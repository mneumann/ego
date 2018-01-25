extern crate ego;
extern crate nsga2;
extern crate rand;
#[macro_use]
extern crate serde_derive;
extern crate serde_json;
extern crate time;

use ego::graph::{self, JsonGML};
use ego::domain_graph::GraphSimilarity;
use ego::cppn::{Expression, GeometricActivationFunction, PopulationFitness, RandomGenomeCreator,
                Reproduction, StartSymmetry};
use ego::mating::MatingMethodWeights;
use ego::prob::Prob;
use ego::weight::{WeightPerturbanceMethod, WeightRange};
use ego::substrate::Position3d;
use nsga2::selection::SelectNSGPMod;
use nsga2::population::UnratedPopulation;
use ego::fitness_graphsimilarity::fitness;
use std::env;
use std::fs::File;
use std::io::Read;
use ego::range::RangeInclusive;

mod substrate_configuration;

#[derive(Debug, Deserialize)]
struct Config {
    evo: EvoConfig,
    fitness: FitnessConfig,
    selection: SelectionConfig,
    reproduction: ReproductionConfig,
    creator: CreatorConfig,
}

#[derive(Debug, Deserialize)]
struct EvoConfig {
    /// The number of individuals in the population
    mu: usize,
    /// The number of offspring to create
    lambda: usize,
    /// The tournament selection size
    k: usize,
    /// Which objectives to use as fitness
    objectives: Vec<usize>,
}

#[derive(Debug, Deserialize)]
struct FitnessConfig {
    /// The target graph to approximate
    target_graph: JsonGML,
    ///
    edge_score: bool,
    /// Number of iterations used by the similarity algorithm
    iters: usize,
    ///
    eps: f32,
    ///
    link_expression_range: RangeInclusive<f64>,
}

#[derive(Debug, Deserialize)]
struct SelectionConfig {
    objective_eps: f64,
}

#[derive(Debug, Deserialize)]
struct ReproductionConfig {
    mating_method_weights: MatingMethodWeights,
    global_mutation_rate: f32,
    global_element_mutation: f32,
    weight_perturbance: WeightPerturbanceMethod,
    link_weight_range: WeightRange,
    activation_functions: Vec<GeometricActivationFunction>,
    mutate_element_prob: Prob,
    link_weight_creation_sigma: f64,
    mutate_add_node_random_link_weight: bool,
    mutate_drop_node_tournament_k: usize,
    mutate_modify_node_tournament_k: usize,
    mate_retries: usize,
}

#[derive(Debug, Deserialize)]
struct CreatorConfig {
    start_connected: bool,
    start_initial_nodes: usize,
    start_activation_functions: Vec<GeometricActivationFunction>,
    link_weight_range: WeightRange,
    start_link_weight_range: WeightRange,
    start_symmetry: Vec<StartSymmetry>,
}

fn main() {
    let mut rng = rand::thread_rng();

    let config_file = env::args().nth(1).expect("config file");
    println!("config file: {}", config_file);

    let mut file = File::open(config_file).expect("Unable to open config file");
    let mut contents = String::new();
    file.read_to_string(&mut contents)
        .expect("Unable to read file");

    let config: Config = serde_json::from_str(&contents).unwrap();

    let domain_fitness_eval = GraphSimilarity {
        target_graph: graph::load_graph_normalized(&config.fitness.target_graph),
        edge_score: config.fitness.edge_score,
        iters: config.fitness.iters,
        eps: config.fitness.eps,
    };

    let node_count = domain_fitness_eval.target_graph_node_count();
    println!("target_graph node_count: {:?}", node_count);

    let substrate_config = substrate_configuration::substrate_configuration(&node_count);

    let selection = SelectNSGPMod {
        objective_eps: config.selection.objective_eps,
    };

    let reproduction = Reproduction {
        mating_method_weights: config.reproduction.mating_method_weights,
        activation_functions: config.reproduction.activation_functions.clone(),
        mutate_element_prob: config.reproduction.mutate_element_prob,
        weight_perturbance: config.reproduction.weight_perturbance,
        link_weight_range: config.reproduction.link_weight_range,
        link_weight_creation_sigma: config.reproduction.link_weight_creation_sigma,
        mutate_add_node_random_link_weight: config.reproduction.mutate_add_node_random_link_weight,
        mutate_drop_node_tournament_k: config.reproduction.mutate_drop_node_tournament_k,
        mutate_modify_node_tournament_k: config.reproduction.mutate_modify_node_tournament_k,
        mate_retries: config.reproduction.mate_retries,
    };

    let random_genome_creator = RandomGenomeCreator {
        link_weight_range: config.creator.link_weight_range,
        start_activation_functions: config.creator.start_activation_functions,
        start_connected: config.creator.start_connected,
        start_link_weight_range: config.creator.start_link_weight_range,
        start_symmetry: config.creator.start_symmetry,
        start_initial_nodes: config.creator.start_initial_nodes,
    };

    let expression = Expression {
        link_expression_range: config.fitness.link_expression_range,
    };

    let global_mutation_rate: f32 = config.reproduction.global_mutation_rate;
    let global_element_mutation: f32 = config.reproduction.global_element_mutation;

    let mut iteration: usize = 0;

    // create `generation 0`
    let mut parents = {
        let mut initial = UnratedPopulation::new();
        for _ in 0..config.evo.mu {
            initial.push(random_genome_creator.create::<_, Position3d>(0, &mut rng));
        }
        let mut rated = initial.rate_in_parallel(&|ind| {
            fitness(ind, &expression, &substrate_config, &domain_fitness_eval)
        });

        PopulationFitness.apply(0, &mut rated);

        rated.select(config.evo.mu, &config.evo.objectives, &selection, &mut rng)
    };

    let mut best_individual_i = 0;
    let mut best_fitness = parents.individuals()[best_individual_i]
        .fitness()
        .domain_fitness;

    for (i, ind) in parents.individuals().iter().enumerate() {
        let fitness = ind.fitness().domain_fitness;
        if fitness > best_fitness {
            best_fitness = fitness;
            best_individual_i = i;
        }
    }

    println!("best individual generation 0: {}", best_individual_i);

    let max_iterations = 100;

    loop {
        if iteration >= max_iterations {
            break;
        }

        let time_before = time::precise_time_ns();

        // create next generation
        iteration += 1;
        let offspring =
            parents.reproduce(&mut rng, config.evo.lambda, config.evo.k, &|rng, p1, p2| {
                reproduction.mate(rng, p1, p2, iteration)
            });
        let mut next_gen = if global_mutation_rate > 0.0 {
            // mutate all individuals of the whole population.
            // XXX: Optimize
            let old = parents.into_unrated().merge(offspring);
            let mut new_unrated = UnratedPopulation::new();
            let prob = Prob::new(global_mutation_rate);
            for ind in old.as_vec().into_iter() {
                // mutate each in
                let mut genome = ind.into_genome();
                if prob.flip(&mut rng) {
                    // mutate that genome
                    genome.mutate_weights(
                        Prob::new(global_element_mutation),
                        &reproduction.weight_perturbance,
                        &reproduction.link_weight_range,
                        &mut rng,
                    );
                }
                new_unrated.push(genome);
            }
            new_unrated.rate_in_parallel(&|ind| {
                fitness(ind, &expression, &substrate_config, &domain_fitness_eval)
            })
        } else {
            // no global mutation.
            let rated_offspring = offspring.rate_in_parallel(&|ind| {
                fitness(ind, &expression, &substrate_config, &domain_fitness_eval)
            });
            parents.merge(rated_offspring)
        };
        PopulationFitness.apply(iteration, &mut next_gen);
        parents = next_gen.select(config.evo.mu, &config.evo.objectives, &selection, &mut rng);

        best_individual_i = 0;
        best_fitness = parents.individuals()[best_individual_i]
            .fitness()
            .domain_fitness;
        for (i, ind) in parents.individuals().iter().enumerate() {
            let fitness = ind.fitness().domain_fitness;
            if fitness > best_fitness {
                best_fitness = fitness;
                best_individual_i = i;
            }
        }

        let time_after = time::precise_time_ns();
        assert!(time_after > time_before);
        let total_ns = time_after - time_before;
        println!(
            "{}\t{}\t{}\t{}",
            iteration, total_ns, best_individual_i, best_fitness
        );
    }
}
