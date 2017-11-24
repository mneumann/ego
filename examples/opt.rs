extern crate ego;
extern crate nsga2;
extern crate rand;
extern crate time;

extern crate toml;
#[macro_use]
extern crate serde_derive;

use ego::graph;
use ego::domain_graph::GraphSimilarity;
use ego::cppn::{GeometricActivationFunction, RandomGenomeCreator, Reproduction, Expression,
                PopulationFitness};
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

mod substrate_configuration;

#[derive(Debug, Deserialize)]
struct Config {
    evo: EvoConfig,
}

#[derive(Debug, Deserialize)]
struct EvoConfig {
    mu: usize,
    lambda: usize,
    k: usize,
    objectives: Vec<usize>,
}

fn main() {

    let mut rng = rand::thread_rng();

    let graph_file = env::args().nth(1).expect("graph file");
    let config_file = env::args().nth(2).expect("config file");
    println!("graph file: {}", graph_file);
    println!("config file: {}", config_file);

    let domain_fitness_eval = GraphSimilarity {
        target_graph: graph::load_graph_normalized(&graph_file),
        edge_score: false,
        iters: 50,
        eps: 0.01,
    };

    // XXX
    let node_count = domain_fitness_eval.target_graph_node_count();
    println!("{:?}", node_count);

    let substrate_config = substrate_configuration::substrate_configuration(&node_count);

    let mut file = File::open(config_file).expect("Unable to open config file");
    let mut contents = String::new();
    file.read_to_string(&mut contents).expect(
        "Unable to read file",
    );

    let config: Config = toml::from_str(&contents).unwrap();

    let selection = SelectNSGPMod { objective_eps: 0.01 };

    let weight_perturbance_sigma = 0.1;
    let link_weight_range = 1.0;
    let reproduction = Reproduction {
        mating_method_weights: MatingMethodWeights {
            mutate_add_node: 2,
            mutate_drop_node: 1,
            mutate_modify_node: 0,
            mutate_connect: 2,
            mutate_disconnect: 2,
            mutate_symmetric_join: 2,
            mutate_symmetric_fork: 2,
            mutate_symmetric_connect: 1,
            mutate_weights: 100,
            crossover_weights: 0,
        },
        activation_functions: vec![
            GeometricActivationFunction::Linear,
            GeometricActivationFunction::LinearClipped,
            //GeometricActivationFunction::Gaussian,
            GeometricActivationFunction::BipolarGaussian,
            GeometricActivationFunction::BipolarSigmoid,
            GeometricActivationFunction::Sine,
            GeometricActivationFunction::Absolute,
        ],
        mutate_element_prob: Prob::new(0.05),
        weight_perturbance: WeightPerturbanceMethod::JiggleGaussian {
            sigma: weight_perturbance_sigma,
        },
        link_weight_range: WeightRange::bipolar(link_weight_range),
        link_weight_creation_sigma: 0.1,

        mutate_add_node_random_link_weight: true,
        mutate_drop_node_tournament_k: 2,
        mutate_modify_node_tournament_k: 2,
        mate_retries: 100,
    };

    let random_genome_creator = RandomGenomeCreator {
        link_weight_range: WeightRange::bipolar(link_weight_range),

        start_activation_functions: vec![
            //GeometricActivationFunction::Linear,
            GeometricActivationFunction::BipolarGaussian,
            GeometricActivationFunction::BipolarSigmoid,
            GeometricActivationFunction::Sine,
        ],
        start_connected: false,
        start_link_weight_range: WeightRange::bipolar(0.1),
        start_symmetry: vec![], // Some(3.0), None, Some(3.0)],
        start_initial_nodes: 0,
    };

    let expression = Expression { link_expression_range: (0.1, 0.5) };

    let global_mutation_rate: f32 = 0.0;
    let global_element_mutation: f32 = 0.0;
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
            iteration,
            total_ns,
            best_individual_i,
            best_fitness
        );
    }
}
