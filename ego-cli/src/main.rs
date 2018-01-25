extern crate ego;
extern crate nsga2;
extern crate rand;
#[macro_use]
extern crate serde_derive;
extern crate serde_json;
extern crate time;

use ego::graph::{self, JsonGML};
use ego::domain_graph::{GraphSimilarity, Neuron};
use ego::cppn::{Expression, PopulationFitness, RandomGenomeCreator, Reproduction, G};
use ego::substrate::{Position3d, SubstrateConfiguration};
use nsga2::selection::SelectNSGPMod;
use nsga2::population::{RankedPopulation, UnratedPopulation};
use ego::fitness_graphsimilarity::fitness;
use ego::fitness::Fitness;
use std::env;
use std::fs::File;
use std::io::Read;
use ego::range::RangeInclusive;
use ego::substrate_configuration::substrate_configuration;
use rand::Rng;

#[derive(Debug, Deserialize)]
struct Config {
    evo: EvoConfig,
    fitness: FitnessConfig,
    selection: SelectionConfig,
    reproduction: Reproduction,
    creator: RandomGenomeCreator,
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

// Extended Config
struct SimulationConfig {
    domain_fitness_eval: GraphSimilarity,
    substrate_config: SubstrateConfiguration<Position3d, Neuron>,
    selection: SelectNSGPMod,
    expression: Expression,
    config: Config,
}

// Saves the simulation state
struct Simulation<R: Rng> {
    simulation_config: SimulationConfig,
    iteration: usize,
    population: RankedPopulation<G, Fitness>,
    rng: Box<R>,
    /// how long did it take to reproduce/create and rank the current population
    current_population_ns: u64,
    /// total simulation duration
    total_ns: u64,
}

fn create_simulation_config(config: Config) -> SimulationConfig {
    let domain_fitness_eval = GraphSimilarity {
        target_graph: graph::load_graph_normalized(&config.fitness.target_graph),
        edge_score: config.fitness.edge_score,
        iters: config.fitness.iters,
        eps: config.fitness.eps,
    };
    let node_count = domain_fitness_eval.target_graph_node_count();

    let substrate_config = substrate_configuration(&node_count);

    let selection = SelectNSGPMod {
        objective_eps: config.selection.objective_eps,
    };

    let expression = Expression {
        link_expression_range: config.fitness.link_expression_range,
    };

    SimulationConfig {
        domain_fitness_eval,
        substrate_config,
        selection,
        expression,
        config,
    }
}

fn create_simulation<R: Rng>(
    simulation_config: SimulationConfig,
    mut rng: Box<R>,
) -> Simulation<R> {
    let time_before = time::precise_time_ns();
    let population = create_generation_0(&simulation_config, &mut rng);
    let time_after = time::precise_time_ns();
    assert!(time_after > time_before);
    let ns = time_after - time_before;

    Simulation {
        simulation_config,
        iteration: 0,
        population,
        current_population_ns: ns,
        total_ns: ns,
        rng,
    }
}

fn create_generation_0<R: Rng>(
    simulation_config: &SimulationConfig,
    rng: &mut R,
) -> RankedPopulation<G, Fitness> {
    // create `generation 0`
    let iteration = 0;
    let mut initial = UnratedPopulation::new();
    for _ in 0..simulation_config.config.evo.mu {
        initial.push(
            simulation_config
                .config
                .creator
                .create::<_, Position3d>(iteration, rng),
        );
    }

    let mut rated = initial.rate_in_parallel(&|ind| {
        fitness(
            ind,
            &simulation_config.expression,
            &simulation_config.substrate_config,
            &simulation_config.domain_fitness_eval,
        )
    });

    PopulationFitness.apply(iteration, &mut rated);

    return rated.select(
        simulation_config.config.evo.mu,
        &simulation_config.config.evo.objectives,
        &simulation_config.selection,
        rng,
    );
}

impl<R: Rng> Simulation<R> {
    fn fittest_individual(&self) -> (usize, f64) {
        let mut best_individual_i = 0;
        let mut best_fitness = self.population.individuals()[best_individual_i]
            .fitness()
            .domain_fitness;

        for (i, ind) in self.population.individuals().iter().enumerate() {
            let fitness = ind.fitness().domain_fitness;
            if fitness > best_fitness {
                best_fitness = fitness;
                best_individual_i = i;
            }
        }
        return (best_individual_i, best_fitness);
    }

    fn print_statistics(&self) {
        let (best_i, best_fit) = self.fittest_individual();
        println!(
            "{}\t{}\t{}\t{}\t{}",
            self.iteration, best_i, best_fit, self.current_population_ns, self.total_ns,
        );
    }

    // create next generation
    fn next_generation(self) -> Simulation<R> {
        let time_before = time::precise_time_ns();
        let Simulation {
            simulation_config,
            iteration,
            population,
            current_population_ns,
            mut rng,
            ..
        } = self;

        let next_iteration = iteration + 1;

        let offspring = population.reproduce(
            &mut rng,
            simulation_config.config.evo.lambda,
            simulation_config.config.evo.k,
            &|rng, p1, p2| {
                simulation_config
                    .config
                    .reproduction
                    .mate(rng, p1, p2, next_iteration)
            },
        );

        let global_mutation_rate = simulation_config.config.reproduction.global_mutation_rate;
        let mut next_gen = if global_mutation_rate.get() > 0.0 {
            // mutate all individuals of the whole population.
            // XXX: Optimize
            let old = population.into_unrated().merge(offspring);
            let mut new_unrated = UnratedPopulation::new();
            //let prob = Prob::new(global_mutation_rate);
            for ind in old.as_vec().into_iter() {
                // mutate each in
                let mut genome = ind.into_genome();
                if global_mutation_rate.flip(&mut rng) {
                    // mutate that genome
                    genome.mutate_weights(
                        simulation_config
                            .config
                            .reproduction
                            .global_element_mutation,
                        &simulation_config.config.reproduction.weight_perturbance,
                        &simulation_config.config.reproduction.link_weight_range,
                        &mut rng,
                    );
                }
                new_unrated.push(genome);
            }
            new_unrated.rate_in_parallel(&|ind| {
                fitness(
                    ind,
                    &simulation_config.expression,
                    &simulation_config.substrate_config,
                    &simulation_config.domain_fitness_eval,
                )
            })
        } else {
            // no global mutation.
            let rated_offspring = offspring.rate_in_parallel(&|ind| {
                fitness(
                    ind,
                    &simulation_config.expression,
                    &simulation_config.substrate_config,
                    &simulation_config.domain_fitness_eval,
                )
            });
            population.merge(rated_offspring)
        };
        PopulationFitness.apply(next_iteration, &mut next_gen);
        let next_population = next_gen.select(
            simulation_config.config.evo.mu,
            &simulation_config.config.evo.objectives,
            &simulation_config.selection,
            &mut rng,
        );
        let time_after = time::precise_time_ns();
        assert!(time_after > time_before);
        let ns = time_after - time_before;

        Simulation {
            simulation_config,
            iteration: next_iteration,
            population: next_population,
            current_population_ns: ns,
            total_ns: current_population_ns + ns,
            rng,
        }
    }
}

fn run_with_config(config: Config) {
    let mut simulation = create_simulation(
        create_simulation_config(config),
        Box::new(rand::thread_rng()),
    );

    println!("iter\tbest_i\tbest_fit\tns_current\tns_total");

    let max_iterations = 100;
    loop {
        simulation.print_statistics();

        if simulation.iteration >= max_iterations {
            break;
        }

        simulation = simulation.next_generation();
    }
}

fn from_command_line() {
    let config_file = env::args().nth(1).expect("config file");
    println!("config file: {}", config_file);

    let mut file = File::open(config_file).expect("Unable to open config file");
    let mut contents = String::new();
    file.read_to_string(&mut contents)
        .expect("Unable to read file");

    let config: Config = serde_json::from_str(&contents).unwrap();

    run_with_config(config);
}

fn main() {
    from_command_line();
}
