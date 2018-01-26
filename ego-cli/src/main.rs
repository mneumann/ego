extern crate ego;
extern crate rand;
extern crate serde_json;

use ego::driver::{Config, SimulationConfig};
use std::env;
use std::fs::File;
use std::io::Read;

fn run_with_config(config: Config) {
    let mut simulation =
        SimulationConfig::new_from_config(config).create_simulation(Box::new(rand::thread_rng()));

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
