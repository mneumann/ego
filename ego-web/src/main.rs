extern crate actix_web;
extern crate ego;
extern crate futures;
extern crate rand;
#[macro_use]
extern crate serde_derive;
extern crate serde_json;

use ego::driver::{Config, SimulationConfig};
use actix_web::*;
use futures::future::Future;

#[derive(Serialize)]
struct HistoryEntry {
    iteration: usize,
    best_fitness: f64,
    /// In seconds
    duration: f64,
}

#[derive(Serialize)]
struct SimulationResponse {
    history: Vec<HistoryEntry>,
}

fn run_with_config(config: Config) -> SimulationResponse {
    let mut simulation =
        SimulationConfig::new_from_config(config).create_simulation(Box::new(rand::thread_rng()));

    let mut response = SimulationResponse { history: vec![] };

    let max_iterations = 100;
    loop {
        response.history.push(HistoryEntry {
            iteration: simulation.iteration,
            best_fitness: simulation.fittest_individual().1,
            duration: simulation.current_population_ns as f64 / 1_000_000_000.0,
        });

        if simulation.iteration >= max_iterations {
            break;
        }

        simulation = simulation.next_generation();
    }
    return response;
}

fn index(req: HttpRequest) -> Box<Future<Item = HttpResponse, Error = Error>> {
    req.json()
        .limit(10_000_000)
        .from_err()
        .and_then(|config: Config| {
            println!("config: {:?}", config);

            let res = run_with_config(config);
            Ok(httpcodes::HTTPOk.build().json(res)?)
        })
        .responder()

    //format!("Hello {}!", &req.match_info()["name"])
}

fn main() {
    HttpServer::new(|| Application::new().resource("/simulation", |r| r.f(index)))
        .bind("127.0.0.1:7878")
        .unwrap()
        .run();
}
