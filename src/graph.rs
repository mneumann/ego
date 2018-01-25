use graph_neighbor_matching::graph::OwnedGraph;
use std::str::FromStr;
use std::f32::{INFINITY, NEG_INFINITY};
use closed01::Closed01;
use petgraph::Directed;
use petgraph::Graph;
use petgraph::graph::NodeIndex;
use std::collections::BTreeMap;
use std::fmt::Debug;

/// Similar to GML, but encoded in JSON.
#[derive(Debug, Deserialize)]
pub struct JsonGML {
    directed: bool,
    name: String,
    description: Option<String>,
    nodes: Vec<JsonGMLNode>,
    edges: Vec<JsonGMLEdge>,
}

#[derive(Debug, Deserialize)]
struct JsonGMLNode {
    id: usize,
    label: Option<String>,
    weight: String,
}

#[derive(Debug, Deserialize)]
struct JsonGMLEdge {
    source: usize,
    target: usize,
    weight: Option<f32>,
}

/// Trait used for dot generation

pub trait NodeLabel {
    fn node_label(&self, _idx: usize) -> Option<String> {
        None
    }
    fn node_shape(&self) -> &'static str {
        "circle"
    }
}

fn determine_edge_value_range<T>(g: &Graph<T, f32, Directed>) -> (f32, f32) {
    let mut w_min = INFINITY;
    let mut w_max = NEG_INFINITY;
    for i in g.raw_edges() {
        w_min = w_min.min(i.weight);
        w_max = w_max.max(i.weight);
    }
    (w_min, w_max)
}

fn normalize_to_closed01(w: f32, range: (f32, f32)) -> Closed01<f32> {
    assert!(range.1 >= range.0);
    let dist = range.1 - range.0;
    if dist == 0.0 {
        Closed01::zero()
    } else {
        Closed01::new((w - range.0) / dist)
    }
}

pub fn load_graph_normalized<N>(json_graph: &JsonGML) -> OwnedGraph<N>
where
    N: Clone + Debug + FromStr<Err = &'static str>,
{
    // We only support directed graphs
    assert!(json_graph.directed);

    let mut node_map: BTreeMap<usize, NodeIndex> = BTreeMap::new();
    let mut graph = Graph::new();

    for node in json_graph.nodes.iter() {
        let idx = graph.add_node(N::from_str(&node.weight).unwrap());
        if let Some(_) = node_map.insert(node.id, idx) {
            panic!("duplicate node id");
        }
    }

    for edge in json_graph.edges.iter() {
        graph.add_edge(
            node_map[&edge.source],
            node_map[&edge.target],
            edge.weight.unwrap_or(0.0),
        );
    }

    let edge_range = determine_edge_value_range(&graph);
    let graph = graph.map(
        |_, nw| nw.clone(),
        |_, &ew| normalize_to_closed01(ew, edge_range),
    );

    OwnedGraph::from_petgraph(&graph)
}
