use super::graph::OwnedGraph;
use super::{ScoreNorm, SimilarityMatrix, WeightedNodeColors};
use graph_io_gml::parse_gml;
use closed01::Closed01;
use asexp::sexp::Sexp;
use petgraph::Directed;
use petgraph::Graph as PetGraph;
use std::f32::{INFINITY, NEG_INFINITY};
use std::fs::File;
use std::io::Read;

fn convert_weight(w: Option<&Sexp>) -> Option<f32> {
    match w {
        Some(s) => s.get_float().map(|f| f as f32),
        None => {
            // use a default
            Some(0.0)
        }
    }
}

fn determine_edge_value_range<T>(g: &PetGraph<T, f32, Directed>) -> (f32, f32) {
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

pub fn load_graph(graph_file: &str) -> OwnedGraph<f32> {
    let graph_str = {
        let mut graph_file = File::open(graph_file).unwrap();
        let mut graph_str = String::new();
        let _ = graph_file.read_to_string(&mut graph_str).unwrap();
        graph_str
    };

    let graph = parse_gml(
        &graph_str,
        &|node_sexp| -> Option<f32> {
            Some(
                node_sexp
                    .and_then(|se| se.get_float().map(|f| f as f32))
                    .unwrap(),
            )
        },
        &convert_weight,
    ).unwrap();

    let edge_range = determine_edge_value_range(&graph);
    let graph = graph.map(
        |_, nw| nw.clone(),
        |_, &ew| normalize_to_closed01(ew, edge_range),
    );

    OwnedGraph::from_petgraph(&graph)
}

pub fn score_graphs(
    a: &OwnedGraph<f32>,
    b: &OwnedGraph<f32>,
    iters: usize,
    eps: f32,
    edge_score: bool,
) -> f32 {
    let mut s = SimilarityMatrix::new(a, b, WeightedNodeColors);
    s.iterate(iters, eps);
    let assignment = s.optimal_node_assignment();
    if edge_score {
        s.score_outgoing_edge_weights_sum_norm(&assignment, ScoreNorm::MaxDegree)
            .get()
    } else {
        s.score_optimal_sum_norm(Some(&assignment), ScoreNorm::MaxDegree)
            .get()
    }
}
