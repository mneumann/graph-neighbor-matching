//! Traits that represent an abstract graph upon which our algorithm operates.

use closed01::Closed01;
use petgraph::{graph::NodeIndex, Directed, Graph as PetGraph};

/// The weight of an edge.
pub type EdgeWeight = Closed01<f32>;

/// The weight of the node color.
pub trait NodeColorWeight {
    fn node_color_weight(&self) -> f32;
}

/// Abstract representation of the edges of a node. Used by the algorithm.
pub trait Edges {
    /// The number of edges
    fn num_edges(&self) -> usize;

    /// Returns the target node of the nth-edge
    fn nth_edge(&self, n: usize) -> Option<usize>;

    /// Returns the nth edge weight. We expect edge weights to be
    /// normalized in the range [0, 1].
    fn nth_edge_weight(&self, n: usize) -> Option<EdgeWeight>;
}

/// Abstract representation of a Graph. Used by the algorithm.
pub trait Graph {
    type EDGE: Edges;
    type NODE: Clone;

    fn num_nodes(&self) -> usize;
    fn node_degree(&self, node_idx: usize) -> usize;
    fn node_value(&self, node_idx: usize) -> &Self::NODE;
    fn in_edges_of(&self, node_idx: usize) -> &Self::EDGE;
    fn out_edges_of(&self, node_idx: usize) -> &Self::EDGE;

    fn to_petgraph(&self) -> PetGraph<Self::NODE, EdgeWeight, Directed> {
        let mut graph = PetGraph::new();
        for i in 0..self.num_nodes() {
            let idx = graph.add_node(self.node_value(i).clone());
            assert!(idx.index() == i);
        }
        for i in 0..self.num_nodes() {
            let in_edges = self.in_edges_of(i);
            for k in 0..in_edges.num_edges() {
                let j = in_edges.nth_edge(k).unwrap();
                let w = in_edges.nth_edge_weight(k).unwrap();
                graph.add_edge(NodeIndex::new(j), NodeIndex::new(i), w);
            }
            let out_edges = self.out_edges_of(i);
            for k in 0..out_edges.num_edges() {
                let j = out_edges.nth_edge(k).unwrap();
                let w = out_edges.nth_edge_weight(k).unwrap();
                graph.add_edge(NodeIndex::new(i), NodeIndex::new(j), w);
            }
        }
        graph
    }
}
