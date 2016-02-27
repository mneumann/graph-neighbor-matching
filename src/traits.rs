use closed01::Closed01;
use std::fmt::Debug;
use petgraph::Graph as PetGraph;
use petgraph::Directed;
use petgraph::graph::NodeIndex;

pub type EdgeWeight = Closed01<f32>;

/// Trait used by the internal algorithm.
pub trait Edges {
    /// The number of edges
    fn num_edges(&self) -> usize;

    /// Returns the target node of the nth-edge
    fn nth_edge(&self, n: usize) -> Option<usize>;

    /// Returns the nth edge weight. We expect edge weights to be
    /// normalized in the range [0, 1].
    fn nth_edge_weight(&self, n: usize) -> Option<EdgeWeight>;
}

pub trait Graph {
    type EDGE: Edges;
    type NODE: Clone;
    fn num_nodes(&self) -> usize;
    fn node_degree(&self, node_idx: usize) -> usize;
    fn node_value(&self, node_idx: usize) -> &Self::NODE;
    fn in_edges_of<'a>(&'a self, node_idx: usize) -> &'a Self::EDGE;
    fn out_edges_of<'a>(&'a self, node_idx: usize) -> &'a Self::EDGE;

    fn to_petgraph(&self) -> PetGraph<Self::NODE, EdgeWeight, Directed> {
        let mut g = PetGraph::new();
        for i in 0..self.num_nodes() {
            let idx = g.add_node(self.node_value(i).clone());
            assert!(idx.index() == i);
        }
        for i in 0..self.num_nodes() {
            let in_edges = self.in_edges_of(i);
            for k in 0..in_edges.num_edges() {
                let j = in_edges.nth_edge(k).unwrap();
                let w = in_edges.nth_edge_weight(k).unwrap();
                g.add_edge(NodeIndex::new(j), NodeIndex::new(i), w);
            }
            let out_edges = self.out_edges_of(i);
            for k in 0..out_edges.num_edges() {
                let j = out_edges.nth_edge(k).unwrap();
                let w = out_edges.nth_edge_weight(k).unwrap();
                g.add_edge(NodeIndex::new(i), NodeIndex::new(j), w);
            }
        }
        return g;
    }
}

pub trait NodeColorMatching<T>: Debug {

    /// Determines how close or distant two nodes with node weights `node_value_i` of graph A, and
    /// `node_value_j` of graph B are. If they have different colors, this method could return 0.0
    /// to describe that they are completely different nodes and as such the neighbor matching will
    /// try to choose a different node.  NOTE: The returned value MUST be in the range [0, 1].
    fn node_color_matching(&self, node_value_i: &T, node_value_j: &T) -> Closed01<f32>;
}

pub trait NodeColorWeight {
    fn node_color_weight(&self) -> f32;
}
