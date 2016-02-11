use closed01::Closed01;
use std::fmt;

/// Trait used by the internal algorithm.
pub trait Edges {
    /// The number of edges
    fn num_edges(&self) -> usize;

    /// Returns the target node of the nth-edge
    fn nth_edge(&self, n: usize) -> Option<usize>;

    /// Returns the nth edge weight. We expect edge weights to be
    /// normalized in the range [0, 1].
    fn nth_edge_weight(&self, n: usize) -> Option<Closed01<f32>>;
}

pub trait Graph {
    type E: Edges;
    fn num_nodes(&self) -> usize;
    fn node_degree(&self, node_idx: usize) -> usize;
    fn in_edges_of<'a>(&'a self, node_idx: usize) -> &'a Self::E;
    fn out_edges_of<'a>(&'a self, node_idx: usize) -> &'a Self::E;
}

pub trait NodeColorMatching: fmt::Debug {
    /// Determines how close or distant two nodes `node_i` of graph A,
    /// and `node_j` of graph B are. If they have different colors,
    /// this method could return 0.0 to describe that they are completely
    /// different nodes and as such the neighbor matching will try to choose
    /// a different node.
    /// NOTE: The returned value MUST be in the range [0, 1].
    fn node_color_matching(&self, node_i: usize, node_j: usize) -> Closed01<f32>;
}
