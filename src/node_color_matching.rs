use crate::NodeColorWeight;
use closed01::Closed01;
use std::fmt::Debug;

/// Describes the closeness of two nodes based on their colors. The color can be thought of as a
/// node type. For example if you have a graph with nodes of type A and B which represent
/// completely different things, you'd assign them a `node_color_matching` value of 0.0. This will
/// tell our algorithm not to try to match these two nodes and their edges.
///
pub trait NodeColorMatching<T>: Debug {
    /// Determines how close or distant two nodes with node weights `node_value_i` of graph A and
    /// `node_value_j` of graph B are. If they have different colors, this method could return 0.0
    /// to describe that they are completely different nodes and as such the neighbor matching will
    /// try to choose a different node.
    ///
    /// NOTE: The returned value MUST be in the range [0, 1].
    fn node_color_matching(&self, node_value_i: &T, node_value_j: &T) -> Closed01<f32>;
}

/// Use `IgnoreNodeColors` to ignore node colors.
#[derive(Debug)]
pub struct IgnoreNodeColors;

impl<T> NodeColorMatching<T> for IgnoreNodeColors {
    fn node_color_matching(&self, _node_i_value: &T, _node_j_value: &T) -> Closed01<f32> {
        Closed01::one()
    }
}

/// Use `WeightedNodeColors` to use the distance of both nodes `node_color_weight` as
/// a measure of the similarity between these nodes.
#[derive(Debug)]
pub struct WeightedNodeColors;

impl<T: NodeColorWeight> NodeColorMatching<T> for WeightedNodeColors {
    fn node_color_matching(&self, node_i_value: &T, node_j_value: &T) -> Closed01<f32> {
        let dist = (node_i_value.node_color_weight() - node_j_value.node_color_weight())
            .abs()
            .min(1.0);

        debug_assert!(dist >= 0.0 && dist <= 1.0);

        Closed01::new(dist).inv()
    }
}
