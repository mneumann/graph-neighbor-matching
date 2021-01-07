use crate::NodeColorWeight;
use closed01::Closed01;
use std::fmt::Debug;

pub trait NodeColorMatching<T>: Debug {
    /// Determines how close or distant two nodes with node weights `node_value_i` of graph A and
    /// `node_value_j` of graph B are. If they have different colors, this method could return 0.0
    /// to describe that they are completely different nodes and as such the neighbor matching will
    /// try to choose a different node.
    ///
    /// NOTE: The returned value MUST be in the range [0, 1].
    fn node_color_matching(&self, node_value_i: &T, node_value_j: &T) -> Closed01<f32>;
}

#[derive(Debug)]
/// Use `IgnoreNodeColors` to ignore node colors.
pub struct IgnoreNodeColors;

impl<T> NodeColorMatching<T> for IgnoreNodeColors {
    fn node_color_matching(&self, _node_i_value: &T, _node_j_value: &T) -> Closed01<f32> {
        Closed01::one()
    }
}

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
