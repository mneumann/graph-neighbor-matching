//! A graph similarity score using neighbor matching according to [this paper][1].
//!
//! [1]: http://arxiv.org/abs/1009.5290 "2010, Mladen Nikolic, Measuring Similarity
//!      of Graph Nodes by Neighbor Matching"
//!
//! TODO: Introduce EdgeWeight trait to abstract edge weight similarity.

pub mod graph;
mod graph_traits;
mod node_color_matching;
mod score_norm;
mod similarity_matrix;

use closed01::Closed01;
pub use {graph_traits::*, node_color_matching::*, score_norm::*, similarity_matrix::*};

impl NodeColorWeight for f32 {
    fn node_color_weight(&self) -> f32 {
        *self
    }
}

pub fn similarity_max_degree<T: Graph>(a: &T, b: &T, num_iters: usize, eps: f32) -> Closed01<f32> {
    let mut s = SimilarityMatrix::new(a, b, IgnoreNodeColors);
    s.iterate(num_iters, eps);
    s.score_optimal_sum_norm(None, ScoreNorm::MaxDegree)
}

pub fn similarity_min_degree<T: Graph>(a: &T, b: &T, num_iters: usize, eps: f32) -> Closed01<f32> {
    let mut s = SimilarityMatrix::new(a, b, IgnoreNodeColors);
    s.iterate(num_iters, eps);
    s.score_optimal_sum_norm(None, ScoreNorm::MinDegree)
}
