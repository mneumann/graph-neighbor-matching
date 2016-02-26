/// A graph similarity score using neighbor matching according to [this paper][1].
///
/// [1]: http://arxiv.org/abs/1009.5290 "2010, Mladen Nikolic, Measuring Similarity
///      of Graph Nodes by Neighbor Matching"
///
/// TODO: Introduce EdgeWeight trait to abstract edge weight similarity.

extern crate nalgebra;
extern crate munkres;
extern crate closed01;
extern crate petgraph;

use nalgebra::{DMat, Shape, ApproxEq};
use munkres::{WeightMatrix, solve_assignment};
use std::cmp;
use std::mem;
use closed01::Closed01;
pub use traits::{NodeColorMatching, Graph, Edges};

pub mod graph;
mod traits;

#[derive(Debug, Copy, Clone)]
pub enum ScoreNorm {
    /// Divide by minimum graph or node degree
    MinDegree,

    /// Divide by maximum graph or node degree
    MaxDegree,
}

#[derive(Debug)]
pub struct IgnoreNodeColors;

impl<T> NodeColorMatching<T> for IgnoreNodeColors {
    fn node_color_matching(&self, _node_i_value: &T, _node_j_value: &T) -> Closed01<f32> {
        Closed01::one()
    }
}

#[inline(always)]
// NOTE: Our weight matrix minimizes the cost, while our similarity matrix
// wants to maximize the similarity score. That's why we have to convert
// the cost with 1.0 - x.
fn similarity_cost(weight: f32) -> f32 {
    debug_assert!(weight >= 0.0 && weight <= 1.0);
    1.0 - weight
}

#[inline]
/// Calculates the similarity of two nodes `i` and `j`.
///
/// `n_i` contains the neighborhood of i (either in or out neighbors, not both)
/// `n_j` contains the neighborhood of j (either in or out neighbors, not both)
/// `x`   the similarity matrix.
fn s_next<T: Edges>(n_i: &T, n_j: &T, x: &DMat<f32>) -> Closed01<f32> {
    let max_deg = cmp::max(n_i.num_edges(), n_j.num_edges());
    let min_deg = cmp::min(n_i.num_edges(), n_j.num_edges());

    debug_assert!(min_deg <= max_deg);

    if max_deg == 0 {
        // in the paper, 0/0 is defined as 1.0
        return Closed01::one();
    }

    if min_deg == 0 {
        return Closed01::zero();
    }

    assert!(min_deg > 0 && max_deg > 0);

    // map indicies from 0..min(degree) to the node indices
    let mapidx = |(a, b)| (n_i.nth_edge(a).unwrap(), n_j.nth_edge(b).unwrap());

    let mut w = WeightMatrix::from_fn(min_deg, |ab| similarity_cost(x[mapidx(ab)]));

    let assignment = solve_assignment(&mut w);
    assert!(assignment.len() == min_deg);

    let sum: f32 = assignment.iter().fold(0.0, |acc, &ab| acc + x[mapidx(ab)]);

    return Closed01::new(sum / max_deg as f32);
}

#[derive(Debug)]
pub struct SimilarityMatrix<'a, F, G, E, N>
    where F: NodeColorMatching<N>,
          G: Graph<EDGE = E, NODE = N> + 'a,
          E: Edges,
          N: Clone
{
    graph_a: &'a G,
    graph_b: &'a G,
    node_color_matching: F,
    // current version of similarity matrix
    current: DMat<f32>,
    // previous version of similarity matrix
    previous: DMat<f32>,
    // current number of iterations
    num_iterations: usize,
}


impl<'a, F, G, E, N> SimilarityMatrix<'a, F, G, E, N>
    where F: NodeColorMatching<N>,
          G: Graph<EDGE = E, NODE = N>,
          E: Edges,
          N: Clone
{
    pub fn new(graph_a: &'a G,
               graph_b: &'a G,
               node_color_matching: F)
               -> SimilarityMatrix<'a, F, G, E, N> {
        // `x` is the node-similarity matrix.
        // we initialize `x`, so that x[i,j]=1 for all i in A.edges() and j in
        // B.edges().
        let x: DMat<f32> = DMat::from_fn(graph_a.num_nodes(), graph_b.num_nodes(), |i, j| {
            if graph_a.node_degree(i) > 0 && graph_b.node_degree(j) > 0 {
                // this is normally set to 1.0 (i.e. without node color matching).
                node_color_matching.node_color_matching(graph_a.node_value(i),
                                                        graph_b.node_value(j))
            } else {
                Closed01::zero()
            }
            .get()
        });

        let new_x: DMat<f32> = DMat::new_zeros(graph_a.num_nodes(), graph_b.num_nodes());

        SimilarityMatrix {
            graph_a: graph_a,
            graph_b: graph_b,
            node_color_matching: node_color_matching,
            current: x,
            previous: new_x,
            num_iterations: 0,
        }
    }

    fn in_eps(&self, eps: f32) -> bool {
        self.previous.approx_eq_eps(&self.current, &eps)
    }

    /// Calculates the next iteration of the similarity matrix (x[k+1]).
    pub fn next(&mut self) {
        {
            let x = &self.current;
            let new_x = &mut self.previous;
            let shape = x.shape();

            for i in 0..shape.0 {
                for j in 0..shape.1 {
                    let scale = self.node_color_matching
                                    .node_color_matching(self.graph_a.node_value(i),
                                                         self.graph_b.node_value(j));
                    let in_score = s_next(self.graph_a.in_edges_of(i),
                                          self.graph_b.in_edges_of(j),
                                          x);
                    let out_score = s_next(self.graph_a.out_edges_of(i),
                                           self.graph_b.out_edges_of(j),
                                           x);
                    new_x[(i, j)] = in_score.average(out_score).mul(scale).get();
                }
            }
        }

        mem::swap(&mut self.previous, &mut self.current);
        self.num_iterations += 1;
    }

    #[inline]
    /// Iteratively calculate the similarity matrix.
    ///
    /// `stop_after_iter`: Stop after iteration (Calculate x(stop_after_iter))
    /// `eps`:   When to stop the iteration
    pub fn iterate(&mut self, stop_after_iter: usize, eps: f32) {
        for _ in 0..stop_after_iter {
            if self.in_eps(eps) {
                break;
            }
            self.next();
        }
    }

    pub fn matrix(&self) -> &DMat<f32> {
        &self.current
    }

    pub fn num_iterations(&self) -> usize {
        self.num_iterations
    }

    pub fn min_nodes(&self) -> usize {
        cmp::min(self.current.nrows(), self.current.ncols())
    }

    pub fn max_nodes(&self) -> usize {
        cmp::max(self.current.nrows(), self.current.ncols())
    }

    pub fn optimal_node_assignment(&self) -> Vec<(usize, usize)> {
        let n = self.min_nodes();
        let assignment = if n > 0 {
            let mut w = WeightMatrix::from_fn(n, |ij| similarity_cost(self.current[ij]));
            solve_assignment(&mut w)
        } else {
            Vec::new()
        };
        assert!(assignment.len() == n);
        assignment
    }

    fn score_optimal_sum(&self, node_assignment: Option<&[(usize, usize)]>) -> f32 {
        match node_assignment {
            Some(node_assignment) => {
                assert!(node_assignment.len() == self.min_nodes());
                node_assignment.iter().fold(0.0, |acc, &ab| acc + self.current[ab])
            }
            None => {
                let node_assignment = self.optimal_node_assignment();
                assert!(node_assignment.len() == self.min_nodes());
                node_assignment.iter().fold(0.0, |acc, &ab| acc + self.current[ab])
            }
        }
    }

    /// Calculate a measure how good the edge weights match up.
    ///
    /// We start by calculating the optimal node assignment between nodes of graph A and graph B,
    /// then compare all outgoing edges of similar-assigned nodes by again using an assignment
    /// between the edge-weight differences of all edge pairs.
    pub fn score_outgoing_edge_weights_sum_norm(&self,
                                                node_assignment: &[(usize, usize)],
                                                norm: ScoreNorm)
                                                -> Closed01<f32> {
        let n = self.min_nodes();
        let m = self.max_nodes();
        debug_assert!(m >= n);

        assert!(node_assignment.len() == n);

        // we sum up all edge weight scores
        let sum: f32 = node_assignment.iter().fold(0.0, |acc, &(node_i, node_j)| {
            let score_ij = self.score_outgoing_edge_weights_of(node_i, node_j);
            acc + score_ij.get()
        });

        assert!(sum >= 0.0 && sum <= n as f32);

        match norm {
            // Not penalize missing nodes.
            ScoreNorm::MinDegree => Closed01::new(sum / n as f32),

            // To penalize for missing nodes, divide by the maximum number of nodes `m`.
            ScoreNorm::MaxDegree => Closed01::new(sum / m as f32),
        }
    }

    /// Calculate a similarity measure of outgoing of nodes `node_i` of graph A and `node_j` of
    /// graph B.  A score of 1.0 means, the edges weights match up perfectly. 0.0 means, no
    /// similarity.
    fn score_outgoing_edge_weights_of(&self, node_i: usize, node_j: usize) -> Closed01<f32> {
        let out_i = self.graph_a.out_edges_of(node_i);
        let out_j = self.graph_b.out_edges_of(node_j);

        let max_deg = cmp::max(out_i.num_edges(), out_j.num_edges());

        if max_deg == 0 {
            // Nodes with no edges are perfectly similar
            return Closed01::one();
        }

        // Calculates the edge weight distance between edges i and j.
        let edge_weight_distance = &|(i, j)| {
            match (out_i.nth_edge_weight(i), out_j.nth_edge_weight(j)) {
                (Some(w_i), Some(w_j)) => w_i.distance(w_j),
                _ => {
                    // Maximum penalty between two weighted edges
                    // NOTE: missing edges could be penalized more, but we already
                    // penalize for that in the node similarity measure.
                    Closed01::one()
                }
            }
            .get()
        };

        let mut w = WeightMatrix::from_fn(max_deg, edge_weight_distance);

        // calculate optimal edge weight assignement.
        let assignment = solve_assignment(&mut w);
        assert!(assignment.len() == max_deg);

        // The sum is the sum of all weight differences on the optimal `path`.
        // It's range is from 0.0 (perfect matching) to max_deg*1.0 (bad matching).
        let sum: f32 = assignment.iter().fold(0.0, |acc, &ij| acc + edge_weight_distance(ij));

        debug_assert!(sum >= 0.0 && sum <= max_deg as f32);

        // we "invert" the normalized sum so that 1.0 means perfect matching and 0.0
        // no matching.
        Closed01::new(sum / max_deg as f32).inv()
    }

    /// Sums the optimal assignment of the node similarities and normalizes (divides)
    /// by the min/max degree of both graphs.
    /// ScoreNorm::MinDegree is used as default in the paper.
    pub fn score_optimal_sum_norm(&self,
                                  node_assignment: Option<&[(usize, usize)]>,
                                  norm: ScoreNorm)
                                  -> Closed01<f32> {
        let n = self.min_nodes();
        let m = self.max_nodes();


        if n > 0 {
            assert!(m > 0);
            let sum = self.score_optimal_sum(node_assignment);
            assert!(sum >= 0.0 && sum <= n as f32);

            match norm {
                // Not penalize missing nodes.
                ScoreNorm::MinDegree => Closed01::new(sum / n as f32),

                // To penalize for missing nodes, divide by the maximum number of nodes `m`.
                ScoreNorm::MaxDegree => Closed01::new(sum / m as f32),
            }
        } else {
            Closed01::zero()
        }
    }

    /// Calculates the average over the whole node similarity matrix. This is faster,
    /// as no assignment has to be found. "Graphs with greater number of automorphisms
    /// would be considered to be more self-similar than graphs without automorphisms."
    pub fn score_average(&self) -> Closed01<f32> {
        let n = self.min_nodes();
        if n > 0 {
            let items = self.current.as_vec();
            let sum: f32 = items.iter().fold(0.0, |acc, &v| acc + v);
            let len = items.len();
            assert!(len > 0);
            Closed01::new(sum / len as f32)
        } else {
            Closed01::zero()
        }
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

#[cfg(test)]
mod tests {
    use super::graph::{Edge, EdgeList, Node, OwnedGraph, GraphBuilder};
    use super::{ScoreNorm, SimilarityMatrix, IgnoreNodeColors};

    fn edge(i: usize) -> Edge {
        Edge::new_unweighted(i)
    }

    fn node(in_edges: Vec<Edge>, out_edges: Vec<Edge>) -> Node<()> {
        Node::new(EdgeList::new(in_edges), EdgeList::new(out_edges), ())
    }

    fn graph(nodes: Vec<Node<()>>) -> OwnedGraph<()> {
        OwnedGraph::new(nodes)
    }

    #[test]
    fn test_matrix() {
        // A: 0 --> 1
        let a = graph(vec![node(vec![], vec![edge(1)]), node(vec![edge(0)], vec![])]);

        // B: 0 <-- 1
        let b = graph(vec![node(vec![edge(1)], vec![]), node(vec![], vec![edge(0)])]);

        let mut s = SimilarityMatrix::new(&a, &b, IgnoreNodeColors);
        s.iterate(100, 0.1);

        println!("{:?}", s);
        assert_eq!(2, s.num_iterations());
        let mat = s.matrix();
        assert_eq!(2, mat.nrows());
        assert_eq!(2, mat.ncols());

        // A and B are isomorphic
        assert_eq!(0.0, mat[(0, 0)]);
        assert_eq!(1.0, mat[(0, 1)]);
        assert_eq!(1.0, mat[(1, 0)]);
        assert_eq!(0.0, mat[(1, 1)]);
    }

    #[test]
    fn test_matrix_iter1() {
        let a = graph(vec![
            node(vec![edge(0), edge(0), edge(0)], vec![edge(0), edge(0), edge(0)]),
        ]);

        let b = graph(vec![
            node(vec![edge(0), edge(0), edge(0), edge(0), edge(0)], vec![edge(0), edge(0), edge(0), edge(0), edge(0)]),
        ]);

        let mut s = SimilarityMatrix::new(&a, &b, IgnoreNodeColors);
        s.iterate(1, 0.1);

        assert_eq!(1, s.num_iterations());
        let mat = s.matrix();
        assert_eq!(3.0 / 5.0, mat[(0, 0)]);
    }

    #[test]
    fn test_score() {
        // A: 0 --> 1
        let a = graph(vec![node(vec![], vec![edge(1)]), node(vec![edge(0)], vec![])]);

        // B: 0 <-- 1
        let b = graph(vec![node(vec![edge(1)], vec![]), node(vec![], vec![edge(0)])]);

        let mut s = SimilarityMatrix::new(&a, &b, IgnoreNodeColors);
        s.iterate(100, 0.1);

        assert_eq!(2, s.num_iterations());

        // The score is 1.0 <=> A and B are isomorphic
        assert_eq!(1.0,
                   s.score_optimal_sum_norm(None, ScoreNorm::MinDegree).get());

        // The score is 1.0 <=> A and B are isomorphic
        assert_eq!(1.0,
                   s.score_optimal_sum_norm(None, ScoreNorm::MaxDegree).get());
    }

    #[test]
    fn test_score_with_graphbuilder() {
        // A: 0 --> 1
        let mut a: GraphBuilder<usize, ()> = GraphBuilder::new();
        a.add_node(0, ());
        a.add_node(1, ());
        a.add_edge_unweighted(0, 1);

        // B: 0 <-- 1
        let mut b: GraphBuilder<usize, ()> = GraphBuilder::new();
        b.add_node(0, ());
        b.add_node(1, ());
        b.add_edge_unweighted(1, 0);

        let ga = a.graph();
        let gb = b.graph();

        let mut s = SimilarityMatrix::new(&ga, &gb, IgnoreNodeColors);
        s.iterate(100, 0.1);

        assert_eq!(2, s.num_iterations());

        // The score is 1.0 <=> A and B are isomorphic
        assert_eq!(1.0,
                   s.score_optimal_sum_norm(None, ScoreNorm::MinDegree).get());

        // The score is 1.0 <=> A and B are isomorphic
        assert_eq!(1.0,
                   s.score_optimal_sum_norm(None, ScoreNorm::MaxDegree).get());
    }

}
