/// A graph similarity score using neighbor matching according to [this paper][1].
///
/// [1]: http://arxiv.org/abs/1009.5290 "2010, Mladen Nikolic, Measuring Similarity
///      of Graph Nodes by Neighbor Matching"
///
/// TODO: Introduce EdgeWeight trait to abstract edge weight similarity.

extern crate nalgebra;
extern crate munkres;

use nalgebra::{DMat, Shape, ApproxEq};
use munkres::{Weights, WeightMatrix, solve_assignment};
use std::cmp;
use std::mem;
use std::fmt;

/// Node index type. Our graphs never exceed 4 billion nodes.
pub type Idx = u32;

/// Encapsulates a floating point number in the range [0, 1] including both endpoints.
#[derive(Copy, Clone, Debug)]
pub struct Closed01<F>(F);

impl Closed01<f32> {
    #[inline(always)]
    pub fn new(f: f32) -> Closed01<f32> {
        assert!(f >= 0.0 && f <= 1.0);
        Closed01(f)
    }

    #[inline(always)]
    pub fn zero() -> Closed01<f32> {
        Closed01(0.0)
    }

    #[inline(always)]
    pub fn one() -> Closed01<f32> {
        Closed01(1.0)
    }

    #[inline(always)]
    pub fn distance(self, other: Closed01<f32>) -> Closed01<f32> {
        let d = (self.0 - other.0).abs();
        debug_assert!(d >= 0.0 && d <= 1.0);
        Closed01(d)
    }

    #[inline(always)]
    pub fn get(self) -> f32 {
        debug_assert!(self.0 >= 0.0 && self.0 <= 1.0);
        self.0
    }

    #[inline(always)]
    /// The average of two values.
    pub fn average(a: Closed01<f32>, b: Closed01<f32>) -> Closed01<f32> {
        let avg = (a.get() + b.get()) / 2.0;
        debug_assert!(avg >= 0.0 && avg <= 1.0);
        Closed01(avg)
    }

    #[inline(always)]
    pub fn scale(&self, scalar: Closed01<f32>) -> Closed01<f32> {
        let s = self.get() * scalar.get();
        debug_assert!(s >= 0.0 && s <= 1.0);
        Closed01(s)
    }
}

trait Edges {
    /// The number of edges
    fn len(&self) -> usize;

    /// Returns the target node of the nth-edge
    fn nth_edge(&self, n: usize) -> Option<usize>;

    /// Returns the nth edge weight. We expect edge weights to be
    /// normalized in the range [0, 1].
    fn nth_edge_weight(&self, _n: usize) -> Option<Closed01<f32>> {
        None
    }
}

impl<'a> Edges for &'a [Idx] {
    #[inline]
    fn len(&self) -> usize {
        let x: &[Idx] = self;
        x.len()
    }
    #[inline]
    fn nth_edge(&self, n: usize) -> Option<usize> {
        self.get(n).map(|&n| n as usize)
    }
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

#[derive(Debug)]
pub struct IgnoreNodeColors;

impl NodeColorMatching for IgnoreNodeColors {
    fn node_color_matching(&self, _node_i: usize, _node_j: usize) -> Closed01<f32> {
        Closed01::one()
    }
}

#[inline]
/// Calculates the similarity of two nodes `i` and `j`.
///
/// `n_i` contains the neighborhood of i (either in or out neighbors, not both)
/// `n_j` contains the neighborhood of j (either in or out neighbors, not both)
/// `x`   the similarity matrix.
fn s_next<T: Edges>(n_i: T, n_j: T, x: &DMat<f32>) -> Closed01<f32> {
    let max_deg = cmp::max(n_i.len(), n_j.len());
    let min_deg = cmp::min(n_i.len(), n_j.len());

    if min_deg == 0 {
        // in the paper, 0/0 is defined as 1.0
        return Closed01::one();
    }

    assert!(min_deg > 0 && max_deg > 0);

    // map indicies from 0..min(degree) to the node indices
    let mapidx = |(a, b)| (n_i.nth_edge(a).unwrap(), n_j.nth_edge(b).unwrap());

    let mut w = WeightMatrix::from_fn(min_deg, |ab| x[mapidx(ab)]);

    let assignment = solve_assignment(&mut w);
    assert!(assignment.len() == min_deg);

    let sum: f32 = assignment.iter().fold(0.0, |acc, &ab| acc + x[mapidx(ab)]);

    return Closed01::new(sum / max_deg as f32);
}

#[derive(Copy, Clone, Debug)]
pub struct Graph<'a> {
    in_edges: &'a [Vec<Idx>],
    out_edges: &'a [Vec<Idx>],
}

impl<'a> Graph<'a> {
    pub fn new(in_edges: &'a [Vec<Idx>], out_edges: &'a [Vec<Idx>]) -> Graph<'a> {
        assert!(in_edges.len() == out_edges.len());
        Graph {
            in_edges: in_edges,
            out_edges: out_edges,
        }
    }

    fn num_nodes(&self) -> usize {
        let n = self.in_edges.len();
        assert!(n == self.out_edges.len());
        n
    }

    #[inline]
    fn node_degree(&self, node_idx: usize) -> usize {
        self.in_edges[node_idx].len() + self.out_edges[node_idx].len()
    }

    #[inline]
    fn in_edges_of(&self, node_idx: usize) -> &[Idx] {
        &self.in_edges[node_idx]
    }

    #[inline]
    fn out_edges_of(&self, node_idx: usize) -> &[Idx] {
        &self.out_edges[node_idx]
    }
}


#[derive(Debug)]
pub struct GraphSimilarityMatrix<'a, F: NodeColorMatching + 'a> {
    graph_a: Graph<'a>,
    graph_b: Graph<'a>,
    node_color_matching: F,
    // current version of similarity matrix
    current: DMat<f32>,
    // previous version of similarity matrix
    previous: DMat<f32>,
    // current number of iterations
    num_iterations: usize,
}

impl<'a, F> GraphSimilarityMatrix<'a, F> where F: NodeColorMatching
{
    pub fn new(graph_a: Graph<'a>,
               graph_b: Graph<'a>,
               node_color_matching: F)
               -> GraphSimilarityMatrix<'a, F> {
        // `x` is the node-similarity matrix.
        // we initialize `x`, so that x[i,j]=1 for all i in A.edges() and j in
        // B.edges().
        let x: DMat<f32> = DMat::from_fn(graph_a.num_nodes(), graph_b.num_nodes(), |i, j| {
            if graph_a.node_degree(i) > 0 && graph_b.node_degree(j) > 0 {
                // this is normally set to 1.0 (i.e. without node color matching).
                node_color_matching.node_color_matching(i, j)
            } else {
                Closed01::zero()
            }
            .get()
        });

        let new_x: DMat<f32> = DMat::new_zeros(graph_a.num_nodes(), graph_b.num_nodes());

        GraphSimilarityMatrix {
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
                    let scale = self.node_color_matching.node_color_matching(i, j);
                    new_x[(i, j)] = Closed01::average(s_next(self.graph_a.in_edges_of(i),
                                                             self.graph_b.in_edges_of(j),
                                                             x),
                                                      s_next(self.graph_a.out_edges_of(i),
                                                             self.graph_b.out_edges_of(j),
                                                             x))
                                        .scale(scale)
                                        .get()
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
            let mut w = WeightMatrix::from_fn(n, |ij| self.current[ij]);
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
    pub fn score_outgoing_edge_weights(&self, node_assignment: &[(usize, usize)]) -> f32 {
        let n = self.min_nodes();
        let m = self.max_nodes();

        assert!(node_assignment.len() == n);

        // we sum up all edge weight scores
        let sum: f32 = node_assignment.iter().fold(0.0, |acc, &(node_i, node_j)| {
            let score_ij = self.score_outgoing_edge_weights_of(node_i, node_j);
            acc + score_ij.get()
        });

        debug_assert!(sum >= 0.0 && sum <= n as f32);

        // to penalize for missing nodes, we divide by the maximum of number of nodes `m`.

        debug_assert!(m >= n);

        let score = sum / m as f32;

        debug_assert!(score >= 0.0 && score <= 1.0);

        score
    }

    /// Calculate a similarity measure of outgoing of nodes `node_i` of graph A and `node_j` of
    /// graph B.  A score of 1.0 means, the edges weights match up perfectly. 0.0 means, no
    /// similarity.
    fn score_outgoing_edge_weights_of(&self, node_i: usize, node_j: usize) -> Closed01<f32> {
        let out_i = self.graph_a.out_edges_of(node_i);
        let out_j = self.graph_b.out_edges_of(node_j);

        let max_deg = cmp::max(out_i.len(), out_j.len());

        if max_deg == 0 {
            // Nodes with no edges are perfectly similar
            return Closed01::one();
        }

        let mut w = WeightMatrix::from_fn(max_deg, |(i, j)| {
            match (out_i.nth_edge_weight(i), out_j.nth_edge_weight(j)) {
                (Some(w_i), Some(w_j)) => {
                    let delta = w_i.distance(w_j);
                    delta
                }
                _ => {
                    // Maximum penalty between two weighted edges
                    // NOTE: missing edges could be penalized more, but we already
                    // penalize for that in the node similarity measure.
                    Closed01::one()
                }
            }
            .get()
        });

        // calculate optimal edge weight assignement.
        let assignment = solve_assignment(&mut w);
        assert!(assignment.len() == max_deg);

        // The sum is the sum of all weight differences on the optimal `path`.
        // It's range is from 0.0 (perfect matching) to max_deg*1.0 (bad matching).
        let sum: f32 = assignment.iter().fold(0.0, |acc, &ij| acc + w.element_at(ij));

        debug_assert!(sum >= 0.0 && sum <= max_deg as f32);

        // we "invert" the normalized sum so that 1.0 means perfect matching and 0.0
        // no matching.
        let score = 1.0 - (sum / max_deg as f32);

        Closed01::new(score)
    }

    /// Sums the optimal assignment of the node similarities and normalizes (divides)
    /// by the min degree of both graphs.
    /// Used as default in the paper.
    pub fn score_sum_norm_min_degree(&self,
                                     node_assignment: Option<&[(usize, usize)]>)
                                     -> Closed01<f32> {
        let n = self.min_nodes();
        if n > 0 {
            Closed01::new(self.score_optimal_sum(node_assignment) / n as f32)
        } else {
            Closed01::zero()
        }
    }

    /// Sums the optimal assignment of the node similarities and normalizes (divides)
    /// by the min degree of both graphs.
    /// Penalizes the difference in size of graphs.
    pub fn score_sum_norm_max_degree(&self,
                                     node_assignment: Option<&[(usize, usize)]>)
                                     -> Closed01<f32> {
        let n = self.min_nodes();
        let m = self.max_nodes();

        if n > 0 {
            assert!(m > 0);
            Closed01::new(self.score_optimal_sum(node_assignment) / m as f32)
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

#[test]
fn test_matrix() {
    // A: 0 --> 1
    let in_a = vec![vec![], vec![0]];
    let out_a = vec![vec![1], vec![]];

    // B: 0 <-- 1
    let in_b = vec![vec![1], vec![]];
    let out_b = vec![vec![], vec![0]];

    let mut s = GraphSimilarityMatrix::new(Graph::new(&in_a, &out_a),
                                           Graph::new(&in_b, &out_b),
                                           IgnoreNodeColors);
    s.iterate(100, 0.1);

    println!("{:?}", s);
    assert_eq!(1, s.num_iterations());
    let mat = s.matrix();
    assert_eq!(2, mat.nrows());
    assert_eq!(2, mat.ncols());

    // A and B are isomorphic
    assert_eq!(1.0, mat[(0, 0)]);
    assert_eq!(1.0, mat[(0, 1)]);
    assert_eq!(1.0, mat[(1, 0)]);
    assert_eq!(1.0, mat[(1, 1)]);
}

#[test]
fn test_matrix_iter1() {
    let in_a = vec![vec![0, 0, 0]];
    let out_a = vec![vec![0, 0, 0]];

    let in_b = vec![vec![0, 0, 0, 0, 0]];
    let out_b = vec![vec![0, 0, 0, 0, 0]];

    let mut s = GraphSimilarityMatrix::new(Graph::new(&in_a, &out_a),
                                           Graph::new(&in_b, &out_b),
                                           IgnoreNodeColors);
    s.iterate(1, 0.1);

    assert_eq!(1, s.num_iterations());
    let mat = s.matrix();
    assert_eq!(3.0 / 5.0, mat[(0, 0)]);
}


#[test]
fn test_score() {
    // A: 0 --> 1
    let in_a = vec![vec![], vec![0]];
    let out_a = vec![vec![1], vec![]];

    // B: 0 <-- 1
    let in_b = vec![vec![1], vec![]];
    let out_b = vec![vec![], vec![0]];

    let mut s = GraphSimilarityMatrix::new(Graph::new(&in_a, &out_a),
                                           Graph::new(&in_b, &out_b),
                                           IgnoreNodeColors);
    s.iterate(100, 0.1);

    assert_eq!(1, s.num_iterations());

    // The score is 1.0 <=> A and B are isomorphic
    assert_eq!(1.0, s.score_sum_norm_min_degree(None).get());

    // The score is 1.0 <=> A and B are isomorphic
    assert_eq!(1.0, s.score_sum_norm_max_degree(None).get());
}
