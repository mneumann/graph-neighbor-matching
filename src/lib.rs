/// A graph similarity score using neighbor matching according to [this paper][1].
///
/// [1]: http://arxiv.org/abs/1009.5290 "2010, Mladen Nikolic, Measuring Similarity
///      of Graph Nodes by Neighbor Matching"
///
/// TODO: Introduce EdgeWeight trait to abstract edge weight similarity.

extern crate nalgebra;
extern crate munkres;
extern crate closed01;

use nalgebra::{DMat, Shape, ApproxEq};
use munkres::{WeightMatrix, solve_assignment};
use std::cmp;
use std::mem;
use std::fmt;
use closed01::Closed01;

trait Edges {
    /// The number of edges
    fn num_edges(&self) -> usize;

    /// Returns the target node of the nth-edge
    fn nth_edge(&self, n: usize) -> Option<usize>;

    /// Returns the nth edge weight. We expect edge weights to be
    /// normalized in the range [0, 1].
    fn nth_edge_weight(&self, n: usize) -> Option<Closed01<f32>>;
}

#[derive(Debug)]
pub struct Edge {
    /// Node index type. Our graphs never exceed 4 billion nodes.
    pointing_node: u32,
    weight: Closed01<f32>,
}

impl Edge {
    pub fn new(node_idx: usize, weight: Closed01<f32>) -> Edge {
        assert!(node_idx <= u32::max_value() as usize);
        Edge {
            pointing_node: node_idx as u32,
            weight: weight,
        }
    }
}

impl<'a> Edges for &'a [Edge] {
    #[inline]
    fn num_edges(&self) -> usize {
        self.len()
    }
    #[inline]
    fn nth_edge(&self, n: usize) -> Option<usize> {
        self.get(n).map(|n| n.pointing_node as usize)
    }
    #[inline]
    fn nth_edge_weight(&self, n: usize) -> Option<Closed01<f32>> {
        self.get(n).map(|n| n.weight)
    }
}

#[derive(Copy, Clone, Debug)]
pub struct Graph<'a> {
    in_edges: &'a [Vec<Edge>],
    out_edges: &'a [Vec<Edge>],
}

impl<'a> Graph<'a> {
    pub fn new(in_edges: &'a [Vec<Edge>], out_edges: &'a [Vec<Edge>]) -> Graph<'a> {
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
    fn in_edges_of(&self, node_idx: usize) -> &[Edge] {
        &self.in_edges[node_idx]
    }

    #[inline]
    fn out_edges_of(&self, node_idx: usize) -> &[Edge] {
        &self.out_edges[node_idx]
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
    let max_deg = cmp::max(n_i.num_edges(), n_j.num_edges());
    let min_deg = cmp::min(n_i.num_edges(), n_j.num_edges());

    debug_assert!(min_deg <= max_deg);

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

#[derive(Debug)]
pub enum ScoreNorm {
    /// Divide by minimum graph or node degree
    MinDegree,

    /// Divide by maximum graph or node degree
    MaxDegree,
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
                    new_x[(i, j)] = Closed01::avg(s_next(self.graph_a.in_edges_of(i),
                                                         self.graph_b.in_edges_of(j),
                                                         x),
                                                  s_next(self.graph_a.out_edges_of(i),
                                                         self.graph_b.out_edges_of(j),
                                                         x))
                                        .mul(scale)
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

        let max_deg = cmp::max(out_i.len(), out_j.len());

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

#[cfg(test)]
fn edge(i: usize) -> Edge {
    Edge::new(i, Closed01::zero())
}

#[test]
fn test_matrix() {
    // A: 0 --> 1
    let in_a = vec![vec![], vec![edge(0)]];
    let out_a = vec![vec![edge(1)], vec![]];

    // B: 0 <-- 1
    let in_b = vec![vec![edge(1)], vec![]];
    let out_b = vec![vec![], vec![edge(0)]];

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
    let in_a = vec![vec![edge(0), edge(0), edge(0)]];
    let out_a = vec![vec![edge(0), edge(0), edge(0)]];

    let in_b = vec![vec![edge(0), edge(0), edge(0), edge(0), edge(0)]];
    let out_b = vec![vec![edge(0), edge(0), edge(0), edge(0), edge(0)]];

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
    let in_a = vec![vec![], vec![edge(0)]];
    let out_a = vec![vec![edge(1)], vec![]];

    // B: 0 <-- 1
    let in_b = vec![vec![edge(1)], vec![]];
    let out_b = vec![vec![], vec![edge(0)]];

    let mut s = GraphSimilarityMatrix::new(Graph::new(&in_a, &out_a),
                                           Graph::new(&in_b, &out_b),
                                           IgnoreNodeColors);
    s.iterate(100, 0.1);

    assert_eq!(1, s.num_iterations());

    // The score is 1.0 <=> A and B are isomorphic
    assert_eq!(1.0,
               s.score_optimal_sum_norm(None, ScoreNorm::MinDegree).get());

    // The score is 1.0 <=> A and B are isomorphic
    assert_eq!(1.0,
               s.score_optimal_sum_norm(None, ScoreNorm::MaxDegree).get());
}
