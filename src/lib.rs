/// A graph similarity score using neighbor matching according to [this paper][1].
///
/// [1]: http://arxiv.org/abs/1009.5290 "2010, Mladen Nikolic, Measuring Similarity
///      of Graph Nodes by Neighbor Matching"

extern crate nalgebra;
extern crate munkres;

use nalgebra::{DMat, Shape, ApproxEq};
use munkres::{WeightMatrix, solve_assignment};
use std::cmp;
use std::mem;

pub type Idx = u32;

trait Edges {
    /// The number of edges
    fn len(&self) -> usize;

    /// Returns the target node of the nth-edge
    fn nth_edge(&self, n: usize) -> usize;

    /// Returns the nth edge weight
    fn nth_edge_weight(&self, n: usize) -> f32 {
        panic!();
    }
}

impl<'a> Edges for &'a [Idx] {
    #[inline]
    fn len(&self) -> usize {
        let x: &[Idx] = self;
        x.len()
    }
    #[inline]
    fn nth_edge(&self, n: usize) -> usize {
        self[n] as usize
    }
}

#[inline]
/// Calculates the similarity of two nodes `i` and `j`.
///
/// `n_i` contains the neighborhood of i (either in or out neighbors, not both)
/// `n_j` contains the neighborhood of j (either in or out neighbors, not both)
/// `x`   the similarity matrix.
fn s_next<T: Edges>(n_i: T, n_j: T, x: &DMat<f32>) -> f32 {
    let max_deg = cmp::max(n_i.len(), n_j.len());
    let min_deg = cmp::min(n_i.len(), n_j.len());

    if min_deg == 0 {
        // in the paper, 0/0 is defined as 1.0
        return 1.0;
    }

    assert!(min_deg > 0 && max_deg > 0);

    // map indicies from 0..min(degree) to the node indices
    let mapidx = |(a, b)| (n_i.nth_edge(a), n_j.nth_edge(b));

    let mut w = WeightMatrix::from_fn(min_deg, |ab| x[mapidx(ab)]);

    let assignment = solve_assignment(&mut w);
    assert!(assignment.len() == min_deg);

    let sum: f32 = assignment.iter().fold(0.0, |acc, &ab| acc + x[mapidx(ab)]);

    return sum / max_deg as f32;
}

#[inline]
/// Calculates x[k+1]
///
/// `node_color_scale((i,j))`: If two nodes `i` (of graph A) and `j` (of graph B)
/// are of different color, this can be set to return 0.0. Alternatively a
/// node-color distance (within 0...1) could be used to penalize.
fn next_x<F>(x: &DMat<f32>,
             new_x: &mut DMat<f32>,
             in_a: &[Vec<Idx>],
             in_b: &[Vec<Idx>],
             out_a: &[Vec<Idx>],
             out_b: &[Vec<Idx>],
             node_color_scale: F)
    where F: Fn((usize, usize)) -> f32
{
    let shape = x.shape();
    assert!(shape == new_x.shape());

    for i in 0..shape.0 {
        for j in 0..shape.1 {
            let in_i: &[Idx] = &in_a[i];
            let in_j: &[Idx] = &in_b[j];
            let out_i: &[Idx] = &out_a[i];
            let out_j: &[Idx] = &out_b[j];
            new_x[(i, j)] = node_color_scale((i, j)) *
                            (s_next(in_i, in_j, x) + s_next(out_i, out_j, x)) /
                            2.0;
        }
    }
}

#[derive(Debug)]
pub struct NodeSimilarityMatrix {
    // number of nodes in graph A
    na: usize,
    // number of nodes in graph B
    nb: usize,
    // current version of similarity matrix
    current: DMat<f32>,
    // previous version of similarity matrix
    previous: DMat<f32>,
    // current number of iterations
    num_iters: usize,
}

impl NodeSimilarityMatrix {
    pub fn init<F>(in_a: &[Vec<Idx>],
                   in_b: &[Vec<Idx>],
                   out_a: &[Vec<Idx>],
                   out_b: &[Vec<Idx>],
                   node_color_scale: &F)
                   -> NodeSimilarityMatrix
        where F: Fn((usize, usize)) -> f32
    {
        let (na, nb) = (in_a.len(), in_b.len());
        assert!((na, nb) == (out_a.len(), out_b.len()));

        // `x` is the node-similarity matrix.
        // we initialize `x`, so that x[i,j]=1 for all i in A.edges() and j in
        // B.edges().
        let x: DMat<f32> = DMat::from_fn(na, nb, |i, j| {
            node_color_scale((i, j)) *
            if in_a[i].len() + out_a[i].len() > 0 && in_b[j].len() + out_b[j].len() > 0 {
                1.0
            } else {
                0.0
            }
        });

        let new_x: DMat<f32> = DMat::new_zeros(na, nb);

        NodeSimilarityMatrix {
            na: na,
            nb: nb,
            current: x,
            previous: new_x,
            num_iters: 0,
        }
    }

    fn in_eps(&self, eps: f32) -> bool {
        self.previous.approx_eq_eps(&self.current, &eps)
    }

    fn next<F>(&mut self,
               in_a: &[Vec<Idx>],
               in_b: &[Vec<Idx>],
               out_a: &[Vec<Idx>],
               out_b: &[Vec<Idx>],
               node_color_scale: &F)
        where F: Fn((usize, usize)) -> f32
    {
        next_x(&self.current,
               &mut self.previous,
               in_a,
               in_b,
               out_a,
               out_b,
               node_color_scale);
        mem::swap(&mut self.previous, &mut self.current);
        self.num_iters += 1;
    }

    #[inline]
    /// Calculates the similarity matrix for two graphs A and B.
    ///
    /// `in_a`:  Incoming edge list for each node of graph A
    /// `in_b`:  Incoming edge list for each node of graph B
    /// `out_a`: Outgoing edge list for each node of graph A
    /// `out_b`: Outgoing edge list for each node of graph B
    /// `eps`:   When to stop the iteration
    /// `stop_after_iter`: Stop after iteration (Calculate x(stop_after_iter))
    pub fn calc<F>(in_a: &[Vec<Idx>],
                   in_b: &[Vec<Idx>],
                   out_a: &[Vec<Idx>],
                   out_b: &[Vec<Idx>],
                   eps: f32,
                   stop_after_iter: usize,
                   node_color_scale: &F)
                   -> NodeSimilarityMatrix
        where F: Fn((usize, usize)) -> f32
    {
        let mut x = NodeSimilarityMatrix::init(in_a, in_b, out_a, out_b, node_color_scale);

        for _ in 0..stop_after_iter {
            if x.in_eps(eps) {
                break;
            }
            x.next(in_a, in_b, out_a, out_b, node_color_scale);
        }

        return x;
    }

    fn matrix(self) -> DMat<f32> {
        self.current
    }

    pub fn num_iterations(&self) -> usize {
        self.num_iters
    }

    fn optimal_node_assignment(&self, n: usize) -> Vec<(usize, usize)> {
        let x = &self.current;
        assert!(n > 0);
        let mut w = WeightMatrix::from_fn(n, |ij| x[ij]);
        let assignment = solve_assignment(&mut w);
        assert!(assignment.len() == n);
        assignment
    }

    fn score_optimal_sum(&self, n: usize) -> f32 {
        self.optimal_node_assignment(n).iter().fold(0.0, |acc, &ab| acc + self.current[ab])
    }

    /// Calculate a measure how good the edge weights match up.
    ///
    /// We start by calculating the optimal node assignment between nodes of graph A and
    /// graph B, then compare all outgoing edges of similar nodes by again using an
    /// assignment.
    pub fn score_outgoing_edge_weights(&self) -> f32 {
        // XXX
        0.0
    }

    /// Sums the optimal assignment of the node similarities and normalizes (divides)
    /// by the min degree of both graphs.
    /// Used as default in the paper.
    pub fn score_sum_norm_min_degree(&self) -> f32 {
        let x = &self.current;
        let n = cmp::min(x.nrows(), x.ncols());
        if n > 0 {
            self.score_optimal_sum(n) / n as f32
        } else {
            0.0
        }
    }

    /// Sums the optimal assignment of the node similarities and normalizes (divides)
    /// by the min degree of both graphs.
    /// Penalizes the difference in size of graphs.
    pub fn score_sum_norm_max_degree(&self) -> f32 {
        let x = &self.current;
        let n = cmp::min(x.nrows(), x.ncols());
        let m = cmp::max(x.nrows(), x.ncols());

        if n > 0 {
            assert!(m > 0);
            self.score_optimal_sum(n) / m as f32
        } else {
            0.0
        }
    }

    /// Calculates the average over the whole node similarity matrix. This is faster,
    /// as no assignment has to be found. "Graphs with greater number of automorphisms
    /// would be considered to be more self-similar than graphs without automorphisms."
    pub fn score_average(&self) -> f32 {
        let x = &self.current;
        let n = cmp::min(x.nrows(), x.ncols());
        if n > 0 {
            let sum: f32 = x.as_vec().iter().fold(0.0, |acc, &v| acc + v);
            let len = x.as_vec().len();
            assert!(len > 0);
            sum / len as f32
        } else {
            0.0
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

    let s = NodeSimilarityMatrix::calc(&in_a, &in_b, &out_a, &out_b, 0.1, 100, &|_| 1.0);
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

    let s = NodeSimilarityMatrix::calc(&in_a, &in_b, &out_a, &out_b, 0.1, 1, &|_| 1.0);
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

    let s = NodeSimilarityMatrix::calc(&in_a, &in_b, &out_a, &out_b, 0.1, 100, &|_| 1.0);

    assert_eq!(1, s.num_iterations());

    // The score is 1.0 <=> A and B are isomorphic
    assert_eq!(1.0, s.score_sum_norm_min_degree());

    // The score is 1.0 <=> A and B are isomorphic
    assert_eq!(1.0, s.score_sum_norm_max_degree());
}
