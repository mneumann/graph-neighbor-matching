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
}

impl<'a> Edges for &'a [Idx] {
    fn len(&self) -> usize {
        let x: &[Idx] = self;
        x.len()
    }
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

#[inline]
/// Calculates the similarity matrix for two graphs A and B.
///
/// `in_a`:  Incoming edge list for each node of graph A
/// `in_b`:  Incoming edge list for each node of graph B
/// `out_a`: Outgoing edge list for each node of graph A
/// `out_b`: Outgoing edge list for each node of graph B
/// `eps`:   When to stop the iteration
/// `stop_after_iter`: Stop after iteration (Calculate x(stop_after_iter))
///
/// Returns (number of iterations, similarity matrix `x`)
pub fn neighbor_matching_matrix<F>(in_a: &[Vec<Idx>],
                                   in_b: &[Vec<Idx>],
                                   out_a: &[Vec<Idx>],
                                   out_b: &[Vec<Idx>],
                                   eps: f32,
                                   stop_after_iter: usize,
                                   node_color_scale: &F)
                                   -> (usize, DMat<f32>)
    where F: Fn((usize, usize)) -> f32
{
    let (na, nb) = (in_a.len(), in_b.len());
    assert!((na, nb) == (out_a.len(), out_b.len()));

    // `x` is the node-similarity matrix.
    // we initialize `x`, so that x[i,j]=1 for all i in A.edges() and j in
    // B.edges().
    let mut x: DMat<f32> = DMat::from_fn(na, nb, |i, j| {
        node_color_scale((i, j)) *
        if in_a[i].len() + out_a[i].len() > 0 && in_b[j].len() + out_b[j].len() > 0 {
            1.0
        } else {
            0.0
        }
    });

    let mut iter = 0;
    let mut new_x: DMat<f32> = DMat::new_zeros(na, nb);

    loop {
        if x.approx_eq_eps(&new_x, &eps) || iter >= stop_after_iter {
            break;
        }

        next_x(&x, &mut new_x, in_a, in_b, out_a, out_b, node_color_scale);
        mem::swap(&mut new_x, &mut x);
        iter += 1;
    }

    (iter, x)
}

/// Different similarity measures can be constructed from the node similarity matrix.
pub enum ScoreMethod {
    /// Sums the optimal assignment of the node similarities and normalizes (divides)
    /// by the min degree of both graphs.
    /// Used as default in the paper.
    SumNormMinDegree,

    /// Sums the optimal assignment of the node similarities and normalizes (divides)
    /// by the min degree of both graphs.
    /// Penalizes the difference in size of graphs.
    SumNormMaxDegree,

    /// Calculates the average over the whole node similarity matrix. This is faster,
    /// as no assignment has to be found. "Graphs with greater number of automorphisms
    /// would be considered to be more self-similar than graphs without automorphisms."
    Average,
}


#[inline]
/// Calculates the similiarity of two graphs.
///
/// For parameters see `neighbor_matching_matrix`.
pub fn neighbor_matching_score<F>(in_a: &[Vec<Idx>],
                                  in_b: &[Vec<Idx>],
                                  out_a: &[Vec<Idx>],
                                  out_b: &[Vec<Idx>],
                                  eps: f32,
                                  stop_after_iter: usize,
                                  node_color_scale: &F,
                                  score_method: ScoreMethod)
                                  -> (usize, f32)
    where F: Fn((usize, usize)) -> f32
{
    let (iter, x) = neighbor_matching_matrix(in_a,
                                             in_b,
                                             out_a,
                                             out_b,
                                             eps,
                                             stop_after_iter,
                                             node_color_scale);
    let n = cmp::min(x.nrows(), x.ncols());
    let m = cmp::max(x.nrows(), x.ncols());
    if n == 0 {
        return (iter, 0.0);
    }

    match score_method {
        ScoreMethod::SumNormMinDegree | ScoreMethod::SumNormMaxDegree => {
            let norm = match score_method {
                ScoreMethod::SumNormMinDegree => n as f32,
                ScoreMethod::SumNormMaxDegree => m as f32,
                _ => unreachable!(),
            };

            let mut w = WeightMatrix::from_fn(n, |ij| x[ij]);
            let assignment = solve_assignment(&mut w);
            assert!(assignment.len() == n);
            let score: f32 = assignment.iter().fold(0.0, |acc, &ab| acc + x[ab]);
            (iter, score / norm)
        }
        ScoreMethod::Average => {
            let sum: f32 = x.as_vec().iter().fold(0.0, |acc, &v| acc + v);
            let len = x.as_vec().len();
            assert!(len > 0);
            (iter, sum / len as f32)
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

    let (iter, mat) = neighbor_matching_matrix(&in_a, &in_b, &out_a, &out_b, 0.1, 100, &|_| 1.0);
    println!("{:?}", mat);
    assert_eq!(iter, 1);
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

    let (iter, mat) = neighbor_matching_matrix(&in_a, &in_b, &out_a, &out_b, 0.1, 1, &|_| 1.0);
    assert_eq!(iter, 1);
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

    let (iter, score) = neighbor_matching_score(&in_a,
                                                &in_b,
                                                &out_a,
                                                &out_b,
                                                0.1,
                                                100,
                                                &|_| 1.0,
                                                ScoreMethod::SumNormMinDegree);
    assert_eq!(iter, 1);

    // The score is 1.0 <=> A and B are isomorphic
    assert_eq!(1.0, score);

    let (iter, score) = neighbor_matching_score(&in_a,
                                                &in_b,
                                                &out_a,
                                                &out_b,
                                                0.1,
                                                100,
                                                &|_| 1.0,
                                                ScoreMethod::SumNormMaxDegree);
    assert_eq!(iter, 1);

    // The score is 1.0 <=> A and B are isomorphic
    assert_eq!(1.0, score);
}
