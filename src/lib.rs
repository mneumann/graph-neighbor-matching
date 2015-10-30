/// A graph similarity score using neighbor matching according to [this paper][1].
///
/// [1]: http://arxiv.org/abs/1009.5290 "2010, Mladen Nikolic, Measuring Similarity of Graph Nodes by Neighbor Matching"

extern crate nalgebra;
extern crate munkres;

use nalgebra::{DMat, Shape, ApproxEq};
use munkres::{WeightMatrix, solve_assignment};
use std::cmp;
use std::mem;

// n_i contains the neighborhood of i (either in or out neighbors, not both)
// n_j contains the neighborhood of j (either in or out neighbors, not both)
fn s_next(n_i: &[usize], n_j: &[usize], x: &DMat<f32>) -> f32 {
    let max_deg = cmp::max(n_i.len(), n_j.len());
    let min_deg = cmp::min(n_i.len(), n_j.len());

    if min_deg == 0 {
        // in the paper, 0/0 is defined as 1.0
        return 1.0;
    }

    // map indicies from 0..min(degree) to the node indices
    let mapidx = |(a, b)| (n_i[a], n_j[b]);

    let mut w = WeightMatrix::from_fn(min_deg, |ab| x[mapidx(ab)]);

    let assignment = solve_assignment(&mut w);
    assert!(assignment.len() == min_deg);

    let sum: f32 = assignment.iter().fold(0.0, |acc, &ab| acc + x[mapidx(ab)]);

    return sum / max_deg as f32;
}

// Calculates x[k+1]
fn next_x(x: &DMat<f32>,
          new_x: &mut DMat<f32>,
          in_a: &[Vec<usize>],
          in_b: &[Vec<usize>],
          out_a: &[Vec<usize>],
          out_b: &[Vec<usize>]) {
    let shape = x.shape();
    assert!(shape == new_x.shape());

    for i in 0..shape.0 {
        for j in 0..shape.1 {
            new_x[(i, j)] = (s_next(&in_a[i], &in_b[j], x) + s_next(&out_a[i], &out_b[j], x)) / 2.0;
        }
    }
}

/// in_a:  Incoming edge list for each node of graph A
/// in_b:  Incoming edge list for each node of graph B
/// out_a: Outgoing edge list for each node of graph A
/// out_b: Outgoing edge list for each node of graph B
/// eps:   When to stop the iteration
/// stop_after_iter: Stop after iteration (Calculate x(stop_after_iter))
///
/// Returns (number of iterations, similarity matrix `x`)
pub fn neighbor_matching_matrix(in_a: &[Vec<usize>],
                                in_b: &[Vec<usize>],
                                out_a: &[Vec<usize>],
                                out_b: &[Vec<usize>],
                                eps: f32,
                                stop_after_iter: usize)
                                -> (usize, DMat<f32>) {
    let (na, nb) = (in_a.len(), in_b.len());
    assert!((na, nb) == (out_a.len(), out_b.len()));

    // `x` is the node-similarity matrix.
    // we initialize `x`, so that x[i,j]=1 for all i in A.edges() and j in
    // B.edges().
    let mut x: DMat<f32> = DMat::from_fn(na, nb, |i, j| {
        // XXX: Is that correct?
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

        next_x(&x, &mut new_x, &in_a, &in_b, &out_a, &out_b);
        mem::swap(&mut new_x, &mut x);
        iter += 1;
    }

    (iter, x)
}


/// For parameters see `neighbor_matching_matrix`.
pub fn neighbor_matching_score(in_a: &[Vec<usize>],
                               in_b: &[Vec<usize>],
                               out_a: &[Vec<usize>],
                               out_b: &[Vec<usize>],
                               eps: f32,
                               stop_after_iter: usize)
                               -> (usize, f32) {


    let (iter, x) = neighbor_matching_matrix(in_a, in_b, out_a, out_b, eps, stop_after_iter);
    let n = cmp::min(x.nrows(), x.ncols());
    if n == 0 {
        return (iter, 0.0);
    }
    let mut w = WeightMatrix::from_fn(n, |ij| x[ij]);

    let assignment = solve_assignment(&mut w);
    assert!(assignment.len() == n);

    let score: f32 = assignment.iter().fold(0.0, |acc, &ab| acc + x[ab]);
    let score = score / n as f32;

    (iter, score)
}

#[test]
fn test_matrix() {
    // A: 0 --> 1
    let in_a = vec![vec![], vec![0]];
    let out_a = vec![vec![1], vec![]];

    // B: 0 <-- 1
    let in_b = vec![vec![1], vec![]];
    let out_b = vec![vec![], vec![0]];

    let (iter, mat) = neighbor_matching_matrix(&in_a, &in_b, &out_a, &out_b, 0.1, 100);
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

    let (iter, mat) = neighbor_matching_matrix(&in_a, &in_b, &out_a, &out_b, 0.1, 1);
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

    let (iter, score) = neighbor_matching_score(&in_a, &in_b, &out_a, &out_b, 0.1, 100);
    assert_eq!(iter, 1);

    // The score is 1.0 <=> A and B are isomorphic
    assert_eq!(1.0, score);
}
