mod common;

use common::{load_graph, score_graphs};
use graph_neighbor_matching::graph::{Edge, EdgeList, GraphBuilder, Node, OwnedGraph};
use graph_neighbor_matching::{IgnoreNodeColors, ScoreNorm, SimilarityMatrix};

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
    let a = graph(vec![
        node(vec![], vec![edge(1)]),
        node(vec![edge(0)], vec![]),
    ]);

    // B: 0 <-- 1
    let b = graph(vec![
        node(vec![edge(1)], vec![]),
        node(vec![], vec![edge(0)]),
    ]);

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
        node(
            vec![edge(0), edge(0), edge(0)],
            vec![edge(0), edge(0), edge(0)],
        ),
    ]);

    let b = graph(vec![
        node(
            vec![edge(0), edge(0), edge(0), edge(0), edge(0)],
            vec![edge(0), edge(0), edge(0), edge(0), edge(0)],
        ),
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
    let a = graph(vec![
        node(vec![], vec![edge(1)]),
        node(vec![edge(0)], vec![]),
    ]);

    // B: 0 <-- 1
    let b = graph(vec![
        node(vec![edge(1)], vec![]),
        node(vec![], vec![edge(0)]),
    ]);

    let mut s = SimilarityMatrix::new(&a, &b, IgnoreNodeColors);
    s.iterate(100, 0.1);

    assert_eq!(2, s.num_iterations());

    // The score is 1.0 <=> A and B are isomorphic
    assert_eq!(
        1.0,
        s.score_optimal_sum_norm(None, ScoreNorm::MinDegree).get()
    );

    // The score is 1.0 <=> A and B are isomorphic
    assert_eq!(
        1.0,
        s.score_optimal_sum_norm(None, ScoreNorm::MaxDegree).get()
    );
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
    assert_eq!(
        1.0,
        s.score_optimal_sum_norm(None, ScoreNorm::MinDegree).get()
    );

    // The score is 1.0 <=> A and B are isomorphic
    assert_eq!(
        1.0,
        s.score_optimal_sum_norm(None, ScoreNorm::MaxDegree).get()
    );
}

#[test]
fn test_isomorphic() {
    let a = load_graph("graphs/skorpion.gml");
    assert_eq!(1.0, score_graphs(&a, &a, 50, 0.1, false));
    assert_eq!(1.0, score_graphs(&a, &a, 1, 0.1, false));
    assert_eq!(1.0, score_graphs(&a, &a, 100, 0.01, true));

    let a = load_graph("graphs/collect_distribute_3_3.gml");
    assert_eq!(1.0, score_graphs(&a, &a, 50, 0.1, false));
    assert_eq!(1.0, score_graphs(&a, &a, 1, 0.1, false));
    assert_eq!(1.0, score_graphs(&a, &a, 100, 0.01, true));
}

#[test]
fn test_similarity() {
    let g = load_graph("graphs/collect_distribute_3_3.gml");
    let a = load_graph("graphs/collect_distribute_3_3a.gml");
    let b = load_graph("graphs/collect_distribute_3_3b.gml");

    // Removing one link -> 79% similarity
    assert_eq!(
        79,
        (score_graphs(&g, &a, 100, 0.01, false) * 100.0) as usize
    );

    // Removing two links -> 64% similarity
    assert_eq!(
        64,
        (score_graphs(&g, &b, 100, 0.01, false) * 100.0) as usize
    );
}

#[test]
fn test_similarity2() {
    let g = load_graph("graphs/skorpion.gml");
    let a = load_graph("graphs/skorpion_approx44.gml");

    assert_eq!(
        44,
        (score_graphs(&g, &a, 100, 0.01, false) * 100.0) as usize
    );
}

#[test]
fn test_similarity_neat() {
    let target = load_graph("graphs/neat/target.gml");
    let g = load_graph("graphs/neat/approx.gml");

    let score1 = score_graphs(&target, &g, 50, 0.01, false);
    let score2 = score_graphs(&g, &target, 50, 0.01, false);

    assert_eq!(56, (score1 * 100.0) as usize);
    assert_eq!(56, (score2 * 100.0) as usize);
}

#[test]
fn test_isomorphic_ffl() {
    let a = load_graph("graphs/ffl/ffl_iso_1_35.gml");
    let b = load_graph("graphs/ffl/ffl_iso_2_35.gml");

    assert_eq!(1.0, score_graphs(&a, &b, 100, 0.01, true));
}
