extern crate test_helper;

use test_helper::{load_graph, score_graphs};

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
