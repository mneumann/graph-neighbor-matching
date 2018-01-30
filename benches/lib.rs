#![feature(test)]

extern crate test;
extern crate test_helper;

use test_helper::{load_graph, score_graphs};
use test::Bencher;

#[bench]
fn bench_similarity(bench: &mut Bencher) {
    let g = load_graph("graphs/collect_distribute_3_3.gml");
    let a = load_graph("graphs/collect_distribute_3_3a.gml");
    let b = load_graph("graphs/collect_distribute_3_3b.gml");

    bench.iter(|| {
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
    });
}

#[bench]
fn bench_similarity2(bench: &mut Bencher) {
    let g = load_graph("graphs/skorpion.gml");
    let a = load_graph("graphs/skorpion_approx44.gml");

    bench.iter(|| {
        assert_eq!(
            44,
            (score_graphs(&g, &a, 100, 0.01, false) * 100.0) as usize
        );
    });
}
