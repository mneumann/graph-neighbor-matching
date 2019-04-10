#[path = "../tests/common/mod.rs"]
mod common;

use common::{load_graph, score_graphs};
use criterion::{criterion_group, criterion_main, Criterion};

fn bench_score_graphs(c: &mut Criterion) {
    c.bench_function("score_graphs(collect_distribute_3_3, a, 79)", |b| {
        let g = load_graph("graphs/collect_distribute_3_3.gml");
        let ga = load_graph("graphs/collect_distribute_3_3a.gml");

        b.iter(|| {
            // Removing one link -> 79% similarity
            assert_eq!(
                79,
                (score_graphs(&g, &ga, 100, 0.01, false) * 100.0) as usize
            );
        })
    });

    c.bench_function("score_graphs(collect_distribute_3_3, b, 64)", |b| {
        let g = load_graph("graphs/collect_distribute_3_3.gml");
        let gb = load_graph("graphs/collect_distribute_3_3b.gml");

        b.iter(|| {
            // Removing two links -> 64% similarity
            assert_eq!(
                64,
                (score_graphs(&g, &gb, 100, 0.01, false) * 100.0) as usize
            );
        })
    });

    c.bench_function("score_graphs(skorpion, 44)", |b| {
        let g = load_graph("graphs/skorpion.gml");
        let ga = load_graph("graphs/skorpion_approx44.gml");

        b.iter(|| {
            assert_eq!(
                44,
                (score_graphs(&g, &ga, 100, 0.01, false) * 100.0) as usize
            );
        })
    });

    c.bench_function("score_graphs(isomorphic_ffl_35)", |b| {
        let ga = load_graph("graphs/ffl/ffl_iso_1_35.gml");
        let gb = load_graph("graphs/ffl/ffl_iso_2_35.gml");

        b.iter(|| {
            assert_eq!(1.0, score_graphs(&ga, &gb, 100, 0.01, true));
        })
    });

    c.bench_function("score_graphs(isomorphic_ffl_100)", |b| {
        let ga = load_graph("graphs/ffl/ffl_iso_1_100.gml");
        let gb = load_graph("graphs/ffl/ffl_iso_2_100.gml");

        b.iter(|| {
            assert_eq!(1.0, score_graphs(&ga, &gb, 100, 0.01, true));
        });
    });
}

criterion_group!(benches, bench_score_graphs);
criterion_main!(benches);
