#[path = "../tests/common/mod.rs"]
mod common;

use common::{load_graph, score_graphs};
use criterion::{criterion_group, criterion_main, Benchmark, Criterion};
use graph_neighbor_matching::graph::OwnedGraph;

const ITERS: usize = 100;
const EPS: f32 = 0.01;

fn assert_similarity(
    expected_similarity_in_percent: usize,
    a: &OwnedGraph<f32>,
    b: &OwnedGraph<f32>,
) {
    let graph_score = score_graphs(a, b, ITERS, EPS, false);
    assert_eq!(
        expected_similarity_in_percent,
        (graph_score * 100.0) as usize
    );
}

fn bench_collect_distribute_a(c: &mut Criterion) {
    let g_collect_distribute_3_3 = load_graph("graphs/collect_distribute_3_3.gml");
    let g_collect_distribute_3_3a = load_graph("graphs/collect_distribute_3_3a.gml");

    c.bench_function("score_graphs/collect_distribute_3_3a/79", move |b| {
        b.iter(|| {
            // Removing one link -> 79% similarity
            assert_similarity(79, &g_collect_distribute_3_3, &g_collect_distribute_3_3a);
        })
    });
}

fn bench_collect_distribute_b(c: &mut Criterion) {
    let g_collect_distribute_3_3 = load_graph("graphs/collect_distribute_3_3.gml");
    let g_collect_distribute_3_3b = load_graph("graphs/collect_distribute_3_3b.gml");

    c.bench_function("score_graphs/collect_distribute_3_3b/64", move |b| {
        b.iter(|| {
            // Removing two links -> 64% similarity
            assert_similarity(64, &g_collect_distribute_3_3, &g_collect_distribute_3_3b);
        })
    });
}

fn bench_skorpion(c: &mut Criterion) {
    let g_skorpion = load_graph("graphs/skorpion.gml");
    let g_skorpion_approx44 = load_graph("graphs/skorpion_approx44.gml");

    c.bench_function("score_graphs/skorpion/44", move |b| {
        b.iter(|| {
            assert_similarity(44, &g_skorpion, &g_skorpion_approx44);
        })
    });
}

fn bench_isomorphic_ffl35(c: &mut Criterion) {
    let g_ffl_iso_1_35 = load_graph("graphs/ffl/ffl_iso_1_35.gml");
    let g_ffl_iso_2_35 = load_graph("graphs/ffl/ffl_iso_2_35.gml");

    c.bench(
        "score_graphs",
        Benchmark::new("isomorphic_ffl_35/100", move |b| {
            b.iter(|| {
                assert_eq!(
                    1.0,
                    score_graphs(&g_ffl_iso_1_35, &g_ffl_iso_2_35, ITERS, EPS, true)
                );
            });
        })
        .sample_size(10),
    );
}

fn bench_isomorphic_ffl100(c: &mut Criterion) {
    let g_ffl_iso_1_100 = load_graph("graphs/ffl/ffl_iso_1_100.gml");
    let g_ffl_iso_2_100 = load_graph("graphs/ffl/ffl_iso_2_100.gml");

    c.bench(
        "score_graphs",
        Benchmark::new("isomorphic_ffl_100/100", move |b| {
            b.iter(|| {
                assert_eq!(
                    1.0,
                    score_graphs(&g_ffl_iso_1_100, &g_ffl_iso_2_100, ITERS, EPS, true)
                );
            });
        })
        .sample_size(10),
    );
}

criterion_group!(
    benches,
    bench_collect_distribute_a,
    bench_collect_distribute_b,
    bench_skorpion,
    bench_isomorphic_ffl35,
    bench_isomorphic_ffl100
);
criterion_main!(benches);
