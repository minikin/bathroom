#![allow(
    missing_docs,
    clippy::missing_docs_in_private_items,
    clippy::unwrap_used,
    clippy::similar_names
)]
use std::collections::HashMap;

use bathroom::BathroomMap;
use criterion::{criterion_group, criterion_main, Criterion};
use proptest::{ prelude::{ any, Strategy}, strategy::ValueTree, test_runner::TestRunner};

const ITEMS_AMOUNT: usize = 1000;
const SAMPLE_SIZE: usize = 10;

fn hash_map_benches(c: &mut Criterion) {
    let mut runner = TestRunner::default();
    let items = any::<[(String, String); ITEMS_AMOUNT]>()
    .new_tree(&mut runner)
    .unwrap()
    .current();


    let mut group = c.benchmark_group("Hash map comparison benchmark");
    group.sample_size(SAMPLE_SIZE);
    let mut bathroom_map = BathroomMap::new();
    let mut rust_map = HashMap::new();
    group.bench_function("bathroom insert", |b| {
        b.iter(
            || {
            for (key, value) in items.clone() {
                bathroom_map.insert(key, value);
            }
            
        });
    });
    group.bench_function("rust std insert", |b| {
        b.iter(
            || {
            for (key, value) in items.clone() {
                rust_map.insert(key, value);
            }
            
        });
    });
    group.bench_function("bathroom get", |b| {
        b.iter(|| {
            for (key, _) in &items {
                let _ = bathroom_map.get(key);
            }
        });
    });
    group.bench_function("rust std get", |b| {
        b.iter(|| {
            for (key, _) in &items {
                let _ = rust_map.get(key);
            }
        });
    });
    group.finish();
}

criterion_group!(benches, hash_map_benches);

criterion_main!(benches);