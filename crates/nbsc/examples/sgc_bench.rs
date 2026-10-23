//! Linear (SGC-style) NBSC-vs-GCN comparison on the canonical Planetoid
//! split, for Cora, Citeseer, and PubMed -- see `sgc.rs` module docs for
//! why this exists (PubMed's 19717 nodes don't fit the dense Burn layers'
//! `n x n` tensors in the memory available on the machine this thesis's
//! live experiments were run on).
//!
//! Includes a weight-decay (L2) grid, addressing the same "was
//! regularization actually tuned, not just defaulted to 0" question as
//! `NBSC_WEIGHT_DECAY` in `thesis_bench.rs`, applied here to both
//! propagators identically and chosen by validation accuracy.
//!
//! Run with: `cargo run --release --example sgc_bench -- <dataset>`
//! where `<dataset>` is `cora`, `citeseer`, or `pubmed` (default: runs all
//! three in sequence).

use nbsc::dataset::Dataset;
use nbsc::sgc::{concat_taps, gcn_propagate_taps, standardize_columns, SoftmaxClassifier};
use nbsc::spectral::{FeatureMatrix, NbscFilterBank};

const K_TAPS: usize = 2;
const EPOCHS: usize = 300;
const LR: f64 = 0.1;
const N_SEEDS: u64 = 3;
const WD_GRID: [f64; 5] = [0.0, 1e-4, 5e-4, 1e-3, 1e-2];

fn mean_std(values: &[f32]) -> (f32, f32) {
    let n = values.len().max(1) as f32;
    let mean = values.iter().sum::<f32>() / n;
    let var = values.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / n;
    (mean, var.sqrt())
}

fn to_feature_matrix(ds: &Dataset) -> FeatureMatrix {
    FeatureMatrix::from_rows(ds.graph.n, ds.features.f, ds.features.data.clone())
}

/// Runs the full pipeline (propagate -> concat -> standardize -> grid
/// search weight decay on val -> report test) for one propagator.
fn run_propagator(
    name: &str,
    feat: &FeatureMatrix,
    labels: &[usize],
    n_classes: usize,
    train_idx: &[usize],
    val_idx: &[usize],
    test_idx: &[usize],
) {
    let std_feat = standardize_columns(feat);
    let mut best_wd = WD_GRID[0];
    let mut best_val = f32::NEG_INFINITY;
    let mut best_test_accs: Vec<f32> = Vec::new();

    for &wd in &WD_GRID {
        let mut val_accs = Vec::with_capacity(N_SEEDS as usize);
        let mut test_accs = Vec::with_capacity(N_SEEDS as usize);
        for seed in 0..N_SEEDS {
            let mut clf = SoftmaxClassifier::new(std_feat.f, n_classes, seed);
            clf.train(&std_feat, labels, train_idx, EPOCHS, LR, wd);
            val_accs.push(clf.accuracy(&std_feat, labels, val_idx));
            test_accs.push(clf.accuracy(&std_feat, labels, test_idx));
        }
        let (val_mean, val_std) = mean_std(&val_accs);
        let (test_mean, test_std) = mean_std(&test_accs);
        println!(
            "  {name:<12} wd={wd:<8.1e} | val {val_mean:.3}+-{val_std:.3} | test {test_mean:.3}+-{test_std:.3} (n_seeds={})",
            val_accs.len()
        );
        if val_mean > best_val {
            best_val = val_mean;
            best_wd = wd;
            best_test_accs = test_accs;
        }
    }
    let (test_mean, test_std) = mean_std(&best_test_accs);
    println!(
        "  {name:<12} BEST (by val): wd={best_wd:<8.1e} | test {test_mean:.3}+-{test_std:.3}\n"
    );
}

fn run_dataset(name: &str, ds: Dataset) {
    println!(
        "\n=== {name} (canonical Planetoid split): n={}, m={}, classes={}, f={}, train={}, val={}, test={} ===",
        ds.graph.n,
        ds.graph.m(),
        ds.num_classes(),
        ds.features.f,
        ds.train_indices().len(),
        ds.val_indices().len(),
        ds.test_indices().len(),
    );

    let feat = to_feature_matrix(&ds);
    let n_classes = ds.num_classes();
    let train_idx = ds.train_indices();
    let val_idx = ds.val_indices();
    let test_idx = ds.test_indices();

    let t = std::time::Instant::now();
    let filter_bank = NbscFilterBank::build(&ds.graph, 60, 0);
    println!("rho_B (non-backtracking spectral radius) = {:.6} ({:.1}s to estimate)", filter_bank.rho_b, t.elapsed().as_secs_f32());

    let t = std::time::Instant::now();
    let nbsc_taps = filter_bank.apply_taps(&ds.graph, &feat, K_TAPS);
    let nbsc_feat = concat_taps(&nbsc_taps);
    println!("NBSC taps [T_0..T_{K_TAPS}] X computed ({:.1}s), concatenated width = {}", t.elapsed().as_secs_f32(), nbsc_feat.f);

    let t = std::time::Instant::now();
    let gcn_taps = gcn_propagate_taps(&ds.graph, &feat, K_TAPS);
    let gcn_feat = concat_taps(&gcn_taps);
    println!("GCN taps [S^0..S^{K_TAPS}] X computed ({:.1}s), concatenated width = {}\n", t.elapsed().as_secs_f32(), gcn_feat.f);

    let t = std::time::Instant::now();
    run_propagator("NBSC-linear", &nbsc_feat, &ds.labels, n_classes, &train_idx, &val_idx, &test_idx);
    println!("  (NBSC-linear grid: {:.1}s)", t.elapsed().as_secs_f32());

    let t = std::time::Instant::now();
    run_propagator("GCN-linear", &gcn_feat, &ds.labels, n_classes, &train_idx, &val_idx, &test_idx);
    println!("  (GCN-linear grid: {:.1}s)", t.elapsed().as_secs_f32());

    // Raw-feature (no propagation) baseline, for reference -- shows how
    // much either propagator actually buys over just the node's own
    // features, exercising the same classifier/regularization pipeline.
    let t = std::time::Instant::now();
    run_propagator("raw (no-prop)", &feat, &ds.labels, n_classes, &train_idx, &val_idx, &test_idx);
    println!("  (raw-feature grid: {:.1}s)", t.elapsed().as_secs_f32());
}

fn main() {
    let which: Vec<String> = std::env::args().skip(1).collect();
    let targets: Vec<String> = if which.is_empty() {
        vec!["cora".into(), "citeseer".into(), "pubmed".into()]
    } else {
        which
    };

    for target in targets {
        match target.as_str() {
            "cora" => run_dataset("Cora", Dataset::load_cora_planetoid().expect("cora_planetoid should load")),
            "citeseer" => run_dataset("Citeseer", Dataset::load_citeseer_planetoid().expect("citeseer_planetoid should load")),
            "pubmed" => run_dataset("PubMed", Dataset::load_pubmed_planetoid().expect("pubmed_planetoid should load")),
            other => eprintln!("unknown dataset '{other}', skipping (expected cora|citeseer|pubmed)"),
        }
    }
}
