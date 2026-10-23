//! Generalized, environment-variable-driven benchmark harness for the
//! thesis's headline NBSC vs. GCN vs. GAT vs. GraphSAGE comparison.
//!
//! This supersedes `benchmark_cora.rs` / `benchmark_citeseer.rs`'s
//! hardcoded-constants pattern with runtime configuration, so the same
//! binary can run:
//!   - the canonical (bit-identical Planetoid) split main table,
//!   - a weight-decay sweep,
//!   - a multi-random-split robustness check (Shchur et al. 2018 style),
//! for either Cora or Citeseer, without editing and recompiling per
//! experiment. `benchmark_cora.rs`/`benchmark_citeseer.rs` are left in
//! place, unmodified, so the original locked baseline numbers in
//! `docs/results_cora_draft.md` remain independently reproducible from
//! the exact file that produced them.
//!
//! ## Environment variables (all optional, with defaults noted)
//! - `NBSC_DATASET`      = `cora` (default) | `citeseer`
//! - `NBSC_SPLIT`        = `canonical` (default) | `random`
//! - `NBSC_SPLIT_SEEDS`  = comma-separated split seeds, only used when
//!                         `NBSC_SPLIT=random` (default `"0"`). One full
//!                         (depths x seeds x architectures) sweep runs
//!                         per split seed, each reported as its own block
//!                         plus an across-split summary line per
//!                         (architecture, depth).
//! - `NBSC_DEPTHS`       = comma-separated depths (default `"2"`)
//! - `NBSC_N_SEEDS`      = number of training/init seeds per config
//!                         (default `3`)
//! - `NBSC_WEIGHT_DECAY` = f64 (default `0.0`)
//! - `NBSC_NORMALIZE`    = `true`|`false` (default `false`) -- LayerNorm
//!                         ablation switch for the NBSC layer only.
//! - `NBSC_ENERGY_REG_LAMBDA` = f64 (default `0.0`) -- weight on an
//!                         *explicit* Dirichlet-energy regularization term
//!                         added to the training loss at every layer's
//!                         activation (`dirichlet_energy_differentiable`
//!                         in `burn_layer.rs`), summed across layers and
//!                         averaged by depth so the effective per-layer
//!                         weight doesn't scale with `NBSC_DEPTHS`. This is
//!                         a second, independent energy-stabilization
//!                         mechanism from `NBSC_NORMALIZE`'s LayerNorm: the
//!                         two can be combined (`normalize=true` changes
//!                         what the activations *are*; this changes what
//!                         the loss *rewards*), and each can be ablated
//!                         alone. Default `0.0` leaves every existing
//!                         number in `docs/results_thesis.md` reproducible
//!                         unchanged. Applies uniformly across whichever
//!                         `NBSC_ARCHS` are run (not NBSC-only) so the
//!                         same loss-side comparison is available for GCN/
//!                         GAT/GraphSAGE too, though the motivating
//!                         hypothesis (§5 of `results_thesis.md`, the
//!                         expansive `A / rho_B` tap) is NBSC-specific.
//! - `NBSC_ARCHS`        = comma-separated subset of
//!                         `nbsc,gcn,gat,sage` (default: all four)
//! - `NBSC_EPOCHS`       = usize (default `150`, matching the original
//!                         Cora/Citeseer baselines for comparability)
//! - `NBSC_CSV_OUT`      = optional path; if set, appends one CSV row per
//!                         (split_seed, depth, seed, architecture) result
//!                         for later aggregation into thesis tables.
//!
//! Run with: `cargo run --release --example thesis_bench --features burn`

use burn::backend::{Autodiff, NdArray};
use burn::module::Module;
use burn::nn::loss::CrossEntropyLossConfig;
use burn::nn::{Linear, LinearConfig};
use burn::optim::{decay::WeightDecayConfig, AdamConfig, GradientsParams, Optimizer};
use burn::tensor::backend::Backend;
use burn::tensor::{Int, Tensor};

use nbsc::burn_layer::{
    degrees_tensor, dense_adjacency_tensor, dirichlet_energy, dirichlet_energy_differentiable,
    GcnLayer, GcnLayerConfig, NbscLayer, NbscLayerConfig,
};
use nbsc::dataset::{stratified_split, Dataset};
use nbsc::gat_layer::{GatLayer, GatLayerConfig};
use nbsc::graph::Graph;
use nbsc::sage_layer::{SageLayer, SageLayerConfig};

use std::fs::OpenOptions;
use std::io::Write;

type Be = Autodiff<NdArray<f32>>;

const HIDDEN: usize = 16;
const K_TAPS: usize = 2;
const GAT_HEADS: usize = 4;
const LR: f64 = 0.01;

fn env_str(name: &str, default: &str) -> String {
    std::env::var(name).unwrap_or_else(|_| default.to_string())
}
fn env_usize(name: &str, default: usize) -> usize {
    env_str(name, &default.to_string()).parse().unwrap_or(default)
}
fn env_f64(name: &str, default: f64) -> f64 {
    env_str(name, &default.to_string()).parse().unwrap_or(default)
}
fn env_bool(name: &str, default: bool) -> bool {
    env_str(name, &default.to_string()).parse().unwrap_or(default)
}
fn env_usize_list(name: &str, default: &[usize]) -> Vec<usize> {
    match std::env::var(name) {
        Ok(v) => v.split(',').filter_map(|s| s.trim().parse().ok()).collect(),
        Err(_) => default.to_vec(),
    }
}
fn env_u64_list(name: &str, default: &[u64]) -> Vec<u64> {
    match std::env::var(name) {
        Ok(v) => v.split(',').filter_map(|s| s.trim().parse().ok()).collect(),
        Err(_) => default.to_vec(),
    }
}
fn env_str_list(name: &str, default: &[&str]) -> Vec<String> {
    match std::env::var(name) {
        Ok(v) => v.split(',').map(|s| s.trim().to_string()).collect(),
        Err(_) => default.iter().map(|s| s.to_string()).collect(),
    }
}

fn int_tensor<B: Backend>(values: &[i64], device: &B::Device) -> Tensor<B, 1, Int> {
    let values_i32: Vec<i32> = values.iter().map(|&v| v as i32).collect();
    Tensor::<B, 1, Int>::from_ints(values_i32.as_slice(), device)
}
fn index_select_rows<B: Backend>(x: &Tensor<B, 2>, idx: &[usize], device: &B::Device) -> Tensor<B, 2> {
    let idx_i64: Vec<i64> = idx.iter().map(|&i| i as i64).collect();
    let idx_t = int_tensor::<B>(&idx_i64, device);
    x.clone().select(0, idx_t)
}
fn accuracy<B: Backend>(logits: &Tensor<B, 2>, labels: &[usize], idx: &[usize]) -> f32 {
    let preds = logits.clone().argmax(1).squeeze::<1>(1);
    let preds_data = preds.into_data().convert::<i64>().value;
    let correct = idx.iter().filter(|&&i| preds_data[i] as usize == labels[i]).count();
    correct as f32 / idx.len().max(1) as f32
}
fn mean_std(values: &[f32]) -> (f32, f32) {
    let n = values.len().max(1) as f32;
    let mean = values.iter().sum::<f32>() / n;
    let var = values.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / n;
    (mean, var.sqrt())
}
fn features_tensor<B: Backend>(ds: &Dataset, device: &B::Device) -> Tensor<B, 2> {
    let data: Vec<f32> = ds.features.data.iter().map(|&v| v as f32).collect();
    Tensor::<B, 1>::from_floats(data.as_slice(), device).reshape([ds.graph.n, ds.features.f])
}

struct EvalResult {
    val_acc: f32,
    test_acc: f32,
    final_energy: f32,
}

#[derive(Module, Debug)]
struct NbscNet<B: Backend> {
    layers: Vec<NbscLayer<B>>,
    head: Linear<B>,
}
impl<B: Backend> NbscNet<B> {
    fn forward_all(&self, x: Tensor<B, 2>) -> (Vec<Tensor<B, 2>>, Tensor<B, 2>) {
        let mut h = x;
        let mut activations = Vec::with_capacity(self.layers.len());
        for layer in &self.layers {
            h = layer.forward(h);
            activations.push(h.clone());
        }
        let logits = self.head.forward(h);
        (activations, logits)
    }
}
#[derive(Module, Debug)]
struct GcnNet<B: Backend> {
    layers: Vec<GcnLayer<B>>,
    head: Linear<B>,
}
impl<B: Backend> GcnNet<B> {
    fn forward_all(&self, x: Tensor<B, 2>) -> (Vec<Tensor<B, 2>>, Tensor<B, 2>) {
        let mut h = x;
        let mut activations = Vec::with_capacity(self.layers.len());
        for layer in &self.layers {
            h = layer.forward(h);
            activations.push(h.clone());
        }
        let logits = self.head.forward(h);
        (activations, logits)
    }
}
#[derive(Module, Debug)]
struct GatNet<B: Backend> {
    layers: Vec<GatLayer<B>>,
    head: Linear<B>,
}
impl<B: Backend> GatNet<B> {
    fn forward_all(&self, x: Tensor<B, 2>) -> (Vec<Tensor<B, 2>>, Tensor<B, 2>) {
        let mut h = x;
        let mut activations = Vec::with_capacity(self.layers.len());
        for layer in &self.layers {
            h = layer.forward(h);
            activations.push(h.clone());
        }
        let logits = self.head.forward(h);
        (activations, logits)
    }
}
#[derive(Module, Debug)]
struct SageNet<B: Backend> {
    layers: Vec<SageLayer<B>>,
    head: Linear<B>,
}
impl<B: Backend> SageNet<B> {
    fn forward_all(&self, x: Tensor<B, 2>) -> (Vec<Tensor<B, 2>>, Tensor<B, 2>) {
        let mut h = x;
        let mut activations = Vec::with_capacity(self.layers.len());
        for layer in &self.layers {
            h = layer.forward(h);
            activations.push(h.clone());
        }
        let logits = self.head.forward(h);
        (activations, logits)
    }
}

fn build_nbsc_net(graph: &Graph, depth: usize, d_in: usize, n_classes: usize, normalize: bool, device: &<Be as Backend>::Device) -> NbscNet<Be> {
    let mut layers = Vec::with_capacity(depth);
    for i in 0..depth {
        let d_i = if i == 0 { d_in } else { HIDDEN };
        layers.push(NbscLayerConfig::new(d_i, HIDDEN, K_TAPS).with_normalize(normalize).init(graph, device));
    }
    let head = LinearConfig::new(HIDDEN, n_classes).init(device);
    NbscNet { layers, head }
}
fn build_gcn_net(graph: &Graph, depth: usize, d_in: usize, n_classes: usize, device: &<Be as Backend>::Device) -> GcnNet<Be> {
    let mut layers = Vec::with_capacity(depth);
    for i in 0..depth {
        let d_i = if i == 0 { d_in } else { HIDDEN };
        layers.push(GcnLayerConfig::new(d_i, HIDDEN).init(graph, device));
    }
    let head = LinearConfig::new(HIDDEN, n_classes).init(device);
    GcnNet { layers, head }
}
fn build_gat_net(graph: &Graph, depth: usize, d_in: usize, n_classes: usize, device: &<Be as Backend>::Device) -> GatNet<Be> {
    let per_head = (HIDDEN / GAT_HEADS).max(1);
    let mut layers = Vec::with_capacity(depth);
    for i in 0..depth {
        let d_i = if i == 0 { d_in } else { per_head * GAT_HEADS };
        layers.push(GatLayerConfig::new(d_i, per_head, GAT_HEADS).init(graph, device));
    }
    let head = LinearConfig::new(per_head * GAT_HEADS, n_classes).init(device);
    GatNet { layers, head }
}
fn build_sage_net(graph: &Graph, depth: usize, d_in: usize, n_classes: usize, device: &<Be as Backend>::Device) -> SageNet<Be> {
    let mut layers = Vec::with_capacity(depth);
    for i in 0..depth {
        let d_i = if i == 0 { d_in } else { HIDDEN };
        layers.push(SageLayerConfig::new(d_i, HIDDEN).init(graph, device));
    }
    let head = LinearConfig::new(HIDDEN, n_classes).init(device);
    SageNet { layers, head }
}

fn weight_decay_config(wd: f64) -> Option<WeightDecayConfig> {
    if wd > 0.0 {
        Some(WeightDecayConfig::new(wd))
    } else {
        None
    }
}

#[allow(clippy::too_many_arguments)]
fn train_generic<F, N>(
    build: F, graph: &Graph, x: Tensor<Be, 2>, labels: &[usize],
    train_idx: &[usize], val_idx: &[usize], test_idx: &[usize],
    device: &<Be as Backend>::Device, seed: u64, epochs: usize, wd: f64, lambda_energy: f64,
) -> EvalResult
where
    F: FnOnce() -> N,
    N: burn::module::AutodiffModule<Be>,
    N: Forward<Be>,
{
    Be::seed(seed);
    let mut net = build();
    let mut optim = AdamConfig::new().with_weight_decay(weight_decay_config(wd)).init();
    let ce = CrossEntropyLossConfig::new().init::<Be>(device);
    let train_targets = int_tensor::<Be>(&train_idx.iter().map(|&i| labels[i] as i64).collect::<Vec<_>>(), device);

    // Built once per run (not per epoch/layer): the "Energy-Stabilized Deep
    // Stacks" regularizer (`NBSC_ENERGY_REG_LAMBDA`) reuses the same dense
    // adjacency/degrees tensors `NbscLayer`/`GcnLayer` build internally.
    let adjacency = dense_adjacency_tensor::<Be>(graph, device);
    let degrees = degrees_tensor::<Be>(graph, device);
    let m = graph.m();

    let mut last_energy = 0.0f32;
    for _ in 0..epochs {
        let (activations, logits) = net.forward_all(x.clone());
        let train_logits = index_select_rows(&logits, train_idx, device);
        let mut loss = ce.forward(train_logits, train_targets.clone());
        if lambda_energy > 0.0 && !activations.is_empty() {
            // Sum the differentiable Dirichlet energy at every layer's
            // activation ("between deep Chebyshev filter taps"), averaged
            // over depth so a lambda tuned at one depth transfers to
            // others without re-tuning for stack length.
            let mut energy_sum = dirichlet_energy_differentiable(&adjacency, &degrees, m, &activations[0]);
            for a in &activations[1..] {
                energy_sum = energy_sum + dirichlet_energy_differentiable(&adjacency, &degrees, m, a);
            }
            let energy_term = energy_sum.mul_scalar(lambda_energy / activations.len() as f64);
            loss = loss + energy_term;
        }
        let grads = loss.backward();
        let grads = GradientsParams::from_grads(grads, &net);
        net = optim.step(LR, net, grads);
        last_energy = activations.last().map(|a| dirichlet_energy(graph, a)).unwrap_or(0.0);
    }
    let (_a, logits) = net.forward_all(x);
    EvalResult {
        val_acc: accuracy(&logits, labels, val_idx),
        test_acc: accuracy(&logits, labels, test_idx),
        final_energy: last_energy,
    }
}

trait Forward<B: Backend> {
    fn forward_all(&self, x: Tensor<B, 2>) -> (Vec<Tensor<B, 2>>, Tensor<B, 2>);
}
impl<B: Backend> Forward<B> for NbscNet<B> {
    fn forward_all(&self, x: Tensor<B, 2>) -> (Vec<Tensor<B, 2>>, Tensor<B, 2>) {
        NbscNet::forward_all(self, x)
    }
}
impl<B: Backend> Forward<B> for GcnNet<B> {
    fn forward_all(&self, x: Tensor<B, 2>) -> (Vec<Tensor<B, 2>>, Tensor<B, 2>) {
        GcnNet::forward_all(self, x)
    }
}
impl<B: Backend> Forward<B> for GatNet<B> {
    fn forward_all(&self, x: Tensor<B, 2>) -> (Vec<Tensor<B, 2>>, Tensor<B, 2>) {
        GatNet::forward_all(self, x)
    }
}
impl<B: Backend> Forward<B> for SageNet<B> {
    fn forward_all(&self, x: Tensor<B, 2>) -> (Vec<Tensor<B, 2>>, Tensor<B, 2>) {
        SageNet::forward_all(self, x)
    }
}

fn report(name: &str, split_seed_label: &str, depth: usize, wd: f64, results: &[EvalResult]) -> (f32, f32, f32, f32) {
    let test_accs: Vec<f32> = results.iter().map(|r| r.test_acc).collect();
    let val_accs: Vec<f32> = results.iter().map(|r| r.val_acc).collect();
    let energies: Vec<f32> = results.iter().map(|r| r.final_energy).collect();
    let (test_mean, test_std) = mean_std(&test_accs);
    let (val_mean, val_std) = mean_std(&val_accs);
    let (energy_mean, energy_std) = mean_std(&energies);
    println!(
        "{name:<10} | split {split_seed_label:<10} | depth {depth:<2} | wd {wd:<8.1e} | val {val_mean:.3}+-{val_std:.3} | test {test_mean:.3}+-{test_std:.3} | energy {energy_mean:.4}+-{energy_std:.4} | n={}",
        results.len()
    );
    (val_mean, val_std, test_mean, test_std)
}

fn csv_append(path: &str, row: &str) {
    let is_new = !std::path::Path::new(path).exists();
    if let Ok(mut f) = OpenOptions::new().create(true).append(true).open(path) {
        if is_new {
            let _ = writeln!(f, "dataset,split_mode,split_seed,depth,weight_decay,normalize,energy_reg_lambda,architecture,seed,val_acc,test_acc,final_energy");
        }
        let _ = writeln!(f, "{row}");
    }
}

fn load_dataset(name: &str, split_mode: &str, split_seed: u64) -> Dataset {
    match (name, split_mode) {
        ("cora", "canonical") => Dataset::load_cora_planetoid().expect("cora_planetoid should load"),
        ("citeseer", "canonical") => Dataset::load_citeseer_planetoid().expect("citeseer_planetoid should load"),
        ("cora", "random") => {
            let mut ds = Dataset::load_planetoid_style(
                &std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("data/cora/cora.content"),
                &std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("data/cora/cora.cites"),
            )
            .expect("cora content/cites should parse");
            let (train, val, test) = stratified_split(&ds.labels, ds.num_classes(), 20, 500, 1000, split_seed);
            ds.train_mask = train;
            ds.val_mask = val;
            ds.test_mask = test;
            ds
        }
        ("citeseer", "random") => {
            let mut ds = Dataset::load_planetoid_style(
                &std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("data/citeseer/citeseer.content"),
                &std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("data/citeseer/citeseer.cites"),
            )
            .expect("citeseer content/cites should parse");
            let (train, val, test) = stratified_split(&ds.labels, ds.num_classes(), 20, 500, 1000, split_seed);
            ds.train_mask = train;
            ds.val_mask = val;
            ds.test_mask = test;
            ds
        }
        _ => panic!("unknown (dataset, split_mode) combination: ({name}, {split_mode})"),
    }
}

fn main() {
    let dataset_name = env_str("NBSC_DATASET", "cora");
    let split_mode = env_str("NBSC_SPLIT", "canonical");
    let split_seeds: Vec<u64> = if split_mode == "canonical" {
        vec![0] // canonical split is fixed; "seed" label is nominal
    } else {
        env_u64_list("NBSC_SPLIT_SEEDS", &[0])
    };
    let depths = env_usize_list("NBSC_DEPTHS", &[2]);
    let n_seeds = env_usize("NBSC_N_SEEDS", 3) as u64;
    let wd = env_f64("NBSC_WEIGHT_DECAY", 0.0);
    let normalize = env_bool("NBSC_NORMALIZE", false);
    let lambda_energy = env_f64("NBSC_ENERGY_REG_LAMBDA", 0.0);
    let archs = env_str_list("NBSC_ARCHS", &["nbsc", "gcn", "gat", "sage"]);
    let epochs = env_usize("NBSC_EPOCHS", 150);
    let csv_out = std::env::var("NBSC_CSV_OUT").ok();

    println!(
        "=== thesis_bench: dataset={dataset_name} split={split_mode} split_seeds={split_seeds:?} depths={depths:?} \
         n_seeds={n_seeds} weight_decay={wd} normalize={normalize} energy_reg_lambda={lambda_energy} archs={archs:?} epochs={epochs} ==="
    );

    let device: <Be as Backend>::Device = Default::default();

    // Accumulate per-(arch,depth) results across split seeds for a
    // cross-split summary line (the Shchur-style robustness check when
    // NBSC_SPLIT=random with multiple NBSC_SPLIT_SEEDS).
    use std::collections::HashMap;
    let mut cross_split_test: HashMap<(String, usize), Vec<f32>> = HashMap::new();

    for &split_seed in &split_seeds {
        let ds = load_dataset(&dataset_name, &split_mode, split_seed);
        let split_label = if split_mode == "canonical" { "canonical".to_string() } else { format!("rand{split_seed}") };
        println!(
            "\n--- {dataset_name} [{split_label}]: n={}, m={}, classes={}, train={}, val={}, test={} ---",
            ds.graph.n, ds.graph.m(), ds.num_classes(), ds.train_indices().len(), ds.val_indices().len(), ds.test_indices().len()
        );

        let x = features_tensor::<Be>(&ds, &device);
        let train_idx = ds.train_indices();
        let val_idx = ds.val_indices();
        let test_idx = ds.test_indices();
        let n_classes = ds.num_classes();
        let d_in = ds.features.f;

        for &depth in &depths {
            let mut results_by_arch: HashMap<String, Vec<EvalResult>> = HashMap::new();
            for arch in &archs {
                results_by_arch.insert(arch.clone(), Vec::with_capacity(n_seeds as usize));
            }

            for seed in 0..n_seeds {
                for arch in &archs {
                    let t = std::time::Instant::now();
                    eprint!("  split={split_label} depth={depth} seed={seed}/{n_seeds} {arch}...  ");
                    let result = match arch.as_str() {
                        "nbsc" => train_generic(
                            || build_nbsc_net(&ds.graph, depth, d_in, n_classes, normalize, &device),
                            &ds.graph, x.clone(), &ds.labels, &train_idx, &val_idx, &test_idx, &device, seed, epochs, wd, lambda_energy,
                        ),
                        "gcn" => train_generic(
                            || build_gcn_net(&ds.graph, depth, d_in, n_classes, &device),
                            &ds.graph, x.clone(), &ds.labels, &train_idx, &val_idx, &test_idx, &device, seed, epochs, wd, lambda_energy,
                        ),
                        "gat" => train_generic(
                            || build_gat_net(&ds.graph, depth, d_in, n_classes, &device),
                            &ds.graph, x.clone(), &ds.labels, &train_idx, &val_idx, &test_idx, &device, seed, epochs, wd, lambda_energy,
                        ),
                        "sage" => train_generic(
                            || build_sage_net(&ds.graph, depth, d_in, n_classes, &device),
                            &ds.graph, x.clone(), &ds.labels, &train_idx, &val_idx, &test_idx, &device, seed, epochs, wd, lambda_energy,
                        ),
                        other => panic!("unknown architecture: {other}"),
                    };
                    eprintln!("done ({:.1}s) val={:.3} test={:.3}", t.elapsed().as_secs_f32(), result.val_acc, result.test_acc);
                    if let Some(path) = &csv_out {
                        csv_append(
                            path,
                            &format!(
                                "{dataset_name},{split_mode},{split_label},{depth},{wd},{normalize},{lambda_energy},{arch},{seed},{:.4},{:.4},{:.6}",
                                result.val_acc, result.test_acc, result.final_energy
                            ),
                        );
                    }
                    results_by_arch.get_mut(arch).unwrap().push(result);
                }
            }

            println!("\n=== {dataset_name} [{split_label}] depth {depth} (wd={wd}) ===");
            for arch in &archs {
                let (_vm, _vs, test_mean, _ts) = report(arch, &split_label, depth, wd, &results_by_arch[arch]);
                cross_split_test.entry((arch.clone(), depth)).or_default().push(test_mean);
            }
        }
    }

    if split_seeds.len() > 1 {
        println!("\n=== cross-split summary (mean of per-split test-accuracy means; robustness check across {} splits) ===", split_seeds.len());
        for &depth in &depths {
            for arch in &archs {
                if let Some(v) = cross_split_test.get(&(arch.clone(), depth)) {
                    let (m, s) = mean_std(v);
                    println!("{arch:<10} | depth {depth:<2} | cross-split test acc {m:.3}+-{s:.3} (n_splits={})", v.len());
                }
            }
        }
    }
}
