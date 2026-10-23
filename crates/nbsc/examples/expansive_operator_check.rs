//! Extends the Cora-only "is A/rho_B expansive?" diagnostic
//! (`docs/results_cora_draft.md`) to all three real datasets, using the
//! same sparse, matrix-free estimators (`estimate_spectral_radius`,
//! `adjacency_operator_norm`) -- no new statistical machinery, just wider
//! coverage of an existing check.

use nbsc::dataset::Dataset;
use nbsc::spectral::{adjacency_operator_norm, estimate_spectral_radius};

fn report(name: &str, ds: &Dataset) {
    let t = std::time::Instant::now();
    let rho_b = estimate_spectral_radius(&ds.graph, 80, 0);
    let a_norm = adjacency_operator_norm(&ds.graph, 80, 0);
    let ratio = a_norm / rho_b;
    println!(
        "{name:<10} n={:<7} m={:<7} rho_B={rho_b:<10.4} ||A||_2={a_norm:<10.4} ||A||_2/rho_B={ratio:<8.4} ({}{:.1}s)",
        ds.graph.n,
        ds.graph.m(),
        if ratio > 1.0 { "EXPANSIVE " } else { "" },
        t.elapsed().as_secs_f32()
    );
}

fn main() {
    report("Cora", &Dataset::load_cora_planetoid().expect("cora"));
    report("Citeseer", &Dataset::load_citeseer_planetoid().expect("citeseer"));
    report("PubMed", &Dataset::load_pubmed_planetoid().expect("pubmed"));
}
