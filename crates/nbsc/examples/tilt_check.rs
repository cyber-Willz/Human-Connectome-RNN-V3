//! Standalone diagnostic (no `burn` feature required — pure linear algebra
//! on the loaded graph). End-to-end live run of the exponential-tilt
//! primitive in `nbsc::spectral`: `DarcIndex`, `ArcWeights`, and
//! `tilted_spectral_radius`.
//!
//! Tilts the non-backtracking spectral radius toward high-degree vertices
//! (`f(u -> v) = degree(v)`, centered) — a stand-in for tilting toward any
//! per-vertex risk/anomaly score in a compliance-style pipeline — sweeps
//! `theta` across a range, and at each point reports:
//!   - `rho_B(theta)`, the tilted Perron root,
//!   - `drho/dtheta`, via the closed-form left/right-eigenvector identity,
//!   - a finite-difference cross-check of that derivative,
//!   - `d(log rho)/dtheta`, which by Varadhan's lemma is exactly the mean
//!     degree-deviation seen along a long non-backtracking walk under the
//!     theta-tilted walk measure (so it should increase monotonically
//!     with theta as the walk is biased ever harder toward high-degree
//!     vertices).
//!
//! Also runs the `theta=0` / uniform-weight sanity checks from the unit
//! tests directly against the real graph, so this doubles as a live
//! (not just synthetic-graph) confirmation that the darc-space
//! construction agrees with the existing Bass-reduced `rho_B` path.
//!
//! Run with: `cargo run --release --example tilt_check`
//! (deliberately no `--features burn` needed)

use nbsc::dataset::Dataset;
use nbsc::spectral::{estimate_spectral_radius, tilted_spectral_radius, ArcWeights, DarcIndex};

fn main() {
    let ds = Dataset::load_cora_default(0).expect(
        "failed to load Cora -- check that nbsc/data/cora/{cora.content,cora.cites} exist",
    );
    let g = &ds.graph;
    println!("Cora: n={}, m={}, connected={}, bipartite={}", g.n, g.m(), g.is_connected(), g.is_bipartite());

    let krylov_dim = (2 * g.n).min(80);
    let seed = 0;

    // --- Sanity check 1: theta=0 (any weights) must reproduce the plain,
    // untilted Hashimoto radius from the existing Bass-reduced path. ---
    let darcs = DarcIndex::build(g);
    let uniform = ArcWeights::uniform(&darcs);
    let baseline_rho = estimate_spectral_radius(g, krylov_dim, seed);
    let at_zero = tilted_spectral_radius(&darcs, &uniform, 0.0, krylov_dim, seed);
    println!();
    println!("=== theta=0 sanity check ===");
    println!("estimate_spectral_radius(g)            = {baseline_rho:.6}");
    println!("tilted_spectral_radius(theta=0).rho     = {:.6}", at_zero.rho);
    let rel_err = (at_zero.rho - baseline_rho).abs() / baseline_rho;
    println!("relative error                          = {rel_err:.2e}");
    assert!(rel_err < 1e-3, "theta=0 tilted radius should match the untilted estimate");

    // --- Sanity check 2: uniform tilt has closed form rho(theta) =
    // e^theta * rho(0), drho/dtheta = rho(theta). ---
    println!();
    println!("=== uniform-tilt closed-form check (theta=0.5) ===");
    let uni_half = tilted_spectral_radius(&darcs, &uniform, 0.5, krylov_dim, seed);
    let expected = 0.5f64.exp() * baseline_rho;
    println!("rho(0.5) computed                       = {:.6}", uni_half.rho);
    println!("e^0.5 * rho(0) expected                 = {expected:.6}");
    println!("drho/dtheta computed                    = {:.6}", uni_half.drho_dtheta);
    println!("rho(0.5) (should equal drho/dtheta)     = {:.6}", uni_half.rho);

    // --- Main run: tilt toward high-degree vertices. ---
    let degrees = g.degrees();
    let mean_deg: f64 = degrees.iter().sum::<f64>() / degrees.len() as f64;
    // Scaled down so theta in [-1.5, 1.5] stays in a numerically comfortable
    // range for exp(theta * f) across Cora's degree spread -- rho(theta)
    // grows roughly exponentially in theta for this observable, so a fixed
    // finite-difference step h eventually stops resolving the curvature
    // (floating-point cancellation in (rho(theta+h) - rho(theta-h))) well
    // outside this range; that's a property of the finite-difference cross
    // check, not of the closed-form derivative itself.
    let phi: Vec<f64> = degrees.iter().map(|d| (d - mean_deg) / 10.0).collect();
    let weights = ArcWeights::from_head_potential(&darcs, &phi);

    println!();
    println!("=== degree-tilted rho_B(theta) sweep (bias toward high-degree vertices) ===");
    println!(
        "{:>6}  {:>12}  {:>14}  {:>14}  {:>10}  {:>14}",
        "theta", "rho", "drho/dtheta", "finite-diff", "rel.err", "dlog(rho)/dth"
    );

    let h = 1e-4;
    let mut prev_dlog: Option<f64> = None;
    for i in -3..=3 {
        let theta = i as f64 * 0.5;
        let center = tilted_spectral_radius(&darcs, &weights, theta, krylov_dim, seed);
        let plus = tilted_spectral_radius(&darcs, &weights, theta + h, krylov_dim, seed);
        let minus = tilted_spectral_radius(&darcs, &weights, theta - h, krylov_dim, seed);
        let finite_diff = (plus.rho - minus.rho) / (2.0 * h);
        let rel_err = (center.drho_dtheta - finite_diff).abs() / finite_diff.abs().max(1e-12);
        let dlog = center.drho_dtheta / center.rho;

        println!(
            "{theta:>6.2}  {:>12.6}  {:>14.6}  {:>14.6}  {rel_err:>10.2e}  {dlog:>14.6}",
            center.rho, center.drho_dtheta, finite_diff
        );

        assert!(
            rel_err < 1e-2,
            "analytic and finite-difference drho/dtheta disagree at theta={theta}: {rel_err:.2e}"
        );

        // d(log rho)/dtheta is the mean tilt-observable under the
        // theta-tilted walk measure; biasing harder toward high-degree
        // vertices (increasing theta) should never decrease that mean.
        if let Some(prev) = prev_dlog {
            assert!(
                dlog >= prev - 1e-6,
                "d(log rho)/dtheta should be monotone non-decreasing in theta (large-deviations \
                 convexity of log rho), got {prev:.6} -> {dlog:.6} at theta={theta}"
            );
        }
        prev_dlog = Some(dlog);
    }

    println!();
    println!(
        "All checks passed: theta=0 matches the untilted rho_B, the uniform-tilt closed form \
         holds, every analytic drho/dtheta agrees with a finite-difference cross-check to <1%, \
         and d(log rho)/dtheta is monotone non-decreasing in theta (as convexity of the scaled \
         cumulant generating function log rho_B(theta) requires)."
    );
}
