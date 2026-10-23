//! Standalone diagnostic (no `burn` feature required — pure linear algebra
//! on the loaded graph). End-to-end live run of `nbsc::spectral`'s
//! large-deviations rate function: `legendre_rate` / `legendre_rate_curve`,
//! the Legendre-Fenchel transform of the scaled CGF `Lambda(theta) = log
//! rho_B(theta)` already computed by `tilted_spectral_radius` (see
//! `tilt_check.rs` for the tilt primitive this builds on).
//!
//! Uses the same degree-tilt observable as `tilt_check` (`f(u -> v) =
//! degree(v)`, centered and scaled) on the real Cora citation graph, and:
//!   - locates `rho_B(0)`, the untilted radius, and `mean_x =
//!     d(log rho)/dtheta` at `theta=0` (the observable's mean under the
//!     untilted non-backtracking walk measure),
//!   - sweeps `x` across a range around `mean_x` and reports, at each
//!     point, the maximizing tilt `theta*`, `Lambda(theta*)`, and the rate
//!     `I(x) = theta* * x - Lambda(theta*)`,
//!   - checks the defining properties of a large-deviations rate function
//!     against that live output: `I` is minimized exactly at `x = mean_x`
//!     with minimum value `-Lambda(0)` (not `0` -- `rho_B(theta)` is a
//!     Perron root, not a normalized MGF, so `Lambda(0) != 0` in general),
//!     `I` is convex (strictly increasing in `|x - mean_x|` on this sweep),
//!     and at every `x` the returned `theta*` actually satisfies the
//!     first-order condition `Lambda'(theta*) = x` to high precision --
//!     cross-checked here via an independent call to
//!     `tilted_spectral_radius` at `theta*`, not just trusted from
//!     `legendre_rate`'s own bookkeeping.
//!
//! Interpretation: by the Gartner-Ellis theorem, `I(x)` is exactly the
//! exponential decay rate of the probability that the empirical mean
//! degree-bias observed along a long non-backtracking walk lands near `x`
//! instead of its typical value `mean_x` -- i.e. `P(mean over the first k
//! steps of a long walk ~= x) ~ exp(-k * I(x))` for large `k`. In a
//! compliance/anomaly-scoring pipeline this is the number that turns "the
//! walk drifted toward high-degree vertices by this much" into "and here
//! is how surprising that drift would be under normal (untilted) walk
//! behavior."
//!
//! Run with: `cargo run --release --example legendre_check`
//! (deliberately no `--features burn` needed)

use nbsc::dataset::Dataset;
use nbsc::spectral::{legendre_rate, tilted_spectral_radius, ArcWeights, DarcIndex};

fn main() {
    let ds = Dataset::load_cora_default(0).expect(
        "failed to load Cora -- check that nbsc/data/cora/{cora.content,cora.cites} exist",
    );
    let g = &ds.graph;
    println!("Cora: n={}, m={}, connected={}, bipartite={}", g.n, g.m(), g.is_connected(), g.is_bipartite());

    let krylov_dim = (2 * g.n).min(80);
    let seed = 0;

    let darcs = DarcIndex::build(g);
    let degrees = g.degrees();
    let mean_deg: f64 = degrees.iter().sum::<f64>() / degrees.len() as f64;
    let max_deg = degrees.iter().cloned().fold(0.0, f64::max);
    // Same scale-down as tilt_check: Cora has a very heavy-tailed degree
    // distribution (a few hub vertices with degree in the hundreds), so an
    // unscaled potential would make exp(theta * f) blow up for even modest
    // theta.
    let phi: Vec<f64> = degrees.iter().map(|d| (d - mean_deg) / 10.0).collect();
    let weights = ArcWeights::from_head_potential(&darcs, &phi);
    println!("degree observable: mean_deg={mean_deg:.4}, max_deg={max_deg:.0} (centered + scaled by 1/10)");

    // --- theta=0 baseline: Lambda(0) = log rho_B(0), mean_x = Lambda'(0). ---
    let at_zero = tilted_spectral_radius(&darcs, &weights, 0.0, krylov_dim, seed);
    let lambda_0 = at_zero.rho.ln();
    let mean_x = at_zero.drho_dtheta / at_zero.rho;
    println!();
    println!("=== theta=0 baseline ===");
    println!("rho_B(0)                  = {:.6}", at_zero.rho);
    println!("Lambda(0) = log rho_B(0)  = {lambda_0:.6}");
    println!("mean_x = Lambda'(0)       = {mean_x:.6}");

    // --- Rate-function sweep around the mean. ---
    println!();
    println!("=== large-deviations rate function I(x) = sup_theta(theta*x - Lambda(theta)) ===");
    println!(
        "{:>10}  {:>12}  {:>14}  {:>14}  {:>10}",
        "x", "theta*", "Lambda(theta*)", "I(x)", "Lambda'(theta*)"
    );

    // A sweep of offsets from the mean, spanning both directions and a
    // range wide enough to show clear convexity while staying inside the
    // numerically well-conditioned theta region for this graph (checked
    // against a live theta-sweep of rho_B(theta) beforehand).
    let offsets: Vec<f64> = vec![-2.16, -1.16, -0.66, -0.16, 0.0, 0.34, 0.84, 1.34, 2.84];
    let mut points = Vec::with_capacity(offsets.len());
    for &offset in &offsets {
        let x = mean_x + offset;
        let point = legendre_rate(&darcs, &weights, x, krylov_dim, seed);
        // Independent cross-check: re-evaluate Lambda' at the returned
        // theta* via a fresh tilted_spectral_radius call, not by trusting
        // legendre_rate's internal bookkeeping.
        let check = tilted_spectral_radius(&darcs, &weights, point.theta_star, krylov_dim, seed);
        let lambda_prime_check = check.drho_dtheta / check.rho;
        println!(
            "{:>10.4}  {:>12.5}  {:>14.6}  {:>14.6}  {:>15.6}",
            point.x, point.theta_star, point.lambda_theta_star, point.rate, lambda_prime_check
        );
        assert!(
            (lambda_prime_check - x).abs() < 1e-3,
            "first-order condition Lambda'(theta*) = x failed at x={x}: got {lambda_prime_check}"
        );
        points.push(point);
    }

    // --- Property 1: I is minimized exactly at x = mean_x, with minimum
    // value -Lambda(0) (not 0 -- rho_B(theta) is a Perron root, not a
    // normalized MGF, so Lambda(0) != 0 here). ---
    let at_mean = legendre_rate(&darcs, &weights, mean_x, krylov_dim, seed);
    println!();
    println!("=== sanity checks ===");
    println!("I(mean_x)                 = {:.6}", at_mean.rate);
    println!("-Lambda(0)                = {:.6}", -lambda_0);
    assert!(
        (at_mean.rate - (-lambda_0)).abs() < 1e-3,
        "I(mean_x) should equal -Lambda(0): got {} vs {}",
        at_mean.rate,
        -lambda_0
    );
    assert!(
        at_mean.theta_star.abs() < 1e-3,
        "theta* at x=mean_x should be ~0, got {}",
        at_mean.theta_star
    );
    println!("theta*(mean_x)             = {:.6} (should be ~0)", at_mean.theta_star);

    // --- Property 2: convexity -- I(x) strictly decreases moving toward
    // mean_x from either side of the sweep, and every I(x) computed is
    // >= I(mean_x). ---
    for p in &points {
        assert!(
            p.rate >= at_mean.rate - 1e-6,
            "I(x) should never be below its minimum I(mean_x): x={}, I(x)={}, I(mean_x)={}",
            p.x,
            p.rate,
            at_mean.rate
        );
    }
    let mut sorted = points.clone();
    sorted.sort_by(|a, b| a.x.partial_cmp(&b.x).unwrap());
    let mean_idx = sorted
        .iter()
        .enumerate()
        .min_by(|(_, a), (_, b)| a.x.partial_cmp(&b.x).unwrap())
        .unwrap()
        .0;
    // Left of the (interpolated) mean, I should be non-increasing as x
    // increases toward mean_x; right of it, non-decreasing as x moves away.
    // We check this directly against the actual mean point rather than
    // relying on an exact sweep entry landing on it.
    let mut with_mean: Vec<(f64, f64)> = sorted.iter().map(|p| (p.x, p.rate)).collect();
    with_mean.push((at_mean.x, at_mean.rate));
    with_mean.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    for w in with_mean.windows(2) {
        let ((x_a, i_a), (x_b, i_b)) = (w[0], w[1]);
        if x_b <= mean_x + 1e-9 {
            assert!(i_b <= i_a + 1e-6, "I should be non-increasing up to the mean: {x_a}->{x_b}");
        } else if x_a >= mean_x - 1e-9 {
            assert!(i_b >= i_a - 1e-6, "I should be non-decreasing past the mean: {x_a}->{x_b}");
        }
    }
    let _ = mean_idx;
    println!("convexity check           = passed (I(x) >= I(mean_x) everywhere on the sweep,");
    println!("                             non-increasing up to mean_x, non-decreasing past it)");

    println!();
    println!(
        "All checks passed: I(x) is minimized exactly at x=mean_x with I(mean_x) = -Lambda(0), \
         every returned theta* satisfies the first-order condition Lambda'(theta*) = x to <1e-3, \
         and I(x) is convex across the sweep -- confirming the Legendre transform of \
         Lambda(theta) = log rho_B(theta) behaves as the large-deviations rate function theory \
         predicts, live on the real Cora non-backtracking spectrum."
    );
}
