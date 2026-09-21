// SPDX-License-Identifier: MIT
#pragma once

/// @file adaptive_metrics.hpp
/// @brief Option-domain adapters for the generic adaptive refinement loop.
///
/// `run_refinement` (adaptive_refinement.hpp) is deliberately ignorant of
/// American options: every domain specific enters through its callbacks.
/// This header owns the mango-side implementations of those callbacks --
/// the FD reference solver, the six-solve reference stencil, and the
/// error-scoring metric -- so the loop itself never links the PDE solver.

#include "mango/option/table/adaptive_grid_types.hpp"
#include "mango/option/table/adaptive_refinement.hpp"
#include "mango/option/option_spec.hpp"
#include "mango/option/grid_spec_types.hpp"
#include <atomic>
#include <expected>
#include <functional>
#include <memory>
#include <optional>
#include <vector>

namespace mango {

/// Operational round trip (edge band tau): the score for one validation
/// point (spec D3, rev 5).
///
/// Runs the product inversion (`invert_price_on_surface`) on the candidate
/// surface at the three stencil targets -- `refs.ref_price` and
/// `refs.ref_price +- refs.delta` -- and reports the largest distance
/// between a recovered volatility and `sigma`.  Both domains of `ctx` are
/// copied into the returned callable.
///
/// The bracket is the product's, taken over the *acceptance band*: the
/// published sigma range of `ctx.sample_bounds` widened by
/// `params.target_iv_error` at each end.  Pre-check, screen, Brent,
/// post-check, the adaptive cap and the configured sigma limits are the
/// shipped ones, unchanged.  The metric is therefore *stronger than the
/// shipped solver at the edges by exactly `target_iv_error`*: a root the
/// solver would refuse today, lying within the user's own tolerance beyond a
/// published edge, is a measurement here.  Making the two coincide again is
/// the query-time follow-up.
///
/// Where the fit domain `ctx.bounds` has no sigma support over part of that
/// band -- the B-spline backends fit exactly the published range -- the
/// surface is extended from its nearest supported edge by first-order
/// extrapolation, `price(sigma) = S(sigma_e) + V(sigma_e) * (sigma -
/// sigma_e)` with the vega held at `V(sigma_e)`.  The extension is at most
/// `target_iv_error` long and the surface is C2, so the model error is
/// `O(vomma * target_iv_error^2)`, negligible at bps scale; the band
/// therefore behaves the same on every backend.
///
/// Outcomes, not verdicts about the surface as a whole:
///  - every target inverted => `PointStatus::Measured` and `iv_error` set;
///  - `refs.resolved == false` => `ReferenceUnresolved`, nothing attempted;
///  - otherwise the most severe inversion failure across the three targets,
///    mapped to the matching `Surface*` status.
///
/// `price_residual` (|surface price - reference price| / strike) is recorded
/// whenever both prices are finite, including on the unresolved path.
///
/// `edge_band_rescue` is the exact-bracket diagnostic: set on a `Measured`
/// point when any recovered volatility falls outside the un-widened product
/// bracket, i.e. when the shipped solver would have refused that query
/// today.  It changes neither `status` nor `iv_error` and gates nothing.
ScoreErrorFn make_round_trip_score_fn(const AdaptiveGridParams& params,
                                      const RefinementContext& ctx,
                                      OptionType option_type);

/// Create a direct FD reference. With reference_maturity, the dividends are
/// anchored to one fixed expiry and rolled to each query's remaining life.
ValidateFn make_validate_fn(double dividend_yield,
                            OptionType option_type,
                            const std::vector<Dividend>& discrete_dividends = {},
                            std::optional<double> reference_maturity = std::nullopt);

/// Accuracy profile the reference oracle solves at. High, not Ultra: the
/// round-trip metric needs a stable two-grid Richardson estimate, not the
/// most expensive grid available.
inline constexpr GridAccuracyProfile kReferenceAccuracy = GridAccuracyProfile::High;

/// Roache's recommended safety factor for a two-grid (uncalibrated-order)
/// Richardson error estimate.
///
/// The D8 calibration test gives it a measured job as well: where the true
/// local order is `p_min` rather than the assumed
/// `kReferenceConvergenceOrder`, the estimate understates by at most
/// `(2^p - 1) / (2^{p_min} - 1)`, and the test asserts this constant covers
/// that ratio over the calibration set (measured 1.69 at 30 days, against
/// 3).  That is what absorbs the order wander the free boundary produces at
/// short maturities.
inline constexpr double kRichardsonSafetyFactor = 3.0;

/// Floor on each stencil uncertainty estimate, relative to the strike of the
/// contract actually solved (spec D1, rev 5): calibrated constant, the
/// oracle's High-vs-Ultra discrepancy scale relative to strike.  Measured
/// 2026-09-19: |V_High - V_Ultra| = 4e-6 and 7e-6 dollars on K = 100, i.e.
/// 4e-8 and 7e-8 of strike; 1e-7 is the conservative rounding.  The D8
/// calibration test asserts the constant is at least
/// max |V_High - V_Ultra| / K.
///
/// Re-measured 2026-09-21 by `//tests:reference_oracle_calibration_test`
/// over D8's six-point set: max |V_High - V_Ultra| / K = 6.34e-8, on the
/// 30-day OTM put.  1e-7 covers it, so the value stands.
///
/// Without it, two discretizations that agree exactly -- both sitting on the
/// obstacle at a near-intrinsic point -- estimate zero uncertainty, and the
/// stencil admits a bracket separated by microdollars that the shipped
/// inversion cannot then invert.
inline constexpr double kReferenceUncertaintyFloor = 1e-7;

/// Assumed observed order of convergence for the reference grid family's
/// two-grid error estimate.  This constant must stay <= the measured minimum
/// usable order, so the estimate below never overstates the family's actual
/// convergence rate.
///
/// Calibrated at 1.0 by the D8 test
/// (`//tests:reference_oracle_calibration_test`, rev 6 run 2026-09-21 on
/// `G-half, G, 2G, 4G`).  It sits below every observed order of the
/// production pair (minimum p_A = 1.31722), so `δ̂` never understates
/// relative to the pair it is applied to.  It was not raised to 1.3, which
/// the triple-A minimum would allow, because the next finer pair's order at
/// 30 days wanders down to 0.68: a lower assumed `p` only enlarges `δ̂`, so
/// staying at 1.0 costs conservatism in the safe direction and buys margin
/// where the order is not settled.  The residual understatement risk is what
/// `kRichardsonSafetyFactor` absorbs -- at p_min = 0.670717 the needed
/// factor is (2^1 - 1)/(2^0.670717 - 1) = 1.69 against the shipped 3, which
/// the calibration test asserts.
///
/// Triple A is `(G-half, G, 2G)`, the order of the very pair this estimate
/// differences; triple B is `(G, 2G, 4G)`, one level finer.  Per point, the
/// range over both tau_iv and all three stencil sigma:
///
///   atm-1y-3div  p_A 1.317..1.341   p_B 1.647..1.660
///   500-trigger  p_A 2.096..2.369   p_B 1.833..2.444
///   otm-30d      p_A 1.559..1.687   p_B 0.671..1.401
///   deep-otm-7d  p_A 1.916..1.996   p_B 1.996..2.112
///   itm-2y       p_A 1.619..1.829   p_B 1.738..1.780
///   atm-6m-call  p_A 2.000          p_B 2.000
///
/// Minimum usable p_A = 1.31722 (atm-1y-3div, tau_iv = 5e-4, sigma-lo),
/// maximum 2.36881; minimum over both triples 0.670717 (otm-30d,
/// tau_iv = 1e-3); 72 usable triples, 0 oscillatory, 0 insufficient.
inline constexpr double kReferenceConvergenceOrder = 1.0;

/// A nested family of explicit PDE grid configs for Richardson-style error
/// estimation: `levels[0]` is the finest grid (from `estimate_pde_grid` at
/// the requested accuracy, rounded to `n = 1 (mod 16)` points so up to three
/// halvings stay odd), and `levels[k]` re-samples the same generator at
/// every 2^k-th node, so `levels[k]` is nested exactly inside `levels[0]`.
struct ReferenceGridFamily {
    std::vector<PDEGridConfig> levels;
    std::vector<size_t> point_counts;
    std::vector<size_t> time_steps;
    /// True when the fine level's point count had to be rounded down to fit
    /// `accuracy.max_spatial_points` instead of up to the next `1 (mod 16)`
    /// count -- the family's own accuracy is then an estimate, not a
    /// validated one.
    ///
    /// The shipped profiles do reach it: it happens whenever the estimate
    /// lands within 15 points of the profile cap, measured for the ITM 2y
    /// put at High, whose estimate of 3495 rounds down to 3489 under the
    /// 3500 cap.  The fine grid is then at most 15 points coarser than the
    /// estimate, which the cap already declared acceptable.
    bool rounded_down = false;
};

/// Build a nested reference grid family for `params` at `accuracy`, with
/// `levels + 1` entries (the fine grid plus `levels` successive halvings).
std::expected<ReferenceGridFamily, ValidationError> make_reference_grid_family(
    const PricingParams& params, const GridAccuracyParams& accuracy, size_t levels);

/// Refine one explicit grid config by an integer `factor`: the same
/// `GridSpec` generator family re-sampled at `factor * (n - 1) + 1` points,
/// with `n_time` multiplied by `factor` and `mandatory_times` carried over.
///
/// Every generator is a pure map of eta = i/(n-1) (grid.hpp `generate()`),
/// so `g` is exactly the every-`factor`-th-node subsequence of the result --
/// the same nesting argument `make_reference_grid_family` uses downward,
/// run upward instead.
///
/// The accuracy profile's `max_spatial_points` cap deliberately does NOT
/// apply: this is a calibration-only construction (spec D8 builds `2G` and
/// `4G` above production's fine grid to observe the order of the pair
/// production actually uses), and the solver accepts any explicit grid.
/// Nothing on the production path calls it.
std::expected<PDEGridConfig, ValidationError> refine_grid_config(
    const PDEGridConfig& g, size_t factor);

/// Solve-attempt/failure counters for the reference stencil, so a caller can
/// see how much PDE work its references cost. Atomic so one counter shared
/// by a whole build can be read while solves run elsewhere. `ReferenceOracle`
/// does not touch it; `make_stencil_refs_fn` records every attempt and every
/// failure on the counter it is handed, and the adaptive builders report the
/// totals as `BuildDiagnostics::reference_solves_fine/coarse` (spec D7).
struct ReferenceSolveCounter {
    std::atomic<size_t> fine_attempts{0};
    std::atomic<size_t> coarse_attempts{0};
    std::atomic<size_t> fine_failures{0};
    std::atomic<size_t> coarse_failures{0};
};

/// Reference oracle: rolls a fixed-expiry dividend schedule onto a query
/// contract exactly as `make_validate_fn` does, and solves it on a caller
/// supplied explicit grid at the oracle's accuracy profile.
struct ReferenceOracle {
    double dividend_yield;
    OptionType option_type;
    std::vector<Dividend> discrete_dividends;
    std::optional<double> reference_maturity;
    GridAccuracyParams accuracy;

    /// Build the PricingParams for one query point, rolling
    /// `discrete_dividends` onto `tau` when `reference_maturity` is set
    /// (segmented/fixed-expiry surfaces), or filtering to `tau` directly
    /// otherwise (ordinary contracts described from now).
    PricingParams contract(double spot, double strike, double tau,
                           double sigma, double rate) const;

    /// Solve `p` on the explicit `grid`.
    std::expected<double, SolverError> solve(const PricingParams& p,
                                             const PDEGridConfig& grid) const;

    /// Solve `p` at the oracle's accuracy profile without a fixed explicit
    /// grid (an auto-estimated grid per `accuracy`). For single-price
    /// callers, such as `make_validate_fn`, that have no reference grid
    /// family in hand.
    std::expected<double, SolverError> solve_estimated(const PricingParams& p) const;
};

/// Solves at one sigma on one grid; injectable for tests.
using StencilSolveFn = std::function<std::expected<double, SolverError>(
    const PricingParams&, const PDEGridConfig&)>;

/// Prepare the six-solve reference stencil for one point (spec D1).
///
/// One `ReferenceGridFamily` is built per preparation, at the widest stencil
/// member (sigma0 + target_iv_error), and all six solves run on that single
/// fine/coarse pair (L2). The order is `[y, y-half, lo, lo-half, hi,
/// hi-half]`; `counter` records fine and coarse attempts and failures.
///
/// The base fine solve failing yields `unexpected` -- the point is invalid.
/// Everything else (sigma0 - target_iv_error <= 0, a failed bracket or coarse
/// solve, a stencil that does not separate, a target the product's query
/// validation rejects) yields a successful `ErrorRefs` with the base price
/// present and `resolved = false`.
///
/// `solve` defaults to `oracle.solve`; tests inject a fake.
PrepareRefsFn make_stencil_refs_fn(const AdaptiveGridParams& params,
                                   ReferenceOracle oracle,
                                   std::shared_ptr<ReferenceSolveCounter> counter,
                                   StencilSolveFn solve = {});

/// Adapt a reference factory to a single-K_ref probe surface (spec D1/L6).
///
/// A sizing handle that prices a `K_ref` probe reaches a query `(S, K)` as
/// `a * probe(S/a, K_ref)` with `a = K/K_ref`, so it approximates
/// `V(S, K; a*D)`, not `V(S, K; D)`: absolute cash dividends do not scale
/// with `a`.  Measuring it against a reference on the user's contract would
/// charge the `(a - 1) * D * dP/dD` residual to the interpolation.
///
/// The returned factory therefore solves the probe's own contract at
/// `(spot/a, K_ref)` through `base` and scales every monetary field --
/// `ref_price`, both bracket prices, all three delta estimates -- by `a`.
/// The sigma coordinates, `resolved` and the step counts are unchanged:
/// every term of the D2 separation test scales alike, so resolution is
/// invariant under the scaling.  A non-positive strike leaves `a = 1`.
PrepareRefsFn make_probe_scaled_refs_fn(PrepareRefsFn base, double K_ref);

/// The D2 separation inequalities alone, on already-prepared refs: all three
/// prices and all three estimates finite, and the three estimated price
/// intervals separated in the expected order. Target validity
/// (`validate_iv_query`) is checked separately, at preparation, because it
/// needs the contract.
bool stencil_resolved(const ErrorRefs& r) noexcept;

}  // namespace mango
