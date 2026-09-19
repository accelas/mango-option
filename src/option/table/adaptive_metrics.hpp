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

/// Round-trip score for one validation point (spec D3).
///
/// Runs the product inversion (`invert_price_on_surface`) on the candidate
/// surface at the three stencil targets -- `refs.ref_price` and
/// `refs.ref_price +- refs.delta` -- over the bracket
/// `effective_sigma_bracket` derives from `ctx.sample_bounds`, and reports
/// the largest distance between a recovered volatility and `sigma`.  Both
/// domains of `ctx` are copied into the returned callable.
///
/// Outcomes, not verdicts about the surface as a whole:
///  - every target inverted => `PointStatus::Measured` and `iv_error` set;
///  - `refs.resolved == false` => `ReferenceUnresolved`, nothing attempted;
///  - otherwise the most severe inversion failure across the three targets,
///    mapped to the matching `Surface*` status.
///
/// `price_residual` (|surface price - reference price| / strike) is recorded
/// whenever both prices are finite, including on the unresolved path.  When
/// the outcome is `SurfaceNoRoot`, the three targets are re-inverted over the
/// bracket widened by `params.target_iv_error` per side (clipped to
/// `ctx.bounds`); success there only sets `edge_band_rescue`, a diagnostic
/// that changes neither `status` nor `iv_error`.
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
inline constexpr double kRichardsonSafetyFactor = 3.0;

/// Assumed observed order of convergence for the reference grid family's
/// two-grid error estimate. Task 11 calibrates the measured minimum order
/// from real surfaces; this constant must stay <= that measured minimum, so
/// the estimate below never overstates the family's actual convergence rate.
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
    bool rounded_down = false;
};

/// Build a nested reference grid family for `params` at `accuracy`, with
/// `levels + 1` entries (the fine grid plus `levels` successive halvings).
std::expected<ReferenceGridFamily, ValidationError> make_reference_grid_family(
    const PricingParams& params, const GridAccuracyParams& accuracy, size_t levels);

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
