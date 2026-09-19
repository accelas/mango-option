// SPDX-License-Identifier: MIT
#pragma once

/// @file adaptive_metrics.hpp
/// @brief Option-domain adapters for the generic adaptive refinement loop.
///
/// `run_refinement` (adaptive_refinement.hpp) is deliberately ignorant of
/// American options: every domain specific enters through its callbacks.
/// This header owns the mango-side implementations of those callbacks --
/// the FD reference solver, the FD-vega reference generator, and the
/// IV-error scoring metric -- so the loop itself never links the PDE
/// solver.

#include "mango/option/table/adaptive_grid_types.hpp"
#include "mango/option/table/adaptive_refinement.hpp"
#include "mango/option/option_spec.hpp"
#include "mango/option/grid_spec_types.hpp"
#include <atomic>
#include <memory>
#include <vector>
#include <optional>

namespace mango {

/// Compute IV error from price error and vega, with floor and cap.
double compute_iv_error(double price_error, double vega,
                        double vega_floor, double target_iv_error);

/// Produce ErrorRefs (FD American price + FD central-difference vega) for
/// one point: base solve + two sigma-bump solves.
/// 2 extra PDE solves per point — acceptable at build time.
/// Any failed or non-finite solve => unexpected.
PrepareRefsFn make_fd_vega_refs_fn(const AdaptiveGridParams& params,
                                    const ValidateFn& validate_fn);

/// Score an interpolated price against cached ErrorRefs using the TV/K
/// filter (skips points where TV/K < 1e-4; IV undefined there) and
/// `compute_iv_error` arithmetic (vega floor + target-level noise clamp).
/// Filtered points return `std::nullopt`, never 0.0: a skip is the absence of
/// a measurement, not a perfect one.
ScoreErrorFn make_iv_score_fn(const AdaptiveGridParams& params,
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
    /// count -- an estimate, not a certified bound on the resulting error.
    bool rounded_down = false;
};

/// Build a nested reference grid family for `params` at `accuracy`, with
/// `levels + 1` entries (the fine grid plus `levels` successive halvings).
std::expected<ReferenceGridFamily, ValidationError> make_reference_grid_family(
    const PricingParams& params, const GridAccuracyParams& accuracy, size_t levels);

/// Solve-attempt/failure counters for the reference oracle, for callers that
/// want visibility into how much PDE work the oracle performs. Atomic so a
/// shared counter can be read from multiple threads while solves run
/// elsewhere; the oracle itself does not update this struct (see
/// `ReferenceOracle::solve`) -- it is a bookkeeping surface for future
/// callers (Task 11+) that track fine vs. coarse solve outcomes.
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
};

}  // namespace mango
