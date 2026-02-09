// SPDX-License-Identifier: MIT
#pragma once
#include "mango/option/option_spec.hpp"  // Dividend
#include "mango/math/root_finding.hpp"
#include <cmath>
#include <limits>
#include <vector>

namespace mango::bench {

inline constexpr double kSpot = 100.0;
inline constexpr double kRate = 0.05;
inline constexpr double kDivYield = 0.02;

// Quarterly $0.50 dividends scaled to maturity
inline std::vector<Dividend> make_div_schedule(double maturity) {
    return {
        Dividend{.calendar_time = maturity * 0.25, .amount = 0.50},
        Dividend{.calendar_time = maturity * 0.50, .amount = 0.50},
        Dividend{.calendar_time = maturity * 0.75, .amount = 0.50},
    };
}

// Brent solver for IV recovery using the library's find_root.
// Returns vol on success, NaN on failure.
template <typename PriceFn>
double brent_solve_iv(PriceFn&& price_fn, double target_price,
                      double a = 0.01, double b = 3.0) {
    auto objective = [&](double vol) { return price_fn(vol) - target_price; };
    RootFindingConfig config{.max_iter = 100, .brent_tol_abs = 1e-6};
    auto result = find_root(objective, a, b, config);
    if (result.has_value()) {
        return result->root;
    }
    return std::numeric_limits<double>::quiet_NaN();
}

// ===========================================================================
// IV failure detection
// ===========================================================================
//
// Two distinct failure modes exist when solving for implied volatility:
//
// 1. IV INDETERMINABILITY (universal, scheme-independent)
//    When option vega ≈ 0, the price is insensitive to volatility. Any σ in
//    a wide range satisfies |price(σ) - target| < tol. Brent converges
//    successfully (small residual, few iterations) but returns an arbitrary σ
//    — typically the bracket lower bound σ_min.
//
//    This occurs for:
//    - Deep OTM options: price ≈ 0 regardless of σ
//    - Deep ITM short-dated options: price ≈ intrinsic regardless of σ
//    - Both cases share: time_value / K ≈ 0
//
//    Detection methods (tested on S=100, K=80..120, T=7d..2y, σ=15%/30%):
//
//    a) Bracket-boundary check (post-solve, zero cost):
//       If solved σ ≈ σ_min or σ ≈ σ_max within margin, IV is unreliable.
//       PERFECT detector: 100% precision, 100% recall on all tested cases.
//       The only false-positive-free method.
//
//    b) Time-value pre-filter (pre-solve, no Brent needed):
//       time_value = price - max(K - S, 0) for puts
//       If time_value / K < threshold, IV is likely indeterminate.
//       100% recall at TV/K < 1e-6 but some false positives on deep ITM
//       options where Brent still solves correctly.
//
//    c) Vega threshold (requires FDM vega, expensive):
//       If |vega| < 0.01, IV is indeterminate.
//       100% recall, slight false positives. Not practical for production
//       since computing vega costs ~2 PDE solves.
//
//    Recommendation for API: use bracket-boundary check after every Brent
//    solve. Return std::unexpected when at_boundary is true.
//
// 2. SURFACE APPROXIMATION ERROR (scheme-dependent)
//    Even when vega is adequate, the interpolated price surface may be
//    inaccurate in certain regions, causing Brent to converge to a wrong σ.
//    This is specific to the interpolation scheme (B-spline, Chebyshev, etc.)
//    and cannot be detected by any input-based filter.
//
//    Tested hypothesis: proximity to the early exercise boundary (|x - x*|)
//    was NOT correlated with interpolation errors. The piecewise boundary
//    detector returned delta=0 for all dividend segments, indicating no clear
//    exercise boundary was found in the V/K_ref surface (unlike the EEP
//    surface used in the vanilla case).
//
//    The observed interpolation failures at σ=15% (e.g. 60d/K=110 at 999 bps)
//    are actually near-intrinsic options with very low TV/K, where the
//    interpolation surface adds enough noise to shift Brent's solution.
//    At σ=30%, only 3 points fail (all with vega < 0.01), confirming that
//    higher vol eliminates most interpolation-induced failures.
//
//    Detection: scheme-specific. Each surface implementation should define
//    its own confidence region or domain bounds.
//
// Summary of detector performance (ground truth: error > 50 bps):
//
//   FDM failures (vega≈0):
//   | Detector              | σ=15% P/R    | σ=30% P/R    |
//   |-----------------------|--------------|--------------|
//   | Bracket boundary      | 100% / 100%  | 100% / 100%  |
//   | TV/K < 1e-6           |  92% / 100%  |  50% / 100%  |
//   | Vega < 0.01           |  92% / 100%  |  33% / 100%  |
//
//   Interpolation failures (surface quality):
//   | Detector              | σ=15% P/R    | σ=30% P/R    |
//   |-----------------------|--------------|--------------|
//   | Bracket boundary      |  64% /  41%  | 100% /  33%  |
//   | TV/K < 1e-4           |  63% /  71%  |  50% / 100%  |
//   | Vega < 0.01           |  67% /  47%  | 100% / 100%  |
//   | |x - x*| < any        | ~10% / ~20%  |   0% /   0%  |
//
// ===========================================================================

/// Full Brent result with convergence diagnostics for IV reliability detection.
///
/// The key field is `at_boundary`: when true, the solved σ landed near the
/// bracket bound, indicating the price is insensitive to vol at this point.
/// This is a perfect detector for IV indeterminability (zero false positives
/// and zero false negatives in all tested configurations).
struct BrentIVResult {
    double iv = std::numeric_limits<double>::quiet_NaN();
    bool converged = false;
    size_t iterations = 0;
    double residual = std::numeric_limits<double>::quiet_NaN();  // |price(σ) - target|
    /// True if solved σ is within `boundary_margin` of bracket bound [a, b].
    /// When true, the IV is unreliable — Brent converged (small residual)
    /// but the root is not meaningful because price(σ) ≈ target for all σ.
    bool at_boundary = false;
};

/// Solve for IV using Brent's method, returning full convergence diagnostics.
///
/// Unlike brent_solve_iv() which discards convergence info, this variant
/// exposes the bracket-boundary flag needed for reliability detection.
///
/// Usage for API-level rejection:
///   auto r = brent_solve_iv_full(price_fn, target);
///   if (!r.converged || r.at_boundary)
///       return std::unexpected(IVError::NotDeterminable);
///   return r.iv;
///
/// @param boundary_margin  Distance from bracket bound to flag as unreliable.
///                         Default 0.005 (50 bps in vol space). Points with
///                         σ <= a + margin or σ >= b - margin are flagged.
template <typename PriceFn>
BrentIVResult brent_solve_iv_full(PriceFn&& price_fn, double target_price,
                                   double a = 0.01, double b = 3.0,
                                   double boundary_margin = 0.005) {
    auto objective = [&](double vol) { return price_fn(vol) - target_price; };
    RootFindingConfig config{.max_iter = 100, .brent_tol_abs = 1e-6};
    auto result = find_root(objective, a, b, config);

    BrentIVResult r;
    if (result.has_value()) {
        r.iv = result->root;
        r.converged = true;
        r.iterations = result->iterations;
        r.residual = result->final_error;
        r.at_boundary = (result->root <= a + boundary_margin) ||
                        (result->root >= b - boundary_margin);
    } else {
        r.converged = false;
        r.iterations = result.error().iterations;
        r.residual = result.error().final_error;
        if (result.error().last_value)
            r.iv = *result.error().last_value;
    }
    return r;
}

}  // namespace mango::bench
