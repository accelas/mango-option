// SPDX-License-Identifier: MIT
/**
 * @file surface_inversion.cpp
 * @brief Bracket screen and price-to-volatility inversion on a price surface.
 */

#include "mango/option/surface_inversion.hpp"

#include "mango/math/root_finding.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>

namespace mango {

// =====================================================================
// Multiple-root bracket screen (spec D8.2)
// =====================================================================

detail::BracketScreen detail::screen_bracket(
    ObjectiveRef objective,
    double sigma_min, double sigma_max,
    double spot, double tolerance)
{
    BracketScreen out;
    out.lo = sigma_min;
    out.hi = sigma_max;

    constexpr size_t kScanPoints = 17;
    // Zero tolerance is a *price* tolerance in dollars, deliberately
    // distinct from `tolerance` (which the root finder uses for both its
    // interval and its objective convergence test).
    const double zero_tol = 1e-9 * spot;
    const double step = (sigma_max - sigma_min) / static_cast<double>(kScanPoints - 1);

    std::array<double, kScanPoints> scan_sigma{};
    std::array<double, kScanPoints> scan_f{};
    std::array<int, kScanPoints> scan_sign{};
    size_t zero_samples = 0;

    for (size_t i = 0; i < kScanPoints; ++i) {
        scan_sigma[i] = (i + 1 == kScanPoints)
            ? sigma_max
            : sigma_min + step * static_cast<double>(i);
        scan_f[i] = objective(scan_sigma[i]);
        if (!std::isfinite(scan_f[i])) {
            out.refusal = IVError{
                .code = IVErrorCode::NumericalInstability,
                .iterations = 0,
                .final_error = std::numeric_limits<double>::quiet_NaN(),
                .last_vol = scan_sigma[i]
            };
            return out;
        }
        scan_sign[i] = (std::abs(scan_f[i]) <= zero_tol)
            ? 0
            : (scan_f[i] > 0.0 ? 1 : -1);
        if (scan_sign[i] == 0) ++zero_samples;
    }

    // Every sample a zero: an unresolved continuum of roots.
    if (zero_samples == kScanPoints) {
        out.refusal = IVError{
            .code = IVErrorCode::MultipleRoots,
            .iterations = 0,
            .final_error = 0.0,
            .last_vol = sigma_min
        };
        return out;
    }

    // Walk the nonzero samples.  Consecutive zeros collapse into one
    // zero run; a run between opposite signs is one transition, a run
    // between equal signs is a tangency contact, and a run at a bracket
    // endpoint is a boundary root.
    size_t transitions = 0;
    size_t tangencies = 0;
    size_t boundary_roots = 0;
    bool leading_boundary = false;
    double lowest_feature_sigma = sigma_max;
    size_t first_lo = 0;
    size_t first_hi = 0;

    auto note_feature = [&](double sigma) {
        lowest_feature_sigma = std::min(lowest_feature_sigma, sigma);
    };

    int last_sign = 0;
    size_t last_idx = 0;
    bool have_last = false;
    for (size_t i = 0; i < kScanPoints; ++i) {
        if (scan_sign[i] == 0) continue;
        if (have_last) {
            if (scan_sign[i] != last_sign) {
                if (transitions == 0) {
                    first_lo = last_idx;
                    first_hi = i;
                }
                ++transitions;
                note_feature(scan_sigma[last_idx]);
            } else if (i > last_idx + 1) {
                ++tangencies;
                note_feature(scan_sigma[last_idx + 1]);
            }
        } else if (i > 0) {
            ++boundary_roots;
            leading_boundary = true;
            note_feature(sigma_min);
        }
        last_sign = scan_sign[i];
        last_idx = i;
        have_last = true;
    }
    if (last_idx + 1 < kScanPoints) {
        ++boundary_roots;
        note_feature(sigma_max);
    }

    // Root features found by the scan.  A tangency counts as two: an
    // even-multiplicity contact is at least a double root, which makes
    // it ambiguous on its own.  Anything beyond a single feature is
    // ambiguous by construction.
    const size_t features = transitions + 2 * tangencies + boundary_roots;
    if (features > 1) {
        out.refusal = IVError{
            .code = IVErrorCode::MultipleRoots,
            .iterations = 0,
            .final_error = static_cast<double>(features),
            .last_vol = lowest_feature_sigma
        };
        return out;
    }

    if (boundary_roots == 1) {
        // Boundary root: honor it only when it also satisfies the
        // solver's configured convergence tolerance.  zero_tol must
        // never silently loosen a user's tighter tolerance.
        const double endpoint = leading_boundary ? sigma_min : sigma_max;
        const double residual =
            std::abs(leading_boundary ? scan_f[0] : scan_f[kScanPoints - 1]);
        if (residual <= tolerance) {
            out.boundary_root = IVSuccess{
                .implied_vol = endpoint,
                .iterations = 0,
                .final_error = residual,
                .vega = std::nullopt,
                .used_rate_approximation = false
            };
            return out;
        }
        // The scan found no true bracket: report what the unscreened
        // path would have reported.
        out.refusal = IVError{
            .code = IVErrorCode::BracketingFailed,
            .iterations = 0,
            .final_error = residual,
            .last_vol = endpoint
        };
        return out;
    }

    if (transitions == 1) {
        out.lo = scan_sigma[first_lo];
        out.hi = scan_sigma[first_hi];
        out.f_lo = scan_f[first_lo];
        out.f_hi = scan_f[first_hi];
        out.check_slope = true;
    }
    // No transition: Brent runs on the full bracket, which reports
    // BracketingFailed exactly as the unscreened path does.
    return out;
}

// =====================================================================
// Effective volatility bracket
// =====================================================================

std::pair<double, double> effective_sigma_bracket(
    double spot, double strike, OptionType type, double target_price,
    const SurfaceInversionPolicy& policy) noexcept
{
    double intrinsic = intrinsic_value(spot, strike, type);

    // Analyze time value to set adaptive bounds
    const double time_value = target_price - intrinsic;
    const double time_value_pct = time_value / target_price;

    double sigma_upper;
    if (time_value_pct > 0.5) {
        sigma_upper = 3.0;  // 300%
    } else if (time_value_pct > 0.2) {
        sigma_upper = 2.0;  // 200%
    } else {
        sigma_upper = 1.5;  // 150%
    }

    double sigma_min = std::max(policy.config_sigma_min, policy.published_sigma_min);
    double sigma_max = std::min({sigma_upper, policy.config_sigma_max,
                                 policy.published_sigma_max});

    if (sigma_min >= sigma_max) {
        sigma_min = policy.published_sigma_min;
        sigma_max = policy.published_sigma_max;
    }

    return {sigma_min, sigma_max};
}

// =====================================================================
// Inversion
// =====================================================================

std::expected<IVSuccess, IVError> invert_price_on_surface(
    detail::ObjectiveRef price, detail::ObjectiveRef vega,
    double target_price, std::pair<double, double> bracket,
    double spot, const SurfaceInversionPolicy& policy) noexcept
{
    const double sigma_min = bracket.first;
    const double sigma_max = bracket.second;

    // Vega pre-check: reject queries where the option has no usable
    // sensitivity to volatility.  Probes are the quartile points of the
    // actual bracket (fixed probe vols could fall outside it entirely) and
    // the maximum is signed: a uniformly negative vega is a broken surface,
    // not a healthy one.  ~600 ns, saves a doomed Brent search.
    if (policy.vega_threshold > 0.0) {
        const double vega_span = sigma_max - sigma_min;
        const double probe_vols[3] = {sigma_min + 0.25 * vega_span,
                                      sigma_min + 0.50 * vega_span,
                                      sigma_min + 0.75 * vega_span};
        double max_vega = -std::numeric_limits<double>::infinity();
        for (double sv : probe_vols) {
            const double v = vega(sv);
            if (!std::isfinite(v)) {
                return std::unexpected(IVError{
                    .code = IVErrorCode::NumericalInstability,
                    .iterations = 0,
                    .final_error = std::numeric_limits<double>::quiet_NaN(),
                    .last_vol = sv
                });
            }
            max_vega = std::max(max_vega, v);
        }
        if (max_vega < policy.vega_threshold) {
            return std::unexpected(IVError{
                .code = IVErrorCode::VegaTooSmall,
                .iterations = 0,
                .final_error = max_vega,
                .last_vol = std::nullopt
            });
        }
    }

    // Define objective function: f(s) = Price(s) - Market_Price
    auto objective = [&](double sigma) -> double {
        return price(sigma) - target_price;
    };

    // Bracket handed to Brent.  The multiple-root screen may narrow it to
    // the single scan interval that contains a sign change.
    double brent_lo = sigma_min;
    double brent_hi = sigma_max;
    bool check_narrowed_slope = false;
    double narrowed_f_lo = 0.0;
    double narrowed_f_hi = 0.0;

    // Multiple-root screen (spec D8.2).  A price surface that is not
    // monotone in sigma admits several implied vols for one market price;
    // Brent would silently return whichever one it lands on.
    // `detail::screen_bracket` samples the objective on a uniform 17-point
    // scan and refuses ambiguous brackets; a single sign change narrows the
    // bracket handed to Brent.
    if (policy.detect_multiple_roots) {
        auto screen = detail::screen_bracket(objective, sigma_min, sigma_max,
                                             spot, policy.tolerance);
        if (screen.refusal.has_value()) {
            return std::unexpected(*screen.refusal);
        }
        if (screen.boundary_root.has_value()) {
            return *screen.boundary_root;
        }
        brent_lo = screen.lo;
        brent_hi = screen.hi;
        check_narrowed_slope = screen.check_slope;
        narrowed_f_lo = screen.f_lo;
        narrowed_f_hi = screen.f_hi;
    }

    // Brent's method
    RootFindingConfig brent_config{
        .max_iter = policy.max_iter,
        .brent_tol_abs = policy.tolerance
    };

    auto result = find_root(objective, brent_lo, brent_hi, brent_config);

    // Check convergence - transform RootFindingError to IVError
    if (!result.has_value()) {
        const auto& root_error = result.error();
        IVErrorCode error_code;
        switch (root_error.code) {
            case RootFindingErrorCode::MaxIterationsExceeded:
                error_code = IVErrorCode::MaxIterationsExceeded;
                break;
            case RootFindingErrorCode::InvalidBracket:
                error_code = IVErrorCode::BracketingFailed;
                break;
            case RootFindingErrorCode::NumericalInstability:
                error_code = IVErrorCode::NumericalInstability;
                break;
            case RootFindingErrorCode::NoProgress:
                error_code = IVErrorCode::NumericalInstability;
                break;
            default:
                error_code = IVErrorCode::NumericalInstability;
                break;
        }

        return std::unexpected(IVError{
            .code = error_code,
            .iterations = root_error.iterations,
            .final_error = root_error.final_error,
            .last_vol = root_error.last_value
        });
    }

    // Post-hoc slope check on the narrowed interval.  A converged root is
    // only trustworthy if the objective rises through it; a falling
    // objective means the surface is non-monotone in sigma there, so the
    // root the screen isolated is not the only one.  Reuses the scan
    // samples — no extra surface evaluations.
    if (check_narrowed_slope) {
        const double slope = (narrowed_f_hi - narrowed_f_lo) / (brent_hi - brent_lo);
        if (!(slope > 0.0)) {
            return std::unexpected(IVError{
                .code = IVErrorCode::MultipleRoots,
                .iterations = result->iterations,
                .final_error = 1.0,
                .last_vol = brent_lo
            });
        }
    }

    return IVSuccess{
        .implied_vol = result->root,
        .iterations = result->iterations,
        .final_error = result->final_error,
        .vega = std::nullopt,
        .used_rate_approximation = false
    };
}

}  // namespace mango
