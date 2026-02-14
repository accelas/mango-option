// SPDX-License-Identifier: MIT
/// Compare current adaptive dimensionless surface against the golden surface
/// from experiment/dimensionless-3d branch.
///
/// Both surfaces store dollar EEP values (V_am - V_eu) in dimensionless coords.
/// The golden surface was built with the old code (European subtraction inside
/// builder); the current code uses the accessor + eep_decompose pattern.
///
/// Usage: bazel run //benchmarks:golden_surface_comparison

#include "mango/option/table/dimensionless/dimensionless_builder.hpp"
#include "mango/option/table/bspline/bspline_surface.hpp"
#include "mango/math/bspline/bspline_nd.hpp"
#include "mango/math/bspline/bspline_basis.hpp"
#include <array>
#include <cmath>
#include <cstdio>
#include <memory>
#include <vector>

using namespace mango;

// ============================================================================
// Reconstruct golden surface from embedded data
// ============================================================================

namespace {
#include "golden_surface_data.inc"

SegmentedDimensionlessSurface::Segment build_golden_segment(
    const double* x, size_t nx,
    const double* tp, size_t ntp,
    const double* lk, size_t nlk,
    const double* coeffs, size_t nc,
    double lk_min, double lk_max)
{
    std::array<std::vector<double>, 3> grids;
    grids[0].assign(x, x + nx);
    grids[1].assign(tp, tp + ntp);
    grids[2].assign(lk, lk + nlk);

    std::array<std::vector<double>, 3> knots;
    knots[0] = clamped_knots_cubic<double>(grids[0]);
    knots[1] = clamped_knots_cubic<double>(grids[1]);
    knots[2] = clamped_knots_cubic<double>(grids[2]);

    auto spline = BSplineND<double, 3>::create(
        grids, knots,
        std::vector<double>(coeffs, coeffs + nc)).value();

    return {.spline = std::make_shared<const BSplineND<double, 3>>(std::move(spline)),
            .lk_min = lk_min, .lk_max = lk_max};
}

std::shared_ptr<SegmentedDimensionlessSurface> load_golden() {
    std::vector<SegmentedDimensionlessSurface::Segment> segments;
    segments.reserve(kNumSegments);

    segments.push_back(build_golden_segment(
        kSeg0X, std::size(kSeg0X), kSeg0Tp, std::size(kSeg0Tp),
        kSeg0Lk, std::size(kSeg0Lk), kSeg0Coeffs, std::size(kSeg0Coeffs),
        kSeg0LkMin, kSeg0LkMax));
    segments.push_back(build_golden_segment(
        kSeg1X, std::size(kSeg1X), kSeg1Tp, std::size(kSeg1Tp),
        kSeg1Lk, std::size(kSeg1Lk), kSeg1Coeffs, std::size(kSeg1Coeffs),
        kSeg1LkMin, kSeg1LkMax));
    segments.push_back(build_golden_segment(
        kSeg2X, std::size(kSeg2X), kSeg2Tp, std::size(kSeg2Tp),
        kSeg2Lk, std::size(kSeg2Lk), kSeg2Coeffs, std::size(kSeg2Coeffs),
        kSeg2LkMin, kSeg2LkMax));

    return std::make_shared<SegmentedDimensionlessSurface>(std::move(segments));
}

}  // namespace

// ============================================================================
// Comparison
// ============================================================================

int main() {
    std::printf("Golden Surface vs Current Adaptive Surface Comparison\n");
    std::printf("=====================================================\n\n");

    // Load golden surface
    std::printf("Loading golden surface (3 segments, ~5K coefficients)...\n");
    auto golden = load_golden();
    std::printf("  Golden: %zu segments, lk range [%.3f, %.3f]\n",
        golden->num_segments(),
        golden->segments().front().lk_min,
        golden->segments().back().lk_max);

    // Build current adaptive surface with same domain
    std::printf("Building current adaptive surface (same domain)...\n");
    DimensionlessAdaptiveParams params;  // defaults match golden's domain
    auto result = build_dimensionless_surface_adaptive(params, 100.0);
    if (!result.has_value()) {
        std::fprintf(stderr, "Adaptive build failed: code=%d\n",
            static_cast<int>(result.error().code));
        return 1;
    }
    auto current = result->surface;
    std::printf("  Current: %zu segments, %d PDE solves, %zu iterations\n",
        result->num_segments, result->total_pde_solves, result->iterations_used);
    std::printf("  Max error: %.6f, target_met: %s\n",
        result->achieved_max_error, result->target_met ? "yes" : "no");
    std::printf("  lk range [%.3f, %.3f]\n\n",
        current->segments().front().lk_min,
        current->segments().back().lk_max);

    // Print per-segment grid sizes
    std::printf("Per-segment grid comparison:\n");
    std::printf("  %-8s  %6s  %6s  %6s  %10s  %10s\n",
        "Seg", "Nx", "Ntp", "Nlk", "lk_min", "lk_max");
    for (size_t s = 0; s < golden->num_segments(); ++s) {
        const auto& gs = golden->segments()[s];
        const auto& sp = *gs.spline;
        std::printf("  gold[%zu]  %6zu  %6zu  %6zu  %10.3f  %10.3f\n",
            s, sp.grid(0).size(), sp.grid(1).size(), sp.grid(2).size(),
            gs.lk_min, gs.lk_max);
    }
    for (size_t s = 0; s < current->num_segments(); ++s) {
        const auto& cs = current->segments()[s];
        const auto& sp = *cs.spline;
        std::printf("  curr[%zu]  %6zu  %6zu  %6zu  %10.3f  %10.3f\n",
            s, sp.grid(0).size(), sp.grid(1).size(), sp.grid(2).size(),
            cs.lk_min, cs.lk_max);
    }

    // Print actual grid values for first segment
    std::printf("\nSegment 0 grid detail:\n");
    auto print_grid = [](const char* label, const std::vector<double>& g) {
        std::printf("  %s [%zu]: ", label, g.size());
        for (size_t i = 0; i < g.size(); ++i) {
            std::printf("%.4f", g[i]);
            if (i + 1 < g.size()) std::printf(", ");
        }
        std::printf("\n");
    };
    {
        const auto& gs = *golden->segments()[0].spline;
        const auto& cs = *current->segments()[0].spline;
        print_grid("gold x  ", gs.grid(0));
        print_grid("curr x  ", cs.grid(0));
        print_grid("gold tp ", gs.grid(1));
        print_grid("curr tp ", cs.grid(1));
        print_grid("gold lk ", gs.grid(2));
        print_grid("curr lk ", cs.grid(2));
    }
    std::printf("\n");

    // Compare at probe points across the physical domain
    // Domain: S/K in [0.65, 1.50], sigma in [0.10, 0.80], rate in [0.005, 0.10]
    // In dimensionless coords:
    //   x = ln(S/K) in [ln(0.65), ln(1.50)] = [-0.431, 0.405]
    //   tau' = sigma^2 * tau / 2, tau in [7/365, 2.0]
    //   ln_kappa = ln(2r/sigma^2)

    constexpr size_t NX = 9, NTP = 8, NLK = 11;
    double x_vals[NX], tp_vals[NTP], lk_vals[NLK];

    // log-moneyness grid
    for (size_t i = 0; i < NX; ++i)
        x_vals[i] = std::log(0.65) + (std::log(1.50) - std::log(0.65))
                     * static_cast<double>(i) / (NX - 1);

    // tau' grid (covering short to long dated)
    double tp_lo = 0.10 * 0.10 * (7.0/365) / 2.0;
    double tp_hi = 0.80 * 0.80 * 2.0 / 2.0;
    for (size_t i = 0; i < NTP; ++i)
        tp_vals[i] = tp_lo + (tp_hi - tp_lo)
                     * static_cast<double>(i) / (NTP - 1);

    // ln_kappa grid
    double lk_lo = std::log(2.0 * 0.005 / (0.80 * 0.80));
    double lk_hi = std::log(2.0 * 0.10 / (0.10 * 0.10));
    for (size_t i = 0; i < NLK; ++i)
        lk_vals[i] = lk_lo + (lk_hi - lk_lo)
                     * static_cast<double>(i) / (NLK - 1);

    // Compare: golden stores normalized EEP (V/K units), current stores dollar
    // EEP (K_ref * V/K). Divide current by K_ref for apples-to-apples comparison.
    constexpr double K_ref = 100.0;

    size_t n_total = 0, n_agree = 0;
    double max_abs_diff = 0, sum_sq_diff = 0, sum_sq_rel = 0;
    size_t n_rel = 0;
    double worst_coords[3] = {};
    double worst_golden = 0, worst_current = 0;

    for (size_t xi = 0; xi < NX; ++xi) {
        for (size_t ti = 0; ti < NTP; ++ti) {
            for (size_t ki = 0; ki < NLK; ++ki) {
                std::array<double, 3> c = {x_vals[xi], tp_vals[ti], lk_vals[ki]};
                double g = golden->value(c);
                double v = current->value(c) / K_ref;  // normalize to V/K units
                double diff = std::abs(g - v);

                n_total++;
                sum_sq_diff += diff * diff;

                if (diff < 1e-4) n_agree++;

                if (g > 1e-6) {
                    double rel = diff / g;
                    sum_sq_rel += rel * rel;
                    n_rel++;
                }

                if (diff > max_abs_diff) {
                    max_abs_diff = diff;
                    worst_coords[0] = c[0];
                    worst_coords[1] = c[1];
                    worst_coords[2] = c[2];
                    worst_golden = g;
                    worst_current = v;
                }
            }
        }
    }

    double rms_abs = std::sqrt(sum_sq_diff / n_total);
    double rms_rel = n_rel > 0 ? std::sqrt(sum_sq_rel / n_rel) : 0;

    std::printf("Comparison over %zu probe points\n", n_total);
    std::printf("================================\n");
    std::printf("  Agree (<1e-4):    %zu/%zu (%.1f%%)\n",
        n_agree, n_total, 100.0 * n_agree / n_total);
    std::printf("  RMS abs diff:     %.6f\n", rms_abs);
    std::printf("  RMS rel diff:     %.4f%% (%zu points with golden>1e-6)\n",
        rms_rel * 100, n_rel);
    std::printf("  Max abs diff:     %.6f\n", max_abs_diff);
    std::printf("    at (x=%.3f, tp=%.4f, lk=%.3f)\n",
        worst_coords[0], worst_coords[1], worst_coords[2]);
    std::printf("    golden=%.6f, current=%.6f\n\n", worst_golden, worst_current);

    // Per-lk slice comparison
    std::printf("Per-lk_kappa slice RMS:\n");
    std::printf("  %-10s  %12s  %12s  %12s\n",
        "ln_kappa", "golden_mean", "current_mean", "RMS_diff");
    for (size_t ki = 0; ki < NLK; ++ki) {
        double g_sum = 0, c_sum = 0, sq_diff = 0;
        size_t n = 0;
        for (size_t xi = 0; xi < NX; ++xi) {
            for (size_t ti = 0; ti < NTP; ++ti) {
                std::array<double, 3> c = {x_vals[xi], tp_vals[ti], lk_vals[ki]};
                double g = golden->value(c);
                double v = current->value(c) / K_ref;
                g_sum += g;
                c_sum += v;
                sq_diff += (g - v) * (g - v);
                n++;
            }
        }
        std::printf("  %10.3f  %12.6f  %12.6f  %12.6f\n",
            lk_vals[ki], g_sum / n, c_sum / n, std::sqrt(sq_diff / n));
    }

    return 0;
}
