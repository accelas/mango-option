// SPDX-License-Identifier: MIT
/**
 * @file dimensionless_adaptive_test.cc
 * @brief Tests for adaptive dimensionless 3D surface builder
 */

#include "mango/option/table/dimensionless/dimensionless_builder.hpp"
#include "mango/option/table/dimensionless/dimensionless_european.hpp"
#include "mango/option/american_option_batch.hpp"
#include "mango/math/cubic_spline_solver.hpp"

#include <gtest/gtest.h>
#include <cmath>

namespace mango {
namespace {

/// Independent reference EEP solve for validation
static double reference_eep_test(double x0, double tau_prime_0, double ln_kappa_0,
                                  double K_ref, OptionType option_type) {
    double kappa = std::exp(ln_kappa_0);
    BatchAmericanOptionSolver solver;
    solver.set_grid_accuracy(make_grid_accuracy(GridAccuracyProfile::Ultra));
    std::vector<double> snap_times = {tau_prime_0};
    solver.set_snapshot_times(std::span<const double>{snap_times});
    double tau_prime_max = std::max(tau_prime_0 * 1.01, 0.02);
    std::vector<PricingParams> batch;
    batch.emplace_back(
        OptionSpec{.spot = K_ref, .strike = K_ref, .maturity = tau_prime_max,
                   .rate = kappa, .dividend_yield = 0.0, .option_type = option_type},
        std::sqrt(2.0));
    auto result = solver.solve_batch(batch, false);
    if (result.results.empty() || !result.results[0].has_value()) return 0.0;
    const auto& sol = result.results[0].value();
    if (sol.num_snapshots() < 1) return 0.0;
    auto grid = sol.grid();
    CubicSpline<double> spline;
    auto err = spline.build(grid->x(), sol.at_time(0));
    if (err.has_value()) return 0.0;
    double american = spline.eval(x0);
    double european = dimensionless_european(x0, tau_prime_0, kappa, option_type);
    return std::max(american - european, 0.0);
}

TEST(DimensionlessAdaptiveTest, ConvergesWithDefaultParams) {
    auto result = build_dimensionless_surface_adaptive();
    ASSERT_TRUE(result.has_value())
        << "Adaptive build failed: code="
        << static_cast<int>(result.error().code);

    EXPECT_TRUE(result->target_met)
        << "Target not met: max_error=" << result->achieved_max_error
        << " target=" << DimensionlessAdaptiveParams{}.target_eep_error
        << " segments=" << result->num_segments
        << " iter=" << result->iterations_used;
    EXPECT_NE(result->surface, nullptr);
    EXPECT_GT(result->total_pde_solves, 0);
    EXPECT_GT(result->iterations_used, 0u);
}

TEST(DimensionlessAdaptiveTest, GridGrowsDuringRefinement) {
    auto result = build_dimensionless_surface_adaptive();
    ASSERT_TRUE(result.has_value());

    // With 3 segments of 10 lk points each = 30 seed.
    // At least some axis should have grown.
    EXPECT_GT(result->final_axes.ln_kappa.size(), 10u)
        << "ln_kappa grid did not grow from seed";
}

TEST(DimensionlessAdaptiveTest, SurfaceAccuracySpotCheck) {
    auto result = build_dimensionless_surface_adaptive();
    ASSERT_TRUE(result.has_value());

    // Query surface at ATM (x=0), moderate dimensionless time, moderate kappa
    double eep = result->surface->value({0.0, 0.04, 0.0});
    EXPECT_GE(eep, 0.0) << "EEP should be non-negative";
    EXPECT_LT(eep, 0.2) << "EEP should be bounded for normalized price";
}

TEST(DimensionlessAdaptiveTest, TighterTargetProducesLargerGrid) {
    DimensionlessAdaptiveParams loose;
    loose.target_eep_error = 0.02;  // very loose
    loose.max_iter = 3;

    DimensionlessAdaptiveParams tight;
    tight.target_eep_error = 2e-3;  // tight (default)
    tight.max_iter = 10;

    auto r_loose = build_dimensionless_surface_adaptive(loose);
    auto r_tight = build_dimensionless_surface_adaptive(tight);

    ASSERT_TRUE(r_loose.has_value());
    ASSERT_TRUE(r_tight.has_value());

    EXPECT_GE(r_tight->final_axes.ln_kappa.size(),
              r_loose->final_axes.ln_kappa.size())
        << "Tighter target should produce same or larger grid";
}

// Dense 3D validation: probe at true random points (not grid-aligned).
// This measures the real worst-case 3D interpolation error.
TEST(DimensionlessAdaptiveTest, Dense3DValidation) {
    auto result = build_dimensionless_surface_adaptive();
    ASSERT_TRUE(result.has_value());

    // Sample 125 probes uniformly in the physical domain
    double x_lo = std::log(0.65), x_hi = std::log(1.50);
    double tp_lo = 0.01, tp_hi = 0.60;
    double lk_lo = std::log(2.0 * 0.005 / 0.64), lk_hi = std::log(2.0 * 0.10 / 0.01);

    constexpr size_t N = 5;
    double max_err = 0, sum_err = 0;
    size_t n_probes = 0;
    std::array<double, 3> worst = {};
    double w_true = 0, w_interp = 0;

    for (size_t i = 0; i < N; ++i) {
        double x = x_lo + (x_hi - x_lo) * (i + 0.5) / N;
        for (size_t j = 0; j < N; ++j) {
            double tp = tp_lo + (tp_hi - tp_lo) * (j + 0.5) / N;
            for (size_t k = 0; k < N; ++k) {
                double lk = lk_lo + (lk_hi - lk_lo) * (k + 0.5) / N;

                double true_eep = reference_eep_test(x, tp, lk, 100.0, OptionType::PUT);
                double interp_eep = result->surface->value({x, tp, lk});
                double err = std::abs(true_eep - interp_eep);

                if (err > max_err) {
                    max_err = err;
                    worst = {x, tp, lk};
                    w_true = true_eep;
                    w_interp = interp_eep;
                }
                sum_err += err;
                n_probes++;
            }
        }
    }

    double avg_err = sum_err / n_probes;
    std::printf("=== Dense 3D validation (%zu probes, %zu segments) ===\n",
        n_probes, result->num_segments);
    std::printf("  max_err=%.6f  avg_err=%.6f\n", max_err, avg_err);
    std::printf("  worst at: x=%.4f (S/K=%.4f)  tp=%.4f  lk=%.4f (kappa=%.4f)\n",
        worst[0], std::exp(worst[0]), worst[1], worst[2], std::exp(worst[2]));
    std::printf("  true_eep=%.6f  interp_eep=%.6f\n", w_true, w_interp);

    EXPECT_LT(max_err, 0.01)
        << "3D interpolation error exceeds $1.00 per $100 strike";
    EXPECT_LT(avg_err, 0.003)
        << "Average 3D interpolation error exceeds $0.30 per $100 strike";
}

TEST(DimensionlessAdaptiveTest, DiagnoseWorstError) {
    auto result = build_dimensionless_surface_adaptive();
    ASSERT_TRUE(result.has_value());

    auto [x, tp, lk] = result->worst_probe;
    std::printf("=== Worst probe diagnostic ===\n");
    std::printf("  Segments: %zu  iter=%zu\n",
        result->num_segments, result->iterations_used);
    std::printf("  max_error=%.6f  avg_error=%.6f  target=%s\n",
        result->achieved_max_error, result->achieved_avg_error,
        result->target_met ? "MET" : "NOT MET");
    std::printf("  Worst at: x=%.4f  tp=%.4f  lk=%.4f (kappa=%.4f)\n",
        x, tp, lk, std::exp(lk));
    std::printf("  true_eep=%.6f  interp_eep=%.6f  err=%.6f\n",
        result->worst_true_eep, result->worst_interp_eep,
        std::abs(result->worst_true_eep - result->worst_interp_eep));
    std::printf("  S/K=%.4f\n", std::exp(x));
    std::printf("  lk union grid: %zu points\n", result->final_axes.ln_kappa.size());

    // Print per-segment info
    for (size_t s = 0; s < result->surface->num_segments(); ++s) {
        auto& seg = result->surface->segments()[s];
        auto& ax = seg.surface->axes();
        std::printf("  seg[%zu]: x=%zu tp=%zu lk=%zu  lk_range=[%.3f, %.3f]\n",
            s, ax.grids[0].size(), ax.grids[1].size(), ax.grids[2].size(),
            seg.lk_min, seg.lk_max);
    }
}

}  // namespace
}  // namespace mango
