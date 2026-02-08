// SPDX-License-Identifier: MIT
/**
 * @file dimensionless_comparison_test.cc
 * @brief Head-to-head comparison: 3D dimensionless surface vs direct PDE solves
 *
 * Builds a 3D EEP surface over (x, tau', ln kappa), then compares reconstructed
 * American prices at random physical parameter points against solve_american_option()
 * ground truth. Reports accuracy metrics, build cost, and query throughput.
 */

#include <gtest/gtest.h>
#include "mango/option/table/dimensionless_builder.hpp"
#include "mango/option/table/dimensionless_inner.hpp"
#include "mango/option/american_option.hpp"
#include <chrono>
#include <cmath>
#include <iostream>
#include <random>
#include <vector>

namespace mango {
namespace {

class DimensionlessComparisonTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Build 3D surface with moderate resolution.
        // Ranges chosen to cover the random test point distribution:
        //   sigma in [0.12, 0.45], rate in [0.02, 0.08], tau in [0.2, 1.5]
        //   => tau' in [0.0014, 0.152], ln_kappa in [-2.32, 2.41]
        //   moneyness S/K in [0.85, 1.15] => log-moneyness in [-0.16, 0.14]
        DimensionlessAxes axes;
        axes.log_moneyness = linspace(-0.30, 0.30, 12);
        axes.tau_prime = linspace(0.001, 0.16, 10);
        axes.ln_kappa = linspace(-2.5, 2.8, 10);

        auto result = build_dimensionless_surface(
            axes, K_ref_, OptionType::PUT, SurfaceContent::EarlyExercisePremium);
        ASSERT_TRUE(result.has_value())
            << "3D build failed with error code: "
            << static_cast<int>(result.error().code)
            << " axis_index=" << result.error().axis_index
            << " count=" << result.error().count;

        inner_ = std::make_unique<DimensionlessEEPInner>(
            result->surface, OptionType::PUT, K_ref_, 0.0);
        n_solves_ = result->n_pde_solves;
        build_sec_ = result->build_time_seconds;
    }

    static std::vector<double> linspace(double lo, double hi, int n) {
        std::vector<double> v(n);
        for (int i = 0; i < n; ++i)
            v[i] = lo + (hi - lo) * i / (n - 1);
        return v;
    }

    static constexpr double K_ref_ = 100.0;
    std::unique_ptr<DimensionlessEEPInner> inner_;
    int n_solves_ = 0;
    double build_sec_ = 0.0;
};

// ===========================================================================
// Test 1: Build cost diagnostics
// ===========================================================================
TEST_F(DimensionlessComparisonTest, BuildCost) {
    std::cout << "\n=== Build Cost ===\n";
    std::cout << "PDE solves: " << n_solves_ << "\n";
    std::cout << "Build time: " << build_sec_ << "s\n";

    // With 10 ln_kappa points, expect exactly 10 PDE solves
    EXPECT_EQ(n_solves_, 10);
    EXPECT_GT(build_sec_, 0.0);
}

// ===========================================================================
// Test 2: Price accuracy vs PDE reference at random points
// ===========================================================================
TEST_F(DimensionlessComparisonTest, PriceAccuracyVsPDE) {
    // Compare 3D surface prices against direct PDE solves at random points
    std::mt19937 rng(42);
    std::uniform_real_distribution<> sigma_dist(0.12, 0.45);
    std::uniform_real_distribution<> rate_dist(0.02, 0.08);
    std::uniform_real_distribution<> tau_dist(0.2, 1.5);
    std::uniform_real_distribution<> moneyness_dist(0.85, 1.15);

    const int N = 50;
    double max_abs_error = 0.0;
    double sum_abs_error = 0.0;
    int within_20c = 0;

    std::cout << "\n=== Price Accuracy (3D vs PDE, N=" << N << ") ===\n";

    for (int i = 0; i < N; ++i) {
        double sigma = sigma_dist(rng);
        double r = rate_dist(rng);
        double tau = tau_dist(rng);
        double m = moneyness_dist(rng);
        double S = K_ref_ * m;

        PriceQuery q{.spot = S, .strike = K_ref_, .tau = tau,
                     .sigma = sigma, .rate = r};

        double price_3d = inner_->price(q);

        // PDE reference
        PricingParams params(
            OptionSpec{
                .spot = S, .strike = K_ref_, .maturity = tau,
                .rate = r, .dividend_yield = 0.0,
                .option_type = OptionType::PUT},
            sigma);
        auto ref = solve_american_option(params);
        ASSERT_TRUE(ref.has_value()) << "PDE solve failed at point " << i;
        double price_ref = ref->value_at(S);

        double err = std::abs(price_3d - price_ref);
        max_abs_error = std::max(max_abs_error, err);
        sum_abs_error += err;
        if (err < 0.20) ++within_20c;
    }

    double mean_error = sum_abs_error / N;

    std::cout << "Max error:    $" << max_abs_error << "\n";
    std::cout << "Mean error:   $" << mean_error << "\n";
    std::cout << "Within $0.20: " << within_20c << "/" << N << "\n";

    // 3D surface should match PDE within $1.00 worst-case
    EXPECT_LT(max_abs_error, 1.0)
        << "Worst-case error exceeds $1.00";
    // Most points should be within $0.20
    EXPECT_GT(within_20c, N * 0.70)
        << "Fewer than 70% of points within $0.20";
}

// ===========================================================================
// Test 3: Query throughput
// ===========================================================================
TEST_F(DimensionlessComparisonTest, QueryTime) {
    PriceQuery q{.spot = 100.0, .strike = 100.0, .tau = 1.0,
                 .sigma = 0.20, .rate = 0.05};

    const int N = 100000;

    // Warm up to avoid cold-cache effects
    volatile double sink = 0.0;
    for (int i = 0; i < 1000; ++i) sink += inner_->price(q);

    auto t0 = std::chrono::steady_clock::now();
    double sum = 0.0;
    for (int i = 0; i < N; ++i) {
        sum += inner_->price(q);
    }
    auto t1 = std::chrono::steady_clock::now();

    double ns_per_query =
        std::chrono::duration<double, std::nano>(t1 - t0).count() / N;

    std::cout << "\n=== Query Time ===\n";
    std::cout << "3D price query: " << ns_per_query << " ns/query\n";
    std::cout << "(sum=" << sum << ")\n";

    // Should be under 10us per query (B-spline eval + European analytical)
    EXPECT_LT(ns_per_query, 10000.0)
        << "Query time exceeds 10us threshold";
}

}  // namespace
}  // namespace mango
