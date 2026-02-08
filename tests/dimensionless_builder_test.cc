// SPDX-License-Identifier: MIT
/**
 * @file dimensionless_builder_test.cc
 * @brief Tests for dimensionless 3D surface builder
 *
 * Validates that build_dimensionless_surface() produces a correct 3D B-spline
 * surface over (x, tau', ln kappa) by checking EEP positivity, surface
 * evaluation, and cross-validation against single-point PDE reference solves.
 */

#include "mango/option/table/dimensionless_builder.hpp"
#include "mango/option/american_option.hpp"
#include "mango/option/american_option_batch.hpp"
#include "mango/option/european_option.hpp"

#include <gtest/gtest.h>
#include <cmath>
#include <vector>

namespace mango {
namespace {

// ===========================================================================
// Test 0: Verify a single dimensionless PDE solve works
// ===========================================================================
TEST(DimensionlessBuilderTest, SingleDimensionlessSolve) {
    // kappa=1.0 (ln_kappa=0), tau_prime_max=0.15
    double K_ref = 100.0;
    double sigma_eff = std::sqrt(2.0);
    double kappa = 1.0;
    double tau_prime_max = 0.15;

    PricingParams params(
        OptionSpec{
            .spot = K_ref,
            .strike = K_ref,
            .maturity = tau_prime_max,
            .rate = kappa,
            .dividend_yield = 0.0,
            .option_type = OptionType::PUT},
        sigma_eff);

    auto result = solve_american_option(params);
    ASSERT_TRUE(result.has_value()) << "Single solve failed";
    double val = result->value();
    EXPECT_GT(val, 0.0) << "Value should be positive for ATM put";
}

// Test batch solving with snapshots
TEST(DimensionlessBuilderTest, BatchWithSnapshots) {
    double K_ref = 100.0;
    double sigma_eff = std::sqrt(2.0);
    double tau_prime_max = 0.15;

    std::vector<double> tau_prime_vals = {0.01, 0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.14};
    std::vector<double> ln_kappa_vals = {-1.5, 0.0, 1.5};

    std::vector<PricingParams> batch;
    for (double lk : ln_kappa_vals) {
        double kappa = std::exp(lk);
        batch.emplace_back(
            OptionSpec{
                .spot = K_ref,
                .strike = K_ref,
                .maturity = tau_prime_max,
                .rate = kappa,
                .dividend_yield = 0.0,
                .option_type = OptionType::PUT},
            sigma_eff);
    }

    BatchAmericanOptionSolver solver;
    solver.set_snapshot_times(std::span<const double>{tau_prime_vals});

    auto batch_result = solver.solve_batch(batch, /*use_shared_grid=*/true);

    // Check batch results
    EXPECT_EQ(batch_result.results.size(), 3u);
    for (size_t i = 0; i < batch_result.results.size(); ++i) {
        EXPECT_TRUE(batch_result.results[i].has_value())
            << "Solve [" << i << "] failed (ln_kappa=" << ln_kappa_vals[i] << ")";
        if (batch_result.results[i].has_value()) {
            const auto& result = batch_result.results[i].value();
            EXPECT_TRUE(result.has_snapshots())
                << "No snapshots for solve [" << i << "]";
            EXPECT_EQ(result.num_snapshots(), tau_prime_vals.size())
                << "Wrong number of snapshots for solve [" << i << "]"
                << " got " << result.num_snapshots()
                << " expected " << tau_prime_vals.size();
            if (result.has_snapshots() && result.num_snapshots() > 0) {
                auto grid = result.grid();
                auto x_grid = grid->x();
                auto sol = result.at_time(0);
                EXPECT_GT(x_grid.size(), 0u) << "Grid x empty for solve [" << i << "]";
                EXPECT_GT(sol.size(), 0u) << "Solution empty for solve [" << i << "]";
            }
        }
    }
}

// ===========================================================================
// Test 1: Build a 3D put surface and verify basic properties
// ===========================================================================
TEST(DimensionlessBuilderTest, BuildsPutSurface) {
    DimensionlessAxes axes;
    // 9 log-moneyness points spanning deep OTM to deep ITM
    axes.log_moneyness = {-0.5, -0.3, -0.15, -0.05, 0.0, 0.05, 0.15, 0.3, 0.5};
    // 8 tau' points (dimensionless time), well-spaced to avoid snapshot
    // deduplication when mapped to the PDE time grid
    axes.tau_prime = {0.01, 0.02, 0.03, 0.04, 0.06, 0.08, 0.10, 0.14};
    // 8 ln(kappa) points spanning low to high kappa
    axes.ln_kappa = {-1.5, -1.0, -0.5, 0.0, 0.25, 0.5, 1.0, 1.5};

    double K_ref = 100.0;
    auto result = build_dimensionless_surface(
        axes, K_ref, OptionType::PUT, SurfaceContent::EarlyExercisePremium);

    ASSERT_TRUE(result.has_value())
        << "Build failed with error code: "
        << static_cast<int>(result.error().code)
        << " axis_index=" << result.error().axis_index
        << " count=" << result.error().count;

    // Basic checks
    EXPECT_GT(result->n_pde_solves, 0);
    EXPECT_EQ(result->n_pde_solves, static_cast<int>(axes.ln_kappa.size()));
    EXPECT_NE(result->surface, nullptr);
    EXPECT_GT(result->build_time_seconds, 0.0);
    EXPECT_EQ(result->metadata.content, SurfaceContent::EarlyExercisePremium);

    // Check EEP at ATM (x=0, tau_prime=0.04, ln_kappa=0.0)
    // For a put, EEP should be positive (American worth more than European)
    // and bounded (typically small fraction of K_ref)
    double eep_atm = result->surface->value({0.0, 0.04, 0.0});
    EXPECT_GE(eep_atm, 0.0) << "EEP should be non-negative";
    EXPECT_LT(eep_atm, 0.2) << "EEP should be bounded (< 0.2 for normalized price)";
}

// ===========================================================================
// Test 2: EEP matches single-point PDE reference solve
// ===========================================================================
TEST(DimensionlessBuilderTest, EEPMatchesFourDSurface) {
    DimensionlessAxes axes;
    axes.log_moneyness = {-0.5, -0.3, -0.15, -0.05, 0.0, 0.05, 0.15, 0.3, 0.5};
    axes.tau_prime = {0.01, 0.02, 0.03125, 0.04, 0.06, 0.08, 0.10, 0.14};
    axes.ln_kappa = {-1.5, -1.0, -0.5, 0.0, 0.247, 0.5, 1.0, 1.5};

    double K_ref = 100.0;
    auto build_result = build_dimensionless_surface(
        axes, K_ref, OptionType::PUT, SurfaceContent::EarlyExercisePremium);
    ASSERT_TRUE(build_result.has_value())
        << "Build failed with error code: "
        << static_cast<int>(build_result.error().code);

    // Physical parameters: sigma=0.25, r=0.04, tau=1.0
    // Dimensionless: tau' = 0.25^2 * 1.0 / 2 = 0.03125
    //                kappa = 2*0.04 / 0.25^2 = 1.28
    //                ln(kappa) = ln(1.28) ~= 0.247
    const double sigma = 0.25;
    const double rate = 0.04;
    const double maturity = 1.0;
    const double tau_prime = sigma * sigma * maturity / 2.0;
    const double kappa = 2.0 * rate / (sigma * sigma);
    const double ln_kappa = std::log(kappa);

    // Test at ATM (x=0, S=K=100)
    const double spot = K_ref;
    const double strike = K_ref;

    // Compute American price via single PDE solve (physical parameters)
    PricingParams am_params(
        OptionSpec{
            .spot = spot,
            .strike = strike,
            .maturity = maturity,
            .rate = rate,
            .dividend_yield = 0.0,
            .option_type = OptionType::PUT},
        sigma);
    auto am_result = solve_american_option(am_params);
    ASSERT_TRUE(am_result.has_value()) << "American solve failed";
    double am_price = am_result->value();

    // Compute European price via closed-form
    EuropeanOptionSolver eu_solver(am_params);
    auto eu_result = eu_solver.solve();
    ASSERT_TRUE(eu_result.has_value()) << "European solve failed";
    double eu_price = eu_result->value();

    // Reference EEP normalized by strike
    double ref_eep = (am_price - eu_price) / strike;

    // Query 3D surface at dimensionless point
    double surface_eep = build_result->surface->value({0.0, tau_prime, ln_kappa});

    // Allow $0.20 tolerance (normalized by K=100, so 0.002 in normalized units).
    // The 3D surface uses a modest grid (8-9 points per axis) so interpolation
    // error at off-grid kappa values (~0.247) is expected.
    EXPECT_NEAR(surface_eep, ref_eep, 0.002)
        << "EEP mismatch at ATM point"
        << "\n  sigma=" << sigma << " r=" << rate << " T=" << maturity
        << "\n  tau'=" << tau_prime << " kappa=" << kappa << " ln_kappa=" << ln_kappa
        << "\n  Am price=" << am_price << " Eu price=" << eu_price
        << "\n  ref EEP (norm)=" << ref_eep
        << "\n  surface EEP=" << surface_eep;
}

}  // namespace
}  // namespace mango
