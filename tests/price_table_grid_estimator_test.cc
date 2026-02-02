// SPDX-License-Identifier: MIT
#include <gtest/gtest.h>
#include "src/option/table/price_table_grid_estimator.hpp"
#include "src/option/american_option.hpp"

namespace mango {
namespace {

// ===========================================================================
// Basic functionality tests
// ===========================================================================

TEST(PriceTableGridEstimatorTest, DefaultParamsAre4D) {
    PriceTableGridAccuracyParams<4> params;

    EXPECT_DOUBLE_EQ(params.target_iv_error, 0.001);  // 10 bps default
    EXPECT_EQ(params.min_points, 4);
    EXPECT_EQ(params.max_points, 50);
    EXPECT_DOUBLE_EQ(params.curvature_weights[0], 1.0);   // moneyness
    EXPECT_DOUBLE_EQ(params.curvature_weights[1], 1.0);   // maturity
    EXPECT_DOUBLE_EQ(params.curvature_weights[2], 1.5);   // volatility (highest)
    EXPECT_DOUBLE_EQ(params.curvature_weights[3], 0.6);   // rate (lowest)
}

TEST(PriceTableGridEstimatorTest, ProfileOrdering) {
    auto low = make_price_table_grid_accuracy(PriceTableGridProfile::Low);
    auto medium = make_price_table_grid_accuracy(PriceTableGridProfile::Medium);
    auto high = make_price_table_grid_accuracy(PriceTableGridProfile::High);
    auto ultra = make_price_table_grid_accuracy(PriceTableGridProfile::Ultra);

    EXPECT_GT(low.target_iv_error, medium.target_iv_error);
    EXPECT_GT(medium.target_iv_error, high.target_iv_error);
    EXPECT_GT(high.target_iv_error, ultra.target_iv_error);

    EXPECT_LE(low.max_points, medium.max_points);
    EXPECT_LE(medium.max_points, high.max_points);
    EXPECT_LE(high.max_points, ultra.max_points);
}

TEST(PriceTableGridEstimatorTest, PdeProfileOrdering) {
    auto low = make_grid_accuracy(GridAccuracyProfile::Low);
    auto medium = make_grid_accuracy(GridAccuracyProfile::Medium);
    auto high = make_grid_accuracy(GridAccuracyProfile::High);
    auto ultra = make_grid_accuracy(GridAccuracyProfile::Ultra);

    EXPECT_GT(low.tol, medium.tol);
    EXPECT_GT(medium.tol, high.tol);
    EXPECT_GT(high.tol, ultra.tol);

    EXPECT_LT(low.min_spatial_points, medium.min_spatial_points);
    EXPECT_LT(medium.min_spatial_points, high.min_spatial_points);
    EXPECT_LT(high.min_spatial_points, ultra.min_spatial_points);

    EXPECT_LT(low.max_spatial_points, medium.max_spatial_points);
    EXPECT_LT(medium.max_spatial_points, high.max_spatial_points);
    EXPECT_LT(high.max_spatial_points, ultra.max_spatial_points);
}

TEST(PriceTableGridEstimatorTest, EstimateGridForPriceTable_DefaultParams) {
    auto estimate = estimate_grid_for_price_table(
        0.8, 1.2,      // moneyness
        0.1, 2.0,      // maturity
        0.10, 0.50,    // volatility
        0.01, 0.06     // rate
    );

    // All grids should have at least min_points (4)
    EXPECT_GE(estimate.grids[0].size(), 4);  // moneyness
    EXPECT_GE(estimate.grids[1].size(), 4);  // maturity
    EXPECT_GE(estimate.grids[2].size(), 4);  // volatility
    EXPECT_GE(estimate.grids[3].size(), 4);  // rate

    // PDE solves = n_sigma * n_rate
    EXPECT_EQ(estimate.estimated_pde_solves,
              estimate.grids[2].size() * estimate.grids[3].size());
}

TEST(PriceTableGridEstimatorTest, EstimateGridForPriceTable_HigherAccuracy) {
    PriceTableGridAccuracyParams<4> high_accuracy;
    high_accuracy.target_iv_error = 0.0001;  // 1 bp target

    PriceTableGridAccuracyParams<4> low_accuracy;
    low_accuracy.target_iv_error = 0.001;   // 10 bp target

    auto high_est = estimate_grid_for_price_table(
        0.8, 1.2, 0.1, 2.0, 0.10, 0.50, 0.01, 0.06, high_accuracy);
    auto low_est = estimate_grid_for_price_table(
        0.8, 1.2, 0.1, 2.0, 0.10, 0.50, 0.01, 0.06, low_accuracy);

    // Higher accuracy should result in more grid points
    EXPECT_GT(high_est.grids[0].size(), low_est.grids[0].size());
    EXPECT_GT(high_est.grids[2].size(), low_est.grids[2].size());  // volatility
}

TEST(PriceTableGridEstimatorTest, MoneynessShouldBeLogUniform) {
    auto estimate = estimate_grid_for_price_table(
        0.8, 1.2, 0.1, 2.0, 0.10, 0.50, 0.01, 0.06);

    const auto& m = estimate.grids[0];
    ASSERT_GE(m.size(), 4);

    // Check bounds
    EXPECT_NEAR(m.front(), 0.8, 0.01);
    EXPECT_NEAR(m.back(), 1.2, 0.01);

    // Check log-uniform spacing: log(m[i+1]) - log(m[i]) should be constant
    if (m.size() >= 3) {
        double log_spacing1 = std::log(m[1]) - std::log(m[0]);
        double log_spacing2 = std::log(m[2]) - std::log(m[1]);
        EXPECT_NEAR(log_spacing1, log_spacing2, 1e-10);
    }
}

TEST(PriceTableGridEstimatorTest, MaturityShouldBeSqrtUniform) {
    auto estimate = estimate_grid_for_price_table(
        0.8, 1.2, 0.1, 2.0, 0.10, 0.50, 0.01, 0.06);

    const auto& tau = estimate.grids[1];
    ASSERT_GE(tau.size(), 4);

    // Check bounds
    EXPECT_NEAR(tau.front(), 0.1, 0.01);
    EXPECT_NEAR(tau.back(), 2.0, 0.01);

    // Check sqrt-uniform spacing: sqrt(tau[i+1]) - sqrt(tau[i]) should be constant
    if (tau.size() >= 3) {
        double sqrt_spacing1 = std::sqrt(tau[1]) - std::sqrt(tau[0]);
        double sqrt_spacing2 = std::sqrt(tau[2]) - std::sqrt(tau[1]);
        EXPECT_NEAR(sqrt_spacing1, sqrt_spacing2, 1e-10);
    }
}

TEST(PriceTableGridEstimatorTest, VolatilityShouldBeUniform) {
    auto estimate = estimate_grid_for_price_table(
        0.8, 1.2, 0.1, 2.0, 0.10, 0.50, 0.01, 0.06);

    const auto& sigma = estimate.grids[2];
    ASSERT_GE(sigma.size(), 4);

    // Check bounds
    EXPECT_NEAR(sigma.front(), 0.10, 0.001);
    EXPECT_NEAR(sigma.back(), 0.50, 0.001);

    // Check uniform spacing
    if (sigma.size() >= 3) {
        double spacing1 = sigma[1] - sigma[0];
        double spacing2 = sigma[2] - sigma[1];
        EXPECT_NEAR(spacing1, spacing2, 1e-10);
    }
}

TEST(PriceTableGridEstimatorTest, RateShouldBeUniform) {
    auto estimate = estimate_grid_for_price_table(
        0.8, 1.2, 0.1, 2.0, 0.10, 0.50, 0.01, 0.06);

    const auto& r = estimate.grids[3];
    ASSERT_GE(r.size(), 4);

    // Check bounds
    EXPECT_NEAR(r.front(), 0.01, 0.001);
    EXPECT_NEAR(r.back(), 0.06, 0.001);

    // Check uniform spacing
    if (r.size() >= 3) {
        double spacing1 = r[1] - r[0];
        double spacing2 = r[2] - r[1];
        EXPECT_NEAR(spacing1, spacing2, 1e-10);
    }
}

TEST(PriceTableGridEstimatorTest, CurvatureWeightsAffectPointAllocation) {
    PriceTableGridAccuracyParams<4> params;
    params.curvature_weights = {1.0, 1.0, 2.0, 0.5};  // Double sigma, halve rate

    auto estimate = estimate_grid_for_price_table(
        0.8, 1.2, 0.1, 2.0, 0.10, 0.50, 0.01, 0.06, params);

    // Volatility (weight 2.0) should have more points than rate (weight 0.5)
    EXPECT_GT(estimate.grids[2].size(), estimate.grids[3].size());
}

// ===========================================================================
// estimate_grid_from_grid_bounds tests
// ===========================================================================

TEST(PriceTableGridEstimatorTest, FromChainBounds_BasicFunctionality) {
    std::vector<double> strikes = {90.0, 95.0, 100.0, 105.0, 110.0};
    double spot = 100.0;
    std::vector<double> maturities = {0.1, 0.25, 0.5, 1.0};
    std::vector<double> vols = {0.15, 0.20, 0.25, 0.30};
    std::vector<double> rates = {0.02, 0.03, 0.04, 0.05};

    auto estimate = estimate_grid_from_grid_bounds(
        strikes, spot, maturities, vols, rates);

    // All grids should be non-empty
    EXPECT_FALSE(estimate.grids[0].empty());
    EXPECT_FALSE(estimate.grids[1].empty());
    EXPECT_FALSE(estimate.grids[2].empty());
    EXPECT_FALSE(estimate.grids[3].empty());

    // Moneyness grid should cover the range (with padding)
    // spot/max_strike = 100/110 ≈ 0.909
    // spot/min_strike = 100/90 ≈ 1.111
    EXPECT_LT(estimate.grids[0].front(), 100.0 / 110.0);  // Below spot/max_strike
    EXPECT_GT(estimate.grids[0].back(), 100.0 / 90.0);    // Above spot/min_strike
}

// ===========================================================================
// Regression tests: empty vector handling
// ===========================================================================

TEST(PriceTableGridEstimatorTest, FromChainBounds_EmptyStrikes_ReturnsEmpty) {
    std::vector<double> strikes = {};  // Empty!
    double spot = 100.0;
    std::vector<double> maturities = {0.1, 0.5};
    std::vector<double> vols = {0.20};
    std::vector<double> rates = {0.03};

    auto estimate = estimate_grid_from_grid_bounds(
        strikes, spot, maturities, vols, rates);

    // Should return empty grids, not crash
    EXPECT_TRUE(estimate.grids[0].empty());
    EXPECT_TRUE(estimate.grids[1].empty());
    EXPECT_TRUE(estimate.grids[2].empty());
    EXPECT_TRUE(estimate.grids[3].empty());
}

TEST(PriceTableGridEstimatorTest, FromChainBounds_EmptyMaturities_ReturnsEmpty) {
    std::vector<double> strikes = {90.0, 100.0, 110.0};
    double spot = 100.0;
    std::vector<double> maturities = {};  // Empty!
    std::vector<double> vols = {0.20};
    std::vector<double> rates = {0.03};

    auto estimate = estimate_grid_from_grid_bounds(
        strikes, spot, maturities, vols, rates);

    EXPECT_TRUE(estimate.grids[0].empty());
}

TEST(PriceTableGridEstimatorTest, FromChainBounds_EmptyVols_ReturnsEmpty) {
    std::vector<double> strikes = {90.0, 100.0, 110.0};
    double spot = 100.0;
    std::vector<double> maturities = {0.1, 0.5};
    std::vector<double> vols = {};  // Empty!
    std::vector<double> rates = {0.03};

    auto estimate = estimate_grid_from_grid_bounds(
        strikes, spot, maturities, vols, rates);

    EXPECT_TRUE(estimate.grids[0].empty());
}

TEST(PriceTableGridEstimatorTest, FromChainBounds_EmptyRates_ReturnsEmpty) {
    std::vector<double> strikes = {90.0, 100.0, 110.0};
    double spot = 100.0;
    std::vector<double> maturities = {0.1, 0.5};
    std::vector<double> vols = {0.20};
    std::vector<double> rates = {};  // Empty!

    auto estimate = estimate_grid_from_grid_bounds(
        strikes, spot, maturities, vols, rates);

    EXPECT_TRUE(estimate.grids[0].empty());
}

TEST(PriceTableGridEstimatorTest, FromChainBounds_ZeroSpot_ReturnsEmpty) {
    std::vector<double> strikes = {90.0, 100.0, 110.0};
    double spot = 0.0;  // Invalid!
    std::vector<double> maturities = {0.1, 0.5};
    std::vector<double> vols = {0.20};
    std::vector<double> rates = {0.03};

    auto estimate = estimate_grid_from_grid_bounds(
        strikes, spot, maturities, vols, rates);

    EXPECT_TRUE(estimate.grids[0].empty());
}

TEST(PriceTableGridEstimatorTest, FromChainBounds_NegativeSpot_ReturnsEmpty) {
    std::vector<double> strikes = {90.0, 100.0, 110.0};
    double spot = -100.0;  // Invalid!
    std::vector<double> maturities = {0.1, 0.5};
    std::vector<double> vols = {0.20};
    std::vector<double> rates = {0.03};

    auto estimate = estimate_grid_from_grid_bounds(
        strikes, spot, maturities, vols, rates);

    EXPECT_TRUE(estimate.grids[0].empty());
}

// ===========================================================================
// Named accessor tests for PriceTableGridEstimate<4>
// ===========================================================================

TEST(PriceTableGridEstimatorTest, NamedAccessors) {
    auto estimate = estimate_grid_for_price_table(
        0.8, 1.2, 0.1, 2.0, 0.10, 0.50, 0.01, 0.06);

    // Named accessors should return same vectors as indexed access
    EXPECT_EQ(estimate.moneyness_grid().size(), estimate.grids[0].size());
    EXPECT_EQ(estimate.maturity_grid().size(), estimate.grids[1].size());
    EXPECT_EQ(estimate.volatility_grid().size(), estimate.grids[2].size());
    EXPECT_EQ(estimate.rate_grid().size(), estimate.grids[3].size());

    // Verify data is the same
    EXPECT_EQ(estimate.moneyness_grid(), estimate.grids[0]);
    EXPECT_EQ(estimate.maturity_grid(), estimate.grids[1]);
}

// ===========================================================================
// Grid generation helper tests (detail namespace)
// ===========================================================================

TEST(PriceTableGridEstimatorTest, UniformGridGeneratesCorrectBounds) {
    auto grid = detail::uniform_grid(1.0, 5.0, 5);

    EXPECT_EQ(grid.size(), 5);
    EXPECT_DOUBLE_EQ(grid.front(), 1.0);
    EXPECT_DOUBLE_EQ(grid.back(), 5.0);

    // Check uniform spacing
    double expected_spacing = (5.0 - 1.0) / 4.0;  // 1.0
    for (size_t i = 1; i < grid.size(); ++i) {
        EXPECT_NEAR(grid[i] - grid[i-1], expected_spacing, 1e-10);
    }
}

TEST(PriceTableGridEstimatorTest, LogUniformGridGeneratesCorrectBounds) {
    auto grid = detail::log_uniform_grid(1.0, 10.0, 5);

    EXPECT_EQ(grid.size(), 5);
    EXPECT_NEAR(grid.front(), 1.0, 1e-10);
    EXPECT_NEAR(grid.back(), 10.0, 1e-10);

    // Check log-uniform spacing
    double log_min = std::log(1.0);
    double log_max = std::log(10.0);
    double expected_log_spacing = (log_max - log_min) / 4.0;
    for (size_t i = 1; i < grid.size(); ++i) {
        double log_spacing = std::log(grid[i]) - std::log(grid[i-1]);
        EXPECT_NEAR(log_spacing, expected_log_spacing, 1e-10);
    }
}

TEST(PriceTableGridEstimatorTest, SqrtUniformGridGeneratesCorrectBounds) {
    auto grid = detail::sqrt_uniform_grid(0.25, 4.0, 5);

    EXPECT_EQ(grid.size(), 5);
    EXPECT_NEAR(grid.front(), 0.25, 1e-10);
    EXPECT_NEAR(grid.back(), 4.0, 1e-10);

    // Check sqrt-uniform spacing
    double sqrt_min = std::sqrt(0.25);  // 0.5
    double sqrt_max = std::sqrt(4.0);   // 2.0
    double expected_sqrt_spacing = (sqrt_max - sqrt_min) / 4.0;  // 0.375
    for (size_t i = 1; i < grid.size(); ++i) {
        double sqrt_spacing = std::sqrt(grid[i]) - std::sqrt(grid[i-1]);
        EXPECT_NEAR(sqrt_spacing, expected_sqrt_spacing, 1e-10);
    }
}

}  // namespace
}  // namespace mango
