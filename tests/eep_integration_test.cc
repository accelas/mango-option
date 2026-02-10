// SPDX-License-Identifier: MIT
/// @file eep_integration_test.cc
/// @brief End-to-end integration tests for EEP decomposition feature

#include "mango/option/table/bspline/bspline_builder.hpp"
#include "mango/option/table/standard_surface.hpp"
#include "mango/option/table/eep/eep_decomposer.hpp"
#include "mango/option/table/price_table_metadata.hpp"
#include "mango/option/american_option.hpp"
#include "mango/pde/core/pde_workspace.hpp"
#include <gtest/gtest.h>
#include <memory_resource>
#include <cmath>

namespace mango {
namespace {

// ===========================================================================
// End-to-end integration tests for EEP decomposition
// ===========================================================================

/// Build a price table and verify that the reconstructed
/// American price from BSplinePriceTable matches a direct PDE solve.
TEST(EEPIntegrationTest, ReconstructedPriceMatchesPDE) {
    // Grid covering a modest range for the price table
    // Each axis needs >= 4 points for B-spline fitting
    std::vector<double> log_moneyness = {std::log(0.90), std::log(0.95), std::log(1.00), std::log(1.05), std::log(1.10)};
    std::vector<double> maturity  = {0.25, 0.50, 0.75, 1.00};
    std::vector<double> vol       = {0.15, 0.20, 0.25, 0.30};
    std::vector<double> rate      = {0.02, 0.03, 0.04, 0.05};

    double K_ref = 100.0;

    // Build with auto-estimated PDE grid, then apply EEP decomposition
    auto setup = PriceTableBuilder::from_vectors(
        log_moneyness, maturity, vol, rate, K_ref,
        GridAccuracyParams{},   // auto-estimate PDE grid
        OptionType::PUT,
        0.0,   // dividend_yield
        0.0);  // max_failure_rate

    ASSERT_TRUE(setup.has_value())
        << "from_vectors failed: code=" << static_cast<int>(setup.error().code);

    auto& [builder, axes] = *setup;
    EEPDecomposer decomposer{OptionType::PUT, K_ref, 0.0};
    auto result = builder.build(axes, SurfaceContent::EarlyExercisePremium,
        [&](PriceTensor& tensor, const PriceTableAxes& a) {
            decomposer.decompose(tensor, a);
        });
    ASSERT_TRUE(result.has_value())
        << "build failed: code=" << static_cast<int>(result.error().code);
    ASSERT_NE(result->surface, nullptr);

    // Verify surface metadata marks EEP content
    EXPECT_EQ(result->surface->metadata().content,
              SurfaceContent::EarlyExercisePremium);

    // Wrap in BSplinePriceTable for reconstruction
    auto wrapper_result = make_standard_surface(result->surface, OptionType::PUT);
    ASSERT_TRUE(wrapper_result.has_value())
        << "make_standard_surface failed: " << wrapper_result.error();
    auto wrapper = std::move(*wrapper_result);

    // Test point: ATM put, 1-year, 20% vol, 5% rate
    double S     = 100.0;
    double K     = 100.0;
    double tau   = 1.0;
    double sigma = 0.20;
    double r     = 0.05;

    double reconstructed = wrapper.price(S, K, tau, sigma, r);
    EXPECT_GT(reconstructed, 0.0) << "Reconstructed price should be positive";

    // Direct PDE solve for comparison
    PricingParams params(OptionSpec{.spot = S, .strike = K, .maturity = tau, .rate = r, .option_type = OptionType::PUT}, sigma);
    auto [grid_spec, time_domain] = estimate_pde_grid(params);

    size_t n_space = grid_spec.n_points();
    size_t ws_size = PDEWorkspace::required_size(n_space);
    std::pmr::synchronized_pool_resource pool;
    std::pmr::vector<double> buffer(ws_size, &pool);

    auto ws = PDEWorkspace::from_buffer(buffer, n_space);
    ASSERT_TRUE(ws.has_value()) << ws.error();

    auto solver = AmericanOptionSolver::create(params, std::move(*ws)).value();
    auto pde_result = solver.solve();
    ASSERT_TRUE(pde_result.has_value())
        << "PDE solve failed: " << static_cast<int>(pde_result.error().code);

    double pde_price = pde_result->value();
    EXPECT_GT(pde_price, 0.0);

    // Compare: reconstructed should be within ~1% of PDE price
    double tol = 0.01 * pde_price;
    EXPECT_NEAR(reconstructed, pde_price, tol)
        << "Reconstructed=" << reconstructed
        << " PDE=" << pde_price
        << " tolerance=" << tol;
}

/// Build a price table and verify that the raw EEP
/// surface produces non-negative values at all grid points.
/// The softplus floor in extract_tensor should guarantee this.
TEST(EEPIntegrationTest, SoftplusFloorEnsuresNonNegative) {
    // Small grid — each axis needs >= 4 points for B-spline fitting
    std::vector<double> log_moneyness = {std::log(0.90), std::log(0.95), std::log(1.00), std::log(1.10)};
    std::vector<double> maturity  = {0.25, 0.50, 0.75, 1.00};
    std::vector<double> vol       = {0.15, 0.20, 0.25, 0.30};
    std::vector<double> rate      = {0.02, 0.03, 0.04, 0.05};

    double K_ref = 100.0;

    auto setup = PriceTableBuilder::from_vectors(
        log_moneyness, maturity, vol, rate, K_ref,
        GridAccuracyParams{},
        OptionType::PUT,
        0.0,   // dividend_yield
        0.0);  // max_failure_rate

    ASSERT_TRUE(setup.has_value())
        << "from_vectors failed: code=" << static_cast<int>(setup.error().code);

    auto& [builder, axes] = *setup;
    EEPDecomposer decomposer{OptionType::PUT, K_ref, 0.0};
    auto result = builder.build(axes, SurfaceContent::EarlyExercisePremium,
        [&](PriceTensor& tensor, const PriceTableAxes& a) {
            decomposer.decompose(tensor, a);
        });
    ASSERT_TRUE(result.has_value())
        << "build failed: code=" << static_cast<int>(result.error().code);
    ASSERT_NE(result->surface, nullptr);

    // Verify metadata
    EXPECT_EQ(result->surface->metadata().content,
              SurfaceContent::EarlyExercisePremium);

    // Query the raw EEP surface at every grid point combination.
    // The B-spline is fitted to softplus-floored data, so values at
    // grid points should be non-negative (or very close due to fitting error).
    const auto& surface = *result->surface;
    const auto& g = surface.axes().grids;

    size_t negative_count = 0;
    double most_negative = 0.0;

    for (double m : g[0]) {
        for (double tau : g[1]) {
            for (double sigma : g[2]) {
                for (double r : g[3]) {
                    double val = surface.value({m, tau, sigma, r});
                    if (val < -1e-10) {
                        ++negative_count;
                        most_negative = std::min(most_negative, val);
                    }
                }
            }
        }
    }

    EXPECT_EQ(negative_count, 0u)
        << "Found " << negative_count << " negative EEP values; "
        << "most negative = " << most_negative;
}

// ===========================================================================
// Regression tests for bugs found during code review
// ===========================================================================

// Regression: make_standard_surface must reject NormalizedPrice content
// Bug: Previously accepted NormalizedPrice but always used EEPPriceTableInner,
// which adds the European component at query time — double-counting it for
// surfaces that already contain full American prices.
TEST(EEPIntegrationTest, MakeStandardWrapperRejectsNormalizedPrice) {
    std::vector<double> log_moneyness = {std::log(0.90), std::log(0.95), std::log(1.00), std::log(1.10)};
    std::vector<double> maturity  = {0.25, 0.50, 0.75, 1.00};
    std::vector<double> vol       = {0.15, 0.20, 0.25, 0.30};
    std::vector<double> rate      = {0.02, 0.03, 0.04, 0.05};

    double K_ref = 100.0;

    auto setup = PriceTableBuilder::from_vectors(
        log_moneyness, maturity, vol, rate, K_ref,
        GridAccuracyParams{},
        OptionType::PUT,
        0.0,   // dividend_yield
        0.0);  // max_failure_rate
    ASSERT_TRUE(setup.has_value());

    auto& [builder, axes] = *setup;

    // Default build produces NormalizedPrice content
    auto result = builder.build(axes);
    ASSERT_TRUE(result.has_value());
    ASSERT_EQ(result->surface->metadata().content,
              SurfaceContent::NormalizedPrice);

    // make_standard_surface must reject this
    auto wrapper_result = make_standard_surface(result->surface, OptionType::PUT);
    EXPECT_FALSE(wrapper_result.has_value())
        << "make_standard_surface should reject NormalizedPrice surfaces";
    EXPECT_NE(wrapper_result.error().find("EEP"), std::string::npos)
        << "Error message should mention EEP; got: " << wrapper_result.error();
}

}  // namespace
}  // namespace mango
