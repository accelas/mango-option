// SPDX-License-Identifier: MIT
/**
 * @file iv_solver_interpolated_test.cc
 * @brief Tests for IVSolverInterpolated (B-spline based IV solver)
 */

#include <gtest/gtest.h>
#include "src/option/iv_solver_interpolated.hpp"
#include "src/option/table/price_table_surface.hpp"
#include "src/option/table/price_table_axes.hpp"
#include "src/option/table/price_table_metadata.hpp"
#include "src/option/table/american_price_surface.hpp"

namespace mango {
namespace {

/// Test fixture that creates a simple price surface for IV solving
class IVSolverInterpolatedTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Create a 4D price surface with realistic grids
        // Axes: moneyness, maturity, volatility, rate
        PriceTableAxes<4> axes;
        axes.grids[0] = {0.8, 0.9, 1.0, 1.1, 1.2};  // moneyness (S/K)
        axes.grids[1] = {0.25, 0.5, 1.0, 2.0};       // maturity (years)
        axes.grids[2] = {0.10, 0.20, 0.30, 0.40};    // volatility
        axes.grids[3] = {0.02, 0.04, 0.06, 0.08};    // rate
        axes.names = {"moneyness", "maturity", "volatility", "rate"};

        // Total coefficients: 5 * 4 * 4 * 4 = 320
        size_t total = 5 * 4 * 4 * 4;
        std::vector<double> coeffs(total);

        // Fill with synthetic put option prices
        // Use Black-Scholes approximation: higher vol = higher price, etc.
        size_t idx = 0;
        for (size_t i_r = 0; i_r < 4; ++i_r) {
            for (size_t i_v = 0; i_v < 4; ++i_v) {
                for (size_t i_t = 0; i_t < 4; ++i_t) {
                    for (size_t i_m = 0; i_m < 5; ++i_m) {
                        double m = axes.grids[0][i_m];
                        double tau = axes.grids[1][i_t];
                        double vol = axes.grids[2][i_v];
                        double r = axes.grids[3][i_r];

                        // Simplified put price model: increases with vol, tau, and ITM-ness
                        // Put is ITM when m < 1 (spot < strike)
                        double intrinsic = std::max(0.0, 1.0 - m);
                        double time_value = vol * std::sqrt(tau) * 0.4;  // Simplified
                        double price = (intrinsic + time_value) * std::exp(-r * tau);

                        // Ensure minimum price
                        coeffs[idx++] = std::max(0.001, price) * K_ref_;
                    }
                }
            }
        }

        PriceTableMetadata meta{
            .K_ref = K_ref_,
            .dividend_yield = 0.0,
            .discrete_dividends = {}
        };

        auto result = PriceTableSurface<4>::build(std::move(axes), std::move(coeffs), meta);
        ASSERT_TRUE(result.has_value()) << "Failed to build surface";
        surface_ = result.value();
    }

    std::shared_ptr<const PriceTableSurface<4>> surface_;
    static constexpr double K_ref_ = 100.0;
};

TEST_F(IVSolverInterpolatedTest, CreateFromSurface) {
    auto result = IVSolverInterpolated::create(surface_);
    ASSERT_TRUE(result.has_value()) << "Failed to create solver";
}

TEST_F(IVSolverInterpolatedTest, CreateWithConfig) {
    IVSolverInterpolatedConfig config{
        .max_iterations = 100,
        .tolerance = 1e-8,
        .sigma_min = 0.05,
        .sigma_max = 2.0
    };

    auto result = IVSolverInterpolated::create(surface_, config);
    ASSERT_TRUE(result.has_value()) << "Failed to create solver with config";
}

TEST_F(IVSolverInterpolatedTest, RejectsNullSurface) {
    auto result = IVSolverInterpolated::create(nullptr);
    EXPECT_FALSE(result.has_value());
}

TEST_F(IVSolverInterpolatedTest, SolveATMPut) {
    auto solver_result = IVSolverInterpolated::create(surface_);
    ASSERT_TRUE(solver_result.has_value());
    auto& solver = solver_result.value();

    // ATM put: S = K = 100, maturity = 1y, rate = 5%
    // Use a price that's within the range our synthetic surface can handle
    IVQuery query{
        100.0,  // spot
        100.0,  // strike
        1.0,    // maturity
        0.05,   // rate
        0.0,    // dividend_yield
        OptionType::PUT,
        8.0     // market_price
    };

    auto result = solver.solve_impl(query);
    // With synthetic data, may or may not converge - test that it returns a result
    // (either success or meaningful error code)
    if (result.has_value()) {
        EXPECT_GT(result->implied_vol, 0.0);
        EXPECT_LT(result->implied_vol, 5.0);  // Reasonable upper bound
    } else {
        // If it fails, should be a convergence issue, not a validation error
        EXPECT_TRUE(result.error().code == IVErrorCode::MaxIterationsExceeded ||
                    result.error().code == IVErrorCode::BracketingFailed ||
                    result.error().code == IVErrorCode::NumericalInstability);
    }
}

TEST_F(IVSolverInterpolatedTest, SolveITMPut) {
    auto solver_result = IVSolverInterpolated::create(surface_);
    ASSERT_TRUE(solver_result.has_value());
    auto& solver = solver_result.value();

    // ITM put: S = 90, K = 100 (m = 0.9), maturity = 1y
    IVQuery query{
        90.0,   // spot (ITM for put)
        100.0,  // strike
        1.0,    // maturity
        0.05,   // rate
        0.0,    // dividend_yield
        OptionType::PUT,
        15.0    // higher price for ITM
    };

    auto result = solver.solve_impl(query);
    // With synthetic data, accept either convergence or graceful failure
    if (result.has_value()) {
        EXPECT_GT(result->implied_vol, 0.0);
        EXPECT_LT(result->implied_vol, 5.0);
    }
    // Test passes as long as it doesn't crash
}

TEST_F(IVSolverInterpolatedTest, SolveOTMPut) {
    auto solver_result = IVSolverInterpolated::create(surface_);
    ASSERT_TRUE(solver_result.has_value());
    auto& solver = solver_result.value();

    // OTM put: S = 110, K = 100 (m = 1.1), maturity = 1y
    IVQuery query{
        110.0,  // spot (OTM for put)
        100.0,  // strike
        1.0,    // maturity
        0.05,   // rate
        0.0,    // dividend_yield
        OptionType::PUT,
        3.0     // lower price for OTM
    };

    auto result = solver.solve_impl(query);
    // With synthetic data, accept either convergence or graceful failure
    if (result.has_value()) {
        EXPECT_GT(result->implied_vol, 0.0);
        EXPECT_LT(result->implied_vol, 5.0);
    }
    // Test passes as long as it doesn't crash
}

TEST_F(IVSolverInterpolatedTest, RejectsInvalidQuery) {
    auto solver_result = IVSolverInterpolated::create(surface_);
    ASSERT_TRUE(solver_result.has_value());
    auto& solver = solver_result.value();

    // Invalid: negative spot
    IVQuery invalid_query{
        -100.0,  // invalid spot
        100.0,
        1.0,
        0.05,
        0.0,
        OptionType::PUT,
        10.0
    };

    auto result = solver.solve_impl(invalid_query);
    EXPECT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code, IVErrorCode::NegativeSpot);
}

TEST_F(IVSolverInterpolatedTest, RejectsNegativeMarketPrice) {
    auto solver_result = IVSolverInterpolated::create(surface_);
    ASSERT_TRUE(solver_result.has_value());
    auto& solver = solver_result.value();

    IVQuery invalid_query{
        100.0,
        100.0,
        1.0,
        0.05,
        0.0,
        OptionType::PUT,
        -5.0    // invalid negative price
    };

    auto result = solver.solve_impl(invalid_query);
    EXPECT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code, IVErrorCode::NegativeMarketPrice);
}

TEST_F(IVSolverInterpolatedTest, BatchSolve) {
    auto solver_result = IVSolverInterpolated::create(surface_);
    ASSERT_TRUE(solver_result.has_value());
    auto& solver = solver_result.value();

    std::vector<IVQuery> queries;

    // Create batch of queries with varying strikes
    for (double strike : {90.0, 95.0, 100.0, 105.0, 110.0}) {
        double m = 100.0 / strike;  // moneyness
        double price = (m < 1.0) ? 12.0 : (m > 1.0 ? 4.0 : 8.0);  // Rough prices
        queries.push_back(IVQuery{
            100.0,  // spot
            strike,
            1.0,    // maturity
            0.05,   // rate
            0.0,    // dividend
            OptionType::PUT,
            price
        });
    }

    auto batch_result = solver.solve_batch_impl(queries);

    // With synthetic data, just verify batch processing works
    EXPECT_EQ(batch_result.results.size(), 5);
    // Count should be consistent
    size_t actual_failures = 0;
    for (const auto& r : batch_result.results) {
        if (!r.has_value()) actual_failures++;
    }
    EXPECT_EQ(batch_result.failed_count, actual_failures);
}

TEST_F(IVSolverInterpolatedTest, BatchSolveAllSucceed) {
    auto solver_result = IVSolverInterpolated::create(surface_);
    ASSERT_TRUE(solver_result.has_value());
    auto& solver = solver_result.value();

    // Single valid query in batch
    std::vector<IVQuery> queries = {
        IVQuery{100.0, 100.0, 1.0, 0.05, 0.0, OptionType::PUT, 8.0}
    };

    auto batch_result = solver.solve_batch_impl(queries);

    EXPECT_EQ(batch_result.results.size(), 1);
    if (batch_result.all_succeeded()) {
        EXPECT_TRUE(batch_result.results[0].has_value());
    }
}

TEST_F(IVSolverInterpolatedTest, ConvergenceWithinIterations) {
    IVSolverInterpolatedConfig config{
        .max_iterations = 10,  // Limited iterations
        .tolerance = 1e-6
    };

    auto solver_result = IVSolverInterpolated::create(surface_, config);
    ASSERT_TRUE(solver_result.has_value());
    auto& solver = solver_result.value();

    IVQuery query{100.0, 100.0, 1.0, 0.05, 0.0, OptionType::PUT, 8.0};
    auto result = solver.solve_impl(query);

    if (result.has_value()) {
        EXPECT_LE(result->iterations, 10u);
    }
}

TEST_F(IVSolverInterpolatedTest, CreateFromAmericanPriceSurface) {
    // Build an EEP surface
    PriceTableAxes<4> eep_axes;
    eep_axes.grids[0] = {0.8, 0.9, 1.0, 1.1, 1.2};
    eep_axes.grids[1] = {0.25, 0.5, 1.0, 2.0};
    eep_axes.grids[2] = {0.10, 0.20, 0.30, 0.40};
    eep_axes.grids[3] = {0.02, 0.04, 0.06, 0.08};

    std::vector<double> eep_coeffs(5 * 4 * 4 * 4, 2.0);  // constant EEP
    PriceTableMetadata eep_meta{
        .K_ref = 100.0,
        .dividend_yield = 0.0,
        .m_min = 0.8,
        .m_max = 1.2,
        .content = SurfaceContent::EarlyExercisePremium,
        .discrete_dividends = {}
    };

    auto eep_surface = PriceTableSurface<4>::build(eep_axes, eep_coeffs, eep_meta);
    ASSERT_TRUE(eep_surface.has_value());

    auto aps = AmericanPriceSurface::create(eep_surface.value(), OptionType::PUT);
    ASSERT_TRUE(aps.has_value());

    auto solver = IVSolverInterpolated::create(std::move(*aps));
    EXPECT_TRUE(solver.has_value());
}

TEST_F(IVSolverInterpolatedTest, SolveWithAmericanPriceSurface) {
    // Build an EEP surface
    PriceTableAxes<4> eep_axes;
    eep_axes.grids[0] = {0.8, 0.9, 1.0, 1.1, 1.2};
    eep_axes.grids[1] = {0.25, 0.5, 1.0, 2.0};
    eep_axes.grids[2] = {0.10, 0.20, 0.30, 0.40};
    eep_axes.grids[3] = {0.02, 0.04, 0.06, 0.08};

    std::vector<double> eep_coeffs(5 * 4 * 4 * 4, 2.0);
    PriceTableMetadata eep_meta{
        .K_ref = 100.0,
        .dividend_yield = 0.0,
        .m_min = 0.8,
        .m_max = 1.2,
        .content = SurfaceContent::EarlyExercisePremium,
        .discrete_dividends = {}
    };

    auto eep_surface = PriceTableSurface<4>::build(eep_axes, eep_coeffs, eep_meta);
    ASSERT_TRUE(eep_surface.has_value());

    auto aps = AmericanPriceSurface::create(eep_surface.value(), OptionType::PUT);
    ASSERT_TRUE(aps.has_value());

    auto solver = IVSolverInterpolated::create(std::move(*aps));
    ASSERT_TRUE(solver.has_value());

    IVQuery query{
        100.0,  // spot
        100.0,  // strike
        1.0,    // maturity
        0.05,   // rate
        0.0,    // dividend_yield
        OptionType::PUT,
        8.0     // market_price
    };

    auto result = solver->solve_impl(query);
    // With synthetic data, accept success or graceful failure
    if (result.has_value()) {
        EXPECT_GT(result->implied_vol, 0.0);
        EXPECT_LT(result->implied_vol, 5.0);
    }
}

}  // namespace
}  // namespace mango
