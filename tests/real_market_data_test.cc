// SPDX-License-Identifier: MIT
/**
 * @file real_market_data_test.cc
 * @brief Tests using real SPY option chain data from yfinance
 *
 * Data source: benchmarks/real_market_data.hpp (auto-generated)
 * Regenerate with: python scripts/download_benchmark_data.py SPY
 */

#include <gtest/gtest.h>
#include "benchmarks/real_market_data.hpp"
#include "src/option/american_option.hpp"
#include "src/option/american_option_batch.hpp"
#include "src/option/iv_solver_fdm.hpp"

using namespace mango;
namespace bdata = mango::benchmark_data;

namespace {

// Helper to convert market data to solver params
AmericanOptionParams make_params(const bdata::RealOptionData& opt, double vol = 0.20) {
    return AmericanOptionParams(
        bdata::SPOT,               // spot
        opt.strike,                // strike
        opt.maturity,              // maturity
        bdata::RISK_FREE_RATE,     // rate
        bdata::DIVIDEND_YIELD,     // dividend_yield
        opt.is_call ? OptionType::CALL : OptionType::PUT,
        vol                        // volatility for grid estimation
    );
}

// Helper to create IVQuery from real data
IVQuery make_iv_query(const bdata::RealOptionData& opt) {
    return IVQuery(
        bdata::SPOT,               // spot
        opt.strike,                // strike
        opt.maturity,              // maturity
        bdata::RISK_FREE_RATE,     // rate
        bdata::DIVIDEND_YIELD,     // dividend_yield
        opt.is_call ? OptionType::CALL : OptionType::PUT,
        opt.market_price           // market_price
    );
}

}  // namespace

// ============================================================================
// Single option pricing tests
// ============================================================================

TEST(RealMarketDataTest, ATMPutPricing) {
    // Price the ATM put option from real market data
    auto params = make_params(bdata::ATM_PUT);

    auto [grid_spec, time_domain] = estimate_grid_for_option(params);
    std::pmr::synchronized_pool_resource pool;
    std::pmr::vector<double> buffer(PDEWorkspace::required_size(grid_spec.n_points()), &pool);
    auto workspace = PDEWorkspace::from_buffer(buffer, grid_spec.n_points());
    ASSERT_TRUE(workspace.has_value()) << workspace.error();

    AmericanOptionSolver solver(params, workspace.value());
    auto result = solver.solve();
    ASSERT_TRUE(result.has_value()) << "Solver failed: " << static_cast<int>(result.error().code);

    double price = result->value_at(params.spot);
    EXPECT_GT(price, 0.0) << "Price should be positive";
    // Model price should be in reasonable range of market price
    // (American options have early exercise premium)
    double market_mid = bdata::ATM_PUT.market_price;
    EXPECT_GT(price, market_mid * 0.5) << "Price too low vs market";
    EXPECT_LT(price, market_mid * 2.0) << "Price too high vs market";
}

TEST(RealMarketDataTest, PutPricingAcrossStrikes) {
    // Test pricing across multiple strikes from real data
    constexpr size_t test_count = 5;  // Test first 5 puts

    for (size_t i = 0; i < std::min(test_count, bdata::REAL_PUTS.size()); ++i) {
        const auto& opt = bdata::REAL_PUTS[i];
        auto params = make_params(opt);

        auto [grid_spec, time_domain] = estimate_grid_for_option(params);
        std::pmr::synchronized_pool_resource pool;
        std::pmr::vector<double> buffer(PDEWorkspace::required_size(grid_spec.n_points()), &pool);
        auto workspace = PDEWorkspace::from_buffer(buffer, grid_spec.n_points());
        ASSERT_TRUE(workspace.has_value()) << "Workspace creation failed for option " << i;

        AmericanOptionSolver solver(params, workspace.value());
        auto result = solver.solve();
        ASSERT_TRUE(result.has_value())
            << "Solver failed for K=" << opt.strike << ": code " << static_cast<int>(result.error().code);

        double price = result->value_at(params.spot);
        EXPECT_GT(price, 0.0) << "Price should be positive for K=" << opt.strike;
    }
}

// ============================================================================
// Batch pricing tests
// ============================================================================

TEST(RealMarketDataTest, BatchPutPricing) {
    // Batch price puts using parallel solver
    std::vector<AmericanOptionParams> batch;
    batch.reserve(bdata::REAL_PUTS.size());

    for (const auto& opt : bdata::REAL_PUTS) {
        batch.push_back(make_params(opt));
    }

    BatchAmericanOptionSolver solver;
    auto batch_result = solver.solve_batch(batch, false);  // per-option grids

    ASSERT_EQ(batch_result.results.size(), bdata::REAL_PUTS.size());

    for (size_t i = 0; i < batch_result.results.size(); ++i) {
        const auto& res = batch_result.results[i];
        ASSERT_TRUE(res.has_value())
            << "Batch solver failed for option " << i << ": code " << static_cast<int>(res.error().code);

        double price = res->value();
        EXPECT_GT(price, 0.0) << "Price should be positive for option " << i;
    }
}

TEST(RealMarketDataTest, BatchCallPricing) {
    // Batch price calls
    std::vector<AmericanOptionParams> batch;
    batch.reserve(bdata::REAL_CALLS.size());

    for (const auto& opt : bdata::REAL_CALLS) {
        batch.push_back(make_params(opt));
    }

    BatchAmericanOptionSolver solver;
    auto batch_result = solver.solve_batch(batch, false);

    ASSERT_EQ(batch_result.results.size(), bdata::REAL_CALLS.size());

    for (size_t i = 0; i < batch_result.results.size(); ++i) {
        const auto& res = batch_result.results[i];
        ASSERT_TRUE(res.has_value())
            << "Batch solver failed for call " << i << ": code " << static_cast<int>(res.error().code);

        double price = res->value();
        EXPECT_GT(price, 0.0) << "Price should be positive for call " << i;
    }
}

// ============================================================================
// IV calculation tests
// ============================================================================

TEST(RealMarketDataTest, IVCalculationFDM) {
    // Calculate IV for ATM put using FDM solver
    auto query = make_iv_query(bdata::ATM_PUT);

    IVSolverFDMConfig config;
    config.root_config.max_iter = 100;
    config.root_config.tolerance = 1e-4;

    IVSolverFDM solver(config);
    auto result = solver.solve_impl(query);

    ASSERT_TRUE(result.has_value())
        << "IV solver failed: error code " << static_cast<int>(result.error().code);

    double iv = result->implied_vol;
    // IV should be in reasonable range (5% to 100%)
    EXPECT_GT(iv, 0.05) << "IV too low";
    EXPECT_LT(iv, 1.0) << "IV too high";
}

TEST(RealMarketDataTest, IVSanityCheck) {
    // Verify IV calculation produces consistent prices
    auto query = make_iv_query(bdata::ATM_PUT);

    IVSolverFDMConfig config;
    IVSolverFDM iv_solver(config);
    auto iv_result = iv_solver.solve_impl(query);
    ASSERT_TRUE(iv_result.has_value());

    double iv = iv_result->implied_vol;

    // Now price with computed IV
    auto params = make_params(bdata::ATM_PUT, iv);

    auto [grid_spec, time_domain] = estimate_grid_for_option(params);
    std::pmr::synchronized_pool_resource pool;
    std::pmr::vector<double> buffer(PDEWorkspace::required_size(grid_spec.n_points()), &pool);
    auto workspace = PDEWorkspace::from_buffer(buffer, grid_spec.n_points());
    ASSERT_TRUE(workspace.has_value());

    AmericanOptionSolver price_solver(params, workspace.value());
    auto price_result = price_solver.solve();
    ASSERT_TRUE(price_result.has_value());

    double model_price = price_result->value_at(params.spot);

    // Model price should match market price (within tolerance)
    EXPECT_NEAR(model_price, bdata::ATM_PUT.market_price, bdata::ATM_PUT.market_price * 0.01)
        << "IV-derived price should match market price";
}
