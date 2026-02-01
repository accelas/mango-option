// SPDX-License-Identifier: MIT
/**
 * @file iv_solver_property_test.cc
 * @brief Property-based tests for implied volatility solvers
 */

#include <gtest/gtest.h>
#include "src/option/iv_solver_fdm.hpp"
#include <cmath>
#include <vector>

using namespace mango;

// Simple test that IV solver can be constructed and used
TEST(IVSolverPropertyTest, BasicConstruction) {
    IVSolverFDMConfig config;
    config.root_config.max_iter = 100;
    config.root_config.tolerance = 1e-6;

    IVSolverFDM solver(config);

    // Just verify construction succeeded
    SUCCEED();
}

// Test IV is always positive when found
TEST(IVSolverPropertyTest, IVAlwaysPositive) {
    IVSolverFDMConfig config;
    config.root_config.max_iter = 100;
    config.root_config.tolerance = 1e-6;

    std::vector<std::tuple<double, double, double, double>> test_cases = {
        // spot, strike, maturity, price
        {100.0, 100.0, 1.0, 10.0},
        {100.0, 100.0, 0.5, 5.0},
        {100.0, 100.0, 2.0, 15.0},
    };

    IVSolverFDM solver(config);

    for (const auto& [spot, strike, maturity, price] : test_cases) {
        IVQuery query(spot, strike, maturity, 0.05, 0.02, OptionType::PUT, price);
        auto result = solver.solve(query);

        if (result.has_value()) {
            EXPECT_GT(result->implied_vol, 0.0)
                << "IV must be positive for spot=" << spot
                << ", strike=" << strike << ", price=" << price;
            EXPECT_FALSE(std::isnan(result->implied_vol))
                << "IV must not be NaN";
            EXPECT_FALSE(std::isinf(result->implied_vol))
                << "IV must not be Inf";
        }
    }
}

// Test IV is within bounds
TEST(IVSolverPropertyTest, IVWithinBounds) {
    IVSolverFDMConfig config;
    config.root_config.max_iter = 100;
    config.root_config.tolerance = 1e-6;

    constexpr double vol_lower = 0.01;
    constexpr double vol_upper = 3.0;

    IVSolverFDM solver(config);

    IVQuery query(100.0, 100.0, 1.0, 0.05, 0.0, OptionType::PUT, 10.0);
    auto result = solver.solve(query);

    if (result.has_value()) {
        EXPECT_GE(result->implied_vol, vol_lower) << "IV below lower bound";
        EXPECT_LE(result->implied_vol, vol_upper) << "IV above upper bound";
    }
}

// Test no NaN/Inf results
TEST(IVSolverPropertyTest, NeverProducesNaNOrInf) {
    IVSolverFDMConfig config;
    config.root_config.max_iter = 100;
    config.root_config.tolerance = 1e-6;

    std::vector<IVQuery> queries = {
        IVQuery(100.0, 100.0, 1.0, 0.05, 0.02, OptionType::PUT, 10.0),
        IVQuery(100.0, 100.0, 1.0, 0.05, 0.02, OptionType::CALL, 10.0),
        IVQuery(100.0, 80.0, 1.0, 0.05, 0.02, OptionType::PUT, 5.0),
    };

    IVSolverFDM solver(config);

    for (const auto& query : queries) {
        auto result = solver.solve(query);

        if (result.has_value()) {
            EXPECT_FALSE(std::isnan(result->implied_vol))
                << "IV is NaN for query with market_price=" << query.market_price;
            EXPECT_FALSE(std::isinf(result->implied_vol))
                << "IV is Inf for query with market_price=" << query.market_price;
        }
    }
}
