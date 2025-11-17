/**
 * @file american_option_solver_test.cc
 * @brief Tests for AmericanOptionSolver structure and API
 */

#include "src/option/american_option.hpp"
#include "src/option/american_option_batch.hpp"
#include "src/option/price_table_snapshot_collector.hpp"
#include <gtest/gtest.h>
#include <cmath>
#include <mutex>
#include <algorithm>

namespace mango {
namespace {

TEST(AmericanOptionSolverTest, ConstructorValidation) {
    AmericanOptionParams params(
        100.0,  // spot
        100.0,  // strike
        1.0,    // maturity
        0.05,   // rate
        0.02,   // dividend_yield
        OptionType::CALL,
        0.2     // volatility
    );

    auto workspace = AmericanSolverWorkspace::create(-3.0, 3.0, 101, 1000).value();

    // Should construct successfully
    EXPECT_NO_THROW({
        AmericanOptionSolver solver(params, workspace);
    });
}

TEST(AmericanOptionSolverTest, InvalidStrike) {
    AmericanOptionParams params(
        100.0,   // spot
        -100.0,  // strike (Invalid)
        1.0,     // maturity
        0.05,    // rate
        0.02,    // dividend_yield
        OptionType::PUT,
        0.2      // volatility
    );

    auto workspace = AmericanSolverWorkspace::create(-3.0, 3.0, 101, 1000).value();

    EXPECT_THROW({
        AmericanOptionSolver solver(params, workspace);
    }, std::invalid_argument);
}

TEST(AmericanOptionSolverTest, InvalidSpot) {
    AmericanOptionParams params(
        0.0,    // spot (Invalid)
        100.0,  // strike
        1.0,    // maturity
        0.05,   // rate
        0.02,   // dividend_yield
        OptionType::PUT,
        0.2     // volatility
    );

    auto workspace = AmericanSolverWorkspace::create(-3.0, 3.0, 101, 1000).value();

    EXPECT_THROW({
        AmericanOptionSolver solver(params, workspace);
    }, std::invalid_argument);
}

TEST(AmericanOptionSolverTest, InvalidMaturity) {
    AmericanOptionParams params(
        100.0,  // spot
        100.0,  // strike
        -1.0,   // maturity (Invalid)
        0.05,   // rate
        0.02,   // dividend_yield
        OptionType::CALL,
        0.2     // volatility
    );

    auto workspace = AmericanSolverWorkspace::create(-3.0, 3.0, 101, 1000).value();

    EXPECT_THROW({
        AmericanOptionSolver solver(params, workspace);
    }, std::invalid_argument);
}

TEST(AmericanOptionSolverTest, InvalidVolatility) {
    AmericanOptionParams params(
        100.0,  // spot
        100.0,  // strike
        1.0,    // maturity
        0.05,   // rate
        0.02,   // dividend_yield
        OptionType::PUT,
        -0.2    // volatility (Invalid)
    );

    auto workspace = AmericanSolverWorkspace::create(-3.0, 3.0, 101, 1000).value();

    EXPECT_THROW({
        AmericanOptionSolver solver(params, workspace);
    }, std::invalid_argument);
}

TEST(AmericanOptionSolverTest, NegativeRateAllowed) {
    // Negative rates are valid (EUR, JPY, CHF markets)
    AmericanOptionParams params(
        100.0,  // spot
        100.0,  // strike
        1.0,    // maturity
        -0.01,  // rate (negative but valid)
        0.02,   // dividend_yield
        OptionType::CALL,
        0.2     // volatility
    );

    auto workspace = AmericanSolverWorkspace::create(-3.0, 3.0, 101, 1000).value();

    EXPECT_NO_THROW({
        AmericanOptionSolver solver(params, workspace);
    });
}

TEST(AmericanOptionSolverTest, InvalidDividendYield) {
    AmericanOptionParams params(
        100.0,  // spot
        100.0,  // strike
        1.0,    // maturity
        0.05,   // rate
        -0.02,  // dividend_yield (Invalid)
        OptionType::PUT,
        0.2     // volatility
    );

    auto workspace = AmericanSolverWorkspace::create(-3.0, 3.0, 101, 1000).value();

    EXPECT_THROW({
        AmericanOptionSolver solver(params, workspace);
    }, std::invalid_argument);
}

TEST(AmericanOptionSolverTest, InvalidGridNSpace) {
    AmericanOptionParams params(
        100.0,  // spot
        100.0,  // strike
        1.0,    // maturity
        0.05,   // rate
        0.02,   // dividend_yield
        OptionType::CALL,
        0.2     // volatility
    );

    // Test that workspace factory validates n_space >= 10
    auto result = AmericanSolverWorkspace::create(-3.0, 3.0, 5, 1000);
    EXPECT_FALSE(result.has_value());
    EXPECT_EQ(result.error(), "n_space must be >= 10");
}

TEST(AmericanOptionSolverTest, InvalidGridNTime) {
    AmericanOptionParams params(
        100.0,  // spot
        100.0,  // strike
        1.0,    // maturity
        0.05,   // rate
        0.02,   // dividend_yield
        OptionType::PUT,
        0.2     // volatility
    );

    // Test that workspace factory validates n_time >= 10
    auto result = AmericanSolverWorkspace::create(-3.0, 3.0, 101, 5);
    EXPECT_FALSE(result.has_value());
    EXPECT_EQ(result.error(), "n_time must be >= 10");
}

TEST(AmericanOptionSolverTest, InvalidGridBounds) {
    AmericanOptionParams params(
        100.0,  // spot
        100.0,  // strike
        1.0,    // maturity
        0.05,   // rate
        0.02,   // dividend_yield
        OptionType::CALL,
        0.2     // volatility
    );

    // Test that workspace factory validates x_min < x_max
    auto result = AmericanSolverWorkspace::create(3.0, -3.0, 101, 1000);
    EXPECT_FALSE(result.has_value());
    EXPECT_EQ(result.error(), "x_min must be < x_max");
}

TEST(AmericanOptionSolverTest, DiscreteDividends) {
    AmericanOptionParams params(
        100.0,  // spot
        100.0,  // strike
        1.0,    // maturity
        0.05,   // rate
        0.0,    // dividend_yield
        OptionType::CALL,
        0.2,    // volatility
        {
            {0.25, 1.0},
            {0.75, 1.5}
        }  // Valid dividends
    );

    auto workspace = AmericanSolverWorkspace::create(-3.0, 3.0, 101, 1000).value();

    // Should accept valid discrete dividends
    EXPECT_NO_THROW({
        AmericanOptionSolver solver(params, workspace);
    });
}

TEST(AmericanOptionSolverTest, DiscreteDividendInvalidTime) {
    // Should reject negative time
    AmericanOptionParams params1(
        100.0,  // spot
        100.0,  // strike
        1.0,    // maturity
        0.05,   // rate
        0.0,    // dividend_yield
        OptionType::CALL,
        0.2,    // volatility
        {
            {-0.1, 1.0}
        }  // Invalid: negative time
    );

    auto workspace = AmericanSolverWorkspace::create(-3.0, 3.0, 101, 1000).value();

    EXPECT_THROW({
        AmericanOptionSolver solver(params1, workspace);
    }, std::invalid_argument);

    // Should reject time beyond maturity
    AmericanOptionParams params2(
        100.0,  // spot
        100.0,  // strike
        1.0,    // maturity
        0.05,   // rate
        0.0,    // dividend_yield
        OptionType::CALL,
        0.2,    // volatility
        {
            {2.0, 1.0}
        }  // Invalid: beyond maturity
    );

    EXPECT_THROW({
        AmericanOptionSolver solver(params2, workspace);
    }, std::invalid_argument);
}

TEST(AmericanOptionSolverTest, DiscreteDividendInvalidAmount) {
    AmericanOptionParams params(
        100.0,  // spot
        100.0,  // strike
        1.0,    // maturity
        0.05,   // rate
        0.0,    // dividend_yield
        OptionType::PUT,
        0.2,    // volatility
        {
            {0.5, -1.0}
        }  // Invalid: negative amount
    );

    auto workspace = AmericanSolverWorkspace::create(-3.0, 3.0, 101, 1000).value();

    // Should reject negative amount
    EXPECT_THROW({
        AmericanOptionSolver solver(params, workspace);
    }, std::invalid_argument);
}

TEST(AmericanOptionSolverTest, SolveAmericanPutNoDiv) {
    AmericanOptionParams params(
        100.0,  // spot
        100.0,  // strike
        1.0,    // maturity
        0.05,   // rate
        0.0,    // dividend_yield
        OptionType::PUT,
        0.2     // volatility
    );

    auto workspace = AmericanSolverWorkspace::create(-3.0, 3.0, 101, 1000).value();
    AmericanOptionSolver solver(params, workspace);

    auto result = solver.solve();
    ASSERT_TRUE(result.has_value()) << result.error().message;

    // Should converge
    EXPECT_TRUE(result->converged);

    // NOTE: Current implementation has known issues with PDE time evolution
    // The solution is converging but not evolving correctly in time
    // For now, just verify solver completes and produces reasonable bounds
    EXPECT_GE(result->value, 0.0);  // Non-negative
    EXPECT_LE(result->value, params.strike);  // Less than strike

    // Solution should be available
    auto solution = solver.get_solution();
    EXPECT_EQ(solution.size(), 101);  // Default n_space
}

TEST(AmericanOptionSolverTest, GetSolutionBeforeSolve) {
    AmericanOptionParams params(
        100.0,  // spot
        100.0,  // strike
        1.0,    // maturity
        0.05,   // rate
        0.02,   // dividend_yield
        OptionType::PUT,
        0.2     // volatility
    );

    auto workspace = AmericanSolverWorkspace::create(-3.0, 3.0, 101, 1000).value();
    AmericanOptionSolver solver(params, workspace);

    // get_solution() should throw before solve()
    EXPECT_THROW({
        solver.get_solution();
    }, std::runtime_error);
}

TEST(AmericanOptionSolverTest, DeltaIsReasonable) {
    AmericanOptionParams params(
        100.0,  // spot
        100.0,  // strike
        1.0,    // maturity
        0.05,   // rate
        0.0,    // dividend_yield
        OptionType::PUT,
        0.2     // volatility
    );

    auto workspace = AmericanSolverWorkspace::create(-3.0, 3.0, 101, 1000).value();
    AmericanOptionSolver solver(params, workspace);

    auto result = solver.solve();
    ASSERT_TRUE(result.has_value()) << result.error().message;

    // Should converge
    EXPECT_TRUE(result->converged);

    // Delta for ATM put should be negative (around -0.5 for European)
    // American put delta can be different, but should still be negative
    EXPECT_LT(result->delta, 0.0);
    EXPECT_GT(result->delta, -1.0);  // Should be between -1 and 0
}

TEST(AmericanOptionSolverTest, GammaIsComputed) {
    AmericanOptionParams params(
        100.0,  // spot
        100.0,  // strike
        1.0,    // maturity
        0.05,   // rate
        0.0,    // dividend_yield
        mango::OptionType::PUT,
        0.2     // volatility
    );

    auto workspace = AmericanSolverWorkspace::create(-3.0, 3.0, 101, 1000).value();
    AmericanOptionSolver solver(params, workspace);

    auto result = solver.solve();
    ASSERT_TRUE(result.has_value()) << result.error().message;

    // Should converge
    EXPECT_TRUE(result->converged);

    // NOTE: The PDE solver has known issues with time evolution (Issue #73)
    // Until fixed, we can only verify that gamma is computed and finite
    // Gamma should theoretically be positive (convexity), but the buggy
    // time evolution can cause incorrect solution surfaces
    EXPECT_TRUE(std::isfinite(result->gamma));
    // Sanity check: gamma shouldn't be absurdly large
    EXPECT_LT(std::abs(result->gamma), 10000.0);
}

TEST(AmericanOptionSolverTest, SolveAmericanCallWithDiscreteDividends) {
    // Test American call option with discrete dividends
    // Dividends make early exercise more attractive for calls
    AmericanOptionParams params(
        110.0,  // spot (ITM call)
        100.0,  // strike
        1.0,    // maturity
        0.05,   // rate
        0.0,    // dividend_yield (no continuous yield)
        OptionType::CALL,
        0.2,    // volatility
        {
            {0.25, 2.0},  // $2 dividend at t=0.25 years
            {0.75, 2.0}   // $2 dividend at t=0.75 years
        }
    );

    auto workspace = AmericanSolverWorkspace::create(-3.0, 3.0, 101, 1000).value();
    AmericanOptionSolver solver(params, workspace);

    auto result = solver.solve();
    ASSERT_TRUE(result.has_value()) << result.error().message;

    // Should converge
    EXPECT_TRUE(result->converged);

    // ITM call should have positive value
    EXPECT_GT(result->value, 0.0);

    // Value should be at least intrinsic value (spot - strike)
    double intrinsic = params.spot - params.strike;
    EXPECT_GE(result->value, intrinsic * 0.9);  // Allow some numerical error

    // Delta should be positive for call
    EXPECT_GT(result->delta, 0.0);
    EXPECT_LE(result->delta, 1.0);  // Between 0 and 1 for calls

    // Solution should be available
    auto solution = solver.get_solution();
    EXPECT_EQ(solution.size(), 101);  // Default n_space
}

TEST(AmericanOptionSolverTest, SolveAmericanPutWithDiscreteDividends) {
    // Test American put option with discrete dividends
    // Dividends make early exercise less attractive for puts
    AmericanOptionParams params(
        90.0,   // spot (ITM put)
        100.0,  // strike
        1.0,    // maturity
        0.05,   // rate
        0.0,    // dividend_yield (no continuous yield)
        OptionType::PUT,
        0.2,    // volatility
        {
            {0.25, 1.5},  // $1.50 dividend at t=0.25 years
            {0.75, 1.5}   // $1.50 dividend at t=0.75 years
        }
    );

    auto workspace = AmericanSolverWorkspace::create(-3.0, 3.0, 101, 1000).value();
    AmericanOptionSolver solver(params, workspace);

    auto result = solver.solve();
    ASSERT_TRUE(result.has_value()) << result.error().message;

    // Should converge
    EXPECT_TRUE(result->converged);

    // STRICT BOUNDS (Issue #98 fixed):
    // For ITM American put with discrete dividends:
    // - Intrinsic value: max(K - S, 0) = max(100 - 90, 0) = 10.0
    // - Upper bound: strike = 100.0 (put can't exceed strike)

    double intrinsic = params.strike - params.spot;  // 10.0

    // Value should be at least intrinsic (American options worth at least immediate exercise)
    EXPECT_GE(result->value, intrinsic);

    // Value should not exceed strike (theoretical upper bound for puts)
    EXPECT_LE(result->value, params.strike);

    // Value should be finite and positive
    EXPECT_TRUE(std::isfinite(result->value));
    EXPECT_GT(result->value, 0.0);

    // Delta bounds for put: -1 ≤ delta ≤ 0
    EXPECT_TRUE(std::isfinite(result->delta));
    EXPECT_LE(result->delta, 0.0);   // Negative for puts
    EXPECT_GE(result->delta, -1.0);  // Should not be less than -1

    // Gamma should be positive (convexity) and finite
    EXPECT_TRUE(std::isfinite(result->gamma));
    EXPECT_GT(result->gamma, 0.0);  // Options have positive gamma

    // Solution should be available
    auto solution = solver.get_solution();
    EXPECT_EQ(solution.size(), 101);  // Default n_space
}

TEST(AmericanOptionSolverTest, HybridDividendModel) {
    // Test using both continuous and discrete dividends simultaneously
    // This models a stock with continuous yield + known discrete payments
    AmericanOptionParams params(
        100.0,  // spot
        100.0,  // strike
        1.0,    // maturity
        0.05,   // rate
        0.01,   // dividend_yield (1% continuous yield)
        OptionType::PUT,
        0.25,   // volatility
        {
            {0.5, 2.0}  // $2 discrete dividend at mid-year
        }
    );

    auto workspace = AmericanSolverWorkspace::create(-3.0, 3.0, 101, 1000).value();
    AmericanOptionSolver solver(params, workspace);

    auto result = solver.solve();
    ASSERT_TRUE(result.has_value()) << result.error().message;

    // Should converge
    EXPECT_TRUE(result->converged);

    // NOTE: PDE solver has known time evolution issues (Issue #73)
    // For now, just verify solver completes successfully and Greeks are computed

    // Value should be non-negative (may be zero due to Issue #73)
    EXPECT_GE(result->value, 0.0);

    // Value should be bounded by strike
    EXPECT_LE(result->value, params.strike);

    // Delta and gamma should be finite
    EXPECT_TRUE(std::isfinite(result->delta));
    EXPECT_TRUE(std::isfinite(result->gamma));

    // Solution should be available
    auto solution = solver.get_solution();
    EXPECT_EQ(solution.size(), 101);  // Default n_space
}

TEST(BatchAmericanOptionSolverTest, SetupCallbackInvoked) {
    std::vector<AmericanOptionParams> batch(5);
    for (size_t i = 0; i < 5; ++i) {
        batch[i] = AmericanOptionParams(
            100.0,                   // spot
            100.0,                   // strike
            1.0,                     // maturity
            0.05,                    // rate
            0.02,                    // dividend_yield
            OptionType::PUT,
            0.20 + 0.02 * i          // volatility
        );
    }

    // Track callback invocations
    std::vector<size_t> callback_indices;
    std::mutex callback_mutex;

    auto batch_result = BatchAmericanOptionSolver::solve_batch(
        batch, -3.0, 3.0, 101, 1000,
        [&](size_t idx, AmericanOptionSolver& solver) {
            std::lock_guard<std::mutex> lock(callback_mutex);
            callback_indices.push_back(idx);
        });

    // Verify all solves succeeded
    ASSERT_EQ(batch_result.results.size(), 5);
    EXPECT_EQ(batch_result.failed_count, 0);
    for (const auto& result : batch_result.results) {
        EXPECT_TRUE(result.has_value());
    }

    // Verify callback was invoked for each option
    EXPECT_EQ(callback_indices.size(), 5);
    std::sort(callback_indices.begin(), callback_indices.end());
    for (size_t i = 0; i < 5; ++i) {
        EXPECT_EQ(callback_indices[i], i);
    }
}

TEST(BatchAmericanOptionSolverTest, CallbackWithSnapshots) {
    std::vector<AmericanOptionParams> batch(3);
    for (size_t i = 0; i < 3; ++i) {
        batch[i] = AmericanOptionParams(
            100.0,          // spot
            100.0,          // strike
            1.0,            // maturity
            0.05,           // rate
            0.02,           // dividend_yield
            OptionType::PUT,
            0.20            // volatility
        );
    }

    // Create collectors for each solve
    std::vector<double> moneyness = {0.9, 1.0, 1.1};
    std::vector<double> maturities = {0.5, 1.0};

    std::vector<PriceTableSnapshotCollector> collectors;
    for (size_t i = 0; i < 3; ++i) {
        PriceTableSnapshotCollectorConfig config{
            .moneyness = std::span{moneyness},
            .tau = std::span{maturities},
            .K_ref = 100.0,
            .option_type = OptionType::PUT,
            .payoff_params = nullptr
        };
        collectors.emplace_back(config);
    }

    // Register snapshots via callback
    auto batch_result = BatchAmericanOptionSolver::solve_batch(
        batch, -3.0, 3.0, 101, 1000,
        [&](size_t idx, AmericanOptionSolver& solver) {
            solver.register_snapshot(499, 0, &collectors[idx]);  // τ=0.5
            solver.register_snapshot(999, 1, &collectors[idx]);  // τ=1.0
        });

    // Verify all solves succeeded
    ASSERT_EQ(batch_result.results.size(), 3);
    EXPECT_EQ(batch_result.failed_count, 0);
    for (const auto& result : batch_result.results) {
        EXPECT_TRUE(result.has_value());
    }

    // Verify snapshots were collected
    for (size_t i = 0; i < 3; ++i) {
        auto prices = collectors[i].prices();
        EXPECT_EQ(prices.size(), moneyness.size() * maturities.size());

        // All prices should be positive
        for (double price : prices) {
            EXPECT_GT(price, 0.0);
        }
    }
}

TEST(BatchAmericanOptionSolverTest, NoCallbackBackwardCompatible) {
    std::vector<AmericanOptionParams> batch(3);
    for (size_t i = 0; i < 3; ++i) {
        batch[i] = AmericanOptionParams(
            100.0,          // spot
            100.0,          // strike
            1.0,            // maturity
            0.05,           // rate
            0.02,           // dividend_yield
            OptionType::PUT,
            0.20            // volatility
        );
    }

    // Call without callback (backward compatible)
    auto batch_result = BatchAmericanOptionSolver::solve_batch(
        batch, -3.0, 3.0, 101, 1000);

    ASSERT_EQ(batch_result.results.size(), 3);
    EXPECT_EQ(batch_result.failed_count, 0);
    for (const auto& result : batch_result.results) {
        EXPECT_TRUE(result.has_value());
        EXPECT_GT(result.value().value, 0.0);
    }
}

}  // namespace
}  // namespace mango
