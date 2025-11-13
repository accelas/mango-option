#include "src/option/option_chain_solver.hpp"
#include <gtest/gtest.h>

namespace mango {
namespace {

TEST(AmericanOptionChainTest, ValidChainPassesValidation) {
    AmericanOptionChain chain{
        .spot = 100.0,
        .maturity = 1.0,
        .volatility = 0.20,
        .rate = 0.05,
        .continuous_dividend_yield = 0.02,
        .option_type = OptionType::PUT,
        .strikes = {90.0, 95.0, 100.0, 105.0, 110.0},
        .discrete_dividends = {}
    };

    auto result = chain.validate();
    EXPECT_TRUE(result.has_value());
}

TEST(AmericanOptionChainTest, NegativeSpotFails) {
    AmericanOptionChain chain{
        .spot = -100.0,
        .maturity = 1.0,
        .volatility = 0.20,
        .rate = 0.05,
        .continuous_dividend_yield = 0.02,
        .option_type = OptionType::PUT,
        .strikes = {100.0}
    };

    auto result = chain.validate();
    EXPECT_FALSE(result.has_value());
    EXPECT_NE(result.error().find("Spot"), std::string::npos);
}

TEST(AmericanOptionChainTest, EmptyStrikesFails) {
    AmericanOptionChain chain{
        .spot = 100.0,
        .maturity = 1.0,
        .volatility = 0.20,
        .rate = 0.05,
        .continuous_dividend_yield = 0.02,
        .option_type = OptionType::PUT,
        .strikes = {},  // Empty!
        .discrete_dividends = {}
    };

    auto result = chain.validate();
    EXPECT_FALSE(result.has_value());
    EXPECT_NE(result.error().find("at least one strike"), std::string::npos);
}

TEST(AmericanOptionChainTest, ZeroMaturityFails) {
    AmericanOptionChain chain{
        .spot = 100.0,
        .maturity = 0.0,  // Invalid!
        .volatility = 0.20,
        .rate = 0.05,
        .continuous_dividend_yield = 0.02,
        .option_type = OptionType::PUT,
        .strikes = {100.0}
    };

    auto result = chain.validate();
    EXPECT_FALSE(result.has_value());
    EXPECT_NE(result.error().find("Maturity"), std::string::npos);
}

TEST(AmericanOptionChainTest, NegativeVolatilityFails) {
    AmericanOptionChain chain{
        .spot = 100.0,
        .maturity = 1.0,
        .volatility = -0.20,  // Invalid!
        .rate = 0.05,
        .continuous_dividend_yield = 0.02,
        .option_type = OptionType::PUT,
        .strikes = {100.0}
    };

    auto result = chain.validate();
    EXPECT_FALSE(result.has_value());
    EXPECT_NE(result.error().find("Volatility"), std::string::npos);
}

TEST(AmericanOptionChainTest, NegativeContinuousDividendFails) {
    AmericanOptionChain chain{
        .spot = 100.0,
        .maturity = 1.0,
        .volatility = 0.20,
        .rate = 0.05,
        .continuous_dividend_yield = -0.02,  // Invalid!
        .option_type = OptionType::PUT,
        .strikes = {100.0}
    };

    auto result = chain.validate();
    EXPECT_FALSE(result.has_value());
    EXPECT_NE(result.error().find("dividend yield"), std::string::npos);
}

TEST(AmericanOptionChainTest, ZeroStrikeFails) {
    AmericanOptionChain chain{
        .spot = 100.0,
        .maturity = 1.0,
        .volatility = 0.20,
        .rate = 0.05,
        .continuous_dividend_yield = 0.02,
        .option_type = OptionType::PUT,
        .strikes = {100.0, 0.0, 110.0}  // Zero strike invalid!
    };

    auto result = chain.validate();
    EXPECT_FALSE(result.has_value());
    EXPECT_NE(result.error().find("strikes must be positive"), std::string::npos);
}

TEST(AmericanOptionChainTest, InvalidDiscreteDividendTimeFails) {
    AmericanOptionChain chain{
        .spot = 100.0,
        .maturity = 1.0,
        .volatility = 0.20,
        .rate = 0.05,
        .continuous_dividend_yield = 0.02,
        .option_type = OptionType::PUT,
        .strikes = {100.0},
        .discrete_dividends = {{1.5, 2.0}}  // time > maturity!
    };

    auto result = chain.validate();
    EXPECT_FALSE(result.has_value());
    EXPECT_NE(result.error().find("dividend time"), std::string::npos);
}

TEST(AmericanOptionChainTest, NegativeDiscreteDividendAmountFails) {
    AmericanOptionChain chain{
        .spot = 100.0,
        .maturity = 1.0,
        .volatility = 0.20,
        .rate = 0.05,
        .continuous_dividend_yield = 0.02,
        .option_type = OptionType::PUT,
        .strikes = {100.0},
        .discrete_dividends = {{0.5, -2.0}}  // negative amount!
    };

    auto result = chain.validate();
    EXPECT_FALSE(result.has_value());
    EXPECT_NE(result.error().find("dividend amount"), std::string::npos);
}

TEST(OptionChainSolverTest, SolvesSimpleChainSequentially) {
    // Create chain: 5 put strikes around ATM
    AmericanOptionChain chain{
        .spot = 100.0,
        .maturity = 1.0,
        .volatility = 0.20,
        .rate = 0.05,
        .continuous_dividend_yield = 0.02,
        .option_type = OptionType::PUT,
        .strikes = {90.0, 95.0, 100.0, 105.0, 110.0},
        .discrete_dividends = {}
    };

    AmericanOptionGrid grid;
    grid.n_space = 101;
    grid.n_time = 1000;
    grid.x_min = -3.0;
    grid.x_max = 3.0;

    auto results = OptionChainSolver::solve_chain(chain, grid);

    ASSERT_EQ(results.size(), 5);

    // All should converge
    for (const auto& [strike, result] : results) {
        ASSERT_TRUE(result.has_value()) << "Strike " << strike << " failed to converge";
        EXPECT_GT(result->value, 0.0) << "Strike " << strike << " has non-positive value";
    }

    // Put values should increase as strike increases (intrinsic value)
    EXPECT_LT(results[0].result->value, results[4].result->value)
        << "Deep OTM put should be cheaper than deep ITM put";
}

TEST(OptionChainSolverTest, HandlesInvalidChainGracefully) {
    // Invalid chain: negative volatility
    AmericanOptionChain chain{
        .spot = 100.0,
        .maturity = 1.0,
        .volatility = -0.20,  // Invalid!
        .rate = 0.05,
        .continuous_dividend_yield = 0.02,
        .option_type = OptionType::PUT,
        .strikes = {100.0}
    };

    AmericanOptionGrid grid;

    auto results = OptionChainSolver::solve_chain(chain, grid);

    ASSERT_EQ(results.size(), 1);
    EXPECT_FALSE(results[0].result.has_value());
    EXPECT_EQ(results[0].result.error().code, SolverErrorCode::InvalidConfiguration);
}

TEST(OptionChainSolverTest, SolvesMultipleChainsInParallel) {
    AmericanOptionGrid grid;
    grid.n_space = 101;
    grid.n_time = 1000;
    grid.x_min = -3.0;
    grid.x_max = 3.0;

    // Create 3 chains with different parameters
    std::vector<AmericanOptionChain> chains = {
        // Chain 1: ATM puts
        {
            .spot = 100.0,
            .maturity = 1.0,
            .volatility = 0.20,
            .rate = 0.05,
            .continuous_dividend_yield = 0.02,
            .option_type = OptionType::PUT,
            .strikes = {95.0, 100.0, 105.0}
        },
        // Chain 2: OTM calls
        {
            .spot = 100.0,
            .maturity = 0.5,
            .volatility = 0.30,
            .rate = 0.03,
            .continuous_dividend_yield = 0.01,
            .option_type = OptionType::CALL,
            .strikes = {105.0, 110.0, 115.0}
        },
        // Chain 3: Deep ITM puts
        {
            .spot = 100.0,
            .maturity = 2.0,
            .volatility = 0.25,
            .rate = 0.04,
            .continuous_dividend_yield = 0.015,
            .option_type = OptionType::PUT,
            .strikes = {120.0, 125.0, 130.0}
        }
    };

    auto all_results = OptionChainSolver::solve_chains(chains, grid);

    ASSERT_EQ(all_results.size(), 3);

    // Check each chain
    for (size_t i = 0; i < chains.size(); ++i) {
        ASSERT_EQ(all_results[i].size(), chains[i].strikes.size())
            << "Chain " << i << " has wrong number of results";

        for (const auto& [strike, result] : all_results[i]) {
            ASSERT_TRUE(result.has_value())
                << "Chain " << i << ", strike " << strike << " failed";
            EXPECT_GT(result->value, 0.0)
                << "Chain " << i << ", strike " << strike << " has non-positive value";
        }
    }
}

TEST(OptionChainSolverTest, ParallelMatchesSequential) {
    // Test that parallel execution gives same results as sequential
    AmericanOptionGrid grid;
    grid.n_space = 101;
    grid.n_time = 1000;
    grid.x_min = -3.0;
    grid.x_max = 3.0;

    AmericanOptionChain chain{
        .spot = 100.0,
        .maturity = 1.0,
        .volatility = 0.20,
        .rate = 0.05,
        .continuous_dividend_yield = 0.02,
        .option_type = OptionType::PUT,
        .strikes = {90.0, 95.0, 100.0, 105.0, 110.0}
    };

    // Solve sequentially
    auto sequential_results = OptionChainSolver::solve_chain(chain, grid);

    // Solve in parallel (single chain, but tests parallel path)
    std::vector<AmericanOptionChain> chains = {chain};
    auto parallel_results = OptionChainSolver::solve_chains(chains, grid);

    ASSERT_EQ(parallel_results.size(), 1);
    ASSERT_EQ(parallel_results[0].size(), sequential_results.size());

    // Compare results
    for (size_t i = 0; i < sequential_results.size(); ++i) {
        ASSERT_TRUE(sequential_results[i].result.has_value());
        ASSERT_TRUE(parallel_results[0][i].result.has_value());

        EXPECT_NEAR(sequential_results[i].result->value,
                   parallel_results[0][i].result->value,
                   1e-10)
            << "Mismatch at strike " << sequential_results[i].strike;
    }
}

}  // namespace
}  // namespace mango
