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

}  // namespace
}  // namespace mango
