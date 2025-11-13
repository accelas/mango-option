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

}  // namespace
}  // namespace mango
