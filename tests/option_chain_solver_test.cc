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

}  // namespace
}  // namespace mango
