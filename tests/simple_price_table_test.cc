// SPDX-License-Identifier: MIT
#include <gtest/gtest.h>
#include "src/simple/price_table.hpp"
#include <filesystem>
#include <cmath>

namespace {

using mango::OptionType;

// Use a small table for fast testing
mango::simple::PriceTableConfig small_config() {
    return {
        .type = OptionType::PUT,
        .strike_ref = 100.0,
        .dividend_yield = 0.0,
        .n_moneyness = 5,
        .n_maturity = 4,
        .n_volatility = 4,
        .n_rate = 4,
        .moneyness_min = 0.8,
        .moneyness_max = 1.2,
        .maturity_min = 0.1,
        .maturity_max = 1.0,
        .vol_min = 0.10,
        .vol_max = 0.40,
        .rate_min = 0.01,
        .rate_max = 0.08,
    };
}

TEST(SimplePriceTableTest, BuildSmallTable) {
    auto result = mango::simple::build_price_table(small_config());
    ASSERT_TRUE(result.has_value()) << result.error();

    auto& table = *result;
    EXPECT_NE(table.surface(), nullptr);
    EXPECT_EQ(table.type(), OptionType::PUT);
    EXPECT_DOUBLE_EQ(table.strike_ref(), 100.0);
}

TEST(SimplePriceTableTest, QueryATMPut) {
    auto result = mango::simple::build_price_table(small_config());
    ASSERT_TRUE(result.has_value()) << result.error();

    // ATM put: moneyness=1.0, tau=0.5, sigma=0.20, rate=0.05
    double price = result->value(1.0, 0.5, 0.20, 0.05);
    EXPECT_GT(price, 0.0);
    EXPECT_LT(price, 20.0);
}

TEST(SimplePriceTableTest, SaveLoadRoundtrip) {
    auto build_result = mango::simple::build_price_table(small_config());
    ASSERT_TRUE(build_result.has_value()) << build_result.error();

    // Save
    auto tmp_path = std::filesystem::temp_directory_path() / "simple_price_table_test.arrow";
    auto save_result = build_result->save(tmp_path);
    ASSERT_TRUE(save_result.has_value()) << save_result.error();

    // Load
    auto load_result = mango::simple::load_price_table(tmp_path);
    ASSERT_TRUE(load_result.has_value()) << load_result.error();

    // Compare a query point
    double original = build_result->value(1.0, 0.5, 0.20, 0.05);
    double loaded   = load_result->value(1.0, 0.5, 0.20, 0.05);
    EXPECT_NEAR(original, loaded, 1e-10);

    // Cleanup
    std::filesystem::remove(tmp_path);
}

TEST(SimplePriceTableTest, MakeIVSolver) {
    auto build_result = mango::simple::build_price_table(small_config());
    ASSERT_TRUE(build_result.has_value()) << build_result.error();

    auto solver_result = mango::simple::make_iv_solver(*build_result);
    ASSERT_TRUE(solver_result.has_value()) << solver_result.error();
}

}  // anonymous namespace
