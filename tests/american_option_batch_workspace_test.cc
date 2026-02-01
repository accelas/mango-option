// SPDX-License-Identifier: MIT
#include <gtest/gtest.h>
#include "src/option/american_option_batch.hpp"

using namespace mango;

TEST(AmericanOptionBatchWorkspaceTest, BatchResultsUnchanged) {
    std::vector<PricingParams> batch;
    for (int i = 0; i < 10; ++i) {
        batch.push_back(PricingParams(
            OptionSpec{.spot = 100.0, .strike = 90.0 + i * 2.0, .maturity = 1.0, .rate = 0.05, .dividend_yield = 0.02, .type = OptionType::PUT}, 0.20));
    }

    BatchAmericanOptionSolver solver;
    auto results = solver.solve_batch(batch);

    EXPECT_EQ(results.results.size(), 10u);
    for (size_t i = 0; i < results.results.size(); ++i) {
        ASSERT_TRUE(results.results[i].has_value()) << "Option " << i << " failed";
        EXPECT_GT(results.results[i]->value(), 0.0);
        EXPECT_LT(results.results[i]->delta(), 0.0);  // Put delta negative
    }
}

TEST(AmericanOptionBatchWorkspaceTest, SharedGridMode) {
    std::vector<PricingParams> batch;
    for (int i = 0; i < 5; ++i) {
        batch.push_back(PricingParams(
            OptionSpec{.spot = 100.0, .strike = 100.0 + i * 5.0, .maturity = 1.0, .rate = 0.05, .dividend_yield = 0.02, .type = OptionType::PUT}, 0.20));
    }

    BatchAmericanOptionSolver solver;
    auto results = solver.solve_batch(batch, /*use_shared_grid=*/true);

    EXPECT_TRUE(results.all_succeeded());
    EXPECT_EQ(results.results.size(), 5u);
}

TEST(AmericanOptionBatchWorkspaceTest, PerOptionGridMode) {
    std::vector<PricingParams> batch;
    for (int i = 0; i < 5; ++i) {
        batch.push_back(PricingParams(
            OptionSpec{.spot = 100.0, .strike = 80.0 + i * 10.0, .maturity = 0.5 + i * 0.25, .rate = 0.05, .dividend_yield = 0.02, .type = OptionType::PUT}, 0.15 + i * 0.05));
    }

    BatchAmericanOptionSolver solver;
    auto results = solver.solve_batch(batch, /*use_shared_grid=*/false);

    EXPECT_TRUE(results.all_succeeded());
    EXPECT_EQ(results.results.size(), 5u);
}

TEST(AmericanOptionBatchWorkspaceTest, MixedCallAndPut) {
    std::vector<PricingParams> batch;
    for (int i = 0; i < 4; ++i) {
        OptionType type = (i % 2 == 0) ? OptionType::CALL : OptionType::PUT;
        batch.push_back(PricingParams(
            OptionSpec{.spot = 100.0, .strike = 95.0 + i * 5.0, .maturity = 1.0, .rate = 0.05, .dividend_yield = 0.02, .type = type}, 0.20));
    }

    BatchAmericanOptionSolver solver;
    auto results = solver.solve_batch(batch);

    EXPECT_EQ(results.results.size(), 4u);
    for (size_t i = 0; i < results.results.size(); ++i) {
        ASSERT_TRUE(results.results[i].has_value()) << "Option " << i << " failed";
        EXPECT_GT(results.results[i]->value(), 0.0);
    }
}

TEST(AmericanOptionBatchWorkspaceTest, EmptyBatch) {
    std::vector<PricingParams> batch;

    BatchAmericanOptionSolver solver;
    auto results = solver.solve_batch(batch);

    EXPECT_EQ(results.results.size(), 0u);
    EXPECT_EQ(results.failed_count, 0);
}

TEST(AmericanOptionBatchWorkspaceTest, SingleOption) {
    std::vector<PricingParams> batch;
    batch.push_back(PricingParams(
            OptionSpec{.spot = 100.0, .strike = 100.0, .maturity = 1.0, .rate = 0.05, .dividend_yield = 0.02, .type = OptionType::PUT}, 0.20));

    BatchAmericanOptionSolver solver;
    auto results = solver.solve_batch(batch);

    EXPECT_EQ(results.results.size(), 1u);
    ASSERT_TRUE(results.results[0].has_value());
    EXPECT_GT(results.results[0]->value(), 0.0);
}
