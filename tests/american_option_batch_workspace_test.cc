// SPDX-License-Identifier: MIT
#include <gtest/gtest.h>
#include "src/option/american_option_batch.hpp"

using namespace mango;

TEST(AmericanOptionBatchWorkspaceTest, BatchResultsUnchanged) {
    std::vector<AmericanOptionParams> batch;
    for (int i = 0; i < 10; ++i) {
        batch.emplace_back(
            100.0,              // spot
            90.0 + i * 2.0,     // strike (varying)
            1.0,                // maturity
            0.05,               // rate
            0.02,               // dividend_yield
            OptionType::PUT,    // type
            0.20                // volatility
        );
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
    std::vector<AmericanOptionParams> batch;
    for (int i = 0; i < 5; ++i) {
        batch.emplace_back(
            100.0,                  // spot
            100.0 + i * 5.0,        // strike (varying)
            1.0,                    // maturity
            0.05,                   // rate
            0.02,                   // dividend_yield
            OptionType::PUT,        // type
            0.20                    // volatility
        );
    }

    BatchAmericanOptionSolver solver;
    auto results = solver.solve_batch(batch, /*use_shared_grid=*/true);

    EXPECT_TRUE(results.all_succeeded());
    EXPECT_EQ(results.results.size(), 5u);
}

TEST(AmericanOptionBatchWorkspaceTest, PerOptionGridMode) {
    std::vector<AmericanOptionParams> batch;
    for (int i = 0; i < 5; ++i) {
        batch.emplace_back(
            100.0,                  // spot
            80.0 + i * 10.0,        // strike (varying widely)
            0.5 + i * 0.25,         // maturity (varying)
            0.05,                   // rate
            0.02,                   // dividend_yield
            OptionType::PUT,        // type
            0.15 + i * 0.05         // volatility (varying)
        );
    }

    BatchAmericanOptionSolver solver;
    auto results = solver.solve_batch(batch, /*use_shared_grid=*/false);

    EXPECT_TRUE(results.all_succeeded());
    EXPECT_EQ(results.results.size(), 5u);
}

TEST(AmericanOptionBatchWorkspaceTest, MixedCallAndPut) {
    std::vector<AmericanOptionParams> batch;
    for (int i = 0; i < 4; ++i) {
        OptionType type = (i % 2 == 0) ? OptionType::CALL : OptionType::PUT;
        batch.emplace_back(
            100.0,                  // spot
            95.0 + i * 5.0,         // strike
            1.0,                    // maturity
            0.05,                   // rate
            0.02,                   // dividend_yield
            type,                   // type (alternating)
            0.20                    // volatility
        );
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
    std::vector<AmericanOptionParams> batch;

    BatchAmericanOptionSolver solver;
    auto results = solver.solve_batch(batch);

    EXPECT_EQ(results.results.size(), 0u);
    EXPECT_EQ(results.failed_count, 0);
}

TEST(AmericanOptionBatchWorkspaceTest, SingleOption) {
    std::vector<AmericanOptionParams> batch;
    batch.emplace_back(
        100.0,                  // spot
        100.0,                  // strike
        1.0,                    // maturity
        0.05,                   // rate
        0.02,                   // dividend_yield
        OptionType::PUT,        // type
        0.20                    // volatility
    );

    BatchAmericanOptionSolver solver;
    auto results = solver.solve_batch(batch);

    EXPECT_EQ(results.results.size(), 1u);
    ASSERT_TRUE(results.results[0].has_value());
    EXPECT_GT(results.results[0]->value(), 0.0);
}
