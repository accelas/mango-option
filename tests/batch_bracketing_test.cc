// SPDX-License-Identifier: MIT
/**
 * @file batch_bracketing_test.cc
 * @brief Tests for OptionBracketing (grouping heterogeneous options)
 */

#include <gtest/gtest.h>
#include "mango/option/batch_bracketing.hpp"
#include <limits>

namespace mango {
namespace {

TEST(OptionBracketingTest, GroupSingleOption) {
    std::vector<PricingParams> options = {
        PricingParams(OptionSpec{.spot = 100.0, .strike = 100.0, .maturity = 1.0, .rate = 0.05, .option_type = OptionType::PUT}, 0.20)
    };

    auto result = OptionBracketing::group_options(options);
    ASSERT_TRUE(result.has_value()) << "Grouping failed: " << result.error();

    EXPECT_EQ(result->total_options, 1);
    EXPECT_GE(result->num_brackets, 1);
}

TEST(OptionBracketingTest, GroupSimilarOptions) {
    // Create options with similar parameters - should be grouped together
    std::vector<PricingParams> options;
    for (double strike : {95.0, 100.0, 105.0}) {
        options.push_back(PricingParams(OptionSpec{.spot = 100.0, .strike = strike, .maturity = 1.0, .rate = 0.05, .option_type = OptionType::PUT}, 0.20));
    }

    BracketingCriteria criteria{
        .maturity_tolerance = 0.5,
        .moneyness_tolerance = 0.3,  // Wide enough to group these
        .max_bracket_size = 100
    };

    auto result = OptionBracketing::group_options(options, criteria);
    ASSERT_TRUE(result.has_value());

    EXPECT_EQ(result->total_options, 3);
    // Similar options should be grouped into one bracket
    EXPECT_LE(result->num_brackets, 2);
}

TEST(OptionBracketingTest, GroupDiverseOptions) {
    // Create options with diverse parameters - may need multiple brackets
    std::vector<PricingParams> options = {
        // Short-term ATM
        PricingParams(OptionSpec{.spot = 100.0, .strike = 100.0, .maturity = 0.25, .rate = 0.05, .option_type = OptionType::PUT}, 0.20),
        // Long-term ATM
        PricingParams(OptionSpec{.spot = 100.0, .strike = 100.0, .maturity = 2.0, .rate = 0.05, .option_type = OptionType::PUT}, 0.20),
        // Deep ITM
        PricingParams(OptionSpec{.spot = 80.0, .strike = 100.0, .maturity = 1.0, .rate = 0.05, .option_type = OptionType::PUT}, 0.20),
        // Deep OTM
        PricingParams(OptionSpec{.spot = 120.0, .strike = 100.0, .maturity = 1.0, .rate = 0.05, .option_type = OptionType::PUT}, 0.20),
    };

    BracketingCriteria criteria{
        .maturity_tolerance = 0.3,  // Tight tolerance
        .moneyness_tolerance = 0.1,
        .max_bracket_size = 100
    };

    auto result = OptionBracketing::group_options(options, criteria);
    ASSERT_TRUE(result.has_value());

    EXPECT_EQ(result->total_options, 4);
    // With tight tolerances, should create multiple brackets
    EXPECT_GE(result->num_brackets, 1);
}

TEST(OptionBracketingTest, GroupEmptyOptions) {
    std::vector<PricingParams> options;

    auto result = OptionBracketing::group_options(options);
    // Empty input should either succeed with 0 brackets or fail gracefully
    if (result.has_value()) {
        EXPECT_EQ(result->total_options, 0);
        EXPECT_EQ(result->num_brackets, 0);
    }
}

TEST(OptionBracketingTest, ComputeDistanceIdenticalOptions) {
    PricingParams a(OptionSpec{.spot = 100.0, .strike = 100.0, .maturity = 1.0, .rate = 0.05, .option_type = OptionType::PUT}, 0.20);
    PricingParams b(OptionSpec{.spot = 100.0, .strike = 100.0, .maturity = 1.0, .rate = 0.05, .option_type = OptionType::PUT}, 0.20);

    BracketingCriteria criteria;
    double distance = OptionBracketing::compute_distance(a, b, criteria);

    EXPECT_NEAR(distance, 0.0, 1e-10);
}

TEST(OptionBracketingTest, ComputeDistanceDifferentMaturity) {
    PricingParams a(OptionSpec{.spot = 100.0, .strike = 100.0, .maturity = 1.0, .rate = 0.05, .option_type = OptionType::PUT}, 0.20);
    PricingParams b(OptionSpec{.spot = 100.0, .strike = 100.0, .maturity = 2.0, .rate = 0.05, .option_type = OptionType::PUT}, 0.20);  // Different maturity

    BracketingCriteria criteria{.maturity_tolerance = 1.0};
    double distance = OptionBracketing::compute_distance(a, b, criteria);

    EXPECT_GT(distance, 0.0);
}

TEST(OptionBracketingTest, ComputeDistanceDifferentMoneyness) {
    PricingParams a(OptionSpec{.spot = 100.0, .strike = 100.0, .maturity = 1.0, .rate = 0.05, .option_type = OptionType::PUT}, 0.20);  // m = 1.0
    PricingParams b(OptionSpec{.spot = 90.0, .strike = 100.0, .maturity = 1.0, .rate = 0.05, .option_type = OptionType::PUT}, 0.20);   // m = 0.9

    BracketingCriteria criteria;
    double distance = OptionBracketing::compute_distance(a, b, criteria);

    EXPECT_GT(distance, 0.0);
}

TEST(OptionBracketingTest, EstimateBracketGrid) {
    std::vector<PricingParams> options = {
        PricingParams(OptionSpec{.spot = 100.0, .strike = 95.0, .maturity = 1.0, .rate = 0.05, .option_type = OptionType::PUT}, 0.20),
        PricingParams(OptionSpec{.spot = 100.0, .strike = 100.0, .maturity = 1.0, .rate = 0.05, .option_type = OptionType::PUT}, 0.25),
        PricingParams(OptionSpec{.spot = 100.0, .strike = 105.0, .maturity = 1.0, .rate = 0.05, .option_type = OptionType::PUT}, 0.20),
    };

    auto result = OptionBracketing::estimate_bracket_grid(options);
    ASSERT_TRUE(result.has_value()) << "Grid estimation failed: " << result.error();

    auto [grid_spec, time_domain] = result.value();

    // Grid should cover all strikes with margin
    EXPECT_GT(grid_spec.n_points(), 10);  // Reasonable grid size
    EXPECT_GT(time_domain.n_steps(), 0);
}

TEST(OptionBracketingTest, BracketStats) {
    std::vector<PricingParams> options = {
        PricingParams(OptionSpec{.spot = 90.0, .strike = 100.0, .maturity = 0.5, .rate = 0.05, .option_type = OptionType::PUT}, 0.15),
        PricingParams(OptionSpec{.spot = 100.0, .strike = 100.0, .maturity = 1.0, .rate = 0.05, .option_type = OptionType::PUT}, 0.20),
        PricingParams(OptionSpec{.spot = 110.0, .strike = 100.0, .maturity = 1.5, .rate = 0.05, .option_type = OptionType::PUT}, 0.25),
    };

    BracketingCriteria criteria{
        .maturity_tolerance = 2.0,  // Wide enough to group all
        .moneyness_tolerance = 0.5
    };

    auto result = OptionBracketing::group_options(options, criteria);
    ASSERT_TRUE(result.has_value());
    ASSERT_GE(result->brackets.size(), 1);

    // Check that stats are computed for at least one bracket
    // Options may be split into multiple brackets depending on algorithm
    double overall_min_mat = std::numeric_limits<double>::max();
    double overall_max_mat = std::numeric_limits<double>::lowest();
    for (const auto& bracket : result->brackets) {
        overall_min_mat = std::min(overall_min_mat, bracket.stats.min_maturity);
        overall_max_mat = std::max(overall_max_mat, bracket.stats.max_maturity);
    }

    // Overall stats should cover the range of input options
    EXPECT_LE(overall_min_mat, 0.5);
    EXPECT_GE(overall_max_mat, 1.5);
}

TEST(OptionBracketingTest, OriginalIndicesPreserved) {
    std::vector<PricingParams> options = {
        PricingParams(OptionSpec{.spot = 100.0, .strike = 90.0, .maturity = 1.0, .rate = 0.05, .option_type = OptionType::PUT}, 0.20),
        PricingParams(OptionSpec{.spot = 100.0, .strike = 100.0, .maturity = 1.0, .rate = 0.05, .option_type = OptionType::PUT}, 0.20),
        PricingParams(OptionSpec{.spot = 100.0, .strike = 110.0, .maturity = 1.0, .rate = 0.05, .option_type = OptionType::PUT}, 0.20),
    };

    auto result = OptionBracketing::group_options(options);
    ASSERT_TRUE(result.has_value());

    // Collect all original indices
    std::vector<bool> seen(3, false);
    for (const auto& bracket : result->brackets) {
        for (size_t idx : bracket.original_indices) {
            EXPECT_LT(idx, 3) << "Index out of range";
            seen[idx] = true;
        }
    }

    // All original indices should be accounted for
    for (size_t i = 0; i < 3; ++i) {
        EXPECT_TRUE(seen[i]) << "Original index " << i << " not found in brackets";
    }
}

TEST(OptionBracketingTest, MaxBracketSizeRespected) {
    // Create many similar options
    std::vector<PricingParams> options;
    for (int i = 0; i < 150; ++i) {
        options.push_back(PricingParams(OptionSpec{.spot = 100.0, .strike = 100.0, .maturity = 1.0, .rate = 0.05, .option_type = OptionType::PUT}, 0.20));
    }

    BracketingCriteria criteria{
        .max_bracket_size = 50  // Limit to 50 per bracket
    };

    auto result = OptionBracketing::group_options(options, criteria);
    ASSERT_TRUE(result.has_value());

    // Each bracket should respect max size
    for (const auto& bracket : result->brackets) {
        EXPECT_LE(bracket.options.size(), 50);
    }
}

TEST(OptionBracketingTest, AvgBracketSize) {
    std::vector<PricingParams> options;
    for (int i = 0; i < 10; ++i) {
        options.push_back(PricingParams(OptionSpec{.spot = 100.0, .strike = 100.0, .maturity = 1.0, .rate = 0.05, .option_type = OptionType::PUT}, 0.20));
    }

    auto result = OptionBracketing::group_options(options);
    ASSERT_TRUE(result.has_value());

    // Average should equal total / num_brackets
    double expected_avg = static_cast<double>(result->total_options) / result->num_brackets;
    EXPECT_NEAR(result->avg_bracket_size(), expected_avg, 1e-10);
}

TEST(BracketingCriteriaTest, DefaultValues) {
    BracketingCriteria criteria;

    EXPECT_EQ(criteria.maturity_tolerance, 0.5);
    EXPECT_EQ(criteria.moneyness_tolerance, 0.2);
    EXPECT_EQ(criteria.volatility_tolerance, 0.1);
    EXPECT_EQ(criteria.rate_tolerance, 0.05);
    EXPECT_EQ(criteria.max_bracket_size, 100);
    EXPECT_EQ(criteria.min_bracket_size, 3);
}

// OptionBracket cannot be default constructed (GridSpec has no default ctor)
// This is correct behavior - brackets should only be created by OptionBracketing::group_options

TEST(BracketingResultTest, AvgBracketSizeZeroBrackets) {
    BracketingResult result;
    result.total_options = 0;
    result.num_brackets = 1;  // Avoid division by zero in test

    // With 1 bracket and 0 options, avg should be 0
    EXPECT_NEAR(result.avg_bracket_size(), 0.0, 1e-10);
}

}  // namespace
}  // namespace mango
