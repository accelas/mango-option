#include <gtest/gtest.h>
#include "src/option/table/adaptive_grid_builder.hpp"
#include <iostream>

namespace mango {
namespace {

TEST(AdaptiveGridBuilderTest, ConstructWithDefaultParams) {
    AdaptiveGridParams params;
    AdaptiveGridBuilder builder(params);

    // Should compile and not crash
    SUCCEED();
}

TEST(AdaptiveGridBuilderTest, ConstructWithCustomParams) {
    AdaptiveGridParams params;
    params.target_iv_error = 0.001;  // 10 bps
    params.max_iterations = 3;

    AdaptiveGridBuilder builder(params);
    SUCCEED();
}

TEST(AdaptiveGridBuilderTest, BuildsWithSyntheticChain) {
    // Create a minimal synthetic chain
    OptionChain chain;
    chain.spot = 100.0;
    chain.dividend_yield = 0.0;

    // Add strikes and maturities
    chain.strikes = {90.0, 95.0, 100.0, 105.0, 110.0};
    chain.maturities = {0.25, 0.5, 1.0};
    chain.implied_vols = {0.18, 0.20, 0.22};  // Some variation
    chain.rates = {0.04, 0.05, 0.06};

    AdaptiveGridParams params;
    params.target_iv_error = 0.002;  // 20 bps - relaxed for test speed
    params.max_iterations = 2;
    params.validation_samples = 8;  // Fewer for test speed

    AdaptiveGridBuilder builder(params);

    auto grid_spec = GridSpec<double>::sinh_spaced(-3.0, 3.0, 51, 2.0).value();
    auto result = builder.build(chain, grid_spec, 200, OptionType::PUT);

    if (!result.has_value()) {
        std::cerr << "Build failed with error code: "
                  << static_cast<int>(result.error().code) << "\n";
    }
    ASSERT_TRUE(result.has_value());

    // Should have at least one iteration
    EXPECT_GE(result->iterations.size(), 1);

    // Surface should be populated
    EXPECT_NE(result->surface, nullptr);

    // Should have done some PDE solves
    EXPECT_GT(result->total_pde_solves, 0);
}

TEST(AdaptiveGridBuilderTest, EmptyChainReturnsError) {
    AdaptiveGridParams params;
    AdaptiveGridBuilder builder(params);

    OptionChain chain;
    chain.spot = 100.0;
    chain.dividend_yield = 0.0;
    // No options added

    auto grid_spec = GridSpec<double>::uniform(-3.0, 3.0, 51).value();
    auto result = builder.build(chain, grid_spec, 100, OptionType::PUT);

    // Should return error for empty chain
    EXPECT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code, PriceTableErrorCode::InvalidConfig);
}

}  // namespace
}  // namespace mango
