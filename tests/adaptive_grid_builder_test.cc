#include <gtest/gtest.h>
#include "src/option/table/adaptive_grid_builder.hpp"

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

TEST(AdaptiveGridBuilderTest, BuildReturnsNotImplemented) {
    AdaptiveGridParams params;
    AdaptiveGridBuilder builder(params);

    OptionChain chain;
    chain.spot = 100.0;
    chain.dividend_yield = 0.0;

    auto grid_spec = GridSpec<double>::uniform(-3.0, 3.0, 51).value();
    auto result = builder.build(chain, grid_spec, 100, OptionType::PUT);

    // Should return error since not implemented
    EXPECT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code, PriceTableErrorCode::InvalidConfig);
}

}  // namespace
}  // namespace mango
