#include <gtest/gtest.h>
#include "src/option/price_table_builder.hpp"

namespace mango {
namespace {

TEST(PriceTableBuilderTest, ConstructFromConfig) {
    PriceTableConfig config;
    PriceTableBuilder<4> builder(config);

    // Just verify construction succeeds
    SUCCEED();
}

TEST(PriceTableBuilderTest, BuildEmpty4DSurface) {
    PriceTableConfig config;
    PriceTableBuilder<4> builder(config);

    PriceTableAxes<4> axes;
    axes.grids[0] = {0.8, 0.9, 1.0, 1.1};
    axes.grids[1] = {0.1, 0.5, 1.0, 1.5};
    axes.grids[2] = {0.15, 0.20, 0.25, 0.30};
    axes.grids[3] = {0.02, 0.04, 0.06, 0.08};

    // This will fail until we implement the pipeline
    // For now, just verify it returns an error
    auto result = builder.build(axes);
    EXPECT_FALSE(result.has_value());  // Not implemented yet
}

} // namespace
} // namespace mango
