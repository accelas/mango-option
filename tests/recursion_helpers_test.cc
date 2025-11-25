#include <gtest/gtest.h>
#include "src/option/table/recursion_helpers.hpp"
#include <vector>

namespace mango {
namespace {

TEST(RecursionHelpersTest, ForEachAxisIndex2D) {
    PriceTableAxes<2> axes;
    axes.grids[0] = {0.9, 1.0};       // 2 points
    axes.grids[1] = {0.1, 0.5, 1.0};  // 3 points

    std::vector<std::array<size_t, 2>> indices;

    for_each_axis_index<0>(axes, [&](const std::array<size_t, 2>& idx) {
        indices.push_back(idx);
    });

    // Should have 2*3 = 6 combinations
    EXPECT_EQ(indices.size(), 6);

    // Check first and last
    EXPECT_EQ(indices[0][0], 0);
    EXPECT_EQ(indices[0][1], 0);
    EXPECT_EQ(indices[5][0], 1);
    EXPECT_EQ(indices[5][1], 2);
}

TEST(RecursionHelpersTest, ForEachAxisIndex4D) {
    PriceTableAxes<4> axes;
    axes.grids[0] = {0.9, 1.0};       // 2
    axes.grids[1] = {0.1, 0.5};       // 2
    axes.grids[2] = {0.15, 0.25};     // 2
    axes.grids[3] = {0.02};           // 1

    size_t count = 0;
    for_each_axis_index<0>(axes, [&](const std::array<size_t, 4>& idx) {
        ++count;
    });

    EXPECT_EQ(count, 2 * 2 * 2 * 1);  // 8 combinations
}

} // namespace
} // namespace mango
