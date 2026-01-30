// SPDX-License-Identifier: MIT
#include <experimental/mdspan>
#include <gtest/gtest.h>
#include <vector>

namespace {
using std::experimental::mdspan;
using std::experimental::extents;
using std::experimental::dextents;
using std::experimental::layout_right;

TEST(MdspanIntegration, BasicUsage) {
    std::vector<double> data{1.0, 2.0, 3.0, 4.0};
    mdspan<double, extents<size_t, 2, 2>> matrix(data.data());

    // C++23 multidimensional subscripting (parentheses protect comma from preprocessor)
    EXPECT_EQ((matrix[0, 0]), 1.0);
    EXPECT_EQ((matrix[0, 1]), 2.0);
    EXPECT_EQ((matrix[1, 0]), 3.0);
    EXPECT_EQ((matrix[1, 1]), 4.0);
}

TEST(MdspanIntegration, LayoutRight) {
    std::vector<int> data(12);
    std::iota(data.begin(), data.end(), 0);

    mdspan<int, extents<size_t, 2, 3, 2>, layout_right> tensor(data.data());

    // C++23 multidimensional subscripting - Row-major: last dimension varies fastest
    EXPECT_EQ((tensor[0, 0, 0]), 0);
    EXPECT_EQ((tensor[0, 0, 1]), 1);
    EXPECT_EQ((tensor[0, 1, 0]), 2);
}
}  // namespace
