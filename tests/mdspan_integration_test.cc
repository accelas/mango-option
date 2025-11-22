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

    // Access via data_handle()[mapping(...)]
    auto& mapping = matrix.mapping();
    auto* ptr = matrix.data_handle();

    EXPECT_EQ(ptr[mapping(0, 0)], 1.0);
    EXPECT_EQ(ptr[mapping(0, 1)], 2.0);
    EXPECT_EQ(ptr[mapping(1, 0)], 3.0);
    EXPECT_EQ(ptr[mapping(1, 1)], 4.0);
}

TEST(MdspanIntegration, LayoutRight) {
    std::vector<int> data(12);
    std::iota(data.begin(), data.end(), 0);

    mdspan<int, extents<size_t, 2, 3, 2>, layout_right> tensor(data.data());

    // Access via data_handle()[mapping(...)]
    auto& mapping = tensor.mapping();
    auto* ptr = tensor.data_handle();

    // Row-major: last dimension varies fastest
    EXPECT_EQ(ptr[mapping(0, 0, 0)], 0);
    EXPECT_EQ(ptr[mapping(0, 0, 1)], 1);
    EXPECT_EQ(ptr[mapping(0, 1, 0)], 2);
}
}  // namespace
