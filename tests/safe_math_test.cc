// SPDX-License-Identifier: MIT
#include <gtest/gtest.h>
#include "mango/math/safe_math.hpp"
#include <limits>
#include <array>
#include <vector>
#include <span>
#include <cmath>

namespace mango {
namespace {

TEST(SafeMathTest, MultiplySmallValues) {
    auto result = safe_multiply(10, 20);
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(result.value(), 200);
}

TEST(SafeMathTest, MultiplyZero) {
    auto result = safe_multiply(0, 1000000);
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(result.value(), 0);
}

TEST(SafeMathTest, MultiplyOne) {
    auto result = safe_multiply(1, std::numeric_limits<size_t>::max());
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(result.value(), std::numeric_limits<size_t>::max());
}

TEST(SafeMathTest, MultiplyOverflow) {
    // Two large values that overflow
    size_t large = std::numeric_limits<size_t>::max() / 2 + 1;
    auto result = safe_multiply(large, 3);
    ASSERT_FALSE(result.has_value());
    EXPECT_EQ(result.error().operand_a, large);
    EXPECT_EQ(result.error().operand_b, 3);
}

TEST(SafeMathTest, MultiplyMaxTimesTwo) {
    size_t max_val = std::numeric_limits<size_t>::max();
    auto result = safe_multiply(max_val, 2);
    ASSERT_FALSE(result.has_value());
}

TEST(SafeMathTest, MultiplyLargeNoOverflow) {
    // A value that when squared stays under SIZE_MAX
    // For 64-bit: (2^32 - 1)^2 = 2^64 - 2^33 + 1, which is < 2^64 - 1
    size_t val = (1ULL << 32) - 1;  // 4294967295
    auto result = safe_multiply(val, val);
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(result.value(), val * val);
}

TEST(SafeMathTest, ProductEmptyContainer) {
    std::vector<size_t> empty;
    auto result = safe_product(empty);
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(result.value(), 1);  // Identity for multiplication
}

TEST(SafeMathTest, ProductSingleElement) {
    std::vector<size_t> values = {42};
    auto result = safe_product(values);
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(result.value(), 42);
}

TEST(SafeMathTest, ProductMultipleElements) {
    std::vector<size_t> values = {2, 3, 4, 5};
    auto result = safe_product(values);
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(result.value(), 120);  // 2*3*4*5 = 120
}

TEST(SafeMathTest, ProductOverflowInMiddle) {
    // Large values that overflow when multiplied together
    size_t large = std::numeric_limits<size_t>::max() / 10;
    std::vector<size_t> values = {2, large, large};  // 2 * large is ok, but then * large overflows
    auto result = safe_product(values);
    ASSERT_FALSE(result.has_value());
}

TEST(SafeMathTest, ProductArray4D) {
    std::array<size_t, 4> shape = {50, 20, 30, 5};
    auto result = safe_product(shape);
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(result.value(), 50 * 20 * 30 * 5);  // 150,000
}

TEST(SafeMathTest, ProductArrayOverflow) {
    // Dimensions that would overflow on 64-bit
    size_t large = 1ULL << 20;  // ~1M
    std::array<size_t, 4> shape = {large, large, large, large};
    auto result = safe_product(shape);
    ASSERT_FALSE(result.has_value());
}

TEST(SafeMathTest, TypicalPriceTableDimensions) {
    // Realistic price table: 50 moneyness × 20 maturity × 25 vol × 3 rate
    std::array<size_t, 4> shape = {50, 20, 25, 3};
    auto result = safe_product(shape);
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(result.value(), 75000);
}

TEST(SafeMathTest, LargePriceTableStillFits) {
    // Larger but still reasonable: 200 × 100 × 100 × 10 = 20M points
    std::array<size_t, 4> shape = {200, 100, 100, 10};
    auto result = safe_product(shape);
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(result.value(), 20000000);  // 20M points
}

TEST(SafeMathTest, ProductSpanStaticExtent) {
    // Fixed-size span allows loop unrolling
    std::array<size_t, 4> shape = {10, 20, 30, 40};
    auto result = safe_product(std::span<const size_t, 4>(shape));
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(result.value(), 10 * 20 * 30 * 40);  // 240,000
}

TEST(SafeMathTest, ProductSpanDynamicExtent) {
    // Dynamic span from vector
    std::vector<size_t> sizes = {5, 6, 7};
    auto result = safe_product(std::span<const size_t>(sizes));
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(result.value(), 5 * 6 * 7);  // 210
}

TEST(SafeMathTest, ProductSpanOverflow) {
    size_t large = 1ULL << 32;
    std::array<size_t, 3> shape = {large, large, large};
    auto result = safe_product(std::span<const size_t, 3>(shape));
    ASSERT_FALSE(result.has_value());
}

}  // namespace
}  // namespace mango
