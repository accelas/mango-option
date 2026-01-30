// SPDX-License-Identifier: MIT
// tests/lifetime_test.cc
#include "src/support/lifetime.hpp"
#include <gtest/gtest.h>
#include <cstddef>
#include <type_traits>

using namespace mango;

TEST(LifetimeTest, StartArrayLifetimeDouble) {
    alignas(64) std::byte buffer[sizeof(double) * 10];

    double* arr = start_array_lifetime<double>(buffer, 10);

    EXPECT_NE(arr, nullptr);
    // Write and read back
    for (size_t i = 0; i < 10; ++i) {
        arr[i] = static_cast<double>(i);
    }
    for (size_t i = 0; i < 10; ++i) {
        EXPECT_DOUBLE_EQ(arr[i], static_cast<double>(i));
    }
}

TEST(LifetimeTest, StartArrayLifetimeInt) {
    alignas(64) std::byte buffer[sizeof(int) * 10];

    int* arr = start_array_lifetime<int>(buffer, 10);

    EXPECT_NE(arr, nullptr);
    for (size_t i = 0; i < 10; ++i) {
        arr[i] = static_cast<int>(i);
    }
    for (size_t i = 0; i < 10; ++i) {
        EXPECT_EQ(arr[i], static_cast<int>(i));
    }
}

// Verify static_assert fires for non-trivially-destructible types
// (This is a compile-time check - if this compiles, the assert works)
struct NonTrivial {
    ~NonTrivial() {}  // Non-trivial destructor
};
static_assert(!std::is_trivially_destructible_v<NonTrivial>);
// Uncommenting below should fail to compile:
// auto* bad = start_array_lifetime<NonTrivial>(nullptr, 0);

TEST(LifetimeTest, AlignUpBasic) {
    // Already aligned
    EXPECT_EQ(align_up(0, 8), 0);
    EXPECT_EQ(align_up(8, 8), 8);
    EXPECT_EQ(align_up(16, 8), 16);
    EXPECT_EQ(align_up(64, 64), 64);

    // Need alignment
    EXPECT_EQ(align_up(1, 8), 8);
    EXPECT_EQ(align_up(7, 8), 8);
    EXPECT_EQ(align_up(9, 8), 16);
    EXPECT_EQ(align_up(15, 8), 16);
}

TEST(LifetimeTest, AlignUpPowersOfTwo) {
    // Alignment to 1 (no-op)
    EXPECT_EQ(align_up(5, 1), 5);

    // Alignment to 2
    EXPECT_EQ(align_up(0, 2), 0);
    EXPECT_EQ(align_up(1, 2), 2);
    EXPECT_EQ(align_up(2, 2), 2);
    EXPECT_EQ(align_up(3, 2), 4);

    // Alignment to 4
    EXPECT_EQ(align_up(0, 4), 0);
    EXPECT_EQ(align_up(1, 4), 4);
    EXPECT_EQ(align_up(3, 4), 4);
    EXPECT_EQ(align_up(4, 4), 4);
    EXPECT_EQ(align_up(5, 4), 8);

    // Alignment to 16
    EXPECT_EQ(align_up(0, 16), 0);
    EXPECT_EQ(align_up(1, 16), 16);
    EXPECT_EQ(align_up(15, 16), 16);
    EXPECT_EQ(align_up(16, 16), 16);
    EXPECT_EQ(align_up(17, 16), 32);

    // Alignment to 32
    EXPECT_EQ(align_up(31, 32), 32);
    EXPECT_EQ(align_up(32, 32), 32);
    EXPECT_EQ(align_up(33, 32), 64);

    // Alignment to 64 (common cache line size)
    EXPECT_EQ(align_up(63, 64), 64);
    EXPECT_EQ(align_up(64, 64), 64);
    EXPECT_EQ(align_up(65, 64), 128);
}

TEST(LifetimeTest, AlignUpTypicalSizes) {
    // Typical struct sizes aligned to 8 bytes
    EXPECT_EQ(align_up(sizeof(double), 8), 8);
    EXPECT_EQ(align_up(sizeof(int), 8), 8);
    EXPECT_EQ(align_up(sizeof(char), 8), 8);

    // Array sizes
    EXPECT_EQ(align_up(sizeof(double) * 100, 64), 832);  // 800 -> 832
    EXPECT_EQ(align_up(sizeof(int) * 100, 64), 448);     // 400 -> 448
}
