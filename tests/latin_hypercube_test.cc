// SPDX-License-Identifier: MIT
#include <gtest/gtest.h>
#include "mango/math/latin_hypercube.hpp"
#include <algorithm>
#include <set>

namespace mango {
namespace {

TEST(LatinHypercubeTest, GeneratesCorrectSize) {
    auto samples = latin_hypercube_4d(64, 42);  // 64 samples, seed=42
    EXPECT_EQ(samples.size(), 64);
}

TEST(LatinHypercubeTest, AllValuesInUnitInterval) {
    auto samples = latin_hypercube_4d(100, 123);
    for (const auto& s : samples) {
        for (size_t d = 0; d < 4; ++d) {
            EXPECT_GE(s[d], 0.0);
            EXPECT_LE(s[d], 1.0);
        }
    }
}

TEST(LatinHypercubeTest, EachBinOccupiedOnce) {
    size_t n = 50;
    auto samples = latin_hypercube_4d(n, 456);

    // For each dimension, verify one sample per bin
    for (size_t d = 0; d < 4; ++d) {
        std::set<size_t> bins;
        for (const auto& s : samples) {
            size_t bin = static_cast<size_t>(s[d] * n);
            bin = std::min(bin, n - 1);  // Handle edge case
            bins.insert(bin);
        }
        EXPECT_EQ(bins.size(), n) << "Dimension " << d << " has repeated bins";
    }
}

TEST(LatinHypercubeTest, DeterministicWithSeed) {
    auto samples1 = latin_hypercube_4d(32, 999);
    auto samples2 = latin_hypercube_4d(32, 999);

    ASSERT_EQ(samples1.size(), samples2.size());
    for (size_t i = 0; i < samples1.size(); ++i) {
        for (size_t d = 0; d < 4; ++d) {
            EXPECT_DOUBLE_EQ(samples1[i][d], samples2[i][d]);
        }
    }
}

TEST(LatinHypercubeTest, DifferentSeedsDifferentSamples) {
    auto samples1 = latin_hypercube_4d(32, 111);
    auto samples2 = latin_hypercube_4d(32, 222);

    // At least some samples should differ
    bool any_different = false;
    for (size_t i = 0; i < samples1.size(); ++i) {
        if (samples1[i][0] != samples2[i][0]) {
            any_different = true;
            break;
        }
    }
    EXPECT_TRUE(any_different);
}

TEST(LatinHypercubeTest, ScaleToCustomBounds) {
    auto samples = latin_hypercube_4d(20, 789);

    // Scale to custom bounds: moneyness [0.8, 1.2], tau [0.1, 2.0], sigma [0.1, 0.5], rate [-0.02, 0.10]
    std::array<std::pair<double, double>, 4> bounds = {{
        {0.8, 1.2},    // moneyness
        {0.1, 2.0},    // tau
        {0.1, 0.5},    // sigma
        {-0.02, 0.10}  // rate
    }};

    auto scaled = scale_lhs_samples(samples, bounds);

    EXPECT_EQ(scaled.size(), samples.size());

    for (const auto& s : scaled) {
        EXPECT_GE(s[0], 0.8);
        EXPECT_LE(s[0], 1.2);
        EXPECT_GE(s[1], 0.1);
        EXPECT_LE(s[1], 2.0);
        EXPECT_GE(s[2], 0.1);
        EXPECT_LE(s[2], 0.5);
        EXPECT_GE(s[3], -0.02);
        EXPECT_LE(s[3], 0.10);
    }
}

// ===========================================================================
// Regression tests for edge cases found during code review
// ===========================================================================

// Regression: Ensure n=0 and n=1 are handled without crashes
TEST(LatinHypercubeTest, DegenerateInputs) {
    // n=0 produces empty vector
    auto s0 = latin_hypercube_4d(0, 42);
    EXPECT_TRUE(s0.empty());

    // n=1 produces single sample in [0,1]^4
    auto s1 = latin_hypercube_4d(1, 42);
    ASSERT_EQ(s1.size(), 1);
    for (size_t d = 0; d < 4; ++d) {
        EXPECT_GE(s1[0][d], 0.0) << "Dimension " << d;
        EXPECT_LE(s1[0][d], 1.0) << "Dimension " << d;
    }
}

// Regression: Zero-range bounds produce constant values
TEST(LatinHypercubeTest, ZeroRangeBounds) {
    auto samples = latin_hypercube_4d(10, 42);
    std::array<std::pair<double, double>, 4> zero_bounds = {{
        {1.0, 1.0}, {2.0, 2.0}, {3.0, 3.0}, {4.0, 4.0}
    }};

    auto scaled = scale_lhs_samples(samples, zero_bounds);
    for (const auto& s : scaled) {
        EXPECT_DOUBLE_EQ(s[0], 1.0);
        EXPECT_DOUBLE_EQ(s[1], 2.0);
        EXPECT_DOUBLE_EQ(s[2], 3.0);
        EXPECT_DOUBLE_EQ(s[3], 4.0);
    }
}

}  // namespace
}  // namespace mango
