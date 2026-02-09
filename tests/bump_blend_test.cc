// SPDX-License-Identifier: MIT
#include <gtest/gtest.h>
#include "bump_blend.hpp"
#include <cmath>

using namespace mango;

TEST(BumpBlendTest, BoundaryValues) {
    EXPECT_NEAR(bump_blend_weight(0.0), 0.0, 1e-10);
    EXPECT_NEAR(bump_blend_weight(1.0), 1.0, 1e-10);
}

TEST(BumpBlendTest, SymmetryAtMidpoint) {
    EXPECT_NEAR(bump_blend_weight(0.5), 0.5, 1e-6);
}

TEST(BumpBlendTest, Monotonicity) {
    for (int i = 0; i < 100; ++i) {
        double t1 = i / 100.0;
        double t2 = (i + 1) / 100.0;
        EXPECT_LE(bump_blend_weight(t1), bump_blend_weight(t2) + 1e-15)
            << "Non-monotone at t=" << t1;
    }
}

TEST(BumpBlendTest, PartitionOfUnity) {
    for (int i = 0; i <= 100; ++i) {
        double t = i / 100.0;
        double w_right = bump_blend_weight(t);
        double w_left = 1.0 - w_right;
        EXPECT_NEAR(w_left + w_right, 1.0, 1e-15);
    }
}

TEST(BumpBlendTest, OverlapWeightRight) {
    EXPECT_NEAR(overlap_weight_right(10.0, 10.0, 20.0), 0.0, 1e-10);
    EXPECT_NEAR(overlap_weight_right(20.0, 10.0, 20.0), 1.0, 1e-10);
    EXPECT_NEAR(overlap_weight_right(15.0, 10.0, 20.0), 0.5, 1e-6);
    EXPECT_NEAR(overlap_weight_right(5.0, 10.0, 20.0), 0.0, 1e-10);
    EXPECT_NEAR(overlap_weight_right(25.0, 10.0, 20.0), 1.0, 1e-10);
}
