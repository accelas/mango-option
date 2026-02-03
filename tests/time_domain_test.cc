// SPDX-License-Identifier: MIT
#include "mango/pde/core/time_domain.hpp"
#include <gtest/gtest.h>

TEST(TimeDomainTest, BasicConfiguration) {
    mango::TimeDomain domain(0.0, 1.0, 0.01);  // t_start, t_end, dt

    EXPECT_DOUBLE_EQ(domain.t_start(), 0.0);
    EXPECT_DOUBLE_EQ(domain.t_end(), 1.0);
    EXPECT_DOUBLE_EQ(domain.dt(), 0.01);
    EXPECT_EQ(domain.n_steps(), 100);  // (1.0 - 0.0) / 0.01
}

TEST(TimeDomainTest, TimePointGeneration) {
    mango::TimeDomain domain(0.0, 1.0, 0.25);

    auto times = domain.time_points();
    EXPECT_EQ(times.size(), 5);  // 0.0, 0.25, 0.5, 0.75, 1.0

    EXPECT_DOUBLE_EQ(times[0], 0.0);
    EXPECT_DOUBLE_EQ(times[2], 0.5);
    EXPECT_DOUBLE_EQ(times[4], 1.0);
}

// ===========================================================================
// Tests for non-uniform time grids with mandatory points
// ===========================================================================

TEST(TimeDomainTest, MandatoryTimePoints) {
    auto td = mango::TimeDomain::with_mandatory_points(0.0, 1.0, 0.25, {0.3});
    auto pts = td.time_points();
    EXPECT_DOUBLE_EQ(pts.front(), 0.0);
    EXPECT_DOUBLE_EQ(pts.back(), 1.0);
    bool found = false;
    for (double p : pts) {
        if (std::abs(p - 0.3) < 1e-14) found = true;
    }
    EXPECT_TRUE(found) << "Mandatory point 0.3 not found in time points";
    for (size_t i = 1; i < pts.size(); ++i) {
        EXPECT_LE(pts[i] - pts[i-1], 0.25 + 1e-10);
    }
    for (size_t i = 1; i < pts.size(); ++i) {
        EXPECT_GT(pts[i], pts[i-1]);
    }
}

TEST(TimeDomainTest, MandatoryTimePointsMultiple) {
    auto td = mango::TimeDomain::with_mandatory_points(0.0, 1.0, 0.5, {0.2, 0.7});
    auto pts = td.time_points();
    auto contains = [&](double v) {
        for (double p : pts) if (std::abs(p - v) < 1e-14) return true;
        return false;
    };
    EXPECT_TRUE(contains(0.2));
    EXPECT_TRUE(contains(0.7));
}

TEST(TimeDomainTest, MandatoryTimePointsEmptyFallback) {
    auto td = mango::TimeDomain::with_mandatory_points(0.0, 1.0, 0.25, {});
    EXPECT_EQ(td.n_steps(), 4u);
    EXPECT_NEAR(td.dt(), 0.25, 1e-14);
}

TEST(TimeDomainTest, MandatoryPointsAtBoundariesIgnored) {
    auto td = mango::TimeDomain::with_mandatory_points(0.0, 1.0, 0.25, {0.0, 1.0});
    EXPECT_EQ(td.n_steps(), 4u);
}

TEST(TimeDomainTest, MandatoryPointsOutOfRangeIgnored) {
    auto td = mango::TimeDomain::with_mandatory_points(0.0, 1.0, 0.25, {-0.5, 1.5});
    EXPECT_EQ(td.n_steps(), 4u);
}
