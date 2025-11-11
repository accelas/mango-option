#include "src/pde/core/time_domain.hpp"
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
