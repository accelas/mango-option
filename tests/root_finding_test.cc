#include "src/math/root_finding.hpp"
#include <gtest/gtest.h>

TEST(RootFindingConfigTest, DefaultValues) {
    mango::RootFindingConfig config;

    EXPECT_EQ(config.max_iter, 100);
    EXPECT_DOUBLE_EQ(config.tolerance, 1e-6);
    EXPECT_DOUBLE_EQ(config.jacobian_fd_epsilon, 1e-7);
    EXPECT_DOUBLE_EQ(config.brent_tol_abs, 1e-6);
}

TEST(RootFindingConfigTest, CustomValues) {
    mango::RootFindingConfig config{
        .max_iter = 50,
        .tolerance = 1e-8,
        .jacobian_fd_epsilon = 1e-9,
        .brent_tol_abs = 1e-8
    };

    EXPECT_EQ(config.max_iter, 50);
    EXPECT_DOUBLE_EQ(config.tolerance, 1e-8);
}
