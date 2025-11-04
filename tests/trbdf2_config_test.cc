#include "src/cpp/trbdf2_config.hpp"
#include <gtest/gtest.h>
#include <cmath>

TEST(TRBDF2ConfigTest, DefaultValues) {
    mango::TRBDF2Config config;

    EXPECT_EQ(config.max_iter, 100);
    EXPECT_DOUBLE_EQ(config.tolerance, 1e-6);

    // γ = 2 - √2 ≈ 0.5857864376269049
    EXPECT_NEAR(config.gamma, 2.0 - std::sqrt(2.0), 1e-10);
}

TEST(TRBDF2ConfigTest, StageWeights) {
    mango::TRBDF2Config config;

    // Stage 1 weight: γ·dt / 2
    // Stage 2 weight (Ascher, Ruuth, Wetton 1995): (1-γ)·dt / (2-γ)

    double dt = 0.01;
    double w1 = config.stage1_weight(dt);
    double w2 = config.stage2_weight(dt);

    double gamma = 2.0 - std::sqrt(2.0);
    EXPECT_NEAR(w1, gamma * dt / 2.0, 1e-12);
    EXPECT_NEAR(w2, (1.0 - gamma) * dt / (2.0 - gamma), 1e-12);
}
