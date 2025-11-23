#include <gtest/gtest.h>
#include "src/option/iv_result.hpp"

using namespace mango;

TEST(IVSuccessTest, BasicConstruction) {
    IVSuccess success{
        .implied_vol = 0.25,
        .iterations = 12,
        .final_error = 1e-8
    };

    EXPECT_DOUBLE_EQ(success.implied_vol, 0.25);
    EXPECT_EQ(success.iterations, 12);
    EXPECT_DOUBLE_EQ(success.final_error, 1e-8);
    EXPECT_FALSE(success.vega.has_value());
}

TEST(IVSuccessTest, ConstructionWithVega) {
    IVSuccess success{
        .implied_vol = 0.30,
        .iterations = 8,
        .final_error = 5e-9,
        .vega = 45.3
    };

    EXPECT_DOUBLE_EQ(success.implied_vol, 0.30);
    EXPECT_EQ(success.iterations, 8);
    EXPECT_DOUBLE_EQ(success.final_error, 5e-9);
    ASSERT_TRUE(success.vega.has_value());
    EXPECT_DOUBLE_EQ(*success.vega, 45.3);
}

TEST(IVSuccessTest, ZeroIterations) {
    // Edge case: convergence on first iteration
    IVSuccess success{
        .implied_vol = 0.20,
        .iterations = 0,
        .final_error = 0.0
    };

    EXPECT_DOUBLE_EQ(success.implied_vol, 0.20);
    EXPECT_EQ(success.iterations, 0);
    EXPECT_DOUBLE_EQ(success.final_error, 0.0);
}

TEST(IVSuccessTest, HighIterations) {
    // Edge case: near max iterations
    IVSuccess success{
        .implied_vol = 0.15,
        .iterations = 99,
        .final_error = 1e-6
    };

    EXPECT_DOUBLE_EQ(success.implied_vol, 0.15);
    EXPECT_EQ(success.iterations, 99);
    EXPECT_DOUBLE_EQ(success.final_error, 1e-6);
}

TEST(IVSuccessTest, VolatilityRange) {
    // Test various realistic volatility values
    IVSuccess low_vol{.implied_vol = 0.05, .iterations = 10, .final_error = 1e-7};
    IVSuccess med_vol{.implied_vol = 0.30, .iterations = 12, .final_error = 1e-8};
    IVSuccess high_vol{.implied_vol = 2.00, .iterations = 15, .final_error = 1e-7};

    EXPECT_DOUBLE_EQ(low_vol.implied_vol, 0.05);
    EXPECT_DOUBLE_EQ(med_vol.implied_vol, 0.30);
    EXPECT_DOUBLE_EQ(high_vol.implied_vol, 2.00);
}
