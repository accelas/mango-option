#include <gtest/gtest.h>
#include "src/option/price_table_builder.hpp"

TEST(PriceTableResultTest, DefaultConstruction) {
    mango::PriceTableResult<4> result;
    EXPECT_EQ(result.surface, nullptr);
    EXPECT_EQ(result.n_pde_solves, 0);
    EXPECT_EQ(result.precompute_time_seconds, 0.0);
    EXPECT_EQ(result.fitting_stats.max_residual_overall, 0.0);
}

TEST(PriceTableResultTest, FieldAssignment) {
    mango::PriceTableResult<4> result;
    result.n_pde_solves = 200;
    result.precompute_time_seconds = 5.5;
    result.fitting_stats.max_residual_overall = 1e-6;

    EXPECT_EQ(result.n_pde_solves, 200);
    EXPECT_EQ(result.precompute_time_seconds, 5.5);
    EXPECT_EQ(result.fitting_stats.max_residual_overall, 1e-6);
}
