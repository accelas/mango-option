// SPDX-License-Identifier: MIT
#include <gtest/gtest.h>
#include "mango/option/table/adaptive_grid_types.hpp"

namespace mango {
namespace {

TEST(AdaptiveGridParamsTest, DefaultValues) {
    AdaptiveGridParams params;

    EXPECT_DOUBLE_EQ(params.target_iv_error, 2e-5);  // 2 bps (High profile)
    EXPECT_EQ(params.max_iter, 5);
    EXPECT_EQ(params.max_points_per_dim, 160);
    EXPECT_EQ(params.validation_samples, 64);
    EXPECT_DOUBLE_EQ(params.refinement_factor, 1.3);
    EXPECT_EQ(params.lhs_seed, 42);
    EXPECT_DOUBLE_EQ(params.vega_floor, 1e-4);
}

TEST(AdaptiveGridParamsTest, CustomValues) {
    AdaptiveGridParams params;
    params.target_iv_error = 0.001;  // 10 bps
    params.max_iter = 10;
    params.max_points_per_dim = 100;
    params.validation_samples = 128;
    params.refinement_factor = 1.5;
    params.lhs_seed = 12345;
    params.vega_floor = 1e-3;

    EXPECT_DOUBLE_EQ(params.target_iv_error, 0.001);
    EXPECT_EQ(params.max_iter, 10);
    EXPECT_EQ(params.max_points_per_dim, 100);
    EXPECT_EQ(params.validation_samples, 128);
    EXPECT_DOUBLE_EQ(params.refinement_factor, 1.5);
    EXPECT_EQ(params.lhs_seed, 12345);
    EXPECT_DOUBLE_EQ(params.vega_floor, 1e-3);
}

TEST(IterationStatsTest, DefaultConstruction) {
    IterationStats stats;

    EXPECT_EQ(stats.iteration, 0);
    EXPECT_EQ(stats.grid_sizes[0], 0);
    EXPECT_EQ(stats.grid_sizes[1], 0);
    EXPECT_EQ(stats.grid_sizes[2], 0);
    EXPECT_EQ(stats.grid_sizes[3], 0);
    EXPECT_EQ(stats.pde_solves_table, 0);
    EXPECT_EQ(stats.pde_solves_validation, 0);
    EXPECT_DOUBLE_EQ(stats.max_error, 0.0);
    EXPECT_DOUBLE_EQ(stats.avg_error, 0.0);
    EXPECT_EQ(stats.refined_dim, -1);
    EXPECT_DOUBLE_EQ(stats.elapsed_seconds, 0.0);
}

TEST(IterationStatsTest, SetValues) {
    IterationStats stats;
    stats.iteration = 2;
    stats.grid_sizes = {10, 8, 6, 4};
    stats.pde_solves_table = 24;
    stats.pde_solves_validation = 64;
    stats.max_error = 0.0003;
    stats.avg_error = 0.0001;
    stats.refined_dim = 1;
    stats.elapsed_seconds = 5.5;

    EXPECT_EQ(stats.iteration, 2);
    EXPECT_EQ(stats.grid_sizes[0], 10);
    EXPECT_EQ(stats.grid_sizes[1], 8);
    EXPECT_EQ(stats.grid_sizes[2], 6);
    EXPECT_EQ(stats.grid_sizes[3], 4);
    EXPECT_EQ(stats.pde_solves_table, 24);
    EXPECT_EQ(stats.pde_solves_validation, 64);
    EXPECT_DOUBLE_EQ(stats.max_error, 0.0003);
    EXPECT_DOUBLE_EQ(stats.avg_error, 0.0001);
    EXPECT_EQ(stats.refined_dim, 1);
    EXPECT_DOUBLE_EQ(stats.elapsed_seconds, 5.5);
}

TEST(AdaptiveResultTest, DefaultConstruction) {
    AdaptiveResult result;

    EXPECT_EQ(result.surface, nullptr);
    EXPECT_TRUE(result.iterations.empty());
    EXPECT_DOUBLE_EQ(result.achieved_max_error, 0.0);
    EXPECT_DOUBLE_EQ(result.achieved_avg_error, 0.0);
    EXPECT_FALSE(result.target_met);
    EXPECT_EQ(result.total_pde_solves, 0);
}

TEST(AdaptiveResultTest, TargetMetFlag) {
    AdaptiveResult result;
    result.achieved_max_error = 0.0003;
    result.target_met = true;

    EXPECT_TRUE(result.target_met);
    EXPECT_LT(result.achieved_max_error, 0.0005);
}

TEST(AdaptiveResultTest, IterationsVector) {
    AdaptiveResult result;

    IterationStats stats1;
    stats1.iteration = 0;
    stats1.max_error = 0.001;

    IterationStats stats2;
    stats2.iteration = 1;
    stats2.max_error = 0.0005;

    result.iterations.push_back(stats1);
    result.iterations.push_back(stats2);

    EXPECT_EQ(result.iterations.size(), 2);
    EXPECT_DOUBLE_EQ(result.iterations[0].max_error, 0.001);
    EXPECT_DOUBLE_EQ(result.iterations[1].max_error, 0.0005);
}

}  // namespace
}  // namespace mango
