#include <gtest/gtest.h>
#include "src/option/american_solver_workspace.hpp"
#include "src/pde/core/grid.hpp"
#include <memory_resource>

namespace mango {
namespace {

TEST(AmericanSolverWorkspaceTest, CreateWithGridSpec) {
    std::pmr::synchronized_pool_resource pool;

    auto grid_spec = GridSpec<double>::sinh_spaced(-3.0, 3.0, 201, 2.0);
    ASSERT_TRUE(grid_spec.has_value());

    auto workspace = AmericanSolverWorkspace::create(
        grid_spec.value(), 1000, &pool);

    ASSERT_TRUE(workspace.has_value());

    auto ws = workspace.value();
    EXPECT_EQ(ws->n_space(), 201);
    EXPECT_EQ(ws->n_time(), 1000);

    // Check we can access new API components
    auto grid_sol = ws->grid_with_solution();
    ASSERT_NE(grid_sol, nullptr);
    EXPECT_EQ(grid_sol->n_space(), 201);

    auto spans = ws->workspace_spans();
    EXPECT_EQ(spans.u_stage().size(), 201);
}

TEST(AmericanSolverWorkspaceTest, GridSpacingAvailable) {
    std::pmr::synchronized_pool_resource pool;
    auto grid_spec = GridSpec<double>::uniform(0.0, 1.0, 101);
    ASSERT_TRUE(grid_spec.has_value());

    auto workspace = AmericanSolverWorkspace::create(
        grid_spec.value(), 1000, &pool);
    ASSERT_TRUE(workspace.has_value());

    // Create GridSpacing from Grid (NEW API)
    auto grid_view = GridView<double>(workspace.value()->grid_with_solution()->x());
    auto spacing = GridSpacing<double>(grid_view);
    EXPECT_TRUE(spacing.is_uniform());
    EXPECT_NEAR(spacing.spacing(), 0.01, 1e-10);
}

TEST(AmericanSolverWorkspaceTest, NullResourceReturnsError) {
    auto grid_spec = GridSpec<double>::uniform(0.0, 1.0, 101);
    ASSERT_TRUE(grid_spec.has_value());

    auto workspace = AmericanSolverWorkspace::create(
        grid_spec.value(), 1000, nullptr);

    EXPECT_FALSE(workspace.has_value());
    EXPECT_EQ(workspace.error(), "Memory resource cannot be null");
}

TEST(AmericanSolverWorkspaceTest, ZeroTimeStepsReturnsError) {
    std::pmr::synchronized_pool_resource pool;
    auto grid_spec = GridSpec<double>::uniform(0.0, 1.0, 101);
    ASSERT_TRUE(grid_spec.has_value());

    auto workspace = AmericanSolverWorkspace::create(
        grid_spec.value(), 0, &pool);

    EXPECT_FALSE(workspace.has_value());
    EXPECT_EQ(workspace.error(), "n_time must be positive");
}

}  // namespace
}  // namespace mango
