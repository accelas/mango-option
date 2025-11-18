#include <gtest/gtest.h>
#include "src/pde/core/pde_workspace_pmr.hpp"
#include <memory_resource>

namespace mango {
namespace {

TEST(PDEWorkspacePMRTest, FactoryCreatesWorkspace) {
    std::pmr::synchronized_pool_resource pool;

    auto grid_spec = GridSpec<double>::uniform(0.0, 1.0, 101);
    ASSERT_TRUE(grid_spec.has_value());

    auto workspace = PDEWorkspace::create(grid_spec.value(), &pool);
    ASSERT_TRUE(workspace.has_value());

    auto ws = workspace.value();
    EXPECT_EQ(ws->logical_size(), 101);
    EXPECT_EQ(ws->padded_size(), 104);  // Rounded to SIMD_WIDTH=8
}

TEST(PDEWorkspacePMRTest, AccessorsReturnPaddedSpans) {
    std::pmr::synchronized_pool_resource pool;
    auto grid_spec = GridSpec<double>::uniform(0.0, 1.0, 101);
    auto ws = PDEWorkspace::create(grid_spec.value(), &pool).value();

    auto u_current = ws->u_current();
    EXPECT_EQ(u_current.size(), 104);  // Padded size

    // Check we can write to all elements
    for (size_t i = 0; i < u_current.size(); ++i) {
        u_current[i] = static_cast<double>(i);
    }
}

TEST(PDEWorkspacePMRTest, GridAccessReturnsCorrectData) {
    std::pmr::synchronized_pool_resource pool;
    auto grid_spec = GridSpec<double>::uniform(0.0, 1.0, 101);
    auto ws = PDEWorkspace::create(grid_spec.value(), &pool).value();

    auto grid = ws->grid();
    EXPECT_EQ(grid.size(), 104);  // Padded
    EXPECT_NEAR(grid[0], 0.0, 1e-14);
    EXPECT_NEAR(grid[100], 1.0, 1e-14);
}

TEST(PDEWorkspacePMRTest, NewtonArraysAccessible) {
    std::pmr::synchronized_pool_resource pool;
    auto grid_spec = GridSpec<double>::uniform(0.0, 1.0, 101);
    auto ws = PDEWorkspace::create(grid_spec.value(), &pool).value();

    // Test Newton array access
    auto jac_diag = ws->jacobian_diag();
    auto jac_upper = ws->jacobian_upper();
    auto jac_lower = ws->jacobian_lower();
    auto residual = ws->residual();
    auto delta_u = ws->delta_u();

    EXPECT_EQ(jac_diag.size(), 104);
    EXPECT_EQ(jac_upper.size(), 104);
    EXPECT_EQ(jac_lower.size(), 104);
    EXPECT_EQ(residual.size(), 104);
    EXPECT_EQ(delta_u.size(), 104);
}

}  // namespace
}  // namespace mango
