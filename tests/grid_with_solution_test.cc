#include "src/pde/core/grid_with_solution.hpp"
#include <gtest/gtest.h>

namespace mango {
namespace {

TEST(GridWithSolutionTest, CreateUniformGrid) {
    // Create uniform grid specification
    auto grid_spec = GridSpec<double>::uniform(0.0, 1.0, 101);
    ASSERT_TRUE(grid_spec.has_value());

    // Create time domain
    TimeDomain time = TimeDomain::from_n_steps(0.0, 1.0, 1000);

    // Create GridWithSolution
    auto grid_result = GridWithSolution<double>::create(grid_spec.value(), time);
    ASSERT_TRUE(grid_result.has_value());

    auto grid = grid_result.value();

    // Verify spatial grid
    EXPECT_EQ(grid->n_space(), 101);
    auto x = grid->x();
    EXPECT_EQ(x.size(), 101);
    EXPECT_DOUBLE_EQ(x[0], 0.0);
    EXPECT_DOUBLE_EQ(x[100], 1.0);

    // Verify time domain
    EXPECT_DOUBLE_EQ(grid->time().t_start(), 0.0);
    EXPECT_DOUBLE_EQ(grid->time().t_end(), 1.0);
    EXPECT_EQ(grid->time().n_steps(), 1000);
    EXPECT_DOUBLE_EQ(grid->dt(), 0.001);

    // Verify solution storage
    auto solution = grid->solution();
    auto solution_prev = grid->solution_prev();
    EXPECT_EQ(solution.size(), 101);
    EXPECT_EQ(solution_prev.size(), 101);

    // Verify solution buffers are independent
    solution[0] = 42.0;
    solution_prev[0] = 84.0;
    EXPECT_DOUBLE_EQ(solution[0], 42.0);
    EXPECT_DOUBLE_EQ(solution_prev[0], 84.0);
}

TEST(GridWithSolutionTest, CreateNonUniformGrid) {
    // Create sinh-spaced grid
    auto grid_spec = GridSpec<double>::sinh_spaced(-3.0, 3.0, 201, 2.0);
    ASSERT_TRUE(grid_spec.has_value());

    TimeDomain time = TimeDomain::from_n_steps(0.0, 2.0, 500);

    auto grid_result = GridWithSolution<double>::create(grid_spec.value(), time);
    ASSERT_TRUE(grid_result.has_value());

    auto grid = grid_result.value();

    // Verify spatial grid
    EXPECT_EQ(grid->n_space(), 201);
    auto x = grid->x();
    EXPECT_EQ(x.size(), 201);

    // Verify grid spacing object
    const auto& spacing = grid->spacing();
    EXPECT_FALSE(spacing.is_uniform());
}

TEST(GridWithSolutionTest, GridSpacingReference) {
    auto grid_spec = GridSpec<double>::uniform(0.0, 1.0, 101);
    ASSERT_TRUE(grid_spec.has_value());

    TimeDomain time = TimeDomain::from_n_steps(0.0, 1.0, 1000);

    auto grid_result = GridWithSolution<double>::create(grid_spec.value(), time);
    ASSERT_TRUE(grid_result.has_value());

    auto grid = grid_result.value();

    // Verify we can safely use reference (Grid outlives solver)
    const GridSpacing<double>& spacing_ref = grid->spacing();
    EXPECT_TRUE(spacing_ref.is_uniform());
    EXPECT_DOUBLE_EQ(spacing_ref.spacing(), 0.01);
}

TEST(GridWithSolutionTest, SolutionBufferModification) {
    auto grid_spec = GridSpec<double>::uniform(0.0, 1.0, 101);
    ASSERT_TRUE(grid_spec.has_value());

    TimeDomain time = TimeDomain::from_n_steps(0.0, 1.0, 1000);

    auto grid_result = GridWithSolution<double>::create(grid_spec.value(), time);
    ASSERT_TRUE(grid_result.has_value());

    auto grid = grid_result.value();

    // Write to solution buffers
    auto solution = grid->solution();
    auto solution_prev = grid->solution_prev();

    for (size_t i = 0; i < solution.size(); ++i) {
        solution[i] = static_cast<double>(i);
        solution_prev[i] = static_cast<double>(i * 2);
    }

    // Read back and verify
    auto solution_const = static_cast<const GridWithSolution<double>*>(grid.get())->solution();
    auto solution_prev_const = static_cast<const GridWithSolution<double>*>(grid.get())->solution_prev();

    for (size_t i = 0; i < solution_const.size(); ++i) {
        EXPECT_DOUBLE_EQ(solution_const[i], static_cast<double>(i));
        EXPECT_DOUBLE_EQ(solution_prev_const[i], static_cast<double>(i * 2));
    }
}

}  // namespace
}  // namespace mango
