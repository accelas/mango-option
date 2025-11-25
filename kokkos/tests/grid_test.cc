#include <gtest/gtest.h>
#include "kokkos/src/pde/core/grid.hpp"

namespace mango::kokkos::test {

// Global setup/teardown for Kokkos - once per test program
class KokkosEnvironment : public ::testing::Environment {
public:
    void SetUp() override {
        Kokkos::initialize();
    }
    void TearDown() override {
        Kokkos::finalize();
    }
};

// Register the global environment
[[maybe_unused]] static ::testing::Environment* const kokkos_env =
    ::testing::AddGlobalTestEnvironment(new KokkosEnvironment);

class GridTest : public ::testing::Test {
    // No per-test setup/teardown needed for Kokkos
};

TEST_F(GridTest, UniformGridCreation) {
    auto grid = Grid<HostMemSpace>::uniform(-1.0, 1.0, 101);
    ASSERT_TRUE(grid.has_value());
    EXPECT_EQ(grid->n_points(), 101);
    EXPECT_DOUBLE_EQ(grid->x_min(), -1.0);
    EXPECT_DOUBLE_EQ(grid->x_max(), 1.0);
}

TEST_F(GridTest, GridPointsAccessible) {
    auto grid = Grid<HostMemSpace>::uniform(-1.0, 1.0, 11).value();
    auto x = grid.x();
    EXPECT_DOUBLE_EQ(x(0), -1.0);
    EXPECT_DOUBLE_EQ(x(10), 1.0);
    EXPECT_DOUBLE_EQ(x(5), 0.0);  // Midpoint
}

TEST_F(GridTest, SolutionStorageWorks) {
    auto grid = Grid<HostMemSpace>::uniform(0.0, 1.0, 5).value();
    auto u = grid.u_current();

    // Initialize solution
    for (size_t i = 0; i < 5; ++i) {
        u(i) = static_cast<double>(i);
    }

    // Verify
    EXPECT_DOUBLE_EQ(u(0), 0.0);
    EXPECT_DOUBLE_EQ(u(4), 4.0);
}

TEST_F(GridTest, InvalidGridSizeRejected) {
    auto grid = Grid<HostMemSpace>::uniform(0.0, 1.0, 1);  // Too few points
    EXPECT_FALSE(grid.has_value());
}

TEST_F(GridTest, SinhSpacedGridCreation) {
    auto grid = Grid<HostMemSpace>::sinh_spaced(-1.0, 1.0, 101, 2.0);
    ASSERT_TRUE(grid.has_value());
    EXPECT_EQ(grid->n_points(), 101);

    // Verify concentration at center (spacing near center < spacing at edges)
    auto x = grid->x();
    double spacing_center = x(51) - x(50);  // Near x=0
    double spacing_edge = x(1) - x(0);       // Near x=-1
    EXPECT_LT(spacing_center, spacing_edge);
}

TEST_F(GridTest, InvalidAlphaRejected) {
    auto grid = Grid<HostMemSpace>::sinh_spaced(0.0, 1.0, 10, -1.0);
    EXPECT_FALSE(grid.has_value());
}

TEST_F(GridTest, SwapSolutionsWorks) {
    auto grid = Grid<HostMemSpace>::uniform(0.0, 1.0, 5).value();
    auto u_current = grid.u_current();
    auto u_prev = grid.u_prev();

    // Set initial values
    u_current(0) = 1.0;
    u_prev(0) = 2.0;

    grid.swap_solutions();

    // After swap, u_current should have old u_prev value
    EXPECT_DOUBLE_EQ(grid.u_current()(0), 2.0);
    EXPECT_DOUBLE_EQ(grid.u_prev()(0), 1.0);
}

}  // namespace mango::kokkos::test
