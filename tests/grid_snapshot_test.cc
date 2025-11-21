#include "src/pde/core/grid.hpp"
#include "src/pde/core/time_domain.hpp"
#include <gtest/gtest.h>
#include <vector>

TEST(GridSnapshotTest, CreateWithoutSnapshots) {
    auto grid_spec = mango::GridSpec<double>::uniform(0.0, 1.0, 11).value();
    auto time_domain = mango::TimeDomain::from_n_steps(0.0, 1.0, 10);

    auto grid_result = mango::Grid<double>::create(grid_spec, time_domain);

    ASSERT_TRUE(grid_result.has_value());
    auto grid = grid_result.value();
    EXPECT_FALSE(grid->has_snapshots());
    EXPECT_EQ(grid->num_snapshots(), 0);
}

TEST(GridSnapshotTest, CreateWithSnapshotTimes) {
    auto grid_spec = mango::GridSpec<double>::uniform(0.0, 1.0, 11).value();
    auto time_domain = mango::TimeDomain::from_n_steps(0.0, 1.0, 10);
    std::vector<double> snapshot_times = {0.0, 0.5, 1.0};

    auto grid_result = mango::Grid<double>::create(grid_spec, time_domain, snapshot_times);

    ASSERT_TRUE(grid_result.has_value());
    auto grid = grid_result.value();
    EXPECT_TRUE(grid->has_snapshots());
    EXPECT_EQ(grid->num_snapshots(), 3);
}
