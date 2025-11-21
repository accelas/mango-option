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

TEST(GridSnapshotTest, TimeToIndexConversion) {
    auto grid_spec = mango::GridSpec<double>::uniform(0.0, 1.0, 11).value();
    auto time_domain = mango::TimeDomain::from_n_steps(0.0, 1.0, 10);  // dt = 0.1

    // Request snapshots at t=0.0, t=0.45, t=1.0
    std::vector<double> snapshot_times = {0.0, 0.45, 1.0};

    auto grid_result = mango::Grid<double>::create(grid_spec, time_domain, snapshot_times);
    ASSERT_TRUE(grid_result.has_value());
    auto grid = grid_result.value();

    // Should snap to nearest: 0.0→state0, 0.45→state5, 1.0→state10
    EXPECT_EQ(grid->num_snapshots(), 3);
    auto times = grid->snapshot_times();
    EXPECT_NEAR(times[0], 0.0, 1e-10);
    EXPECT_NEAR(times[1], 0.5, 1e-10);  // Snapped to nearest
    EXPECT_NEAR(times[2], 1.0, 1e-10);
}

TEST(GridSnapshotTest, OutOfRangeTimeRejected) {
    auto grid_spec = mango::GridSpec<double>::uniform(0.0, 1.0, 11).value();
    auto time_domain = mango::TimeDomain::from_n_steps(0.0, 1.0, 10);

    std::vector<double> bad_times = {-0.1};  // Negative time
    auto grid_result = mango::Grid<double>::create(grid_spec, time_domain, bad_times);

    EXPECT_FALSE(grid_result.has_value());
    EXPECT_TRUE(grid_result.error().find("out of range") != std::string::npos);
}
