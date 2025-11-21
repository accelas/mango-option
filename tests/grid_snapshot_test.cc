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

TEST(GridSnapshotTest, RecordAndRetrieve) {
    auto grid_spec = mango::GridSpec<double>::uniform(0.0, 1.0, 11).value();
    auto time_domain = mango::TimeDomain::from_n_steps(0.0, 1.0, 10);
    std::vector<double> snapshot_times = {0.0, 0.5, 1.0};

    auto grid = mango::Grid<double>::create(grid_spec, time_domain, snapshot_times).value();

    // Record snapshots at different states
    std::vector<double> state0(11, 1.0);  // All 1.0
    std::vector<double> state5(11, 2.0);  // All 2.0
    std::vector<double> state10(11, 3.0); // All 3.0

    EXPECT_TRUE(grid->should_record(0));
    grid->record(0, state0);

    EXPECT_TRUE(grid->should_record(5));
    grid->record(5, state5);

    EXPECT_TRUE(grid->should_record(10));
    grid->record(10, state10);

    // Retrieve and verify
    auto snap0 = grid->at(0);
    EXPECT_EQ(snap0.size(), 11);
    EXPECT_DOUBLE_EQ(snap0[0], 1.0);

    auto snap1 = grid->at(1);
    EXPECT_DOUBLE_EQ(snap1[0], 2.0);

    auto snap2 = grid->at(2);
    EXPECT_DOUBLE_EQ(snap2[0], 3.0);
}

TEST(GridSnapshotTest, ShouldRecordOnlyRequestedStates) {
    auto grid_spec = mango::GridSpec<double>::uniform(0.0, 1.0, 11).value();
    auto time_domain = mango::TimeDomain::from_n_steps(0.0, 1.0, 10);
    std::vector<double> snapshot_times = {0.5};  // Only middle

    auto grid = mango::Grid<double>::create(grid_spec, time_domain, snapshot_times).value();

    EXPECT_FALSE(grid->should_record(0));
    EXPECT_TRUE(grid->should_record(5));
    EXPECT_FALSE(grid->should_record(10));
}
