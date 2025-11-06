#include "src/snapshot.hpp"
#include <gtest/gtest.h>
#include <vector>

TEST(SnapshotTest, StructLayout) {
    // Create test data
    std::vector<double> x = {0.0, 0.5, 1.0};
    std::vector<double> dx = {0.5, 0.5};
    std::vector<double> u = {1.0, 2.0, 3.0};
    std::vector<double> Lu = {0.0, 1.0, 0.0};
    std::vector<double> du = {2.0, 2.0, 2.0};
    std::vector<double> d2u = {0.0, 0.0, 0.0};

    // Create snapshot
    mango::Snapshot snap{
        .time = 0.5,
        .user_index = 42,
        .spatial_grid = std::span{x},
        .dx = std::span{dx},
        .solution = std::span{u},
        .spatial_operator = std::span{Lu},
        .first_derivative = std::span{du},
        .second_derivative = std::span{d2u},
        .problem_params = nullptr
    };

    // Verify fields
    EXPECT_DOUBLE_EQ(snap.time, 0.5);
    EXPECT_EQ(snap.user_index, 42u);
    EXPECT_EQ(snap.spatial_grid.size(), 3u);
    EXPECT_EQ(snap.solution.size(), 3u);
    EXPECT_DOUBLE_EQ(snap.solution[1], 2.0);
    EXPECT_EQ(snap.problem_params, nullptr);
}
