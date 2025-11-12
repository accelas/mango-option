#include "src/pde/memory/pde_workspace.hpp"
#include "src/pde/core/grid.hpp"
#include <gtest/gtest.h>
#include <algorithm>
#include <cstring>

TEST(PDEWorkspaceBatchTest, HasBatchQuery) {
    auto grid_result = mango::GridSpec<>::uniform(0.0, 1.0, 101);
    ASSERT_TRUE(grid_result.has_value());
    auto grid = grid_result.value().generate();

    // Single-contract mode (batch_width = 0)
    mango::PDEWorkspace single_contract(101, grid.span(), 0);
    EXPECT_FALSE(single_contract.has_batch());
    EXPECT_EQ(single_contract.batch_width(), 0);

    // Batch mode (batch_width = 4)
    mango::PDEWorkspace batched(101, grid.span(), 4);
    EXPECT_TRUE(batched.has_batch());
    EXPECT_EQ(batched.batch_width(), 4);
}

TEST(PDEWorkspaceBatchTest, PackScatterRoundTrip) {
    auto grid_result = mango::GridSpec<>::uniform(0.0, 1.0, 101);
    ASSERT_TRUE(grid_result.has_value());
    auto grid = grid_result.value().generate();

    constexpr size_t batch_width = 4;
    mango::PDEWorkspace workspace(101, grid.span(), batch_width);

    // Initialize per-lane SoA buffers with unique test pattern
    for (size_t lane = 0; lane < batch_width; ++lane) {
        auto u_lane = workspace.u_lane(lane);
        for (size_t i = 0; i < u_lane.size(); ++i) {
            u_lane[i] = static_cast<double>(lane * 1000 + i);
        }
    }

    // Save original values for verification
    std::vector<std::vector<double>> original_values(batch_width);
    for (size_t lane = 0; lane < batch_width; ++lane) {
        auto u_lane = workspace.u_lane(lane);
        original_values[lane].assign(u_lane.begin(), u_lane.end());
    }

    // Pack: SoA → AoS
    workspace.pack_to_batch_slice();

    // Zero out the per-lane buffers to verify scatter writes correctly
    for (size_t lane = 0; lane < batch_width; ++lane) {
        auto u_lane = workspace.u_lane(lane);
        std::fill(u_lane.begin(), u_lane.end(), 0.0);
    }

    // Copy batch slice to lu_batch (simulating stencil operation)
    auto u_batch = workspace.batch_slice();
    auto lu_batch = workspace.lu_batch();
    std::copy(u_batch.begin(), u_batch.end(), lu_batch.begin());

    // Scatter: AoS → SoA
    workspace.scatter_from_batch_slice();

    // Verify bitwise identity: lu_lane[i] == original u_lane[i]
    for (size_t lane = 0; lane < batch_width; ++lane) {
        auto lu_lane = workspace.lu_lane(lane);
        const auto& original = original_values[lane];

        ASSERT_EQ(lu_lane.size(), original.size());

        for (size_t i = 0; i < lu_lane.size(); ++i) {
            // Use memcmp for bitwise identity (stricter than EXPECT_DOUBLE_EQ)
            EXPECT_EQ(std::memcmp(&lu_lane[i], &original[i], sizeof(double)), 0)
                << "Mismatch at lane=" << lane << ", i=" << i
                << ": lu_lane[i]=" << lu_lane[i]
                << ", original[i]=" << original[i];
        }
    }
}
