#include <gtest/gtest.h>
#include <memory>
#include <expected>
#include "src/support/memory/solver_memory_arena.hpp"

namespace mango {
namespace memory {
namespace testing {

class SolverMemoryArenaTest : public ::testing::Test {
protected:
    void SetUp() override {}
    void TearDown() override {}
};

TEST_F(SolverMemoryArenaTest, CanCreateArena) {
    // Test that we can create a SolverMemoryArena with the factory method
    constexpr size_t arena_size = 1024 * 1024; // 1MB arena

    auto arena_result = SolverMemoryArena::create(arena_size);

    // Should succeed
    ASSERT_TRUE(arena_result.has_value()) << "Failed to create arena: " << arena_result.error();

    auto arena = std::move(arena_result.value());
    ASSERT_NE(arena, nullptr);

    // Test basic properties
    EXPECT_EQ(arena->get_stats().total_size, arena_size);
    EXPECT_EQ(arena->get_stats().active_workspace_count, 0);
    EXPECT_EQ(arena->get_stats().used_size, 0);

    // Test resource access
    EXPECT_NE(arena->resource(), nullptr);
}

TEST_F(SolverMemoryArenaTest, CanIncrementAndDecrementActiveCount) {
    constexpr size_t arena_size = 1024 * 1024; // 1MB arena

    auto arena_result = SolverMemoryArena::create(arena_size);
    ASSERT_TRUE(arena_result.has_value());
    auto arena = std::move(arena_result.value());

    // Initially should be 0
    EXPECT_EQ(arena->get_stats().active_workspace_count, 0);

    // Increment active count
    arena->increment_active();
    EXPECT_EQ(arena->get_stats().active_workspace_count, 1);

    // Increment again
    arena->increment_active();
    EXPECT_EQ(arena->get_stats().active_workspace_count, 2);

    // Decrement
    arena->decrement_active();
    EXPECT_EQ(arena->get_stats().active_workspace_count, 1);

    // Decrement again
    arena->decrement_active();
    EXPECT_EQ(arena->get_stats().active_workspace_count, 0);
}

TEST_F(SolverMemoryArenaTest, TryResetFailsWhenWorkspacesActive) {
    constexpr size_t arena_size = 1024 * 1024; // 1MB arena

    auto arena_result = SolverMemoryArena::create(arena_size);
    ASSERT_TRUE(arena_result.has_value());
    auto arena = std::move(arena_result.value());

    // Increment active count
    arena->increment_active();

    // Try reset should fail when workspaces are active
    auto reset_result = arena->try_reset();
    EXPECT_FALSE(reset_result.has_value());
    EXPECT_EQ(reset_result.error(), "Cannot reset: active workspaces exist");
}

TEST_F(SolverMemoryArenaTest, TryResetSucceedsWhenNoWorkspacesActive) {
    constexpr size_t arena_size = 1024 * 1024; // 1MB arena

    auto arena_result = SolverMemoryArena::create(arena_size);
    ASSERT_TRUE(arena_result.has_value());
    auto arena = std::move(arena_result.value());

    // Reset should succeed when no workspaces are active
    auto reset_result = arena->try_reset();
    EXPECT_TRUE(reset_result.has_value());
}

TEST_F(SolverMemoryArenaTest, TryResetAfterAllWorkspacesInactive) {
    constexpr size_t arena_size = 1024 * 1024; // 1MB arena

    auto arena_result = SolverMemoryArena::create(arena_size);
    ASSERT_TRUE(arena_result.has_value());
    auto arena = std::move(arena_result.value());

    // Activate and deactivate workspaces
    arena->increment_active();
    arena->increment_active();
    EXPECT_EQ(arena->get_stats().active_workspace_count, 2);

    // Deactivate all
    arena->decrement_active();
    arena->decrement_active();
    EXPECT_EQ(arena->get_stats().active_workspace_count, 0);

    // Reset should succeed now
    auto reset_result = arena->try_reset();
    EXPECT_TRUE(reset_result.has_value());
}

}  // namespace testing
}  // namespace memory
}  // namespace mango