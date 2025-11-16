#include <gtest/gtest.h>
#include <memory>
#include <expected>
#include <memory_resource>
#include <future>
#include <thread>
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

    {
        auto token1 = SolverMemoryArena::ActiveWorkspaceToken(arena);
        EXPECT_EQ(arena->get_stats().active_workspace_count, 1);

        auto token2 = SolverMemoryArena::ActiveWorkspaceToken(arena);
        EXPECT_EQ(arena->get_stats().active_workspace_count, 2);

        token2.reset();
        EXPECT_EQ(arena->get_stats().active_workspace_count, 1);
    }

    EXPECT_EQ(arena->get_stats().active_workspace_count, 0);
}

TEST_F(SolverMemoryArenaTest, TryResetFailsWhenWorkspacesActive) {
    constexpr size_t arena_size = 1024 * 1024; // 1MB arena

    auto arena_result = SolverMemoryArena::create(arena_size);
    ASSERT_TRUE(arena_result.has_value());
    auto arena = std::move(arena_result.value());

    auto token = SolverMemoryArena::ActiveWorkspaceToken(arena);

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

    {
        auto token1 = SolverMemoryArena::ActiveWorkspaceToken(arena);
        auto token2 = SolverMemoryArena::ActiveWorkspaceToken(arena);
        EXPECT_EQ(arena->get_stats().active_workspace_count, 2);
    }

    EXPECT_EQ(arena->get_stats().active_workspace_count, 0);

    auto reset_result = arena->try_reset();
    EXPECT_TRUE(reset_result.has_value());
}

TEST_F(SolverMemoryArenaTest, ResourceCanBeUsedForPmrAllocationsAndReset) {
    constexpr size_t arena_size = 64 * 1024;  // 64KB arena for test

    auto arena_result = SolverMemoryArena::create(arena_size);
    ASSERT_TRUE(arena_result.has_value());
    auto arena = std::move(arena_result.value());

    std::pmr::vector<int> numbers(arena->resource());
    numbers.resize(256, 42);
    EXPECT_EQ(numbers.size(), 256);

    // Capture used size before reset. It should be >0 once we allocate.
    auto before_reset = arena->get_stats();
    EXPECT_GT(before_reset.used_size, 0u);

    // Release workspace token to allow reset
    auto reset_result = arena->try_reset();
    EXPECT_TRUE(reset_result.has_value());
    auto after_reset = arena->get_stats();
    EXPECT_EQ(after_reset.used_size, 0u);
}

TEST_F(SolverMemoryArenaTest, TryResetFailsWhileTokenAliveInAnotherThread) {
    constexpr size_t arena_size = 1024 * 1024; // 1MB arena

    auto arena_result = SolverMemoryArena::create(arena_size);
    ASSERT_TRUE(arena_result.has_value());
    auto arena = std::move(arena_result.value());

    std::promise<void> token_ready;
    std::promise<void> release_token;
    auto release_future = release_token.get_future();

    std::thread worker([arena, &token_ready, release_future = std::move(release_future)]() mutable {
        SolverMemoryArena::ActiveWorkspaceToken token(arena);
        token_ready.set_value();
        release_future.wait();
    });

    token_ready.get_future().wait();

    auto reset_result = arena->try_reset();
    EXPECT_FALSE(reset_result.has_value());
    EXPECT_EQ(reset_result.error(), "Cannot reset: active workspaces exist");

    release_token.set_value();
    worker.join();

    auto reset_after = arena->try_reset();
    EXPECT_TRUE(reset_after.has_value());
}

}  // namespace testing
}  // namespace memory
}  // namespace mango
