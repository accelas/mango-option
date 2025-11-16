#include "solver_memory_arena.hpp"
#include <vector>
#include <algorithm>

namespace mango {
namespace memory {

std::expected<std::shared_ptr<SolverMemoryArena>, std::string> SolverMemoryArena::create(size_t arena_size) {
    try {
        MANGO_TRACE_ALGO_START(MODULE_MEMORY, arena_size, 0, 0);

        auto arena = std::shared_ptr<SolverMemoryArena>(new SolverMemoryArena(arena_size));

        MANGO_TRACE_ALGO_COMPLETE(MODULE_MEMORY, 1, 0);

        return arena;
    } catch (const std::exception& e) {
        return std::unexpected(std::string("Failed to create SolverMemoryArena: ") + e.what());
    } catch (...) {
        return std::unexpected("Failed to create SolverMemoryArena: unknown exception");
    }
}

SolverMemoryArena::SolverMemoryArena(size_t arena_size)
    : active_workspace_count_(0),
      arena_size_(arena_size) {

    // Create the three-level PMR hierarchy: pool → arena → tracker

    // Level 1: Create buffer storage for the arena
    arena_buffer_.resize(arena_size);

    // Level 2: Create the underlying memory resource (UnifiedMemoryResource)
    memory_resource_ = std::make_unique<UnifiedMemoryResource>(arena_size);

    // Level 3: Create the monotonic buffer resource (arena) on top of the buffer
    arena_resource_ = std::make_unique<std::pmr::monotonic_buffer_resource>(
        arena_buffer_.data(), arena_size, memory_resource_->pmr_resource());

    // Level 1: Pool options for memory resource
    pool_options_ = std::make_unique<std::pmr::pool_options>();
    pool_options_->max_blocks_per_chunk = 64;
    pool_options_->largest_required_pool_block = 1024;
}

std::expected<void, std::string> SolverMemoryArena::try_reset() {
    std::lock_guard<std::mutex> lock(mutex_);

    MANGO_TRACE_ALGO_PROGRESS(MODULE_MEMORY, active_workspace_count_, 0, 0);

    if (active_workspace_count_ > 0) {
        return std::unexpected("Cannot reset: active workspaces exist");
    }

    // Reset the arena
    arena_resource_->release();

    MANGO_TRACE_ALGO_COMPLETE(MODULE_MEMORY, 1, 0);

    return {};
}

void SolverMemoryArena::increment_active() {
    std::lock_guard<std::mutex> lock(mutex_);
    ++active_workspace_count_;
    MANGO_TRACE_ALGO_PROGRESS(MODULE_MEMORY, active_workspace_count_, 0, 1);
}

void SolverMemoryArena::decrement_active() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (active_workspace_count_ > 0) {
        --active_workspace_count_;
        MANGO_TRACE_ALGO_PROGRESS(MODULE_MEMORY, active_workspace_count_, 0, 0);
    }
}

std::pmr::memory_resource* SolverMemoryArena::resource() {
    return arena_resource_.get();
}

SolverMemoryArenaStats SolverMemoryArena::get_stats() const {
    std::lock_guard<std::mutex> lock(mutex_);

    SolverMemoryArenaStats stats;
    stats.total_size = arena_size_;
    stats.active_workspace_count = active_workspace_count_;
    stats.used_size = memory_resource_->bytes_allocated();

    return stats;
}

}  // namespace memory
}  // namespace mango