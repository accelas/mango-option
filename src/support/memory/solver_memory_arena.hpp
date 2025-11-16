#ifndef MANGO_SUPPORT_MEMORY_SOLVER_MEMORY_ARENA_HPP_
#define MANGO_SUPPORT_MEMORY_SOLVER_MEMORY_ARENA_HPP_

#include <memory>
#include <mutex>
#include <expected>
#include <cstring>
#include <memory_resource>
#include <vector>
#include <atomic>
#include "common/ivcalc_trace.h"
#include "unified_memory_resource.hpp"

namespace mango {
namespace memory {

// Memory module identifier for tracing
#define MODULE_MEMORY 8

/**
 * @brief Statistics for the solver memory arena
 */
struct SolverMemoryArenaStats {
    size_t total_size;              // Total size of the arena
    size_t used_size;               // Currently used size
    size_t active_workspace_count;  // Number of active workspaces
};

/**
 * @brief Memory arena for solver workspaces with PMR hierarchy
 *
 * Provides a thread-safe memory arena for solver workspaces with a three-level
 * PMR hierarchy: pool → arena → tracker. The arena supports reset operations
 * when no workspaces are active.
 */
class SolverMemoryArena {
public:
    ~SolverMemoryArena() = default;

    // Delete copy operations
    SolverMemoryArena(const SolverMemoryArena&) = delete;
    SolverMemoryArena& operator=(const SolverMemoryArena&) = delete;

    // Allow move operations
    SolverMemoryArena(SolverMemoryArena&&) = default;
    SolverMemoryArena& operator=(SolverMemoryArena&&) = default;

    /**
     * @brief Factory method to create a SolverMemoryArena
     *
     * @param arena_size Size of the memory arena in bytes
     * @return Expected containing shared_ptr to arena or error message
     */
    static std::expected<std::shared_ptr<SolverMemoryArena>, std::string> create(size_t arena_size);

    /**
     * @brief Try to reset the arena
     *
     * @return Expected containing success or error message if reset failed
     */
    std::expected<void, std::string> try_reset();

    /**
     * @brief Increment the active workspace count
     */
    void increment_active();

    /**
     * @brief Decrement the active workspace count
     */
    void decrement_active();

    /**
     * @brief Get the memory resource for this arena
     *
     * @return Pointer to the memory resource
     */
    std::pmr::memory_resource* resource();

    /**
     * @brief Get arena statistics
     *
     * @return Current arena statistics
     */
    SolverMemoryArenaStats get_stats() const;

private:
    /**
     * @brief Thread-safe counting memory resource wrapper
     *
     * Wraps an upstream memory resource and tracks bytes allocated/deallocated
     * using atomic operations for lock-free thread safety.
     */
    class CountingMemoryResource : public std::pmr::memory_resource {
    public:
        explicit CountingMemoryResource(std::pmr::memory_resource* upstream)
            : upstream_(upstream), bytes_used_(0) {}

        [[nodiscard]] size_t bytes_used() const {
            return bytes_used_.load(std::memory_order_relaxed);
        }

        void reset() {
            bytes_used_.store(0, std::memory_order_relaxed);
        }

    protected:
        void* do_allocate(size_t bytes, size_t alignment) override {
            void* ptr = upstream_->allocate(bytes, alignment);
            bytes_used_.fetch_add(bytes, std::memory_order_relaxed);
            return ptr;
        }

        void do_deallocate(void* ptr, size_t bytes, size_t alignment) override {
            upstream_->deallocate(ptr, bytes, alignment);
            bytes_used_.fetch_sub(bytes, std::memory_order_relaxed);
        }

        bool do_is_equal(const std::pmr::memory_resource& other) const noexcept override {
            return upstream_->is_equal(other);
        }

    private:
        std::pmr::memory_resource* upstream_;
        std::atomic<size_t> bytes_used_;
    };
    /**
     * @brief Private constructor
     *
     * @param arena_size Size of the memory arena in bytes
     */
    explicit SolverMemoryArena(size_t arena_size);

    // Member variables
    std::unique_ptr<UnifiedMemoryResource> upstream_resource_;  // Level 1: underlying resource
    std::unique_ptr<CountingMemoryResource> counting_resource_;  // Level 2: counting wrapper
    std::unique_ptr<std::pmr::monotonic_buffer_resource> arena_resource_;  // Level 3: arena
    std::unique_ptr<std::pmr::pool_options> pool_options_;
    std::vector<char> arena_buffer_;  // Storage buffer for the arena

    mutable std::mutex mutex_;
    size_t active_workspace_count_;
    size_t arena_size_;
};

}  // namespace memory
}  // namespace mango

#endif  // MANGO_SUPPORT_MEMORY_SOLVER_MEMORY_ARENA_HPP_