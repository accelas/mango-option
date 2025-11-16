#ifndef MANGO_SUPPORT_MEMORY_SOLVER_MEMORY_ARENA_HPP_
#define MANGO_SUPPORT_MEMORY_SOLVER_MEMORY_ARENA_HPP_

#include <memory>
#include <mutex>
#include <expected>
#include <cstring>
#include <memory_resource>
#include <vector>
#include <atomic>
#include <utility>
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
    /**
     * @brief RAII token that tracks active workspace usage.
     *
     * Creating a token increments the arena's active count and destroying it
     * decrements the count. Tokens are move-only and keep the arena alive via
     * shared_ptr ownership while in scope.
     */
    class ActiveWorkspaceToken {
    public:
        ActiveWorkspaceToken() = default;
        explicit ActiveWorkspaceToken(std::shared_ptr<SolverMemoryArena> arena)
            : arena_(std::move(arena)) {
            if (arena_) {
                arena_->increment_active();
            }
        }

        ActiveWorkspaceToken(const ActiveWorkspaceToken&) = delete;
        ActiveWorkspaceToken& operator=(const ActiveWorkspaceToken&) = delete;

        ActiveWorkspaceToken(ActiveWorkspaceToken&& other) noexcept
            : arena_(std::move(other.arena_)) {}

        ActiveWorkspaceToken& operator=(ActiveWorkspaceToken&& other) noexcept {
            if (this != &other) {
                release();
                arena_ = std::move(other.arena_);
            }
            return *this;
        }

        ~ActiveWorkspaceToken() { release(); }

        [[nodiscard]] SolverMemoryArena* get() const noexcept { return arena_.get(); }
        [[nodiscard]] std::shared_ptr<SolverMemoryArena> shared() const noexcept { return arena_; }
        [[nodiscard]] std::pmr::memory_resource* resource() const {
            return arena_ ? arena_->resource() : nullptr;
        }
        explicit operator bool() const noexcept { return static_cast<bool>(arena_); }

    private:
        void release() {
            if (arena_) {
                arena_->decrement_active();
                arena_.reset();
            }
        }

        std::shared_ptr<SolverMemoryArena> arena_;
    };

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
     * @brief Thread-safe wrapper around synchronized_pool_resource with tracking
     *
     * Wraps a synchronized_pool_resource and tracks bytes allocated/deallocated
     * with atomic operations for thread-safe memory accounting.
     */
    class TrackingSynchronizedResource : public std::pmr::memory_resource {
    public:
        explicit TrackingSynchronizedResource(std::pmr::memory_resource* upstream)
            : upstream_(upstream),
              pool_(std::make_unique<std::pmr::synchronized_pool_resource>(
                  std::pmr::pool_options{64, 1024}, upstream)),
              bytes_used_(0) {}

        [[nodiscard]] size_t bytes_used() const {
            return bytes_used_.load(std::memory_order_relaxed);
        }

        void reset() {
            pool_->release();
            bytes_used_.store(0, std::memory_order_relaxed);
        }

    protected:
        void* do_allocate(size_t bytes, size_t alignment) override {
            void* ptr = pool_->allocate(bytes, alignment);
            bytes_used_.fetch_add(bytes, std::memory_order_relaxed);
            return ptr;
        }

        void do_deallocate(void* ptr, size_t bytes, size_t alignment) override {
            pool_->deallocate(ptr, bytes, alignment);
            bytes_used_.fetch_sub(bytes, std::memory_order_relaxed);
        }

        bool do_is_equal(const std::pmr::memory_resource& other) const noexcept override {
            return pool_->is_equal(other);
        }

    private:
        std::pmr::memory_resource* upstream_;
        std::unique_ptr<std::pmr::synchronized_pool_resource> pool_;
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
    std::unique_ptr<TrackingSynchronizedResource> arena_resource_;  // Level 2+3: tracking+pool
    std::unique_ptr<std::pmr::pool_options> pool_options_;
    std::vector<char> arena_buffer_;  // Storage buffer (unused but kept for future)

    mutable std::mutex mutex_;
    size_t active_workspace_count_;
    size_t arena_size_;
};

}  // namespace memory
}  // namespace mango

#endif  // MANGO_SUPPORT_MEMORY_SOLVER_MEMORY_ARENA_HPP_
