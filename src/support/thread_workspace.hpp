// src/support/thread_workspace.hpp

#pragma once

#include <memory_resource>
#include <span>
#include <cstddef>
#include <cstdlib>   // std::aligned_alloc, std::free
#include <cassert>
#include <memory>    // std::unique_ptr

namespace mango {

/// Per-thread workspace buffer with 64-byte alignment guarantee
///
/// Primary: pmr::monotonic_buffer_resource over 64-byte aligned storage (Linux/glibc via std::aligned_alloc)
/// Fallback: thread-local pmr::unsynchronized_pool_resource (if exhausted)
///
/// Design principles:
/// - Buffer provides raw byte storage (std::byte) with 64-byte alignment
/// - Workspace consumers handle typed access; alignment is guaranteed by buffer
/// - pmr::memory_resource access is still available for callers that expect PMR
/// - Buffer is allocated once per thread, reused across iterations
/// - No release() - memory stays valid for parallel region lifetime
///
/// IMPORTANT: Alignment guarantee
/// The underlying storage is allocated with a 64-byte-aligned helper (std::aligned_alloc,
/// posix_memalign, or _aligned_malloc) to ensure the base pointer is cache-line aligned.
/// This is required for AVX-512 aligned loads and cache-line optimization. `std::pmr::vector`
/// only guarantees `alignof(max_align_t)` which is typically 16 bytes - insufficient for
/// our SIMD requirements.
///
/// Example:
///   MANGO_PRAGMA_PARALLEL
///   {
///       ThreadWorkspaceBuffer buffer(MyWorkspace::required_bytes(n));
///
///       // Create workspace ONCE per thread (starts object lifetimes)
///       auto ws = MyWorkspace::from_bytes(buffer.bytes(), n).value();
///
///       MANGO_PRAGMA_FOR_STATIC
///       for (size_t i = 0; i < count; ++i) {
///           // Reuse ws - solver overwrites spans each iteration.
///           // No re-initialization needed for trivial types.
///           solver.fit_with_workspace(input[i], ws);
///       }
///   }
///
/// Lifecycle notes:
/// - from_bytes() starts object lifetimes via start_array_lifetime()
/// - Workspace is created ONCE per thread, NOT per iteration
/// - Solver methods overwrite workspace arrays each iteration
/// - For B-spline fitting: band_storage, lapack_storage, pivots, coeffs
///   are all written fresh each fit - no stale state accumulates
///
class ThreadWorkspaceBuffer {
public:
    static constexpr size_t ALIGNMENT = 64;  // Cache line / AVX-512

    /// Construct with expected byte count (64-byte aligned)
    explicit ThreadWorkspaceBuffer(size_t byte_count)
        : size_(align_up(byte_count, ALIGNMENT))
        , storage_(allocate_aligned(size_), &ThreadWorkspaceBuffer::free_aligned)
        , fallback_(get_fallback_pool())
        , resource_(storage_.get(), size_, fallback_)
        , byte_view_(static_cast<std::byte*>(storage_.get()), size_)
    {
        if (!storage_) {
            throw std::bad_alloc{};
        }
        // Verify alignment (should always succeed with aligned allocation helper)
        assert(reinterpret_cast<std::uintptr_t>(storage_.get()) % ALIGNMENT == 0);
    }

    // Move constructor (monotonic_buffer_resource is not movable, so we manually handle it)
    ThreadWorkspaceBuffer(ThreadWorkspaceBuffer&& other) noexcept
        : size_(other.size_)
        , storage_(std::move(other.storage_))
        , fallback_(other.fallback_)
        , resource_(storage_.get(), size_, fallback_)
        , byte_view_(static_cast<std::byte*>(storage_.get()), size_)
    {
        other.size_ = 0;
        other.byte_view_ = std::span<std::byte>{};
    }

    // Move assignment
    ThreadWorkspaceBuffer& operator=(ThreadWorkspaceBuffer&& other) noexcept {
        if (this != &other) {
            storage_.reset();
            size_ = other.size_;
            storage_ = std::move(other.storage_);
            fallback_ = other.fallback_;
            // Reconstruct resource_ with new storage
            resource_.~monotonic_buffer_resource();
            new (&resource_) std::pmr::monotonic_buffer_resource(storage_.get(), size_, fallback_);
            byte_view_ = std::span<std::byte>(static_cast<std::byte*>(storage_.get()), size_);

            other.size_ = 0;
            other.byte_view_ = std::span<std::byte>{};
        }
        return *this;
    }

    // Not copyable
    ThreadWorkspaceBuffer(const ThreadWorkspaceBuffer&) = delete;
    ThreadWorkspaceBuffer& operator=(const ThreadWorkspaceBuffer&) = delete;

    /// Get byte span view of buffer (stable for lifetime of object)
    std::span<std::byte> bytes() noexcept { return byte_view_; }
    std::span<const std::byte> bytes() const noexcept { return byte_view_; }

    /// Access underlying pmr resource (for legacy callers still using pmr::vector)
    std::pmr::memory_resource& resource() noexcept { return resource_; }

    size_t size() const noexcept { return size_; }

private:
    static constexpr size_t align_up(size_t n, size_t alignment) {
        return (n + alignment - 1) & ~(alignment - 1);
    }

    static std::pmr::memory_resource* get_fallback_pool() {
        // unsynchronized_pool_resource is safe here because the pool is thread_local
        // No synchronization overhead needed for single-threaded access
        thread_local std::pmr::unsynchronized_pool_resource pool;
        return &pool;
    }

    /// Allocate 64-byte aligned storage
    static void* allocate_aligned(size_t byte_count) {
        byte_count = align_up(byte_count, ALIGNMENT);
        void* ptr = std::aligned_alloc(ALIGNMENT, byte_count);
        if (!ptr) {
            throw std::bad_alloc{};
        }
        return ptr;
    }

    static void free_aligned(void* ptr) noexcept { std::free(ptr); }

    size_t size_;
    std::unique_ptr<void, decltype(&std::free)> storage_;
    std::pmr::memory_resource* fallback_;  // Non-owning
    std::pmr::monotonic_buffer_resource resource_;
    std::span<std::byte> byte_view_;
};

} // namespace mango
