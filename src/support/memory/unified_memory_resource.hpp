#pragma once

#include <memory_resource>
#include <cstddef>

namespace mango::memory {

/**
 * RAII wrapper around std::pmr::monotonic_buffer_resource
 *
 * Provides workspace-owned allocator with:
 * - Zero-cost reset() between solves
 * - 64-byte default alignment for AVX-512
 * - Manual bytes_allocated() tracking
 *
 * Thread-safe: each workspace owns one instance (no shared state)
 */
class UnifiedMemoryResource {
public:
    explicit UnifiedMemoryResource(size_t initial_buffer_size = 1024 * 1024,
                                  std::pmr::memory_resource* resource = nullptr)
        : upstream_(resource ? resource : std::pmr::get_default_resource())
        , monotonic_(initial_buffer_size, upstream_)
        , bytes_allocated_(0)
    {}

    /**
     * Allocate memory with specified alignment
     *
     * @param bytes Size in bytes
     * @param alignment Alignment requirement (default: 64 for AVX-512)
     * @return Pointer to allocated memory
     */
    void* allocate(size_t bytes, size_t alignment = 64) {
        void* ptr = monotonic_.allocate(bytes, alignment);
        bytes_allocated_ += bytes;
        return ptr;
    }

    /**
     * Reset for reuse (zero-cost between solves)
     *
     * WARNING: Invalidates all previously allocated pointers!
     */
    void reset() {
        monotonic_.release();
        bytes_allocated_ = 0;
    }

    /// Query total bytes allocated
    size_t bytes_allocated() const { return bytes_allocated_; }

    /// Get PMR resource
    std::pmr::memory_resource* pmr_resource() { return &monotonic_; }

    /// Get PMR resource (const version)
    const std::pmr::memory_resource* pmr_resource() const { return &monotonic_; }

private:
    std::pmr::memory_resource* upstream_;
    std::pmr::monotonic_buffer_resource monotonic_;
    size_t bytes_allocated_;  // Manual tracking (PMR doesn't expose this)
};

} // namespace mango::memory
