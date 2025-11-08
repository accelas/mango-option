#pragma once

#include <memory_resource>
#include <cstddef>
#include <cstdlib>

#ifdef _MSC_VER
#include <malloc.h>
#endif

namespace mango {

/**
 * Aligned memory resource for std::pmr containers
 *
 * Provides portable 64-byte aligned allocations via std::pmr interface:
 * - Uses _aligned_malloc/_aligned_free on MSVC
 * - Uses posix_memalign on POSIX systems
 * - Uses std::aligned_alloc on C++17 compliant platforms
 *
 * This resource can back std::pmr::vector and other PMR containers,
 * enabling SIMD-friendly allocations with standard value semantics.
 */
class AlignedMemoryResource : public std::pmr::memory_resource {
public:
    explicit AlignedMemoryResource(std::size_t alignment = 64)
        : alignment_(alignment)
    {}

    ~AlignedMemoryResource() override = default;

private:
    void* do_allocate(std::size_t bytes, std::size_t) override {
        // Round up to multiple of alignment
        const std::size_t aligned_bytes = (bytes + alignment_ - 1) & ~(alignment_ - 1);

        void* ptr = nullptr;

#if defined(_MSC_VER)
        // MSVC: use _aligned_malloc
        ptr = _aligned_malloc(aligned_bytes, alignment_);
#elif defined(__APPLE__) || !defined(_ISOC11_SOURCE)
        // macOS or systems without C11: use posix_memalign
        if (posix_memalign(&ptr, alignment_, aligned_bytes) != 0) {
            ptr = nullptr;
        }
#else
        // C++17 / C11: use std::aligned_alloc
        ptr = std::aligned_alloc(alignment_, aligned_bytes);
#endif

        if (!ptr) {
            throw std::bad_alloc();
        }

        return ptr;
    }

    void do_deallocate(void* ptr, std::size_t, std::size_t) override {
        if (!ptr) return;

#if defined(_MSC_VER)
        _aligned_free(ptr);
#else
        std::free(ptr);
#endif
    }

    bool do_is_equal(const std::pmr::memory_resource& other) const noexcept override {
        return this == &other;
    }

    std::size_t alignment_;
};

} // namespace mango
