#pragma once

#include <cstddef>
#include <cstdlib>
#include <new>
#include <limits>

#ifdef _MSC_VER
#include <malloc.h>
#endif

namespace mango {

/**
 * Aligned allocator for std::vector and other STL containers
 *
 * Provides portable 64-byte aligned allocation across platforms:
 * - Uses std::aligned_alloc on C++17 compliant platforms
 * - Uses _aligned_malloc/_aligned_free on MSVC
 * - Uses posix_memalign on POSIX systems without aligned_alloc
 *
 * This enables efficient SIMD vectorization (AVX-512) while preserving
 * container value semantics and portability.
 */
template<typename T, std::size_t Alignment = 64>
class AlignedAllocator {
public:
    using value_type = T;
    using size_type = std::size_t;
    using difference_type = std::ptrdiff_t;

    template<typename U>
    struct rebind {
        using other = AlignedAllocator<U, Alignment>;
    };

    AlignedAllocator() noexcept = default;

    template<typename U>
    AlignedAllocator(const AlignedAllocator<U, Alignment>&) noexcept {}

    T* allocate(size_type n) {
        if (n > std::numeric_limits<size_type>::max() / sizeof(T)) {
            throw std::bad_array_new_length();
        }

        const size_type bytes = n * sizeof(T);
        const size_type aligned_bytes = (bytes + Alignment - 1) & ~(Alignment - 1);

        void* ptr = nullptr;

#if defined(_MSC_VER)
        // MSVC: use _aligned_malloc
        ptr = _aligned_malloc(aligned_bytes, Alignment);
#elif defined(__APPLE__) || !defined(_ISOC11_SOURCE)
        // macOS or systems without C11 aligned_alloc: use posix_memalign
        if (posix_memalign(&ptr, Alignment, aligned_bytes) != 0) {
            ptr = nullptr;
        }
#else
        // C++17 / C11: use std::aligned_alloc
        ptr = std::aligned_alloc(Alignment, aligned_bytes);
#endif

        if (!ptr) {
            throw std::bad_alloc();
        }

        return static_cast<T*>(ptr);
    }

    void deallocate(T* ptr, size_type) noexcept {
        if (!ptr) return;

#if defined(_MSC_VER)
        _aligned_free(ptr);
#else
        std::free(ptr);
#endif
    }

    template<typename U>
    bool operator==(const AlignedAllocator<U, Alignment>&) const noexcept {
        return true;
    }

    template<typename U>
    bool operator!=(const AlignedAllocator<U, Alignment>&) const noexcept {
        return false;
    }
};

} // namespace mango
