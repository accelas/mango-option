// SPDX-License-Identifier: MIT
// src/support/aligned_allocator.hpp

#pragma once

#include <cstddef>
#include <cstdlib>
#include <new>

namespace mango {

/// Allocator for 64-byte aligned memory (AVX-512 compatible)
///
/// Standard-compliant allocator for use with std::vector and other containers.
/// Uses std::aligned_alloc for allocation with guaranteed 64-byte alignment.
///
/// Usage:
///   std::vector<double, AlignedAllocator<double>> vec(100);
///   // vec.data() is guaranteed 64-byte aligned
///
template<typename T, size_t Alignment = 64>
class AlignedAllocator {
public:
    using value_type = T;
    using size_type = std::size_t;
    using difference_type = std::ptrdiff_t;
    using propagate_on_container_move_assignment = std::true_type;

    static constexpr size_t alignment = Alignment;

    AlignedAllocator() noexcept = default;

    template<typename U>
    AlignedAllocator(const AlignedAllocator<U, Alignment>&) noexcept {}

    [[nodiscard]] T* allocate(size_type n) {
        if (n == 0) {
            return nullptr;
        }

        // std::aligned_alloc requires size to be multiple of alignment
        size_type byte_count = n * sizeof(T);
        size_type aligned_size = (byte_count + Alignment - 1) & ~(Alignment - 1);

        void* ptr = std::aligned_alloc(Alignment, aligned_size);
        if (!ptr) {
            throw std::bad_alloc();
        }
        return static_cast<T*>(ptr);
    }

    void deallocate(T* ptr, size_type) noexcept {
        std::free(ptr);
    }

    template<typename U>
    struct rebind {
        using other = AlignedAllocator<U, Alignment>;
    };
};

template<typename T, typename U, size_t A>
bool operator==(const AlignedAllocator<T, A>&, const AlignedAllocator<U, A>&) noexcept {
    return true;
}

template<typename T, typename U, size_t A>
bool operator!=(const AlignedAllocator<T, A>&, const AlignedAllocator<U, A>&) noexcept {
    return false;
}

/// Convenience alias for 64-byte aligned vector
template<typename T>
using AlignedVector = std::vector<T, AlignedAllocator<T, 64>>;

} // namespace mango
