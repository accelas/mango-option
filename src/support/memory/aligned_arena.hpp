#pragma once

#include <memory>
#include <vector>
#include <cstddef>
#include <expected>
#include <string>

namespace mango {
namespace memory {

/// Arena allocator with guaranteed alignment for SIMD operations
///
/// Provides 64-byte aligned memory allocation for AVX-512 compatibility.
/// Uses shared_ptr for automatic lifetime management.
class AlignedArena {
public:
    /// Factory method to create arena
    ///
    /// @param bytes Total size in bytes
    /// @param align Alignment requirement (default 64 for AVX-512)
    /// @return Shared pointer to arena or error message
    [[nodiscard]] static std::expected<std::shared_ptr<AlignedArena>, std::string>
    create(size_t bytes, size_t align = 64);

    /// Allocate aligned memory for count doubles
    ///
    /// @param count Number of doubles to allocate
    /// @return Pointer to aligned memory or nullptr if insufficient space
    [[nodiscard]] double* allocate(size_t count);

    /// Share this arena (increment reference count)
    ///
    /// @return Shared pointer to this arena
    [[nodiscard]] std::shared_ptr<AlignedArena> share();

    /// Get total capacity in bytes
    [[nodiscard]] size_t capacity() const noexcept { return buffer_.size(); }

    /// Get current offset in bytes
    [[nodiscard]] size_t offset() const noexcept { return offset_; }

private:
    explicit AlignedArena(size_t bytes, size_t align);

    std::vector<std::byte> buffer_;
    size_t align_;
    size_t offset_;

    // Use weak_ptr instead of shared_ptr to prevent circular references.
    // The arena is owned by external shared_ptr returned from create().
    // This weak_ptr allows share() to return a shared_ptr without creating cycles.
    std::weak_ptr<AlignedArena> self_;
};

} // namespace memory
} // namespace mango
