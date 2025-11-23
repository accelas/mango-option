#include "src/support/memory/aligned_arena.hpp"
#include <cstring>
#include <cstdlib>

namespace mango {
namespace memory {

AlignedArena::AlignedArena(size_t bytes, size_t align)
    : align_(align), offset_(0) {
    // Allocate extra space to guarantee alignment
    buffer_.resize(bytes + align);
}

std::expected<std::shared_ptr<AlignedArena>, std::string>
AlignedArena::create(size_t bytes, size_t align) {
    if (bytes == 0) {
        return std::unexpected("Arena size must be positive");
    }
    if (align == 0 || (align & (align - 1)) != 0) {
        return std::unexpected("Alignment must be a power of 2");
    }

    auto arena = std::shared_ptr<AlignedArena>(new AlignedArena(bytes, align));
    arena->self_ = arena;
    return arena;
}

double* AlignedArena::allocate(size_t count) {
    const size_t bytes = count * sizeof(double);

    // Calculate current position in buffer
    uintptr_t current_addr = reinterpret_cast<uintptr_t>(buffer_.data()) + offset_;

    // Align the current position to the required boundary
    uintptr_t aligned_addr = (current_addr + align_ - 1) & ~(align_ - 1);
    size_t aligned_offset = aligned_addr - reinterpret_cast<uintptr_t>(buffer_.data());

    // Check if we have enough space after alignment
    if (aligned_offset + bytes > buffer_.size()) {
        return nullptr;  // Out of memory
    }

    double* ptr = reinterpret_cast<double*>(buffer_.data() + aligned_offset);
    offset_ = aligned_offset + bytes;  // Update offset to end of allocation

    return ptr;
}

std::shared_ptr<AlignedArena> AlignedArena::share() {
    return self_.lock();
}

} // namespace memory
} // namespace mango
