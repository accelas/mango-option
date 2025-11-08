#pragma once

#include <memory_resource>
#include <cstddef>

namespace mango {

/**
 * Aligned memory resource for std::pmr containers
 *
 * Provides 64-byte aligned allocations via std::pmr interface using
 * the default memory resource with explicit alignment.
 *
 * This resource backs std::pmr::vector and other PMR containers,
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
        return std::pmr::get_default_resource()->allocate(bytes, alignment_);
    }

    void do_deallocate(void* ptr, std::size_t bytes, std::size_t) override {
        std::pmr::get_default_resource()->deallocate(ptr, bytes, alignment_);
    }

    bool do_is_equal(const std::pmr::memory_resource& other) const noexcept override {
        return this == &other;
    }

    std::size_t alignment_;
};

} // namespace mango
