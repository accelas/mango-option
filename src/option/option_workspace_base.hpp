#pragma once

#include "src/support/memory/workspace_base.hpp"
#include <memory_resource>
#include <vector>

namespace mango {

/**
 * OptionWorkspaceBase: PMR-aware base class for option-related workspaces
 *
 * Extends WorkspaceBase to provide PMR-compatible data structures
 * for option pricing components like price tables, B-splines, and
 * normalized chain solvers.
 *
 * Key features:
 * - PMR vector types for efficient memory management
 * - Zero-copy interfaces using std::span
 * - Automatic alignment for SIMD operations
 * - Reusable buffer patterns for repeated solves
 */
class OptionWorkspaceBase : public WorkspaceBase {
public:
    explicit OptionWorkspaceBase(size_t initial_buffer_size = 1024 * 1024)
        : WorkspaceBase(initial_buffer_size)
        , pmr_resource_(&resource_)
    {}

    /// PMR vector type for double arrays
    using pmr_vector = std::pmr::vector<double>;

    /// PMR vector type for size_t arrays
    using pmr_size_vector = std::pmr::vector<size_t>;

    /**
     * Create a PMR vector with automatic alignment and padding
     *
     * @param size Number of elements
     * @param pad_to_simd Whether to pad to SIMD width (default: true)
     * @return PMR vector with allocated memory
     */
    pmr_vector create_pmr_vector(size_t size, bool pad_to_simd = true) {
        size_t padded_size = pad_to_simd ? WorkspaceBase::pad_to_simd(size) : size;
        return pmr_vector(padded_size, &pmr_resource_);
    }

    /**
     * Create a PMR vector from existing data with optional padding
     *
     * @param data Source data span
     * @param pad_to_simd Whether to pad to SIMD width (default: true)
     * @return PMR vector with copied and padded data
     */
    pmr_vector create_pmr_vector_from_span(std::span<const double> data, bool pad_to_simd = true) {
        pmr_vector result = create_pmr_vector(data.size(), pad_to_simd);
        std::copy(data.begin(), data.end(), result.begin());
        return result;
    }

    /**
     * Get a span view of PMR vector data (excluding padding)
     *
     * @param vec PMR vector
     * @param logical_size Logical size (excluding padding)
     * @return Span view of the logical data
     */
    static std::span<const double> get_logical_span(const pmr_vector& vec, size_t logical_size) {
        return std::span<const double>(vec.data(), logical_size);
    }

    /**
     * Get a mutable span view of PMR vector data (excluding padding)
     *
     * @param vec PMR vector
     * @param logical_size Logical size (excluding padding)
     * @return Mutable span view of the logical data
     */
    static std::span<double> get_logical_span(pmr_vector& vec, size_t logical_size) {
        return std::span<double>(vec.data(), logical_size);
    }

    /**
     * Create a zero-copy interface for const data
     * If data is already in PMR format, use directly. Otherwise, create a copy.
     *
     * @param data Source data
     * @return PMR vector (either zero-copy or copied)
     */
    pmr_vector create_zero_copy_or_copy(std::span<const double> data) {
        // For now, always copy. Future optimization: detect if data is already in our arena
        return create_pmr_vector_from_span(data);
    }

protected:
    /// PMR memory resource adapter for use with std::pmr types
    class PmrAdapter : public std::pmr::memory_resource {
    public:
        explicit PmrAdapter(memory::UnifiedMemoryResource* resource)
            : resource_(resource) {}

    private:
        void* do_allocate(size_t bytes, size_t alignment) override {
            return resource_->allocate(bytes, alignment);
        }

        void do_deallocate(void* ptr, size_t bytes, size_t alignment) override {
            // PMR deallocate is a no-op for monotonic_buffer_resource
            (void)ptr;
            (void)bytes;
            (void)alignment;
        }

        bool do_is_equal(const std::pmr::memory_resource& other) const noexcept override {
            return this == &other;
        }

        memory::UnifiedMemoryResource* resource_;
    };

    PmrAdapter pmr_resource_;
};

} // namespace mango