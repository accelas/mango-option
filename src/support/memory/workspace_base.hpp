#pragma once

#include "unified_memory_resource.hpp"
#include <cassert>
#include <cstddef>
#include <algorithm>

namespace mango {

/// Tile metadata for operator-level tiling
struct TileMetadata {
    size_t tile_start;      ///< Start index in original grid
    size_t tile_size;       ///< Actual elements (not padded)
    size_t padded_size;     ///< Rounded to SIMD_WIDTH
    size_t alignment;       ///< Byte alignment
};

/**
 * WorkspaceBase: Base class providing allocator and tiling infrastructure
 *
 * Provides reusable functionality for all workspace types:
 * - UnifiedMemoryResource allocator
 * - Tiling metadata generation
 * - SIMD padding utilities
 */
class WorkspaceBase {
public:
    explicit WorkspaceBase(size_t initial_buffer_size = 1024 * 1024)
        : resource_(initial_buffer_size)
    {}

    /// AVX-512: 8 doubles per vector
    static constexpr size_t SIMD_WIDTH = 8;

    /// Round up to SIMD_WIDTH boundary
    static constexpr size_t pad_to_simd(size_t n) {
        return ((n + SIMD_WIDTH - 1) / SIMD_WIDTH) * SIMD_WIDTH;
    }

    /// Query total bytes allocated
    size_t bytes_allocated() const { return resource_.bytes_allocated(); }

protected:
    memory::UnifiedMemoryResource resource_;
};

} // namespace mango
