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

    /**
     * Generate tile metadata for operator-level tiling
     *
     * Distributes n elements across num_tiles, with remainder spread
     * across first tiles. All tiles SIMD-padded.
     *
     * @param n Total number of elements
     * @param tile_idx Index of this tile [0, num_tiles)
     * @param num_tiles Total number of tiles
     * @return TileMetadata for this tile
     */
    static TileMetadata tile_info(size_t n, size_t tile_idx, size_t num_tiles) {
        assert(num_tiles > 0 && "num_tiles must be positive");
        assert(tile_idx < num_tiles && "tile_idx out of bounds");

        const size_t base_tile_size = n / num_tiles;
        const size_t remainder = n % num_tiles;
        const size_t tile_size = base_tile_size + (tile_idx < remainder ? 1 : 0);
        const size_t tile_start = tile_idx * base_tile_size + std::min(tile_idx, remainder);
        const size_t padded_size = pad_to_simd(tile_size);

        return {tile_start, tile_size, padded_size, 64};
    }

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
