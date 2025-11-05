#pragma once

#include <cstddef>
#include <algorithm>

namespace mango {

/**
 * CacheBlockConfig: Configuration for cache-aware grid blocking
 *
 * Splits large grids into cache-friendly blocks to improve memory locality.
 * Small grids (n < 5000) use no blocking (single block with overlap=1 for stencils).
 * Large grids use adaptive blocking based on cache size.
 */
struct CacheBlockConfig {
    size_t block_size;  // Points per block
    size_t n_blocks;    // Number of blocks
    size_t overlap;     // Halo size (points of overlap between blocks)

    // L1 cache size (per core): 32 KB typical
    // Each grid point: ~24 bytes (u_current, u_next, workspace)
    // L1-optimal: ~1000 points (24 KB)
    static constexpr size_t L1_CACHE_SIZE = 32 * 1024;

    // L2 cache size (per core): 256 KB typical
    // L2-optimal: ~8000 points (192 KB)
    static constexpr size_t L2_CACHE_SIZE = 256 * 1024;

    // Bytes per grid point (conservative estimate)
    static constexpr size_t BYTES_PER_POINT = 24;

    // Create config for specific cache level
    static CacheBlockConfig for_cache(size_t n_points, size_t cache_size) {
        const size_t optimal_points = cache_size / BYTES_PER_POINT;

        if (n_points <= optimal_points) {
            // Fits in cache - use single block with halo
            return CacheBlockConfig{n_points, 1, 1};
        }

        // Multiple blocks needed
        const size_t n_blocks = (n_points + optimal_points - 1) / optimal_points;
        const size_t block_size = (n_points + n_blocks - 1) / n_blocks;
        const size_t overlap = 1;  // Stencil width

        return CacheBlockConfig{block_size, n_blocks, overlap};
    }

    // L1-optimized: ~1000 points (24 KB for 3 arrays)
    static CacheBlockConfig l1_blocked(size_t n) {
        return for_cache(n, L1_CACHE_SIZE);
    }

    // L2-optimized: ~8000 points (192 KB for 3 arrays)
    static CacheBlockConfig l2_blocked(size_t n) {
        return for_cache(n, L2_CACHE_SIZE);
    }

    // Adaptive: single block for small grids, L1-blocked for large
    static CacheBlockConfig adaptive(size_t n, size_t threshold = 5000) {
        if (n < threshold) {
            return CacheBlockConfig{n, 1, 1};  // Single block with halo
        }
        return l1_blocked(n);
    }
};

} // namespace mango
