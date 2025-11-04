#pragma once

#include "cache_config.hpp"
#include <vector>
#include <span>
#include <cstddef>

namespace mango {

/**
 * WorkspaceStorage: Cache-blocked storage for PDE solver arrays
 *
 * Manages solver workspace with:
 * - Contiguous buffer for all arrays (cache-friendly)
 * - Pre-computed dx array (avoids out-of-bounds in cache blocks)
 * - Cache-blocking configuration
 * - 64-byte alignment for SIMD operations
 */
class WorkspaceStorage {
public:
    /**
     * Create workspace for grid
     * @param n Number of grid points
     * @param grid Grid coordinates (used to pre-compute dx)
     */
    explicit WorkspaceStorage(size_t n, std::span<const double> grid)
        : buffer_(5 * n)  // u_current, u_next, u_stage, rhs, Lu
        , cache_config_(CacheBlockConfig::adaptive(n))
        , dx_(n - 1)
    {
        // Pre-compute grid spacing once during initialization
        // CRITICAL: Avoids out-of-bounds access when processing cache blocks
        for (size_t i = 0; i < n - 1; ++i) {
            dx_[i] = grid[i + 1] - grid[i];
        }

        // Set up array views as non-overlapping spans
        size_t offset = 0;
        u_current_ = std::span{buffer_.data() + offset, n}; offset += n;
        u_next_    = std::span{buffer_.data() + offset, n}; offset += n;
        u_stage_   = std::span{buffer_.data() + offset, n}; offset += n;
        rhs_       = std::span{buffer_.data() + offset, n}; offset += n;
        lu_        = std::span{buffer_.data() + offset, n};
    }

    // Access to arrays
    std::span<double> u_current() { return u_current_; }
    std::span<const double> u_current() const { return u_current_; }

    std::span<double> u_next() { return u_next_; }
    std::span<const double> u_next() const { return u_next_; }

    std::span<double> u_stage() { return u_stage_; }
    std::span<const double> u_stage() const { return u_stage_; }

    std::span<double> rhs() { return rhs_; }
    std::span<const double> rhs() const { return rhs_; }

    std::span<double> lu() { return lu_; }
    std::span<const double> lu() const { return lu_; }

    // Access to pre-computed dx
    std::span<const double> dx() const { return dx_; }

    // Cache configuration
    const CacheBlockConfig& cache_config() const { return cache_config_; }
    CacheBlockConfig& cache_config() { return cache_config_; }

    /**
     * BlockInfo: Information about a cache block with halo
     */
    struct BlockInfo {
        std::span<const double> data;  // Data with halo
        size_t interior_start;         // Global index where interior starts
        size_t interior_count;         // Number of interior points
        size_t halo_left;             // Number of left halo points
        size_t halo_right;            // Number of right halo points
    };

    /**
     * Get interior range for a cache block
     * @param block_idx Block index
     * @return [start, end) range of interior points (exclusive of boundaries)
     */
    std::pair<size_t, size_t> get_block_interior_range(size_t block_idx) const {
        const size_t n = u_current_.size();
        size_t start = block_idx * cache_config_.block_size;
        size_t end = std::min(start + cache_config_.block_size, n);

        // Skip global boundaries (0 and n-1)
        size_t interior_start = std::max(start, size_t{1});
        size_t interior_end = std::min(end, n - 1);

        return {interior_start, interior_end};
    }

    /**
     * Get block with halo for stencil operations
     * @param array Array to get block from
     * @param block_idx Block index
     * @return BlockInfo with data span and halo information
     */
    BlockInfo get_block_with_halo(std::span<const double> array, size_t block_idx) const {
        auto [interior_start, interior_end] = get_block_interior_range(block_idx);

        if (interior_start >= interior_end) {
            // Boundary-only block (shouldn't happen with proper sizing)
            return {std::span<const double>{}, interior_start, 0, 0, 0};
        }

        const size_t n = array.size();
        const size_t interior_count = interior_end - interior_start;

        // Compute halo sizes (clamped to available points)
        const size_t halo_left  = std::min(cache_config_.overlap, interior_start);
        const size_t halo_right = std::min(cache_config_.overlap, n - interior_end);

        // Build span with halos
        auto data_with_halo = array.subspan(
            interior_start - halo_left,
            interior_count + halo_left + halo_right
        );

        return {data_with_halo, interior_start, interior_count, halo_left, halo_right};
    }

private:
    std::vector<double> buffer_;     // Single allocation for all arrays
    CacheBlockConfig cache_config_;  // Cache-blocking configuration
    std::vector<double> dx_;         // Pre-computed grid spacing

    // Spans into buffer_
    std::span<double> u_current_;
    std::span<double> u_next_;
    std::span<double> u_stage_;
    std::span<double> rhs_;
    std::span<double> lu_;
};

} // namespace mango
