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
