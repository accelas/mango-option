#pragma once

#include "cache_config.hpp"
#include <span>
#include <cstddef>
#include <cstdlib>
#include <new>
#include <algorithm>
#include <vector>

namespace mango {

/// Workspace storage for PDE solver arrays
///
/// **CPU-only implementation** - SYCL GPU specialization deferred to v2.1.
///
/// Manages all solver state in a single contiguous buffer for cache efficiency.
/// Arrays: u_current, u_next, u_stage, rhs, Lu, psi (6n doubles total).
///
/// **64-byte alignment** - Uses aligned allocation for AVX-512 SIMD vectorization.
/// The buffer is aligned to 64 bytes to enable efficient vector loads/stores.
///
/// **Pre-computed dx array** - Grid spacing computed once during construction
/// to avoid redundant S[i+1] - S[i] calculations in stencil operations.
///
/// **Cache-blocking** - Adaptive strategy based on grid size:
/// - n < 5000: Single block (no blocking overhead)
/// - n ≥ 5000: L1-blocked (~1000 points per block, ~32 KB working set)
///
/// Future GPU version (v2.1) will use SYCL unified shared memory (USM)
/// with explicit device allocation and host-device synchronization.
class WorkspaceStorage {
public:
    /// Construct workspace for n grid points
    ///
    /// @param n Number of grid points
    /// @param grid Grid coordinates for pre-computing dx
    /// @param threshold Cache-blocking threshold (default 5000)
    ///
    /// Allocates 6n doubles (u_current, u_next, u_stage, rhs, Lu, psi)
    /// plus (n-1) doubles for pre-computed dx array.
    /// Memory is 64-byte aligned for AVX-512 SIMD operations.
    explicit WorkspaceStorage(size_t n, std::span<const double> grid, size_t threshold = 5000)
        : n_(n)
        , cache_config_(CacheBlockConfig::adaptive(n, threshold))
        , dx_(n - 1)
    {
        // Allocate 64-byte aligned buffer for SIMD vectorization
        constexpr size_t alignment = 64;  // AVX-512 alignment
        const size_t buffer_size = 6 * n;

        buffer_ = static_cast<double*>(std::aligned_alloc(alignment, buffer_size * sizeof(double)));
        if (buffer_ == nullptr) {
            throw std::bad_alloc();
        }
        // Pre-compute grid spacing once during initialization
        // CRITICAL: Avoids out-of-bounds access when processing cache blocks
        for (size_t i = 0; i < n - 1; ++i) {
            dx_[i] = grid[i + 1] - grid[i];
        }

        // Set up array views as non-overlapping spans
        size_t offset = 0;
        u_current_ = std::span{buffer_ + offset, n}; offset += n;
        u_next_    = std::span{buffer_ + offset, n}; offset += n;
        u_stage_   = std::span{buffer_ + offset, n}; offset += n;
        rhs_       = std::span{buffer_ + offset, n}; offset += n;
        lu_        = std::span{buffer_ + offset, n}; offset += n;
        psi_       = std::span{buffer_ + offset, n}; offset += n;
    }

    /// Destructor - free aligned memory
    ~WorkspaceStorage() {
        std::free(buffer_);
    }

    // Disable copy/move to prevent double-free issues
    WorkspaceStorage(const WorkspaceStorage&) = delete;
    WorkspaceStorage& operator=(const WorkspaceStorage&) = delete;
    WorkspaceStorage(WorkspaceStorage&&) = delete;
    WorkspaceStorage& operator=(WorkspaceStorage&&) = delete;

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

    std::span<double> psi_buffer() { return psi_; }
    std::span<const double> psi_buffer() const { return psi_; }

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
    size_t n_;                       // Number of grid points
    double* buffer_;                 // 64-byte aligned buffer for all arrays (CPU memory)
    CacheBlockConfig cache_config_;  // Cache-blocking configuration (CPU-only)
    std::vector<double> dx_;         // Pre-computed grid spacing

    // Spans into buffer_
    std::span<double> u_current_;
    std::span<double> u_next_;
    std::span<double> u_stage_;
    std::span<double> rhs_;
    std::span<double> lu_;
    std::span<double> psi_;
};

} // namespace mango
