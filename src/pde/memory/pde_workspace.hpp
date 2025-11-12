#pragma once

#include "workspace_base.hpp"
#include <span>
#include <cassert>
#include <algorithm>
#include <vector>
#include <experimental/simd>

namespace mango {

/**
 * PDEWorkspace: workspace for PDE solver with SoA layout
 *
 * Full Structure-of-Arrays layout for SIMD-friendly access:
 * - Each state array separate and SIMD-padded
 * - Zero-initialized padding for safe tail processing
 * - Dual accessors: logical size and padded size
 *
 * LIFETIME REQUIREMENTS:
 * - The `grid` span passed to constructor must remain valid for the lifetime
 *   of this workspace (stored for reset() reinit).
 *
 * INVALIDATION WARNING:
 * - reset() invalidates all previously returned std::span objects.
 * - After reset(), caller MUST re-acquire spans via accessors.
 */
class PDEWorkspace : public WorkspaceBase {
public:
    explicit PDEWorkspace(size_t n, std::span<const double> grid,
                         size_t batch_width = 0,  // NEW: 0 = single-contract mode
                         size_t initial_buffer_size = 1024 * 1024)
        : WorkspaceBase(initial_buffer_size)
        , n_(n)
        , padded_n_(pad_to_simd(n))
        , grid_(grid)
        , batch_width_(batch_width)  // NEW
        , u_batch_(nullptr)           // NEW
        , lu_batch_(nullptr)          // NEW
    {
        assert(!grid.empty() && "grid must not be empty");
        assert(grid.size() == n && "grid size must match n");
        allocate_and_initialize();
    }

    // SoA array accessors (logical size)
    std::span<double> u_current() { return {u_current_, n_}; }
    std::span<const double> u_current() const { return {u_current_, n_}; }

    std::span<double> u_next() { return {u_next_, n_}; }
    std::span<const double> u_next() const { return {u_next_, n_}; }

    std::span<double> u_stage() { return {u_stage_, n_}; }
    std::span<const double> u_stage() const { return {u_stage_, n_}; }

    std::span<double> rhs() { return {rhs_, n_}; }
    std::span<const double> rhs() const { return {rhs_, n_}; }

    std::span<double> lu() { return {lu_, n_}; }
    std::span<const double> lu() const { return {lu_, n_}; }

    std::span<double> psi_buffer() { return {psi_, n_}; }
    std::span<const double> psi_buffer() const { return {psi_, n_}; }

    // Batch AoS accessors (for horizontal SIMD stencil)
    std::span<double> batch_slice() {
        return {u_batch_, n_ * batch_width_};
    }
    std::span<const double> batch_slice() const {
        return {u_batch_, n_ * batch_width_};
    }

    std::span<double> lu_batch() {
        return {lu_batch_, n_ * batch_width_};
    }
    std::span<const double> lu_batch() const {
        return {lu_batch_, n_ * batch_width_};
    }

    // Per-lane SoA accessors (for Newton machinery)
    std::span<double> u_lane(size_t lane) {
        assert(lane < batch_width_ && "lane out of range");
        return u_lanes_[lane];
    }
    std::span<const double> u_lane(size_t lane) const {
        assert(lane < batch_width_ && "lane out of range");
        return u_lanes_[lane];
    }

    std::span<double> lu_lane(size_t lane) {
        assert(lane < batch_width_ && "lane out of range");
        return {lu_lane_buffers_[lane], n_};
    }
    std::span<const double> lu_lane(size_t lane) const {
        assert(lane < batch_width_ && "lane out of range");
        return lu_lanes_[lane];
    }

    // Padded accessors for SIMD kernels
    //
    // CRITICAL: Padded spans do NOT include front guard cells!
    // - Stencil operators accessing u[i-1] must use start >= 1
    // - Boundary points (i=0, i=n-1) handled separately by boundary condition code
    // - Padding is ONLY at the tail for safe SIMD overread
    //
    // Example safe usage:
    //   auto u_padded = workspace.u_current_padded();
    //   operator.compute(u_padded, Lu, start=1, end=n-1);  // ✓ Safe
    //   operator.compute(u_padded, Lu, start=0, end=n);    // ✗ UNSAFE: u[-1] access!
    std::span<double> u_current_padded() { return {u_current_, padded_n_}; }
    std::span<const double> u_current_padded() const { return {u_current_, padded_n_}; }

    std::span<double> u_next_padded() { return {u_next_, padded_n_}; }
    std::span<const double> u_next_padded() const { return {u_next_, padded_n_}; }

    std::span<double> lu_padded() { return {lu_, padded_n_}; }
    std::span<const double> lu_padded() const { return {lu_, padded_n_}; }

    // Grid spacing (SIMD-padded, zero-filled tail)
    std::span<const double> dx() const { return {dx_, n_ - 1}; }
    std::span<const double> dx_padded() const { return {dx_, pad_to_simd(n_ - 1)}; }

    /// Tile metadata for this workspace
    ///
    /// NOTE: Currently unused in production code. Reserved for future multi-level
    /// cache tiling optimizations (L1/L2/L3 blocking). The infrastructure exists
    /// but benchmarking showed no benefit on modern CPUs with large caches.
    /// See CLAUDE.md "Cache Blocking (Removed)" section for details.
    TileMetadata tile_info(size_t tile_idx, size_t num_tiles) const {
        return WorkspaceBase::tile_info(n_, tile_idx, num_tiles);
    }

    /**
     * Reset and reinitialize
     * WARNING: Invalidates all previously returned spans!
     */
    void reset() {
        resource_.reset();
        allocate_and_initialize();
    }

    size_t logical_size() const { return n_; }
    size_t padded_size() const { return padded_n_; }

    // Batch mode queries
    bool has_batch() const { return batch_width_ > 0; }
    size_t batch_width() const { return batch_width_; }

    /**
     * Pack per-lane SoA buffers into AoS batch slice
     *
     * Performs vectorized transpose from per-lane Structure-of-Arrays layout
     * to Array-of-Structures batch slice. Uses std::experimental::simd for
     * optimal performance with scalar tail handling.
     *
     * Memory layout transformation:
     *   SoA: u_lane[0][i], u_lane[1][i], ..., u_lane[W-1][i]  (W separate arrays)
     *   AoS: u_batch[i*W + 0], u_batch[i*W + 1], ..., u_batch[i*W + W-1]  (interleaved)
     *
     * PRECONDITION: batch_width_ > 0 (batch mode enabled)
     */
    void pack_to_batch_slice() {
        assert(batch_width_ > 0 && "pack requires batch mode");

        using simd_t = std::experimental::native_simd<double>;
        constexpr size_t simd_width = simd_t::size();

        for (size_t i = 0; i < n_; ++i) {
            size_t lane = 0;

            // Vectorized transpose
            for (; lane + simd_width <= batch_width_; lane += simd_width) {
                simd_t chunk;
                for (size_t k = 0; k < simd_width; ++k) {
                    chunk[k] = u_lane_buffers_[lane + k][i];
                }
                chunk.copy_to(&u_batch_[i * batch_width_ + lane],
                             std::experimental::element_aligned);
            }

            // Scalar tail
            for (; lane < batch_width_; ++lane) {
                u_batch_[i * batch_width_ + lane] = u_lane_buffers_[lane][i];
            }
        }
    }

private:
    void allocate_and_initialize() {
        allocate_arrays();
        precompute_grid_spacing();
        allocate_batch_buffers();  // NEW
    }

    void allocate_batch_buffers() {
        if (batch_width_ == 0) return;  // Single-contract mode

        const size_t aos_size = padded_n_ * batch_width_;
        const size_t aos_bytes = aos_size * sizeof(double);

        // Allocate AoS buffers
        u_batch_ = static_cast<double*>(resource_.allocate(aos_bytes));
        lu_batch_ = static_cast<double*>(resource_.allocate(aos_bytes));

        // Zero-initialize
        std::fill(u_batch_, u_batch_ + aos_size, 0.0);
        std::fill(lu_batch_, lu_batch_ + aos_size, 0.0);

        // Allocate per-lane SoA buffers
        const size_t lane_bytes = padded_n_ * sizeof(double);
        u_lane_buffers_.resize(batch_width_);
        lu_lane_buffers_.resize(batch_width_);
        u_lanes_.reserve(batch_width_);
        lu_lanes_.reserve(batch_width_);

        for (size_t lane = 0; lane < batch_width_; ++lane) {
            u_lane_buffers_[lane] = static_cast<double*>(resource_.allocate(lane_bytes));
            lu_lane_buffers_[lane] = static_cast<double*>(resource_.allocate(lane_bytes));

            std::fill(u_lane_buffers_[lane], u_lane_buffers_[lane] + padded_n_, 0.0);
            std::fill(lu_lane_buffers_[lane], lu_lane_buffers_[lane] + padded_n_, 0.0);

            u_lanes_.push_back(std::span<double>{u_lane_buffers_[lane], n_});
            lu_lanes_.push_back(std::span<const double>{lu_lane_buffers_[lane], n_});
        }
    }

    void allocate_arrays() {
        const size_t array_bytes = padded_n_ * sizeof(double);
        u_current_ = static_cast<double*>(resource_.allocate(array_bytes));
        u_next_    = static_cast<double*>(resource_.allocate(array_bytes));
        u_stage_   = static_cast<double*>(resource_.allocate(array_bytes));
        rhs_       = static_cast<double*>(resource_.allocate(array_bytes));
        lu_        = static_cast<double*>(resource_.allocate(array_bytes));
        psi_       = static_cast<double*>(resource_.allocate(array_bytes));

        // Zero-initialize entire buffers (including padding)
        std::fill(u_current_, u_current_ + padded_n_, 0.0);
        std::fill(u_next_, u_next_ + padded_n_, 0.0);
        std::fill(u_stage_, u_stage_ + padded_n_, 0.0);
        std::fill(rhs_, rhs_ + padded_n_, 0.0);
        std::fill(lu_, lu_ + padded_n_, 0.0);
        std::fill(psi_, psi_ + padded_n_, 0.0);
    }

    void precompute_grid_spacing() {
        const size_t dx_padded = pad_to_simd(n_ - 1);
        const size_t dx_bytes = dx_padded * sizeof(double);
        dx_ = static_cast<double*>(resource_.allocate(dx_bytes));

        for (size_t i = 0; i < n_ - 1; ++i) {
            dx_[i] = grid_[i + 1] - grid_[i];
        }
        // Zero padding for safe SIMD tail
        std::fill(dx_ + (n_ - 1), dx_ + dx_padded, 0.0);
    }

    size_t n_;
    size_t padded_n_;
    std::span<const double> grid_;  // Caller must keep alive!

    // SoA arrays (separate, SIMD-aligned)
    double* u_current_;
    double* u_next_;
    double* u_stage_;
    double* rhs_;
    double* lu_;
    double* psi_;
    double* dx_;

    // Batch support
    size_t batch_width_;  // 0 = single-contract, >0 = batch mode

    // AoS buffers (for horizontal SIMD stencil)
    double* u_batch_;   // [n * batch_width]
    double* lu_batch_;  // [n * batch_width]

    // Per-lane SoA buffers (for Newton machinery)
    std::vector<double*> u_lane_buffers_;   // batch_width buffers of [n]
    std::vector<double*> lu_lane_buffers_;  // batch_width buffers of [n]
    std::vector<std::span<double>> u_lanes_;
    std::vector<std::span<const double>> lu_lanes_;
};

} // namespace mango
