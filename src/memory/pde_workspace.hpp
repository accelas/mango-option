#pragma once

#include "workspace_base.hpp"
#include <span>
#include <cassert>
#include <algorithm>

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
                         size_t initial_buffer_size = 1024 * 1024)
        : WorkspaceBase(initial_buffer_size)
        , n_(n)
        , padded_n_(pad_to_simd(n))
        , grid_(grid)
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

private:
    void allocate_and_initialize() {
        allocate_arrays();
        precompute_grid_spacing();
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
};

} // namespace mango
