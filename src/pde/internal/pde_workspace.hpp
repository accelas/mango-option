// SPDX-License-Identifier: MIT
#pragma once

#include <span>
#include <cstdint>
#include <bit>
#include <expected>
#include <string>
#include <format>
#include <algorithm>
#include <limits>
#include "mango/math/tridiagonal_matrix_view.hpp"

namespace mango {

/**
 * PDEWorkspace: Named spans to caller-managed PMR buffers
 *
 * Provides zero-copy access to temporary arrays for PDE solver.
 * All arrays are padded to 8-element boundaries for SIMD safety.
 * Caller manages buffer lifetime and allocation strategy.
 *
 * Arrays (16 regular + tridiag @ 2n):
 * - dx (n-1): Grid spacing
 * - u_stage (n): Stage buffer for TR-BDF2
 * - rhs (n): Right-hand side vector
 * - lu (n): Spatial operator output
 * - psi (n): Obstacle constraint
 * - jacobian_diag (n): Jacobian main diagonal
 * - jacobian_upper (n-1): Jacobian upper diagonal
 * - jacobian_lower (n-1): Jacobian lower diagonal
 * - residual (n): Newton residual
 * - delta_u (n): Newton correction
 * - newton_u_old (n): Previous Newton iterate
 * - u_next (n): Next solution buffer
 * - reserved1 (n): Reserved for future use
 * - d2u_scratch (n): Second derivative scratch (SpatialOperator)
 * - du_scratch (n): First derivative scratch (SpatialOperator)
 * - a_f_cache (n): Per-node Il'in-fitted diffusion coefficient cache
 *   (SpatialOperator, #472) -- amortizes the std::tanh in
 *   fitted_diffusion() across the repeated apply()/assemble_jacobian()
 *   calls of a single (a, b) sample instead of paying it per node per call
 * - fitted_cache_meta (3, padded to 8): validity metadata for a_f_cache,
 *   co-located with it in this buffer rather than held by SpatialOperator
 *   (#472 follow-up) so that when two SpatialOperator instances share one
 *   PDEWorkspace, whichever last filled a_f_cache also owns the key that
 *   says so -- a stale key living on a different object can no longer read
 *   back the wrong data. Slot 0 = cached a, slot 1 = cached b (NaN until
 *   first fill), slot 2 = the sampled GridSpacing's identity as
 *   std::uintptr_t bits (0 until first fill; never compared as a double,
 *   since a pointer's bit pattern can form a NaN payload)
 * - tridiag_workspace (2n): Thomas solver workspace
 * - active_mask (n bytes): LCP active-set mask (uint8_t), carved from a
 *   double-aligned tail block so PDEWorkspace stays a single span<double>
 */
struct PDEWorkspace {
    static constexpr size_t SIMD_WIDTH = 8;

    static constexpr size_t pad_to_simd(size_t n) {
        return ((n + SIMD_WIDTH - 1) / SIMD_WIDTH) * SIMD_WIDTH;
    }

    /// Calculate required buffer size (16 arrays + tridiag @ 2n +
    /// fitted_cache_meta + active_mask tail)
    static constexpr size_t required_size(size_t n) {
        size_t n_padded = pad_to_simd(n);
        size_t n_minus_1_padded = pad_to_simd(n - 1);

        // 13 arrays @ n (padded)
        size_t regular_n = 13 * n_padded;

        // 3 arrays @ (n-1) (padded): dx, jacobian_upper, jacobian_lower
        size_t arrays_n_minus_1 = 3 * n_minus_1_padded;

        // tridiag_workspace @ 2n (padded)
        size_t tridiag = pad_to_simd(2 * n);

        // active_mask: n bytes (uint8_t), carved from the tail of this
        // double array. Round up to whole doubles, then pad to the same
        // SIMD-safe element boundary (in units of doubles) as every other
        // array here.
        size_t mask_doubles = pad_to_simd((n + sizeof(double) - 1) / sizeof(double));

        // fitted_cache_meta: 3 doubles (cached a, b, grid-key bits),
        // fixed-size (independent of n), padded to the same SIMD-safe
        // boundary as every other array here.
        size_t meta_padded = pad_to_simd(3);

        return regular_n + arrays_n_minus_1 + tridiag + mask_doubles + meta_padded;
    }

    /// Create workspace spans from buffer (without grid, dx not initialized)
    static std::expected<PDEWorkspace, std::string>
    from_buffer(std::span<double> buffer, size_t n) {
        if (n < 2) {
            return std::unexpected("Grid size must be at least 2");
        }

        size_t required = required_size(n);

        if (buffer.size() < required) {
            return std::unexpected(std::format(
                "Workspace buffer too small: {} < {} required for n={}",
                buffer.size(), required, n));
        }

        size_t n_padded = pad_to_simd(n);
        size_t n_minus_1_padded = pad_to_simd(n - 1);
        PDEWorkspace workspace;
        workspace.n_ = n;

        size_t offset = 0;

        // Slice arrays (n each, padded)
        workspace.dx_ = buffer.subspan(offset, n_minus_1_padded);
        offset += n_minus_1_padded;

        workspace.u_stage_ = buffer.subspan(offset, n_padded);
        offset += n_padded;

        workspace.rhs_ = buffer.subspan(offset, n_padded);
        offset += n_padded;

        workspace.lu_ = buffer.subspan(offset, n_padded);
        offset += n_padded;

        workspace.psi_ = buffer.subspan(offset, n_padded);
        offset += n_padded;

        workspace.jacobian_diag_ = buffer.subspan(offset, n_padded);
        offset += n_padded;

        workspace.jacobian_upper_ = buffer.subspan(offset, n_minus_1_padded);
        offset += n_minus_1_padded;

        workspace.jacobian_lower_ = buffer.subspan(offset, n_minus_1_padded);
        offset += n_minus_1_padded;

        workspace.residual_ = buffer.subspan(offset, n_padded);
        offset += n_padded;

        workspace.delta_u_ = buffer.subspan(offset, n_padded);
        offset += n_padded;

        workspace.newton_u_old_ = buffer.subspan(offset, n_padded);
        offset += n_padded;

        workspace.u_next_ = buffer.subspan(offset, n_padded);
        offset += n_padded;

        // Reserved for future (3 × n)
        workspace.reserved1_ = buffer.subspan(offset, n_padded);
        offset += n_padded;

        workspace.d2u_scratch_ = buffer.subspan(offset, n_padded);
        offset += n_padded;

        workspace.du_scratch_ = buffer.subspan(offset, n_padded);
        offset += n_padded;

        workspace.a_f_cache_ = buffer.subspan(offset, n_padded);
        offset += n_padded;

        // fitted_cache_meta: 3 doubles co-located with a_f_cache so
        // validity travels with the data (#472 follow-up). Slots 0/1
        // (cached a, b) start at NaN and slot 2 (grid-key bits) starts at
        // 0, so the first ensure_fitted_cache() call on this buffer always
        // misses regardless of which SpatialOperator touches it first.
        static_assert(sizeof(std::uintptr_t) <= sizeof(std::uint64_t),
                      "grid key must round-trip through a uint64_t bit_cast");
        size_t meta_padded = pad_to_simd(3);
        workspace.fitted_cache_meta_ = buffer.subspan(offset, meta_padded);
        offset += meta_padded;
        workspace.fitted_cache_meta_[0] = std::numeric_limits<double>::quiet_NaN();
        workspace.fitted_cache_meta_[1] = std::numeric_limits<double>::quiet_NaN();
        workspace.fitted_cache_meta_[2] = std::bit_cast<double>(std::uint64_t{0});

        // tridiag_workspace (2n, padded)
        size_t tridiag_padded = pad_to_simd(2 * n);
        workspace.tridiag_workspace_ = buffer.subspan(offset, tridiag_padded);
        offset += tridiag_padded;

        // active_mask: n bytes (uint8_t), carved from a double-aligned tail
        // block. Reinterpreting a live double array's storage as
        // unsigned-char-family bytes is well-defined (the aliasing
        // exception for char types), so no lifetime start is needed.
        size_t mask_doubles = pad_to_simd((n + sizeof(double) - 1) / sizeof(double));
        auto mask_storage = buffer.subspan(offset, mask_doubles);
        workspace.active_mask_ = std::span<uint8_t>(
            reinterpret_cast<uint8_t*>(mask_storage.data()),
            mask_doubles * sizeof(double));

        return workspace;
    }

    /// Create workspace spans from buffer and initialize dx from grid
    static std::expected<PDEWorkspace, std::string>
    from_buffer_and_grid(std::span<double> buffer,
                        std::span<const double> grid,
                        size_t n) {
        if (grid.size() != n) {
            return std::unexpected(std::format(
                "Grid size mismatch: {} != {}", grid.size(), n));
        }

        auto workspace_result = from_buffer(buffer, n);
        if (!workspace_result.has_value()) {
            return std::unexpected(workspace_result.error());
        }

        auto workspace = workspace_result.value();

        // Compute dx from grid
        auto dx_span = workspace.dx();
        for (size_t i = 0; i < n - 1; ++i) {
            dx_span[i] = grid[i + 1] - grid[i];
        }

        return workspace;
    }

    // Accessors - return logical size spans (not padded)

    std::span<double> dx() { return dx_.subspan(0, n_ - 1); }
    std::span<const double> dx() const { return dx_.subspan(0, n_ - 1); }

    std::span<double> u_stage() { return u_stage_.subspan(0, n_); }
    std::span<const double> u_stage() const { return u_stage_.subspan(0, n_); }

    std::span<double> rhs() { return rhs_.subspan(0, n_); }
    std::span<const double> rhs() const { return rhs_.subspan(0, n_); }

    std::span<double> lu() { return lu_.subspan(0, n_); }
    std::span<const double> lu() const { return lu_.subspan(0, n_); }

    std::span<double> psi() { return psi_.subspan(0, n_); }
    std::span<const double> psi() const { return psi_.subspan(0, n_); }

    std::span<double> jacobian_diag() { return jacobian_diag_.subspan(0, n_); }
    std::span<const double> jacobian_diag() const { return jacobian_diag_.subspan(0, n_); }

    std::span<double> jacobian_upper() { return jacobian_upper_.subspan(0, n_ - 1); }
    std::span<const double> jacobian_upper() const { return jacobian_upper_.subspan(0, n_ - 1); }

    std::span<double> jacobian_lower() { return jacobian_lower_.subspan(0, n_ - 1); }
    std::span<const double> jacobian_lower() const { return jacobian_lower_.subspan(0, n_ - 1); }

    std::span<double> residual() { return residual_.subspan(0, n_); }
    std::span<const double> residual() const { return residual_.subspan(0, n_); }

    std::span<double> delta_u() { return delta_u_.subspan(0, n_); }
    std::span<const double> delta_u() const { return delta_u_.subspan(0, n_); }

    std::span<double> newton_u_old() { return newton_u_old_.subspan(0, n_); }
    std::span<const double> newton_u_old() const { return newton_u_old_.subspan(0, n_); }

    std::span<double> u_next() { return u_next_.subspan(0, n_); }
    std::span<const double> u_next() const { return u_next_.subspan(0, n_); }

    std::span<double> reserved1() { return reserved1_.subspan(0, n_); }
    std::span<const double> reserved1() const { return reserved1_.subspan(0, n_); }

    std::span<double> d2u_scratch() { return d2u_scratch_.subspan(0, n_); }
    std::span<const double> d2u_scratch() const { return d2u_scratch_.subspan(0, n_); }

    std::span<double> du_scratch() { return du_scratch_.subspan(0, n_); }
    std::span<const double> du_scratch() const { return du_scratch_.subspan(0, n_); }

    /// Per-node fitted-diffusion coefficient cache (SpatialOperator, #472).
    /// Owned by the workspace so it survives across the repeated
    /// apply()/assemble_jacobian() calls of one Newton stage without a
    /// per-call heap allocation. Validity is tracked by fitted_cache_meta()
    /// below, co-located in this same buffer so it travels with the data.
    std::span<double> a_f_cache() { return a_f_cache_.subspan(0, n_); }
    std::span<const double> a_f_cache() const { return a_f_cache_.subspan(0, n_); }

    /// Validity metadata for a_f_cache (#472 follow-up): slot 0 = cached
    /// a, slot 1 = cached b, slot 2 = the sampled GridSpacing's identity
    /// as std::uintptr_t bits (see SpatialOperator::ensure_fitted_cache).
    /// Co-located with a_f_cache in this workspace buffer -- rather than
    /// held as members on SpatialOperator -- so that when two
    /// SpatialOperator instances share one PDEWorkspace, the key that
    /// says whether a_f_cache is valid lives with the data it describes:
    /// whichever operator last filled the cache also owns the key, so a
    /// second operator reading a stale local key can no longer read back
    /// data it didn't write. Behavior when two operators alternate over
    /// one workspace is "correct but recomputes on every switch" rather
    /// than "corrupt" -- this is a shared single-slot cache, not a
    /// per-operator one.
    std::span<double> fitted_cache_meta() { return fitted_cache_meta_.subspan(0, 3); }
    std::span<const double> fitted_cache_meta() const { return fitted_cache_meta_.subspan(0, 3); }

    std::span<double> tridiag_workspace() { return tridiag_workspace_.subspan(0, 2 * n_); }
    std::span<const double> tridiag_workspace() const { return tridiag_workspace_.subspan(0, 2 * n_); }

    /// LCP active-set mask: active_mask()[i] == 1 iff node i is clamped to
    /// the obstacle. Written by solve_thomas_projected2, consumed by
    /// validate_lcp_kkt.
    std::span<uint8_t> active_mask() { return active_mask_.subspan(0, n_); }
    std::span<const uint8_t> active_mask() const { return active_mask_.subspan(0, n_); }

    /// Get TridiagonalMatrixView providing unified access to tridiagonal Jacobian
    ///
    /// This is the preferred way to access the Jacobian matrix. It provides
    /// type safety and clearer intent than accessing the three arrays separately.
    TridiagonalMatrixView jacobian() {
        return TridiagonalMatrixView(jacobian_lower(), jacobian_diag(), jacobian_upper());
    }

    size_t size() const { return n_; }

private:
    size_t n_;
    std::span<double> dx_;
    std::span<double> u_stage_;
    std::span<double> rhs_;
    std::span<double> lu_;
    std::span<double> psi_;
    std::span<double> jacobian_diag_;
    std::span<double> jacobian_upper_;
    std::span<double> jacobian_lower_;
    std::span<double> residual_;
    std::span<double> delta_u_;
    std::span<double> newton_u_old_;
    std::span<double> u_next_;
    std::span<double> tridiag_workspace_;
    std::span<double> reserved1_;
    std::span<double> d2u_scratch_;
    std::span<double> du_scratch_;
    std::span<double> a_f_cache_;
    std::span<double> fitted_cache_meta_;
    std::span<uint8_t> active_mask_;
};

}  // namespace mango
