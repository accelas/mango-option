#pragma once

#include <span>
#include <expected>
#include <string>
#include <format>
#include <algorithm>

namespace mango {

/**
 * PDEWorkspace: Named spans to caller-managed PMR buffers
 *
 * Provides zero-copy access to temporary arrays for PDE solver.
 * All arrays are padded to 8-element boundaries for SIMD safety.
 * Caller manages buffer lifetime and allocation strategy.
 *
 * Arrays (15 regular + tridiag @ 2n):
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
 * - reserved1-3 (3 × n): Reserved for future use
 * - tridiag_workspace (2n): Thomas solver workspace
 */
struct PDEWorkspace {
    static constexpr size_t SIMD_WIDTH = 8;

    static constexpr size_t pad_to_simd(size_t n) {
        return ((n + SIMD_WIDTH - 1) / SIMD_WIDTH) * SIMD_WIDTH;
    }

    /// Calculate required buffer size (15 arrays + tridiag @ 2n)
    static size_t required_size(size_t n) {
        size_t n_padded = pad_to_simd(n);
        size_t n_minus_1_padded = pad_to_simd(n - 1);

        // 12 arrays @ n (padded)
        size_t regular_n = 12 * n_padded;

        // 3 arrays @ (n-1) (padded): dx, jacobian_upper, jacobian_lower
        size_t arrays_n_minus_1 = 3 * n_minus_1_padded;

        // tridiag_workspace @ 2n (padded)
        size_t tridiag = pad_to_simd(2 * n);

        return regular_n + arrays_n_minus_1 + tridiag;
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

        workspace.reserved2_ = buffer.subspan(offset, n_padded);
        offset += n_padded;

        workspace.reserved3_ = buffer.subspan(offset, n_padded);
        offset += n_padded;

        // tridiag_workspace (2n, padded)
        size_t tridiag_padded = pad_to_simd(2 * n);
        workspace.tridiag_workspace_ = buffer.subspan(offset, tridiag_padded);

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

    std::span<double> tridiag_workspace() { return tridiag_workspace_.subspan(0, 2 * n_); }
    std::span<const double> tridiag_workspace() const { return tridiag_workspace_.subspan(0, 2 * n_); }

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
    std::span<double> reserved2_;
    std::span<double> reserved3_;
};

}  // namespace mango
