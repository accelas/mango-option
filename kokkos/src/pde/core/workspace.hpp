#pragma once

#include <Kokkos_Core.hpp>
#include <expected>
#include "kokkos/src/support/execution_space.hpp"

namespace mango::kokkos {

/// Workspace error codes
enum class WorkspaceError {
    InvalidSize,
    AllocationFailed
};

/// PDE solver workspace with Kokkos Views
///
/// Owns all temporary buffers needed by TR-BDF2 solver.
/// Template on MemSpace for CPU/GPU portability.
template <typename MemSpace>
class PDEWorkspace {
public:
    using view_type = Kokkos::View<double*, MemSpace>;

    /// Factory method
    [[nodiscard]] static std::expected<PDEWorkspace, WorkspaceError>
    create(size_t n) {
        if (n < 2) {
            return std::unexpected(WorkspaceError::InvalidSize);
        }

        PDEWorkspace ws;
        ws.n_ = n;

        // Allocate all buffers
        ws.u_stage_ = view_type("u_stage", n);
        ws.rhs_ = view_type("rhs", n);
        ws.lu_ = view_type("lu", n);
        ws.psi_ = view_type("psi", n);

        // Jacobian (tridiagonal)
        ws.jacobian_diag_ = view_type("jacobian_diag", n);
        ws.jacobian_lower_ = view_type("jacobian_lower", n - 1);
        ws.jacobian_upper_ = view_type("jacobian_upper", n - 1);

        // Newton iteration
        ws.residual_ = view_type("residual", n);
        ws.delta_u_ = view_type("delta_u", n);

        // Thomas solver workspace
        ws.thomas_c_prime_ = view_type("thomas_c_prime", n);
        ws.thomas_d_prime_ = view_type("thomas_d_prime", n);

        return ws;
    }

    // Accessors
    [[nodiscard]] size_t n() const { return n_; }

    [[nodiscard]] view_type u_stage() const { return u_stage_; }
    [[nodiscard]] view_type rhs() const { return rhs_; }
    [[nodiscard]] view_type lu() const { return lu_; }
    [[nodiscard]] view_type psi() const { return psi_; }

    [[nodiscard]] view_type jacobian_diag() const { return jacobian_diag_; }
    [[nodiscard]] view_type jacobian_lower() const { return jacobian_lower_; }
    [[nodiscard]] view_type jacobian_upper() const { return jacobian_upper_; }

    [[nodiscard]] view_type residual() const { return residual_; }
    [[nodiscard]] view_type delta_u() const { return delta_u_; }

    [[nodiscard]] view_type thomas_c_prime() const { return thomas_c_prime_; }
    [[nodiscard]] view_type thomas_d_prime() const { return thomas_d_prime_; }

private:
    PDEWorkspace() = default;

    size_t n_ = 0;

    view_type u_stage_;
    view_type rhs_;
    view_type lu_;
    view_type psi_;

    view_type jacobian_diag_;
    view_type jacobian_lower_;
    view_type jacobian_upper_;

    view_type residual_;
    view_type delta_u_;

    view_type thomas_c_prime_;
    view_type thomas_d_prime_;
};

}  // namespace mango::kokkos
