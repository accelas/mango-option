#pragma once

#include "workspace.hpp"
#include <vector>
#include <span>
#include <cstddef>

namespace mango {

/// Workspace for Newton-Raphson iteration
///
/// **Memory Strategy (Hybrid Allocation):**
/// - Allocates: 8n doubles (Jacobian: 3n-2, residual: n, delta_u: n, u_old: n, tridiag: 2n)
/// - Borrows: 2n doubles from WorkspaceStorage as scratch space (u_stage, rhs)
/// - Total: 8n allocated + 2n borrowed (vs. 11n if everything owned)
///
/// **Safety of borrowing:**
/// - u_stage: Not used during Newton (operates on u_current)
/// - rhs: Passed as const to Newton solve(), safe to reuse for Lu_perturb scratch
/// - Lu: Read-only during Jacobian build, safe to reference
///
/// **Memory reduction:** 11n → 8n allocated (27% reduction in Newton-specific memory)
class NewtonWorkspace {
public:
    /// Construct workspace borrowing scratch arrays from PDE workspace
    ///
    /// @param n Grid size
    /// @param pde_ws PDE workspace to borrow scratch space from
    NewtonWorkspace(size_t n, WorkspaceStorage& pde_ws)
        : n_(n)
        , buffer_(compute_buffer_size(n))
        , Lu_(pde_ws.lu())
        , u_perturb_(pde_ws.u_stage())
        , Lu_perturb_(pde_ws.rhs())
    {
        setup_owned_arrays();
    }

    // Owned arrays (allocated in buffer_)
    std::span<double> jacobian_lower() { return jacobian_lower_; }
    std::span<double> jacobian_diag() { return jacobian_diag_; }
    std::span<double> jacobian_upper() { return jacobian_upper_; }
    std::span<double> residual() { return residual_; }
    std::span<double> delta_u() { return delta_u_; }
    std::span<double> u_old() { return u_old_; }
    std::span<double> tridiag_workspace() { return tridiag_workspace_; }

    // Borrowed arrays (spans into PDE workspace)
    std::span<const double> Lu() const { return Lu_; }
    std::span<double> u_perturb() { return u_perturb_; }
    std::span<double> Lu_perturb() { return Lu_perturb_; }

private:
    size_t n_;
    std::vector<double> buffer_;  // Single allocation for owned arrays

    // Owned spans (point into buffer_)
    std::span<double> jacobian_lower_;      // n-1
    std::span<double> jacobian_diag_;       // n
    std::span<double> jacobian_upper_;      // n-1
    std::span<double> residual_;            // n
    std::span<double> delta_u_;             // n
    std::span<double> u_old_;               // n
    std::span<double> tridiag_workspace_;   // 2n (CRITICAL: Thomas needs 2n)

    // Borrowed spans (point into WorkspaceStorage)
    std::span<double> Lu_;          // n (read-only during Jacobian)
    std::span<double> u_perturb_;   // n (scratch, from u_stage)
    std::span<double> Lu_perturb_;  // n (scratch, from rhs)

    static constexpr size_t compute_buffer_size(size_t n) {
        // jacobian: (n-1) + n + (n-1) = 3n - 2
        // residual: n
        // delta_u: n
        // u_old: n
        // tridiag_workspace: 2n (CRITICAL FIX from design review)
        return 3*n - 2 + n + n + n + 2*n;  // = 8n - 2
    }

    void setup_owned_arrays() {
        size_t offset = 0;
        jacobian_lower_      = std::span{buffer_.data() + offset, n_ - 1}; offset += n_ - 1;
        jacobian_diag_       = std::span{buffer_.data() + offset, n_};     offset += n_;
        jacobian_upper_      = std::span{buffer_.data() + offset, n_ - 1}; offset += n_ - 1;
        residual_            = std::span{buffer_.data() + offset, n_};     offset += n_;
        delta_u_             = std::span{buffer_.data() + offset, n_};     offset += n_;
        u_old_               = std::span{buffer_.data() + offset, n_};     offset += n_;
        tridiag_workspace_   = std::span{buffer_.data() + offset, 2*n_};
    }
};

}  // namespace mango
