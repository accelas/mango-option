#pragma once

#include <cmath>
#include <cstddef>
#include <optional>
#include <string>

namespace mango {

/// Method for handling obstacle constraints in American options
enum class ObstacleMethod {
    Heuristic,  ///< Empirical active set heuristic (tuned parameters)
    PDAS        ///< Primal-Dual Active Set (Hintermüller-Ito-Kunisch)
};

/// Result from Newton-Raphson iteration for implicit PDE stages
struct NewtonResult {
    bool converged;                              ///< Convergence status
    size_t iterations;                           ///< Number of iterations performed
    double final_error;                          ///< Final error measure
    std::optional<std::string> failure_reason;   ///< Optional failure diagnostic
};

/// TR-BDF2 time-stepping configuration
///
/// TR-BDF2 is a composite two-stage method:
/// - Stage 1: Trapezoidal rule to t_n + γ·dt
/// - Stage 2: BDF2 from t_n to t_n+1
///
/// γ = 2 - √2 ≈ 0.5857864376269049 (optimal for L-stability)
///
/// Each implicit stage is solved using Newton-Raphson iteration (no obstacle)
/// or PDAS iteration (with obstacle constraint).
struct TRBDF2Config {
    /// Stage 1 parameter (γ = 2 - √2)
    double gamma = 2.0 - std::sqrt(2.0);

    /// Maximum Newton iterations per stage (no obstacle)
    size_t max_iter = 20;

    /// Convergence tolerance for Newton solver (relative error)
    double tolerance = 1e-6;

    /// Finite difference epsilon for Jacobian computation
    double jacobian_fd_epsilon = 1e-7;

    /// Obstacle constraint method (default: Heuristic until PDAS is validated)
    ObstacleMethod obstacle_method = ObstacleMethod::Heuristic;

    /// Maximum PDAS iterations per stage (with obstacle)
    size_t pdas_max_iter = 20;

    /// PDAS convergence tolerance for complementarity residual
    double pdas_tol = 1e-8;

    /// PDAS parameter β ∈ (0,1) for θ = β/L_max
    /// Literature suggests β = 0.9 (Hintermüller-Ito-Kunisch 2003)
    double pdas_beta = 0.9;

    /// Compute weight for Stage 1 update
    ///
    /// Stage 1: u^{n+γ} = u^n + w1 * [L(u^n) + L(u^{n+γ})]
    /// where w1 = γ·dt / 2
    double stage1_weight(double dt) const {
        return gamma * dt / 2.0;
    }

    /// Compute weight for Stage 2 update (BDF2 implicit weight)
    ///
    /// Stage 2: u^{n+1} - w2·L(u^{n+1}) = alpha·u^{n+γ} + beta·u^n
    /// where w2 = (1-γ)·dt / (2-γ)  (Ascher, Ruuth, Wetton 1995)
    double stage2_weight(double dt) const {
        return (1.0 - gamma) * dt / (2.0 - gamma);
    }
};

}  // namespace mango
