#pragma once

#include <cmath>
#include <cstddef>

namespace mango {

/// TR-BDF2 time-stepping configuration
///
/// TR-BDF2 is a composite two-stage method:
/// - Stage 1: Trapezoidal rule to t_n + γ·dt
/// - Stage 2: BDF2 from t_n to t_n+1
///
/// γ = 2 - √2 ≈ 0.5857864376269049 (optimal for L-stability)
struct TRBDF2Config {
    /// Maximum iterations for implicit solver
    size_t max_iter = 100;

    /// Convergence tolerance (relative error)
    double tolerance = 1e-6;

    /// Stage 1 parameter (γ = 2 - √2)
    double gamma = 2.0 - std::sqrt(2.0);

    /// Under-relaxation parameter for fixed-point iteration
    double omega = 0.7;

    /// Cache blocking threshold (apply blocking when n >= threshold)
    size_t cache_blocking_threshold = 5000;

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
    /// where w2 = (1-γ)²·dt / (γ(2γ-1))
    double stage2_weight(double dt) const {
        return (1.0 - gamma) * (1.0 - gamma) * dt / (gamma * (2.0 * gamma - 1.0));
    }
};

}  // namespace mango
