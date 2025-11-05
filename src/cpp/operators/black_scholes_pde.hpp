#pragma once

namespace mango::operators {

/**
 * BlackScholesPDE: Pure Black-Scholes PDE operator
 *
 * Implements the Black-Scholes PDE in log-moneyness coordinates x = ln(S/K):
 *   ∂V/∂t = L(V)
 *   L(V) = (σ²/2)·∂²V/∂x² + (r-d-σ²/2)·∂V/∂x - r·V
 *
 * Single Responsibility: Mathematical PDE formula only
 * - NO grid knowledge
 * - NO discretization knowledge
 * - Takes pre-computed derivatives as input
 * - Returns operator evaluation
 */
template<typename T = double>
class BlackScholesPDE {
public:
    /**
     * Construct Black-Scholes operator
     * @param sigma Volatility (σ)
     * @param r Risk-free rate
     * @param d Continuous dividend yield
     */
    BlackScholesPDE(T sigma, T r, T d)
        : half_sigma_sq_(T(0.5) * sigma * sigma)
        , drift_(r - d - half_sigma_sq_)
        , discount_rate_(r)
    {}

    /**
     * Apply operator: L(V) = (σ²/2)·∂²V/∂x² + (r-d-σ²/2)·∂V/∂x - r·V
     *
     * @param d2v_dx2 Second derivative ∂²V/∂x²
     * @param dv_dx First derivative ∂V/∂x
     * @param v Value V
     * @return L(V)
     */
    T operator()(T d2v_dx2, T dv_dx, T v) const {
        return half_sigma_sq_ * d2v_dx2 + drift_ * dv_dx - discount_rate_ * v;
    }

    /**
     * Compute first derivative coefficient: (r - d - σ²/2)
     * Used for finite difference Jacobian construction
     */
    T first_derivative_coeff() const { return drift_; }

    /**
     * Compute second derivative coefficient: σ²/2
     * Used for finite difference Jacobian construction
     */
    T second_derivative_coeff() const { return half_sigma_sq_; }

    /**
     * Compute discount rate: r
     * Used for finite difference Jacobian construction
     */
    T discount_rate() const { return discount_rate_; }

private:
    T half_sigma_sq_;    // σ²/2
    T drift_;            // r - d - σ²/2
    T discount_rate_;    // r
};

} // namespace mango::operators
