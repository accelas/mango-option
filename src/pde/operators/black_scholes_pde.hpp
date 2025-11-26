/**
 * @file black_scholes_pde.hpp
 * @brief Black-Scholes PDE operator in log-moneyness coordinates
 */

#pragma once

#include <type_traits>
#include <functional>

namespace mango::operators {

/**
 * BlackScholesPDE: Black-Scholes PDE operator in log-moneyness coordinates
 *
 * Implements the Black-Scholes PDE in log-moneyness coordinates x = ln(S/K):
 *   ∂V/∂t = L(V)
 *   L(V) = (σ²/2)·∂²V/∂x² + (r(t)-d-σ²/2)·∂V/∂x - r(t)·V
 *
 * Supports both constant rate and time-varying rate via callable.
 *
 * @tparam T Scalar type (double)
 * @tparam RateFn Rate function type: double(double) or similar callable
 */
template<typename T = double, typename RateFn = T>
class BlackScholesPDE {
public:
    /**
     * Construct with callable rate function
     *
     * @param sigma Volatility
     * @param rate_fn Rate function: rate_fn(t) -> r(t)
     * @param d Continuous dividend yield
     */
    template<typename Fn,
             typename = std::enable_if_t<std::is_invocable_r_v<T, Fn, double>>>
    BlackScholesPDE(T sigma, Fn&& rate_fn, T d)
        : half_sigma_sq_(T(0.5) * sigma * sigma)
        , dividend_(d)
        , rate_fn_(std::forward<Fn>(rate_fn))
    {}

    /**
     * Construct with constant rate (backward compatible)
     *
     * @param sigma Volatility
     * @param r Constant risk-free rate
     * @param d Continuous dividend yield
     */
    template<typename U = RateFn,
             typename = std::enable_if_t<std::is_same_v<U, T>>>
    BlackScholesPDE(T sigma, T r, T d)
        : half_sigma_sq_(T(0.5) * sigma * sigma)
        , dividend_(d)
        , rate_fn_(r)
    {}

    /**
     * Apply operator with time parameter (for time-varying rate)
     *
     * L(V) = (σ²/2)·∂²V/∂x² + (r(t)-d-σ²/2)·∂V/∂x - r(t)·V
     *
     * @param d2v_dx2 Second derivative ∂²V/∂x²
     * @param dv_dx First derivative ∂V/∂x
     * @param v Value V
     * @param t Current time
     * @return L(V)
     */
    T operator()(T d2v_dx2, T dv_dx, T v, double t) const {
        T r = get_rate(t);
        T drift = r - dividend_ - half_sigma_sq_;
        return half_sigma_sq_ * d2v_dx2 + drift * dv_dx - r * v;
    }

    /**
     * Apply operator without time (backward compatible, for constant rate)
     */
    T operator()(T d2v_dx2, T dv_dx, T v) const {
        return (*this)(d2v_dx2, dv_dx, v, 0.0);
    }

    /**
     * Compute first derivative coefficient: (r - d - σ²/2)
     * Used for analytical Jacobian construction
     */
    T first_derivative_coeff(double t = 0.0) const {
        return get_rate(t) - dividend_ - half_sigma_sq_;
    }

    /**
     * Compute second derivative coefficient: σ²/2
     * Used for analytical Jacobian construction
     */
    T second_derivative_coeff() const { return half_sigma_sq_; }

    /**
     * Compute discount rate: r
     * Used for analytical Jacobian construction
     */
    T discount_rate(double t = 0.0) const { return get_rate(t); }

private:
    T get_rate(double t) const {
        if constexpr (std::is_invocable_v<RateFn, double>) {
            return rate_fn_(t);
        } else {
            return rate_fn_;  // Constant rate
        }
    }

    T half_sigma_sq_;    // σ²/2
    T dividend_;         // d
    RateFn rate_fn_;     // r(t) or constant r
};

// Deduction guides
template<typename T, typename Fn>
BlackScholesPDE(T, Fn&&, T) -> BlackScholesPDE<T, std::decay_t<Fn>>;

template<typename T>
BlackScholesPDE(T, T, T) -> BlackScholesPDE<T, T>;

} // namespace mango::operators
