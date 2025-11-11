/**
 * @file iv_solver_interpolated.hpp
 * @brief Implied volatility solver using B-spline price interpolation
 *
 * Solves for implied volatility using Newton's method with interpolated
 * option prices from pre-computed 4D B-spline surface. Achieves ~30µs
 * IV calculation vs ~143ms with FDM (4,800× speedup).
 *
 * Usage:
 *   // After building price table
 *   BSpline4D_FMA price_surface = ...;  // from PriceTable4DBuilder
 *
 *   // Create IV solver
 *   IVSolverInterpolated iv_solver(price_surface, K_ref);
 *
 *   // Solve for IV
 *   IVQuery query{
 *       .market_price = 10.45,
 *       .spot = 100.0,
 *       .maturity = 1.0,
 *       .rate = 0.05
 *   };
 *
 *   auto result = iv_solver.solve(query);
 *   if (result.converged) {
 *       std::cout << "IV: " << result.implied_vol << "\n";
 *   }
 *
 * Algorithm:
 * - Newton-Raphson iteration: σ_{n+1} = σ_n - f(σ_n)/f'(σ_n)
 * - f(σ) = Price(m, τ, σ, r) - Market_Price
 * - f'(σ) = Vega(m, τ, σ, r) ≈ [Price(σ+ε) - Price(σ-ε)] / (2ε)
 * - Adaptive bounds based on intrinsic value analysis
 * - Typical convergence: 3-5 iterations
 *
 * Performance:
 * - B-spline eval: ~500ns per price query
 * - Vega computation: ~1µs (2 B-spline evals + FD)
 * - Newton iterations: 3-5 typical
 * - Total IV solve: ~10-30µs
 */

#pragma once

#include "src/interpolation/bspline_4d.hpp"
#include "src/option/american_option.hpp"  // For OptionType enum
#include "src/option/iv_types.hpp"
#include <cmath>
#include <optional>
#include <string>

namespace mango {

/// Query parameters for IV calculation
struct IVQuery {
    double market_price;  ///< Observed option price in market
    double spot;          ///< Current underlying price
    double strike;        ///< Strike price (use K_ref if not specified)
    double maturity;      ///< Time to maturity in years
    double rate;          ///< Risk-free rate
    OptionType option_type = OptionType::PUT;  ///< CALL or PUT (default PUT for backwards compatibility)
};

/// Configuration for IV solver
struct IVSolverConfig {
    int max_iterations = 50;      ///< Maximum Newton iterations
    double tolerance = 1e-6;       ///< Price convergence tolerance
    double vega_epsilon = 1e-4;    ///< Finite difference step for vega
    double sigma_min = 0.01;       ///< Minimum volatility (1%)
    double sigma_max = 3.0;        ///< Maximum volatility (300%)
};

/// Interpolation-based IV Solver
///
/// Uses pre-computed B-spline price surface for ultra-fast IV calculation.
/// Solves: Find σ such that Price(m, τ, σ, r) = Market_Price
///
/// Thread-safe: Each thread should have its own IVSolverInterpolated instance
class IVSolverInterpolated {
public:
    /// Constructor
    ///
    /// @param price_surface Pre-computed 4D B-spline price evaluator
    /// @param K_ref Reference strike price used for price table construction
    /// @param m_range Moneyness bounds (min, max) - queries outside are rejected
    /// @param tau_range Maturity bounds (min, max)
    /// @param sigma_range Volatility bounds (min, max)
    /// @param r_range Rate bounds (min, max)
    /// @param config Solver configuration
    IVSolverInterpolated(
        const BSpline4D_FMA& price_surface,
        double K_ref,
        std::pair<double, double> m_range,
        std::pair<double, double> tau_range,
        std::pair<double, double> sigma_range,
        std::pair<double, double> r_range,
        const IVSolverConfig& config = {})
        : price_surface_(price_surface)
        , K_ref_(K_ref)
        , m_range_(m_range)
        , tau_range_(tau_range)
        , sigma_range_(sigma_range)
        , r_range_(r_range)
        , config_(config)
    {
        if (K_ref <= 0.0) {
            throw std::invalid_argument("K_ref must be positive");
        }
    }

    /// Solve for implied volatility
    ///
    /// @param query Market data and option parameters
    /// @return IV result with convergence status
    IVResult solve(const IVQuery& query) const;

private:
    const BSpline4D_FMA& price_surface_;
    double K_ref_;
    std::pair<double, double> m_range_, tau_range_, sigma_range_, r_range_;
    IVSolverConfig config_;

    /// Evaluate option price using B-spline interpolation with strike scaling
    ///
    /// The price surface stores prices for reference strike K_ref.
    /// For options with different strikes, we scale: V(K) = V(K_ref) * (K/K_ref)
    ///
    /// @param moneyness m = S/K
    /// @param maturity Time to maturity
    /// @param vol Volatility
    /// @param rate Risk-free rate
    /// @param strike Actual strike (for scaling)
    /// @return Scaled price for given strike
    double eval_price(double moneyness, double maturity, double vol, double rate, double strike) const {
        double price_Kref = price_surface_.eval(moneyness, maturity, vol, rate);
        double scale_factor = strike / K_ref_;
        return price_Kref * scale_factor;
    }

    /// Compute vega using finite differences
    double compute_vega(double moneyness, double maturity, double vol, double rate, double strike) const {
        const double eps = config_.vega_epsilon;

        const double price_up = eval_price(moneyness, maturity, vol + eps, rate, strike);
        const double price_dn = eval_price(moneyness, maturity, vol - eps, rate, strike);

        return (price_up - price_dn) / (2.0 * eps);
    }

    /// Check if query parameters are within surface bounds
    bool is_in_bounds(const IVQuery& query, double vol) const {
        // CRITICAL: Use K_ref for moneyness, not query.strike!
        // The surface is built with m = S/K_ref
        const double m = query.spot / K_ref_;

        return m >= m_range_.first && m <= m_range_.second &&
               query.maturity >= tau_range_.first && query.maturity <= tau_range_.second &&
               vol >= sigma_range_.first && vol <= sigma_range_.second &&
               query.rate >= r_range_.first && query.rate <= r_range_.second;
    }

    /// Validate query parameters
    std::optional<std::string> validate_query(const IVQuery& query) const;

    /// Determine adaptive volatility bounds based on intrinsic value
    std::pair<double, double> adaptive_bounds(const IVQuery& query) const;
};

}  // namespace mango
