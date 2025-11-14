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
 *   auto surface_result = PriceTableSurface::create(...);
 *   auto solver = IVSolverInterpolated::create(surface_result.value());
 *
 *   // Solve for IV
 *   OptionSpec spec{
 *       .spot = 100.0,
 *       .strike = 100.0,
 *       .maturity = 1.0,
 *       .rate = 0.05,
 *       .dividend_yield = 0.02,
 *       .type = OptionType::PUT
 *   };
 *   IVQuery query{.option = spec, .market_price = 10.45};
 *
 *   IVResult result = solver->solve(query);
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

#include "src/option/iv_solver_base.hpp"
#include "src/option/option_spec.hpp"
#include "src/option/iv_types.hpp"
#include "src/interpolation/bspline_4d.hpp"
#include "src/support/expected.hpp"
#include <cmath>
#include <memory>
#include <optional>
#include <string>
#include <span>

namespace mango {

class PriceTableSurface;

/// Configuration for interpolation-based IV solver
struct IVSolverInterpolatedConfig {
    int max_iterations = 50;      ///< Maximum Newton iterations
    double tolerance = 1e-6;       ///< Price convergence tolerance
    double sigma_min = 0.01;       ///< Minimum volatility (1%)
    double sigma_max = 3.0;        ///< Maximum volatility (300%)
};

/// Interpolation-based IV Solver
///
/// Uses pre-computed B-spline price surface for ultra-fast IV calculation.
/// Solves: Find σ such that Price(m, τ, σ, r) = Market_Price
///
/// Thread-safe: Fully thread-safe for both single and batch queries (immutable spline)
class IVSolverInterpolated : public IVSolverBase {
public:
    /// Create solver from PriceTableSurface
    ///
    /// @param surface Pre-computed price table surface
    /// @param config Solver configuration
    /// @return IV solver or error message
    static expected<IVSolverInterpolated, std::string> create(
        std::shared_ptr<const BSpline4D> spline,
        double K_ref,
        std::pair<double, double> m_range,
        std::pair<double, double> tau_range,
        std::pair<double, double> sigma_range,
        std::pair<double, double> r_range,
        const IVSolverInterpolatedConfig& config = {});

    /// Convenience factory that derives metadata from PriceTableSurface
    static expected<IVSolverInterpolated, std::string> create(
        const PriceTableSurface& surface,
        const IVSolverInterpolatedConfig& config = {});

    /// Solve for implied volatility (single query)
    ///
    /// Implementation for IVSolverBase::solve().
    /// Uses Newton-Raphson method with B-spline price interpolation.
    ///
    /// @param query Option specification and market price
    /// @return IVResult with convergence status and implied volatility
    IVResult solve_impl(const IVQuery& query) const noexcept;

    /// Solve for implied volatility (batch with OpenMP)
    ///
    /// Implementation for IVSolverBase::solve_batch().
    /// Trivially parallel since B-spline is immutable and thread-safe.
    ///
    /// @param queries Input queries
    /// @param results Output buffer (must match queries.size())
    void solve_batch_impl(std::span<const IVQuery> queries,
                         std::span<IVResult> results) const noexcept;

private:
    /// Private constructor (use create() factory methods)
    IVSolverInterpolated(
        std::shared_ptr<const BSpline4D> spline,
        double K_ref,
        std::pair<double, double> m_range,
        std::pair<double, double> tau_range,
        std::pair<double, double> sigma_range,
        std::pair<double, double> r_range,
        const IVSolverInterpolatedConfig& config)
        : spline_(std::move(spline))
        , K_ref_(K_ref)
        , m_range_(m_range)
        , tau_range_(tau_range)
        , sigma_range_(sigma_range)
        , r_range_(r_range)
        , config_(config)
    {}

    std::shared_ptr<const BSpline4D> spline_;
    double K_ref_;
    std::pair<double, double> m_range_, tau_range_, sigma_range_, r_range_;
    IVSolverInterpolatedConfig config_;

    /// Evaluate option price using B-spline interpolation with strike scaling
    ///
    /// The price surface stores prices for reference strike K_ref.
    /// For options with different strikes, we scale: V(K) = V(K_ref) * (K/K_ref)
    ///
    /// @param moneyness m = S/K_ref (not S/K!)
    /// @param maturity Time to maturity
    /// @param vol Volatility
    /// @param rate Risk-free rate
    /// @param strike Actual strike (for scaling)
    /// @return Scaled price for given strike
    double eval_price(double moneyness, double maturity, double vol, double rate, double strike) const {
        double price_Kref = spline_->eval(moneyness, maturity, vol, rate);
        double scale_factor = strike / K_ref_;
        return price_Kref * scale_factor;
    }

    /// Compute vega using analytic B-spline derivative
    double compute_vega(double moneyness, double maturity, double vol, double rate, double strike) const {
        double price_unused, vega_Kref;
        spline_->eval_price_and_vega_analytic(
            moneyness, maturity, vol, rate,
            price_unused, vega_Kref);

        const double scale_factor = strike / K_ref_;
        return vega_Kref * scale_factor;
    }

    /// Check if query parameters are within surface bounds
    bool is_in_bounds(const IVQuery& query, double vol) const {
        // CRITICAL: Use K_ref for moneyness, not query.option.strike!
        const double m = query.option.spot / K_ref_;

        return m >= m_range_.first && m <= m_range_.second &&
               query.option.maturity >= tau_range_.first && query.option.maturity <= tau_range_.second &&
               vol >= sigma_range_.first && vol <= sigma_range_.second &&
               query.option.rate >= r_range_.first && query.option.rate <= r_range_.second;
    }

    /// Validate query parameters
    std::optional<std::string> validate_query(const IVQuery& query) const;

    /// Determine adaptive volatility bounds based on intrinsic value
    std::pair<double, double> adaptive_bounds(const IVQuery& query) const;
};

}  // namespace mango
