// SPDX-License-Identifier: MIT
/**
 * @file iv_solver_interpolated.hpp
 * @brief Implied volatility solver using B-spline price interpolation
 *
 * Solves for implied volatility using Newton's method with interpolated
 * option prices from pre-computed 4D B-spline surface. Achieves ~30us
 * IV calculation vs ~19ms with FDM (5,400x speedup).
 *
 * Usage:
 *   // After building price table
 *   auto surface_result = PriceTableSurface::create(...);
 *   auto solver = IVSolverInterpolatedStandard::create(surface_result.value());
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
 *   auto result = solver->solve(query);
 *   if (result.has_value()) {
 *       std::cout << "IV: " << result->implied_vol << "\n";
 *   } else {
 *       std::cerr << "Error: " <<  "Error code: " << static_cast<int>(result.error().code) << "\n";
 *   }
 *
 * Algorithm:
 * - Newton-Raphson iteration: s_{n+1} = s_n - f(s_n)/f'(s_n)
 * - f(s) = Price(m, tau, s, r) - Market_Price
 * - f'(s) = Vega(m, tau, s, r) ~ [Price(s+e) - Price(s-e)] / (2e)
 * - Adaptive bounds based on intrinsic value analysis
 * - Typical convergence: 3-5 iterations
 *
 * Performance:
 * - B-spline eval: ~500ns per price query
 * - Vega computation: ~1us (2 B-spline evals + FD)
 * - Newton iterations: 3-5 typical
 * - Total IV solve: ~10-30us
 *
 * Template parameter:
 * - Surface must satisfy the PriceSurface concept (price, vega, bounds)
 */

#pragma once

#include "src/option/option_spec.hpp"
#include "src/option/iv_result.hpp"
#include "src/option/table/price_surface_concept.hpp"
#include "src/option/table/american_price_surface.hpp"
#include "src/support/error_types.hpp"
#include "src/support/parallel.hpp"
#include "src/math/root_finding.hpp"
#include <expected>
#include <cmath>
#include <algorithm>
#include <vector>
#include <optional>

namespace mango {

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
/// Solves: Find s such that Price(m, tau, s, r) = Market_Price
///
/// Thread-safe: Fully thread-safe for both single and batch queries (immutable spline)
///
/// Rate handling: The price surface uses a scalar rate axis (designed for SOFR/flat rates).
/// When a YieldCurve is provided, it is collapsed to a zero rate: -ln(D(T))/T.
/// This provides a reasonable approximation but does not capture term structure dynamics.
/// For full yield curve support, use IVSolverFDM instead.
/// When rate approximation is used, IVSuccess::used_rate_approximation is set to true.
///
/// @tparam Surface A type satisfying the PriceSurface concept
template <PriceSurface Surface>
class IVSolverInterpolated {
public:
    /// Create solver from a PriceSurface
    ///
    /// The surface must provide price(), vega(), and bounds accessors.
    ///
    /// @param surface Pre-built price surface
    /// @param config Solver configuration
    /// @return IV solver or ValidationError
    static std::expected<IVSolverInterpolated, ValidationError> create(
        Surface surface,
        const IVSolverInterpolatedConfig& config = {});

    /// Solve for implied volatility (single query)
    ///
    /// Uses Newton-Raphson method with B-spline price interpolation.
    ///
    /// @param query Option specification and market price
    /// @return Success with IV and diagnostics, or error with details
    std::expected<IVSuccess, IVError> solve(const IVQuery& query) const noexcept;

    /// Solve for implied volatility (batch with OpenMP)
    ///
    /// Trivially parallel since B-spline is immutable and thread-safe.
    ///
    /// @param queries Input queries (as vector for convenience)
    /// @return BatchIVResult with individual results and failure count
    BatchIVResult solve_batch(const std::vector<IVQuery>& queries) const noexcept;

private:
    /// Private constructor (use create() factory method)
    IVSolverInterpolated(
        Surface surface,
        std::pair<double, double> m_range,
        std::pair<double, double> tau_range,
        std::pair<double, double> sigma_range,
        std::pair<double, double> r_range,
        OptionType option_type,
        double dividend_yield,
        const IVSolverInterpolatedConfig& config)
        : surface_(std::move(surface))
        , m_range_(m_range)
        , tau_range_(tau_range)
        , sigma_range_(sigma_range)
        , r_range_(r_range)
        , config_(config)
        , option_type_(option_type)
        , dividend_yield_(dividend_yield)
    {}

    Surface surface_;
    std::pair<double, double> m_range_, tau_range_, sigma_range_, r_range_;
    IVSolverInterpolatedConfig config_;
    OptionType option_type_;
    double dividend_yield_;

    /// Evaluate option price using B-spline interpolation with strike scaling
    double eval_price(double moneyness, double maturity, double vol, double rate, double strike) const;

    /// Compute vega using partial derivative w.r.t. volatility (axis 2)
    double compute_vega(double moneyness, double maturity, double vol, double rate, double strike) const;

    /// Check if query parameters are within surface bounds
    bool is_in_bounds(const IVQuery& query, double vol) const {
        const double m = query.spot / query.strike;

        // Extract zero rate for bounds check - must match what solve uses
        // Using get_zero_rate() ensures consistency: -ln(D(T))/T for curves
        double rate_value = get_zero_rate(query.rate, query.maturity);

        return m >= m_range_.first && m <= m_range_.second &&
               query.maturity >= tau_range_.first && query.maturity <= tau_range_.second &&
               vol >= sigma_range_.first && vol <= sigma_range_.second &&
               rate_value >= r_range_.first && rate_value <= r_range_.second;
    }

    /// Validate query parameters
    std::optional<ValidationError> validate_query(const IVQuery& query) const;

    /// Determine adaptive volatility bounds based on intrinsic value
    std::pair<double, double> adaptive_bounds(const IVQuery& query) const;
};

/// Type alias for backward compatibility: IVSolverInterpolated with AmericanPriceSurface
using IVSolverInterpolatedStandard = IVSolverInterpolated<AmericanPriceSurface>;

// =====================================================================
// Template implementation (must be in header for template instantiation)
// =====================================================================

template <PriceSurface Surface>
std::expected<IVSolverInterpolated<Surface>, ValidationError>
IVSolverInterpolated<Surface>::create(
    Surface surface,
    const IVSolverInterpolatedConfig& config)
{
    // Use concept accessors for bounds extraction
    auto m_range = std::make_pair(surface.m_min(), surface.m_max());
    auto tau_range = std::make_pair(surface.tau_min(), surface.tau_max());
    auto sigma_range = std::make_pair(surface.sigma_min(), surface.sigma_max());
    auto r_range = std::make_pair(surface.rate_min(), surface.rate_max());

    // Validate bounds
    if (m_range.first >= m_range.second ||
        tau_range.first >= tau_range.second ||
        sigma_range.first >= sigma_range.second ||
        r_range.first >= r_range.second) {
        return std::unexpected(ValidationError(ValidationErrorCode::InvalidGridSize, 0.0));
    }

    auto option_type = surface.option_type();
    auto dividend_yield = surface.dividend_yield();

    return IVSolverInterpolated(
        std::move(surface),
        m_range,
        tau_range,
        sigma_range,
        r_range,
        option_type,
        dividend_yield,
        config);
}

template <PriceSurface Surface>
double IVSolverInterpolated<Surface>::eval_price(
    double moneyness, double maturity, double vol, double rate, double strike) const
{
    double spot = moneyness * strike;
    return surface_.price(spot, strike, maturity, vol, rate);
}

template <PriceSurface Surface>
double IVSolverInterpolated<Surface>::compute_vega(
    double moneyness, double maturity, double vol, double rate, double strike) const
{
    double spot = moneyness * strike;
    return surface_.vega(spot, strike, maturity, vol, rate);
}

template <PriceSurface Surface>
std::optional<ValidationError>
IVSolverInterpolated<Surface>::validate_query(const IVQuery& query) const
{
    if (query.type != option_type_) {
        return ValidationError{ValidationErrorCode::OptionTypeMismatch,
            static_cast<double>(query.type), 0};
    }

    if (std::abs(query.dividend_yield - dividend_yield_) > 1e-10) {
        return ValidationError{ValidationErrorCode::DividendYieldMismatch,
            query.dividend_yield, 0};
    }

    // Use common validation for option spec, market price, and arbitrage checks
    auto validation = validate_iv_query(query);
    if (!validation.has_value()) {
        return validation.error();
    }

    return std::nullopt;
}

template <PriceSurface Surface>
std::pair<double, double>
IVSolverInterpolated<Surface>::adaptive_bounds(const IVQuery& query) const
{
    // Compute intrinsic value based on option type
    double intrinsic;
    if (query.type == OptionType::CALL) {
        intrinsic = std::max(query.spot - query.strike, 0.0);
    } else {  // PUT
        intrinsic = std::max(query.strike - query.spot, 0.0);
    }

    // Analyze time value to set adaptive bounds
    const double time_value = query.market_price - intrinsic;
    const double time_value_pct = time_value / query.market_price;

    double sigma_upper;
    if (time_value_pct > 0.5) {
        sigma_upper = 3.0;  // 300%
    } else if (time_value_pct > 0.2) {
        sigma_upper = 2.0;  // 200%
    } else {
        sigma_upper = 1.5;  // 150%
    }

    double sigma_min = std::max(config_.sigma_min, sigma_range_.first);
    double sigma_max = std::min({sigma_upper, config_.sigma_max, sigma_range_.second});

    if (sigma_min >= sigma_max) {
        sigma_min = sigma_range_.first;
        sigma_max = sigma_range_.second;
    }

    return {sigma_min, sigma_max};
}

template <PriceSurface Surface>
std::expected<IVSuccess, IVError>
IVSolverInterpolated<Surface>::solve(const IVQuery& query) const noexcept
{
    // Validate input using centralized validation
    auto error = validate_query(query);
    if (error.has_value()) {
        // Convert ValidationError to IVError using shared mapping
        return std::unexpected(validation_error_to_iv_error(*error));
    }

    const double moneyness = query.spot / query.strike;

    // Get adaptive bounds
    auto [sigma_min, sigma_max] = adaptive_bounds(query);

    // Check if query is within surface bounds
    if (!is_in_bounds(query, sigma_min) || !is_in_bounds(query, sigma_max)) {
        return std::unexpected(IVError{
            .code = IVErrorCode::InvalidGridConfig,
            .iterations = 0,
            .final_error = 0.0,
            .last_vol = std::nullopt
        });
    }

    // Extract zero rate for surface lookup
    // For yield curves, use zero rate = -ln(D(T))/T which matches how surfaces are built
    // Using instantaneous forward rate curve.rate(T) would be incorrect as it only
    // reflects the rate at maturity, not the integrated discount factor
    //
    // Note: When a YieldCurve is provided, we collapse it to a single zero rate.
    // This loses term structure dynamics. For full curve support, use IVSolverFDM.
    const bool rate_is_curve = is_yield_curve(query.rate);
    double rate_value = get_zero_rate(query.rate, query.maturity);

    // Define objective function: f(s) = Price(s) - Market_Price
    auto objective = [&](double sigma) -> double {
        return eval_price(moneyness, query.maturity, sigma, rate_value, query.strike) - query.market_price;
    };

    // Define derivative (vega): df/ds = dPrice/ds
    auto derivative = [&](double sigma) -> double {
        return compute_vega(moneyness, query.maturity, sigma, rate_value, query.strike);
    };

    // Use generic bounded Newton-Raphson
    RootFindingConfig newton_config{
        .max_iter = static_cast<size_t>(std::max(0, config_.max_iterations)),
        .tolerance = config_.tolerance
    };

    const double sigma0 = (sigma_min + sigma_max) / 2.0;  // Initial guess
    auto result = newton_find_root(objective, derivative, sigma0, sigma_min, sigma_max, newton_config);

    // Check convergence - transform RootFindingError to IVError
    if (!result.has_value()) {
        const auto& root_error = result.error();
        IVErrorCode error_code;
        switch (root_error.code) {
            case RootFindingErrorCode::MaxIterationsExceeded:
                error_code = IVErrorCode::MaxIterationsExceeded;
                break;
            case RootFindingErrorCode::InvalidBracket:
                error_code = IVErrorCode::BracketingFailed;
                break;
            case RootFindingErrorCode::NumericalInstability:
                error_code = IVErrorCode::NumericalInstability;
                break;
            case RootFindingErrorCode::NoProgress:
                error_code = IVErrorCode::NumericalInstability;
                break;
            default:
                error_code = IVErrorCode::NumericalInstability;
                break;
        }

        return std::unexpected(IVError{
            .code = error_code,
            .iterations = root_error.iterations,
            .final_error = root_error.final_error,
            .last_vol = root_error.last_value
        });
    }

    // Compute final vega for the result
    double final_vega = derivative(result->root);

    // Return success
    return IVSuccess{
        .implied_vol = result->root,
        .iterations = result->iterations,
        .final_error = result->final_error,
        .vega = final_vega,
        .used_rate_approximation = rate_is_curve
    };
}

template <PriceSurface Surface>
BatchIVResult
IVSolverInterpolated<Surface>::solve_batch(const std::vector<IVQuery>& queries) const noexcept
{
    std::vector<std::expected<IVSuccess, IVError>> results(queries.size());
    size_t failed_count = 0;

    // Trivially parallel: B-spline is immutable and thread-safe
    MANGO_PRAGMA_PARALLEL_FOR
    for (size_t i = 0; i < queries.size(); ++i) {
        results[i] = solve(queries[i]);
        if (!results[i].has_value()) {
            MANGO_PRAGMA_ATOMIC
            ++failed_count;
        }
    }

    return BatchIVResult{
        .results = std::move(results),
        .failed_count = failed_count
    };
}

}  // namespace mango
