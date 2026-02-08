// SPDX-License-Identifier: MIT
/**
 * @file interpolated_iv_solver.hpp
 * @brief Implied volatility solver using B-spline price interpolation
 *
 * Provides:
 * - InterpolatedIVSolver<Surface>: Newton-Raphson IV solver on any PriceSurface
 * - AnyIVSolver: type-erased wrapper for convenient use
 * - make_interpolated_iv_solver(): factory that builds the price surface and solver
 *
 * Two construction paths:
 * 1. Direct: build your own PriceSurface, then InterpolatedIVSolver::create()
 * 2. Factory: fill IVSolverFactoryConfig, call make_interpolated_iv_solver()
 *
 * Grid density is controlled via IVGrid.  When `adaptive` is set, the grid
 * values serve as domain bounds for automatic refinement; otherwise they are
 * exact interpolation knots.
 */

#pragma once

#include "mango/option/option_spec.hpp"
#include "mango/option/iv_result.hpp"
#include "mango/option/table/price_surface_concept.hpp"
#include "mango/option/table/standard_surface.hpp"
#include "mango/option/table/spliced_surface.hpp"
#include "mango/option/table/adaptive_grid_types.hpp"
#include "mango/support/error_types.hpp"
#include "mango/support/parallel.hpp"
#include "mango/math/root_finding.hpp"
#include <expected>
#include <cmath>
#include <algorithm>
#include <vector>
#include <variant>
#include <optional>

namespace mango {

/// Configuration for interpolation-based IV solver
struct InterpolatedIVSolverConfig {
    size_t max_iter = 50;          ///< Maximum Newton iterations
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
/// For full yield curve support, use IVSolver instead.
/// When rate approximation is used, IVSuccess::used_rate_approximation is set to true.
///
/// @tparam Surface A type satisfying the PriceSurface concept
template <PriceSurface Surface>
class InterpolatedIVSolver {
public:
    /// Create solver from a PriceSurface
    ///
    /// The surface must provide price(), vega(), and bounds accessors.
    ///
    /// @param surface Pre-built price surface
    /// @param config Solver configuration
    /// @return IV solver or ValidationError
    static std::expected<InterpolatedIVSolver, ValidationError> create(
        Surface surface,
        const InterpolatedIVSolverConfig& config = {});

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
    InterpolatedIVSolver(
        Surface surface,
        std::pair<double, double> m_range,
        std::pair<double, double> tau_range,
        std::pair<double, double> sigma_range,
        std::pair<double, double> r_range,
        OptionType option_type,
        double dividend_yield,
        const InterpolatedIVSolverConfig& config)
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
    InterpolatedIVSolverConfig config_;
    OptionType option_type_;
    double dividend_yield_;

    /// Evaluate option price using B-spline interpolation with strike scaling
    double eval_price(double moneyness, double maturity, double vol, double rate, double strike) const;

    /// Compute vega using partial derivative w.r.t. volatility (axis 2)
    double compute_vega(double moneyness, double maturity, double vol, double rate, double strike) const;

    /// Check if query parameters are within surface bounds
    bool is_in_bounds(const IVQuery& query, double vol) const {
        const double x = std::log(query.spot / query.strike);

        // Extract zero rate for bounds check - must match what solve uses
        // Using get_zero_rate() ensures consistency: -ln(D(T))/T for curves
        double rate_value = get_zero_rate(query.rate, query.maturity);

        return x >= m_range_.first && x <= m_range_.second &&
               query.maturity >= tau_range_.first && query.maturity <= tau_range_.second &&
               vol >= sigma_range_.first && vol <= sigma_range_.second &&
               rate_value >= r_range_.first && rate_value <= r_range_.second;
    }

    /// Validate query parameters
    std::optional<ValidationError> validate_query(const IVQuery& query) const;

    /// Determine adaptive volatility bounds based on intrinsic value
    std::pair<double, double> adaptive_bounds(const IVQuery& query) const;
};

/// Type alias for backward compatibility: InterpolatedIVSolver with StandardSurfaceWrapper
using DefaultInterpolatedIVSolver = InterpolatedIVSolver<StandardSurfaceWrapper>;

// =====================================================================
// Factory: config types, type-erased solver, and factory function
// =====================================================================

/// Standard path: continuous dividends only, maturity grid for interpolation
struct StandardIVPath {
    std::vector<double> maturity_grid;
};

/// Segmented path: discrete dividends with multi-K_ref surface
struct SegmentedIVPath {
    double maturity = 1.0;
    std::vector<Dividend> discrete_dividends;
    MultiKRefConfig kref_config;  ///< defaults to auto
};

/// Configuration for the IV solver factory
struct IVSolverFactoryConfig {
    OptionType option_type = OptionType::PUT;
    double spot = 100.0;
    double dividend_yield = 0.0;
    IVGrid grid;                                    ///< Grid points (exact or domain bounds)
    std::optional<AdaptiveGridParams> adaptive;     ///< If set, refine grid adaptively
    InterpolatedIVSolverConfig solver_config;       ///< Newton config
    std::variant<StandardIVPath, SegmentedIVPath> path;
};

/// Type-erased IV solver wrapping either path
class AnyIVSolver {
public:
    /// Solve for implied volatility (single query)
    std::expected<IVSuccess, IVError> solve(const IVQuery& query) const;

    /// Solve for implied volatility (batch with OpenMP)
    BatchIVResult solve_batch(const std::vector<IVQuery>& queries) const;

    /// Constructor from standard solver
    explicit AnyIVSolver(InterpolatedIVSolver<StandardSurfaceWrapper> solver);

    /// Constructor from segmented solver (spliced surface)
    explicit AnyIVSolver(InterpolatedIVSolver<MultiKRefSurfaceWrapper<>> solver);

private:
    using SolverVariant = std::variant<
        InterpolatedIVSolver<StandardSurfaceWrapper>,
        InterpolatedIVSolver<MultiKRefSurfaceWrapper<>>
    >;
    SolverVariant solver_;
};

/// Factory function: build price surface and IV solver from config
///
/// If path holds StandardIVPath, uses the StandardSurface path.
/// If path holds SegmentedIVPath, uses the MultiKRefSurface path.
/// If adaptive is set, uses AdaptiveGridBuilder
/// to automatically refine grid density until the target IV error is met.
///
/// @param config Solver configuration
/// @return Type-erased AnyIVSolver or ValidationError
std::expected<AnyIVSolver, ValidationError> make_interpolated_iv_solver(const IVSolverFactoryConfig& config);

// =====================================================================
// Template implementation (must be in header for template instantiation)
// =====================================================================

template <PriceSurface Surface>
std::expected<InterpolatedIVSolver<Surface>, ValidationError>
InterpolatedIVSolver<Surface>::create(
    Surface surface,
    const InterpolatedIVSolverConfig& config)
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

    return InterpolatedIVSolver(
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
double InterpolatedIVSolver<Surface>::eval_price(
    double moneyness, double maturity, double vol, double rate, double strike) const
{
    double spot = moneyness * strike;
    return surface_.price(spot, strike, maturity, vol, rate);
}

template <PriceSurface Surface>
double InterpolatedIVSolver<Surface>::compute_vega(
    double moneyness, double maturity, double vol, double rate, double strike) const
{
    double spot = moneyness * strike;
    return surface_.vega(spot, strike, maturity, vol, rate);
}

template <PriceSurface Surface>
std::optional<ValidationError>
InterpolatedIVSolver<Surface>::validate_query(const IVQuery& query) const
{
    if (query.option_type != option_type_) {
        return ValidationError{ValidationErrorCode::OptionTypeMismatch,
            static_cast<double>(query.option_type), 0};
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
InterpolatedIVSolver<Surface>::adaptive_bounds(const IVQuery& query) const
{
    double intrinsic = intrinsic_value(query.spot, query.strike, query.option_type);

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
InterpolatedIVSolver<Surface>::solve(const IVQuery& query) const noexcept
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
    // This loses term structure dynamics. For full curve support, use IVSolver.
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
        .max_iter = config_.max_iter,
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
InterpolatedIVSolver<Surface>::solve_batch(const std::vector<IVQuery>& queries) const noexcept
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
