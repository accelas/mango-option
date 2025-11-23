/**
 * @file iv_solver_interpolated.cpp
 * @brief Implementation of interpolation-based IV solver
 */

#include "src/option/iv_solver_interpolated.hpp"
#include "src/option/price_table_4d_builder.hpp"
#include "src/support/parallel.hpp"
#include "src/math/root_finding.hpp"
#include <algorithm>
#include <cmath>

namespace mango {

std::expected<IVSolverInterpolated, std::string> IVSolverInterpolated::create(
    std::shared_ptr<const BSpline4D> spline,
    double K_ref,
    std::pair<double, double> m_range,
    std::pair<double, double> tau_range,
    std::pair<double, double> sigma_range,
    std::pair<double, double> r_range,
    const IVSolverInterpolatedConfig& config)
{
    if (!spline) {
        return std::unexpected(std::string("BSpline4D pointer is null"));
    }
    if (K_ref <= 0.0) {
        return std::unexpected(std::string("K_ref must be positive"));
    }

    return IVSolverInterpolated(
        std::move(spline), K_ref, m_range, tau_range, sigma_range, r_range, config);
}

std::expected<IVSolverInterpolated, std::string> IVSolverInterpolated::create(
    const PriceTableSurface& surface,
    const IVSolverInterpolatedConfig& config)
{
    if (!surface.valid()) {
        return std::unexpected(std::string("PriceTableSurface is not initialized (workspace is null)"));
    }

    auto spline = std::make_shared<BSpline4D>(*surface.workspace());
    return create(
        std::move(spline),
        surface.K_ref(),
        surface.moneyness_range(),
        surface.maturity_range(),
        surface.volatility_range(),
        surface.rate_range(),
        config);
}

std::optional<std::string> IVSolverInterpolated::validate_query(const IVQuery& query) const {
    // Use common validation for option spec, market price, and arbitrage checks
    auto validation = validate_iv_query(query);
    if (!validation) {
        return validation.error();
    }

    return std::nullopt;
}

std::pair<double, double> IVSolverInterpolated::adaptive_bounds(const IVQuery& query) const {
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

// Using IVResult for backward compatibility - will be migrated to std::expected in future task
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
IVResult IVSolverInterpolated::solve_impl(const IVQuery& query) const noexcept {
    // Validate input
    auto error = validate_query(query);
    if (error.has_value()) {
        return IVResult{
            .converged = false,
            .iterations = 0,
            .implied_vol = 0.0,
            .final_error = 0.0,
            .failure_reason = *error,
            .vega = std::nullopt
        };
    }

    const double moneyness = query.spot / query.strike;

    // Get adaptive bounds
    auto [sigma_min, sigma_max] = adaptive_bounds(query);

    // Check if query is within surface bounds
    if (!is_in_bounds(query, sigma_min) || !is_in_bounds(query, sigma_max)) {
        return IVResult{
            .converged = false,
            .iterations = 0,
            .implied_vol = 0.0,
            .final_error = 0.0,
            .failure_reason = "Query parameters out of surface bounds. "
                              "Use PDE-based IV solver for out-of-grid queries.",
            .vega = std::nullopt
        };
    }

    // Define objective function: f(σ) = Price(σ) - Market_Price
    auto objective = [&](double sigma) -> double {
        return eval_price(moneyness, query.maturity, sigma, query.rate, query.strike) - query.market_price;
    };

    // Define derivative (vega): df/dσ = ∂Price/∂σ
    auto derivative = [&](double sigma) -> double {
        return compute_vega(moneyness, query.maturity, sigma, query.rate, query.strike);
    };

    // Use generic bounded Newton-Raphson
    RootFindingConfig newton_config{
        .max_iter = static_cast<size_t>(std::max(0, config_.max_iterations)),
        .tolerance = config_.tolerance
    };

    const double sigma0 = (sigma_min + sigma_max) / 2.0;  // Initial guess
    auto result = newton_find_root(objective, derivative, sigma0, sigma_min, sigma_max, newton_config);

    // Compute final vega for the result
    std::optional<double> final_vega = std::nullopt;
    if (result.root.has_value()) {
        final_vega = derivative(result.root.value());
    }

    // Convert RootFindingResult to IVResult
    return IVResult{
        .converged = result.converged,
        .iterations = result.iterations,
        .implied_vol = result.root.value_or(sigma0),
        .final_error = result.final_error,
        .failure_reason = result.failure_reason,
        .vega = final_vega
    };
}

void IVSolverInterpolated::solve_batch_impl(std::span<const IVQuery> queries,
                                             std::span<IVResult> results) const noexcept {
    // Trivially parallel: B-spline is immutable and thread-safe
    MANGO_PRAGMA_PARALLEL_FOR
    for (size_t i = 0; i < queries.size(); ++i) {
        results[i] = solve_impl(queries[i]);
    }
}
#pragma GCC diagnostic pop

} // namespace mango
