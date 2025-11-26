/**
 * @file iv_solver_interpolated.cpp
 * @brief Implementation of interpolation-based IV solver
 */

#include "src/option/iv_solver_interpolated.hpp"
#include "src/option/table/price_table_surface.hpp"
#include "src/support/parallel.hpp"
#include "src/math/root_finding.hpp"
#include <algorithm>
#include <cmath>

namespace mango {

std::expected<IVSolverInterpolated, ValidationError> IVSolverInterpolated::create(
    std::shared_ptr<const PriceTableSurface<4>> surface,
    const IVSolverInterpolatedConfig& config)
{
    // Validate surface pointer
    if (!surface) {
        return std::unexpected(ValidationError(
            ValidationErrorCode::InvalidGridSize,
            0.0));
    }

    // Extract metadata from surface
    const auto& axes = surface->axes();
    const auto& meta = surface->metadata();

    // Validate K_ref
    if (meta.K_ref <= 0.0) {
        return std::unexpected(ValidationError(
            ValidationErrorCode::InvalidStrike,
            meta.K_ref));
    }

    // Extract ranges from axes
    // Grid indices: 0=moneyness, 1=maturity, 2=volatility, 3=rate
    if (axes.grids[0].empty() || axes.grids[1].empty() ||
        axes.grids[2].empty() || axes.grids[3].empty()) {
        return std::unexpected(ValidationError(
            ValidationErrorCode::InvalidGridSize,
            0.0));
    }

    auto m_range = std::make_pair(axes.grids[0].front(), axes.grids[0].back());
    auto tau_range = std::make_pair(axes.grids[1].front(), axes.grids[1].back());
    auto sigma_range = std::make_pair(axes.grids[2].front(), axes.grids[2].back());
    auto r_range = std::make_pair(axes.grids[3].front(), axes.grids[3].back());

    return IVSolverInterpolated(
        std::move(surface),
        meta.K_ref,
        m_range,
        tau_range,
        sigma_range,
        r_range,
        config);
}

double IVSolverInterpolated::eval_price(double moneyness, double maturity, double vol, double rate, double strike) const {
    double price_Kref = surface_->value({moneyness, maturity, vol, rate});
    double scale_factor = strike / K_ref_;
    return price_Kref * scale_factor;
}

double IVSolverInterpolated::compute_vega(double moneyness, double maturity, double vol, double rate, double strike) const {
    double vega_Kref = surface_->partial(2, {moneyness, maturity, vol, rate});
    const double scale_factor = strike / K_ref_;
    return vega_Kref * scale_factor;
}

std::optional<ValidationError> IVSolverInterpolated::validate_query(const IVQuery& query) const {
    // Use common validation for option spec, market price, and arbitrage checks
    auto validation = validate_iv_query(query);
    if (!validation.has_value()) {
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

std::expected<IVSuccess, IVError> IVSolverInterpolated::solve_impl(const IVQuery& query) const noexcept {
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

    // Define objective function: f(σ) = Price(σ) - Market_Price
    auto objective = [&](double sigma) -> double {
        return eval_price(moneyness, query.maturity, sigma, rate_value, query.strike) - query.market_price;
    };

    // Define derivative (vega): df/dσ = ∂Price/∂σ
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

BatchIVResult IVSolverInterpolated::solve_batch_impl(const std::vector<IVQuery>& queries) const noexcept {
    std::vector<std::expected<IVSuccess, IVError>> results(queries.size());
    size_t failed_count = 0;

    // Trivially parallel: B-spline is immutable and thread-safe
    MANGO_PRAGMA_PARALLEL_FOR
    for (size_t i = 0; i < queries.size(); ++i) {
        results[i] = solve_impl(queries[i]);
        if (!results[i].has_value()) {
            #pragma omp atomic
            ++failed_count;
        }
    }

    return BatchIVResult{
        .results = std::move(results),
        .failed_count = failed_count
    };
}

} // namespace mango
