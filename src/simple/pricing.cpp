// SPDX-License-Identifier: MIT
#include "src/simple/pricing.hpp"
#include "src/option/american_option.hpp"
#include "src/option/american_option_batch.hpp"
#include "src/option/iv_solver_fdm.hpp"
#include "src/pde/core/pde_workspace.hpp"
#include "src/pde/core/grid.hpp"
#include <sstream>
#include <vector>

namespace mango::simple {

std::expected<double, std::string> price(
    double spot, double strike, double maturity,
    double volatility, double rate,
    double dividend_yield,
    OptionType type)
{
    PricingParams params;
    params.spot = spot;
    params.strike = strike;
    params.maturity = maturity;
    params.rate = rate;
    params.dividend_yield = dividend_yield;
    params.type = type;
    params.volatility = volatility;

    // Validate parameters before grid estimation
    auto validation = validate_pricing_params(params);
    if (!validation.has_value()) {
        std::ostringstream oss;
        oss << validation.error();
        return std::unexpected(oss.str());
    }

    // Auto-estimate grid
    auto [grid_spec, time_domain] = estimate_grid_for_option(params);
    size_t n = grid_spec.n_points();
    auto grid_buffer = grid_spec.generate();

    // Allocate workspace buffer
    std::vector<double> buffer(PDEWorkspace::required_size(n));
    auto ws_result = PDEWorkspace::from_buffer_and_grid(
        buffer, grid_buffer.span(), n);
    if (!ws_result.has_value()) {
        return std::unexpected(ws_result.error());
    }

    // Create solver with custom grid config (bypasses re-estimation)
    auto solver_result = AmericanOptionSolver::create(
        params, std::move(*ws_result), std::nullopt,
        std::make_pair(grid_spec, time_domain));
    if (!solver_result.has_value()) {
        std::ostringstream oss;
        oss << solver_result.error();
        return std::unexpected(oss.str());
    }

    // Solve
    auto result = solver_result->solve();
    if (!result.has_value()) {
        std::ostringstream oss;
        oss << result.error();
        return std::unexpected(oss.str());
    }

    return result->value();
}

std::expected<double, std::string> implied_vol(
    double spot, double strike, double maturity,
    double market_price, double rate,
    double dividend_yield,
    OptionType type)
{
    IVQuery query;
    query.spot = spot;
    query.strike = strike;
    query.maturity = maturity;
    query.rate = rate;
    query.dividend_yield = dividend_yield;
    query.type = type;
    query.market_price = market_price;

    IVSolverFDMConfig config;
    IVSolverFDM solver(config);
    auto result = solver.solve_impl(query);
    if (!result.has_value()) {
        const auto& err = result.error();
        std::ostringstream oss;
        oss << "IVError{code=" << static_cast<int>(err.code)
            << ", iterations=" << err.iterations
            << ", final_error=" << err.final_error;
        if (err.last_vol.has_value()) {
            oss << ", last_vol=" << *err.last_vol;
        }
        oss << "}";
        return std::unexpected(oss.str());
    }

    return result->implied_vol;
}

BatchPriceResult price_batch(const std::vector<PricingParams>& params) {
    BatchAmericanOptionSolver solver;
    // solve_batch automatically routes to normalized chain solver when eligible
    auto batch_result = solver.solve_batch(params);

    BatchPriceResult result;
    result.failed_count = batch_result.failed_count;
    result.prices.reserve(batch_result.results.size());

    for (auto& r : batch_result.results) {
        if (r.has_value()) {
            result.prices.push_back(r->value());
        } else {
            std::ostringstream oss;
            oss << r.error();
            result.prices.push_back(std::unexpected(oss.str()));
        }
    }

    return result;
}

BatchIVResult implied_vol_batch(const std::vector<IVQuery>& queries) {
    IVSolverFDMConfig config;
    IVSolverFDM solver(config);
    auto batch_result = solver.solve_batch_impl(queries);

    BatchIVResult result;
    result.failed_count = batch_result.failed_count;
    result.vols.reserve(batch_result.results.size());

    for (auto& r : batch_result.results) {
        if (r.has_value()) {
            result.vols.push_back(r->implied_vol);
        } else {
            const auto& err = r.error();
            std::ostringstream oss;
            oss << "IVError{code=" << static_cast<int>(err.code)
                << ", iterations=" << err.iterations
                << ", final_error=" << err.final_error;
            if (err.last_vol.has_value()) {
                oss << ", last_vol=" << *err.last_vol;
            }
            oss << "}";
            result.vols.push_back(std::unexpected(oss.str()));
        }
    }

    return result;
}

}  // namespace mango::simple
