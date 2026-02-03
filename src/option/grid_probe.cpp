// SPDX-License-Identifier: MIT
/**
 * @file grid_probe.cpp
 * @brief Implementation of Richardson-style grid adequacy probing
 */

#include "src/option/grid_probe.hpp"
#include "src/pde/core/pde_workspace.hpp"
#include "src/support/ivcalc_trace.h"
#include <memory_resource>
#include <algorithm>
#include <cmath>

namespace mango {

namespace {

/// Minimum number of time steps for accuracy (TR-BDF2 is L-stable, no CFL needed)
constexpr size_t kMinTimeSteps = 50;

/// Solve American option at given Nx and return result
/// Returns nullopt on solver failure
std::optional<AmericanOptionResult> solve_at_nx(
    const PricingParams& params,
    size_t Nx,
    const GridAccuracyParams& accuracy)
{
    // Build grid with specified Nx
    GridAccuracyParams local_accuracy = accuracy;
    local_accuracy.min_spatial_points = Nx;
    local_accuracy.max_spatial_points = Nx;

    auto [grid_spec, time_domain] = estimate_pde_grid(params, local_accuracy);

    // Enforce Nt floor for short maturities
    if (time_domain.n_steps() < kMinTimeSteps) {
        time_domain = TimeDomain::from_n_steps(0.0, params.maturity, kMinTimeSteps);
    }

    // Allocate workspace
    size_t n = grid_spec.n_points();
    std::pmr::vector<double> buffer(PDEWorkspace::required_size(n),
                                    std::pmr::get_default_resource());

    auto workspace_result = PDEWorkspace::from_buffer(buffer, n);
    if (!workspace_result.has_value()) {
        return std::nullopt;
    }

    // Collect mandatory tau values for discrete dividends
    std::vector<double> mandatory_tau;
    for (const auto& div : params.discrete_dividends) {
        double tau = params.maturity - div.calendar_time;
        if (tau > 0.0 && tau < params.maturity) {
            mandatory_tau.push_back(tau);
        }
    }

    // Create solver
    auto solver_result = AmericanOptionSolver::create(
        params, workspace_result.value(),
        PDEGridConfig{
            .grid_spec = grid_spec,
            .n_time = time_domain.n_steps(),
            .mandatory_times = std::move(mandatory_tau)
        });

    if (!solver_result.has_value()) {
        return std::nullopt;
    }

    auto solve_result = solver_result.value().solve();
    if (!solve_result.has_value()) {
        return std::nullopt;
    }

    return std::move(solve_result.value());
}

/// Compute delta via finite difference at given spot
/// Uses the intersection of both grid domains with 1% margin
/// Returns 0 if spot is outside valid range or h is too small
double compute_delta_at_spot(const AmericanOptionResult& result, double spot) {
    // Get grid bounds from the result's grid
    auto grid = result.grid();
    auto x_span = grid->x();

    double x_min = x_span.front();
    double x_max = x_span.back();

    // Convert spot to log-moneyness
    double x = std::log(spot / result.strike());

    // Apply 1% margin
    double margin = 0.01 * (x_max - x_min);
    double lo = x_min + margin;
    double hi = x_max - margin;

    // Compute symmetric h clamped to stay within bounds
    double h_max = std::max(0.0, std::min(x - lo, hi - x));

    // If h is too small, skip delta check
    if (h_max < 1e-6) {
        return 0.0;
    }

    // Use a reasonable h (1% of spot in log-moneyness)
    double h = std::min(h_max, 0.01);

    // Finite difference: (V(S+) - V(S-)) / (2h * S)
    // In log-moneyness: S+ = K*exp(x+h), S- = K*exp(x-h)
    double K = result.strike();
    double S_plus = K * std::exp(x + h);
    double S_minus = K * std::exp(x - h);

    double V_plus = result.value_at(S_plus);
    double V_minus = result.value_at(S_minus);

    // Delta in terms of spot: dV/dS = (V+ - V-) / (S+ - S-)
    return (V_plus - V_minus) / (S_plus - S_minus);
}

}  // namespace

std::expected<ProbeResult, ValidationError> probe_grid_adequacy(
    const PricingParams& params,
    double target_error,
    size_t initial_Nx,
    size_t max_iterations)
{
    // Validate target_error
    if (target_error <= 0.0) {
        return std::unexpected(ValidationError(
            ValidationErrorCode::InvalidBounds,
            target_error));
    }

    // Trace: calibration start
    MANGO_TRACE_GRID_PROBE_START(initial_Nx, target_error, max_iterations);

    // Use default accuracy params for grid estimation
    GridAccuracyParams accuracy{};

    size_t Nx = initial_Nx;
    double estimated_error = std::numeric_limits<double>::max();
    bool converged = false;
    size_t iteration = 0;

    // Store the last successful grid/time_domain
    std::optional<GridSpec<double>> best_grid;
    std::optional<TimeDomain> best_time_domain;

    for (iteration = 1; iteration <= max_iterations && !converged; ++iteration) {
        // Solve at Nx
        auto result_nx = solve_at_nx(params, Nx, accuracy);
        if (!result_nx.has_value()) {
            // Solver failed - use what we have
            break;
        }

        // Solve at 2*Nx
        auto result_2nx = solve_at_nx(params, 2 * Nx, accuracy);
        if (!result_2nx.has_value()) {
            // Solver failed - use Nx grid
            best_grid = result_nx->grid()->spacing().grid().is_uniform()
                ? GridSpec<double>::uniform(
                    result_nx->grid()->x().front(),
                    result_nx->grid()->x().back(),
                    result_nx->grid()->n_space()).value()
                : GridSpec<double>::sinh_spaced(
                    result_nx->grid()->x().front(),
                    result_nx->grid()->x().back(),
                    result_nx->grid()->n_space(),
                    accuracy.alpha).value();
            best_time_domain = result_nx->grid()->time();
            break;
        }

        // Compute price difference
        double P1 = result_nx->value_at(params.spot);
        double P2 = result_2nx->value_at(params.spot);
        double price_diff = std::abs(P2 - P1);

        // Compute delta difference using intersection domain
        // Get bounds from both grids
        auto grid_nx = result_nx->grid();
        auto grid_2nx = result_2nx->grid();

        double x_min_nx = grid_nx->x().front();
        double x_max_nx = grid_nx->x().back();
        double x_min_2nx = grid_2nx->x().front();
        double x_max_2nx = grid_2nx->x().back();

        // Intersection with 1% margin
        double x_min = std::max(x_min_nx, x_min_2nx);
        double x_max = std::min(x_max_nx, x_max_2nx);
        double margin = 0.01 * (x_max - x_min);
        double lo = x_min + margin;
        double hi = x_max - margin;

        // Compute delta difference
        double delta_diff = 0.0;
        double x_spot = std::log(params.spot / params.strike);

        // Only compute delta if spot is within valid range
        if (x_spot >= lo && x_spot <= hi) {
            double h_max = std::max(0.0, std::min(x_spot - lo, hi - x_spot));

            if (h_max >= 1e-6) {
                double delta1 = compute_delta_at_spot(*result_nx, params.spot);
                double delta2 = compute_delta_at_spot(*result_2nx, params.spot);
                delta_diff = std::abs(delta2 - delta1);
            }
        }

        // Convergence check
        // Price: |P2 - P1| <= max(target_error, 0.001 * max(|P1|, |P2|, 0.10))
        double price_tol = std::max(target_error,
            0.001 * std::max({std::abs(P1), std::abs(P2), 0.10}));
        bool price_converged = price_diff <= price_tol;

        // Delta: |delta2 - delta1| <= 0.01
        bool delta_converged = delta_diff <= 0.01;

        estimated_error = price_diff;

        // Trace: iteration complete
        bool iter_converged = price_converged && delta_converged;
        MANGO_TRACE_GRID_PROBE_ITERATION(iteration, Nx, price_diff, iter_converged ? 1 : 0);

        if (iter_converged) {
            converged = true;
            // Use the finer grid (2*Nx) since it's more accurate
            best_grid = GridSpec<double>::sinh_spaced(
                grid_2nx->x().front(),
                grid_2nx->x().back(),
                grid_2nx->n_space(),
                accuracy.alpha).value();
            best_time_domain = grid_2nx->time();
        } else {
            // Not converged - prepare for next iteration
            // Store current best (the finer grid)
            best_grid = GridSpec<double>::sinh_spaced(
                grid_2nx->x().front(),
                grid_2nx->x().back(),
                grid_2nx->n_space(),
                accuracy.alpha).value();
            best_time_domain = grid_2nx->time();

            // Double Nx for next iteration
            Nx *= 2;
        }
    }

    // If we don't have a grid yet, create one from initial parameters
    if (!best_grid.has_value() || !best_time_domain.has_value()) {
        auto [grid_spec, time_domain] = estimate_pde_grid(params, accuracy);
        if (time_domain.n_steps() < kMinTimeSteps) {
            time_domain = TimeDomain::from_n_steps(0.0, params.maturity, kMinTimeSteps);
        }
        best_grid = grid_spec;
        best_time_domain = time_domain;
    }

    size_t final_iterations = iteration > max_iterations ? max_iterations : iteration;

    // Trace: calibration complete
    MANGO_TRACE_GRID_PROBE_COMPLETE(
        best_grid.value().n_points(),
        estimated_error,
        final_iterations,
        converged ? 1 : 0);

    return ProbeResult{
        .grid = best_grid.value(),
        .time_domain = best_time_domain.value(),
        .estimated_error = estimated_error,
        .probe_iterations = final_iterations,
        .converged = converged
    };
}

}  // namespace mango
