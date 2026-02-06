// SPDX-License-Identifier: MIT
#include "mango/option/grid_spec_types.hpp"

namespace mango {

GridAccuracyParams make_grid_accuracy(GridAccuracyProfile profile) {
    GridAccuracyParams params;
    switch (profile) {
        case GridAccuracyProfile::Low:
            params.tol = 5e-3;
            params.min_spatial_points = 150;
            params.max_spatial_points = 1500;
            params.max_time_steps = 6000;
            break;
        case GridAccuracyProfile::Medium:
            params.tol = 5e-5;
            params.min_spatial_points = 201;
            params.max_spatial_points = 2500;
            params.max_time_steps = 12000;
            break;
        case GridAccuracyProfile::High:
            params.tol = 1e-5;
            params.min_spatial_points = 301;
            params.max_spatial_points = 3500;
            params.max_time_steps = 16000;
            break;
        case GridAccuracyProfile::Ultra:
            params.tol = 5e-6;
            params.min_spatial_points = 401;
            params.max_spatial_points = 5000;
            params.max_time_steps = 20000;
            break;
    }
    return params;
}

std::pair<GridSpec<double>, TimeDomain> estimate_pde_grid(
    const PricingParams& params,
    const GridAccuracyParams& accuracy)
{
    // Domain bounds (centered on current moneyness)
    double sigma_sqrt_T = params.volatility * std::sqrt(params.maturity);
    double x0 = std::log(params.spot / params.strike);

    double x_min = x0 - accuracy.n_sigma * sigma_sqrt_T;
    double x_max = x0 + accuracy.n_sigma * sigma_sqrt_T;

    // Spatial resolution (target truncation error)
    double dx_target = params.volatility * std::sqrt(accuracy.tol);
    size_t Nx = static_cast<size_t>(std::ceil((x_max - x_min) / dx_target));
    Nx = std::clamp(Nx, accuracy.min_spatial_points, accuracy.max_spatial_points);

    // Ensure odd number of points (for centered stencils)
    if (Nx % 2 == 0) Nx++;

    // Widen grid for dividend shift: spline evaluates at x'=ln(exp(x)-D/K)
    // Only consider dividends strictly within (0, T) — same filter as mandatory tau
    double max_d_over_k = 0.0;
    for (const auto& div : params.discrete_dividends) {
        double tau = params.maturity - div.calendar_time;
        if (tau > 0.0 && tau < params.maturity) {
            max_d_over_k = std::max(max_d_over_k, div.amount / params.strike);
        }
    }
    if (max_d_over_k > 0.0 && std::exp(x_min) > max_d_over_k) {
        x_min = std::log(std::exp(x_min) - max_d_over_k);
    } else if (max_d_over_k > 0.0) {
        x_min -= 1.0;  // conservative extension
    }

    // Build cluster list for multi-sinh grid.
    // Strike cluster at x=0 (payoff kink) is the dominant coarse-grid error source.
    // Spot cluster at x0 (query point) is secondary for value_at()/delta().
    // When S≈K, auto-merge combines them. Skip strike cluster if outside domain.
    std::vector<MultiSinhCluster<double>> clusters;
    clusters.push_back({.center_x = x0, .alpha = accuracy.alpha, .weight = 0.5});
    if (0.0 >= x_min && 0.0 <= x_max) {
        clusters.push_back({.center_x = 0.0, .alpha = accuracy.alpha, .weight = 1.0});
    }
    auto grid_spec = GridSpec<double>::multi_sinh_spaced(x_min, x_max, Nx, std::move(clusters));

    // Temporal resolution: compute actual dx_min from generated grid.
    // The old formula dx_min = dx_avg·exp(-α) was an approximation that became
    // wildly pessimistic at high α (e.g., α≈4 gave ~7x more time steps than needed).
    // TR-BDF2 is L-stable so CFL is for accuracy, not stability.
    auto grid_buf = grid_spec.value().generate();
    auto pts = grid_buf.view().span();
    double dx_min = pts[1] - pts[0];
    for (size_t i = 2; i < pts.size(); ++i) {
        dx_min = std::min(dx_min, pts[i] - pts[i - 1]);
    }

    double dt = accuracy.c_t * dx_min;
    size_t Nt = static_cast<size_t>(std::ceil(params.maturity / dt));
    Nt = std::min(Nt, accuracy.max_time_steps);  // Upper bound

    // Convert discrete dividend calendar times to time-to-expiry (tau)
    std::vector<double> mandatory_tau;
    for (const auto& div : params.discrete_dividends) {
        double tau = params.maturity - div.calendar_time;
        if (tau > 0.0 && tau < params.maturity) {
            mandatory_tau.push_back(tau);
        }
    }

    // Apply max_time_steps cap to both uniform and non-uniform paths
    double dt_capped = std::max(dt, params.maturity / static_cast<double>(accuracy.max_time_steps));

    TimeDomain time_domain = mandatory_tau.empty()
        ? TimeDomain::from_n_steps(0.0, params.maturity, Nt)
        : TimeDomain::with_mandatory_points(0.0, params.maturity, dt_capped, mandatory_tau);

    return {grid_spec.value(), time_domain};
}

std::pair<GridSpec<double>, TimeDomain> estimate_batch_pde_grid(
    std::span<const PricingParams> params,
    const GridAccuracyParams& accuracy)
{
    if (params.empty()) {
        // Return minimal valid sinh grid for empty batch
        auto grid_spec = GridSpec<double>::sinh_spaced(-3.0, 3.0, 101, accuracy.alpha);
        TimeDomain time_domain = TimeDomain::from_n_steps(0.0, 1.0, 100);
        return {grid_spec.value(), time_domain};
    }

    double global_x_min = std::numeric_limits<double>::max();
    double global_x_max = std::numeric_limits<double>::lowest();
    size_t global_Nx = 0;
    size_t global_Nt = 0;
    double max_maturity = 0.0;

    // Estimate grid for each option and take union/maximum
    for (const auto& p : params) {
        auto [grid_spec, time_domain] = estimate_pde_grid(p, accuracy);
        global_x_min = std::min(global_x_min, grid_spec.x_min());
        global_x_max = std::max(global_x_max, grid_spec.x_max());
        global_Nx = std::max(global_Nx, grid_spec.n_points());
        global_Nt = std::max(global_Nt, time_domain.n_steps());
        max_maturity = std::max(max_maturity, p.maturity);
    }

    // Create multi-sinh grid with strike cluster at x=0
    // Batch uses shared strike, so x=0 is the common payoff kink
    auto grid_spec = GridSpec<double>::multi_sinh_spaced(global_x_min, global_x_max, global_Nx, {
        {.center_x = 0.0, .alpha = accuracy.alpha, .weight = 1.0},
    });

    // Collect union of all dividend tau values across the batch
    std::vector<double> all_mandatory_tau;
    for (const auto& p : params) {
        for (const auto& div : p.discrete_dividends) {
            double tau = p.maturity - div.calendar_time;
            if (tau > 0.0 && tau < max_maturity) {
                all_mandatory_tau.push_back(tau);
            }
        }
    }

    double global_dt = max_maturity / static_cast<double>(global_Nt);
    double dt_capped = std::max(global_dt,
        max_maturity / static_cast<double>(accuracy.max_time_steps));

    TimeDomain time_domain = all_mandatory_tau.empty()
        ? TimeDomain::from_n_steps(0.0, max_maturity, global_Nt)
        : TimeDomain::with_mandatory_points(0.0, max_maturity, dt_capped, all_mandatory_tau);

    return {grid_spec.value(), time_domain};
}

}  // namespace mango
