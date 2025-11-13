/**
 * @file normalized_chain_solver.cpp
 * @brief Implementation of normalized chain solver
 */

#include "src/option/normalized_chain_solver.hpp"
#include <cmath>
#include <algorithm>

namespace mango {

expected<void, std::string> NormalizedSolveRequest::validate() const {
    if (sigma <= 0.0) {
        return unexpected("Volatility must be positive");
    }
    // Note: rate can be negative (EUR, JPY markets)
    if (dividend < 0.0) {
        return unexpected("Dividend yield must be non-negative");
    }
    if (x_min >= x_max) {
        return unexpected("x_min must be < x_max");
    }
    if (n_space < 3) {
        return unexpected("n_space must be ≥ 3");
    }
    if (n_time < 1) {
        return unexpected("n_time must be ≥ 1");
    }
    if (T_max <= 0.0) {
        return unexpected("T_max must be positive");
    }
    if (tau_snapshots.empty()) {
        return unexpected("tau_snapshots must be non-empty");
    }

    // Validate snapshot times
    for (double tau : tau_snapshots) {
        if (tau <= 0.0 || tau > T_max) {
            return unexpected("Snapshot times must be in (0, T_max]");
        }
    }

    return {};
}

expected<NormalizedWorkspace, std::string> NormalizedWorkspace::create(
    const NormalizedSolveRequest& request)
{
    // Validate request
    auto validation = request.validate();
    if (!validation) {
        return unexpected(validation.error());
    }

    NormalizedWorkspace workspace;

    // Create PDE workspace
    auto pde_workspace = AmericanSolverWorkspace::create(
        request.x_min, request.x_max, request.n_space, request.n_time);
    if (!pde_workspace) {
        return unexpected("Failed to create PDE workspace: " + pde_workspace.error());
    }
    workspace.pde_workspace_ = std::move(pde_workspace.value());

    // Allocate x grid
    workspace.x_grid_.resize(request.n_space);
    double dx = (request.x_max - request.x_min) / (request.n_space - 1);
    for (size_t i = 0; i < request.n_space; ++i) {
        workspace.x_grid_[i] = request.x_min + i * dx;
    }

    // Allocate tau grid (copy from request)
    workspace.tau_grid_.assign(request.tau_snapshots.begin(), request.tau_snapshots.end());
    std::sort(workspace.tau_grid_.begin(), workspace.tau_grid_.end());

    // Allocate values array (Nx × Ntau)
    workspace.values_.resize(request.n_space * workspace.tau_grid_.size(), 0.0);

    return workspace;
}

NormalizedSurfaceView NormalizedWorkspace::surface_view() {
    return NormalizedSurfaceView(x_grid_, tau_grid_, values_);
}

double NormalizedSurfaceView::interpolate(double x, double tau) const {
    // Find x interval [x_grid[i], x_grid[i+1]]
    auto x_it = std::lower_bound(x_grid_.begin(), x_grid_.end(), x);
    if (x_it == x_grid_.begin()) {
        x_it = x_grid_.begin() + 1;  // Clamp to first interval
    } else if (x_it == x_grid_.end()) {
        x_it = x_grid_.end() - 1;  // Clamp to last interval
    }
    size_t i_x = x_it - x_grid_.begin() - 1;

    // Find tau interval [tau_grid[j], tau_grid[j+1]]
    auto tau_it = std::lower_bound(tau_grid_.begin(), tau_grid_.end(), tau);
    if (tau_it == tau_grid_.begin()) {
        tau_it = tau_grid_.begin() + 1;
    } else if (tau_it == tau_grid_.end()) {
        tau_it = tau_grid_.end() - 1;
    }
    size_t i_tau = tau_it - tau_grid_.begin() - 1;

    // Bilinear interpolation
    double x0 = x_grid_[i_x];
    double x1 = x_grid_[i_x + 1];
    double tau0 = tau_grid_[i_tau];
    double tau1 = tau_grid_[i_tau + 1];

    double fx = (x - x0) / (x1 - x0);
    double ft = (tau - tau0) / (tau1 - tau0);

    // Values stored row-major: values[i*Ntau + j]
    size_t Ntau = tau_grid_.size();
    double v00 = values_[i_x * Ntau + i_tau];
    double v01 = values_[i_x * Ntau + (i_tau + 1)];
    double v10 = values_[(i_x + 1) * Ntau + i_tau];
    double v11 = values_[(i_x + 1) * Ntau + (i_tau + 1)];

    return (1.0 - fx) * (1.0 - ft) * v00 +
           (1.0 - fx) * ft * v01 +
           fx * (1.0 - ft) * v10 +
           fx * ft * v11;
}

expected<void, std::string> NormalizedChainSolver::check_eligibility(
    const NormalizedSolveRequest& request,
    std::span<const double> moneyness_grid)
{
    // Check grid spacing
    double dx = (request.x_max - request.x_min) / (request.n_space - 1);
    if (dx > EligibilityLimits::MAX_DX) {
        return unexpected(
            "Grid spacing " + std::to_string(dx) +
            " exceeds limit " + std::to_string(EligibilityLimits::MAX_DX) +
            " (Von Neumann stability requirement)");
    }

    // Check domain width
    double width = request.x_max - request.x_min;
    if (width > EligibilityLimits::MAX_WIDTH) {
        return unexpected(
            "Domain width " + std::to_string(width) +
            " exceeds limit " + std::to_string(EligibilityLimits::MAX_WIDTH) +
            " (convergence degrades beyond 5.8 log-units)");
    }

    // Check margins
    // Moneyness convention: m = K/S, so x = ln(S/K) = -ln(m)
    // x_min_data = -ln(m_max), x_max_data = -ln(m_min)
    if (moneyness_grid.empty()) {
        return unexpected("Moneyness grid is empty");
    }

    auto [m_min_it, m_max_it] = std::ranges::minmax_element(moneyness_grid);
    double m_min = *m_min_it;
    double m_max = *m_max_it;

    if (m_min <= 0.0 || m_max <= 0.0) {
        return unexpected("Moneyness values must be positive (m = K/S > 0)");
    }

    double x_min_data = -std::log(m_max);
    double x_max_data = -std::log(m_min);

    double margin_left = x_min_data - request.x_min;
    double margin_right = request.x_max - x_max_data;
    double min_margin = EligibilityLimits::min_margin(dx);

    if (margin_left < min_margin) {
        return unexpected(
            "Left margin " + std::to_string(margin_left) +
            " < required " + std::to_string(min_margin) +
            " (need ≥6 ghost cells to avoid boundary reflection)");
    }

    if (margin_right < min_margin) {
        return unexpected(
            "Right margin " + std::to_string(margin_right) +
            " < required " + std::to_string(min_margin) +
            " (need ≥6 ghost cells to avoid boundary reflection)");
    }

    // Check ratio (derived from width + margin constraints)
    double ratio = m_max / m_min;
    double max_ratio_limit = EligibilityLimits::max_ratio(dx);
    if (ratio > max_ratio_limit) {
        return unexpected(
            "Moneyness ratio " + std::to_string(ratio) +
            " exceeds limit " + std::to_string(max_ratio_limit) +
            " (derived from width=" + std::to_string(EligibilityLimits::MAX_WIDTH) +
            " and margin=" + std::to_string(min_margin) + ")");
    }

    return {};
}

}  // namespace mango
