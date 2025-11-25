#pragma once

/// @file american_option.hpp
/// @brief American option pricing with Kokkos
///
/// Implements finite difference solver for American options in
/// log-moneyness coordinates using implicit Euler time-stepping.

#include <Kokkos_Core.hpp>
#include <expected>
#include <cmath>
#include "kokkos/src/pde/core/grid.hpp"
#include "kokkos/src/pde/core/workspace.hpp"
#include "kokkos/src/pde/operators/black_scholes.hpp"
#include "kokkos/src/math/thomas_solver.hpp"
#include "kokkos/src/support/execution_space.hpp"

namespace mango::kokkos {

enum class OptionType { Call, Put };

/// Option pricing parameters
struct PricingParams {
    double strike;
    double spot;
    double maturity;
    double volatility;
    double rate;
    double dividend_yield;
    OptionType type;
};

/// Pricing result with price and Greeks
struct PricingResult {
    double price;
    double delta;
};

/// Solver error codes
enum class SolverError {
    InvalidParams,
    ConvergenceFailed,
    GridError
};

/// American option solver using finite differences
///
/// Solves Black-Scholes PDE in log-moneyness coordinates with
/// early exercise via obstacle projection.
///
/// Mathematical formulation:
///   V_t = 0.5*sigma^2*V_xx + (r-q-0.5*sigma^2)*V_x - r*V
///   subject to V >= payoff (early exercise constraint)
///
/// Accuracy: O(dt) + O(dx^2)
template <typename MemSpace>
class AmericanOptionSolver {
public:
    using view_type = Kokkos::View<double*, MemSpace>;

    explicit AmericanOptionSolver(const PricingParams& params,
                                   size_t n_space = 201,
                                   size_t n_time = 1000)
        : params_(params), n_space_(n_space), n_time_(n_time) {}

    [[nodiscard]] std::expected<PricingResult, SolverError> solve() {
        // Grid bounds in log-moneyness
        double sigma_sqrt_T = params_.volatility * std::sqrt(params_.maturity);
        double x0 = std::log(params_.spot / params_.strike);
        double x_min = x0 - 5.0 * sigma_sqrt_T;
        double x_max = x0 + 5.0 * sigma_sqrt_T;

        // Create grid
        auto grid_result = Grid<MemSpace>::uniform(x_min, x_max, n_space_);
        if (!grid_result.has_value()) {
            return std::unexpected(SolverError::GridError);
        }
        auto grid = std::move(grid_result.value());

        // Create workspace
        auto ws_result = PDEWorkspace<MemSpace>::create(n_space_);
        if (!ws_result.has_value()) {
            return std::unexpected(SolverError::GridError);
        }
        auto workspace = std::move(ws_result.value());

        double dx = (x_max - x_min) / static_cast<double>(n_space_ - 1);
        double dt = params_.maturity / static_cast<double>(n_time_);

        // Initialize with payoff
        auto u = grid.u_current();
        auto x = grid.x();
        initialize_payoff(x, u);

        // Black-Scholes operator
        BlackScholesOperator<MemSpace> bs_op(params_.volatility, params_.rate,
                                              params_.dividend_yield);

        // Assemble Jacobian (constant for linear PDE)
        auto lower = workspace.jacobian_lower();
        auto diag = workspace.jacobian_diag();
        auto upper = workspace.jacobian_upper();
        bs_op.assemble_jacobian(dt, dx, lower, diag, upper);

        // Apply Dirichlet boundary conditions to matrix
        apply_boundary_conditions_to_matrix(lower, diag, upper);

        // Pre-allocated temporaries for Thomas solver
        auto lower_temp = workspace.thomas_lower_temp();
        auto diag_temp = workspace.thomas_diag_temp();
        auto upper_temp = workspace.thomas_upper_temp();

        ThomasSolver<MemSpace> thomas;
        auto rhs = workspace.rhs();
        auto solution = workspace.delta_u();

        // Cache boundary x values (copy once, not per timestep)
        double x_left = x_min;
        double x_right = x_max;

        // Time stepping (backward from T to 0)
        for (size_t step = 0; step < n_time_; ++step) {
            // RHS = u^n
            Kokkos::deep_copy(rhs, u);

            // Boundary conditions on RHS
            set_boundary_rhs(x, rhs, x_left, x_right, step, dt);

            // Copy matrix for Thomas (it modifies in-place)
            Kokkos::deep_copy(lower_temp, lower);
            Kokkos::deep_copy(diag_temp, diag);
            Kokkos::deep_copy(upper_temp, upper);

            // Solve linear system
            auto result = thomas.solve(lower_temp, diag_temp, upper_temp, rhs, solution);

            // Apply obstacle (early exercise)
            apply_obstacle(x, solution);

            // Copy solution back
            Kokkos::deep_copy(u, solution);
        }

        // Interpolate price at spot
        double price = interpolate_at_spot(x, u, x0);
        double delta = compute_delta(x, u, x0);

        return PricingResult{.price = price, .delta = delta};
    }

private:
    void initialize_payoff(view_type x, view_type u) {
        const double K = params_.strike;
        const bool is_put = (params_.type == OptionType::Put);
        const size_t n = n_space_;

        Kokkos::parallel_for("init_payoff", n,
            KOKKOS_LAMBDA(const size_t i) {
                double S = K * std::exp(x(i));
                if (is_put) {
                    u(i) = (K > S) ? (K - S) : 0.0;
                } else {
                    u(i) = (S > K) ? (S - K) : 0.0;
                }
            });
        Kokkos::fence();
    }

    void apply_boundary_conditions_to_matrix(view_type lower, view_type diag, view_type upper) {
        const size_t n = n_space_;
        // Use parallel_for for GPU efficiency
        Kokkos::parallel_for("apply_bc_matrix", n,
            KOKKOS_LAMBDA(const size_t i) {
                if (i == 0) {
                    diag(i) = 1.0;
                    upper(i) = 0.0;
                } else if (i == n - 1) {
                    diag(i) = 1.0;
                    lower(i - 1) = 0.0;
                }
            });
        Kokkos::fence();
    }

    void set_boundary_rhs(view_type x, view_type rhs, double x_left, double x_right,
                          size_t step, double dt) {
        // Compute boundary values (no device-host copy needed)
        double t = params_.maturity - static_cast<double>(step + 1) * dt;
        double K = params_.strike;
        double r = params_.rate;
        double q = params_.dividend_yield;
        const size_t n = n_space_;
        const bool is_put = (params_.type == OptionType::Put);

        double bc_left, bc_right;
        if (is_put) {
            // Left BC: deep ITM put value
            double S_left = K * std::exp(x_left);
            bc_left = K * std::exp(-r * t) - S_left * std::exp(-q * t);
            // Right BC: deep OTM put = 0
            bc_right = 0.0;
        } else {
            // Left BC: deep OTM call = 0
            bc_left = 0.0;
            // Right BC: deep ITM call value
            double S_right = K * std::exp(x_right);
            bc_right = S_right * std::exp(-q * t) - K * std::exp(-r * t);
        }

        // Apply BCs via parallel_for (no host mirror needed)
        Kokkos::parallel_for("apply_bc_rhs", n,
            KOKKOS_LAMBDA(const size_t i) {
                if (i == 0) {
                    rhs(i) = bc_left;
                } else if (i == n - 1) {
                    rhs(i) = bc_right;
                }
            });
        Kokkos::fence();
    }

    void apply_obstacle(view_type x, view_type u) {
        const double K = params_.strike;
        const bool is_put = (params_.type == OptionType::Put);
        const size_t n = n_space_;

        Kokkos::parallel_for("apply_obstacle", n,
            KOKKOS_LAMBDA(const size_t i) {
                double S = K * std::exp(x(i));
                double intrinsic = is_put ? ((K > S) ? (K - S) : 0.0)
                                          : ((S > K) ? (S - K) : 0.0);
                if (u(i) < intrinsic) {
                    u(i) = intrinsic;
                }
            });
        Kokkos::fence();
    }

    double interpolate_at_spot(view_type x, view_type u, double x0) {
        auto x_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, x);
        auto u_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, u);

        // Find bracketing indices
        size_t i = 0;
        while (i < n_space_ - 1 && x_h(i + 1) < x0) ++i;

        // Linear interpolation
        double t_interp = (x0 - x_h(i)) / (x_h(i + 1) - x_h(i));
        return u_h(i) * (1.0 - t_interp) + u_h(i + 1) * t_interp;
    }

    double compute_delta(view_type x, view_type u, double x0) {
        auto x_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, x);
        auto u_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, u);

        size_t i = 0;
        while (i < n_space_ - 1 && x_h(i + 1) < x0) ++i;

        // dV/dS = (1/S) * dV/dx
        double dV_dx = (u_h(i + 1) - u_h(i)) / (x_h(i + 1) - x_h(i));
        return dV_dx / params_.spot;
    }

    PricingParams params_;
    size_t n_space_;
    size_t n_time_;
};

}  // namespace mango::kokkos
