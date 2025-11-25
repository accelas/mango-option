#pragma once

/// @file american_option.hpp
/// @brief American option pricing with Kokkos
///
/// Implements finite difference solver for American options in
/// log-moneyness coordinates using TR-BDF2 time-stepping with sinh-spaced grids.
/// Feature parity with the original C++ implementation.

#include <Kokkos_Core.hpp>
#include <expected>
#include <cmath>
#include "kokkos/src/pde/core/grid.hpp"
#include "kokkos/src/pde/core/workspace.hpp"
#include "kokkos/src/pde/core/pde_solver.hpp"
#include "kokkos/src/pde/operators/black_scholes.hpp"
#include "kokkos/src/math/thomas_solver.hpp"
#include "kokkos/src/support/execution_space.hpp"

namespace mango::kokkos {

enum class OptionType { Call, Put };

/// Grid accuracy parameters for automatic estimation
///
/// Controls spatial/temporal resolution tradeoffs for American option PDE solver.
/// Matches the original C++ version's GridAccuracyParams.
struct GridAccuracyParams {
    /// Domain half-width in units of σ√T (default: 5.0 covers ±5 std devs)
    double n_sigma = 5.0;

    /// Sinh clustering strength (default: 2.0 concentrates points near strike)
    double alpha = 2.0;

    /// Target spatial truncation error (default: 1e-2 for ~1e-3 price accuracy)
    /// - 1e-2: Fast mode (~100-150 points, ~5ms per option)
    /// - 1e-3: Medium accuracy (~300-400 points, ~50ms per option)
    /// - 1e-6: High accuracy mode (~1200 points, ~300ms per option)
    double tol = 1e-2;

    /// CFL safety factor for time step (default: 0.75)
    double c_t = 0.75;

    /// Minimum spatial grid points (default: 100)
    size_t min_spatial_points = 100;

    /// Maximum spatial grid points (default: 1200)
    size_t max_spatial_points = 1200;

    /// Maximum time steps (default: 5000)
    size_t max_time_steps = 5000;
};

/// Grid parameters for estimation
struct GridParams {
    double x_min;
    double x_max;
    size_t n_points;
    double alpha;  // sinh clustering parameter
};

/// Time domain parameters
struct TimeParams {
    double T;
    size_t n_steps;
};

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

/// Estimate grid parameters for batch pricing
///
/// Automatically determines appropriate spatial/temporal discretization
/// for batch option pricing where all options share the same maturity and volatility.
///
/// @param maturity Time to expiration (years)
/// @param volatility Annualized volatility
/// @param accuracy Grid accuracy parameters
/// @return Pair of (n_space, n_time)
inline std::pair<size_t, size_t> estimate_batch_grid(
    double maturity,
    double volatility,
    const GridAccuracyParams& accuracy = GridAccuracyParams{})
{
    // Domain width in log-moneyness
    double sigma_sqrt_T = volatility * std::sqrt(maturity);
    double domain_width = 2.0 * accuracy.n_sigma * sigma_sqrt_T;

    // Spatial resolution based on truncation error tolerance
    double dx_target = volatility * std::sqrt(accuracy.tol);
    size_t Nx = static_cast<size_t>(std::ceil(domain_width / dx_target));
    Nx = std::clamp(Nx, accuracy.min_spatial_points, accuracy.max_spatial_points);

    // Ensure odd number of points (for centered stencils)
    if (Nx % 2 == 0) Nx++;

    // Time step based on CFL condition
    // For sinh grid with clustering α, dx_min ≈ dx_avg · exp(-α)
    double dx_avg = domain_width / static_cast<double>(Nx);
    double dx_min = dx_avg * std::exp(-accuracy.alpha);  // Sinh clustering factor
    double dt = accuracy.c_t * dx_min;
    size_t Nt = static_cast<size_t>(std::ceil(maturity / dt));
    Nt = std::min(Nt, accuracy.max_time_steps);  // Upper bound for stability

    return {Nx, Nt};
}

/// Estimate grid parameters for American option pricing
///
/// Automatically determines appropriate spatial/temporal discretization
/// based on option characteristics (volatility, maturity, moneyness).
///
/// @param params Option pricing parameters
/// @param accuracy Grid accuracy parameters
/// @return Pair of (GridParams, TimeParams)
inline std::pair<GridParams, TimeParams> estimate_grid_for_option(
    const PricingParams& params,
    const GridAccuracyParams& accuracy = GridAccuracyParams{})
{
    // Domain bounds (centered on current moneyness)
    double sigma_sqrt_T = params.volatility * std::sqrt(params.maturity);
    double x0 = std::log(params.spot / params.strike);
    double x_min = x0 - accuracy.n_sigma * sigma_sqrt_T;
    double x_max = x0 + accuracy.n_sigma * sigma_sqrt_T;

    // Use batch estimation for grid size
    auto [Nx, Nt] = estimate_batch_grid(params.maturity, params.volatility, accuracy);

    return {GridParams{x_min, x_max, Nx, accuracy.alpha}, TimeParams{params.maturity, Nt}};
}

/// American option solver using finite differences
///
/// Solves Black-Scholes PDE in log-moneyness coordinates with
/// early exercise via obstacle projection.
///
/// Mathematical formulation:
///   V_t = 0.5*sigma^2*V_xx + (r-q-0.5*sigma^2)*V_x - r*V
///   subject to V >= payoff (early exercise constraint)
///
/// Accuracy: O(dt²) + O(dx²) with TR-BDF2 time stepping
template <typename MemSpace>
class AmericanOptionSolver {
public:
    using view_type = Kokkos::View<double*, MemSpace>;

    /// Construct solver with automatic grid estimation
    explicit AmericanOptionSolver(const PricingParams& params)
        : params_(params) {
        // Auto-estimate grid
        auto [grid_params, time_params] = estimate_grid_for_option(params);
        grid_params_ = grid_params;
        time_params_ = time_params;
    }

    /// Construct solver with explicit grid parameters
    AmericanOptionSolver(const PricingParams& params,
                         GridParams grid_params,
                         TimeParams time_params)
        : params_(params), grid_params_(grid_params), time_params_(time_params) {}

    [[nodiscard]] std::expected<PricingResult, SolverError> solve() {
        // Create sinh-spaced grid (concentrates points near strike)
        auto grid_result = Grid<MemSpace>::sinh_spaced(
            grid_params_.x_min, grid_params_.x_max,
            grid_params_.n_points, grid_params_.alpha);
        if (!grid_result.has_value()) {
            return std::unexpected(SolverError::GridError);
        }
        auto grid = std::move(grid_result.value());

        // Create workspace
        auto ws_result = PDEWorkspace<MemSpace>::create(grid_params_.n_points);
        if (!ws_result.has_value()) {
            return std::unexpected(SolverError::GridError);
        }
        auto workspace = std::move(ws_result.value());

        // Initialize with payoff
        auto u = grid.u_current();
        auto x = grid.x();
        initialize_payoff(x, u);

        // Black-Scholes operator
        BlackScholesOperator<MemSpace> bs_op(params_.volatility, params_.rate,
                                              params_.dividend_yield);

        // TR-BDF2 solver
        TRBDF2Solver<MemSpace> solver(grid, workspace);

        // Average grid spacing (passed but not used in non-uniform operator)
        double dx_avg = (grid_params_.x_max - grid_params_.x_min) /
                       static_cast<double>(grid_params_.n_points - 1);

        // Define callbacks for TR-BDF2 with non-uniform grid support
        auto assemble_jacobian = [&](double coeff_dt, view_type lower,
                                      view_type diag, view_type upper) {
            bs_op.assemble_jacobian(coeff_dt, x, lower, diag, upper);
        };

        auto apply_operator = [&](view_type u_in, view_type Lu_out) {
            bs_op.apply(x, u_in, Lu_out, dx_avg);
        };

        auto apply_bc_matrix = [&](view_type lower, view_type diag, view_type upper) {
            apply_boundary_conditions_to_matrix(lower, diag, upper);
        };

        auto apply_bc_rhs = [&](view_type rhs, double t, double dt, size_t step) {
            set_boundary_rhs(x, rhs, grid_params_.x_min, grid_params_.x_max, t, dt, step);
        };

        auto compute_obstacle_fn = [&](view_type x_grid, view_type psi) {
            compute_obstacle(x_grid, psi);
        };

        // Solve with TR-BDF2 using Projected Thomas for obstacle constraint
        solver.solve(time_params_.T, time_params_.n_steps,
                     assemble_jacobian, apply_operator, apply_bc_matrix, apply_bc_rhs,
                     compute_obstacle_fn);

        // Interpolate price at spot
        double x0 = std::log(params_.spot / params_.strike);
        double price = interpolate_at_spot(x, u, x0);
        double delta = compute_delta(x, u, x0);

        return PricingResult{.price = price, .delta = delta};
    }

private:
    /// Initialize with normalized payoff: V/K
    /// Put: max(1 - exp(x), 0)  where x = ln(S/K)
    /// Call: max(exp(x) - 1, 0)
    void initialize_payoff(view_type x, view_type u) {
        const bool is_put = (params_.type == OptionType::Put);
        const size_t n = grid_params_.n_points;

        Kokkos::parallel_for("init_payoff", n,
            KOKKOS_LAMBDA(const size_t i) {
                double exp_x = std::exp(x(i));
                if (is_put) {
                    u(i) = (1.0 > exp_x) ? (1.0 - exp_x) : 0.0;
                } else {
                    u(i) = (exp_x > 1.0) ? (exp_x - 1.0) : 0.0;
                }
            });
        Kokkos::fence();
    }

    void apply_boundary_conditions_to_matrix(view_type lower, view_type diag, view_type upper) {
        const size_t n = grid_params_.n_points;
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

    /// Set boundary conditions in RHS
    /// Uses normalized coordinates (V/K) with simple intrinsic BCs
    /// t is PDE time (0 to T), financial time to expiry is τ = T - t
    /// Put: Left BC = max(1-exp(x), 0), Right BC = 0
    /// Call: Left BC = 0, Right BC = exp(x) - exp(-r*τ)
    void set_boundary_rhs(view_type x, view_type rhs, double x_left, double x_right,
                          double t, double /*dt*/, size_t /*step*/) {
        const size_t n = grid_params_.n_points;
        const bool is_put = (params_.type == OptionType::Put);
        const double r = params_.rate;
        const double T = time_params_.T;

        // Financial time to expiry: τ = T - t
        // At t=0 (start of integration), τ = T (full time to expiry)
        // At t=T (end of integration), τ = 0 (at expiry)
        double tau = T - t;

        double bc_left, bc_right;
        if (is_put) {
            // Put: Left is deep ITM (intrinsic), Right is deep OTM (0)
            double exp_x_left = std::exp(x_left);
            bc_left = (1.0 > exp_x_left) ? (1.0 - exp_x_left) : 0.0;
            bc_right = 0.0;
        } else {
            // Call: Left is deep OTM (0), Right is deep ITM
            bc_left = 0.0;
            double exp_x_right = std::exp(x_right);
            // For calls, right BC is exp(x) - exp(-r*τ) where τ is time to expiry
            double discount = std::exp(-r * tau);
            bc_right = exp_x_right - discount;
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

    /// Compute obstacle values (for Projected Thomas solver)
    /// Uses normalized coordinates: ψ = intrinsic/K
    /// Put: max(1 - exp(x), 0)
    /// Call: max(exp(x) - 1, 0)
    void compute_obstacle(view_type x, view_type psi) {
        const bool is_put = (params_.type == OptionType::Put);
        const size_t n = grid_params_.n_points;

        Kokkos::parallel_for("compute_obstacle", n,
            KOKKOS_LAMBDA(const size_t i) {
                double exp_x = std::exp(x(i));
                if (is_put) {
                    psi(i) = (1.0 > exp_x) ? (1.0 - exp_x) : 0.0;
                } else {
                    psi(i) = (exp_x > 1.0) ? (exp_x - 1.0) : 0.0;
                }
            });
        Kokkos::fence();
    }

    /// Interpolate normalized price at spot and convert to actual price
    /// Returns V = (V/K) * K
    double interpolate_at_spot(view_type x, view_type u, double x0) {
        auto x_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, x);
        auto u_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, u);

        // Find bracketing indices
        size_t i = 0;
        while (i < grid_params_.n_points - 1 && x_h(i + 1) < x0) ++i;

        // Linear interpolation of normalized price V/K
        double t_interp = (x0 - x_h(i)) / (x_h(i + 1) - x_h(i));
        double normalized_price = u_h(i) * (1.0 - t_interp) + u_h(i + 1) * t_interp;

        // Convert to actual price: V = (V/K) * K
        return normalized_price * params_.strike;
    }

    /// Compute delta: dV/dS
    /// In normalized coordinates: V = K * v(x) where x = ln(S/K)
    /// dV/dS = dV/dx * dx/dS = K * dv/dx * (1/S) = (K/S) * dv/dx
    double compute_delta(view_type x, view_type u, double x0) {
        auto x_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, x);
        auto u_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, u);

        size_t i = 0;
        while (i < grid_params_.n_points - 1 && x_h(i + 1) < x0) ++i;

        // dv/dx (normalized derivative)
        double dv_dx = (u_h(i + 1) - u_h(i)) / (x_h(i + 1) - x_h(i));

        // dV/dS = (K/S) * dv/dx
        return (params_.strike / params_.spot) * dv_dx;
    }

    PricingParams params_;
    GridParams grid_params_;
    TimeParams time_params_;
};

}  // namespace mango::kokkos
