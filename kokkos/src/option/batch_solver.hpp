#pragma once

/// @file batch_solver.hpp
/// @brief Batched American option pricing with Kokkos
///
/// Provides parallel pricing of multiple American options sharing:
/// - Same volatility, rate, and dividend yield
/// - Same maturity
/// - Different strikes (for building price tables)
///
/// This is the key performance-critical component for:
/// 1. Price table precomputation
/// 2. Implied volatility calibration
///
/// Design choices:
/// - Options batched along first dimension
/// - Grid points along second dimension
/// - Thomas solver runs independently per option
/// - Obstacle projection vectorized across options

#include <Kokkos_Core.hpp>
#include <expected>
#include <cmath>
#include "kokkos/src/pde/core/grid.hpp"
#include "kokkos/src/pde/core/workspace.hpp"
#include "kokkos/src/pde/operators/black_scholes.hpp"
#include "kokkos/src/math/thomas_solver.hpp"
#include "kokkos/src/support/execution_space.hpp"

namespace mango::kokkos {

/// Batch pricing parameters (shared across all options)
struct BatchPricingParams {
    double maturity;
    double volatility;
    double rate;
    double dividend_yield;
    bool is_put;
};

/// Error codes for batch solver
enum class BatchSolverError {
    InvalidParams,
    GridError,
    AllocationFailed
};

/// Result for a single option in the batch
struct BatchOptionResult {
    double price;
    double delta;
};

/// Batched American option solver
///
/// Solves N American options simultaneously with shared PDE parameters
/// but different strikes. Uses 2D parallelism:
/// - Outer: options (batch dimension)
/// - Inner: spatial grid points
///
/// @tparam MemSpace Kokkos memory space
template <typename MemSpace>
class BatchAmericanSolver {
public:
    using view_1d = Kokkos::View<double*, MemSpace>;
    using view_2d = Kokkos::View<double**, MemSpace>;

    /// Create batch solver with given grid specification
    ///
    /// @param params Shared pricing parameters (vol, rate, maturity)
    /// @param strikes View of strike prices (defines batch size)
    /// @param spots View of spot prices (one per option)
    /// @param n_space Spatial grid points
    /// @param n_time Time steps
    BatchAmericanSolver(const BatchPricingParams& params,
                        view_1d strikes,
                        view_1d spots,
                        size_t n_space = 201,
                        size_t n_time = 1000)
        : params_(params)
        , strikes_(strikes)
        , spots_(spots)
        , n_batch_(strikes.extent(0))
        , n_space_(n_space)
        , n_time_(n_time)
    {
        // Allocate 2D arrays: [n_batch, n_space]
        u_current_ = view_2d("u_current", n_batch_, n_space_);
        u_prev_ = view_2d("u_prev", n_batch_, n_space_);
        rhs_ = view_2d("rhs", n_batch_, n_space_);
        solution_ = view_2d("solution", n_batch_, n_space_);

        // Allocate per-option tridiagonal matrices
        // Note: lower/upper have n_space-1 elements
        jacobian_diag_ = view_2d("jacobian_diag", n_batch_, n_space_);
        jacobian_lower_ = view_2d("jacobian_lower", n_batch_, n_space_ - 1);
        jacobian_upper_ = view_2d("jacobian_upper", n_batch_, n_space_ - 1);

        // Thomas solver temporaries
        thomas_diag_temp_ = view_2d("thomas_diag_temp", n_batch_, n_space_);
        thomas_lower_temp_ = view_2d("thomas_lower_temp", n_batch_, n_space_ - 1);
        thomas_upper_temp_ = view_2d("thomas_upper_temp", n_batch_, n_space_ - 1);

        // Grid coordinates (shared across batch, but stored per-option for strike-dependent bounds)
        x_ = view_2d("x", n_batch_, n_space_);

        // Results
        results_ = Kokkos::View<BatchOptionResult*, MemSpace>("results", n_batch_);
    }

    /// Solve all options in the batch
    ///
    /// @return View of results (one per option)
    [[nodiscard]] std::expected<Kokkos::View<BatchOptionResult*, MemSpace>, BatchSolverError> solve() {
        // Copy params to device-accessible values
        const double sigma = params_.volatility;
        const double r = params_.rate;
        const double q = params_.dividend_yield;
        const double T = params_.maturity;
        const bool is_put = params_.is_put;
        const size_t n_batch = n_batch_;
        const size_t n_space = n_space_;
        const size_t n_time = n_time_;

        // Black-Scholes coefficients
        const double half_sigma_sq = 0.5 * sigma * sigma;
        const double drift = r - q - half_sigma_sq;

        // Compute dt once
        const double dt = T / static_cast<double>(n_time);

        // Access Views
        auto x = x_;
        auto strikes = strikes_;
        auto spots = spots_;
        auto u_current = u_current_;
        auto u_prev = u_prev_;
        auto rhs = rhs_;
        auto solution = solution_;
        auto jacobian_diag = jacobian_diag_;
        auto jacobian_lower = jacobian_lower_;
        auto jacobian_upper = jacobian_upper_;
        auto thomas_diag_temp = thomas_diag_temp_;
        auto thomas_lower_temp = thomas_lower_temp_;
        auto thomas_upper_temp = thomas_upper_temp_;
        auto results = results_;

        // ===== Step 1: Initialize grid and payoff for each option =====
        // Each option has its own grid centered at its log-moneyness
        const double sigma_sqrt_T = sigma * std::sqrt(T);

        Kokkos::parallel_for("init_batch",
            Kokkos::MDRangePolicy<Kokkos::Rank<2>,
                typename MemSpace::execution_space>({0, 0}, {n_batch, n_space}),
            KOKKOS_LAMBDA(const size_t opt, const size_t i) {
                const double K = strikes(opt);
                const double S = spots(opt);
                const double x0 = Kokkos::log(S / K);

                // Grid bounds in log-moneyness
                const double x_min = x0 - 5.0 * sigma_sqrt_T;
                const double x_max = x0 + 5.0 * sigma_sqrt_T;
                const double dx = (x_max - x_min) / static_cast<double>(n_space - 1);

                // Set grid coordinate
                x(opt, i) = x_min + static_cast<double>(i) * dx;

                // Initialize with payoff
                const double S_grid = K * Kokkos::exp(x(opt, i));
                if (is_put) {
                    u_current(opt, i) = (K > S_grid) ? (K - S_grid) : 0.0;
                } else {
                    u_current(opt, i) = (S_grid > K) ? (S_grid - K) : 0.0;
                }
            });
        Kokkos::fence();

        // ===== Step 2: Assemble Jacobian for each option =====
        // I - dt * L where L is Black-Scholes operator
        Kokkos::parallel_for("assemble_jacobian_batch",
            Kokkos::MDRangePolicy<Kokkos::Rank<2>,
                typename MemSpace::execution_space>({0, 0}, {n_batch, n_space}),
            KOKKOS_LAMBDA(const size_t opt, const size_t i) {
                const double K = strikes(opt);
                const double S = spots(opt);
                const double x0 = Kokkos::log(S / K);
                const double x_min = x0 - 5.0 * sigma_sqrt_T;
                const double x_max = x0 + 5.0 * sigma_sqrt_T;
                const double dx = (x_max - x_min) / static_cast<double>(n_space - 1);
                const double dx_sq = dx * dx;
                const double two_dx = 2.0 * dx;

                // L coefficients
                const double L_lower = half_sigma_sq / dx_sq - drift / two_dx;
                const double L_diag = -2.0 * half_sigma_sq / dx_sq - r;
                const double L_upper = half_sigma_sq / dx_sq + drift / two_dx;

                // Boundary conditions (Dirichlet)
                if (i == 0) {
                    jacobian_diag(opt, i) = 1.0;
                    jacobian_upper(opt, i) = 0.0;
                } else if (i == n_space - 1) {
                    jacobian_diag(opt, i) = 1.0;
                    jacobian_lower(opt, i - 1) = 0.0;
                } else {
                    // Interior: I - dt*L
                    jacobian_diag(opt, i) = 1.0 - dt * L_diag;
                    jacobian_lower(opt, i - 1) = -dt * L_lower;
                    jacobian_upper(opt, i) = -dt * L_upper;
                }
            });
        Kokkos::fence();

        // ===== Step 3: Time stepping =====
        for (size_t step = 0; step < n_time; ++step) {
            const double t = T - static_cast<double>(step + 1) * dt;

            // Copy current to RHS
            Kokkos::deep_copy(rhs, u_current);

            // Apply boundary conditions to RHS
            Kokkos::parallel_for("apply_bc_rhs_batch", n_batch,
                KOKKOS_LAMBDA(const size_t opt) {
                    const double K = strikes(opt);
                    const double S = spots(opt);
                    const double x0 = Kokkos::log(S / K);
                    const double x_left = x0 - 5.0 * sigma_sqrt_T;
                    const double x_right = x0 + 5.0 * sigma_sqrt_T;

                    double bc_left, bc_right;
                    if (is_put) {
                        double S_left = K * Kokkos::exp(x_left);
                        bc_left = K * Kokkos::exp(-r * t) - S_left * Kokkos::exp(-q * t);
                        bc_right = 0.0;
                    } else {
                        bc_left = 0.0;
                        double S_right = K * Kokkos::exp(x_right);
                        bc_right = S_right * Kokkos::exp(-q * t) - K * Kokkos::exp(-r * t);
                    }

                    rhs(opt, 0) = bc_left;
                    rhs(opt, n_space - 1) = bc_right;
                });
            Kokkos::fence();

            // Copy Jacobian to temps (Thomas modifies in place)
            Kokkos::deep_copy(thomas_diag_temp, jacobian_diag);
            Kokkos::deep_copy(thomas_lower_temp, jacobian_lower);
            Kokkos::deep_copy(thomas_upper_temp, jacobian_upper);

            // Solve tridiagonal system per option
            Kokkos::parallel_for("thomas_batch", n_batch,
                KOKKOS_LAMBDA(const size_t opt) {
                    // Extract 1D subviews for this option
                    // Note: We inline Thomas algorithm here for efficiency
                    const size_t n = n_space;

                    // Forward elimination
                    for (size_t i = 1; i < n; ++i) {
                        double w = thomas_lower_temp(opt, i - 1) / thomas_diag_temp(opt, i - 1);
                        thomas_diag_temp(opt, i) -= w * thomas_upper_temp(opt, i - 1);
                        rhs(opt, i) -= w * rhs(opt, i - 1);
                    }

                    // Back substitution
                    solution(opt, n - 1) = rhs(opt, n - 1) / thomas_diag_temp(opt, n - 1);
                    for (int i = static_cast<int>(n) - 2; i >= 0; --i) {
                        solution(opt, i) = (rhs(opt, i) - thomas_upper_temp(opt, i) * solution(opt, i + 1))
                                         / thomas_diag_temp(opt, i);
                    }
                });
            Kokkos::fence();

            // Apply obstacle (early exercise)
            Kokkos::parallel_for("apply_obstacle_batch",
                Kokkos::MDRangePolicy<Kokkos::Rank<2>,
                    typename MemSpace::execution_space>({0, 0}, {n_batch, n_space}),
                KOKKOS_LAMBDA(const size_t opt, const size_t i) {
                    const double K = strikes(opt);
                    const double S_grid = K * Kokkos::exp(x(opt, i));
                    const double intrinsic = is_put ?
                        ((K > S_grid) ? (K - S_grid) : 0.0) :
                        ((S_grid > K) ? (S_grid - K) : 0.0);

                    if (solution(opt, i) < intrinsic) {
                        solution(opt, i) = intrinsic;
                    }
                });
            Kokkos::fence();

            // Copy solution back to current
            Kokkos::deep_copy(u_current, solution);
        }

        // ===== Step 4: Interpolate results at spot price =====
        Kokkos::parallel_for("interpolate_batch", n_batch,
            KOKKOS_LAMBDA(const size_t opt) {
                const double K = strikes(opt);
                const double S = spots(opt);
                const double x0 = Kokkos::log(S / K);

                // Find bracketing indices
                size_t idx = 0;
                while (idx < n_space - 1 && x(opt, idx + 1) < x0) ++idx;

                // Linear interpolation
                const double t_interp = (x0 - x(opt, idx)) / (x(opt, idx + 1) - x(opt, idx));
                const double price = u_current(opt, idx) * (1.0 - t_interp) +
                                     u_current(opt, idx + 1) * t_interp;

                // Delta: dV/dS = (1/S) * dV/dx
                const double dV_dx = (u_current(opt, idx + 1) - u_current(opt, idx)) /
                                     (x(opt, idx + 1) - x(opt, idx));
                const double delta = dV_dx / S;

                results(opt).price = price;
                results(opt).delta = delta;
            });
        Kokkos::fence();

        return results;
    }

    /// Get batch size
    [[nodiscard]] size_t batch_size() const noexcept { return n_batch_; }

    /// Get spatial grid size
    [[nodiscard]] size_t n_space() const noexcept { return n_space_; }

private:
    BatchPricingParams params_;
    view_1d strikes_;
    view_1d spots_;
    size_t n_batch_;
    size_t n_space_;
    size_t n_time_;

    // 2D arrays [batch, space]
    view_2d u_current_;
    view_2d u_prev_;
    view_2d rhs_;
    view_2d solution_;
    view_2d x_;

    // Per-option Jacobian
    view_2d jacobian_diag_;
    view_2d jacobian_lower_;
    view_2d jacobian_upper_;

    // Thomas solver temps
    view_2d thomas_diag_temp_;
    view_2d thomas_lower_temp_;
    view_2d thomas_upper_temp_;

    // Results
    Kokkos::View<BatchOptionResult*, MemSpace> results_;
};

}  // namespace mango::kokkos
