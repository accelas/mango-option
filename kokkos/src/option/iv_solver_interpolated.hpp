#pragma once

/// @file iv_solver_interpolated.hpp
/// @brief GPU-accelerated implied volatility solver using price table interpolation
///
/// Solves for implied volatility using Newton's method with interpolated
/// option prices from pre-computed 4D price table. Enables batch IV
/// calculation in parallel on GPU.
///
/// Algorithm:
/// - Newton-Raphson iteration: σ_{n+1} = σ_n - f(σ_n)/f'(σ_n)
/// - f(σ) = PriceTable.lookup(m, τ, σ, r) * (K/K_ref) - Market_Price
/// - f'(σ) = Vega(σ) ≈ [Price(σ+ε) - Price(σ-ε)] / (2ε) * (K/K_ref)
/// - All computation on GPU with Kokkos::parallel_for

#include <Kokkos_Core.hpp>
#include <expected>
#include <array>
#include <cmath>
#include "kokkos/src/math/root_finding.hpp"
#include "kokkos/src/option/price_table.hpp"
#include "kokkos/src/option/iv_common.hpp"
#include "kokkos/src/support/execution_space.hpp"

namespace mango::kokkos {

/// Configuration for interpolation-based IV solver
struct IVSolverConfig {
    int max_iterations = 50;      ///< Maximum Newton iterations
    double tolerance = 1e-6;       ///< Price convergence tolerance
    double sigma_min = 0.01;       ///< Minimum volatility (1%)
    double sigma_max = 3.0;        ///< Maximum volatility (300%)
};

/// IV result
struct IVResult {
    double implied_vol;
    size_t iterations;
    double final_error;
    bool converged;
};

/// Error codes for IV solver
enum class IVSolverError {
    InvalidQuery,
    TableMismatch,
    AllocationFailed
};

/// Interpolation-based IV Solver (GPU-accelerated)
///
/// Uses pre-computed price table for ultra-fast IV calculation in parallel.
/// Solves: Find σ such that PriceTable(m, τ, σ, r) * (K/K_ref) = Market_Price
///
/// @tparam MemSpace Kokkos memory space (HostSpace, CudaSpace, etc.)
template <typename MemSpace>
class IVSolverInterpolated {
public:
    using view_query = Kokkos::View<IVQuery*, MemSpace>;
    using view_result = Kokkos::View<IVResult*, MemSpace>;

    /// Create solver from price table
    ///
    /// @param table Pre-computed 4D price table
    /// @param config Solver configuration
    /// @return IV solver or error
    [[nodiscard]] static std::expected<IVSolverInterpolated, IVSolverError> create(
        const PriceTable4D& table,
        const IVSolverConfig& config = {})
    {
        // Validate table
        if (table.shape[0] < 2 || table.shape[1] < 2 ||
            table.shape[2] < 2 || table.shape[3] < 2) {
            return std::unexpected(IVSolverError::TableMismatch);
        }

        return IVSolverInterpolated(table, config);
    }

    /// Solve batch of IV queries in parallel
    ///
    /// @param queries Input queries (device View)
    /// @return View of IV results (device)
    [[nodiscard]] std::expected<view_result, IVSolverError>
    solve_batch(const view_query& queries) const
    {
        const size_t n = queries.extent(0);

        // Allocate result View
        view_result results("iv_results", n);

        // Copy table data to device for kernel access
        // We'll use the table's lookup method which handles interpolation
        const auto& table = table_;
        const auto& config = config_;

        // Convert table grids to device Views for kernel
        const size_t n_m = table.shape[0];
        const size_t n_tau = table.shape[1];
        const size_t n_sigma = table.shape[2];
        const size_t n_r = table.shape[3];

        // Copy grids to device
        Kokkos::View<double*, MemSpace> m_grid_d("m_grid", n_m);
        Kokkos::View<double*, MemSpace> tau_grid_d("tau_grid", n_tau);
        Kokkos::View<double*, MemSpace> sigma_grid_d("sigma_grid", n_sigma);
        Kokkos::View<double*, MemSpace> r_grid_d("r_grid", n_r);

        auto m_h = Kokkos::create_mirror_view(m_grid_d);
        auto tau_h = Kokkos::create_mirror_view(tau_grid_d);
        auto sigma_h = Kokkos::create_mirror_view(sigma_grid_d);
        auto r_h = Kokkos::create_mirror_view(r_grid_d);

        for (size_t i = 0; i < n_m; ++i) m_h(i) = table.moneyness_grid[i];
        for (size_t i = 0; i < n_tau; ++i) tau_h(i) = table.maturity_grid[i];
        for (size_t i = 0; i < n_sigma; ++i) sigma_h(i) = table.vol_grid[i];
        for (size_t i = 0; i < n_r; ++i) r_h(i) = table.rate_grid[i];

        Kokkos::deep_copy(m_grid_d, m_h);
        Kokkos::deep_copy(tau_grid_d, tau_h);
        Kokkos::deep_copy(sigma_grid_d, sigma_h);
        Kokkos::deep_copy(r_grid_d, r_h);

        // Copy price tensor to device
        Kokkos::View<double****, MemSpace> prices_d("prices", n_m, n_tau, n_sigma, n_r);
        auto prices_h = table.prices();
        Kokkos::deep_copy(prices_d, prices_h);

        // Capture bounds for kernel
        const double sigma_min = config.sigma_min;
        const double sigma_max = config.sigma_max;
        const double tolerance = config.tolerance;
        const size_t max_iter = static_cast<size_t>(config.max_iterations);

        // Capture Views by value for kernel
        auto m_grid_kernel = m_grid_d;
        auto tau_grid_kernel = tau_grid_d;
        auto sigma_grid_kernel = sigma_grid_d;
        auto r_grid_kernel = r_grid_d;
        auto prices_kernel = prices_d;

        // Parallel solve for each query
        Kokkos::parallel_for("iv_solve_batch", n,
            KOKKOS_LAMBDA(const size_t i) {
                const auto& query = queries(i);

                // Compute moneyness
                const double m = query.spot / query.strike;
                const double tau = query.maturity;
                const double r = query.rate;
                const double market_price = query.market_price;
                const double strike = query.strike;

                // K_ref is implicitly 1.0 in the price table (prices are normalized)
                // We need to scale by strike to get absolute price
                const double K_ref = 100.0;  // Assumed reference strike
                const double scale_factor = strike / K_ref;

                // Initial guess (midpoint)
                double sigma = (sigma_min + sigma_max) * 0.5;

                // Manual Newton iteration (avoid nested lambdas which don't work on device)
                bool converged = false;
                size_t iter = 0;
                double final_error = 0.0;

                // Find fixed indices for m, tau, r (don't change during iteration)
                size_t im = 0;
                while (im < n_m - 1 && m_grid_kernel(im + 1) < m) ++im;
                size_t it_idx = 0;
                while (it_idx < n_tau - 1 && tau_grid_kernel(it_idx + 1) < tau) ++it_idx;
                size_t ir = 0;
                while (ir < n_r - 1 && r_grid_kernel(ir + 1) < r) ++ir;

                double tm = (m - m_grid_kernel(im)) / (m_grid_kernel(im + 1) - m_grid_kernel(im));
                double tt = (tau - tau_grid_kernel(it_idx)) / (tau_grid_kernel(it_idx + 1) - tau_grid_kernel(it_idx));
                double tr = (r - r_grid_kernel(ir)) / (r_grid_kernel(ir + 1) - r_grid_kernel(ir));

                for (iter = 0; iter < max_iter; ++iter) {
                    // Find index for current sigma
                    size_t is = 0;
                    while (is < n_sigma - 1 && sigma_grid_kernel(is + 1) < sigma) ++is;
                    double ts = (sigma - sigma_grid_kernel(is)) / (sigma_grid_kernel(is + 1) - sigma_grid_kernel(is));

                    // Evaluate price via inline 4D interpolation
                    double price = 0.0;
                    for (int dm = 0; dm <= 1; ++dm) {
                        for (int dt = 0; dt <= 1; ++dt) {
                            for (int ds = 0; ds <= 1; ++ds) {
                                for (int dr = 0; dr <= 1; ++dr) {
                                    double wm = dm ? tm : (1.0 - tm);
                                    double wt = dt ? tt : (1.0 - tt);
                                    double ws = ds ? ts : (1.0 - ts);
                                    double wr = dr ? tr : (1.0 - tr);
                                    price += wm * wt * ws * wr * prices_kernel(im + dm, it_idx + dt, is + ds, ir + dr);
                                }
                            }
                        }
                    }

                    double fx = price * scale_factor - market_price;
                    final_error = Kokkos::fabs(fx);

                    // Check convergence
                    if (final_error < tolerance) {
                        converged = true;
                        break;
                    }

                    // Compute derivative via finite difference
                    const double eps = 1e-5;
                    double sigma_plus = sigma + eps;
                    double sigma_minus = sigma - eps;
                    if (sigma_plus > sigma_max) sigma_plus = sigma_max;
                    if (sigma_minus < sigma_min) sigma_minus = sigma_min;

                    // Lookup at sigma_plus
                    size_t is_plus = 0;
                    while (is_plus < n_sigma - 1 && sigma_grid_kernel(is_plus + 1) < sigma_plus) ++is_plus;
                    double ts_plus = (sigma_plus - sigma_grid_kernel(is_plus)) / (sigma_grid_kernel(is_plus + 1) - sigma_grid_kernel(is_plus));

                    double price_plus = 0.0;
                    for (int dm = 0; dm <= 1; ++dm) {
                        for (int dt = 0; dt <= 1; ++dt) {
                            for (int ds = 0; ds <= 1; ++ds) {
                                for (int dr = 0; dr <= 1; ++dr) {
                                    double wm = dm ? tm : (1.0 - tm);
                                    double wt = dt ? tt : (1.0 - tt);
                                    double ws = ds ? ts_plus : (1.0 - ts_plus);
                                    double wr = dr ? tr : (1.0 - tr);
                                    price_plus += wm * wt * ws * wr * prices_kernel(im + dm, it_idx + dt, is_plus + ds, ir + dr);
                                }
                            }
                        }
                    }

                    // Lookup at sigma_minus
                    size_t is_minus = 0;
                    while (is_minus < n_sigma - 1 && sigma_grid_kernel(is_minus + 1) < sigma_minus) ++is_minus;
                    double ts_minus = (sigma_minus - sigma_grid_kernel(is_minus)) / (sigma_grid_kernel(is_minus + 1) - sigma_grid_kernel(is_minus));

                    double price_minus = 0.0;
                    for (int dm = 0; dm <= 1; ++dm) {
                        for (int dt = 0; dt <= 1; ++dt) {
                            for (int ds = 0; ds <= 1; ++ds) {
                                for (int dr = 0; dr <= 1; ++dr) {
                                    double wm = dm ? tm : (1.0 - tm);
                                    double wt = dt ? tt : (1.0 - tt);
                                    double ws = ds ? ts_minus : (1.0 - ts_minus);
                                    double wr = dr ? tr : (1.0 - tr);
                                    price_minus += wm * wt * ws * wr * prices_kernel(im + dm, it_idx + dt, is_minus + ds, ir + dr);
                                }
                            }
                        }
                    }

                    double dfx = (price_plus - price_minus) * scale_factor / (sigma_plus - sigma_minus);

                    // Check for flat derivative
                    if (Kokkos::fabs(dfx) < 1e-10) {
                        break;
                    }

                    // Newton step
                    sigma = sigma - fx / dfx;
                    if (sigma < sigma_min) sigma = sigma_min;
                    if (sigma > sigma_max) sigma = sigma_max;
                }

                // Store result
                results(i) = IVResult{
                    .implied_vol = converged ? sigma : 0.0,
                    .iterations = converged ? (iter + 1) : iter,
                    .final_error = final_error,
                    .converged = converged
                };
            });

        Kokkos::fence();

        return results;
    }

private:
    PriceTable4D table_;
    IVSolverConfig config_;

    /// Private constructor (use create() factory)
    IVSolverInterpolated(
        const PriceTable4D& table,
        const IVSolverConfig& config)
        : table_(table)
        , config_(config)
    {}

    /// Device-callable lookup function (4D linear interpolation)
    KOKKOS_INLINE_FUNCTION
    static double lookup_device(
        double moneyness, double maturity, double vol, double rate,
        const Kokkos::View<double*, MemSpace>& m_grid,
        const Kokkos::View<double*, MemSpace>& tau_grid,
        const Kokkos::View<double*, MemSpace>& sigma_grid,
        const Kokkos::View<double*, MemSpace>& r_grid,
        const Kokkos::View<double****, MemSpace>& prices,
        size_t n_m, size_t n_tau, size_t n_sigma, size_t n_r) noexcept
    {
        // Find bracketing indices for moneyness
        size_t im = 0;
        double tm = 0.0;
        if (moneyness <= m_grid(0)) {
            im = 0;
            tm = 0.0;
        } else if (moneyness >= m_grid(n_m - 1)) {
            im = n_m - 2;
            tm = 1.0;
        } else {
            while (im < n_m - 1 && m_grid(im + 1) < moneyness) ++im;
            tm = (moneyness - m_grid(im)) / (m_grid(im + 1) - m_grid(im));
        }

        // Find bracketing indices for maturity
        size_t it = 0;
        double tt = 0.0;
        if (maturity <= tau_grid(0)) {
            it = 0;
            tt = 0.0;
        } else if (maturity >= tau_grid(n_tau - 1)) {
            it = n_tau - 2;
            tt = 1.0;
        } else {
            while (it < n_tau - 1 && tau_grid(it + 1) < maturity) ++it;
            tt = (maturity - tau_grid(it)) / (tau_grid(it + 1) - tau_grid(it));
        }

        // Find bracketing indices for volatility
        size_t is = 0;
        double ts = 0.0;
        if (vol <= sigma_grid(0)) {
            is = 0;
            ts = 0.0;
        } else if (vol >= sigma_grid(n_sigma - 1)) {
            is = n_sigma - 2;
            ts = 1.0;
        } else {
            while (is < n_sigma - 1 && sigma_grid(is + 1) < vol) ++is;
            ts = (vol - sigma_grid(is)) / (sigma_grid(is + 1) - sigma_grid(is));
        }

        // Find bracketing indices for rate
        size_t ir = 0;
        double tr = 0.0;
        if (rate <= r_grid(0)) {
            ir = 0;
            tr = 0.0;
        } else if (rate >= r_grid(n_r - 1)) {
            ir = n_r - 2;
            tr = 1.0;
        } else {
            while (ir < n_r - 1 && r_grid(ir + 1) < rate) ++ir;
            tr = (rate - r_grid(ir)) / (r_grid(ir + 1) - r_grid(ir));
        }

        // Trilinear interpolation in 4D (16 corners)
        double result = 0.0;
        for (int dm = 0; dm <= 1; ++dm) {
            double wm = dm ? tm : (1.0 - tm);
            for (int dt = 0; dt <= 1; ++dt) {
                double wt = dt ? tt : (1.0 - tt);
                for (int ds = 0; ds <= 1; ++ds) {
                    double ws = ds ? ts : (1.0 - ts);
                    for (int dr = 0; dr <= 1; ++dr) {
                        double wr = dr ? tr : (1.0 - tr);
                        double w = wm * wt * ws * wr;
                        result += w * prices(im + dm, it + dt, is + ds, ir + dr);
                    }
                }
            }
        }

        return result;
    }
};

}  // namespace mango::kokkos
