#pragma once

/// @file price_table.hpp
/// @brief Price table precomputation and lookup with Kokkos
///
/// Provides GPU-accelerated price table building:
/// 1. Solve many options in parallel (BatchAmericanSolver)
/// 2. Store results in 4D tensor (m, τ, σ, r)
/// 3. Build B-spline interpolator for fast lookups
///
/// Design goals:
/// - Minimize GPU-CPU transfers during build
/// - Support efficient batched queries after build
/// - Compatible with existing BSplineND interpolator

#include <Kokkos_Core.hpp>
#include <expected>
#include <array>
#include <cmath>
#include "kokkos/src/option/batch_solver.hpp"
#include "kokkos/src/math/bspline_nd.hpp"
#include "kokkos/src/math/bspline_basis.hpp"
#include "kokkos/src/support/execution_space.hpp"

namespace mango::kokkos {

/// Configuration for price table builder
struct PriceTableConfig {
    size_t n_space = 101;     ///< Spatial grid points per option
    size_t n_time = 500;      ///< Time steps per option
    double K_ref = 100.0;     ///< Reference strike for normalization
    double q = 0.0;           ///< Dividend yield
    bool is_put = true;       ///< Option type
};

/// Error codes for price table
enum class PriceTableError {
    InvalidDimensions,
    SolverFailed,
    FittingFailed,
    AllocationFailed
};

// Forward declaration
template <typename MemSpace>
class PriceTableBuilder4D;

/// 4D price table result
struct PriceTable4D {
    /// Lookup price at given coordinates
    double lookup(double moneyness, double maturity, double vol, double rate) const;

    /// Get raw price tensor
    [[nodiscard]] Kokkos::View<double****, Kokkos::HostSpace> prices() const {
        return prices_;
    }

    /// Grid dimensions
    std::array<size_t, 4> shape;

    // Grid coordinates (stored on host for lookups)
    std::vector<double> moneyness_grid;
    std::vector<double> maturity_grid;
    std::vector<double> vol_grid;
    std::vector<double> rate_grid;

private:
    template <typename MemSpace>
    friend class PriceTableBuilder4D;
    Kokkos::View<double****, Kokkos::HostSpace> prices_;
};

/// Builder for 4D price tables
///
/// Dimensions: [moneyness, maturity, volatility, rate]
///
/// Build process:
/// 1. For each (vol, rate) pair, solve batch of (moneyness, maturity) options
/// 2. Store results in 4D tensor
/// 3. Optionally fit B-spline for fast interpolation
template <typename MemSpace>
class PriceTableBuilder4D {
public:
    using view_1d = Kokkos::View<double*, MemSpace>;

    /// Construct builder with grid specifications
    ///
    /// @param moneyness Moneyness grid points (S/K ratios)
    /// @param maturity Maturity grid points (years)
    /// @param vol Volatility grid points
    /// @param rate Rate grid points
    /// @param config Additional configuration
    PriceTableBuilder4D(
        view_1d moneyness,
        view_1d maturity,
        view_1d vol,
        view_1d rate,
        const PriceTableConfig& config = PriceTableConfig{})
        : moneyness_(moneyness)
        , maturity_(maturity)
        , vol_(vol)
        , rate_(rate)
        , config_(config)
        , n_m_(moneyness.extent(0))
        , n_tau_(maturity.extent(0))
        , n_sigma_(vol.extent(0))
        , n_r_(rate.extent(0))
    {}

    /// Build the price table
    ///
    /// Solves American options for all grid points.
    /// Results stored in 4D tensor on host.
    ///
    /// @return Price table or error
    [[nodiscard]] std::expected<PriceTable4D, PriceTableError> build() {
        // Copy grids to host for result construction
        auto m_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, moneyness_);
        auto tau_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, maturity_);
        auto sigma_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, vol_);
        auto r_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, rate_);

        // Allocate result tensor [m, tau, sigma, r]
        PriceTable4D result;
        result.shape = {n_m_, n_tau_, n_sigma_, n_r_};
        result.prices_ = Kokkos::View<double****, Kokkos::HostSpace>(
            "prices", n_m_, n_tau_, n_sigma_, n_r_);

        // Copy grids to result
        result.moneyness_grid.resize(n_m_);
        result.maturity_grid.resize(n_tau_);
        result.vol_grid.resize(n_sigma_);
        result.rate_grid.resize(n_r_);

        for (size_t i = 0; i < n_m_; ++i) result.moneyness_grid[i] = m_h(i);
        for (size_t i = 0; i < n_tau_; ++i) result.maturity_grid[i] = tau_h(i);
        for (size_t i = 0; i < n_sigma_; ++i) result.vol_grid[i] = sigma_h(i);
        for (size_t i = 0; i < n_r_; ++i) result.rate_grid[i] = r_h(i);

        // For each (sigma, r) pair, solve a batch of options
        // This is the outer loop on host; batched solving happens on device
        for (size_t i_sigma = 0; i_sigma < n_sigma_; ++i_sigma) {
            for (size_t i_r = 0; i_r < n_r_; ++i_r) {
                // For each maturity, solve batch of moneyness options
                for (size_t i_tau = 0; i_tau < n_tau_; ++i_tau) {
                    auto slice_result = solve_moneyness_slice(
                        tau_h(i_tau), sigma_h(i_sigma), r_h(i_r));

                    if (!slice_result.has_value()) {
                        return std::unexpected(slice_result.error());
                    }

                    // Copy results to tensor
                    auto prices_h = Kokkos::create_mirror_view_and_copy(
                        Kokkos::HostSpace{}, slice_result.value());

                    for (size_t i_m = 0; i_m < n_m_; ++i_m) {
                        result.prices_(i_m, i_tau, i_sigma, i_r) = prices_h(i_m);
                    }
                }
            }
        }

        return result;
    }

private:
    /// Solve for all moneyness values at fixed (tau, sigma, r)
    [[nodiscard]] std::expected<view_1d, PriceTableError>
    solve_moneyness_slice(double tau, double sigma, double r) {
        // Create batch params
        BatchPricingParams params{
            .maturity = tau,
            .volatility = sigma,
            .rate = r,
            .dividend_yield = config_.q,
            .is_put = config_.is_put
        };

        // Convert moneyness to spot/strike pairs
        // moneyness = S/K, so S = moneyness * K_ref
        const double K = config_.K_ref;
        const size_t n = n_m_;

        view_1d strikes("strikes", n);
        view_1d spots("spots", n);

        auto moneyness = moneyness_;

        Kokkos::parallel_for("setup_strikes", n,
            KOKKOS_LAMBDA(const size_t i) {
                double m = moneyness(i);
                strikes(i) = K;           // Fixed reference strike
                spots(i) = m * K;         // Spot from moneyness
            });
        Kokkos::fence();

        // Solve batch
        BatchAmericanSolver<MemSpace> solver(params, strikes, spots,
                                              config_.n_space, config_.n_time);

        auto batch_result = solver.solve();
        if (!batch_result.has_value()) {
            return std::unexpected(PriceTableError::SolverFailed);
        }

        // Extract just prices
        view_1d prices("prices", n);
        auto results = batch_result.value();

        Kokkos::parallel_for("extract_prices", n,
            KOKKOS_LAMBDA(const size_t i) {
                prices(i) = results(i).price;
            });
        Kokkos::fence();

        return prices;
    }

    view_1d moneyness_;
    view_1d maturity_;
    view_1d vol_;
    view_1d rate_;
    PriceTableConfig config_;
    size_t n_m_, n_tau_, n_sigma_, n_r_;
};

// Implementation of PriceTable4D lookup
inline double PriceTable4D::lookup(double moneyness, double maturity,
                                    double vol, double rate) const {
    // Simple linear interpolation in 4D
    // Find bracketing indices in each dimension
    auto find_bracket = [](const std::vector<double>& grid, double x) -> std::pair<size_t, double> {
        if (x <= grid.front()) return {0, 0.0};
        if (x >= grid.back()) return {grid.size() - 2, 1.0};

        size_t i = 0;
        while (i < grid.size() - 1 && grid[i + 1] < x) ++i;

        double t = (x - grid[i]) / (grid[i + 1] - grid[i]);
        return {i, t};
    };

    auto [im, tm] = find_bracket(moneyness_grid, moneyness);
    auto [it, tt] = find_bracket(maturity_grid, maturity);
    auto [is, ts] = find_bracket(vol_grid, vol);
    auto [ir, tr] = find_bracket(rate_grid, rate);

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
                    result += w * prices_(im + dm, it + dt, is + ds, ir + dr);
                }
            }
        }
    }

    return result;
}

}  // namespace mango::kokkos
