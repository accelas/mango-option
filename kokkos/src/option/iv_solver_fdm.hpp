#pragma once

/// @file iv_solver_fdm.hpp
/// @brief FDM-based implied volatility solver with Kokkos
///
/// Implements implied volatility calculation using finite difference methods
/// (FDM) to price American options and Brent's root-finding to solve for
/// implied volatility.
///
/// Design:
/// - Each IV solve uses Brent's method (inherently serial)
/// - Objective function uses AmericanOptionSolver (device-parallel PDE solve)
/// - Batch solving: serial loop over queries, each with parallel PDE solve
/// - Future optimization: Could batch multiple PDE solves across IV queries

#include <Kokkos_Core.hpp>
#include <expected>
#include <cmath>
#include <functional>
#include "kokkos/src/option/american_option.hpp"
#include "kokkos/src/option/iv_common.hpp"
#include "kokkos/src/math/root_finding.hpp"
#include "kokkos/src/support/execution_space.hpp"

namespace mango::kokkos {

/// Configuration for FDM-based IV solver
struct IVSolverFDMConfig {
    /// Maximum iterations for Brent's method
    size_t max_iterations = 100;

    /// Absolute tolerance for convergence
    double tolerance = 1e-6;

    /// Minimum volatility bound for search
    double sigma_min = 0.01;

    /// Maximum volatility bound for search
    double sigma_max = 3.0;

    /// Number of spatial grid points for PDE solver
    size_t n_space = 101;

    /// Number of time steps for PDE solver
    size_t n_time = 500;
};

/// Result codes for IV solver
enum class IVResultCode {
    Success,
    MaxIterationsExceeded,
    BracketingFailed,
    NumericalInstability,
    InvalidParams,
    ArbitrageViolation
};

/// Result from IV solve
struct IVResultFDM {
    double implied_vol;
    size_t iterations;
    double final_error;
    bool converged;
    IVResultCode code;
};

/// FDM-based Implied Volatility Solver for American Options
///
/// Finds the volatility parameter that makes the American option's
/// theoretical price (from PDE solver) match the observed market price.
///
/// Algorithm:
/// - Uses Brent's method for root-finding (robust, no derivatives needed)
/// - Each iteration solves American option PDE for candidate volatility
/// - Nested approach: serial Brent iterations, parallel PDE solve per iteration
///
/// Performance:
/// - Single IV solve: ~10-20ms (dominated by PDE solves in Brent iterations)
/// - Each Brent iteration: ~1-2ms for PDE solve
/// - Typical convergence: 5-10 iterations
///
/// @tparam MemSpace Kokkos memory space (HostSpace for CPU, CudaSpace for GPU)
template <typename MemSpace>
class IVSolverFDM {
public:
    /// Construct solver with configuration
    explicit IVSolverFDM(const IVSolverFDMConfig& config = IVSolverFDMConfig{})
        : config_(config) {}

    /// Solve for implied volatility (single query)
    ///
    /// Uses Brent's method to find the volatility that makes the
    /// American option's theoretical price match the market price.
    ///
    /// @param query Option specification and market price
    /// @return Result with implied_vol, iterations, convergence status
    [[nodiscard]] std::expected<IVResultFDM, IVResultCode> solve(const IVQuery& query) {
        // Validate inputs
        if (query.spot <= 0.0 || query.strike <= 0.0 || query.maturity <= 0.0) {
            return std::unexpected(IVResultCode::InvalidParams);
        }
        if (query.market_price <= 0.0 || !std::isfinite(query.market_price)) {
            return std::unexpected(IVResultCode::InvalidParams);
        }

        // Check arbitrage bounds
        double intrinsic = (query.type == OptionType::Put)
            ? std::max(0.0, query.strike - query.spot)
            : std::max(0.0, query.spot - query.strike);
        if (query.market_price < intrinsic - 1e-6) {
            return std::unexpected(IVResultCode::ArbitrageViolation);
        }

        // Adaptive volatility bounds based on time value
        double time_value = query.market_price - intrinsic;
        double time_value_ratio = time_value / query.market_price;

        double vol_upper;
        if (time_value_ratio > 0.5) {
            vol_upper = 3.0;  // High time value: ATM/OTM
        } else if (time_value_ratio > 0.2) {
            vol_upper = 2.0;  // Moderate time value
        } else {
            vol_upper = 1.5;  // Low time value: deep ITM
        }

        double vol_lower = config_.sigma_min;

        // Create objective function: f(sigma) = Price(sigma) - Market_Price
        auto objective = [this, &query](double sigma) -> double {
            return this->objective_function(query, sigma);
        };

        // Configure Brent solver
        RootFindingConfig root_config{
            .max_iter = config_.max_iterations,
            .tolerance = config_.tolerance,
            .brent_tol_abs = config_.tolerance
        };

        // Solve for root
        auto brent_result = brent_find_root(objective, vol_lower, vol_upper, root_config);

        if (!brent_result.has_value()) {
            // Map root-finding error to IV error code
            IVResultCode code;
            switch (brent_result.error().code) {
                case RootFindingErrorCode::InvalidBracket:
                    code = IVResultCode::BracketingFailed;
                    break;
                case RootFindingErrorCode::MaxIterationsExceeded:
                    code = IVResultCode::MaxIterationsExceeded;
                    break;
                case RootFindingErrorCode::NumericalInstability:
                case RootFindingErrorCode::NoProgress:
                    code = IVResultCode::NumericalInstability;
                    break;
                default:
                    code = IVResultCode::NumericalInstability;
                    break;
            }

            return std::unexpected(code);
        }

        // Success
        return IVResultFDM{
            .implied_vol = brent_result->root,
            .iterations = brent_result->iterations,
            .final_error = brent_result->final_error,
            .converged = true,
            .code = IVResultCode::Success
        };
    }

    /// Solve batch of IV queries
    ///
    /// Since Brent's method requires sequential evaluations, batch IV solving
    /// cannot parallelize at the IV level. However, each PDE solve benefits
    /// from Kokkos parallelism.
    ///
    /// @param queries View of IV queries
    /// @return View of results (one per query)
    [[nodiscard]] Kokkos::View<IVResultFDM*, MemSpace>
    solve_batch(const Kokkos::View<IVQuery*, MemSpace>& queries) {
        const size_t n_queries = queries.extent(0);
        Kokkos::View<IVResultFDM*, MemSpace> results("iv_results", n_queries);

        // Create host mirrors for processing
        auto queries_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, queries);
        auto results_h = Kokkos::create_mirror_view(results);

        // Serial loop over queries (Brent is inherently serial)
        // Each query's PDE solve runs in parallel on device
        for (size_t i = 0; i < n_queries; ++i) {
            auto result = solve(queries_h(i));
            if (result.has_value()) {
                results_h(i) = result.value();
            } else {
                results_h(i) = IVResultFDM{
                    .implied_vol = 0.0,
                    .iterations = 0,
                    .final_error = std::numeric_limits<double>::infinity(),
                    .converged = false,
                    .code = result.error()
                };
            }
        }

        // Copy results back to device
        Kokkos::deep_copy(results, results_h);
        return results;
    }

private:
    IVSolverFDMConfig config_;

    /// Objective function for root-finding: f(sigma) = Price(sigma) - Market_Price
    ///
    /// This function is called by Brent's method during each iteration.
    /// It prices the American option using the PDE solver with the given
    /// candidate volatility.
    ///
    /// @param query Option specification and market price
    /// @param volatility Candidate volatility
    /// @return Difference between theoretical price and market price
    double objective_function(const IVQuery& query, double volatility) const {
        // Create pricing parameters with candidate volatility
        PricingParams params{
            .strike = query.strike,
            .spot = query.spot,
            .maturity = query.maturity,
            .volatility = volatility,
            .rate = query.rate,
            .dividend_yield = query.dividend_yield,
            .type = query.type
        };

        // Solve American option PDE
        AmericanOptionSolver<MemSpace> solver(params, config_.n_space, config_.n_time);
        auto result = solver.solve();

        if (!result.has_value()) {
            // PDE solver failed - return NaN to signal error to Brent
            return std::numeric_limits<double>::quiet_NaN();
        }

        // Return difference: Price(sigma) - Market_Price
        return result->price - query.market_price;
    }
};

}  // namespace mango::kokkos
