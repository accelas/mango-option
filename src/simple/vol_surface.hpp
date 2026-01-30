// SPDX-License-Identifier: MIT
/**
 * @file vol_surface.hpp
 * @brief Volatility surface types and computation
 */

#pragma once

#include "src/simple/option_chain.hpp"
#include "src/option/iv_solver_interpolated.hpp"
#include "src/option/iv_solver_fdm.hpp"
#include <expected>
#include <memory>
#include <ranges>
#include <vector>

namespace mango::simple {

/// Single point on the volatility smile
struct VolatilitySmile {
    Timestamp expiry{""};
    double tau = 0.0;  // Time to expiry in years
    Price spot{0.0};

    struct Point {
        OptionType type;          // CALL or PUT
        Price strike{0.0};
        double moneyness = 0.0;   // log(K/S)
        std::optional<double> iv_bid;
        std::optional<double> iv_ask;
        std::optional<double> iv_mid;
        std::optional<double> iv_last;
    };

    std::vector<Point> points;  // All points (calls and puts together)

    /// Get filtered view of calls only
    [[nodiscard]] auto calls() const {
        return points | std::views::filter([](const Point& pt) {
            return pt.type == OptionType::CALL;
        });
    }

    /// Get filtered view of puts only
    [[nodiscard]] auto puts() const {
        return points | std::views::filter([](const Point& pt) {
            return pt.type == OptionType::PUT;
        });
    }
};

/// Complete volatility surface
struct VolatilitySurface {
    std::string symbol;
    Timestamp quote_time{""};
    Price spot{0.0};

    std::vector<VolatilitySmile> smiles;

    /// Interpolate IV at arbitrary (strike, tau)
    std::optional<double> iv_at(double strike, double tau) const;
};

/// Error during surface computation
struct ComputeError {
    std::string message;
    size_t failed_count = 0;
};

/// Configuration for IV computation
struct IVComputeConfig {
    enum class Method {
        Interpolated,  // Fast: use precomputed tables (~30µs)
        FDM            // Accurate: solve PDE per option (~143ms)
    };

    Method method = Method::Interpolated;
    double tolerance = 1e-6;
    int max_iterations = 50;
};

/// Price source for IV calculation
enum class PriceSource {
    Bid,
    Ask,
    Mid,
    Last
};

/// Compute volatility surface from option chain
///
/// @param chain Option chain with quotes
/// @param ctx Market context (rate, valuation time)
/// @param solver Interpolated IV solver (required for Method::Interpolated)
/// @param config Computation configuration
/// @param price_source Which price to use for IV
/// @return Volatility surface or error
std::expected<VolatilitySurface, ComputeError> compute_vol_surface(
    const OptionChain& chain,
    const MarketContext& ctx,
    const mango::IVSolverInterpolated* solver = nullptr,
    const IVComputeConfig& config = {},
    PriceSource price_source = PriceSource::Mid);

}  // namespace mango::simple
