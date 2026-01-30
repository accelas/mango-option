// SPDX-License-Identifier: MIT
#pragma once

#include "src/option/option_spec.hpp"
#include "src/option/table/price_table_surface.hpp"
#include "src/option/iv_solver_interpolated.hpp"
#include <expected>
#include <filesystem>
#include <memory>
#include <string>

namespace mango::simple {

using mango::OptionType;

/// Configuration for building a price table
struct PriceTableConfig {
    OptionType type = OptionType::PUT;
    double strike_ref = 100.0;
    double dividend_yield = 0.0;
    size_t n_moneyness = 21;
    size_t n_maturity = 15;
    size_t n_volatility = 15;
    size_t n_rate = 7;
    double moneyness_min = 0.5;
    double moneyness_max = 2.0;
    double maturity_min = 0.01;
    double maturity_max = 2.0;
    double vol_min = 0.05;
    double vol_max = 1.0;
    double rate_min = -0.01;
    double rate_max = 0.10;
};

/// Pre-computed price table for fast option pricing and IV calculation
///
/// Wraps PriceTableSurface<4> with convenience methods for
/// querying, persistence (Arrow IPC), and IV solver creation.
class PriceTable {
public:
    /// Query the price surface
    ///
    /// @param moneyness S/K ratio
    /// @param tau Time to maturity (years)
    /// @param sigma Volatility
    /// @param rate Risk-free rate
    /// @return Interpolated option price (normalized to strike_ref)
    double value(double moneyness, double tau, double sigma, double rate) const;

    /// Save to Arrow IPC file
    std::expected<void, std::string> save(const std::filesystem::path& path) const;

    /// Access the underlying surface
    std::shared_ptr<const PriceTableSurface<4>> surface() const { return surface_; }

    /// Option type
    OptionType type() const { return type_; }

    /// Reference strike
    double strike_ref() const { return strike_ref_; }

private:
    friend std::expected<PriceTable, std::string> build_price_table(const PriceTableConfig&);
    friend std::expected<PriceTable, std::string> load_price_table(const std::filesystem::path&);

    std::shared_ptr<const PriceTableSurface<4>> surface_;
    OptionType type_ = OptionType::PUT;
    double strike_ref_ = 100.0;
};

/// Build a price table by solving the PDE across a parameter grid
///
/// @param config Table configuration (grid ranges, resolution)
/// @return PriceTable or error string
std::expected<PriceTable, std::string> build_price_table(const PriceTableConfig& config = {});

/// Load a previously saved price table from Arrow IPC file
///
/// @param path Path to Arrow IPC file
/// @return PriceTable or error string
std::expected<PriceTable, std::string> load_price_table(const std::filesystem::path& path);

/// Create an interpolation-based IV solver from a price table
///
/// @param table Pre-computed price table
/// @return IVSolverInterpolated or error string
std::expected<IVSolverInterpolated, std::string> make_iv_solver(const PriceTable& table);

}  // namespace mango::simple
