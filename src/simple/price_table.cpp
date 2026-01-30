// SPDX-License-Identifier: MIT
#include "src/simple/price_table.hpp"
#include "src/option/table/price_table_builder.hpp"
#include "src/option/table/price_table_workspace.hpp"
#include "src/pde/core/grid.hpp"
#include <cmath>
#include <sstream>

namespace mango::simple {

// ---------------------------------------------------------------------------
// PriceTable methods
// ---------------------------------------------------------------------------

double PriceTable::value(double moneyness, double tau, double sigma, double rate) const {
    return surface_->value({moneyness, tau, sigma, rate});
}

std::expected<void, std::string> PriceTable::save(const std::filesystem::path& path) const {
    // Extract data from surface for workspace creation
    const auto& axes = surface_->axes();      // log-moneyness internally
    const auto& meta = surface_->metadata();
    const auto& coeffs = surface_->coefficients();

    auto ws_result = PriceTableWorkspace::create(
        axes.grids[0],   // log-moneyness grid
        axes.grids[1],   // maturity grid
        axes.grids[2],   // volatility grid
        axes.grids[3],   // rate grid
        coeffs,
        meta.K_ref,
        meta.dividend_yield,
        meta.m_min,
        meta.m_max);

    if (!ws_result.has_value()) {
        return std::unexpected(ws_result.error());
    }

    uint8_t opt_type = (type_ == OptionType::PUT) ? 0 : 1;
    return ws_result->save(path.string(), "TABLE", opt_type);
}

// ---------------------------------------------------------------------------
// Free functions
// ---------------------------------------------------------------------------

namespace {

std::vector<double> linspace(double lo, double hi, size_t n) {
    std::vector<double> v(n);
    if (n == 1) {
        v[0] = (lo + hi) * 0.5;
        return v;
    }
    for (size_t i = 0; i < n; ++i) {
        v[i] = lo + static_cast<double>(i) * (hi - lo) / static_cast<double>(n - 1);
    }
    return v;
}

}  // anonymous namespace

std::expected<PriceTable, std::string> build_price_table(const PriceTableConfig& config) {
    auto moneyness_grid = linspace(config.moneyness_min, config.moneyness_max, config.n_moneyness);
    auto maturity_grid  = linspace(config.maturity_min,  config.maturity_max,  config.n_maturity);
    auto vol_grid       = linspace(config.vol_min,       config.vol_max,       config.n_volatility);
    auto rate_grid      = linspace(config.rate_min,      config.rate_max,      config.n_rate);

    // PDE spatial grid
    auto grid_spec_result = GridSpec<double>::uniform(-3.0, 3.0, 101);
    if (!grid_spec_result.has_value()) {
        return std::unexpected("Failed to create PDE grid");
    }

    auto builder_result = PriceTableBuilder<4>::from_vectors(
        moneyness_grid, maturity_grid, vol_grid, rate_grid,
        config.strike_ref,
        grid_spec_result.value(),
        1000,   // n_time steps
        config.type,
        config.dividend_yield);

    if (!builder_result.has_value()) {
        std::ostringstream oss;
        oss << builder_result.error();
        return std::unexpected(oss.str());
    }

    auto& [builder, axes] = *builder_result;
    auto build_result = builder.build(axes);

    if (!build_result.has_value()) {
        std::ostringstream oss;
        oss << build_result.error();
        return std::unexpected(oss.str());
    }

    PriceTable table;
    table.surface_ = build_result->surface;
    table.type_ = config.type;
    table.strike_ref_ = config.strike_ref;
    return table;
}

std::expected<PriceTable, std::string> load_price_table(const std::filesystem::path& path) {
    auto ws_result = PriceTableWorkspace::load(path.string());
    if (!ws_result.has_value()) {
        return std::unexpected(
            "Failed to load price table (error code " +
            std::to_string(static_cast<int>(ws_result.error())) + ")");
    }

    auto& ws = *ws_result;

    // Reconstruct moneyness grids from log-moneyness
    std::vector<double> moneyness_grid(ws.log_moneyness().size());
    for (size_t i = 0; i < moneyness_grid.size(); ++i) {
        moneyness_grid[i] = std::exp(ws.log_moneyness()[i]);
    }

    // Build axes in moneyness space (PriceTableSurface::build transforms to log)
    PriceTableAxes<4> axes;
    axes.grids[0] = std::move(moneyness_grid);
    axes.grids[1] = std::vector<double>(ws.maturity().begin(), ws.maturity().end());
    axes.grids[2] = std::vector<double>(ws.volatility().begin(), ws.volatility().end());
    axes.grids[3] = std::vector<double>(ws.rate().begin(), ws.rate().end());
    axes.names = {"moneyness", "maturity", "volatility", "rate"};

    std::vector<double> coeffs(ws.coefficients().begin(), ws.coefficients().end());

    PriceTableMetadata meta;
    meta.K_ref = ws.K_ref();
    meta.dividend_yield = ws.dividend_yield();
    meta.m_min = ws.m_min();
    meta.m_max = ws.m_max();

    auto surface_result = PriceTableSurface<4>::build(
        std::move(axes), std::move(coeffs), std::move(meta));

    if (!surface_result.has_value()) {
        std::ostringstream oss;
        oss << surface_result.error();
        return std::unexpected(oss.str());
    }

    PriceTable table;
    table.surface_ = *surface_result;
    table.strike_ref_ = ws.K_ref();
    // We don't store option type in workspace, default to PUT
    table.type_ = OptionType::PUT;
    return table;
}

std::expected<IVSolverInterpolated, std::string> make_iv_solver(const PriceTable& table) {
    auto result = IVSolverInterpolated::create(table.surface());
    if (!result.has_value()) {
        std::ostringstream oss;
        oss << result.error();
        return std::unexpected(oss.str());
    }
    return std::move(*result);
}

}  // namespace mango::simple
