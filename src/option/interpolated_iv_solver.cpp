// SPDX-License-Identifier: MIT
/**
 * @file interpolated_iv_solver.cpp
 * @brief Explicit template instantiations + factory implementation
 *
 * The solver template is in the header (interpolated_iv_solver.hpp).
 * This file provides explicit instantiations for common surface types
 * and the factory implementation (make_interpolated_iv_solver).
 */

#include "mango/option/interpolated_iv_solver.hpp"
#include "mango/option/table/adaptive_grid_builder.hpp"
#include "mango/option/table/price_table_builder.hpp"
#include "mango/option/table/segmented_price_table_builder.hpp"
#include "mango/option/table/spliced_surface_builder.hpp"
#include <algorithm>
#include <cmath>

namespace mango {

// =====================================================================
// Explicit template instantiations
// =====================================================================

template class InterpolatedIVSolver<AmericanPriceSurface>;
template class InterpolatedIVSolver<MultiKRefSurfaceWrapper<>>;
template class InterpolatedIVSolver<StrikeSurfaceWrapper<>>;

// =====================================================================
// Factory internals
// =====================================================================

namespace {

std::expected<std::vector<double>, ValidationError>
to_log_moneyness(const std::vector<double>& moneyness) {
    std::vector<double> log_m;
    log_m.reserve(moneyness.size());
    for (double m : moneyness) {
        if (m <= 0.0 || !std::isfinite(m)) {
            return std::unexpected(ValidationError{
                ValidationErrorCode::InvalidBounds, m});
        }
        log_m.push_back(std::log(m));
    }
    return log_m;
}

/// Build a MultiKRefSurface<> for manual grid path
std::expected<MultiKRefSurface<>, PriceTableError> build_multi_kref_manual(
    double spot,
    OptionType option_type,
    const DividendSpec& dividends,
    const IVGrid& log_grid,
    double maturity,
    const MultiKRefConfig& kref_config)
{
    // Generate K_refs if not provided
    std::vector<double> K_refs = kref_config.K_refs;
    if (K_refs.empty()) {
        // Auto-generate K_refs around spot
        K_refs.reserve(static_cast<size_t>(kref_config.K_ref_count));
        double log_low = std::log(spot) - kref_config.K_ref_span;
        double log_high = std::log(spot) + kref_config.K_ref_span;
        for (int i = 0; i < kref_config.K_ref_count; ++i) {
            double t = static_cast<double>(i) / (kref_config.K_ref_count - 1);
            K_refs.push_back(std::exp(log_low + t * (log_high - log_low)));
        }
    }

    std::vector<MultiKRefEntry> entries;
    entries.reserve(K_refs.size());

    for (double K_ref : K_refs) {
        // Build SegmentedSurface for this K_ref
        SegmentedPriceTableBuilder::Config seg_config{
            .K_ref = K_ref,
            .option_type = option_type,
            .dividends = dividends,
            .grid = log_grid,
            .maturity = maturity,
            .tau_points_per_segment = 5,
        };

        auto surface = SegmentedPriceTableBuilder::build(seg_config);
        if (!surface.has_value()) {
            return std::unexpected(surface.error());
        }

        entries.push_back(MultiKRefEntry{
            .K_ref = K_ref,
            .surface = std::move(*surface),
        });
    }

    return build_multi_kref_surface(std::move(entries));
}

/// Extract bounds from an IVGrid for surface wrapper construction
struct GridBounds {
    double m_min, m_max;
    double sigma_min, sigma_max;
    double rate_min, rate_max;
};

GridBounds extract_bounds(const IVGrid& grid) {
    auto minmax_m = std::minmax_element(grid.moneyness.begin(), grid.moneyness.end());
    auto minmax_v = std::minmax_element(grid.vol.begin(), grid.vol.end());
    auto minmax_r = std::minmax_element(grid.rate.begin(), grid.rate.end());
    // Moneyness must be positive for log conversion
    if (*minmax_m.first <= 0.0) {
        return {};  // Zero-initialized; caller validates non-empty bounds
    }
    return {
        .m_min = std::log(*minmax_m.first), .m_max = std::log(*minmax_m.second),
        .sigma_min = *minmax_v.first, .sigma_max = *minmax_v.second,
        .rate_min = *minmax_r.first, .rate_max = *minmax_r.second,
    };
}

}  // anonymous namespace

// =====================================================================
// AnyIVSolver: type-erased wrapper
// =====================================================================

AnyIVSolver::AnyIVSolver(InterpolatedIVSolver<AmericanPriceSurface> solver)
    : solver_(std::move(solver))
{}

AnyIVSolver::AnyIVSolver(InterpolatedIVSolver<MultiKRefSurfaceWrapper<>> solver)
    : solver_(std::move(solver))
{}

AnyIVSolver::AnyIVSolver(InterpolatedIVSolver<StrikeSurfaceWrapper<>> solver)
    : solver_(std::move(solver))
{}

std::expected<IVSuccess, IVError> AnyIVSolver::solve(const IVQuery& query) const {
    return std::visit([&](const auto& solver) {
        return solver.solve(query);
    }, solver_);
}

BatchIVResult AnyIVSolver::solve_batch(const std::vector<IVQuery>& queries) const {
    return std::visit([&](const auto& solver) {
        return solver.solve_batch(queries);
    }, solver_);
}

// ---------------------------------------------------------------------------
// Helper: wrap surface into AnyIVSolver
// ---------------------------------------------------------------------------

static std::expected<AnyIVSolver, ValidationError>
wrap_surface(std::shared_ptr<const PriceTableSurface<4>> surface,
             OptionType option_type,
             const InterpolatedIVSolverConfig& solver_config) {
    auto aps = AmericanPriceSurface::create(surface, option_type);
    if (!aps.has_value()) {
        return std::unexpected(aps.error());
    }

    auto solver = InterpolatedIVSolver<AmericanPriceSurface>::create(
        std::move(*aps), solver_config);
    if (!solver.has_value()) {
        return std::unexpected(ValidationError{
            ValidationErrorCode::InvalidGridSize, 0.0});
    }

    return AnyIVSolver(std::move(*solver));
}

// ---------------------------------------------------------------------------
// Factory: standard path with adaptive grid refinement
// ---------------------------------------------------------------------------

static std::expected<AnyIVSolver, ValidationError>
build_standard_adaptive(const IVSolverFactoryConfig& config,
                        const StandardIVPath& path) {
    OptionGrid chain;
    chain.spot = config.spot;
    chain.dividend_yield = config.dividend_yield;

    chain.strikes.reserve(config.grid.moneyness.size());
    for (double m : config.grid.moneyness) {
        chain.strikes.push_back(config.spot / m);
    }
    chain.maturities = path.maturity_grid;
    chain.implied_vols = config.grid.vol;
    chain.rates = config.grid.rate;

    // Use auto-estimated grid with High profile for better accuracy
    // (Fixed 101x500 grid was too coarse, causing ~600 bps IV errors)
    GridAccuracyParams accuracy = make_grid_accuracy(GridAccuracyProfile::High);

    AdaptiveGridBuilder builder(*config.adaptive);
    auto result = builder.build(chain, accuracy, config.option_type);

    if (!result.has_value()) {
        return std::unexpected(ValidationError{
            ValidationErrorCode::InvalidGridSize, 0.0});
    }

    return wrap_surface(result->surface, config.option_type, config.solver_config);
}

// ---------------------------------------------------------------------------
// Factory: standard path (dispatch manual vs adaptive)
// ---------------------------------------------------------------------------

static std::expected<AnyIVSolver, ValidationError>
build_standard(const IVSolverFactoryConfig& config, const StandardIVPath& path) {
    if (config.adaptive.has_value()) {
        return build_standard_adaptive(config, path);
    }

    // Manual grid: build price table directly
    auto log_m = to_log_moneyness(config.grid.moneyness);
    if (!log_m.has_value()) {
        return std::unexpected(log_m.error());
    }
    auto setup = PriceTableBuilder<4>::from_vectors(
        std::move(*log_m), path.maturity_grid, config.grid.vol, config.grid.rate,
        config.spot, GridAccuracyParams{}, config.option_type,
        config.dividend_yield);
    if (!setup.has_value()) {
        return std::unexpected(ValidationError{
            ValidationErrorCode::InvalidGridSize, 0.0});
    }

    auto& [builder, axes] = *setup;
    auto table_result = builder.build(axes);
    if (!table_result.has_value()) {
        return std::unexpected(ValidationError{
            ValidationErrorCode::InvalidGridSize, 0.0});
    }

    return wrap_surface(table_result->surface, config.option_type,
                        config.solver_config);
}

// ---------------------------------------------------------------------------
// Factory: segmented path helpers
// ---------------------------------------------------------------------------

/// Wrap a StrikeSurface into AnyIVSolver
static std::expected<AnyIVSolver, ValidationError>
wrap_strike_surface(StrikeSurface<> surface,
                    const GridBounds& b, double maturity,
                    OptionType option_type, double dividend_yield,
                    const InterpolatedIVSolverConfig& solver_config) {
    StrikeSurfaceWrapper<>::Bounds bounds{
        .m_min = b.m_min, .m_max = b.m_max,
        .tau_min = 0.0, .tau_max = maturity,
        .sigma_min = b.sigma_min, .sigma_max = b.sigma_max,
        .rate_min = b.rate_min, .rate_max = b.rate_max,
    };

    auto wrapper = StrikeSurfaceWrapper<>(
        std::move(surface), bounds, option_type, dividend_yield);

    auto solver = InterpolatedIVSolver<StrikeSurfaceWrapper<>>::create(
        std::move(wrapper), solver_config);
    if (!solver.has_value()) {
        return std::unexpected(ValidationError{
            ValidationErrorCode::InvalidGridSize, 0.0});
    }
    return AnyIVSolver(std::move(*solver));
}

/// Wrap a MultiKRefSurface into AnyIVSolver
static std::expected<AnyIVSolver, ValidationError>
wrap_multi_kref_surface(MultiKRefSurface<> surface,
                        const GridBounds& b, double maturity,
                        OptionType option_type, double dividend_yield,
                        const InterpolatedIVSolverConfig& solver_config) {
    MultiKRefSurfaceWrapper<>::Bounds bounds{
        .m_min = b.m_min, .m_max = b.m_max,
        .tau_min = 0.0, .tau_max = maturity,
        .sigma_min = b.sigma_min, .sigma_max = b.sigma_max,
        .rate_min = b.rate_min, .rate_max = b.rate_max,
    };

    auto wrapper = MultiKRefSurfaceWrapper<>(
        std::move(surface), bounds, option_type, dividend_yield);

    auto solver = InterpolatedIVSolver<MultiKRefSurfaceWrapper<>>::create(
        std::move(wrapper), solver_config);
    if (!solver.has_value()) {
        return std::unexpected(ValidationError{
            ValidationErrorCode::InvalidGridSize, 0.0});
    }
    return AnyIVSolver(std::move(*solver));
}

/// Build per-strike surface manually (no adaptive grid)
static std::expected<StrikeSurface<>, ValidationError>
build_manual_strike_surface(const IVSolverFactoryConfig& config,
                            const SegmentedIVPath& path,
                            const std::vector<double>& strikes,
                            const IVGrid& log_grid) {
    DividendSpec dividends{
        .dividend_yield = config.dividend_yield,
        .discrete_dividends = path.discrete_dividends
    };

    std::vector<StrikeEntry> entries;
    entries.reserve(strikes.size());
    for (double strike : strikes) {
        SegmentedPriceTableBuilder::Config seg_config{
            .K_ref = strike,
            .option_type = config.option_type,
            .dividends = dividends,
            .grid = log_grid,
            .maturity = path.maturity,
            .tau_points_per_segment = 5,
        };

        auto surface = SegmentedPriceTableBuilder::build(seg_config);
        if (!surface.has_value()) {
            return std::unexpected(ValidationError{
                ValidationErrorCode::InvalidGridSize, 0.0});
        }

        entries.push_back(StrikeEntry{
            .strike = strike,
            .surface = std::move(*surface),
        });
    }

    auto surface = build_strike_surface(std::move(entries), /*use_nearest=*/true);
    if (!surface.has_value()) {
        return std::unexpected(ValidationError{
            ValidationErrorCode::InvalidGridSize, 0.0});
    }
    return std::move(*surface);
}

// ---------------------------------------------------------------------------
// Factory: segmented path (discrete dividends)
// ---------------------------------------------------------------------------

static std::expected<AnyIVSolver, ValidationError>
build_segmented(const IVSolverFactoryConfig& config, const SegmentedIVPath& path) {
    MultiKRefConfig kref_config = path.kref_config;
    const auto& grid = config.grid;
    const bool use_per_strike = !path.strike_grid.empty();
    if (use_per_strike) {
        kref_config.K_refs = path.strike_grid;
    }

    auto log_m = to_log_moneyness(grid.moneyness);
    if (!log_m.has_value()) {
        return std::unexpected(log_m.error());
    }
    IVGrid log_grid = grid;
    log_grid.moneyness = std::move(*log_m);

    auto b = extract_bounds(grid);

    if (config.adaptive.has_value()) {
        AdaptiveGridBuilder builder(*config.adaptive);
        SegmentedAdaptiveConfig seg_config{
            .spot = config.spot,
            .option_type = config.option_type,
            .dividend_yield = config.dividend_yield,
            .discrete_dividends = path.discrete_dividends,
            .maturity = path.maturity,
            .kref_config = kref_config,
        };

        if (use_per_strike) {
            auto result = builder.build_segmented_strike(
                seg_config, path.strike_grid,
                {log_grid.moneyness, log_grid.vol, log_grid.rate});
            if (!result.has_value()) {
                return std::unexpected(ValidationError{
                    ValidationErrorCode::InvalidGridSize, 0.0});
            }
            return wrap_strike_surface(std::move(result->surface),
                b, path.maturity, config.option_type,
                config.dividend_yield, config.solver_config);
        }

        auto result = builder.build_segmented(
            seg_config, {log_grid.moneyness, log_grid.vol, log_grid.rate});
        if (!result.has_value()) {
            return std::unexpected(ValidationError{
                ValidationErrorCode::InvalidGridSize, 0.0});
        }
        return wrap_multi_kref_surface(std::move(result->surface),
            b, path.maturity, config.option_type,
            config.dividend_yield, config.solver_config);
    }

    // Manual grid path
    if (use_per_strike) {
        auto surface = build_manual_strike_surface(
            config, path, kref_config.K_refs, log_grid);
        if (!surface.has_value()) {
            return std::unexpected(surface.error());
        }
        return wrap_strike_surface(std::move(*surface),
            b, path.maturity, config.option_type,
            config.dividend_yield, config.solver_config);
    }

    DividendSpec dividends{
        .dividend_yield = config.dividend_yield,
        .discrete_dividends = path.discrete_dividends
    };

    auto surface = build_multi_kref_manual(
        config.spot, config.option_type, dividends,
        log_grid, path.maturity, kref_config);
    if (!surface.has_value()) {
        return std::unexpected(ValidationError{
            ValidationErrorCode::InvalidGridSize, 0.0});
    }
    return wrap_multi_kref_surface(std::move(*surface),
        b, path.maturity, config.option_type,
        config.dividend_yield, config.solver_config);
}

// ---------------------------------------------------------------------------
// Public factory
// ---------------------------------------------------------------------------

std::expected<AnyIVSolver, ValidationError> make_interpolated_iv_solver(const IVSolverFactoryConfig& config) {
    return std::visit([&](const auto& path) -> std::expected<AnyIVSolver, ValidationError> {
        using T = std::decay_t<decltype(path)>;
        if constexpr (std::is_same_v<T, StandardIVPath>) {
            return build_standard(config, path);
        } else {
            return build_segmented(config, path);
        }
    }, config.path);
}

}  // namespace mango
