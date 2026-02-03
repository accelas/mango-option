// SPDX-License-Identifier: MIT
#include "mango/option/iv_solver_factory.hpp"
#include "mango/option/table/adaptive_grid_builder.hpp"
#include "mango/option/table/price_table_builder.hpp"
#include <type_traits>

namespace mango {

// ---------------------------------------------------------------------------
// AnyIVSolver: type-erased wrapper
// ---------------------------------------------------------------------------

AnyIVSolver::AnyIVSolver(InterpolatedIVSolver<AmericanPriceSurface> solver)
    : solver_(std::move(solver))
{}

AnyIVSolver::AnyIVSolver(InterpolatedIVSolver<SegmentedMultiKRefSurface> solver)
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
                        const StandardIVPath& path,
                        const AdaptiveGrid& grid) {
    OptionGrid chain;
    chain.spot = config.spot;
    chain.dividend_yield = config.dividend_yield;

    chain.strikes.reserve(grid.moneyness.size());
    for (double m : grid.moneyness) {
        chain.strikes.push_back(config.spot / m);
    }
    chain.maturities = path.maturity_grid;
    chain.implied_vols = grid.vol;
    chain.rates = grid.rate;

    auto grid_spec = GridSpec<double>::sinh_spaced(-3.0, 3.0, 101, 2.0);
    if (!grid_spec.has_value()) {
        return std::unexpected(ValidationError{
            ValidationErrorCode::InvalidGridSize, 0.0});
    }

    AdaptiveGridBuilder builder(grid.params);
    auto result = builder.build(chain, *grid_spec, 500, config.option_type);

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
    return std::visit([&](const auto& grid) -> std::expected<AnyIVSolver, ValidationError> {
        using G = std::decay_t<decltype(grid)>;
        if constexpr (std::is_same_v<G, AdaptiveGrid>) {
            return build_standard_adaptive(config, path, grid);
        }

        // Manual grid: build price table directly
        auto setup = PriceTableBuilder<4>::from_vectors(
            grid.moneyness, path.maturity_grid, grid.vol, grid.rate,
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
    }, config.grid);
}

// ---------------------------------------------------------------------------
// Factory: segmented path (discrete dividends)
// ---------------------------------------------------------------------------

static std::expected<AnyIVSolver, ValidationError>
build_segmented(const IVSolverFactoryConfig& config, const SegmentedIVPath& path) {
    return std::visit([&](const auto& grid) -> std::expected<AnyIVSolver, ValidationError> {
        using G = std::decay_t<decltype(grid)>;

        if constexpr (std::is_same_v<G, AdaptiveGrid>) {
            // Adaptive grid for segmented path
            AdaptiveGridBuilder builder(grid.params);
            SegmentedAdaptiveConfig seg_config{
                .spot = config.spot,
                .option_type = config.option_type,
                .dividend_yield = config.dividend_yield,
                .discrete_dividends = path.discrete_dividends,
                .maturity = path.maturity,
                .kref_config = path.kref_config,
            };
            auto surface = builder.build_segmented(
                seg_config, grid.moneyness, grid.vol, grid.rate);
            if (!surface.has_value()) {
                return std::unexpected(ValidationError{
                    ValidationErrorCode::InvalidGridSize, 0.0});
            }

            auto solver = InterpolatedIVSolver<SegmentedMultiKRefSurface>::create(
                std::move(*surface), config.solver_config);
            if (!solver.has_value()) {
                return std::unexpected(ValidationError{
                    ValidationErrorCode::InvalidGridSize, 0.0});
            }
            return AnyIVSolver(std::move(*solver));
        }

        // Manual grid: existing path (unchanged)
        SegmentedMultiKRefBuilder::Config seg_config{
            .spot = config.spot,
            .option_type = config.option_type,
            .dividends = {.dividend_yield = config.dividend_yield, .discrete_dividends = path.discrete_dividends},
            .moneyness_grid = grid.moneyness,
            .maturity = path.maturity,
            .vol_grid = grid.vol,
            .rate_grid = grid.rate,
            .kref_config = path.kref_config,
        };

        auto surface = SegmentedMultiKRefBuilder::build(seg_config);
        if (!surface.has_value()) {
            return std::unexpected(surface.error());
        }

        auto solver = InterpolatedIVSolver<SegmentedMultiKRefSurface>::create(
            std::move(*surface), config.solver_config);
        if (!solver.has_value()) {
            return std::unexpected(ValidationError{
                ValidationErrorCode::InvalidGridSize, 0.0});
        }

        return AnyIVSolver(std::move(*solver));
    }, config.grid);
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
