// SPDX-License-Identifier: MIT
#include "src/option/iv_solver_factory.hpp"
#include "src/option/table/price_table_builder.hpp"
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
// Factory: standard path (no discrete dividends)
// ---------------------------------------------------------------------------

static std::expected<AnyIVSolver, ValidationError>
build_standard(const IVSolverFactoryConfig& config, const StandardIVPath& path) {
    // Use spot as K_ref (ATM reference strike)
    double K_ref = config.spot;

    auto setup = PriceTableBuilder<4>::from_vectors(
        config.moneyness_grid,
        path.maturity_grid,
        config.vol_grid,
        config.rate_grid,
        K_ref,
        GridAccuracyParams{},
        config.option_type,
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

    auto aps = AmericanPriceSurface::create(
        table_result->surface, config.option_type);
    if (!aps.has_value()) {
        return std::unexpected(aps.error());
    }

    auto solver = InterpolatedIVSolver<AmericanPriceSurface>::create(
        std::move(*aps), config.solver_config);
    if (!solver.has_value()) {
        return std::unexpected(ValidationError{
            ValidationErrorCode::InvalidGridSize, 0.0});
    }

    return AnyIVSolver(std::move(*solver));
}

// ---------------------------------------------------------------------------
// Factory: segmented path (discrete dividends)
// ---------------------------------------------------------------------------

static std::expected<AnyIVSolver, ValidationError>
build_segmented(const IVSolverFactoryConfig& config, const SegmentedIVPath& path) {
    SegmentedMultiKRefBuilder::Config seg_config{
        .spot = config.spot,
        .option_type = config.option_type,
        .dividends = {.dividend_yield = config.dividend_yield, .discrete_dividends = path.discrete_dividends},
        .moneyness_grid = config.moneyness_grid,
        .maturity = path.maturity,
        .vol_grid = config.vol_grid,
        .rate_grid = config.rate_grid,
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
