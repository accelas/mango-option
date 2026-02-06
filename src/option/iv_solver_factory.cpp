// SPDX-License-Identifier: MIT
#include "mango/option/iv_solver_factory.hpp"
#include "mango/option/table/adaptive_grid_builder.hpp"
#include "mango/option/table/price_table_builder.hpp"
#include "mango/option/table/segmented_price_table_builder.hpp"
#include "mango/option/table/spliced_surface_builder.hpp"
#include <algorithm>
#include <cmath>
#include <type_traits>

namespace mango {

namespace {

/// Build a MultiKRefSurface<> for manual grid path
std::expected<MultiKRefSurface<>, PriceTableError> build_multi_kref_manual(
    double spot,
    OptionType option_type,
    const DividendSpec& dividends,
    const ManualGrid& grid,
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
            .grid = grid,
            .maturity = maturity,
            .tau_points_per_segment = 5,
            .skip_moneyness_expansion = false,
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

}  // anonymous namespace

// ---------------------------------------------------------------------------
// AnyIVSolver: type-erased wrapper
// ---------------------------------------------------------------------------

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

    // Use auto-estimated grid with High profile for better accuracy
    // (Fixed 101×500 grid was too coarse, causing ~600 bps IV errors)
    GridAccuracyParams accuracy = make_grid_accuracy(GridAccuracyProfile::High);

    AdaptiveGridBuilder builder(grid.params);
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
    MultiKRefConfig kref_config = path.kref_config;
    return std::visit([&](const auto& grid) -> std::expected<AnyIVSolver, ValidationError> {
        using G = std::decay_t<decltype(grid)>;
        const bool use_per_strike = !path.strike_grid.empty();
        if (use_per_strike) {
            kref_config.K_refs = path.strike_grid;
        }

        if constexpr (std::is_same_v<G, AdaptiveGrid>) {
            // Adaptive grid for segmented path
            AdaptiveGridBuilder builder(grid.params);
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
                    {grid.moneyness, grid.vol, grid.rate});
                if (!result.has_value()) {
                    return std::unexpected(ValidationError{
                        ValidationErrorCode::InvalidGridSize, 0.0});
                }

                auto minmax_m = std::minmax_element(grid.moneyness.begin(), grid.moneyness.end());
                auto minmax_v = std::minmax_element(grid.vol.begin(), grid.vol.end());
                auto minmax_r = std::minmax_element(grid.rate.begin(), grid.rate.end());

                StrikeSurfaceWrapper<>::Bounds bounds{
                    .m_min = *minmax_m.first,
                    .m_max = *minmax_m.second,
                    .tau_min = 0.0,
                    .tau_max = path.maturity,
                    .sigma_min = *minmax_v.first,
                    .sigma_max = *minmax_v.second,
                    .rate_min = *minmax_r.first,
                    .rate_max = *minmax_r.second,
                };

                auto wrapper = StrikeSurfaceWrapper<>(
                    std::move(result->surface), bounds, config.option_type, config.dividend_yield);

                auto solver = InterpolatedIVSolver<StrikeSurfaceWrapper<>>::create(
                    std::move(wrapper), config.solver_config);
                if (!solver.has_value()) {
                    return std::unexpected(ValidationError{
                        ValidationErrorCode::InvalidGridSize, 0.0});
                }
                return AnyIVSolver(std::move(*solver));
            }

            auto result = builder.build_segmented(
                seg_config, {grid.moneyness, grid.vol, grid.rate});
            if (!result.has_value()) {
                return std::unexpected(ValidationError{
                    ValidationErrorCode::InvalidGridSize, 0.0});
            }

            auto minmax_m = std::minmax_element(grid.moneyness.begin(), grid.moneyness.end());
            auto minmax_v = std::minmax_element(grid.vol.begin(), grid.vol.end());
            auto minmax_r = std::minmax_element(grid.rate.begin(), grid.rate.end());

            MultiKRefSurfaceWrapper<>::Bounds bounds{
                .m_min = *minmax_m.first,
                .m_max = *minmax_m.second,
                .tau_min = 0.0,
                .tau_max = path.maturity,
                .sigma_min = *minmax_v.first,
                .sigma_max = *minmax_v.second,
                .rate_min = *minmax_r.first,
                .rate_max = *minmax_r.second,
            };

            auto wrapper = MultiKRefSurfaceWrapper<>(
                std::move(result->surface), bounds, config.option_type, config.dividend_yield);

            auto solver = InterpolatedIVSolver<MultiKRefSurfaceWrapper<>>::create(
                std::move(wrapper), config.solver_config);
            if (!solver.has_value()) {
                return std::unexpected(ValidationError{
                    ValidationErrorCode::InvalidGridSize, 0.0});
            }
            return AnyIVSolver(std::move(*solver));
        } else if constexpr (std::is_same_v<G, ManualGrid>) {
            // Manual grid: build MultiKRefSurface using new spliced surface types
            DividendSpec dividends{
            .dividend_yield = config.dividend_yield,
            .discrete_dividends = path.discrete_dividends
        };

        if (use_per_strike) {
            std::vector<StrikeEntry> entries;
            entries.reserve(kref_config.K_refs.size());
            for (double strike : kref_config.K_refs) {
                SegmentedPriceTableBuilder::Config seg_config{
                    .K_ref = strike,
                    .option_type = config.option_type,
                    .dividends = dividends,
                    .grid = grid,
                    .maturity = path.maturity,
                    .tau_points_per_segment = 5,
                    .skip_moneyness_expansion = false,
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

            auto minmax_m = std::minmax_element(grid.moneyness.begin(), grid.moneyness.end());
            auto minmax_v = std::minmax_element(grid.vol.begin(), grid.vol.end());
            auto minmax_r = std::minmax_element(grid.rate.begin(), grid.rate.end());

            StrikeSurfaceWrapper<>::Bounds bounds{
                .m_min = *minmax_m.first,
                .m_max = *minmax_m.second,
                .tau_min = 0.0,
                .tau_max = path.maturity,
                .sigma_min = *minmax_v.first,
                .sigma_max = *minmax_v.second,
                .rate_min = *minmax_r.first,
                .rate_max = *minmax_r.second,
            };

            auto wrapper = StrikeSurfaceWrapper<>(
                std::move(*surface), bounds, config.option_type, config.dividend_yield);

            auto solver = InterpolatedIVSolver<StrikeSurfaceWrapper<>>::create(
                std::move(wrapper), config.solver_config);
            if (!solver.has_value()) {
                return std::unexpected(ValidationError{
                    ValidationErrorCode::InvalidGridSize, 0.0});
            }

            return AnyIVSolver(std::move(*solver));
        }

        auto surface = build_multi_kref_manual(
            config.spot, config.option_type, dividends,
            grid, path.maturity, kref_config);
        if (!surface.has_value()) {
            return std::unexpected(ValidationError{
                ValidationErrorCode::InvalidGridSize, 0.0});
        }

        auto minmax_m = std::minmax_element(grid.moneyness.begin(), grid.moneyness.end());
        auto minmax_v = std::minmax_element(grid.vol.begin(), grid.vol.end());
        auto minmax_r = std::minmax_element(grid.rate.begin(), grid.rate.end());

        MultiKRefSurfaceWrapper<>::Bounds bounds{
            .m_min = *minmax_m.first,
            .m_max = *minmax_m.second,
            .tau_min = 0.0,
            .tau_max = path.maturity,
            .sigma_min = *minmax_v.first,
            .sigma_max = *minmax_v.second,
            .rate_min = *minmax_r.first,
            .rate_max = *minmax_r.second,
        };

        auto wrapper = MultiKRefSurfaceWrapper<>(
            std::move(*surface), bounds, config.option_type, config.dividend_yield);

        auto solver = InterpolatedIVSolver<MultiKRefSurfaceWrapper<>>::create(
            std::move(wrapper), config.solver_config);
        if (!solver.has_value()) {
            return std::unexpected(ValidationError{
                ValidationErrorCode::InvalidGridSize, 0.0});
        }

        return AnyIVSolver(std::move(*solver));
        }
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
