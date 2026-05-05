// SPDX-License-Identifier: MIT

#include "mango/option/price_table_factory.hpp"

#include "mango/math/bspline/bspline_basis.hpp"
#include "mango/math/bspline/bspline_nd_separable.hpp"
#include "mango/math/chebyshev/chebyshev_nodes.hpp"
#include "mango/option/table/bspline/bspline_3d_surface.hpp"
#include "mango/option/table/bspline/bspline_adaptive.hpp"
#include "mango/option/table/bspline/bspline_builder.hpp"
#include "mango/option/table/bspline/bspline_segmented_builder.hpp"
#include "mango/option/table/bspline/bspline_surface.hpp"
#include "mango/option/table/bspline/bspline_tensor_accessor.hpp"
#include "mango/option/table/chebyshev/chebyshev_3d_surface.hpp"
#include "mango/option/table/chebyshev/chebyshev_adaptive.hpp"
#include "mango/option/table/chebyshev/chebyshev_surface.hpp"
#include "mango/option/table/chebyshev/chebyshev_table_builder.hpp"
#include "mango/option/table/dimensionless/dimensionless_3d_accessor.hpp"
#include "mango/option/table/dimensionless/dimensionless_builder.hpp"
#include "mango/option/table/eep/analytical_eep.hpp"
#include "mango/option/table/eep/eep_decomposer.hpp"
#include "mango/option/table/parquet/parquet_io.hpp"
#include "mango/option/table/serialization/from_data.hpp"
#include "mango/option/table/serialization/to_data.hpp"
#include "mango/option/table/transforms/dimensionless_3d.hpp"

#include <algorithm>
#include <cmath>
#include <memory>
#include <type_traits>
#include <variant>

namespace mango {
namespace {

using PriceTableVariant = std::variant<
    std::shared_ptr<const BSplinePriceTable>,
    std::shared_ptr<const BSplineMultiKRefSurface>,
    std::shared_ptr<const ChebyshevSurface>,
    std::shared_ptr<const ChebyshevMultiKRefSurface>,
    std::shared_ptr<const BSpline3DPriceTable>,
    std::shared_ptr<const Chebyshev3DPriceTable>>;

}  // namespace

struct AnyPriceTable::Impl {
    PriceTableVariant table;

    template <typename T>
    explicit Impl(T t)
        : table(std::make_shared<const T>(std::move(t))) {}
};

namespace {

template <typename Table>
AnyPriceTable make_any_price_table(Table table) {
    return AnyPriceTable(
        std::make_unique<AnyPriceTable::Impl>(std::move(table)));
}

template <typename Table>
constexpr const char* surface_type_for_table() {
    using Inner = typename Table::inner_type;
    if constexpr (std::is_same_v<Inner, BSplineLeaf>) {
        return surface_types::kBSpline4D;
    } else if constexpr (std::is_same_v<Inner, BSplineMultiKRefInner>) {
        return surface_types::kBSpline4DSegmented;
    } else if constexpr (std::is_same_v<Inner, ChebyshevLeaf>) {
        return surface_types::kChebyshev4DRaw;
    } else if constexpr (std::is_same_v<Inner, ChebyshevMultiKRefInner>) {
        return surface_types::kChebyshev4DSegmented;
    } else if constexpr (std::is_same_v<Inner, BSpline3DLeaf>) {
        return surface_types::kBSpline3D;
    } else if constexpr (std::is_same_v<Inner, Chebyshev3DLeaf>) {
        return surface_types::kChebyshev3DRaw;
    } else {
        static_assert(!sizeof(Table), "Unsupported AnyPriceTable surface type");
    }
}

ValidationError to_validation_error(const PriceTableError& error) {
    switch (error.code) {
        case PriceTableErrorCode::NonPositiveValue:
            return ValidationError{ValidationErrorCode::InvalidBounds, 0.0,
                                   error.axis_index};
        case PriceTableErrorCode::InsufficientGridPoints:
        case PriceTableErrorCode::GridNotSorted:
            return ValidationError{ValidationErrorCode::InvalidGridSize,
                                   static_cast<double>(error.count),
                                   error.axis_index};
        default:
            return ValidationError{ValidationErrorCode::InvalidGridSize,
                                   static_cast<double>(error.count),
                                   error.axis_index};
    }
}

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

struct GridBounds {
    double m_min = 0.0, m_max = 0.0;
    double sigma_min = 0.0, sigma_max = 0.0;
    double rate_min = 0.0, rate_max = 0.0;
};

GridBounds extract_bounds(const IVGrid& grid) {
    if (grid.moneyness.empty() || grid.vol.empty() || grid.rate.empty()) {
        return {};
    }
    auto minmax_m = std::minmax_element(grid.moneyness.begin(), grid.moneyness.end());
    auto minmax_v = std::minmax_element(grid.vol.begin(), grid.vol.end());
    auto minmax_r = std::minmax_element(grid.rate.begin(), grid.rate.end());
    if (*minmax_m.first <= 0.0) {
        return {};
    }
    return {
        .m_min = std::log(*minmax_m.first),
        .m_max = std::log(*minmax_m.second),
        .sigma_min = *minmax_v.first,
        .sigma_max = *minmax_v.second,
        .rate_min = *minmax_r.first,
        .rate_max = *minmax_r.second,
    };
}

std::expected<BSplineMultiKRefInner, PriceTableError> build_multi_kref_manual(
    double spot,
    OptionType option_type,
    const DividendSpec& dividends,
    const IVGrid& log_grid,
    double maturity,
    const MultiKRefConfig& kref_config)
{
    std::vector<double> K_refs = kref_config.K_refs;
    if (K_refs.empty()) {
        if (kref_config.K_ref_count < 1) {
            return std::unexpected(PriceTableError{
                PriceTableErrorCode::InvalidConfig});
        }
        if (kref_config.K_ref_count == 1) {
            K_refs.push_back(spot);
        } else {
            K_refs.reserve(static_cast<size_t>(kref_config.K_ref_count));
            const double log_low = std::log(spot) - kref_config.K_ref_span;
            const double log_high = std::log(spot) + kref_config.K_ref_span;
            for (int i = 0; i < kref_config.K_ref_count; ++i) {
                const double t = static_cast<double>(i) /
                                 static_cast<double>(kref_config.K_ref_count - 1);
                K_refs.push_back(std::exp(log_low + t * (log_high - log_low)));
            }
        }
    }

    std::vector<BSplineMultiKRefEntry> entries;
    entries.reserve(K_refs.size());
    for (double K_ref : K_refs) {
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
        entries.push_back(BSplineMultiKRefEntry{
            .K_ref = K_ref,
            .surface = std::move(*surface),
        });
    }

    return build_multi_kref_surface(std::move(entries));
}

BSplineMultiKRefSurface wrap_multi_kref_surface(
    BSplineMultiKRefInner surface,
    const GridBounds& b,
    double maturity,
    OptionType option_type,
    double dividend_yield)
{
    SurfaceBounds bounds{
        .m_min = b.m_min, .m_max = b.m_max,
        .tau_min = 0.0, .tau_max = maturity,
        .sigma_min = b.sigma_min, .sigma_max = b.sigma_max,
        .rate_min = b.rate_min, .rate_max = b.rate_max,
    };

    return BSplineMultiKRefSurface(
        std::move(surface), bounds, option_type, dividend_yield);
}

std::expected<BSplineMultiKRefSurface, ValidationError>
build_bspline_segmented_table(const IVSolverFactoryConfig& config,
                              const DiscreteDividendConfig& divs) {
    auto log_m = to_log_moneyness(config.grid.moneyness);
    if (!log_m.has_value()) {
        return std::unexpected(log_m.error());
    }

    IVGrid log_grid = config.grid;
    log_grid.moneyness = std::move(*log_m);
    const auto b = extract_bounds(config.grid);

    if (config.adaptive.has_value()) {
        SegmentedAdaptiveConfig seg_config{
            .spot = config.spot,
            .option_type = config.option_type,
            .dividend_yield = config.dividend_yield,
            .discrete_dividends = divs.discrete_dividends,
            .maturity = divs.maturity,
            .kref_config = divs.kref_config,
        };

        auto result = build_adaptive_bspline_segmented(
            *config.adaptive, seg_config,
            {log_grid.moneyness, log_grid.vol, log_grid.rate});
        if (!result.has_value()) {
            return std::unexpected(to_validation_error(result.error()));
        }

        return wrap_multi_kref_surface(
            std::move(result->surface), b, divs.maturity,
            config.option_type, config.dividend_yield);
    }

    DividendSpec dividends{
        .dividend_yield = config.dividend_yield,
        .discrete_dividends = divs.discrete_dividends,
    };

    auto surface = build_multi_kref_manual(
        config.spot, config.option_type, dividends,
        log_grid, divs.maturity, divs.kref_config);
    if (!surface.has_value()) {
        return std::unexpected(to_validation_error(surface.error()));
    }

    return wrap_multi_kref_surface(
        std::move(*surface), b, divs.maturity,
        config.option_type, config.dividend_yield);
}

std::expected<BSplinePriceTable, ValidationError>
build_bspline_continuous_table(const IVSolverFactoryConfig& config,
                               const BSplineBackend& backend) {
    if (config.adaptive.has_value()) {
        OptionGrid chain;
        chain.spot = config.spot;
        chain.dividend_yield = config.dividend_yield;
        chain.strikes.reserve(config.grid.moneyness.size());
        for (double m : config.grid.moneyness) {
            chain.strikes.push_back(config.spot / m);
        }
        chain.maturities = backend.maturity_grid;
        chain.implied_vols = config.grid.vol;
        chain.rates = config.grid.rate;

        auto result = build_adaptive_bspline(
            *config.adaptive, chain,
            make_grid_accuracy(GridAccuracyProfile::High),
            config.option_type);
        if (!result.has_value()) {
            return std::unexpected(to_validation_error(result.error()));
        }

        auto table = make_bspline_surface(
            std::move(result->spline), chain.spot, chain.dividend_yield,
            config.option_type);
        if (!table.has_value()) {
            return std::unexpected(ValidationError{
                ValidationErrorCode::InvalidGridSize, 0.0});
        }
        return std::move(*table);
    }

    auto log_m = to_log_moneyness(config.grid.moneyness);
    if (!log_m.has_value()) {
        return std::unexpected(log_m.error());
    }

    auto setup = PriceTableBuilder::from_vectors(
        std::move(*log_m), backend.maturity_grid, config.grid.vol, config.grid.rate,
        config.spot, GridAccuracyParams{}, config.option_type,
        config.dividend_yield);
    if (!setup.has_value()) {
        return std::unexpected(to_validation_error(setup.error()));
    }

    auto& [builder, axes] = *setup;
    auto table_result = builder.build(axes,
        [&](PriceTensor& tensor, const PriceTableAxes& a) {
            BSplineTensorAccessor accessor(tensor, a, config.spot);
            eep_decompose(
                accessor,
                AnalyticalEEP(config.option_type, config.dividend_yield));
        });
    if (!table_result.has_value()) {
        return std::unexpected(to_validation_error(table_result.error()));
    }

    auto table = make_bspline_surface(
        table_result->spline, config.spot, config.dividend_yield,
        config.option_type);
    if (!table.has_value()) {
        return std::unexpected(ValidationError{
            ValidationErrorCode::InvalidGridSize, 0.0});
    }
    return std::move(*table);
}

std::expected<AnyPriceTable, ValidationError>
build_bspline_table(const IVSolverFactoryConfig& config,
                    const BSplineBackend& backend) {
    if (config.discrete_dividends.has_value()) {
        auto table = build_bspline_segmented_table(
            config, *config.discrete_dividends);
        if (!table.has_value()) {
            return std::unexpected(table.error());
        }
        return make_any_price_table(std::move(*table));
    }

    auto table = build_bspline_continuous_table(config, backend);
    if (!table.has_value()) {
        return std::unexpected(table.error());
    }
    return make_any_price_table(std::move(*table));
}

std::expected<ChebyshevMultiKRefSurface, ValidationError>
build_chebyshev_segmented_table(const IVSolverFactoryConfig& config,
                                const DiscreteDividendConfig& divs) {
    auto log_m = to_log_moneyness(config.grid.moneyness);
    if (!log_m.has_value()) {
        return std::unexpected(log_m.error());
    }

    SegmentedAdaptiveConfig seg_config{
        .spot = config.spot,
        .option_type = config.option_type,
        .dividend_yield = config.dividend_yield,
        .discrete_dividends = divs.discrete_dividends,
        .maturity = divs.maturity,
        .kref_config = divs.kref_config,
    };

    IVGrid log_grid{std::move(*log_m), config.grid.vol, config.grid.rate};

    if (config.adaptive.has_value()) {
        auto result = build_adaptive_chebyshev_segmented(
            *config.adaptive, seg_config, log_grid);
        if (!result.has_value()) {
            return std::unexpected(to_validation_error(result.error()));
        }
        return std::move(result->surface);
    }

    auto surface = build_chebyshev_segmented_manual(seg_config, log_grid);
    if (!surface.has_value()) {
        return std::unexpected(to_validation_error(surface.error()));
    }
    return std::move(*surface);
}

std::expected<ChebyshevSurface, ValidationError>
build_chebyshev_continuous_table(const IVSolverFactoryConfig& config,
                                 const ChebyshevBackend& backend) {
    const auto b = extract_bounds(config.grid);

    ChebyshevTableConfig cheb_config{
        .num_pts = backend.num_pts,
        .domain = Domain<4>{
            .lo = {b.m_min, std::min(0.01, backend.maturity * 0.5),
                   b.sigma_min, b.rate_min},
            .hi = {b.m_max, backend.maturity, b.sigma_max, b.rate_max},
        },
        .K_ref = config.spot,
        .option_type = config.option_type,
        .dividend_yield = config.dividend_yield,
    };

    auto result = build_chebyshev_table(cheb_config);
    if (!result.has_value()) {
        return std::unexpected(to_validation_error(result.error()));
    }
    return std::move(result->surface);
}

std::expected<AnyPriceTable, ValidationError>
build_chebyshev_table(const IVSolverFactoryConfig& config,
                      const ChebyshevBackend& backend) {
    if (config.discrete_dividends.has_value()) {
        auto table = build_chebyshev_segmented_table(
            config, *config.discrete_dividends);
        if (!table.has_value()) {
            return std::unexpected(table.error());
        }
        return make_any_price_table(std::move(*table));
    }

    auto table = build_chebyshev_continuous_table(config, backend);
    if (!table.has_value()) {
        return std::unexpected(table.error());
    }
    return make_any_price_table(std::move(*table));
}

struct DimlessDomain {
    double sigma_min = 0.0, sigma_max = 0.0;
    double rate_min = 0.0, rate_max = 0.0;
    double tau_min = 0.0, tp_min = 0.0, tp_max = 0.0;
    double lk_min = 0.0, lk_max = 0.0;
    double m_min = 0.0, m_max = 0.0;
};

DimlessDomain compute_dimless_domain(const GridBounds& b, double maturity) {
    DimlessDomain d;
    d.sigma_min = b.sigma_min;
    d.sigma_max = b.sigma_max;
    d.rate_min = b.rate_min;
    d.rate_max = b.rate_max;
    d.m_min = b.m_min;
    d.m_max = b.m_max;
    d.tau_min = 0.01;
    d.tp_min = d.sigma_min * d.sigma_min * d.tau_min / 2.0;
    d.tp_max = d.sigma_max * d.sigma_max * maturity / 2.0;
    d.lk_min = std::log(2.0 * d.rate_min /
                        (d.sigma_max * d.sigma_max));
    d.lk_max = std::log(2.0 * d.rate_max /
                        (d.sigma_min * d.sigma_min));
    return d;
}

SurfaceBounds dimless_bounds(const DimlessDomain& d, double maturity) {
    return {
        .m_min = d.m_min, .m_max = d.m_max,
        .tau_min = d.tau_min, .tau_max = maturity,
        .sigma_min = d.sigma_min, .sigma_max = d.sigma_max,
        .rate_min = d.rate_min, .rate_max = d.rate_max,
    };
}

struct DimensionlessGridSpec {
    DimensionlessAxes axes;
    DimlessDomain domain;
};

std::expected<DimensionlessGridSpec, ValidationError>
build_dimensionless_grid(const IVGrid& grid, double maturity) {
    auto log_m = to_log_moneyness(grid.moneyness);
    if (!log_m.has_value()) {
        return std::unexpected(log_m.error());
    }

    auto b = extract_bounds(grid);
    auto d = compute_dimless_domain(b, maturity);

    auto linspace = [](double lo, double hi, size_t n) {
        std::vector<double> values(n);
        for (size_t i = 0; i < n; ++i) {
            values[i] = lo + (hi - lo) * static_cast<double>(i) /
                             static_cast<double>(n - 1);
        }
        return values;
    };

    DimensionlessAxes axes{
        .log_moneyness = linspace(d.m_min, d.m_max, 12),
        .tau_prime = linspace(d.tp_min, d.tp_max, 10),
        .ln_kappa = linspace(d.lk_min, d.lk_max, 10),
    };

    return DimensionlessGridSpec{std::move(axes), d};
}

std::expected<BSpline3DPriceTable, ValidationError>
build_dimensionless_bspline_table(const IVSolverFactoryConfig& config,
                                  const DimensionlessBackend& backend) {
    auto grid_spec = build_dimensionless_grid(config.grid, backend.maturity);
    if (!grid_spec.has_value()) {
        return std::unexpected(grid_spec.error());
    }

    auto& axes = grid_spec->axes;
    auto& d = grid_spec->domain;

    auto pde = solve_dimensionless_pde(
        axes, config.spot, config.option_type);
    if (!pde.has_value()) {
        return std::unexpected(to_validation_error(pde.error()));
    }

    Dimensionless3DAccessor accessor(pde->values, axes, config.spot);
    eep_decompose(accessor, AnalyticalEEP(config.option_type, 0.0));

    std::array<std::vector<double>, 3> grids = {
        axes.log_moneyness, axes.tau_prime, axes.ln_kappa};
    auto fitter_result = BSplineNDSeparable<double, 3>::create(grids);
    if (!fitter_result.has_value()) {
        return std::unexpected(ValidationError{
            ValidationErrorCode::InvalidGridSize, 0.0});
    }

    auto fit_result = fitter_result->fit(std::move(pde->values));
    if (!fit_result.has_value()) {
        return std::unexpected(ValidationError{
            ValidationErrorCode::InvalidGridSize, 0.0});
    }

    typename BSplineND<double, 3>::KnotArray knots;
    typename BSplineND<double, 3>::GridArray grids_3d;
    grids_3d[0] = axes.log_moneyness;
    grids_3d[1] = axes.tau_prime;
    grids_3d[2] = axes.ln_kappa;
    for (size_t dim = 0; dim < 3; ++dim) {
        knots[dim] = clamped_knots_cubic(grids_3d[dim]);
    }

    auto spline_result = BSplineND<double, 3>::create(
        std::move(grids_3d), std::move(knots),
        std::move(fit_result->coefficients));
    if (!spline_result.has_value()) {
        return std::unexpected(ValidationError{
            ValidationErrorCode::InvalidGridSize, 0.0});
    }

    auto spline_ptr = std::make_shared<const BSplineND<double, 3>>(
        std::move(*spline_result));
    SharedBSplineInterp<3> interp(spline_ptr);
    DimensionlessTransform3D xform;
    BSpline3DTransformLeaf leaf(std::move(interp), xform, config.spot);
    AnalyticalEEP eep(config.option_type, 0.0);
    BSpline3DLeaf eep_leaf(std::move(leaf), std::move(eep));

    return BSpline3DPriceTable(
        std::move(eep_leaf), dimless_bounds(d, backend.maturity),
        config.option_type, 0.0);
}

std::expected<Chebyshev3DPriceTable, ValidationError>
build_dimensionless_chebyshev_table(const IVSolverFactoryConfig& config,
                                    const DimensionlessBackend& backend) {
    auto log_m = to_log_moneyness(config.grid.moneyness);
    if (!log_m.has_value()) {
        return std::unexpected(log_m.error());
    }

    const auto b = extract_bounds(config.grid);
    const auto d = compute_dimless_domain(b, backend.maturity);

    auto x_nodes = chebyshev_nodes(
        backend.chebyshev_pts[0], d.m_min, d.m_max);
    auto tp_nodes = chebyshev_nodes(
        backend.chebyshev_pts[1], d.tp_min, d.tp_max);
    auto lk_nodes = chebyshev_nodes(
        backend.chebyshev_pts[2], d.lk_min, d.lk_max);

    DimensionlessAxes axes{
        .log_moneyness = x_nodes,
        .tau_prime = tp_nodes,
        .ln_kappa = lk_nodes,
    };

    auto pde = solve_dimensionless_pde(
        axes, config.spot, config.option_type);
    if (!pde.has_value()) {
        return std::unexpected(to_validation_error(pde.error()));
    }

    Dimensionless3DAccessor accessor(pde->values, axes, config.spot);
    eep_decompose(accessor, AnalyticalEEP(config.option_type, 0.0));

    Domain<3> domain{
        .lo = {d.m_min, d.tp_min, d.lk_min},
        .hi = {d.m_max, d.tp_max, d.lk_max},
    };

    auto cheb = ChebyshevInterpolant<3, RawTensor<3>>::build_from_values(
        std::span<const double>(pde->values), domain, backend.chebyshev_pts);

    DimensionlessTransform3D xform;
    Chebyshev3DTransformLeaf leaf(std::move(cheb), xform, config.spot);
    AnalyticalEEP eep_fn(config.option_type, 0.0);
    Chebyshev3DLeaf eep_leaf(std::move(leaf), std::move(eep_fn));

    return Chebyshev3DPriceTable(
        std::move(eep_leaf), dimless_bounds(d, backend.maturity),
        config.option_type, 0.0);
}

std::expected<AnyPriceTable, ValidationError>
build_dimensionless_table(const IVSolverFactoryConfig& config,
                          const DimensionlessBackend& backend) {
    if (std::abs(config.dividend_yield) > 1e-12) {
        return std::unexpected(ValidationError{
            ValidationErrorCode::InvalidDividend, config.dividend_yield});
    }
    if (config.discrete_dividends.has_value() &&
        !config.discrete_dividends->discrete_dividends.empty()) {
        return std::unexpected(ValidationError{
            ValidationErrorCode::InvalidDividend,
            static_cast<double>(
                config.discrete_dividends->discrete_dividends.size())});
    }

    const auto b = extract_bounds(config.grid);
    if (b.rate_min <= 0.0) {
        return std::unexpected(ValidationError{
            ValidationErrorCode::InvalidRate, b.rate_min});
    }
    if (b.sigma_min <= 0.0) {
        return std::unexpected(ValidationError{
            ValidationErrorCode::InvalidVolatility, b.sigma_min});
    }

    if (backend.interpolant == DimensionlessBackend::Interpolant::Chebyshev) {
        auto table = build_dimensionless_chebyshev_table(config, backend);
        if (!table.has_value()) {
            return std::unexpected(table.error());
        }
        return make_any_price_table(std::move(*table));
    }

    auto table = build_dimensionless_bspline_table(config, backend);
    if (!table.has_value()) {
        return std::unexpected(table.error());
    }
    return make_any_price_table(std::move(*table));
}

ParquetCompression to_parquet_compression(PriceTableCompression compression) {
    switch (compression) {
        case PriceTableCompression::NONE:
            return ParquetCompression::NONE;
        case PriceTableCompression::SNAPPY:
            return ParquetCompression::SNAPPY;
        case PriceTableCompression::ZSTD:
            return ParquetCompression::ZSTD;
    }
    return ParquetCompression::ZSTD;
}

template <typename Inner>
std::expected<AnyPriceTable, PriceTableError>
load_as(const PriceTableData& data) {
    auto table = from_data<Inner>(data);
    if (!table.has_value()) {
        return std::unexpected(table.error());
    }
    return make_any_price_table(std::move(*table));
}

}  // namespace

AnyPriceTable::AnyPriceTable(std::unique_ptr<Impl> impl)
    : impl_(std::move(impl)) {}

AnyPriceTable::AnyPriceTable(AnyPriceTable&&) noexcept = default;
AnyPriceTable& AnyPriceTable::operator=(AnyPriceTable&&) noexcept = default;
AnyPriceTable::~AnyPriceTable() = default;

std::string AnyPriceTable::surface_type() const {
    return std::visit([](const auto& table_ptr) {
        using Table = std::remove_cv_t<
            typename std::decay_t<decltype(table_ptr)>::element_type>;
        return std::string(surface_type_for_table<Table>());
    }, impl_->table);
}

OptionType AnyPriceTable::option_type() const noexcept {
    return std::visit([](const auto& table_ptr) noexcept {
        return table_ptr->option_type();
    }, impl_->table);
}

double AnyPriceTable::dividend_yield() const noexcept {
    return std::visit([](const auto& table_ptr) noexcept {
        return table_ptr->dividend_yield();
    }, impl_->table);
}

std::expected<void, ValidationError>
AnyPriceTable::validate_pricing_params(const PricingParams& params) const {
    auto base_validation = mango::validate_pricing_params(params);
    if (!base_validation.has_value()) {
        return base_validation;
    }

    if (params.option_type != option_type()) {
        return std::unexpected(ValidationError{
            ValidationErrorCode::OptionTypeMismatch,
            static_cast<double>(params.option_type)});
    }

    if (std::abs(params.dividend_yield - dividend_yield()) > 1e-10) {
        return std::unexpected(ValidationError{
            ValidationErrorCode::DividendYieldMismatch,
            params.dividend_yield});
    }

    const double log_moneyness = std::log(params.spot / params.strike);
    const double rate = get_zero_rate(params.rate, params.maturity);
    return std::visit([&](const auto& table_ptr)
        -> std::expected<void, ValidationError> {
        if (log_moneyness < table_ptr->m_min() ||
            log_moneyness > table_ptr->m_max()) {
            return std::unexpected(ValidationError{
                ValidationErrorCode::OutOfRange, log_moneyness, 0});
        }
        if (params.maturity < table_ptr->tau_min() ||
            params.maturity > table_ptr->tau_max()) {
            return std::unexpected(ValidationError{
                ValidationErrorCode::OutOfRange, params.maturity, 1});
        }
        if (params.volatility < table_ptr->sigma_min() ||
            params.volatility > table_ptr->sigma_max()) {
            return std::unexpected(ValidationError{
                ValidationErrorCode::OutOfRange, params.volatility, 2});
        }
        if (rate < table_ptr->rate_min() || rate > table_ptr->rate_max()) {
            return std::unexpected(ValidationError{
                ValidationErrorCode::OutOfRange, rate, 3});
        }
        return {};
    }, impl_->table);
}

double AnyPriceTable::price(const PricingParams& params) const {
    const double rate = get_zero_rate(params.rate, params.maturity);
    return std::visit([&](const auto& table_ptr) {
        return table_ptr->price(params.spot, params.strike, params.maturity,
                                params.volatility, rate);
    }, impl_->table);
}

double AnyPriceTable::vega(const PricingParams& params) const {
    const double rate = get_zero_rate(params.rate, params.maturity);
    return std::visit([&](const auto& table_ptr) {
        return table_ptr->vega(params.spot, params.strike, params.maturity,
                               params.volatility, rate);
    }, impl_->table);
}

std::expected<double, GreekError>
AnyPriceTable::delta(const PricingParams& params) const {
    return std::visit([&](const auto& table_ptr) {
        return table_ptr->delta(params);
    }, impl_->table);
}

std::expected<double, GreekError>
AnyPriceTable::gamma(const PricingParams& params) const {
    return std::visit([&](const auto& table_ptr) {
        return table_ptr->gamma(params);
    }, impl_->table);
}

std::expected<double, GreekError>
AnyPriceTable::theta(const PricingParams& params) const {
    return std::visit([&](const auto& table_ptr) {
        return table_ptr->theta(params);
    }, impl_->table);
}

std::expected<double, GreekError>
AnyPriceTable::rho(const PricingParams& params) const {
    return std::visit([&](const auto& table_ptr) {
        return table_ptr->rho(params);
    }, impl_->table);
}

std::expected<AnyInterpIVSolver, ValidationError>
AnyPriceTable::make_iv_solver(
    const InterpolatedIVSolverConfig& config) const {
    return std::visit([&](const auto& table_ptr)
        -> std::expected<AnyInterpIVSolver, ValidationError> {
        using Table = std::remove_cv_t<
            typename std::decay_t<decltype(table_ptr)>::element_type>;
        using SharedSurface = detail::SharedPriceTableSurface<Table>;
        auto solver = InterpolatedIVSolver<SharedSurface>::create(
            SharedSurface(table_ptr), config);
        if (!solver.has_value()) {
            return std::unexpected(solver.error());
        }
        return make_any_interpolated_solver(std::move(*solver));
    }, impl_->table);
}

std::expected<IVSuccess, IVError>
AnyPriceTable::solve_iv(
    const IVQuery& query,
    const InterpolatedIVSolverConfig& config) const {
    auto solver = make_iv_solver(config);
    if (!solver.has_value()) {
        return std::unexpected(validation_error_to_iv_error(solver.error()));
    }
    return solver->solve(query);
}

PriceTableData AnyPriceTable::to_data() const {
    return std::visit([](const auto& table_ptr) {
        return mango::to_data(*table_ptr);
    }, impl_->table);
}

std::expected<void, PriceTableError>
AnyPriceTable::save(
    const std::filesystem::path& path,
    PriceTableCompression compression) const {
    return write_parquet(
        to_data(), path,
        ParquetWriteOptions{
            .compression = to_parquet_compression(compression),
        });
}

std::expected<AnyPriceTable, ValidationError>
make_price_table(const IVSolverFactoryConfig& config) {
    return std::visit([&](const auto& backend)
        -> std::expected<AnyPriceTable, ValidationError> {
        using Backend = std::decay_t<decltype(backend)>;
        if constexpr (std::is_same_v<Backend, BSplineBackend>) {
            return build_bspline_table(config, backend);
        } else if constexpr (std::is_same_v<Backend, ChebyshevBackend>) {
            return build_chebyshev_table(config, backend);
        } else {
            return build_dimensionless_table(config, backend);
        }
    }, config.backend);
}

std::expected<AnyPriceTable, PriceTableError>
load_price_table(const std::filesystem::path& path) {
    auto data = read_parquet(path);
    if (!data.has_value()) {
        return std::unexpected(data.error());
    }

    const auto& type = data->surface_type;
    if (type == surface_types::kBSpline4D) {
        return load_as<BSplineLeaf>(*data);
    }
    if (type == surface_types::kBSpline4DSegmented) {
        return load_as<BSplineMultiKRefInner>(*data);
    }
    if (type == surface_types::kChebyshev4D ||
        type == surface_types::kChebyshev4DRaw) {
        return load_as<ChebyshevLeaf>(*data);
    }
    if (type == surface_types::kChebyshev4DSegmented) {
        return load_as<ChebyshevMultiKRefInner>(*data);
    }
    if (type == surface_types::kBSpline3D) {
        return load_as<BSpline3DLeaf>(*data);
    }
    if (type == surface_types::kChebyshev3D ||
        type == surface_types::kChebyshev3DRaw) {
        return load_as<Chebyshev3DLeaf>(*data);
    }

    return std::unexpected(PriceTableError{
        PriceTableErrorCode::InvalidConfig});
}

std::expected<AnyInterpIVSolver, ValidationError>
make_interpolated_iv_solver(const IVSolverFactoryConfig& config) {
    auto table = make_price_table(config);
    if (!table.has_value()) {
        return std::unexpected(table.error());
    }
    return table->make_iv_solver(config.solver_config);
}

}  // namespace mango
