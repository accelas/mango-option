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
#include "mango/option/table/adaptive_grid_types.hpp"
#include "mango/option/table/bspline/bspline_adaptive.hpp"
#include "mango/option/table/bspline/bspline_surface.hpp"
#include "mango/option/table/bspline/bspline_tensor_accessor.hpp"
#include "mango/option/table/bspline/bspline_builder.hpp"
#include "mango/option/table/bspline/bspline_segmented_builder.hpp"
#include "mango/option/table/chebyshev/chebyshev_adaptive.hpp"
#include "mango/option/table/chebyshev/chebyshev_surface.hpp"
#include "mango/option/table/chebyshev/chebyshev_table_builder.hpp"
#include "mango/option/table/bspline/bspline_3d_surface.hpp"
#include "mango/option/table/chebyshev/chebyshev_3d_surface.hpp"
#include "mango/option/table/dimensionless/dimensionless_builder.hpp"
#include "mango/option/table/eep/dimensionless_3d_accessor.hpp"
#include "mango/option/table/transforms/dimensionless_3d.hpp"
#include "mango/math/bspline_nd_separable.hpp"
#include "mango/math/chebyshev/chebyshev_nodes.hpp"
#include "mango/math/chebyshev/tucker_tensor.hpp"
#include <algorithm>
#include <cmath>
#include <variant>

namespace mango {

// =====================================================================
// Explicit template instantiations
// =====================================================================

template class InterpolatedIVSolver<BSplinePriceTable>;
template class InterpolatedIVSolver<BSplineMultiKRefSurface>;
template class InterpolatedIVSolver<ChebyshevSurface>;
template class InterpolatedIVSolver<ChebyshevRawSurface>;
template class InterpolatedIVSolver<BSpline3DPriceTable>;
template class InterpolatedIVSolver<Chebyshev3DPriceTable>;

// =====================================================================
// Factory internals
// =====================================================================

/// Surface leaf that wraps a type-erased price_fn and computes vega via FD.
/// Used for segmented Chebyshev which produces multi-K_ref blended price_fn.
class FDVegaLeaf {
public:
    using PriceFn = std::function<double(double, double, double, double, double)>;

    explicit FDVegaLeaf(PriceFn fn) : fn_(std::move(fn)) {}

    double price(double spot, double strike,
                 double tau, double sigma, double rate) const {
        return fn_(spot, strike, tau, sigma, rate);
    }

    double vega(double spot, double strike,
                double tau, double sigma, double rate) const {
        double eps = std::max(1e-4, 0.01 * sigma);
        double sigma_up = sigma + eps;
        double sigma_dn = std::max(1e-4, sigma - eps);
        double eff_eps = (sigma_up - sigma_dn) / 2.0;
        return (fn_(spot, strike, tau, sigma_up, rate) -
                fn_(spot, strike, tau, sigma_dn, rate)) / (2.0 * eff_eps);
    }

private:
    PriceFn fn_;
};

using ChebyshevSegmentedSurface = PriceTable<FDVegaLeaf>;
template class InterpolatedIVSolver<ChebyshevSegmentedSurface>;

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

/// Build a BSplineMultiKRefInner for manual grid path
std::expected<BSplineMultiKRefInner, PriceTableError> build_multi_kref_manual(
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
        if (kref_config.K_ref_count < 1) {
            return std::unexpected(PriceTableError{PriceTableErrorCode::InvalidConfig});
        }
        // Auto-generate K_refs around spot
        if (kref_config.K_ref_count == 1) {
            K_refs.push_back(spot);
        } else {
            K_refs.reserve(static_cast<size_t>(kref_config.K_ref_count));
            double log_low = std::log(spot) - kref_config.K_ref_span;
            double log_high = std::log(spot) + kref_config.K_ref_span;
            for (int i = 0; i < kref_config.K_ref_count; ++i) {
                double t = static_cast<double>(i) / (kref_config.K_ref_count - 1);
                K_refs.push_back(std::exp(log_low + t * (log_high - log_low)));
            }
        }
    }

    std::vector<BSplineMultiKRefEntry> entries;
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

        entries.push_back(BSplineMultiKRefEntry{
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
    if (grid.moneyness.empty() || grid.vol.empty() || grid.rate.empty()) {
        return {};  // Zero-initialized; caller validates non-empty bounds
    }
    auto minmax_m = std::minmax_element(grid.moneyness.begin(), grid.moneyness.end());
    auto minmax_v = std::minmax_element(grid.vol.begin(), grid.vol.end());
    auto minmax_r = std::minmax_element(grid.rate.begin(), grid.rate.end());
    // Moneyness must be positive for log conversion
    if (*minmax_m.first <= 0.0) {
        return {};
    }
    return {
        .m_min = std::log(*minmax_m.first), .m_max = std::log(*minmax_m.second),
        .sigma_min = *minmax_v.first, .sigma_max = *minmax_v.second,
        .rate_min = *minmax_r.first, .rate_max = *minmax_r.second,
    };
}

}  // anonymous namespace

// =====================================================================
// AnyIVSolver: pimpl implementation
// =====================================================================

struct AnyIVSolver::Impl {
    using SolverVariant = std::variant<
        InterpolatedIVSolver<BSplinePriceTable>,
        InterpolatedIVSolver<BSplineMultiKRefSurface>,
        InterpolatedIVSolver<ChebyshevSurface>,
        InterpolatedIVSolver<ChebyshevRawSurface>,
        InterpolatedIVSolver<ChebyshevSegmentedSurface>,
        InterpolatedIVSolver<BSpline3DPriceTable>,
        InterpolatedIVSolver<Chebyshev3DPriceTable>
    >;
    SolverVariant solver;

    template <typename T>
    explicit Impl(T s) : solver(std::move(s)) {}
};

AnyIVSolver::AnyIVSolver(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}
AnyIVSolver::AnyIVSolver(AnyIVSolver&&) noexcept = default;
AnyIVSolver& AnyIVSolver::operator=(AnyIVSolver&&) noexcept = default;
AnyIVSolver::~AnyIVSolver() = default;

std::expected<IVSuccess, IVError> AnyIVSolver::solve(const IVQuery& query) const {
    return std::visit([&](const auto& solver) {
        return solver.solve(query);
    }, impl_->solver);
}

BatchIVResult AnyIVSolver::solve_batch(const std::vector<IVQuery>& queries) const {
    return std::visit([&](const auto& solver) {
        return solver.solve_batch(queries);
    }, impl_->solver);
}

/// Helper: wrap a typed solver into AnyIVSolver via pimpl
template <typename Surface>
static AnyIVSolver make_any_solver(InterpolatedIVSolver<Surface> solver) {
    return AnyIVSolver(std::make_unique<AnyIVSolver::Impl>(std::move(solver)));
}

// ---------------------------------------------------------------------------
// Helper: wrap surface into AnyIVSolver
// ---------------------------------------------------------------------------

static std::expected<AnyIVSolver, ValidationError>
wrap_surface(std::shared_ptr<const PriceTableSurface> surface,
             OptionType option_type,
             const InterpolatedIVSolverConfig& solver_config) {
    auto wrapper = make_bspline_surface(surface, option_type);
    if (!wrapper.has_value()) {
        return std::unexpected(ValidationError{
            ValidationErrorCode::InvalidGridSize, 0.0});
    }

    auto solver = InterpolatedIVSolver<BSplinePriceTable>::create(
        std::move(*wrapper), solver_config);
    if (!solver.has_value()) {
        return std::unexpected(ValidationError{
            ValidationErrorCode::InvalidGridSize, 0.0});
    }

    return make_any_solver(std::move(*solver));
}

// ---------------------------------------------------------------------------
// Factory: B-spline + continuous (adaptive)
// ---------------------------------------------------------------------------

static std::expected<AnyIVSolver, ValidationError>
build_bspline_adaptive(const IVSolverFactoryConfig& config,
                       const BSplineBackend& backend) {
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

    // Use auto-estimated grid with High profile for better accuracy
    // (Fixed 101x500 grid was too coarse, causing ~600 bps IV errors)
    GridAccuracyParams accuracy = make_grid_accuracy(GridAccuracyProfile::High);

    auto result = build_adaptive_bspline(*config.adaptive, chain, accuracy, config.option_type);

    if (!result.has_value()) {
        return std::unexpected(ValidationError{
            ValidationErrorCode::InvalidGridSize, 0.0});
    }

    return wrap_surface(std::move(result->surface), config.option_type, config.solver_config);
}

// ---------------------------------------------------------------------------
// Factory: B-spline + continuous (dispatch manual vs adaptive)
// ---------------------------------------------------------------------------

static std::expected<AnyIVSolver, ValidationError>
build_bspline(const IVSolverFactoryConfig& config, const BSplineBackend& backend) {
    if (config.adaptive.has_value()) {
        return build_bspline_adaptive(config, backend);
    }

    // Manual grid: build price table directly
    auto log_m = to_log_moneyness(config.grid.moneyness);
    if (!log_m.has_value()) {
        return std::unexpected(log_m.error());
    }
    auto setup = PriceTableBuilder::from_vectors(
        std::move(*log_m), backend.maturity_grid, config.grid.vol, config.grid.rate,
        config.spot, GridAccuracyParams{}, config.option_type,
        config.dividend_yield);
    if (!setup.has_value()) {
        return std::unexpected(ValidationError{
            ValidationErrorCode::InvalidGridSize, 0.0});
    }

    auto& [builder, axes] = *setup;

    // Standard path: decompose tensor to EEP before B-spline fitting
    auto table_result = builder.build(axes,
        [&](PriceTensor& tensor, const PriceTableAxes& a) {
            BSplineTensorAccessor accessor(tensor, a, config.spot);
            eep_decompose(accessor, AnalyticalEEP(config.option_type, config.dividend_yield));
        });
    if (!table_result.has_value()) {
        return std::unexpected(ValidationError{
            ValidationErrorCode::InvalidGridSize, 0.0});
    }

    return wrap_surface(table_result->surface, config.option_type, config.solver_config);
}

// ---------------------------------------------------------------------------
// Factory: B-spline + discrete dividends helpers
// ---------------------------------------------------------------------------

/// Wrap a BSplineMultiKRefInner into AnyIVSolver
static std::expected<AnyIVSolver, ValidationError>
wrap_multi_kref_surface(BSplineMultiKRefInner surface,
                        const GridBounds& b, double maturity,
                        OptionType option_type, double dividend_yield,
                        const InterpolatedIVSolverConfig& solver_config) {
    SurfaceBounds bounds{
        .m_min = b.m_min, .m_max = b.m_max,
        .tau_min = 0.0, .tau_max = maturity,
        .sigma_min = b.sigma_min, .sigma_max = b.sigma_max,
        .rate_min = b.rate_min, .rate_max = b.rate_max,
    };

    auto wrapper = BSplineMultiKRefSurface(
        std::move(surface), bounds, option_type, dividend_yield);

    auto solver = InterpolatedIVSolver<BSplineMultiKRefSurface>::create(
        std::move(wrapper), solver_config);
    if (!solver.has_value()) {
        return std::unexpected(ValidationError{
            ValidationErrorCode::InvalidGridSize, 0.0});
    }
    return make_any_solver(std::move(*solver));
}

// ---------------------------------------------------------------------------
// Factory: B-spline + discrete dividends
// ---------------------------------------------------------------------------

static std::expected<AnyIVSolver, ValidationError>
build_bspline_segmented(const IVSolverFactoryConfig& config,
                        const DiscreteDividendConfig& divs) {
    const auto& kref_config = divs.kref_config;
    const auto& grid = config.grid;

    auto log_m = to_log_moneyness(grid.moneyness);
    if (!log_m.has_value()) {
        return std::unexpected(log_m.error());
    }
    IVGrid log_grid = grid;
    log_grid.moneyness = std::move(*log_m);

    auto b = extract_bounds(grid);

    if (config.adaptive.has_value()) {
        SegmentedAdaptiveConfig seg_config{
            .spot = config.spot,
            .option_type = config.option_type,
            .dividend_yield = config.dividend_yield,
            .discrete_dividends = divs.discrete_dividends,
            .maturity = divs.maturity,
            .kref_config = kref_config,
        };

        auto result = build_adaptive_bspline_segmented(
            *config.adaptive, seg_config,
            {log_grid.moneyness, log_grid.vol, log_grid.rate});
        if (!result.has_value()) {
            return std::unexpected(ValidationError{
                ValidationErrorCode::InvalidGridSize, 0.0});
        }
        return wrap_multi_kref_surface(std::move(result->surface),
            b, divs.maturity, config.option_type,
            config.dividend_yield, config.solver_config);
    }

    DividendSpec dividends{
        .dividend_yield = config.dividend_yield,
        .discrete_dividends = divs.discrete_dividends
    };

    auto surface = build_multi_kref_manual(
        config.spot, config.option_type, dividends,
        log_grid, divs.maturity, kref_config);
    if (!surface.has_value()) {
        return std::unexpected(ValidationError{
            ValidationErrorCode::InvalidGridSize, 0.0});
    }
    return wrap_multi_kref_surface(std::move(*surface),
        b, divs.maturity, config.option_type,
        config.dividend_yield, config.solver_config);
}

// ---------------------------------------------------------------------------
// Factory: Chebyshev + continuous
// ---------------------------------------------------------------------------

static std::expected<AnyIVSolver, ValidationError>
build_chebyshev(const IVSolverFactoryConfig& config,
                const ChebyshevBackend& backend) {
    auto b = extract_bounds(config.grid);

    ChebyshevTableConfig cheb_config{
        .num_pts = backend.num_pts,
        .domain = Domain<4>{
            .lo = {b.m_min, std::min(0.01, backend.maturity * 0.5), b.sigma_min, b.rate_min},
            .hi = {b.m_max, backend.maturity, b.sigma_max, b.rate_max},
        },
        .K_ref = config.spot,
        .option_type = config.option_type,
        .dividend_yield = config.dividend_yield,
        .tucker_epsilon = backend.tucker_epsilon,
    };

    auto result = build_chebyshev_table(cheb_config);
    if (!result.has_value()) {
        return std::unexpected(ValidationError{
            ValidationErrorCode::InvalidGridSize, 0.0});
    }

    return std::visit([&](auto&& surface) -> std::expected<AnyIVSolver, ValidationError> {
        auto solver = InterpolatedIVSolver<std::decay_t<decltype(surface)>>::create(
            std::move(surface), config.solver_config);
        if (!solver.has_value()) {
            return std::unexpected(solver.error());
        }
        return make_any_solver(std::move(*solver));
    }, std::move(result->surface));
}

// ---------------------------------------------------------------------------
// Factory: Chebyshev + discrete dividends
// ---------------------------------------------------------------------------

static std::expected<AnyIVSolver, ValidationError>
build_chebyshev_segmented(const IVSolverFactoryConfig& config,
                          const ChebyshevBackend& /* backend */,
                          const DiscreteDividendConfig& divs) {
    // Segmented Chebyshev requires adaptive grid builder
    if (!config.adaptive.has_value()) {
        return std::unexpected(ValidationError{
            ValidationErrorCode::InvalidGridSize, 0.0});
    }

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
    auto result = build_adaptive_chebyshev_segmented(
        *config.adaptive, seg_config, log_grid);
    if (!result.has_value()) {
        return std::unexpected(ValidationError{
            ValidationErrorCode::InvalidGridSize, 0.0});
    }

    // Wrap the type-erased price_fn in FDVegaLeaf for Newton-based IV solving
    auto b = extract_bounds(config.grid);
    SurfaceBounds bounds{
        .m_min = b.m_min, .m_max = b.m_max,
        .tau_min = std::min(0.01, divs.maturity * 0.5), .tau_max = divs.maturity,
        .sigma_min = b.sigma_min, .sigma_max = b.sigma_max,
        .rate_min = b.rate_min, .rate_max = b.rate_max,
    };

    FDVegaLeaf leaf(std::move(result->price_fn));
    ChebyshevSegmentedSurface surface(
        std::move(leaf), bounds, config.option_type, config.dividend_yield);

    auto solver = InterpolatedIVSolver<ChebyshevSegmentedSurface>::create(
        std::move(surface), config.solver_config);
    if (!solver.has_value()) {
        return std::unexpected(solver.error());
    }
    return make_any_solver(std::move(*solver));
}

// ---------------------------------------------------------------------------
// Factory: Dimensionless 3D
// ---------------------------------------------------------------------------

/// Compute dimensionless domain bounds from config.
struct DimlessDomain {
    double sigma_min, sigma_max, rate_min, rate_max;
    double tau_min, tp_min, tp_max, lk_min, lk_max;
    double m_min, m_max;
};

static DimlessDomain compute_dimless_domain(
    const GridBounds& b, double maturity)
{
    DimlessDomain d;
    d.sigma_min = b.sigma_min;  d.sigma_max = b.sigma_max;
    d.rate_min  = b.rate_min;   d.rate_max  = b.rate_max;
    d.m_min     = b.m_min;      d.m_max     = b.m_max;
    d.tau_min   = 0.01;
    d.tp_min = d.sigma_min * d.sigma_min * d.tau_min / 2.0;
    d.tp_max = d.sigma_max * d.sigma_max * maturity / 2.0;
    d.lk_min = std::log(2.0 * d.rate_min / (d.sigma_max * d.sigma_max));
    d.lk_max = std::log(2.0 * d.rate_max / (d.sigma_min * d.sigma_min));
    return d;
}

static SurfaceBounds dimless_bounds(const DimlessDomain& d, double maturity) {
    return {
        .m_min = d.m_min, .m_max = d.m_max,
        .tau_min = d.tau_min, .tau_max = maturity,
        .sigma_min = d.sigma_min, .sigma_max = d.sigma_max,
        .rate_min = d.rate_min, .rate_max = d.rate_max,
    };
}

static std::expected<AnyIVSolver, ValidationError>
build_dimensionless_bspline(const IVSolverFactoryConfig& config,
                            const DimensionlessBackend& backend) {
    auto log_m = to_log_moneyness(config.grid.moneyness);
    if (!log_m.has_value()) return std::unexpected(log_m.error());

    auto b = extract_bounds(config.grid);
    auto d = compute_dimless_domain(b, backend.maturity);

    auto linspace = [](double lo, double hi, size_t n) {
        std::vector<double> v(n);
        for (size_t i = 0; i < n; ++i)
            v[i] = lo + (hi - lo) * static_cast<double>(i) / static_cast<double>(n - 1);
        return v;
    };

    DimensionlessAxes axes{
        .log_moneyness = linspace(d.m_min, d.m_max, 12),
        .tau_prime = linspace(d.tp_min, d.tp_max, 10),
        .ln_kappa = linspace(d.lk_min, d.lk_max, 10),
    };

    // 1. PDE solve -> raw V/K
    auto pde = solve_dimensionless_pde(axes, config.spot, config.option_type);
    if (!pde.has_value()) {
        return std::unexpected(ValidationError{
            ValidationErrorCode::InvalidGridSize, 0.0});
    }

    // 2. EEP decompose via accessor
    Dimensionless3DAccessor accessor(pde->values, axes, config.spot);
    eep_decompose(accessor, AnalyticalEEP(config.option_type, 0.0));

    // 3. Fit B-spline on (now dollar EEP) values
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

    // 4. Build surface with actual K_ref
    PriceTableAxesND<3> surface_axes;
    surface_axes.grids[0] = axes.log_moneyness;
    surface_axes.grids[1] = axes.tau_prime;
    surface_axes.grids[2] = axes.ln_kappa;
    surface_axes.names = {"log_moneyness", "tau_prime", "ln_kappa"};

    auto surface = PriceTableSurfaceND<3>::build(
        std::move(surface_axes), std::move(fit_result->coefficients), config.spot);
    if (!surface.has_value()) {
        return std::unexpected(ValidationError{
            ValidationErrorCode::InvalidGridSize, 0.0});
    }

    // 5. Wrap in layered PriceTable (K_ref = config.spot)
    SharedBSplineInterp<3> interp(std::move(surface.value()));
    DimensionlessTransform3D xform;
    BSpline3DTransformLeaf leaf(std::move(interp), xform, config.spot);
    AnalyticalEEP eep(config.option_type, 0.0);
    BSpline3DLeaf eep_leaf(std::move(leaf), std::move(eep));

    BSpline3DPriceTable table(
        std::move(eep_leaf), dimless_bounds(d, backend.maturity),
        config.option_type, 0.0);

    auto solver = InterpolatedIVSolver<BSpline3DPriceTable>::create(
        std::move(table), config.solver_config);
    if (!solver.has_value()) return std::unexpected(solver.error());
    return make_any_solver(std::move(*solver));
}

static std::expected<AnyIVSolver, ValidationError>
build_dimensionless_chebyshev(const IVSolverFactoryConfig& config,
                              const DimensionlessBackend& backend) {
    auto log_m = to_log_moneyness(config.grid.moneyness);
    if (!log_m.has_value()) return std::unexpected(log_m.error());

    auto b = extract_bounds(config.grid);
    auto d = compute_dimless_domain(b, backend.maturity);

    // Generate Chebyshev nodes per axis
    auto x_nodes  = chebyshev_nodes(backend.chebyshev_pts[0], d.m_min, d.m_max);
    auto tp_nodes = chebyshev_nodes(backend.chebyshev_pts[1], d.tp_min, d.tp_max);
    auto lk_nodes = chebyshev_nodes(backend.chebyshev_pts[2], d.lk_min, d.lk_max);

    DimensionlessAxes axes{
        .log_moneyness = x_nodes,
        .tau_prime = tp_nodes,
        .ln_kappa = lk_nodes,
    };

    // 1. PDE solve -> raw V/K
    auto pde = solve_dimensionless_pde(axes, config.spot, config.option_type);
    if (!pde.has_value()) {
        return std::unexpected(ValidationError{
            ValidationErrorCode::InvalidGridSize, 0.0});
    }

    // 2. EEP decompose via accessor
    Dimensionless3DAccessor accessor(pde->values, axes, config.spot);
    eep_decompose(accessor, AnalyticalEEP(config.option_type, 0.0));

    // 3. Fit Chebyshev on (now dollar EEP) values
    Domain<3> domain{
        .lo = {d.m_min, d.tp_min, d.lk_min},
        .hi = {d.m_max, d.tp_max, d.lk_max},
    };

    auto cheb = ChebyshevInterpolant<3, TuckerTensor<3>>::build_from_values(
        std::span<const double>(pde->values),
        domain, backend.chebyshev_pts, backend.tucker_epsilon);

    // 4. Wrap in layered PriceTable (K_ref = config.spot)
    DimensionlessTransform3D xform;
    Chebyshev3DTransformLeaf leaf(std::move(cheb), xform, config.spot);
    AnalyticalEEP eep_fn(config.option_type, 0.0);
    Chebyshev3DLeaf eep_leaf(std::move(leaf), std::move(eep_fn));

    Chebyshev3DPriceTable table(
        std::move(eep_leaf), dimless_bounds(d, backend.maturity),
        config.option_type, 0.0);

    auto solver = InterpolatedIVSolver<Chebyshev3DPriceTable>::create(
        std::move(table), config.solver_config);
    if (!solver.has_value()) return std::unexpected(solver.error());
    return make_any_solver(std::move(*solver));
}

static std::expected<AnyIVSolver, ValidationError>
build_dimensionless(const IVSolverFactoryConfig& config,
                    const DimensionlessBackend& backend) {
    // Dimensionless backend requires q=0, no discrete dividends, r > 0.
    if (std::abs(config.dividend_yield) > 1e-12) {
        return std::unexpected(ValidationError{
            ValidationErrorCode::InvalidDividend, config.dividend_yield});
    }
    if (config.discrete_dividends.has_value() &&
        !config.discrete_dividends->discrete_dividends.empty()) {
        return std::unexpected(ValidationError{
            ValidationErrorCode::InvalidDividend,
            static_cast<double>(config.discrete_dividends->discrete_dividends.size())});
    }

    // ln(2r/sigma^2) requires strictly positive rate and sigma bounds.
    auto b = extract_bounds(config.grid);
    if (b.rate_min <= 0.0) {
        return std::unexpected(ValidationError{
            ValidationErrorCode::InvalidRate, b.rate_min});
    }
    if (b.sigma_min <= 0.0) {
        return std::unexpected(ValidationError{
            ValidationErrorCode::InvalidVolatility, b.sigma_min});
    }

    if (backend.interpolant == DimensionlessBackend::Interpolant::Chebyshev)
        return build_dimensionless_chebyshev(config, backend);
    return build_dimensionless_bspline(config, backend);
}

// ---------------------------------------------------------------------------
// Public factory
// ---------------------------------------------------------------------------

std::expected<AnyIVSolver, ValidationError> make_interpolated_iv_solver(const IVSolverFactoryConfig& config) {
    const bool has_divs = config.discrete_dividends.has_value();

    return std::visit([&](const auto& backend) -> std::expected<AnyIVSolver, ValidationError> {
        using B = std::decay_t<decltype(backend)>;
        if constexpr (std::is_same_v<B, BSplineBackend>) {
            if (has_divs)
                return build_bspline_segmented(config, *config.discrete_dividends);
            else
                return build_bspline(config, backend);
        } else if constexpr (std::is_same_v<B, DimensionlessBackend>) {
            return build_dimensionless(config, backend);
        } else {
            if (has_divs)
                return build_chebyshev_segmented(config, backend, *config.discrete_dividends);
            else
                return build_chebyshev(config, backend);
        }
    }, config.backend);
}

}  // namespace mango
