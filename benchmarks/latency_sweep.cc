// SPDX-License-Identifier: MIT
/// @file latency_sweep.cc
/// @brief Comprehensive latency sweep: PDE pricing, FDM IV, surface queries, interp IV
///
/// Four sections parametrized across 9 strikes x 3 maturities, reporting both
/// latency and accuracy per scenario.
///
/// Section A: BM_PDE_Pricing     — raw PDE solve latency + accuracy
/// Section B: BM_FDM_IV          — FDM-based IV solve latency + accuracy
/// Section C: BM_Surface_Query   — 7 surface types, price query latency + accuracy
/// Section D: BM_Interp_IV       — 3 interpolated IV backends, latency + accuracy
///
/// Usage:
///   bazel run //benchmarks:latency_sweep

#include "iv_benchmark_common.hpp"
#include "mango/option/american_option.hpp"
#include "mango/option/iv_solver.hpp"
#include "mango/option/interpolated_iv_solver.hpp"
#include "mango/option/table/bspline/bspline_adaptive.hpp"
#include "mango/option/table/bspline/bspline_builder.hpp"
#include "mango/option/table/bspline/bspline_surface.hpp"
#include "mango/option/table/bspline/bspline_tensor_accessor.hpp"
#include "mango/option/table/bspline/bspline_segmented_builder.hpp"
#include "mango/option/table/bspline/bspline_3d_surface.hpp"
#include "mango/option/table/chebyshev/chebyshev_table_builder.hpp"
#include "mango/option/table/chebyshev/chebyshev_3d_surface.hpp"
#include "mango/option/table/chebyshev/chebyshev_adaptive.hpp"
#include "mango/option/table/dimensionless/dimensionless_builder.hpp"
#include "mango/option/table/dimensionless/dimensionless_3d_accessor.hpp"
#include "mango/option/table/eep/analytical_eep.hpp"
#include "mango/math/chebyshev/chebyshev_nodes.hpp"
#include "mango/math/bspline/bspline_nd_separable.hpp"
#include "mango/math/bspline/bspline_nd.hpp"
#include "mango/math/bspline/bspline_basis.hpp"
#include "mango/option/option_grid.hpp"
#include "mango/pde/core/pde_workspace.hpp"
#include <benchmark/benchmark.h>
#include <array>
#include <cmath>
#include <format>
#include <memory>
#include <memory_resource>
#include <stdexcept>
#include <vector>

using namespace mango;
using namespace mango::bench;

namespace {

// ============================================================================
// Test matrix
// ============================================================================

static constexpr double kVol = 0.20;

static constexpr size_t kNStrikes = 9;
static constexpr std::array<double, kNStrikes> kStrikes = {
    80.0, 85.0, 90.0, 95.0, 100.0, 105.0, 110.0, 115.0, 120.0
};

static constexpr size_t kNMaturities = 3;
static constexpr std::array<double, kNMaturities> kMaturities = {0.25, 1.0, 2.0};

// ============================================================================
// Utility
// ============================================================================

auto linspace(double lo, double hi, int n) {
    std::vector<double> v(n);
    for (int i = 0; i < n; ++i)
        v[i] = lo + (hi - lo) * i / (n - 1);
    return v;
}

/// Compute error in basis points.  For near-zero reference prices, use
/// absolute error normalized by strike to avoid division-by-zero.
double error_bps(double computed, double reference, double strike) {
    if (!std::isfinite(computed) || !std::isfinite(reference))
        return std::numeric_limits<double>::quiet_NaN();
    if (std::abs(reference) < 1e-8)
        return std::abs(computed - reference) * 10000.0 / strike;
    return std::abs(computed - reference) / reference * 10000.0;
}

// ============================================================================
// Reference prices (computed once via mango PDE solver)
// ============================================================================

using PriceGrid = std::array<std::array<double, kNStrikes>, kNMaturities>;

/// Discrete dividend used by segmented surfaces.
static const std::vector<Dividend> kSegDividends = {{.calendar_time = 0.25, .amount = 2.0}};

/// Solve American put for all (maturity, strike) pairs at given accuracy.
static PriceGrid
solve_prices(const GridAccuracyParams& accuracy, double q,
             const std::vector<Dividend>& discrete_divs = {}) {
    PriceGrid result;
    for (size_t ti = 0; ti < kNMaturities; ++ti) {
        for (size_t ki = 0; ki < kNStrikes; ++ki) {
            PricingParams params(
                OptionSpec{.spot = kSpot, .strike = kStrikes[ki],
                    .maturity = kMaturities[ti],
                    .rate = kRate, .dividend_yield = q,
                    .option_type = OptionType::PUT},
                kVol, discrete_divs);

            auto [grid_spec, time_domain] = estimate_pde_grid(params, accuracy);
            size_t n = grid_spec.n_points();
            std::pmr::vector<double> buffer(
                PDEWorkspace::required_size(n), std::pmr::get_default_resource());
            auto workspace = PDEWorkspace::from_buffer(buffer, n);
            if (!workspace) {
                result[ti][ki] = std::numeric_limits<double>::quiet_NaN();
                continue;
            }
            auto solver = AmericanOptionSolver::create(
                params, *workspace,
                PDEGridConfig{.grid_spec = grid_spec,
                              .n_time = time_domain.n_steps()});
            if (!solver) {
                result[ti][ki] = std::numeric_limits<double>::quiet_NaN();
                continue;
            }
            auto res = solver->solve();
            result[ti][ki] = res ? res->value_at(kSpot)
                                 : std::numeric_limits<double>::quiet_NaN();
        }
    }
    return result;
}

static constexpr GridAccuracyParams kHighAccuracy{
    .min_spatial_points = 1001, .max_spatial_points = 1201};
static constexpr GridAccuracyParams kTableAccuracy{
    .min_spatial_points = 201, .max_spatial_points = 301};

/// High-accuracy reference (q=kDivYield): surfaces 0,1,2
static const PriceGrid& get_high_accuracy_prices() {
    static auto prices = solve_prices(kHighAccuracy, kDivYield);
    return prices;
}

/// Table-accuracy reference (q=kDivYield): surfaces 0,1,2
static const PriceGrid& get_table_accuracy_prices() {
    static auto prices = solve_prices(kTableAccuracy, kDivYield);
    return prices;
}

/// High-accuracy reference (q=0): dimensionless surfaces 3,4
static const PriceGrid& get_q0_prices() {
    static auto prices = solve_prices(kHighAccuracy, 0.0);
    return prices;
}

/// High-accuracy reference with discrete dividends: segmented surfaces 5,6
static const PriceGrid& get_segmented_prices() {
    static auto prices = solve_prices(kHighAccuracy, 0.0, kSegDividends);
    return prices;
}

/// Per-surface reference price lookup.
double ref_price_for(int surf_idx, size_t ti, size_t ki) {
    if (surf_idx <= 2) return get_high_accuracy_prices()[ti][ki];
    if (surf_idx <= 4) return get_q0_prices()[ti][ki];
    return get_segmented_prices()[ti][ki];
}

/// Per-surface table-accuracy reference (only for standard 4D).
double table_price_for(int surf_idx, size_t ti, size_t ki) {
    if (surf_idx <= 2) return get_table_accuracy_prices()[ti][ki];
    return std::numeric_limits<double>::quiet_NaN();  // no table-accuracy for q=0/segmented
}

// ============================================================================
// Section A: BM_PDE_Pricing
// ============================================================================

static void BM_PDE_Pricing(benchmark::State& state) {
    size_t ki = static_cast<size_t>(state.range(0));
    size_t ti = static_cast<size_t>(state.range(1));
    double K = kStrikes[ki];
    double T = kMaturities[ti];
    double ref_price = get_high_accuracy_prices()[ti][ki];

    PricingParams params(
        OptionSpec{.spot = kSpot, .strike = K, .maturity = T,
            .rate = kRate, .dividend_yield = kDivYield,
            .option_type = OptionType::PUT},
        kVol);

    auto [grid_spec, time_domain] = estimate_pde_grid(params);
    size_t n = grid_spec.n_points();
    size_t nt = time_domain.n_steps();

    // Pre-allocate workspace outside timing loop
    std::pmr::vector<double> buffer(
        PDEWorkspace::required_size(n), std::pmr::get_default_resource());

    double price = 0.0;
    for (auto _ : state) {
        auto workspace = PDEWorkspace::from_buffer(buffer, n);
        auto solver = AmericanOptionSolver::create(
            params, *workspace,
            PDEGridConfig{.grid_spec = grid_spec, .n_time = nt});
        auto result = solver->solve();
        price = result ? result->value_at(kSpot) : 0.0;
        benchmark::DoNotOptimize(price);
    }

    state.SetLabel(std::format("K={:.0f} T={:.2f}", K, T));
    state.counters["strike"] = K;
    state.counters["maturity"] = T;
    state.counters["moneyness"] = kSpot / K;
    state.counters["n_space"] = static_cast<double>(n);
    state.counters["n_time"] = static_cast<double>(nt);
    state.counters["price"] = price;
    state.counters["price_err_bps"] = error_bps(price, ref_price, K);
}

BENCHMARK(BM_PDE_Pricing)
    ->ArgsProduct({
        benchmark::CreateDenseRange(0, 8, 1),
        benchmark::CreateDenseRange(0, 2, 1),
    })
    ->Unit(benchmark::kMillisecond);

// ============================================================================
// Section B: BM_FDM_IV
// ============================================================================

static void BM_FDM_IV(benchmark::State& state) {
    size_t ki = static_cast<size_t>(state.range(0));
    size_t ti = static_cast<size_t>(state.range(1));
    double K = kStrikes[ki];
    double T = kMaturities[ti];
    double ref_price = get_high_accuracy_prices()[ti][ki];

    if (!std::isfinite(ref_price)) {
        state.SkipWithError("Reference price not available");
        return;
    }

    OptionSpec spec{
        .spot = kSpot, .strike = K, .maturity = T,
        .rate = kRate, .dividend_yield = kDivYield,
        .option_type = OptionType::PUT
    };
    IVQuery query(spec, ref_price);

    IVSolverConfig config;
    IVSolver solver(config);

    std::expected<IVSuccess, IVError> last_result;
    for (auto _ : state) {
        last_result = solver.solve(query);
        if (!last_result) {
            state.SkipWithError("FDM IV solve failed");
            return;
        }
        benchmark::DoNotOptimize(last_result);
    }

    double iv = last_result->implied_vol;
    double iv_err_bps = std::abs(iv - kVol) * 10000.0;

    state.SetLabel(std::format("K={:.0f} T={:.2f}", K, T));
    state.counters["strike"] = K;
    state.counters["maturity"] = T;
    state.counters["iv"] = iv;
    state.counters["iv_err_bps"] = iv_err_bps;
    state.counters["iters"] = static_cast<double>(last_result->iterations);
}

BENCHMARK(BM_FDM_IV)
    ->ArgsProduct({
        benchmark::CreateDenseRange(0, 8, 1),
        benchmark::CreateDenseRange(0, 2, 1),
    })
    ->Unit(benchmark::kMillisecond);

// ============================================================================
// Section C: Surface builders (static, built once)
// ============================================================================

// Surface 0: B-spline 4D (adaptive grid, ~2 bps target)
const BSplinePriceTable& GetBSpline4D() {
    static BSplinePriceTable* surface = [] {
        OptionGrid chain;
        chain.spot = kSpot;
        chain.dividend_yield = kDivYield;
        // Match interp_iv_safety.cc factory pattern: S/K moneyness → strikes
        for (double m : {0.70, 0.80, 0.90, 1.00, 1.10, 1.20, 1.30})
            chain.strikes.push_back(kSpot / m);
        chain.maturities = {0.01, 0.03, 0.06, 0.12, 0.20,
                            0.35, 0.60, 1.0, 1.5, 2.0, 2.5};
        chain.implied_vols = {0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50};
        chain.rates = {0.01, 0.03, 0.05, 0.10};

        AdaptiveGridParams params{.target_iv_error = 2e-5};  // 2 bps
        auto result = build_adaptive_bspline(
            params, chain,
            make_grid_accuracy(GridAccuracyProfile::High), OptionType::PUT);
        if (!result) throw std::runtime_error("BSpline4D adaptive: build failed");

        auto wrapper = make_bspline_surface(
            result->spline, result->K_ref, result->dividend_yield,
            OptionType::PUT);
        if (!wrapper) throw std::runtime_error("BSpline4D: make_bspline_surface failed");

        return new BSplinePriceTable(std::move(*wrapper));
    }();
    return *surface;
}

// Surface 1: Chebyshev 4D raw (adaptive CC-level refinement)
const ChebyshevRawSurface& GetChebyshev4DRaw() {
    static ChebyshevRawSurface* surface = [] {
        OptionGrid chain;
        chain.spot = kSpot;
        chain.dividend_yield = kDivYield;
        for (double m : {0.70, 0.80, 0.90, 1.00, 1.10, 1.20, 1.30})
            chain.strikes.push_back(kSpot / m);
        chain.maturities = {0.01, 0.03, 0.06, 0.12, 0.20,
                            0.35, 0.60, 1.0, 1.5, 2.0, 2.5};
        chain.implied_vols = {0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50};
        chain.rates = {0.01, 0.03, 0.05, 0.10};

        AdaptiveGridParams params{.target_iv_error = 5e-4};  // 50 bps (CC refinement)
        auto result = build_adaptive_chebyshev(params, chain, OptionType::PUT);
        if (!result) throw std::runtime_error("Chebyshev4D adaptive: build failed");

        return new ChebyshevRawSurface(std::move(*result->surface));
    }();
    return *surface;
}

// Surface 2: Chebyshev 4D (Tucker compressed)
const ChebyshevTableResult& GetChebyshev4DTucker() {
    static ChebyshevTableResult* surface = [] {
        ChebyshevTableConfig config{
            .num_pts = {12, 8, 8, 5},
            .domain = Domain<4>{
                .lo = {-0.50, 0.02, 0.05, 0.00},
                .hi = { 0.50, 2.50, 0.50, 0.12},
            },
            .K_ref = 100.0,
            .option_type = OptionType::PUT,
            .dividend_yield = kDivYield,
            .tucker_epsilon = 1e-6,
        };
        auto result = build_chebyshev_table(config);
        if (!result) throw std::runtime_error("Chebyshev4D Tucker: build failed");
        return new ChebyshevTableResult(std::move(*result));
    }();
    return *surface;
}

// Surface 3: B-spline 3D dimensionless (q=0)
const BSpline3DPriceTable& GetDimensionless3D() {
    static BSpline3DPriceTable* surface = [] {
        constexpr double K_ref = 100.0;

        DimensionlessAxes axes;
        axes.log_moneyness = {-0.50, -0.30, -0.20, -0.10, -0.05,
                               0.0, 0.05, 0.10, 0.20, 0.30, 0.50};
        axes.tau_prime = {0.0005, 0.001, 0.005, 0.01, 0.02,
                          0.04, 0.06, 0.08, 0.12, 0.16};
        axes.ln_kappa = {-2.5, -1.5, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 2.8};

        auto pde = solve_dimensionless_pde(axes, K_ref, OptionType::PUT);
        if (!pde) throw std::runtime_error("Dimensionless3D: PDE solve failed");

        Dimensionless3DAccessor accessor(pde->values, axes, K_ref);
        eep_decompose(accessor, AnalyticalEEP(OptionType::PUT, 0.0));

        std::array<std::vector<double>, 3> grids = {
            axes.log_moneyness, axes.tau_prime, axes.ln_kappa};
        auto fitter = BSplineNDSeparable<double, 3>::create(grids);
        if (!fitter) throw std::runtime_error("Dimensionless3D: fitter create failed");
        auto fit = fitter->fit(std::move(pde->values));
        if (!fit) throw std::runtime_error("Dimensionless3D: fit failed");

        std::array<std::vector<double>, 3> bspline_knots;
        for (size_t i = 0; i < 3; ++i) {
            bspline_knots[i] = clamped_knots_cubic(grids[i]);
        }
        auto spline = BSplineND<double, 3>::create(
            grids, std::move(bspline_knots), std::move(fit->coefficients));
        if (!spline) throw std::runtime_error("Dimensionless3D: spline create failed");

        auto spline_ptr = std::make_shared<const BSplineND<double, 3>>(
            std::move(spline.value()));

        SharedBSplineInterp<3> interp(std::move(spline_ptr));
        DimensionlessTransform3D xform;
        BSpline3DTransformLeaf leaf(std::move(interp), xform, K_ref);
        AnalyticalEEP eep(OptionType::PUT, 0.0);
        BSpline3DLeaf eep_leaf(std::move(leaf), std::move(eep));

        constexpr double sigma_min = 0.05, sigma_max = 0.80;
        SurfaceBounds bounds{
            .m_min = axes.log_moneyness.front(),
            .m_max = axes.log_moneyness.back(),
            .tau_min = 2.0 * axes.tau_prime.front() / (sigma_max * sigma_max),
            .tau_max = 2.0 * axes.tau_prime.back() / (sigma_min * sigma_min),
            .sigma_min = sigma_min,
            .sigma_max = sigma_max,
            .rate_min = 0.005,
            .rate_max = 0.10,
        };

        return new BSpline3DPriceTable(
            std::move(eep_leaf), bounds, OptionType::PUT, 0.0);
    }();
    return *surface;
}

// Surface 4: Chebyshev 3D (Tucker, dimensionless, q=0)
const Chebyshev3DPriceTable& GetChebyshev3D() {
    static Chebyshev3DPriceTable* surface = [] {
        constexpr double K_ref = 100.0;
        constexpr std::array<size_t, 3> num_pts = {14, 12, 10};

        constexpr double m_min = -0.50, m_max = 0.50;
        constexpr double tp_min = 0.0005, tp_max = 0.16;
        constexpr double lk_min = -2.5, lk_max = 2.8;

        auto x_nodes  = chebyshev_nodes(num_pts[0], m_min, m_max);
        auto tp_nodes = chebyshev_nodes(num_pts[1], tp_min, tp_max);
        auto lk_nodes = chebyshev_nodes(num_pts[2], lk_min, lk_max);

        DimensionlessAxes axes{
            .log_moneyness = x_nodes,
            .tau_prime = tp_nodes,
            .ln_kappa = lk_nodes,
        };

        auto pde = solve_dimensionless_pde(axes, K_ref, OptionType::PUT);
        if (!pde) throw std::runtime_error("Chebyshev3D: PDE solve failed");

        Dimensionless3DAccessor accessor(pde->values, axes, K_ref);
        eep_decompose(accessor, AnalyticalEEP(OptionType::PUT, 0.0));

        Domain<3> domain{
            .lo = {m_min, tp_min, lk_min},
            .hi = {m_max, tp_max, lk_max},
        };

        // tucker_epsilon=0 avoids AVX-512 alignment bug
        auto cheb = ChebyshevInterpolant<3, TuckerTensor<3>>::build_from_values(
            std::span<const double>(pde->values), domain, num_pts, 0.0);

        DimensionlessTransform3D xform;
        Chebyshev3DTransformLeaf leaf(std::move(cheb), xform, K_ref);
        AnalyticalEEP eep(OptionType::PUT, 0.0);
        Chebyshev3DLeaf eep_leaf(std::move(leaf), std::move(eep));

        constexpr double sigma_min = 0.05, sigma_max = 0.80;
        SurfaceBounds bounds{
            .m_min = m_min, .m_max = m_max,
            .tau_min = 2.0 * tp_min / (sigma_max * sigma_max),
            .tau_max = 2.0 * tp_max / (sigma_min * sigma_min),
            .sigma_min = sigma_min, .sigma_max = sigma_max,
            .rate_min = 0.005, .rate_max = 0.10,
        };

        return new Chebyshev3DPriceTable(
            std::move(eep_leaf), bounds, OptionType::PUT, 0.0);
    }();
    return *surface;
}

// Surface 5: Segmented B-spline (discrete dividends, maturity=2.5)
const BSplineSegmentedSurface& GetSegmented() {
    static BSplineSegmentedSurface* surface = [] {
        auto log_m = [](std::initializer_list<double> ms) {
            std::vector<double> out;
            for (double m : ms) out.push_back(std::log(m));
            return out;
        };

        SegmentedPriceTableBuilder::Config config{
            .K_ref = 100.0,
            .option_type = OptionType::PUT,
            .dividends = {
                .dividend_yield = 0.0,
                .discrete_dividends = {{.calendar_time = 0.25, .amount = 2.0}},
            },
            .grid = IVGrid{
                .moneyness = log_m({0.70, 0.80, 0.85, 0.90, 0.95, 1.00,
                                    1.05, 1.10, 1.15, 1.20, 1.30}),
                .vol = {0.08, 0.15, 0.20, 0.30, 0.40, 0.50},
                .rate = {0.00, 0.03, 0.05, 0.07, 0.10},
            },
            .maturity = 2.5,
        };

        auto result = SegmentedPriceTableBuilder::build(config);
        if (!result) throw std::runtime_error("Segmented: build failed");
        return new BSplineSegmentedSurface(std::move(*result));
    }();
    return *surface;
}

// Surface 6: Segmented Chebyshev (discrete dividends, maturity=2.5)
const ChebyshevMultiKRefSurface& GetChebSegmented() {
    static ChebyshevMultiKRefSurface* surface = [] {
        auto log_m = [](std::initializer_list<double> ms) {
            std::vector<double> out;
            for (double m : ms) out.push_back(std::log(m));
            return out;
        };

        SegmentedAdaptiveConfig seg_config{
            .spot = 100.0,
            .option_type = OptionType::PUT,
            .dividend_yield = 0.0,
            .discrete_dividends = {{.calendar_time = 0.25, .amount = 2.0}},
            .maturity = 2.5,
            .kref_config = {.K_refs = {80.0, 100.0, 120.0}},
        };

        IVGrid grid{
            .moneyness = log_m({0.70, 0.80, 0.85, 0.90, 0.95, 1.00,
                                1.05, 1.10, 1.15, 1.20, 1.30}),
            .vol = {0.08, 0.15, 0.20, 0.30, 0.40, 0.50},
            .rate = {0.00, 0.03, 0.05, 0.07, 0.10},
        };

        auto result = build_chebyshev_segmented_manual(seg_config, grid);
        if (!result) throw std::runtime_error("ChebSegmented: build failed");
        return new ChebyshevMultiKRefSurface(std::move(*result));
    }();
    return *surface;
}

// ============================================================================
// Section C: BM_Surface_Query
// ============================================================================

static constexpr size_t kNSurfaces = 7;

static const char* kSurfaceNames[] = {
    "bspline4d", "cheb4d_raw", "cheb4d_tucker",
    "bspline3d", "cheb3d", "seg_bspline", "seg_cheb"
};

/// Query price from surface by index.
/// For 3D dimensionless surfaces (idx 3,4): q=0, so we pass q=0 implicitly
/// (the surface was built with q=0). The surface .price() handles it.
/// For segmented surfaces (idx 5,6): built with discrete dividends (q=0).
double query_price(int surf_idx, double S, double K, double tau,
                   double sigma, double r) {
    switch (surf_idx) {
        case 0: return GetBSpline4D().price(S, K, tau, sigma, r);
        case 1: return GetChebyshev4DRaw().price(S, K, tau, sigma, r);
        case 2: return GetChebyshev4DTucker().price(S, K, tau, sigma, r);
        case 3: return GetDimensionless3D().price(S, K, tau, sigma, r);
        case 4: return GetChebyshev3D().price(S, K, tau, sigma, r);
        case 5: return GetSegmented().price(S, K, tau, sigma, r);
        case 6: return GetChebSegmented().price(S, K, tau, sigma, r);
        default: return std::numeric_limits<double>::quiet_NaN();
    }
}

static void BM_Surface_Query(benchmark::State& state) {
    int surf_idx = static_cast<int>(state.range(0));
    size_t ki = static_cast<size_t>(state.range(1));
    size_t ti = static_cast<size_t>(state.range(2));
    double K = kStrikes[ki];
    double T = kMaturities[ti];
    double ref_price = ref_price_for(surf_idx, ti, ki);
    double table_price = table_price_for(surf_idx, ti, ki);

    double price = 0.0;
    for (auto _ : state) {
        price = query_price(surf_idx, kSpot, K, T, kVol, kRate);
        benchmark::DoNotOptimize(price);
    }

    // Time value / strike: measures how far from intrinsic the price is.
    // Low TV/K (<1e-4) means the point is deep OTM/ITM where interpolation
    // errors dominate and IV recovery is unreliable.
    double intrinsic = std::max(K - kSpot, 0.0);  // put intrinsic
    double tv_per_k = (ref_price - intrinsic) / K;

    state.SetLabel(std::format("{} K={:.0f} T={:.2f}",
                               kSurfaceNames[surf_idx], K, T));
    state.counters["strike"] = K;
    state.counters["maturity"] = T;
    state.counters["price"] = price;
    state.counters["price_err_bps"] = error_bps(price, ref_price, K);
    state.counters["interp_err_bps"] = error_bps(price, table_price, K);
    state.counters["tv_per_k"] = tv_per_k;
}

BENCHMARK(BM_Surface_Query)
    ->ArgsProduct({
        benchmark::CreateDenseRange(0, static_cast<int>(kNSurfaces) - 1, 1),
        benchmark::CreateDenseRange(0, 8, 1),
        benchmark::CreateDenseRange(0, 2, 1),
    })
    ->Unit(benchmark::kNanosecond);

// ============================================================================
// Section D: BM_Interp_IV — interpolated IV solvers
// ============================================================================

// Backend 0: BSpline 4D
const InterpolatedIVSolver<BSplinePriceTable>& GetBSpline4DIVSolver() {
    static auto* solver = [] {
        auto wrapper = GetBSpline4D();  // copies surface for ownership
        auto s = InterpolatedIVSolver<BSplinePriceTable>::create(std::move(wrapper));
        if (!s) throw std::runtime_error("BSpline4D IV solver: create failed");
        return new InterpolatedIVSolver<BSplinePriceTable>(std::move(*s));
    }();
    return *solver;
}

// Backend 1: Chebyshev 4D raw
const InterpolatedIVSolver<ChebyshevRawSurface>& GetChebyshev4DIVSolver() {
    static auto* solver = [] {
        auto s = InterpolatedIVSolver<ChebyshevRawSurface>::create(GetChebyshev4DRaw());
        if (!s) throw std::runtime_error("Chebyshev4D raw IV solver: create failed");
        return new InterpolatedIVSolver<ChebyshevRawSurface>(std::move(*s));
    }();
    return *solver;
}

// Backend 2: BSpline 3D dimensionless (via factory, q=0 required)
const AnyInterpIVSolver& GetDimensionless3DIVSolver() {
    static auto* solver = [] {
        IVSolverFactoryConfig config{
            .option_type = OptionType::PUT,
            .spot = kSpot,
            .dividend_yield = 0.0,  // dimensionless surfaces require q=0
            .grid = IVGrid{
                .moneyness = {0.70, 0.80, 0.90, 1.00, 1.10, 1.20, 1.30},
                .vol = {0.05, 0.10, 0.20, 0.30, 0.50},
                .rate = {0.01, 0.03, 0.05, 0.10},
            },
            .backend = DimensionlessBackend{.maturity = 2.5},
        };
        auto s = make_interpolated_iv_solver(config);
        if (!s) throw std::runtime_error("Dimensionless3D IV solver: factory failed");
        return new AnyInterpIVSolver(std::move(*s));
    }();
    return *solver;
}

static constexpr size_t kNIVBackends = 3;

static const char* kIVBackendNames[] = {
    "bspline4d", "cheb4d_raw", "bspline3d"
};

std::expected<IVSuccess, IVError>
solve_interp_iv(int backend_idx, const IVQuery& query) {
    switch (backend_idx) {
        case 0: return GetBSpline4DIVSolver().solve(query);
        case 1: return GetChebyshev4DIVSolver().solve(query);
        case 2: return GetDimensionless3DIVSolver().solve(query);
        default:
            return std::unexpected(IVError{
                .code = IVErrorCode::NumericalInstability,
                .iterations = 0, .final_error = 0.0, .last_vol = std::nullopt
            });
    }
}

/// Dividend yield per IV backend.
static constexpr double kBackendQ[] = {kDivYield, kDivYield, 0.0};

static void BM_Interp_IV(benchmark::State& state) {
    int backend_idx = static_cast<int>(state.range(0));
    size_t ki = static_cast<size_t>(state.range(1));
    size_t ti = static_cast<size_t>(state.range(2));
    double K = kStrikes[ki];
    double T = kMaturities[ti];

    // Use matching dividend yield for reference and query
    double q = kBackendQ[backend_idx];
    double ref_price = (q == kDivYield) ? get_high_accuracy_prices()[ti][ki]
                                        : get_q0_prices()[ti][ki];
    double table_price = (q == kDivYield) ? get_table_accuracy_prices()[ti][ki]
                                          : std::numeric_limits<double>::quiet_NaN();

    if (!std::isfinite(ref_price)) {
        state.SkipWithError("Reference price not available");
        return;
    }

    OptionSpec spec{
        .spot = kSpot, .strike = K, .maturity = T,
        .rate = kRate, .dividend_yield = q,
        .option_type = OptionType::PUT
    };
    IVQuery query(spec, ref_price);

    std::expected<IVSuccess, IVError> last_result;
    for (auto _ : state) {
        last_result = solve_interp_iv(backend_idx, query);
        if (!last_result) {
            state.SkipWithError("Interp IV solve failed");
            return;
        }
        benchmark::DoNotOptimize(last_result);
    }

    double iv = last_result->implied_vol;
    double iv_err_bps = std::abs(iv - kVol) * 10000.0;

    // Interpolation-only IV error: recover IV from table-accuracy price
    double interp_iv_err_bps = std::numeric_limits<double>::quiet_NaN();
    if (std::isfinite(table_price)) {
        IVQuery table_query(spec, table_price);
        auto interp_result = solve_interp_iv(backend_idx, table_query);
        if (interp_result) {
            interp_iv_err_bps = std::abs(interp_result->implied_vol - kVol) * 10000.0;
        }
    }

    state.SetLabel(std::format("{} K={:.0f} T={:.2f}",
                               kIVBackendNames[backend_idx], K, T));
    state.counters["solver"] = backend_idx;
    state.counters["strike"] = K;
    state.counters["maturity"] = T;
    state.counters["iv"] = iv;
    state.counters["iv_err_bps"] = iv_err_bps;
    state.counters["interp_iv_err_bps"] = interp_iv_err_bps;
    state.counters["iters"] = static_cast<double>(last_result->iterations);
}

BENCHMARK(BM_Interp_IV)
    ->ArgsProduct({
        benchmark::CreateDenseRange(0, static_cast<int>(kNIVBackends) - 1, 1),
        benchmark::CreateDenseRange(0, 8, 1),
        benchmark::CreateDenseRange(0, 2, 1),
    })
    ->Unit(benchmark::kMicrosecond);

}  // namespace

int main(int argc, char** argv) {
    // Pre-initialize all static surfaces before benchmarks run.
    // Without this, the first benchmark case for each surface includes
    // the build time (seconds of PDE solves) in the measurement.
    (void)GetBSpline4D();
    (void)GetChebyshev4DRaw();
    (void)GetChebyshev4DTucker();
    (void)GetDimensionless3D();
    (void)GetChebyshev3D();
    (void)GetSegmented();
    (void)GetChebSegmented();

    // Pre-initialize reference price caches
    (void)get_high_accuracy_prices();
    (void)get_table_accuracy_prices();
    (void)get_q0_prices();
    (void)get_segmented_prices();

    // Pre-initialize IV solvers
    (void)GetBSpline4DIVSolver();
    (void)GetChebyshev4DIVSolver();
    (void)GetDimensionless3DIVSolver();

    benchmark::Initialize(&argc, argv);
    benchmark::RunSpecifiedBenchmarks();
    benchmark::Shutdown();
    return 0;
}
