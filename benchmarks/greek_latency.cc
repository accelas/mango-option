// SPDX-License-Identifier: MIT
/// @file greek_latency.cc
/// @brief Latency benchmark: per-query time for price, vega, delta, gamma, theta, rho
///
/// Builds each surface type once, then benchmarks each Greek query
/// individually on a single ATM point. Reports ns/query.
///
/// Usage:
///   bazel run //benchmarks:greek_latency

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
#include "mango/option/table/greek_types.hpp"
#include "mango/math/chebyshev/chebyshev_nodes.hpp"
#include "mango/math/bspline/bspline_nd_separable.hpp"
#include "mango/math/bspline/bspline_nd.hpp"
#include "mango/math/bspline/bspline_basis.hpp"
#include "mango/option/option_spec.hpp"
#include <benchmark/benchmark.h>
#include <cmath>
#include <memory>
#include <stdexcept>

using namespace mango;

namespace {

auto linspace(double lo, double hi, int n) {
    std::vector<double> v(n);
    for (int i = 0; i < n; ++i)
        v[i] = lo + (hi - lo) * i / (n - 1);
    return v;
}

// Shared query point
constexpr double S = 100.0, K = 100.0, tau = 0.5, sigma = 0.20, rate = 0.05;

PricingParams MakeParams(double q = 0.02) {
    return PricingParams(
        OptionSpec{.spot = S, .strike = K, .maturity = tau,
            .rate = rate, .dividend_yield = q,
            .option_type = OptionType::PUT},
        sigma);
}

// ===========================================================================
// Surface 1: B-spline 4D (StandardTransform4D)
// ===========================================================================

const BSplinePriceTable& GetBSpline4D() {
    static BSplinePriceTable* surface = [] {
        auto m_grid    = linspace(-0.40, 0.40, 15);  // log-moneyness ln(S/K)
        auto tau_grid  = linspace(0.05, 2.50, 10);
        auto vol_grid  = linspace(0.08, 0.50, 8);
        auto rate_grid = linspace(0.00, 0.12, 6);

        constexpr double K_ref = 100.0;
        constexpr double q = 0.02;

        auto setup = PriceTableBuilder::from_vectors(
            m_grid, tau_grid, vol_grid, rate_grid, K_ref,
            GridAccuracyParams{}, OptionType::PUT, q);
        if (!setup) throw std::runtime_error("BSpline4D: from_vectors failed");

        auto [builder, axes] = std::move(*setup);
        auto table = builder.build(axes,
            [&](PriceTensor& tensor, const PriceTableAxes& a) {
                BSplineTensorAccessor accessor(tensor, a, K_ref);
                eep_decompose(accessor, AnalyticalEEP(OptionType::PUT, q));
            });
        if (!table) throw std::runtime_error("BSpline4D: build failed");

        auto wrapper = make_bspline_surface(table->spline, table->K_ref, q, OptionType::PUT);
        if (!wrapper) throw std::runtime_error("BSpline4D: make_bspline_surface failed");

        return new BSplinePriceTable(std::move(*wrapper));
    }();
    return *surface;
}

static void BM_BSpline4D_Price(benchmark::State& state) {
    const auto& surf = GetBSpline4D();
    for (auto _ : state) {
        benchmark::DoNotOptimize(surf.price(S, K, tau, sigma, rate));
    }
}
BENCHMARK(BM_BSpline4D_Price);

static void BM_BSpline4D_Vega(benchmark::State& state) {
    const auto& surf = GetBSpline4D();
    for (auto _ : state) {
        benchmark::DoNotOptimize(surf.vega(S, K, tau, sigma, rate));
    }
}
BENCHMARK(BM_BSpline4D_Vega);

static void BM_BSpline4D_Delta(benchmark::State& state) {
    const auto& surf = GetBSpline4D();
    auto params = MakeParams();
    for (auto _ : state) {
        benchmark::DoNotOptimize(surf.delta(params));
    }
}
BENCHMARK(BM_BSpline4D_Delta);

static void BM_BSpline4D_Gamma(benchmark::State& state) {
    const auto& surf = GetBSpline4D();
    auto params = MakeParams();
    for (auto _ : state) {
        benchmark::DoNotOptimize(surf.gamma(params));
    }
}
BENCHMARK(BM_BSpline4D_Gamma);

static void BM_BSpline4D_Theta(benchmark::State& state) {
    const auto& surf = GetBSpline4D();
    auto params = MakeParams();
    for (auto _ : state) {
        benchmark::DoNotOptimize(surf.theta(params));
    }
}
BENCHMARK(BM_BSpline4D_Theta);

static void BM_BSpline4D_Rho(benchmark::State& state) {
    const auto& surf = GetBSpline4D();
    auto params = MakeParams();
    for (auto _ : state) {
        benchmark::DoNotOptimize(surf.rho(params));
    }
}
BENCHMARK(BM_BSpline4D_Rho);

static void BM_BSpline4D_All(benchmark::State& state) {
    const auto& surf = GetBSpline4D();
    auto params = MakeParams();
    for (auto _ : state) {
        benchmark::DoNotOptimize(surf.price(S, K, tau, sigma, rate));
        benchmark::DoNotOptimize(surf.vega(S, K, tau, sigma, rate));
        benchmark::DoNotOptimize(surf.delta(params));
        benchmark::DoNotOptimize(surf.gamma(params));
        benchmark::DoNotOptimize(surf.theta(params));
        benchmark::DoNotOptimize(surf.rho(params));
    }
}
BENCHMARK(BM_BSpline4D_All);

// ===========================================================================
// Surface 2: Chebyshev 4D (FD fallback for gamma)
// ===========================================================================

const ChebyshevTableResult& GetChebyshev4D() {
    static ChebyshevTableResult* surface = [] {
        ChebyshevTableConfig config{
            .num_pts = {12, 8, 8, 5},
            .domain = Domain<4>{
                .lo = {-0.30, 0.02, 0.05, 0.01},
                .hi = { 0.30, 2.00, 0.50, 0.10},
            },
            .K_ref = 100.0,
            .option_type = OptionType::PUT,
            .dividend_yield = 0.02,
        };
        auto result = build_chebyshev_table(config);
        if (!result) throw std::runtime_error("Chebyshev4D: build failed");
        return new ChebyshevTableResult(std::move(*result));
    }();
    return *surface;
}

static void BM_Chebyshev4D_Price(benchmark::State& state) {
    const auto& surf = GetChebyshev4D();
    for (auto _ : state) {
        benchmark::DoNotOptimize(surf.price(S, K, tau, sigma, rate));
    }
}
BENCHMARK(BM_Chebyshev4D_Price);

static void BM_Chebyshev4D_Vega(benchmark::State& state) {
    const auto& surf = GetChebyshev4D();
    for (auto _ : state) {
        benchmark::DoNotOptimize(surf.vega(S, K, tau, sigma, rate));
    }
}
BENCHMARK(BM_Chebyshev4D_Vega);

static void BM_Chebyshev4D_Delta(benchmark::State& state) {
    const auto& surf = GetChebyshev4D();
    auto params = MakeParams();
    for (auto _ : state) {
        benchmark::DoNotOptimize(surf.delta(params));
    }
}
BENCHMARK(BM_Chebyshev4D_Delta);

static void BM_Chebyshev4D_Gamma(benchmark::State& state) {
    const auto& surf = GetChebyshev4D();
    auto params = MakeParams();
    for (auto _ : state) {
        benchmark::DoNotOptimize(surf.gamma(params));
    }
}
BENCHMARK(BM_Chebyshev4D_Gamma);

static void BM_Chebyshev4D_Theta(benchmark::State& state) {
    const auto& surf = GetChebyshev4D();
    auto params = MakeParams();
    for (auto _ : state) {
        benchmark::DoNotOptimize(surf.theta(params));
    }
}
BENCHMARK(BM_Chebyshev4D_Theta);

static void BM_Chebyshev4D_Rho(benchmark::State& state) {
    const auto& surf = GetChebyshev4D();
    auto params = MakeParams();
    for (auto _ : state) {
        benchmark::DoNotOptimize(surf.rho(params));
    }
}
BENCHMARK(BM_Chebyshev4D_Rho);

static void BM_Chebyshev4D_All(benchmark::State& state) {
    const auto& surf = GetChebyshev4D();
    auto params = MakeParams();
    for (auto _ : state) {
        benchmark::DoNotOptimize(surf.price(S, K, tau, sigma, rate));
        benchmark::DoNotOptimize(surf.vega(S, K, tau, sigma, rate));
        benchmark::DoNotOptimize(surf.delta(params));
        benchmark::DoNotOptimize(surf.gamma(params));
        benchmark::DoNotOptimize(surf.theta(params));
        benchmark::DoNotOptimize(surf.rho(params));
    }
}
BENCHMARK(BM_Chebyshev4D_All);

// ===========================================================================
// Surface 3: Dimensionless 3D B-spline (DimensionlessTransform3D)
// ===========================================================================

const BSpline3DPriceTable& GetDimensionless3D() {
    static BSpline3DPriceTable* surface = [] {
        constexpr double K_ref = 100.0;

        DimensionlessAxes axes;
        axes.log_moneyness = {-0.30, -0.20, -0.10, -0.05, 0.0, 0.05, 0.10, 0.20, 0.30};
        axes.tau_prime = {0.001, 0.005, 0.01, 0.02, 0.04, 0.06, 0.08, 0.12, 0.16};
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

        std::array<std::vector<double>, 3> bspline_grids = {
            axes.log_moneyness, axes.tau_prime, axes.ln_kappa};
        std::array<std::vector<double>, 3> bspline_knots;
        for (size_t i = 0; i < 3; ++i) {
            bspline_knots[i] = clamped_knots_cubic(bspline_grids[i]);
        }
        auto spline = BSplineND<double, 3>::create(
            bspline_grids, std::move(bspline_knots), std::move(fit->coefficients));
        if (!spline) throw std::runtime_error("Dimensionless3D: spline create failed");

        auto spline_ptr = std::make_shared<const BSplineND<double, 3>>(
            std::move(spline.value()));

        SharedBSplineInterp<3> interp(std::move(spline_ptr));
        DimensionlessTransform3D xform;
        BSpline3DTransformLeaf leaf(std::move(interp), xform, K_ref);
        AnalyticalEEP eep(OptionType::PUT, 0.0);
        BSpline3DLeaf eep_leaf(std::move(leaf), std::move(eep));

        constexpr double sigma_min = 0.10, sigma_max = 0.80;
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

static void BM_Dim3D_Price(benchmark::State& state) {
    const auto& surf = GetDimensionless3D();
    for (auto _ : state) {
        benchmark::DoNotOptimize(surf.price(S, K, tau, sigma, rate));
    }
}
BENCHMARK(BM_Dim3D_Price);

static void BM_Dim3D_Vega(benchmark::State& state) {
    const auto& surf = GetDimensionless3D();
    for (auto _ : state) {
        benchmark::DoNotOptimize(surf.vega(S, K, tau, sigma, rate));
    }
}
BENCHMARK(BM_Dim3D_Vega);

static void BM_Dim3D_Delta(benchmark::State& state) {
    const auto& surf = GetDimensionless3D();
    auto params = MakeParams(0.0);  // 3D surface has q=0
    for (auto _ : state) {
        benchmark::DoNotOptimize(surf.delta(params));
    }
}
BENCHMARK(BM_Dim3D_Delta);

static void BM_Dim3D_Gamma(benchmark::State& state) {
    const auto& surf = GetDimensionless3D();
    auto params = MakeParams(0.0);
    for (auto _ : state) {
        benchmark::DoNotOptimize(surf.gamma(params));
    }
}
BENCHMARK(BM_Dim3D_Gamma);

static void BM_Dim3D_Theta(benchmark::State& state) {
    const auto& surf = GetDimensionless3D();
    auto params = MakeParams(0.0);
    for (auto _ : state) {
        benchmark::DoNotOptimize(surf.theta(params));
    }
}
BENCHMARK(BM_Dim3D_Theta);

static void BM_Dim3D_Rho(benchmark::State& state) {
    const auto& surf = GetDimensionless3D();
    auto params = MakeParams(0.0);
    for (auto _ : state) {
        benchmark::DoNotOptimize(surf.rho(params));
    }
}
BENCHMARK(BM_Dim3D_Rho);

static void BM_Dim3D_All(benchmark::State& state) {
    const auto& surf = GetDimensionless3D();
    auto params = MakeParams(0.0);
    for (auto _ : state) {
        benchmark::DoNotOptimize(surf.price(S, K, tau, sigma, rate));
        benchmark::DoNotOptimize(surf.vega(S, K, tau, sigma, rate));
        benchmark::DoNotOptimize(surf.delta(params));
        benchmark::DoNotOptimize(surf.gamma(params));
        benchmark::DoNotOptimize(surf.theta(params));
        benchmark::DoNotOptimize(surf.rho(params));
    }
}
BENCHMARK(BM_Dim3D_All);

// ===========================================================================
// Surface 4: Segmented B-spline (discrete dividends, SplitSurface routing)
// ===========================================================================

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
                .moneyness = log_m({0.80, 0.85, 0.90, 0.95, 1.00,
                                    1.05, 1.10, 1.15, 1.20}),
                .vol = {0.15, 0.20, 0.25, 0.30, 0.40},
                .rate = {0.03, 0.05, 0.07, 0.09},
            },
            .maturity = 1.0,
        };

        auto result = SegmentedPriceTableBuilder::build(config);
        if (!result) throw std::runtime_error("Segmented: build failed");
        return new BSplineSegmentedSurface(std::move(*result));
    }();
    return *surface;
}

static void BM_Segmented_Price(benchmark::State& state) {
    const auto& surf = GetSegmented();
    for (auto _ : state) {
        benchmark::DoNotOptimize(surf.price(S, K, tau, sigma, rate));
    }
}
BENCHMARK(BM_Segmented_Price);

static void BM_Segmented_Vega(benchmark::State& state) {
    const auto& surf = GetSegmented();
    for (auto _ : state) {
        benchmark::DoNotOptimize(surf.vega(S, K, tau, sigma, rate));
    }
}
BENCHMARK(BM_Segmented_Vega);

static void BM_Segmented_Delta(benchmark::State& state) {
    const auto& surf = GetSegmented();
    auto params = MakeParams(0.0);
    for (auto _ : state) {
        benchmark::DoNotOptimize(surf.greek(Greek::Delta, params));
    }
}
BENCHMARK(BM_Segmented_Delta);

static void BM_Segmented_Gamma(benchmark::State& state) {
    const auto& surf = GetSegmented();
    auto params = MakeParams(0.0);
    for (auto _ : state) {
        benchmark::DoNotOptimize(surf.gamma(params));
    }
}
BENCHMARK(BM_Segmented_Gamma);

static void BM_Segmented_Theta(benchmark::State& state) {
    const auto& surf = GetSegmented();
    auto params = MakeParams(0.0);
    for (auto _ : state) {
        benchmark::DoNotOptimize(surf.greek(Greek::Theta, params));
    }
}
BENCHMARK(BM_Segmented_Theta);

static void BM_Segmented_Rho(benchmark::State& state) {
    const auto& surf = GetSegmented();
    auto params = MakeParams(0.0);
    for (auto _ : state) {
        benchmark::DoNotOptimize(surf.greek(Greek::Rho, params));
    }
}
BENCHMARK(BM_Segmented_Rho);

static void BM_Segmented_All(benchmark::State& state) {
    const auto& surf = GetSegmented();
    auto params = MakeParams(0.0);
    for (auto _ : state) {
        benchmark::DoNotOptimize(surf.price(S, K, tau, sigma, rate));
        benchmark::DoNotOptimize(surf.vega(S, K, tau, sigma, rate));
        benchmark::DoNotOptimize(surf.greek(Greek::Delta, params));
        benchmark::DoNotOptimize(surf.gamma(params));
        benchmark::DoNotOptimize(surf.greek(Greek::Theta, params));
        benchmark::DoNotOptimize(surf.greek(Greek::Rho, params));
    }
}
BENCHMARK(BM_Segmented_All);

// ===========================================================================
// Surface 6: Chebyshev 3D (DimensionlessTransform3D, FD fallback for gamma)
// ===========================================================================

const Chebyshev3DPriceTable& GetChebyshev3D() {
    static Chebyshev3DPriceTable* surface = [] {
        constexpr double K_ref = 100.0;
        constexpr std::array<size_t, 3> num_pts = {12, 10, 8};

        // Domain in dimensionless coordinates
        constexpr double m_min = -0.30, m_max = 0.30;
        constexpr double tp_min = 0.001, tp_max = 0.16;
        constexpr double lk_min = -2.5, lk_max = 2.8;

        // Generate Chebyshev nodes for PDE solve
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

        auto cheb = ChebyshevInterpolant<3, RawTensor<3>>::build_from_values(
            std::span<const double>(pde->values), domain, num_pts);

        DimensionlessTransform3D xform;
        Chebyshev3DTransformLeaf leaf(std::move(cheb), xform, K_ref);
        AnalyticalEEP eep(OptionType::PUT, 0.0);
        Chebyshev3DLeaf eep_leaf(std::move(leaf), std::move(eep));

        constexpr double sigma_min = 0.10, sigma_max = 0.80;
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

static void BM_Cheb3D_Price(benchmark::State& state) {
    const auto& surf = GetChebyshev3D();
    for (auto _ : state) {
        benchmark::DoNotOptimize(surf.price(S, K, tau, sigma, rate));
    }
}
BENCHMARK(BM_Cheb3D_Price);

static void BM_Cheb3D_Vega(benchmark::State& state) {
    const auto& surf = GetChebyshev3D();
    for (auto _ : state) {
        benchmark::DoNotOptimize(surf.vega(S, K, tau, sigma, rate));
    }
}
BENCHMARK(BM_Cheb3D_Vega);

static void BM_Cheb3D_Delta(benchmark::State& state) {
    const auto& surf = GetChebyshev3D();
    auto params = MakeParams(0.0);
    for (auto _ : state) {
        benchmark::DoNotOptimize(surf.delta(params));
    }
}
BENCHMARK(BM_Cheb3D_Delta);

static void BM_Cheb3D_Gamma(benchmark::State& state) {
    const auto& surf = GetChebyshev3D();
    auto params = MakeParams(0.0);
    for (auto _ : state) {
        benchmark::DoNotOptimize(surf.gamma(params));
    }
}
BENCHMARK(BM_Cheb3D_Gamma);

static void BM_Cheb3D_Theta(benchmark::State& state) {
    const auto& surf = GetChebyshev3D();
    auto params = MakeParams(0.0);
    for (auto _ : state) {
        benchmark::DoNotOptimize(surf.theta(params));
    }
}
BENCHMARK(BM_Cheb3D_Theta);

static void BM_Cheb3D_Rho(benchmark::State& state) {
    const auto& surf = GetChebyshev3D();
    auto params = MakeParams(0.0);
    for (auto _ : state) {
        benchmark::DoNotOptimize(surf.rho(params));
    }
}
BENCHMARK(BM_Cheb3D_Rho);

static void BM_Cheb3D_All(benchmark::State& state) {
    const auto& surf = GetChebyshev3D();
    auto params = MakeParams(0.0);
    for (auto _ : state) {
        benchmark::DoNotOptimize(surf.price(S, K, tau, sigma, rate));
        benchmark::DoNotOptimize(surf.vega(S, K, tau, sigma, rate));
        benchmark::DoNotOptimize(surf.delta(params));
        benchmark::DoNotOptimize(surf.gamma(params));
        benchmark::DoNotOptimize(surf.theta(params));
        benchmark::DoNotOptimize(surf.rho(params));
    }
}
BENCHMARK(BM_Cheb3D_All);

// ===========================================================================
// Surface 7: Segmented Chebyshev (discrete dividends, multi-K_ref)
// ===========================================================================

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
            .maturity = 1.0,
            .kref_config = {.K_refs = {80.0, 100.0, 120.0}},
        };

        IVGrid grid{
            .moneyness = log_m({0.80, 0.85, 0.90, 0.95, 1.00,
                                1.05, 1.10, 1.15, 1.20}),
            .vol = {0.15, 0.20, 0.25, 0.30, 0.40},
            .rate = {0.03, 0.05, 0.07, 0.09},
        };

        auto result = build_chebyshev_segmented_manual(seg_config, grid);
        if (!result) throw std::runtime_error("ChebSegmented: build failed");
        return new ChebyshevMultiKRefSurface(std::move(*result));
    }();
    return *surface;
}

static void BM_ChebSeg_Price(benchmark::State& state) {
    const auto& surf = GetChebSegmented();
    for (auto _ : state) {
        benchmark::DoNotOptimize(surf.price(S, K, tau, sigma, rate));
    }
}
BENCHMARK(BM_ChebSeg_Price);

static void BM_ChebSeg_Vega(benchmark::State& state) {
    const auto& surf = GetChebSegmented();
    for (auto _ : state) {
        benchmark::DoNotOptimize(surf.vega(S, K, tau, sigma, rate));
    }
}
BENCHMARK(BM_ChebSeg_Vega);

static void BM_ChebSeg_Delta(benchmark::State& state) {
    const auto& surf = GetChebSegmented();
    auto params = MakeParams(0.0);
    for (auto _ : state) {
        benchmark::DoNotOptimize(surf.delta(params));
    }
}
BENCHMARK(BM_ChebSeg_Delta);

static void BM_ChebSeg_Gamma(benchmark::State& state) {
    const auto& surf = GetChebSegmented();
    auto params = MakeParams(0.0);
    for (auto _ : state) {
        benchmark::DoNotOptimize(surf.gamma(params));
    }
}
BENCHMARK(BM_ChebSeg_Gamma);

static void BM_ChebSeg_Theta(benchmark::State& state) {
    const auto& surf = GetChebSegmented();
    auto params = MakeParams(0.0);
    for (auto _ : state) {
        benchmark::DoNotOptimize(surf.theta(params));
    }
}
BENCHMARK(BM_ChebSeg_Theta);

static void BM_ChebSeg_Rho(benchmark::State& state) {
    const auto& surf = GetChebSegmented();
    auto params = MakeParams(0.0);
    for (auto _ : state) {
        benchmark::DoNotOptimize(surf.rho(params));
    }
}
BENCHMARK(BM_ChebSeg_Rho);

static void BM_ChebSeg_All(benchmark::State& state) {
    const auto& surf = GetChebSegmented();
    auto params = MakeParams(0.0);
    for (auto _ : state) {
        benchmark::DoNotOptimize(surf.price(S, K, tau, sigma, rate));
        benchmark::DoNotOptimize(surf.vega(S, K, tau, sigma, rate));
        benchmark::DoNotOptimize(surf.delta(params));
        benchmark::DoNotOptimize(surf.gamma(params));
        benchmark::DoNotOptimize(surf.theta(params));
        benchmark::DoNotOptimize(surf.rho(params));
    }
}
BENCHMARK(BM_ChebSeg_All);

}  // namespace

BENCHMARK_MAIN();
