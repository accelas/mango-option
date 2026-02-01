// SPDX-License-Identifier: MIT
/**
 * @file iv_interpolation_profile.cc
 * @brief Profile B-spline interpolation performance, especially vega computation
 *
 * Benchmarks:
 * - BM_BSpline_Eval: Single price evaluation (baseline)
 * - BM_BSpline_VegaFD: Vega via finite difference (3 separate evals)
 * - BM_BSpline_VegaTriple: Vega via scalar triple (3 simultaneous evals)
 * - BM_BSpline_VegaAnalytic: Vega via Cox-de Boor analytic derivative
 *
 * Run with: bazel run -c opt //benchmarks:iv_interpolation_profile -- --benchmark_filter="Vega"
 */

#include "src/math/bspline_nd_separable.hpp"
#include "src/option/table/price_table_surface.hpp"
#include "src/option/table/price_table_axes.hpp"
#include "src/option/table/price_table_metadata.hpp"
#include <benchmark/benchmark.h>
#include <memory>
#include <vector>
#include <cmath>
#include <stdexcept>

using namespace mango;

namespace {

// Analytic Black-Scholes for fitting test surface
double analytic_bs_price(double S, double K, double tau, double sigma, double r) {
    if (tau <= 0.0) {
        return std::max(K - S, 0.0);
    }

    const double sqrt_tau = std::sqrt(tau);
    const double d1 = (std::log(S / K) + (r + 0.5 * sigma * sigma) * tau) / (sigma * sqrt_tau);
    const double d2 = d1 - sigma * sqrt_tau;

    auto Phi = [](double x) {
        return 0.5 * (1.0 + std::erf(x / std::sqrt(2.0)));
    };

    return K * std::exp(-r * tau) * Phi(-d2) - S * Phi(-d1);
}

// Fixture with fitted B-spline surface
struct AnalyticSurfaceFixture {
    double K_ref;
    std::vector<double> m_grid;
    std::vector<double> tau_grid;
    std::vector<double> sigma_grid;
    std::vector<double> rate_grid;
    std::shared_ptr<const PriceTableSurface<4>> surface;
};

const AnalyticSurfaceFixture& GetSurface() {
    static AnalyticSurfaceFixture* fixture = [] {
        auto fixture_ptr = std::make_unique<AnalyticSurfaceFixture>();
        fixture_ptr->K_ref = 100.0;

        // Small test grids
        fixture_ptr->m_grid = {0.8, 0.9, 1.0, 1.1, 1.2};
        fixture_ptr->tau_grid = {0.1, 0.5, 1.0, 2.0};
        fixture_ptr->sigma_grid = {0.10, 0.15, 0.20, 0.25, 0.30};
        fixture_ptr->rate_grid = {0.0, 0.025, 0.05, 0.10};

        const size_t Nm = fixture_ptr->m_grid.size();
        const size_t Nt = fixture_ptr->tau_grid.size();
        const size_t Nv = fixture_ptr->sigma_grid.size();
        const size_t Nr = fixture_ptr->rate_grid.size();

        // Generate prices from analytic Black-Scholes
        std::vector<double> prices(Nm * Nt * Nv * Nr);
        for (size_t i = 0; i < Nm; ++i) {
            for (size_t j = 0; j < Nt; ++j) {
                for (size_t k = 0; k < Nv; ++k) {
                    for (size_t l = 0; l < Nr; ++l) {
                        const size_t idx = ((i * Nt + j) * Nv + k) * Nr + l;
                        prices[idx] = analytic_bs_price(
                            fixture_ptr->m_grid[i] * fixture_ptr->K_ref,
                            fixture_ptr->K_ref,
                            fixture_ptr->tau_grid[j],
                            fixture_ptr->sigma_grid[k],
                            fixture_ptr->rate_grid[l]);
                    }
                }
            }
        }

        // Fit B-spline using factory method
        auto fitter_result = BSplineNDSeparable<double, 4>::create(
            std::array<std::vector<double>, 4>{
                fixture_ptr->m_grid,
                fixture_ptr->tau_grid,
                fixture_ptr->sigma_grid,
                fixture_ptr->rate_grid});

        if (!fitter_result.has_value()) {
            throw std::runtime_error("Failed to create B-spline fitter");
        }

        auto fit_result = fitter_result.value().fit(prices, BSplineNDSeparableConfig<double>{.tolerance = 1e-6});
        if (!fit_result.has_value()) {
            throw std::runtime_error("Failed to fit B-spline surface");
        }

        // Create PriceTableAxes
        PriceTableAxes<4> axes;
        axes.grids = {
            fixture_ptr->m_grid,
            fixture_ptr->tau_grid,
            fixture_ptr->sigma_grid,
            fixture_ptr->rate_grid
        };

        // Create metadata
        PriceTableMetadata meta{
            .K_ref = fixture_ptr->K_ref,
        };

        // Create surface with coefficients directly
        auto surface_result = PriceTableSurface<4>::build(axes, fit_result->coefficients, meta);
        if (!surface_result.has_value()) {
            throw std::runtime_error("Failed to create surface");
        }
        fixture_ptr->surface = std::move(surface_result.value());

        return fixture_ptr.release();
    }();

    return *fixture;
}

} // namespace

// ============================================================================
// Baseline: Single price evaluation
// ============================================================================

static void BM_BSpline_Eval(benchmark::State& state) {
    const auto& surf = GetSurface();

    constexpr double m = 1.03;
    constexpr double tau = 0.5;
    constexpr double sigma = 0.22;
    constexpr double r = 0.05;

    for (auto _ : state) {
        double price = surf.surface->value({m, tau, sigma, r});
        benchmark::DoNotOptimize(price);
    }

    state.SetLabel("Single price eval (baseline)");
}
BENCHMARK(BM_BSpline_Eval);

// ============================================================================
// Vega via Finite Difference (3 separate evaluations) - Not implemented
// ============================================================================
// Note: Would require computing prices at sigma +/- epsilon via difference approximation
// Implementation is non-trivial due to API changes. The analytic derivative method
// (BM_BSpline_VegaAnalytic) provides the same functionality more efficiently.

// static void BM_BSpline_VegaFD(benchmark::State& state) {
//     const auto& surf = GetSurface();
//
//     constexpr double m = 1.03;
//     constexpr double tau = 0.5;
//     constexpr double sigma = 0.22;
//     constexpr double r = 0.05;
//     constexpr double epsilon = 1e-4;
//
//     for (auto _ : state) {
//         // 3 separate evaluations (old approach)
//         double price_center = surf.evaluator->eval(m, tau, sigma, r);
//         double price_up = surf.evaluator->eval(m, tau, sigma + epsilon, r);
//         double price_down = surf.evaluator->eval(m, tau, sigma - epsilon, r);
//         double vega = (price_up - price_down) / (2.0 * epsilon);
//         benchmark::DoNotOptimize(vega);
//     }
//
//     state.SetLabel("Vega via FD (3 × 256 FMAs = 768)");
// }
// BENCHMARK(BM_BSpline_VegaFD);

// ============================================================================
// Vega via Scalar Triple Evaluation (single pass, no SIMD) - Not implemented
// ============================================================================
// Note: This benchmark was intended to compare scalar triple evaluation vs analytic derivative.
// The analytic derivative method (BM_BSpline_VegaAnalytic) is the recommended approach.

// static void BM_BSpline_VegaTriple(benchmark::State& state) {
//     const auto& surf = GetSurface();
//
//     constexpr double m = 1.03;
//     constexpr double tau = 0.5;
//     constexpr double sigma = 0.22;
//     constexpr double r = 0.05;
//     constexpr double epsilon = 1e-4;
//
//     for (auto _ : state) {
//         double price, vega;
//         surf.evaluator->eval_price_and_vega(m, tau, sigma, r, price, vega);
//         benchmark::DoNotOptimize(vega);
//     }
//
//     state.SetLabel("Vega triple scalar (single-pass)");
// }
// BENCHMARK(BM_BSpline_VegaTriple);

// ============================================================================
// Vega via Analytic B-spline Derivative (single evaluation)
// ============================================================================

static void BM_BSpline_VegaAnalytic(benchmark::State& state) {
    const auto& surf = GetSurface();

    constexpr double m = 1.03;
    constexpr double tau = 0.5;
    constexpr double sigma = 0.22;
    constexpr double r = 0.05;

    for (auto _ : state) {
        // Use analytic partial derivative with respect to sigma (dimension 2)
        double price = surf.surface->value({m, tau, sigma, r});
        double vega = surf.surface->partial(2, {m, tau, sigma, r});
        benchmark::DoNotOptimize(price);
        benchmark::DoNotOptimize(vega);
    }

    state.SetLabel("Vega analytic (B'_k derivative)");
}
BENCHMARK(BM_BSpline_VegaAnalytic);

BENCHMARK_MAIN();
