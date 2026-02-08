// SPDX-License-Identifier: MIT
//
// Head-to-head benchmark: Chebyshev-Tucker vs B-spline for dimensionless EEP.
// Sweeps grid density and prints convergence table.

#include "mango/option/table/dimensionless/chebyshev_tucker.hpp"
#include "mango/option/table/dimensionless/dimensionless_builder.hpp"
#include "mango/option/table/dimensionless/dimensionless_european.hpp"
#include "mango/option/american_option_batch.hpp"
#include "mango/math/cubic_spline_solver.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

namespace mango {
namespace {

// ===========================================================================
// Domain constants (from DimensionlessAdaptiveParams defaults)
// ===========================================================================

constexpr double SIGMA_MIN = 0.10, SIGMA_MAX = 0.80;
constexpr double RATE_MIN = 0.005, RATE_MAX = 0.10;
constexpr double TAU_MIN = 7.0 / 365, TAU_MAX = 2.0;
constexpr double MONEYNESS_MIN = 0.65, MONEYNESS_MAX = 1.50;

// Dimensionless domain bounds
const double X_MIN = std::log(MONEYNESS_MIN);           // -0.431
const double X_MAX = std::log(MONEYNESS_MAX);            //  0.405
const double TP_MIN = std::max(
    SIGMA_MIN * SIGMA_MIN * TAU_MIN / 2.0, 0.005);       //  0.005
const double TP_MAX = SIGMA_MAX * SIGMA_MAX * TAU_MAX / 2.0;  // 0.64
const double LK_MIN = std::log(2.0 * RATE_MIN / (SIGMA_MAX * SIGMA_MAX));  // -5.37
const double LK_MAX = std::log(2.0 * RATE_MAX / (SIGMA_MIN * SIGMA_MIN));  //  2.996

constexpr double K_REF = 100.0;
constexpr auto OPTION_TYPE = OptionType::PUT;

// Segment boundaries for config 2 (matching PR 386 logic)
constexpr size_t N_SEGMENTS = 3;

// Tucker epsilon sweep (pinned in design doc)
constexpr std::array<double, 6> TUCKER_EPSILONS = {
    1e-2, 1e-4, 1e-6, 1e-8, 1e-10, 1e-12
};

// ===========================================================================
// Reference EEP solve
// ===========================================================================

double reference_eep(double x0, double tp0, double lk0) {
    double kappa = std::exp(lk0);
    BatchAmericanOptionSolver solver;
    solver.set_grid_accuracy(make_grid_accuracy(GridAccuracyProfile::Ultra));
    std::vector<double> snap = {tp0};
    solver.set_snapshot_times(std::span<const double>{snap});
    double tp_max = std::max(tp0 * 1.01, 0.02);
    std::vector<PricingParams> batch;
    batch.emplace_back(
        OptionSpec{.spot = K_REF, .strike = K_REF, .maturity = tp_max,
                   .rate = kappa, .dividend_yield = 0.0, .option_type = OPTION_TYPE},
        std::sqrt(2.0));
    auto result = solver.solve_batch(batch, /*use_shared_grid=*/false);
    if (result.results.empty() || !result.results[0].has_value()) return 0.0;
    const auto& sol = result.results[0].value();
    if (sol.num_snapshots() < 1) return 0.0;
    CubicSpline<double> spline;
    if (spline.build(sol.grid()->x(), sol.at_time(0)).has_value()) return 0.0;
    double am = spline.eval(x0);
    double eu = dimensionless_european(x0, tp0, kappa, OPTION_TYPE);
    return std::max(am - eu, 0.0);
}

// ===========================================================================
// Probe generation (520 probes: 500 LHS + 8 corners + 12 edge midpoints)
// ===========================================================================

struct Probe { double x, tp, lk, true_eep; };

std::vector<Probe> generate_probes() {
    std::vector<Probe> probes;

    // 500 seeded LHS probes
    std::mt19937 rng(42);
    std::uniform_real_distribution<double> dx(X_MIN, X_MAX);
    std::uniform_real_distribution<double> dtp(TP_MIN, TP_MAX);
    std::uniform_real_distribution<double> dlk(LK_MIN, LK_MAX);

    for (size_t i = 0; i < 500; ++i) {
        double x = dx(rng), tp = dtp(rng), lk = dlk(rng);
        probes.push_back({x, tp, lk, reference_eep(x, tp, lk)});
    }

    // 8 corner points
    for (double x : {X_MIN, X_MAX})
        for (double tp : {TP_MIN, TP_MAX})
            for (double lk : {LK_MIN, LK_MAX})
                probes.push_back({x, tp, lk, reference_eep(x, tp, lk)});

    // 12 edge midpoints (vary one axis, fix other two at midpoint)
    double x_mid = (X_MIN + X_MAX) / 2;
    double tp_mid = (TP_MIN + TP_MAX) / 2;
    double lk_mid = (LK_MIN + LK_MAX) / 2;

    for (double x : {X_MIN, X_MAX}) {
        probes.push_back({x, tp_mid, lk_mid, reference_eep(x, tp_mid, lk_mid)});
    }
    for (double tp : {TP_MIN, TP_MAX}) {
        probes.push_back({x_mid, tp, lk_mid, reference_eep(x_mid, tp, lk_mid)});
    }
    for (double lk : {LK_MIN, LK_MAX}) {
        probes.push_back({x_mid, tp_mid, lk, reference_eep(x_mid, tp_mid, lk)});
    }
    // Remaining 6 edge midpoints: vary two axes at extremes, fix one at mid
    for (double x : {X_MIN, X_MAX}) {
        for (double tp : {TP_MIN, TP_MAX}) {
            probes.push_back({x, tp, lk_mid, reference_eep(x, tp, lk_mid)});
        }
    }
    // That gives: 500 + 8 + 2 + 2 + 2 + 4 = 518.
    // Add the last 2: vary (x, lk) at extremes, tp at mid
    for (double x : {X_MIN, X_MAX}) {
        probes.push_back({x, tp_mid, LK_MIN, reference_eep(x, tp_mid, LK_MIN)});
    }
    // Total: 520

    return probes;
}

// ===========================================================================
// B-spline headroom helper
// ===========================================================================

double spline_headroom(double domain_width, size_t n_knots) {
    size_t n = std::max(n_knots, size_t{4});
    return 3.0 * domain_width / static_cast<double>(n - 1);
}

std::vector<double> linspace(double lo, double hi, size_t n) {
    std::vector<double> v(n);
    for (size_t i = 0; i < n; ++i)
        v[i] = lo + (hi - lo) * static_cast<double>(i) / static_cast<double>(n - 1);
    return v;
}

// ===========================================================================
// Config 1: B-spline unsegmented
// ===========================================================================

struct ErrorResult {
    double max_err, avg_err;
    double build_seconds;
    int n_pde_solves;
};

ErrorResult eval_bspline_unsegmented(
    size_t num_pts, const std::vector<Probe>& probes)
{
    // Generate grids with headroom
    double hx = spline_headroom(X_MAX - X_MIN, num_pts);
    double htp = spline_headroom(TP_MAX - TP_MIN, num_pts);
    double hlk = spline_headroom(LK_MAX - LK_MIN, num_pts);
    auto x_grid = linspace(X_MIN - hx, X_MAX + hx, num_pts);
    auto tp_grid = linspace(std::max(TP_MIN - htp, 1e-3), TP_MAX + htp, num_pts);
    auto lk_grid = linspace(LK_MIN - hlk, LK_MAX + hlk, num_pts);

    DimensionlessAxes axes{x_grid, tp_grid, lk_grid};

    auto t0 = std::chrono::steady_clock::now();
    auto result = build_dimensionless_surface(axes, K_REF, OPTION_TYPE,
                                              SurfaceContent::EarlyExercisePremium);
    auto t1 = std::chrono::steady_clock::now();

    if (!result.has_value()) {
        std::fprintf(stderr, "  B-spline unseg build failed at num_pts=%zu\n", num_pts);
        return {999.0, 999.0, 0.0, 0};
    }

    double max_err = 0, sum_err = 0;
    for (const auto& p : probes) {
        double interp = result->surface->value({p.x, p.tp, p.lk});
        double err = std::abs(interp - p.true_eep);
        max_err = std::max(max_err, err);
        sum_err += err;
    }

    return {
        max_err, sum_err / static_cast<double>(probes.size()),
        std::chrono::duration<double>(t1 - t0).count(),
        result->n_pde_solves
    };
}

// ===========================================================================
// Config 2: B-spline 3-segment (matching PR 386 boundary logic)
// ===========================================================================

ErrorResult eval_bspline_segmented(
    size_t num_pts, const std::vector<Probe>& probes)
{
    double hx = spline_headroom(X_MAX - X_MIN, num_pts);
    double htp = spline_headroom(TP_MAX - TP_MIN, num_pts);
    auto x_grid = linspace(X_MIN - hx, X_MAX + hx, num_pts);
    auto tp_grid = linspace(std::max(TP_MIN - htp, 1e-3), TP_MAX + htp, num_pts);

    double lk_seg_width = (LK_MAX - LK_MIN) / static_cast<double>(N_SEGMENTS);

    auto t0 = std::chrono::steady_clock::now();

    std::vector<SegmentedDimensionlessSurface::Segment> segments;
    int total_solves = 0;

    for (size_t s = 0; s < N_SEGMENTS; ++s) {
        double seg_lk_min_phys = LK_MIN + lk_seg_width * static_cast<double>(s);
        double seg_lk_max_phys = LK_MIN + lk_seg_width * static_cast<double>(s + 1);
        double hlk = spline_headroom(seg_lk_max_phys - seg_lk_min_phys, num_pts);
        auto lk_grid = linspace(seg_lk_min_phys - hlk, seg_lk_max_phys + hlk, num_pts);

        DimensionlessAxes axes{x_grid, tp_grid, lk_grid};
        auto result = build_dimensionless_surface(axes, K_REF, OPTION_TYPE,
                                                  SurfaceContent::EarlyExercisePremium);
        if (!result.has_value()) {
            std::fprintf(stderr, "  B-spline seg[%zu] build failed at num_pts=%zu\n", s, num_pts);
            return {999.0, 999.0, 0.0, 0};
        }

        segments.push_back({result->surface, seg_lk_min_phys, seg_lk_max_phys});
        total_solves += result->n_pde_solves;
    }

    auto surface = std::make_shared<SegmentedDimensionlessSurface>(std::move(segments));
    auto t1 = std::chrono::steady_clock::now();

    double max_err = 0, sum_err = 0;
    for (const auto& p : probes) {
        double interp = surface->value({p.x, p.tp, p.lk});
        double err = std::abs(interp - p.true_eep);
        max_err = std::max(max_err, err);
        sum_err += err;
    }

    return {
        max_err, sum_err / static_cast<double>(probes.size()),
        std::chrono::duration<double>(t1 - t0).count(),
        total_solves
    };
}

// ===========================================================================
// Config 3: Chebyshev full tensor (no Tucker)
// ===========================================================================

ErrorResult eval_chebyshev_full(
    size_t num_pts, const std::vector<Probe>& probes)
{
    ChebyshevTuckerDomain domain{
        .bounds = {{{X_MIN, X_MAX}, {TP_MIN, TP_MAX}, {LK_MIN, LK_MAX}}},
    };
    // epsilon near machine precision => effectively full rank
    ChebyshevTuckerConfig config{
        .num_pts = {num_pts, num_pts, num_pts},
        .epsilon = 1e-15,
    };

    auto eep_fn = [](double x, double tp, double lk) -> double {
        return reference_eep(x, tp, lk);
    };

    auto t0 = std::chrono::steady_clock::now();
    auto interp = ChebyshevTucker3D::build(eep_fn, domain, config);
    auto t1 = std::chrono::steady_clock::now();

    double max_err = 0, sum_err = 0;
    for (const auto& p : probes) {
        double val = interp.eval({p.x, p.tp, p.lk});
        double err = std::abs(val - p.true_eep);
        max_err = std::max(max_err, err);
        sum_err += err;
    }

    return {
        max_err, sum_err / static_cast<double>(probes.size()),
        std::chrono::duration<double>(t1 - t0).count(),
        static_cast<int>(num_pts * num_pts * num_pts)  // One solve per grid point
    };
}

// ===========================================================================
// Config 4: Chebyshev-Tucker (epsilon sweep)
// ===========================================================================

struct TuckerSweepResult {
    double epsilon;
    std::array<size_t, 3> ranks;
    double max_err, avg_err;
    size_t compressed_size;
};

std::vector<TuckerSweepResult> eval_chebyshev_tucker(
    size_t num_pts, const std::vector<Probe>& probes)
{
    ChebyshevTuckerDomain domain{
        .bounds = {{{X_MIN, X_MAX}, {TP_MIN, TP_MAX}, {LK_MIN, LK_MAX}}},
    };

    auto eep_fn = [](double x, double tp, double lk) -> double {
        return reference_eep(x, tp, lk);
    };

    std::vector<TuckerSweepResult> results;

    for (double eps : TUCKER_EPSILONS) {
        ChebyshevTuckerConfig config{
            .num_pts = {num_pts, num_pts, num_pts},
            .epsilon = eps,
        };

        auto interp = ChebyshevTucker3D::build(eep_fn, domain, config);

        double max_err = 0, sum_err = 0;
        for (const auto& p : probes) {
            double val = interp.eval({p.x, p.tp, p.lk});
            double err = std::abs(val - p.true_eep);
            max_err = std::max(max_err, err);
            sum_err += err;
        }

        results.push_back({
            eps, interp.ranks(), max_err, sum_err / static_cast<double>(probes.size()),
            interp.compressed_size()
        });
    }

    return results;
}

}  // anonymous namespace
}  // namespace mango

// ===========================================================================
// Main
// ===========================================================================

int main() {
    using namespace mango;

    std::printf("=== Chebyshev-Tucker vs B-spline benchmark ===\n");
    std::printf("Domain: x=[%.3f,%.3f] tp=[%.4f,%.3f] lk=[%.3f,%.3f]\n",
                X_MIN, X_MAX, TP_MIN, TP_MAX, LK_MIN, LK_MAX);

    std::printf("\nGenerating reference probes...\n");
    auto probes = generate_probes();
    std::printf("  %zu probes generated\n\n", probes.size());

    // Sweep num_pts
    for (size_t num_pts : {6, 8, 10, 12, 15, 18, 20, 25, 30}) {
        std::printf("=== num_pts = %zu ===\n", num_pts);

        // Config 1: B-spline unsegmented
        auto bs1 = eval_bspline_unsegmented(num_pts, probes);
        std::printf("  B-spline unseg:  max_err=%.6f  avg_err=%.6f  "
                     "solves=%d  build=%.1fs\n",
                     bs1.max_err, bs1.avg_err, bs1.n_pde_solves, bs1.build_seconds);

        // Config 2: B-spline 3-segment
        auto bs3 = eval_bspline_segmented(num_pts, probes);
        std::printf("  B-spline 3-seg:  max_err=%.6f  avg_err=%.6f  "
                     "solves=%d  build=%.1fs\n",
                     bs3.max_err, bs3.avg_err, bs3.n_pde_solves, bs3.build_seconds);

        // Config 3: Chebyshev full (only at smaller sizes due to cost)
        if (num_pts <= 20) {
            auto ch_full = eval_chebyshev_full(num_pts, probes);
            std::printf("  Cheb full:       max_err=%.6f  avg_err=%.6f  "
                         "solves=%d  build=%.1fs\n",
                         ch_full.max_err, ch_full.avg_err,
                         ch_full.n_pde_solves, ch_full.build_seconds);
        }

        // Config 4: Chebyshev-Tucker sweep (only at a few sizes)
        if (num_pts == 10 || num_pts == 15 || num_pts == 20) {
            auto tucker_results = eval_chebyshev_tucker(num_pts, probes);
            for (const auto& tr : tucker_results) {
                std::printf("  Cheb-Tucker e=%.0e: max=%.6f avg=%.6f "
                             "R=(%zu,%zu,%zu) size=%zu\n",
                             tr.epsilon, tr.max_err, tr.avg_err,
                             tr.ranks[0], tr.ranks[1], tr.ranks[2],
                             tr.compressed_size);
            }
        }

        std::printf("\n");
    }

    return 0;
}
