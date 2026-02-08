// SPDX-License-Identifier: MIT
//
// Quick accuracy comparison: Chebyshev-Tucker vs B-spline for dimensionless EEP.
// Uses the same batch PDE solver for both methods (fair cost comparison).

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

constexpr double SIGMA_MIN = 0.10, SIGMA_MAX = 0.80;
constexpr double RATE_MIN = 0.005, RATE_MAX = 0.10;
constexpr double TAU_MIN = 7.0 / 365, TAU_MAX = 2.0;
constexpr double MONEYNESS_MIN = 0.65, MONEYNESS_MAX = 1.50;

const double X_MIN = std::log(MONEYNESS_MIN);
const double X_MAX = std::log(MONEYNESS_MAX);
const double TP_MIN = std::max(SIGMA_MIN * SIGMA_MIN * TAU_MIN / 2.0, 0.005);
const double TP_MAX = SIGMA_MAX * SIGMA_MAX * TAU_MAX / 2.0;
const double LK_MIN = std::log(2.0 * RATE_MIN / (SIGMA_MAX * SIGMA_MAX));
const double LK_MAX = std::log(2.0 * RATE_MAX / (SIGMA_MIN * SIGMA_MIN));

constexpr double K_REF = 100.0;
constexpr auto OPTION_TYPE = OptionType::PUT;
constexpr size_t N_SEGMENTS = 3;

// ===========================================================================
// Reference EEP via individual PDE solve (for probe ground truth only)
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
    auto result = solver.solve_batch(batch, false);
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
// Efficient Chebyshev EEP tensor build via batch PDE solver
// Same cost as B-spline: one PDE per ln κ node.
// ===========================================================================

struct ChebyshevBuildResult {
    ChebyshevTucker3D interp;
    int n_pde_solves;
    double build_seconds;
};

ChebyshevBuildResult build_chebyshev_efficient(
    size_t num_pts, double epsilon)
{
    ChebyshevTuckerDomain dom{
        .bounds = {{{X_MIN, X_MAX}, {TP_MIN, TP_MAX}, {LK_MIN, LK_MAX}}}};
    ChebyshevTuckerConfig cfg{
        .num_pts = {num_pts, num_pts, num_pts}, .epsilon = epsilon};

    // Generate Chebyshev nodes per axis
    auto x_nodes = chebyshev_nodes(num_pts, X_MIN, X_MAX);
    auto tp_nodes = chebyshev_nodes(num_pts, TP_MIN, TP_MAX);
    auto lk_nodes = chebyshev_nodes(num_pts, LK_MIN, LK_MAX);

    auto t0 = std::chrono::steady_clock::now();

    // Create batch: one PDE per ln κ node (same pattern as dimensionless_builder)
    const double sigma_eff = std::sqrt(2.0);
    const double tp_max = tp_nodes.back() * 1.01;

    std::vector<PricingParams> batch;
    batch.reserve(num_pts);
    for (size_t k = 0; k < num_pts; ++k) {
        double kappa = std::exp(lk_nodes[k]);
        batch.emplace_back(
            OptionSpec{.spot = K_REF, .strike = K_REF, .maturity = tp_max,
                       .rate = kappa, .dividend_yield = 0.0, .option_type = OPTION_TYPE},
            sigma_eff);
    }

    // Solve batch with Chebyshev τ' nodes as snapshots
    BatchAmericanOptionSolver solver;
    solver.set_grid_accuracy(make_grid_accuracy(GridAccuracyProfile::Ultra));
    solver.set_snapshot_times(std::span<const double>{tp_nodes});
    auto batch_result = solver.solve_batch(batch, /*use_shared_grid=*/true);

    // Extract EEP tensor: resample PDE solution at Chebyshev x nodes
    std::vector<double> tensor(num_pts * num_pts * num_pts, 0.0);

    for (size_t k = 0; k < num_pts; ++k) {
        if (!batch_result.results[k].has_value()) continue;
        const auto& result = batch_result.results[k].value();
        auto x_grid = result.grid()->x();
        double kappa = std::exp(lk_nodes[k]);

        for (size_t j = 0; j < num_pts; ++j) {
            auto spatial = result.at_time(j);
            CubicSpline<double> spline;
            if (spline.build(x_grid, spatial).has_value()) continue;

            for (size_t i = 0; i < num_pts; ++i) {
                double am = spline.eval(x_nodes[i]);
                double eu = dimensionless_european(
                    x_nodes[i], tp_nodes[j], kappa, OPTION_TYPE);
                tensor[i * num_pts * num_pts + j * num_pts + k] =
                    std::max(am - eu, 0.0);
            }
        }
    }

    auto interp = ChebyshevTucker3D::build_from_values(tensor, dom, cfg);
    auto t1 = std::chrono::steady_clock::now();

    return {std::move(interp), static_cast<int>(num_pts),
            std::chrono::duration<double>(t1 - t0).count()};
}

// ===========================================================================
// Probes
// ===========================================================================

struct Probe { double x, tp, lk, true_eep; };

std::vector<Probe> generate_probes(size_t n_random) {
    std::vector<Probe> probes;
    std::mt19937 rng(42);
    std::uniform_real_distribution<double> dx(X_MIN, X_MAX);
    std::uniform_real_distribution<double> dtp(TP_MIN, TP_MAX);
    std::uniform_real_distribution<double> dlk(LK_MIN, LK_MAX);

    for (size_t i = 0; i < n_random; ++i) {
        double x = dx(rng), tp = dtp(rng), lk = dlk(rng);
        probes.push_back({x, tp, lk, reference_eep(x, tp, lk)});
        if ((i + 1) % 20 == 0)
            std::fprintf(stderr, "  probe %zu/%zu\n", i + 1, n_random);
    }
    for (double x : {X_MIN, X_MAX})
        for (double tp : {TP_MIN, TP_MAX})
            for (double lk : {LK_MIN, LK_MAX})
                probes.push_back({x, tp, lk, reference_eep(x, tp, lk)});
    return probes;
}

// ===========================================================================
// Helpers
// ===========================================================================

double spline_headroom(double domain_width, size_t n_knots) {
    return 3.0 * domain_width / static_cast<double>(std::max(n_knots, size_t{4}) - 1);
}

std::vector<double> linspace(double lo, double hi, size_t n) {
    std::vector<double> v(n);
    for (size_t i = 0; i < n; ++i)
        v[i] = lo + (hi - lo) * static_cast<double>(i) / static_cast<double>(n - 1);
    return v;
}

struct ErrorResult { double max_err, avg_err; int n_solves; double build_s; };

ErrorResult measure_error(const std::vector<Probe>& probes,
                          auto eval_fn) {
    double mx = 0, sm = 0;
    for (const auto& p : probes) {
        double e = std::abs(eval_fn(p.x, p.tp, p.lk) - p.true_eep);
        mx = std::max(mx, e);
        sm += e;
    }
    return {mx, sm / static_cast<double>(probes.size()), 0, 0};
}

// ===========================================================================
// B-spline builders
// ===========================================================================

ErrorResult eval_bspline_unseg(size_t num_pts, const std::vector<Probe>& probes) {
    double hx = spline_headroom(X_MAX - X_MIN, num_pts);
    double htp = spline_headroom(TP_MAX - TP_MIN, num_pts);
    double hlk = spline_headroom(LK_MAX - LK_MIN, num_pts);
    DimensionlessAxes axes{
        linspace(X_MIN - hx, X_MAX + hx, num_pts),
        linspace(std::max(TP_MIN - htp, 1e-3), TP_MAX + htp, num_pts),
        linspace(LK_MIN - hlk, LK_MAX + hlk, num_pts)};
    auto t0 = std::chrono::steady_clock::now();
    auto res = build_dimensionless_surface(axes, K_REF, OPTION_TYPE,
                                           SurfaceContent::EarlyExercisePremium);
    auto t1 = std::chrono::steady_clock::now();
    if (!res) return {999, 999, 0, 0};
    auto err = measure_error(probes, [&](double x, double tp, double lk) {
        return res->surface->value({x, tp, lk});
    });
    err.n_solves = res->n_pde_solves;
    err.build_s = std::chrono::duration<double>(t1 - t0).count();
    return err;
}

ErrorResult eval_bspline_seg(size_t num_pts, const std::vector<Probe>& probes) {
    double hx = spline_headroom(X_MAX - X_MIN, num_pts);
    double htp = spline_headroom(TP_MAX - TP_MIN, num_pts);
    auto x_grid = linspace(X_MIN - hx, X_MAX + hx, num_pts);
    auto tp_grid = linspace(std::max(TP_MIN - htp, 1e-3), TP_MAX + htp, num_pts);
    double lk_seg_w = (LK_MAX - LK_MIN) / static_cast<double>(N_SEGMENTS);
    auto t0 = std::chrono::steady_clock::now();
    std::vector<SegmentedDimensionlessSurface::Segment> segs;
    int total = 0;
    for (size_t s = 0; s < N_SEGMENTS; ++s) {
        double lo = LK_MIN + lk_seg_w * static_cast<double>(s);
        double hi = LK_MIN + lk_seg_w * static_cast<double>(s + 1);
        double hlk = spline_headroom(hi - lo, num_pts);
        DimensionlessAxes axes{x_grid, tp_grid, linspace(lo - hlk, hi + hlk, num_pts)};
        auto res = build_dimensionless_surface(axes, K_REF, OPTION_TYPE,
                                               SurfaceContent::EarlyExercisePremium);
        if (!res) return {999, 999, 0, 0};
        segs.push_back({res->surface, lo, hi});
        total += res->n_pde_solves;
    }
    auto surf = std::make_shared<SegmentedDimensionlessSurface>(std::move(segs));
    auto t1 = std::chrono::steady_clock::now();
    auto err = measure_error(probes, [&](double x, double tp, double lk) {
        return surf->value({x, tp, lk});
    });
    err.n_solves = total;
    err.build_s = std::chrono::duration<double>(t1 - t0).count();
    return err;
}

}  // namespace
}  // namespace mango

int main() {
    using namespace mango;

    std::printf("=== Chebyshev-Tucker vs B-spline (fair cost comparison) ===\n");
    std::printf("Domain: x=[%.3f,%.3f] tp=[%.4f,%.3f] lk=[%.3f,%.3f]\n",
                X_MIN, X_MAX, TP_MIN, TP_MAX, LK_MIN, LK_MAX);
    std::printf("ln kappa range: %.2f units\n\n", LK_MAX - LK_MIN);

    std::fprintf(stderr, "Generating 100 reference probes + 8 corners...\n");
    auto probes = generate_probes(100);
    std::printf("%zu probes generated\n\n", probes.size());

    std::printf("%-5s  %-24s  %10s  %10s  %6s  %7s\n",
                "N", "Method", "max_err", "avg_err", "solves", "build_s");
    std::printf("-----  ------------------------  ----------  ----------  ------  -------\n");

    for (size_t n : {6, 8, 10, 12, 15, 20, 25}) {
        std::fprintf(stderr, "\n--- num_pts=%zu ---\n", n);

        // B-spline unsegmented
        std::fprintf(stderr, "  B-spline unseg...\n");
        auto bs1 = eval_bspline_unseg(n, probes);
        std::printf("%-5zu  %-24s  %10.6f  %10.6f  %6d  %7.1f\n",
                    n, "B-spline unseg", bs1.max_err, bs1.avg_err,
                    bs1.n_solves, bs1.build_s);

        // B-spline 3-segment
        std::fprintf(stderr, "  B-spline 3-seg...\n");
        auto bs3 = eval_bspline_seg(n, probes);
        std::printf("%-5zu  %-24s  %10.6f  %10.6f  %6d  %7.1f\n",
                    n, "B-spline 3-seg", bs3.max_err, bs3.avg_err,
                    bs3.n_solves, bs3.build_s);

        // Chebyshev-Tucker (efficient batch build, same PDE cost)
        std::fprintf(stderr, "  Chebyshev efficient (eps=1e-14)...\n");
        auto ch = build_chebyshev_efficient(n, 1e-14);
        auto ch_err = measure_error(probes, [&](double x, double tp, double lk) {
            return ch.interp.eval({x, tp, lk});
        });
        std::printf("%-5zu  %-24s  %10.6f  %10.6f  %6d  %7.1f\n",
                    n, "Chebyshev (full)", ch_err.max_err, ch_err.avg_err,
                    ch.n_pde_solves, ch.build_seconds);

        // Tucker compressed (eps=1e-4)
        std::fprintf(stderr, "  Chebyshev efficient (eps=1e-4)...\n");
        auto ct = build_chebyshev_efficient(n, 1e-4);
        auto ct_err = measure_error(probes, [&](double x, double tp, double lk) {
            return ct.interp.eval({x, tp, lk});
        });
        auto ranks = ct.interp.ranks();
        std::printf("%-5zu  %-24s  %10.6f  %10.6f  %6d  %7.1f  R=(%zu,%zu,%zu)\n",
                    n, "Chebyshev-Tucker e=1e-4", ct_err.max_err, ct_err.avg_err,
                    ct.n_pde_solves, ct.build_seconds,
                    ranks[0], ranks[1], ranks[2]);

        std::printf("\n");
        std::fflush(stdout);
    }

    return 0;
}
