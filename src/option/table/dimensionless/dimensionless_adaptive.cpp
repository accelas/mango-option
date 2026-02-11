// SPDX-License-Identifier: MIT
#include "mango/option/table/dimensionless/dimensionless_builder.hpp"
#include "mango/option/table/dimensionless/dimensionless_european.hpp"
#include "mango/option/table/eep/dimensionless_3d_accessor.hpp"
#include "mango/option/table/eep/analytical_eep.hpp"
#include "mango/math/bspline_nd_separable.hpp"
#include "mango/option/table/bspline/bspline_surface.hpp"
#include "mango/option/american_option_batch.hpp"
#include "mango/option/grid_spec_types.hpp"
#include "mango/math/cubic_spline_solver.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <random>

namespace mango {

namespace {

std::vector<double> linspace(double lo, double hi, size_t n) {
    std::vector<double> v(n);
    for (size_t i = 0; i < n; ++i)
        v[i] = lo + (hi - lo) * static_cast<double>(i) / static_cast<double>(n - 1);
    return v;
}

double spline_headroom(double domain_width, size_t n_knots) {
    size_t n = std::max(n_knots, size_t{4});
    return 3.0 * domain_width / static_cast<double>(n - 1);
}

/// Solve dimensionless PDE at a single probe point for reference EEP
double reference_eep(double x0, double tau_prime_0, double ln_kappa_0,
                     double K_ref, OptionType option_type) {
    double kappa = std::exp(ln_kappa_0);
    double sigma_eff = std::sqrt(2.0);

    BatchAmericanOptionSolver solver;
    solver.set_grid_accuracy(make_grid_accuracy(GridAccuracyProfile::Ultra));

    std::vector<double> snap_times = {tau_prime_0};
    solver.set_snapshot_times(std::span<const double>{snap_times});

    double tau_prime_max = std::max(tau_prime_0 * 1.01, 0.02);
    std::vector<PricingParams> batch;
    batch.emplace_back(
        OptionSpec{
            .spot = K_ref,
            .strike = K_ref,
            .maturity = tau_prime_max,
            .rate = kappa,
            .dividend_yield = 0.0,
            .option_type = option_type},
        sigma_eff);

    auto result = solver.solve_batch(batch, /*use_shared_grid=*/false);
    if (result.results.empty() || !result.results[0].has_value()) {
        return 0.0;
    }

    const auto& sol = result.results[0].value();
    if (sol.num_snapshots() < 1) return 0.0;

    auto grid = sol.grid();
    auto x_grid = grid->x();
    auto spatial = sol.at_time(0);

    CubicSpline<double> spline;
    auto err = spline.build(x_grid, spatial);
    if (err.has_value()) return 0.0;

    double american = spline.eval(x0);
    double european = dimensionless_european(x0, tau_prime_0, kappa, option_type);
    return std::max(american - european, 0.0);
}

struct SegmentDomain {
    double x_min, x_max;
    double tp_min, tp_max;
    double lk_min, lk_max;
    double lk_min_phys, lk_max_phys;
};

struct SegmentBuildResult {
    SegmentedDimensionlessSurface::Segment segment;
    int n_pde_solves = 0;
};

std::expected<SegmentBuildResult, PriceTableError>
build_segment(const SegmentDomain& dom,
              const std::vector<double>& x_grid,
              const std::vector<double>& tp_grid,
              size_t n_lk_points,
              double K_ref, OptionType option_type)
{
    auto lk_grid = linspace(dom.lk_min, dom.lk_max, n_lk_points);
    DimensionlessAxes axes{x_grid, tp_grid, lk_grid};

    // 1. PDE solve -> raw V/K
    auto pde = solve_dimensionless_pde(axes, K_ref, option_type);
    if (!pde.has_value()) return std::unexpected(pde.error());

    // 2. EEP decompose via accessor
    Dimensionless3DAccessor accessor(pde->values, axes, K_ref);
    eep_decompose(accessor, AnalyticalEEP(option_type, 0.0));

    // 3. Fit B-spline on dollar EEP values
    std::array<std::vector<double>, 3> grids = {
        axes.log_moneyness, axes.tau_prime, axes.ln_kappa};
    auto fitter_result = BSplineNDSeparable<double, 3>::create(grids);
    if (!fitter_result.has_value()) {
        return std::unexpected(PriceTableError{PriceTableErrorCode::FittingFailed});
    }
    auto fit_result = fitter_result->fit(std::move(pde->values));
    if (!fit_result.has_value()) {
        return std::unexpected(PriceTableError{PriceTableErrorCode::FittingFailed});
    }

    // 4. Build surface with actual K_ref
    PriceTableAxesND<3> surface_axes;
    surface_axes.grids[0] = axes.log_moneyness;
    surface_axes.grids[1] = axes.tau_prime;
    surface_axes.grids[2] = axes.ln_kappa;
    surface_axes.names = {"log_moneyness", "tau_prime", "ln_kappa"};

    auto surface = PriceTableSurfaceND<3>::build(
        std::move(surface_axes), std::move(fit_result->coefficients), K_ref);
    if (!surface.has_value()) {
        return std::unexpected(PriceTableError{PriceTableErrorCode::SurfaceBuildFailed});
    }

    return SegmentBuildResult{
        .segment = {.surface = std::move(surface.value()), .lk_min = dom.lk_min_phys, .lk_max = dom.lk_max_phys},
        .n_pde_solves = pde->n_pde_solves,
    };
}

}  // anonymous namespace

// ===========================================================================
// SegmentedDimensionlessSurface::value
// ===========================================================================

double SegmentedDimensionlessSurface::value(
    const std::array<double, 3>& coords) const
{
    double lk = coords[2];

    for (size_t i = 0; i < segments_.size(); ++i) {
        if (lk <= segments_[i].lk_max || i == segments_.size() - 1) {
            double val = segments_[i].surface->value(coords);

            // Blend with next segment near upper boundary
            if (i + 1 < segments_.size()) {
                double blend_lo = segments_[i].lk_max;
                double blend_hi = segments_[i + 1].lk_min;
                if (lk >= blend_hi && lk <= blend_lo) {
                    double t = (lk - blend_hi) / (blend_lo - blend_hi);
                    double val_next = segments_[i + 1].surface->value(coords);
                    val = (1.0 - t) * val_next + t * val;
                }
            }

            return std::max(val, 0.0);
        }
    }

    return std::max(segments_.back().surface->value(coords), 0.0);
}

// ===========================================================================
// Adaptive builder
// ===========================================================================

std::expected<DimensionlessAdaptiveResult, PriceTableError>
build_dimensionless_surface_adaptive(
    const DimensionlessAdaptiveParams& params,
    double K_ref)
{
    auto t_start = std::chrono::steady_clock::now();

    constexpr size_t N_SEED = 10;
    constexpr size_t N_PROBES = 50;

    // Physical domain bounds in dimensionless coordinates
    double x_min_phys = std::log(params.moneyness_min);
    double x_max_phys = std::log(params.moneyness_max);
    double tp_min_phys = std::max(
        params.sigma_min * params.sigma_min * params.tau_min / 2.0, 0.005);
    double tp_max_phys = params.sigma_max * params.sigma_max * params.tau_max / 2.0;
    double lk_min_phys = std::log(2.0 * params.rate_min / (params.sigma_max * params.sigma_max));
    double lk_max_phys = std::log(2.0 * params.rate_max / (params.sigma_min * params.sigma_min));

    // x and tp grids with headroom (shared across all segments)
    double hx = spline_headroom(x_max_phys - x_min_phys, N_SEED);
    double htp = spline_headroom(tp_max_phys - tp_min_phys, N_SEED);
    auto x_grid = linspace(x_min_phys - hx, x_max_phys + hx, N_SEED);
    auto tp_grid = linspace(std::max(tp_min_phys - htp, 1e-3), tp_max_phys + htp, N_SEED);

    // Split ln kappa into segments
    size_t n_seg = std::max(params.lk_segments, size_t{1});
    double lk_seg_width = (lk_max_phys - lk_min_phys) / static_cast<double>(n_seg);

    std::vector<SegmentDomain> seg_domains(n_seg);
    for (size_t s = 0; s < n_seg; ++s) {
        auto& d = seg_domains[s];
        d.lk_min_phys = lk_min_phys + lk_seg_width * static_cast<double>(s);
        d.lk_max_phys = lk_min_phys + lk_seg_width * static_cast<double>(s + 1);
        double hlk = spline_headroom(d.lk_max_phys - d.lk_min_phys, N_SEED);
        d.lk_min = d.lk_min_phys - hlk;
        d.lk_max = d.lk_max_phys + hlk;
        d.x_min = x_grid.front();
        d.x_max = x_grid.back();
        d.tp_min = tp_grid.front();
        d.tp_max = tp_grid.back();
    }

    // Build all segments
    std::vector<SegmentedDimensionlessSurface::Segment> segments;
    int total_pde_solves = 0;
    DimensionlessAxes final_axes{x_grid, tp_grid, {}};

    for (size_t s = 0; s < n_seg; ++s) {
        auto seg_result = build_segment(
            seg_domains[s], x_grid, tp_grid, N_SEED, K_ref, params.option_type);
        if (!seg_result.has_value()) {
            return std::unexpected(seg_result.error());
        }
        segments.push_back(std::move(seg_result->segment));
        total_pde_solves += seg_result->n_pde_solves;
    }

    auto surface = std::make_shared<SegmentedDimensionlessSurface>(std::move(segments));

    // Adaptive refinement loop
    std::mt19937 rng(42);

    for (size_t iter = 0; iter < params.max_iter; ++iter) {
        double max_error = 0.0;
        double sum_error = 0.0;
        size_t n_probes = 0;
        std::array<double, 3> worst_loc = {};
        double worst_true = 0.0, worst_interp = 0.0;

        std::vector<double> seg_max_error(n_seg, 0.0);

        std::uniform_real_distribution<double> dist_x(x_min_phys, x_max_phys);
        std::uniform_real_distribution<double> dist_tp(tp_min_phys, tp_max_phys);
        std::uniform_real_distribution<double> dist_lk(lk_min_phys, lk_max_phys);

        for (size_t p = 0; p < N_PROBES; ++p) {
            double x0 = dist_x(rng);
            double tp0 = dist_tp(rng);
            double lk0 = dist_lk(rng);

            if (tp0 < 1e-4) continue;

            double true_eep = reference_eep(x0, tp0, lk0, K_ref, params.option_type);
            double interp_eep = surface->value({x0, tp0, lk0});
            double err = std::abs(true_eep - interp_eep);

            if (err > max_error) {
                max_error = err;
                worst_loc = {x0, tp0, lk0};
                worst_true = true_eep;
                worst_interp = interp_eep;
            }
            sum_error += err;
            n_probes++;

            for (size_t s = 0; s < n_seg; ++s) {
                if (lk0 >= seg_domains[s].lk_min_phys &&
                    lk0 <= seg_domains[s].lk_max_phys) {
                    seg_max_error[s] = std::max(seg_max_error[s], err);
                    break;
                }
            }
        }

        total_pde_solves += static_cast<int>(n_probes);
        double avg_error = n_probes > 0 ? sum_error / static_cast<double>(n_probes) : 0.0;
        bool converged = (max_error <= params.target_eep_error);

        if (converged || iter == params.max_iter - 1) {
            // Build final_axes as union of all segment lk grids
            for (const auto& seg : surface->segments()) {
                auto& lk_ax = seg.surface->axes().grids[2];
                for (double v : lk_ax) {
                    if (v >= lk_min_phys && v <= lk_max_phys)
                        final_axes.ln_kappa.push_back(v);
                }
            }
            std::sort(final_axes.ln_kappa.begin(), final_axes.ln_kappa.end());
            final_axes.ln_kappa.erase(
                std::unique(final_axes.ln_kappa.begin(), final_axes.ln_kappa.end()),
                final_axes.ln_kappa.end());

            auto t_end = std::chrono::steady_clock::now();
            return DimensionlessAdaptiveResult{
                .surface = surface,
                .final_axes = final_axes,
                .achieved_max_error = max_error,
                .achieved_avg_error = avg_error,
                .target_met = converged,
                .total_pde_solves = total_pde_solves,
                .total_build_time_seconds = std::chrono::duration<double>(t_end - t_start).count(),
                .iterations_used = iter + 1,
                .num_segments = n_seg,
                .worst_probe = worst_loc,
                .worst_true_eep = worst_true,
                .worst_interp_eep = worst_interp,
            };
        }

        // Refine worst segment
        size_t worst_seg = 0;
        double worst_seg_err = 0.0;
        for (size_t s = 0; s < n_seg; ++s) {
            if (seg_max_error[s] > worst_seg_err) {
                worst_seg_err = seg_max_error[s];
                worst_seg = s;
            }
        }

        auto& seg = surface->segments()[worst_seg];
        size_t cur_lk = seg.surface->axes().grids[2].size();
        size_t new_lk = std::min(
            static_cast<size_t>(static_cast<double>(cur_lk) * params.refinement_factor),
            params.max_points_per_dim);
        if (new_lk <= cur_lk) {
            size_t cur_x = x_grid.size();
            size_t new_x = std::min(
                static_cast<size_t>(static_cast<double>(cur_x) * params.refinement_factor),
                params.max_points_per_dim);
            if (new_x > cur_x) {
                x_grid = linspace(x_grid.front(), x_grid.back(), new_x);
            }
            size_t cur_tp = tp_grid.size();
            size_t new_tp = std::min(
                static_cast<size_t>(static_cast<double>(cur_tp) * params.refinement_factor),
                params.max_points_per_dim);
            if (new_tp > cur_tp) {
                tp_grid = linspace(tp_grid.front(), tp_grid.back(), new_tp);
            }
            new_lk = cur_lk;
        }

        // Rebuild all segments with updated grids
        std::vector<SegmentedDimensionlessSurface::Segment> new_segments;
        for (size_t s = 0; s < n_seg; ++s) {
            size_t lk_pts = (s == worst_seg) ? new_lk :
                surface->segments()[s].surface->axes().grids[2].size();
            auto seg_result = build_segment(
                seg_domains[s], x_grid, tp_grid, lk_pts, K_ref, params.option_type);
            if (!seg_result.has_value()) {
                return std::unexpected(seg_result.error());
            }
            new_segments.push_back(std::move(seg_result->segment));
            total_pde_solves += seg_result->n_pde_solves;
        }
        surface = std::make_shared<SegmentedDimensionlessSurface>(std::move(new_segments));
    }

    return std::unexpected(PriceTableError{PriceTableErrorCode::ExtractionFailed});
}

}  // namespace mango
