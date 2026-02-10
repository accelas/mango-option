// SPDX-License-Identifier: MIT
#include "mango/option/table/segmented_price_table_builder.hpp"
#include "mango/option/table/dividend_utils.hpp"
#include "mango/option/table/price_table_builder.hpp"
#include "mango/option/american_option.hpp"
#include <algorithm>
#include <cmath>
#include <numeric>

namespace mango {

namespace {

constexpr int kCubicSplineDegree = 3;

/// Context for sampling a previous segment's surface as an initial condition.
struct ChainedICContext {
    std::shared_ptr<const PriceTableSurface> prev_surface;
    double K_ref;
    double prev_tau_end;   ///< Previous segment's local τ at its far boundary
    double boundary_div;   ///< Discrete dividend amount at this boundary
};

/// Generate a τ grid for a segment [tau_start, tau_end].
/// When tau_target_dt > 0, scales points proportionally to segment width.
/// Otherwise falls back to constant min_points.
std::vector<double> make_segment_tau_grid(
    double tau_start, double tau_end, int min_points,
    double tau_target_dt = 0.0, int tau_points_min = 4, int tau_points_max = 30)
{
    double seg_width = tau_end - tau_start;

    int n;
    if (tau_target_dt > 0.0) {
        // Width-proportional: wider segments get more points
        n = static_cast<int>(std::ceil(seg_width / tau_target_dt)) + 1;
        n = std::clamp(n, tau_points_min, tau_points_max);
    } else {
        // Legacy constant mode
        n = std::max(min_points, 4);
    }

    std::vector<double> grid;
    grid.reserve(static_cast<size_t>(n));

    double step = (tau_end - tau_start) / static_cast<double>(n - 1);
    for (int i = 0; i < n; ++i) {
        grid.push_back(tau_start + step * static_cast<double>(i));
    }

    return grid;
}

/// Filter dividends: delegates to shared filter_and_merge_dividends().
std::vector<Dividend> filter_dividends(
    const std::vector<Dividend>& divs, double T)
{
    return filter_and_merge_dividends(divs, T);
}

/// Append an upper guard band sized by cubic-spline support in log-moneyness.
///
/// The interpolation axis is log-moneyness. Appending enough local knot
/// intervals keeps the original upper domain away from clamped endpoint basis
/// effects and adds one diffusion-length guard for the widest segment.
void append_upper_tail_log_moneyness(std::vector<double>& log_grid,
                                     double sigma_max,
                                     double max_segment_width) {
    if (log_grid.size() < 2) return;

    std::sort(log_grid.begin(), log_grid.end());
    log_grid.erase(std::unique(log_grid.begin(), log_grid.end()), log_grid.end());
    if (log_grid.size() < 2) return;

    const size_t n = log_grid.size();
    double h_upper = 0.0;
    const size_t first = (n >= static_cast<size_t>(kCubicSplineDegree + 1))
        ? (n - static_cast<size_t>(kCubicSplineDegree + 1))
        : 0;
    for (size_t i = first; i + 1 < n; ++i) {
        h_upper = std::max(h_upper, log_grid[i + 1] - log_grid[i]);
    }
    if (!(h_upper > 0.0)) return;

    const double support_headroom =
        static_cast<double>(kCubicSplineDegree) * h_upper;
    const double diffusion_headroom =
        (sigma_max > 0.0 && max_segment_width > 0.0)
        ? (sigma_max * std::sqrt(max_segment_width))
        : 0.0;
    const double required_headroom =
        std::max(support_headroom, diffusion_headroom);
    const int tail_points = std::max(
        kCubicSplineDegree,
        static_cast<int>(std::ceil(required_headroom / h_upper)));

    const double x_max = log_grid.back();
    for (int i = 1; i <= tail_points; ++i) {
        log_grid.push_back(x_max + h_upper * static_cast<double>(i));
    }
}

}  // namespace

std::expected<SegmentedPriceSurface, PriceTableError>
SegmentedPriceTableBuilder::build(const Config& config) {
    // =====================================================================
    // Validate inputs
    // =====================================================================
    if (config.K_ref <= 0.0) {
        return std::unexpected(PriceTableError{PriceTableErrorCode::InvalidConfig});
    }
    if (config.maturity <= 0.0) {
        return std::unexpected(PriceTableError{PriceTableErrorCode::InvalidConfig});
    }
    if (config.grid.moneyness.size() < 4) {
        return std::unexpected(PriceTableError{PriceTableErrorCode::InsufficientGridPoints, 0});
    }
    if (config.grid.vol.size() < 4) {
        return std::unexpected(PriceTableError{PriceTableErrorCode::InsufficientGridPoints, 2});
    }
    if (config.grid.rate.size() < 4) {
        return std::unexpected(PriceTableError{PriceTableErrorCode::InsufficientGridPoints, 3});
    }

    const double T = config.maturity;
    const double K_ref = config.K_ref;

    // =====================================================================
    // Step 1: Filter and sort dividends
    // =====================================================================
    auto dividends = filter_dividends(config.dividends.discrete_dividends, T);

    // =====================================================================
    // Step 2: Compute segment boundaries in τ-space
    // =====================================================================
    // With N dividends at calendar times t_1 < ... < t_N the segment
    // boundaries in τ are:
    //   {0, T - t_N, T - t_{N-1}, ..., T - t_1, T}
    // Segment 0 (closest to expiry) covers [0, T - t_N].
    // Segment k covers (boundary[k], boundary[k+1]].
    std::vector<double> boundaries;
    boundaries.push_back(0.0);
    for (auto it = dividends.rbegin(); it != dividends.rend(); ++it) {
        boundaries.push_back(T - it->calendar_time);
    }
    boundaries.push_back(T);

    // Number of segments
    const size_t n_segments = boundaries.size() - 1;

    // =====================================================================
    // Step 3: Expand log-moneyness grid downward
    // =====================================================================
    std::vector<double> expanded_log_m_grid = config.grid.moneyness;
    std::sort(expanded_log_m_grid.begin(), expanded_log_m_grid.end());
    expanded_log_m_grid.erase(
        std::unique(expanded_log_m_grid.begin(), expanded_log_m_grid.end()),
        expanded_log_m_grid.end());

    if (expanded_log_m_grid.size() < 2) {
        return std::unexpected(PriceTableError{
            PriceTableErrorCode::InsufficientGridPoints, 0});
    }
    if (!std::isfinite(expanded_log_m_grid.front()) ||
        !std::isfinite(expanded_log_m_grid.back())) {
        return std::unexpected(PriceTableError{PriceTableErrorCode::InvalidConfig});
    }

    double total_div = 0.0;
    for (const auto& div : dividends) {
        total_div += div.amount;
    }

    // Expand lower side in moneyness-space, then map back to log-moneyness.
    const double x_min = expanded_log_m_grid.front();
    const double m_min = std::exp(x_min);
    double m_min_expanded = std::max(m_min - total_div / K_ref, 0.01);
    double x_min_expanded = std::log(m_min_expanded);

    if (x_min_expanded < expanded_log_m_grid.front()) {
        double step = (expanded_log_m_grid.front() - x_min_expanded) / 3.0;
        for (int i = 2; i >= 0; --i) {
            double x = x_min_expanded + step * static_cast<double>(i);
            if (x < expanded_log_m_grid.front()) {
                expanded_log_m_grid.insert(expanded_log_m_grid.begin(), x);
            }
        }
    }

    // Add right-tail headroom. Scale by one diffusion length in log-space
    // (sigma_max * sqrt(max segment width)) and at least one cubic-support
    // band so the original upper domain stays away from endpoint effects.
    double sigma_max = *std::max_element(
        config.grid.vol.begin(), config.grid.vol.end());
    double max_segment_width = 0.0;
    for (size_t i = 0; i + 1 < boundaries.size(); ++i) {
        max_segment_width =
            std::max(max_segment_width, boundaries[i + 1] - boundaries[i]);
    }
    append_upper_tail_log_moneyness(
        expanded_log_m_grid, sigma_max, max_segment_width);

    std::sort(expanded_log_m_grid.begin(), expanded_log_m_grid.end());
    expanded_log_m_grid.erase(
        std::unique(expanded_log_m_grid.begin(), expanded_log_m_grid.end()),
        expanded_log_m_grid.end());

    // Ensure at least 4 log-moneyness points
    if (expanded_log_m_grid.size() < 4) {
        return std::unexpected(PriceTableError{PriceTableErrorCode::InsufficientGridPoints, 0});
    }

    for (double x : expanded_log_m_grid) {
        if (!std::isfinite(x)) {
            return std::unexpected(PriceTableError{
                PriceTableErrorCode::InvalidConfig});
        }
    }

    // =====================================================================
    // Step 4: Build segments (last first, then backward)
    // =====================================================================
    // Build each segment's surface, then assemble into SegmentedSurface.
    // Index 0 = closest to expiry.
    std::vector<SegmentConfig> segment_configs;
    segment_configs.reserve(n_segments);

    // The "previous" surface, used to generate chained ICs for earlier segments.
    std::shared_ptr<const PriceTableSurface> prev_surface;

    for (size_t seg_idx = 0; seg_idx < n_segments; ++seg_idx) {
        double tau_start = boundaries[seg_idx];
        double tau_end = boundaries[seg_idx + 1];
        double seg_width = tau_end - tau_start;

        // Local τ grid for this segment
        auto local_tau = make_segment_tau_grid(
            0.0, seg_width, config.tau_points_per_segment,
            config.tau_target_dt, config.tau_points_min, config.tau_points_max);

        // Build PriceTableBuilderND for this segment
        auto setup = PriceTableBuilder::from_vectors(
            expanded_log_m_grid, local_tau, config.grid.vol, config.grid.rate,
            K_ref, config.pde_accuracy, config.option_type,
            config.dividends.dividend_yield);

        if (!setup.has_value()) {
            return std::unexpected(PriceTableError{PriceTableErrorCode::InvalidConfig});
        }

        auto& [builder, axes] = *setup;
        builder.set_allow_tau_zero(true);

        // ------ Manual build path (used for all segments) ------

        // 1. Create batch params
        auto batch_params = builder.make_batch(axes);

        // 2. Estimate PDE grid (same as builder.build() would)
        auto [est_grid, est_td] = builder.estimate_pde_grid(batch_params, axes);
        PDEGridSpec custom_grid = PDEGridConfig{est_grid, est_td.n_steps(), {}};

        // 3. Create batch solver with snapshot times
        BatchAmericanOptionSolver batch_solver;
        batch_solver.set_snapshot_times(axes.grids[1]);

        // 4. Build setup callback (chained segments only)
        BatchAmericanOptionSolver::SetupCallback setup_callback = nullptr;

        if (seg_idx > 0) {

            const auto& vol_grid = config.grid.vol;
            const auto& rate_grid = config.grid.rate;
            const size_t Nr = rate_grid.size();

            // τ at the boundary of the previous segment (in its local coords)
            double prev_seg_tau_local_end = boundaries[seg_idx] - boundaries[seg_idx - 1];

            // Dividend amount at this boundary.
            double boundary_div = dividends[dividends.size() - seg_idx].amount;

            ChainedICContext ic_ctx{
                .prev_surface = prev_surface,
                .K_ref = K_ref,
                .prev_tau_end = prev_seg_tau_local_end,
                .boundary_div = boundary_div,
            };

            setup_callback = [ic_ctx, &vol_grid, &rate_grid, Nr](
                size_t index, AmericanOptionSolver& solver)
            {
                double sigma = vol_grid[index / Nr];
                double rate = rate_grid[index % Nr];

                // IC maps log-moneyness x → normalized price u = V/K_ref.
                // Jump condition at dividend date: V(t⁻, S) = V(t⁺, S - D).
                solver.set_initial_condition(
                    [ic_ctx, sigma, rate](
                        std::span<const double> x, std::span<double> u)
                    {
                        for (size_t i = 0; i < x.size(); ++i) {
                            double spot = ic_ctx.K_ref * std::exp(x[i]);
                            double spot_adj = std::max(spot - ic_ctx.boundary_div, 1e-8);
                            double x_adj = std::log(spot_adj / ic_ctx.K_ref);
                            double raw = ic_ctx.prev_surface->value(
                                {x_adj, ic_ctx.prev_tau_end, sigma, rate});
                            u[i] = raw;
                        }
                    });
            };
        }

        // 5. Solve batch with estimated grid
        auto batch_result = batch_solver.solve_batch(
            batch_params, true, setup_callback, custom_grid);

        // 6. Failure rate check
        // Segment 0: strict (0.0), matching builder.build() default.
        // Chained segments: lenient (0.5), matching old behavior (no check).
        if (!batch_result.results.empty()) {
            const double max_rate = (seg_idx == 0) ? 0.0 : 0.5;
            const double failure_rate = static_cast<double>(batch_result.failed_count) /
                                        static_cast<double>(batch_result.results.size());
            if (failure_rate > max_rate) {
                return std::unexpected(PriceTableError{PriceTableErrorCode::ExtractionFailed});
            }
        }

        // 7. Extract tensor
        auto extraction = builder.extract_tensor(batch_result, axes);
        if (!extraction.has_value()) {
            return std::unexpected(PriceTableError{PriceTableErrorCode::ExtractionFailed});
        }

        // 8. Repair failures
        auto repair = builder.repair_failed_slices(
            extraction->tensor, extraction->failed_pde,
            extraction->failed_spline, axes);
        if (!repair.has_value()) {
            return std::unexpected(PriceTableError{PriceTableErrorCode::RepairFailed});
        }

        // 9. Fit B-spline coefficients
        auto fit_result = builder.fit_coeffs(extraction->tensor, axes);
        if (!fit_result.has_value()) {
            return std::unexpected(PriceTableError{PriceTableErrorCode::FittingFailed});
        }
        auto coeffs = std::move(fit_result->coefficients);

        // 10. Build PriceTableSurfaceND
        PriceTableMetadata metadata{
            .K_ref = K_ref,
            .dividends = {.dividend_yield = config.dividends.dividend_yield},
            .content = SurfaceContent::NormalizedPrice,
        };

        auto surface = PriceTableSurface::build(
            axes, std::move(coeffs), metadata);
        if (!surface.has_value()) {
            return std::unexpected(PriceTableError{PriceTableErrorCode::SurfaceBuildFailed});
        }

        auto surface_ptr = *surface;

        segment_configs.push_back(SegmentConfig{
            .surface = surface_ptr,
            .tau_start = tau_start,
            .tau_end = tau_end,
        });
        prev_surface = surface_ptr;
    }

    // =====================================================================
    // Step 5: Assemble SegmentedSurface
    // =====================================================================
    SegmentedConfig seg_config{
        .segments = std::move(segment_configs),
        .K_ref = K_ref,
    };

    return build_segmented_surface(std::move(seg_config));
}

}  // namespace mango
