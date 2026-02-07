// SPDX-License-Identifier: MIT
#include "mango/option/table/segmented_price_table_builder.hpp"
#include "mango/option/table/price_table_builder.hpp"
#include "mango/option/table/american_price_surface.hpp"
#include "mango/option/american_option.hpp"
#include <algorithm>
#include <cmath>
#include <numeric>

namespace mango {

namespace {

/// Context for sampling a previous segment's surface as an initial condition.
struct ChainedICContext {
    const AmericanPriceSurface* prev;
    double K_ref;
    double prev_tau_end;   ///< Previous segment's local τ at its far boundary
    double boundary_div;   ///< Discrete dividend amount at this boundary
    bool prev_is_eep;      ///< Whether previous segment uses EEP decomposition
};

/// Generate a τ grid for a segment [tau_start, tau_end].
/// When tau_target_dt > 0, scales points proportionally to segment width.
/// Otherwise falls back to constant min_points.
/// The first segment (closest to expiry) uses a small ε instead of exactly 0
/// to avoid PDE degeneracy.
std::vector<double> make_segment_tau_grid(
    double tau_start, double tau_end, int min_points, bool is_last_segment,
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

    double effective_start = tau_start;
    if (is_last_segment && tau_start == 0.0) {
        // For EEP mode the first tau must be > 0, but never exceed segment width
        effective_start = std::min(0.01, tau_end * 0.5);
    }

    double step = (tau_end - effective_start) / static_cast<double>(n - 1);
    for (int i = 0; i < n; ++i) {
        grid.push_back(effective_start + step * static_cast<double>(i));
    }

    return grid;
}

/// Filter dividends: keep only those strictly inside (0, T).  Sort by
/// calendar time.  Merge any duplicates at the same date.
std::vector<Dividend> filter_dividends(
    const std::vector<Dividend>& divs, double T)
{
    std::vector<Dividend> filtered;
    for (const auto& div : divs) {
        if (div.calendar_time > 0.0 && div.calendar_time < T && div.amount > 0.0) {
            filtered.push_back(div);
        }
    }
    std::sort(filtered.begin(), filtered.end(),
              [](const Dividend& a, const Dividend& b) { return a.calendar_time < b.calendar_time; });

    // Merge same-date dividends
    std::vector<Dividend> merged;
    for (const auto& div : filtered) {
        if (!merged.empty() && std::abs(merged.back().calendar_time - div.calendar_time) < 1e-12) {
            merged.back().amount += div.amount;
        } else {
            merged.push_back(div);
        }
    }
    return merged;
}

}  // namespace

std::expected<SegmentedSurface<>, PriceTableError>
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
    // Step 3: Expand moneyness grid downward
    // =====================================================================
    std::vector<double> expanded_m_grid = config.grid.moneyness;
    if (!config.skip_moneyness_expansion) {
        double total_div = 0.0;
        for (const auto& div : dividends) {
            total_div += div.amount;
        }
        double m_min_expanded = config.grid.moneyness.front() - total_div / K_ref;
        if (m_min_expanded < 0.01) m_min_expanded = 0.01;

        if (m_min_expanded < expanded_m_grid.front()) {
            // Insert extra points at the low end
            double step = (expanded_m_grid.front() - m_min_expanded) / 3.0;
            for (int i = 2; i >= 0; --i) {
                double val = m_min_expanded + step * static_cast<double>(i);
                if (val > 0.0 && val < expanded_m_grid.front()) {
                    expanded_m_grid.insert(expanded_m_grid.begin(), val);
                }
            }
        }
    }
    // Always sort and deduplicate (even when expansion is skipped,
    // the caller might pass unsorted or duplicate points)
    std::sort(expanded_m_grid.begin(), expanded_m_grid.end());
    expanded_m_grid.erase(
        std::unique(expanded_m_grid.begin(), expanded_m_grid.end()),
        expanded_m_grid.end());

    // Ensure at least 4 moneyness points
    if (expanded_m_grid.size() < 4) {
        return std::unexpected(PriceTableError{PriceTableErrorCode::InsufficientGridPoints, 0});
    }

    // Convert moneyness to log-moneyness for from_vectors() (only when
    // grid came from user-facing IVGrid, not from adaptive builder)
    if (!config.skip_moneyness_expansion) {
        for (auto& m : expanded_m_grid) {
            if (m <= 0.0) {
                return std::unexpected(PriceTableError{
                    PriceTableErrorCode::InvalidConfig});
            }
            m = std::log(m);
        }
    }

    // =====================================================================
    // Step 4: Build segments (last first, then backward)
    // =====================================================================
    // We'll store AmericanPriceSurface for each segment, then assemble.
    // Index 0 = closest to expiry.
    std::vector<SegmentConfig> segment_configs;
    segment_configs.reserve(n_segments);

    // The "previous" surface, used to generate chained ICs for earlier segments.
    AmericanPriceSurface* prev_surface_ptr = nullptr;

    for (size_t seg_idx = 0; seg_idx < n_segments; ++seg_idx) {
        double tau_start = boundaries[seg_idx];
        double tau_end = boundaries[seg_idx + 1];
        double seg_width = tau_end - tau_start;

        bool is_last_segment = (seg_idx == 0);

        // Local τ grid for this segment
        auto local_tau = make_segment_tau_grid(
            0.0, seg_width, config.tau_points_per_segment, is_last_segment,
            config.tau_target_dt, config.tau_points_min, config.tau_points_max);

        // Determine surface content mode
        SurfaceContent content = is_last_segment
            ? SurfaceContent::EarlyExercisePremium
            : SurfaceContent::RawPrice;

        // Build PriceTableBuilder for this segment
        auto setup = PriceTableBuilder<4>::from_vectors(
            expanded_m_grid, local_tau, config.grid.vol, config.grid.rate,
            K_ref, config.pde_accuracy, config.option_type,
            config.dividends.dividend_yield);

        if (!setup.has_value()) {
            return std::unexpected(PriceTableError{PriceTableErrorCode::InvalidConfig});
        }

        auto& [builder, axes] = *setup;
        builder.set_surface_content(content);

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

        if (!is_last_segment) {
            // Chained segment: τ=0 is the boundary, needs custom IC
            builder.set_allow_tau_zero(true);

            const auto& vol_grid = config.grid.vol;
            const auto& rate_grid = config.grid.rate;
            const size_t Nr = rate_grid.size();

            // τ at the boundary of the previous segment (in its local coords)
            double prev_seg_tau_local_end = boundaries[seg_idx] - boundaries[seg_idx - 1];

            // Dividend amount at this boundary.
            double boundary_div = dividends[dividends.size() - seg_idx].amount;

            AmericanPriceSurface* prev = prev_surface_ptr;
            ChainedICContext ic_ctx{
                .prev = prev,
                .K_ref = K_ref,
                .prev_tau_end = prev_seg_tau_local_end,
                .boundary_div = boundary_div,
                .prev_is_eep = (prev->metadata().content ==
                                SurfaceContent::EarlyExercisePremium),
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
                            double raw = ic_ctx.prev->price(
                                spot_adj, ic_ctx.K_ref, ic_ctx.prev_tau_end,
                                sigma, rate);
                            // EEP returns actual price V; RawPrice returns V/K_ref
                            u[i] = ic_ctx.prev_is_eep ? raw / ic_ctx.K_ref : raw;
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
            const double max_rate = is_last_segment ? 0.0 : 0.5;
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

        // 10. Build PriceTableSurface and AmericanPriceSurface
        PriceTableMetadata metadata{
            .K_ref = K_ref,
            .dividends = {.dividend_yield = config.dividends.dividend_yield},
            .content = content,
        };

        auto surface = PriceTableSurface<4>::build(
            axes, std::move(coeffs), metadata);
        if (!surface.has_value()) {
            return std::unexpected(PriceTableError{PriceTableErrorCode::SurfaceBuildFailed});
        }

        auto aps = AmericanPriceSurface::create(*surface, config.option_type);
        if (!aps.has_value()) {
            return std::unexpected(PriceTableError{PriceTableErrorCode::SurfaceBuildFailed});
        }

        segment_configs.push_back(SegmentConfig{
            .surface = std::move(*aps),
            .tau_start = tau_start,
            .tau_end = tau_end,
        });
        prev_surface_ptr = &segment_configs.back().surface;
    }

    // =====================================================================
    // Step 5: Assemble SegmentedSurface
    // =====================================================================
    SegmentedConfig seg_config{
        .segments = std::move(segment_configs),
        .dividends = dividends,
        .K_ref = K_ref,
        .T = T,
    };

    return build_segmented_surface(std::move(seg_config));
}

}  // namespace mango
