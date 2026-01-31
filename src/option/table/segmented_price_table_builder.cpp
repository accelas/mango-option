// SPDX-License-Identifier: MIT
#include "src/option/table/segmented_price_table_builder.hpp"
#include "src/option/table/price_table_builder.hpp"
#include "src/option/table/american_price_surface.hpp"
#include "src/option/american_option.hpp"
#include <algorithm>
#include <cmath>
#include <numeric>

namespace mango {

namespace {

/// Generate a τ grid for a segment [tau_start, tau_end] with at least
/// `min_points` points (including endpoints).  The first segment (closest
/// to expiry) uses a small ε instead of exactly 0 to avoid PDE degeneracy.
std::vector<double> make_segment_tau_grid(
    double tau_start, double tau_end, int min_points, bool is_last_segment)
{
    // Ensure at least 4 points (B-spline minimum)
    int n = std::max(min_points, 4);

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
std::vector<std::pair<double, double>> filter_dividends(
    const std::vector<std::pair<double, double>>& divs, double T)
{
    std::vector<std::pair<double, double>> filtered;
    for (auto [t, d] : divs) {
        if (t > 0.0 && t < T && d > 0.0) {
            filtered.emplace_back(t, d);
        }
    }
    std::sort(filtered.begin(), filtered.end(),
              [](const auto& a, const auto& b) { return a.first < b.first; });

    // Merge same-date dividends
    std::vector<std::pair<double, double>> merged;
    for (auto& [t, d] : filtered) {
        if (!merged.empty() && std::abs(merged.back().first - t) < 1e-12) {
            merged.back().second += d;
        } else {
            merged.emplace_back(t, d);
        }
    }
    return merged;
}

}  // namespace

std::expected<SegmentedPriceSurface, ValidationError>
SegmentedPriceTableBuilder::build(const Config& config) {
    // =====================================================================
    // Validate inputs
    // =====================================================================
    if (config.K_ref <= 0.0) {
        return std::unexpected(ValidationError{
            ValidationErrorCode::InvalidStrike, config.K_ref, 0});
    }
    if (config.maturity <= 0.0) {
        return std::unexpected(ValidationError{
            ValidationErrorCode::InvalidMaturity, config.maturity, 0});
    }
    if (config.moneyness_grid.size() < 4) {
        return std::unexpected(ValidationError{
            ValidationErrorCode::InvalidBounds, 0.0, 0});
    }
    if (config.vol_grid.size() < 4) {
        return std::unexpected(ValidationError{
            ValidationErrorCode::InvalidBounds, 0.0, 1});
    }
    if (config.rate_grid.size() < 4) {
        return std::unexpected(ValidationError{
            ValidationErrorCode::InvalidBounds, 0.0, 2});
    }

    const double T = config.maturity;
    const double K_ref = config.K_ref;

    // =====================================================================
    // Step 1: Filter and sort dividends
    // =====================================================================
    auto dividends = filter_dividends(config.dividends, T);

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
        boundaries.push_back(T - it->first);
    }
    boundaries.push_back(T);

    // Number of segments
    const size_t n_segments = boundaries.size() - 1;

    // =====================================================================
    // Step 3: Expand moneyness grid downward
    // =====================================================================
    double total_div = 0.0;
    for (auto& [t, d] : dividends) {
        total_div += d;
    }
    double m_min_expanded = config.moneyness_grid.front() - total_div / K_ref;
    if (m_min_expanded < 0.01) m_min_expanded = 0.01;

    std::vector<double> expanded_m_grid = config.moneyness_grid;
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
    // Sort and deduplicate
    std::sort(expanded_m_grid.begin(), expanded_m_grid.end());
    expanded_m_grid.erase(
        std::unique(expanded_m_grid.begin(), expanded_m_grid.end()),
        expanded_m_grid.end());

    // Ensure at least 4 moneyness points
    if (expanded_m_grid.size() < 4) {
        return std::unexpected(ValidationError{
            ValidationErrorCode::InvalidBounds, 0.0, 0});
    }

    // =====================================================================
    // Step 4: Build segments (last first, then backward)
    // =====================================================================
    // We'll store AmericanPriceSurface for each segment, then assemble.
    // Index 0 = closest to expiry.
    std::vector<SegmentedPriceSurface::Segment> segments;
    segments.reserve(n_segments);

    // The "previous" surface, used to generate chained ICs for earlier segments.
    // After building segment 0, this holds segment 0's surface; after segment 1,
    // segment 1's surface; etc.
    AmericanPriceSurface* prev_surface_ptr = nullptr;

    for (size_t seg_idx = 0; seg_idx < n_segments; ++seg_idx) {
        double tau_start = boundaries[seg_idx];
        double tau_end = boundaries[seg_idx + 1];
        double seg_width = tau_end - tau_start;

        bool is_last_segment = (seg_idx == 0);

        // Local τ grid for this segment
        auto local_tau = make_segment_tau_grid(
            0.0, seg_width, config.tau_points_per_segment, is_last_segment);

        // Determine surface content mode
        SurfaceContent content = is_last_segment
            ? SurfaceContent::EarlyExercisePremium
            : SurfaceContent::RawPrice;

        // Build PriceTableBuilder for this segment
        auto setup = PriceTableBuilder<4>::from_vectors(
            expanded_m_grid, local_tau, config.vol_grid, config.rate_grid,
            K_ref, GridAccuracyParams{}, config.option_type,
            config.dividend_yield);

        if (!setup.has_value()) {
            return std::unexpected(ValidationError{
                ValidationErrorCode::InvalidBounds,
                static_cast<double>(seg_idx), 0});
        }

        auto& [builder, axes] = *setup;
        builder.set_surface_content(content);

        if (!is_last_segment) {
            // Chained segment: τ=0 is the boundary, needs custom IC
            builder.set_allow_tau_zero(true);

            // The previous segment's surface provides the IC.
            // We need per-(sigma, rate) ICs.  The SetupCallback receives
            // the batch index and the solver; from the batch index we can
            // recover sigma and rate indices (row-major over σ × r).
            const auto& vol_grid = config.vol_grid;
            const auto& rate_grid = config.rate_grid;
            const size_t Nr = rate_grid.size();

            // τ at the boundary of the previous segment (in its local coords)
            // The previous segment covers [boundaries[seg_idx-1], boundaries[seg_idx]]
            // so its local τ_end = boundaries[seg_idx] - boundaries[seg_idx-1].
            double prev_seg_tau_local_end = boundaries[seg_idx] - boundaries[seg_idx - 1];

            // Capture prev_surface by pointer (it's in the segments vector)
            AmericanPriceSurface* prev = prev_surface_ptr;

            // Set a per-solve custom IC via the builder's SetupCallback.
            // PriceTableBuilder itself doesn't have a SetupCallback setter,
            // but it does have set_initial_condition() which sets a GLOBAL IC.
            // Since the IC doesn't know sigma/rate, we use a different approach:
            //
            // We use the global custom IC that captures the previous surface.
            // The IC function gets x values and must produce u values.
            // For RawPrice surfaces, price(spot, K_ref, tau, sigma, rate)
            // returns V/K_ref which IS the PDE normalized solution.
            //
            // Problem: the IC lambda doesn't know which sigma/rate it's being
            // called with.  The PriceTableBuilder's solve_batch passes the
            // custom_ic_ to ALL solves via a SetupCallback that doesn't
            // differentiate between (sigma, rate) combinations.
            //
            // Looking at price_table_builder.cpp line 269-274:
            //   if (custom_ic_) {
            //       auto ic = *custom_ic_;
            //       setup_cb = [ic](size_t, AmericanOptionSolver& s) {
            //           s.set_initial_condition(ic);
            //       };
            //   }
            // So the same IC is applied to all solves.  This won't work for
            // per-(sigma, rate) ICs.
            //
            // Solution: we override this by setting a SetupCallback-aware IC.
            // We'll build the batch manually and use solve_batch with a
            // per-index SetupCallback that captures sigma/rate from the index.

            // Actually, looking more closely, the SetupCallback IS per-index:
            //   setup_cb = [ic](size_t /*index*/, AmericanOptionSolver& s) { ... };
            // So we can use the index to determine sigma and rate.
            //
            // BUT: PriceTableBuilder::build() creates the SetupCallback
            // internally and doesn't expose a way to set an external one.
            // The custom_ic_ path creates a lambda that ignores the index.
            //
            // We have two options:
            //   A) Modify PriceTableBuilder to accept a SetupCallback (breaks API).
            //   B) Work around it by building manually.
            //
            // Let's use option B: we'll use the PriceTableBuilder's infrastructure
            // but with a workaround.  Since the IC function is called once per solve,
            // and the batch iterates over (sigma, rate) in a fixed order, we can use
            // a stateful IC that advances through the grid.  But that's fragile.
            //
            // Better approach: use a single IC function that evaluates the previous
            // surface.  For each x value, we compute the price for ALL possible
            // (sigma, rate) pairs -- but the IC is called per-solve with specific
            // (sigma, rate).  Wait, the IC doesn't receive sigma/rate at all.
            //
            // The cleanest workaround: use a thread-local or atomic counter to
            // track which (sigma, rate) pair is being solved.  The batch ordering
            // is deterministic: sigma_idx * Nr + rate_idx.  But with OpenMP
            // parallelism, the ordering isn't sequential.
            //
            // Best approach: The previous segment's AmericanPriceSurface with EEP
            // mode already stores the FULL American price for ANY (sigma, rate).
            // The first segment's surface works correctly.  For the chained IC,
            // we need to evaluate it.  But the IC function signature is
            // f(x, u) with no sigma/rate info.
            //
            // Final decision: Build the chained segments using BatchAmericanOptionSolver
            // directly with a per-index SetupCallback, instead of going through
            // PriceTableBuilder.  This gives us full control.
            //
            // ACTUALLY: re-reading the task description again, the simplest
            // approach that works is: set_initial_condition with a lambda that
            // uses the previous surface at the max-local-tau.  But the IC
            // doesn't know sigma or rate.
            //
            // COMPROMISE: For RawPrice chained segments, we can build a separate
            // PriceTableBuilder for each (sigma, rate) pair.  That's O(Nsigma * Nr)
            // builders, each with 1 batch entry... that's just as expensive as
            // the batch approach.
            //
            // OK, I realize from re-reading the code more carefully that there IS
            // a way.  The solve_batch in PriceTableBuilder creates a batch and
            // passes a SetupCallback.  If custom_ic_ is set, it creates a callback
            // that sets the same IC for all solves.  What if we DON'T set custom_ic_
            // and instead manipulate the solve_batch process ourselves?
            //
            // Actually, the simplest clean approach: since PriceTableBuilder's
            // build() method is monolithic, let's compose the pieces manually
            // for chained segments.  We'll:
            //   1. Create a PriceTableBuilder to get the config
            //   2. Use make_batch_internal to get the batch params
            //   3. Use a BatchAmericanOptionSolver with per-index setup callback
            //   4. Use extract_tensor_internal, repair, fit_coeffs
            //   5. Build the surface
            //
            // This requires the internal APIs that PriceTableBuilder exposes.

            // ------ Manual build for chained segment ------

            // Create batch params from the builder
            auto batch_params = builder.make_batch_for_testing(axes);

            // Create batch solver with per-index setup
            BatchAmericanOptionSolver batch_solver;
            batch_solver.set_snapshot_times(axes.grids[1]);

            // Build setup callback that sets per-solve IC
            auto setup_callback = [prev, K_ref, prev_seg_tau_local_end,
                                   &vol_grid, &rate_grid, Nr](
                size_t index, AmericanOptionSolver& solver)
            {
                size_t sigma_idx = index / Nr;
                size_t rate_idx = index % Nr;
                double sigma = vol_grid[sigma_idx];
                double rate = rate_grid[rate_idx];

                // The IC maps log-moneyness x -> normalized price u = V/K
                // prev->price(spot, K_ref, tau, sigma, rate) for EEP returns
                // actual price V.  For RawPrice it returns V/K_ref.
                // Since K_ref is the strike for all segments, we need u = V/K_ref.
                //
                // For EEP (segment 0): price() returns actual V.
                //   u[i] = price(K_ref * exp(x[i]), K_ref, tau, sigma, rate) / K_ref
                //
                // For RawPrice (later segments): price() returns V/K_ref directly.
                //   u[i] = price(K_ref * exp(x[i]), K_ref, tau, sigma, rate)
                //
                // To handle both uniformly, we check the metadata.

                bool prev_is_eep = (prev->metadata().content ==
                                    SurfaceContent::EarlyExercisePremium);

                solver.set_initial_condition(
                    [prev, K_ref, prev_seg_tau_local_end, sigma, rate,
                     prev_is_eep](
                        std::span<const double> x, std::span<double> u)
                    {
                        for (size_t i = 0; i < x.size(); ++i) {
                            double spot = K_ref * std::exp(x[i]);
                            double raw = prev->price(
                                spot, K_ref, prev_seg_tau_local_end,
                                sigma, rate);
                            if (prev_is_eep) {
                                // EEP returns actual price; normalize
                                u[i] = raw / K_ref;
                            } else {
                                // RawPrice returns V/K_ref already
                                u[i] = raw;
                            }
                        }
                    });
            };

            // Estimate grid using the builder's internal approach
            auto batch_result = batch_solver.solve_batch(
                batch_params, true, setup_callback);

            // Extract tensor
            auto extraction = builder.extract_tensor_for_testing(batch_result, axes);
            if (!extraction.has_value()) {
                return std::unexpected(ValidationError{
                    ValidationErrorCode::InvalidBounds,
                    static_cast<double>(seg_idx), 1});
            }

            // Repair failures
            auto repair = builder.repair_failed_slices_for_testing(
                extraction->tensor, extraction->failed_pde,
                extraction->failed_spline, axes);
            if (!repair.has_value()) {
                return std::unexpected(ValidationError{
                    ValidationErrorCode::InvalidBounds,
                    static_cast<double>(seg_idx), 2});
            }

            // Fit B-spline coefficients
            auto coeffs = builder.fit_coeffs_for_testing(extraction->tensor, axes);
            if (!coeffs.has_value()) {
                return std::unexpected(ValidationError{
                    ValidationErrorCode::InvalidBounds,
                    static_cast<double>(seg_idx), 3});
            }

            // Build PriceTableSurface manually
            PriceTableMetadata metadata{
                .K_ref = K_ref,
                .dividend_yield = config.dividend_yield,
                .content = content,
            };

            auto surface = PriceTableSurface<4>::build(
                axes, std::move(*coeffs), metadata);
            if (!surface.has_value()) {
                return std::unexpected(ValidationError{
                    ValidationErrorCode::InvalidBounds,
                    static_cast<double>(seg_idx), 4});
            }

            auto aps = AmericanPriceSurface::create(*surface, config.option_type);
            if (!aps.has_value()) {
                return std::unexpected(ValidationError{
                    ValidationErrorCode::InvalidBounds,
                    static_cast<double>(seg_idx), 5});
            }

            segments.push_back(SegmentedPriceSurface::Segment{
                .surface = std::move(*aps),
                .tau_start = tau_start,
                .tau_end = tau_end,
            });
            prev_surface_ptr = &segments.back().surface;

        } else {
            // Last segment (closest to expiry): standard build with EEP
            auto result = builder.build(axes);
            if (!result.has_value()) {
                return std::unexpected(ValidationError{
                    ValidationErrorCode::InvalidBounds,
                    static_cast<double>(seg_idx), 0});
            }

            auto aps = AmericanPriceSurface::create(
                result->surface, config.option_type);
            if (!aps.has_value()) {
                return std::unexpected(ValidationError{
                    ValidationErrorCode::InvalidBounds,
                    static_cast<double>(seg_idx), 1});
            }

            segments.push_back(SegmentedPriceSurface::Segment{
                .surface = std::move(*aps),
                .tau_start = tau_start,
                .tau_end = tau_end,
            });
            prev_surface_ptr = &segments.back().surface;
        }
    }

    // =====================================================================
    // Step 5: Assemble SegmentedPriceSurface
    // =====================================================================
    SegmentedPriceSurface::Config sps_config;
    sps_config.segments = std::move(segments);
    sps_config.dividends = dividends;
    sps_config.K_ref = K_ref;
    sps_config.T = T;

    return SegmentedPriceSurface::create(std::move(sps_config));
}

}  // namespace mango
