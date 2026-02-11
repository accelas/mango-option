// SPDX-License-Identifier: MIT
#include "mango/option/table/chebyshev/chebyshev_adaptive.hpp"
#include "mango/option/table/adaptive_grid_types.hpp"
#include "mango/option/table/adaptive_refinement.hpp"
#include "mango/math/chebyshev/chebyshev_nodes.hpp"
#include "mango/option/american_option_batch.hpp"
#include "mango/option/option_spec.hpp"
#include "mango/option/table/eep/eep_decomposer.hpp"
#include "mango/option/table/chebyshev/chebyshev_surface.hpp"
#include "mango/option/table/chebyshev/chebyshev_pde_cache.hpp"
#include "mango/option/table/split_surface.hpp"
#include "mango/option/dividend_utils.hpp"
#include "mango/option/table/splits/multi_kref.hpp"
#include "mango/option/table/splits/tau_segment.hpp"
#include <algorithm>
#include <any>
#include <cmath>
#include <chrono>
#include <limits>
#include <random>
#include <span>

namespace mango {

namespace {

constexpr double kMinPositive = 1e-6;

// ============================================================================
// Chebyshev refinement and build strategies
// ============================================================================

/// Config for Chebyshev build callback (shared across refinement iterations)
struct ChebyshevBuildConfig {
    double K_ref;
    OptionType option_type;
    double dividend_yield = 0.0;
};

/// State for Chebyshev CC-level refinement.
/// All 4 dimensions use Clenshaw-Curtis levels for nested node placement.
/// Node count at level l = 2^l + 1.
struct ChebyshevRefinementState {
    size_t m_level = 5;       // CC level for moneyness (initial: 33 nodes)
    size_t tau_level = 3;     // CC level for tau (initial: 9 nodes)
    size_t sigma_level = 2;   // CC level for sigma (initial: 5 nodes)
    size_t rate_level = 1;    // CC level for rate (initial: 3 nodes)
    size_t max_level = 7;     // ceiling per dimension (2^7+1 = 129 nodes)
    // Frozen extended domain bounds
    double m_lo, m_hi, tau_lo, tau_hi;
    double sigma_lo, sigma_hi, rate_lo, rate_hi;
    std::vector<double> seg_boundaries;  // empty = vanilla (no segmentation)
    std::vector<bool> seg_is_gap;        // true for synthetic dividend gap segments
};

/// Config for segmented Chebyshev build (discrete dividends, no EEP)
struct SegmentedChebyshevBuildConfig {
    double K_ref;
    OptionType option_type;
    double dividend_yield;
    std::vector<Dividend> discrete_dividends;
    std::vector<double> seg_boundaries;
    std::vector<bool> seg_is_gap;  ///< true for synthetic dividend gap segments
};

/// Create a BuildFn for the adaptive refinement loop that builds Chebyshev surfaces.
/// Reuses PDE solutions across refinement iterations via ChebyshevPDECache.
/// The last_surface side-channel captures the typed surface from each build.
static BuildFn make_chebyshev_build_fn(
    ChebyshevPDECache& cache,
    const ChebyshevBuildConfig& config,
    std::shared_ptr<ChebyshevRawSurface>& last_surface)
{
    // Track tau grid size to detect tau refinement (requires full re-solve)
    auto last_tau_size = std::make_shared<size_t>(0);

    return [&cache, config, last_tau_size, &last_surface](
        std::span<const double> m_nodes,
        std::span<const double> tau_nodes,
        std::span<const double> sigma_nodes,
        std::span<const double> rate_nodes)
        -> std::expected<SurfaceHandle, PriceTableError>
    {
        // Tau change invalidates all cached slices (tau_idx keys become stale)
        if (tau_nodes.size() != *last_tau_size) {
            cache.clear();
            *last_tau_size = tau_nodes.size();
        }

        // 1. Find missing (sigma, rate) pairs
        auto missing = cache.missing_pairs(sigma_nodes, rate_nodes);

        // 2. Batch-solve only missing pairs
        size_t new_solves = 0;
        if (!missing.empty()) {
            std::vector<PricingParams> batch;
            batch.reserve(missing.size());
            for (auto [si, ri] : missing) {
                batch.emplace_back(
                    OptionSpec{.spot = config.K_ref, .strike = config.K_ref,
                               .maturity = tau_nodes.back() * 1.01,
                               .rate = rate_nodes[ri],
                               .dividend_yield = config.dividend_yield,
                               .option_type = config.option_type},
                    sigma_nodes[si]);
            }
            BatchAmericanOptionSolver solver;
            solver.set_grid_accuracy(
                make_grid_accuracy(GridAccuracyProfile::Ultra));
            std::vector<double> tau_vec(tau_nodes.begin(), tau_nodes.end());
            solver.set_snapshot_times(std::span<const double>(tau_vec));
            auto batch_result = solver.solve_batch(
                std::span<const PricingParams>(batch), /*use_shared_grid=*/true);
            new_solves = batch.size() - batch_result.failed_count;

            for (size_t bi = 0; bi < missing.size(); ++bi) {
                auto [si, ri] = missing[bi];
                if (!batch_result.results[bi].has_value()) continue;
                const auto& result = batch_result.results[bi].value();
                auto grid = result.grid();
                auto x_grid = grid->x();
                for (size_t j = 0; j < tau_nodes.size(); ++j) {
                    auto spatial = result.at_time(j);
                    cache.store_slice(sigma_nodes[si], rate_nodes[ri],
                                      j, x_grid, spatial);
                }
            }
            cache.record_pde_solves(new_solves);
        }

        // 3. Extract EEP tensor
        const size_t Nm = m_nodes.size();
        const size_t Nt = tau_nodes.size();
        const size_t Ns = sigma_nodes.size();
        const size_t Nr = rate_nodes.size();
        std::vector<double> eep_values(Nm * Nt * Ns * Nr);

        for (size_t si = 0; si < Ns; ++si) {
            double sigma = sigma_nodes[si];
            for (size_t ri = 0; ri < Nr; ++ri) {
                double rate = rate_nodes[ri];
                for (size_t ti = 0; ti < Nt; ++ti) {
                    auto* spline = cache.get_slice(sigma, rate, ti);
                    if (!spline) continue;
                    double tau = tau_nodes[ti];
                    for (size_t mi = 0; mi < Nm; ++mi) {
                        double m = m_nodes[mi];
                        double am = spline->eval(m) * config.K_ref;
                        double spot_node = config.K_ref * std::exp(m);

                        double eep = compute_eep(
                            am, spot_node, config.K_ref, tau, sigma, rate,
                            AnalyticalEEP(config.option_type, config.dividend_yield));

                        size_t flat = mi * (Nt*Ns*Nr)
                                    + ti * (Ns*Nr)
                                    + si * Nr + ri;
                        eep_values[flat] = eep;
                    }
                }
            }
        }

        // 4. Build interpolant and wrap in surface
        Domain<4> domain{
            .lo = {m_nodes.front(), tau_nodes.front(),
                   sigma_nodes.front(), rate_nodes.front()},
            .hi = {m_nodes.back(), tau_nodes.back(),
                   sigma_nodes.back(), rate_nodes.back()},
        };
        std::array<size_t, 4> num_pts = {Nm, Nt, Ns, Nr};

        auto interp = ChebyshevInterpolant<4, RawTensor<4>>::
            build_from_values(std::span<const double>(eep_values),
                              domain, num_pts);

        ChebyshevRawTransformLeaf tleaf(
            std::move(interp), StandardTransform4D{}, config.K_ref);
        ChebyshevRawLeaf leaf(std::move(tleaf),
            AnalyticalEEP(config.option_type, config.dividend_yield));

        SurfaceBounds bounds{
            .m_min = m_nodes.front(), .m_max = m_nodes.back(),
            .tau_min = tau_nodes.front(), .tau_max = tau_nodes.back(),
            .sigma_min = sigma_nodes.front(),
            .sigma_max = sigma_nodes.back(),
            .rate_min = rate_nodes.front(),
            .rate_max = rate_nodes.back()};

        auto shared = std::make_shared<ChebyshevRawSurface>(
            std::move(leaf), bounds,
            config.option_type, config.dividend_yield);
        last_surface = shared;

        return SurfaceHandle{
            .price = [shared](double spot, double strike, double tau,
                              double sigma, double rate) {
                return shared->price(spot, strike, tau, sigma, rate);
            },
            .pde_solves = new_solves,
        };
    };
}

/// Create a BuildFn for segmented Chebyshev surfaces (discrete dividends).
/// Stores V/K_ref directly (TransformLeaf, no EEP decomposition) with local
/// tau coordinates per segment.
static BuildFn make_segmented_chebyshev_build_fn(
    ChebyshevPDECache& cache,
    const SegmentedChebyshevBuildConfig& config,
    const ChebyshevRefinementState& state)
{
    auto last_tau_size = std::make_shared<size_t>(0);

    return [&cache, &config, &state, last_tau_size](
        std::span<const double> m_nodes,
        std::span<const double> tau_nodes,
        std::span<const double> sigma_nodes,
        std::span<const double> rate_nodes)
        -> std::expected<SurfaceHandle, PriceTableError>
    {
        if (tau_nodes.size() != *last_tau_size) {
            cache.clear();
            *last_tau_size = tau_nodes.size();
        }

        // 1. Batch-solve missing (sigma, rate) pairs
        auto missing = cache.missing_pairs(sigma_nodes, rate_nodes);
        size_t new_solves = 0;
        if (!missing.empty()) {
            std::vector<PricingParams> batch;
            batch.reserve(missing.size());
            for (auto [si, ri] : missing) {
                PricingParams p(
                    OptionSpec{.spot = config.K_ref, .strike = config.K_ref,
                               .maturity = tau_nodes.back() * 1.01,
                               .rate = rate_nodes[ri],
                               .dividend_yield = config.dividend_yield,
                               .option_type = config.option_type},
                    sigma_nodes[si]);
                p.discrete_dividends = config.discrete_dividends;
                batch.push_back(std::move(p));
            }

            BatchAmericanOptionSolver solver;
            solver.set_grid_accuracy(
                make_grid_accuracy(GridAccuracyProfile::Ultra));
            std::vector<double> tau_vec(tau_nodes.begin(), tau_nodes.end());
            solver.set_snapshot_times(std::span<const double>(tau_vec));
            auto batch_result = solver.solve_batch(
                std::span<const PricingParams>(batch), true);
            new_solves = batch.size() - batch_result.failed_count;

            for (size_t bi = 0; bi < missing.size(); ++bi) {
                auto [si, ri] = missing[bi];
                if (!batch_result.results[bi].has_value()) continue;
                const auto& result = batch_result.results[bi].value();
                auto grid = result.grid();
                auto x_grid = grid->x();
                for (size_t j = 0; j < tau_nodes.size(); ++j) {
                    auto spatial = result.at_time(j);
                    cache.store_slice(sigma_nodes[si], rate_nodes[ri],
                                      j, x_grid, spatial);
                }
            }
            cache.record_pde_solves(new_solves);
        }

        // 2. Map tau nodes to segments
        const auto& seg = config.seg_boundaries;
        const size_t n_seg = seg.size() - 1;
        std::vector<std::vector<size_t>> seg_tau_indices(n_seg);
        for (size_t ti = 0; ti < tau_nodes.size(); ++ti) {
            double t = tau_nodes[ti];
            size_t s = 0;
            for (size_t k = 0; k < n_seg; ++k) {
                // Skip gap segments -- CGL nodes at boundaries belong
                // to the adjacent real segment, not the narrow gap.
                if (config.seg_is_gap[k]) continue;
                if (t >= seg[k] && t <= seg[k + 1]) {
                    s = k;
                    break;
                }
            }
            seg_tau_indices[s].push_back(ti);
        }

        // 3. Build per-segment Chebyshev tensors (V/K_ref, no EEP)
        const size_t Nm = m_nodes.size();
        const size_t Ns = sigma_nodes.size();
        const size_t Nr = rate_nodes.size();

        std::vector<ChebyshevSegmentedLeaf> leaves;
        leaves.reserve(n_seg);

        for (size_t s = 0; s < n_seg; ++s) {
            const auto& tau_idx = seg_tau_indices[s];
            const size_t Nt_seg = tau_idx.size();

            if (Nt_seg == 0) {
                Domain<4> domain{
                    .lo = {m_nodes.front(), seg[s],
                           sigma_nodes.front(), rate_nodes.front()},
                    .hi = {m_nodes.back(), seg[s + 1],
                           sigma_nodes.back(), rate_nodes.back()},
                };
                std::array<size_t, 4> num_pts = {2, 2, 2, 2};
                std::vector<double> zeros(16, 0.0);
                auto interp = ChebyshevInterpolant<4, RawTensor<4>>::
                    build_from_values(std::span<const double>(zeros),
                                      domain, num_pts);
                leaves.emplace_back(std::move(interp), StandardTransform4D{},
                                    config.K_ref);
                continue;
            }

            std::vector<double> local_tau(Nt_seg);
            for (size_t j = 0; j < Nt_seg; ++j) {
                local_tau[j] = tau_nodes[tau_idx[j]] - seg[s];
            }

            std::vector<double> values(Nm * Nt_seg * Ns * Nr, 0.0);
            for (size_t si = 0; si < Ns; ++si) {
                double sigma = sigma_nodes[si];
                for (size_t ri = 0; ri < Nr; ++ri) {
                    double rate = rate_nodes[ri];
                    for (size_t jt = 0; jt < Nt_seg; ++jt) {
                        auto* spline = cache.get_slice(
                            sigma, rate, tau_idx[jt]);
                        if (!spline) continue;
                        for (size_t mi = 0; mi < Nm; ++mi) {
                            double v_over_k = spline->eval(m_nodes[mi]);
                            size_t flat =
                                mi * (Nt_seg * Ns * Nr)
                                + jt * (Ns * Nr)
                                + si * Nr + ri;
                            values[flat] = v_over_k;
                        }
                    }
                }
            }

            Domain<4> domain{
                .lo = {m_nodes.front(), local_tau.front(),
                       sigma_nodes.front(), rate_nodes.front()},
                .hi = {m_nodes.back(), local_tau.back(),
                       sigma_nodes.back(), rate_nodes.back()},
            };
            std::array<size_t, 4> num_pts = {Nm, Nt_seg, Ns, Nr};
            auto interp = ChebyshevInterpolant<4, RawTensor<4>>::
                build_from_values(std::span<const double>(values),
                                  domain, num_pts);
            leaves.emplace_back(std::move(interp), StandardTransform4D{},
                                config.K_ref);
        }

        // 4. Direct evaluation lambda.
        // TransformLeaf: leaf.price() = interp(log(S/K), tau, sigma, r) * K/K_ref
        // Multiply by K_ref to get V * K/K_ref (homogeneity scaling).
        auto leaves_shared =
            std::make_shared<std::vector<ChebyshevSegmentedLeaf>>(
                std::move(leaves));
        auto seg_copy = std::make_shared<std::vector<double>>(
            seg.begin(), seg.end());
        auto gap_copy = std::make_shared<std::vector<bool>>(
            config.seg_is_gap.begin(), config.seg_is_gap.end());
        double K_ref = config.K_ref;

        return SurfaceHandle{
            .price = [leaves_shared, seg_copy, gap_copy, K_ref, n_seg](
                double spot, double strike, double tau,
                double sigma, double rate) {
                // Find segment for tau (reverse scan for proper boundary handling)
                const auto& bounds = *seg_copy;
                const auto& is_gap = *gap_copy;
                size_t seg_idx = 0;
                for (size_t i = n_seg; i > 0; --i) {
                    size_t j = i - 1;
                    if (j == 0 ? (tau >= bounds[j] && tau <= bounds[j + 1])
                               : (tau > bounds[j] && tau <= bounds[j + 1])) {
                        seg_idx = j;
                        break;
                    }
                }
                if (tau <= bounds.front()) seg_idx = 0;
                else if (tau >= bounds.back()) seg_idx = n_seg - 1;

                // If tau lands in a gap segment, route to nearest real
                // segment by distance from the gap midpoint.
                if (is_gap[seg_idx]) {
                    double gap_mid = (bounds[seg_idx] + bounds[seg_idx + 1]) * 0.5;
                    // Search outward for nearest non-gap segment
                    size_t left = seg_idx, right = seg_idx;
                    while (left > 0 && is_gap[left - 1]) --left;
                    if (left > 0) left = left - 1;  // non-gap to the left
                    else left = n_seg;               // sentinel: no left
                    while (right + 1 < n_seg && is_gap[right + 1]) ++right;
                    if (right + 1 < n_seg) right = right + 1;  // non-gap to the right
                    else right = n_seg;                         // sentinel: no right
                    if (left < n_seg && right < n_seg) {
                        seg_idx = (tau <= gap_mid) ? left : right;
                    } else if (left < n_seg) {
                        seg_idx = left;
                    } else if (right < n_seg) {
                        seg_idx = right;
                    }
                }

                // Local tau within segment
                double local_tau = std::clamp(
                    tau - bounds[seg_idx],
                    0.0, bounds[seg_idx + 1] - bounds[seg_idx]);

                double v_over_kref = (*leaves_shared)[seg_idx].price(
                    spot, strike, local_tau, sigma, rate);
                return v_over_kref * K_ref;
            },
            .pde_solves = new_solves,
        };
    };
}

/// Create a RefineFn for Chebyshev CC-level refinement.
/// All 4 dimensions use nested CC levels (2^l+1 nodes at level l).
///
/// Balanced strategy: use error_bins worst_dim as primary signal, but
/// prevent runaway anisotropy by refusing to bump a dimension that is
/// already >2 levels ahead of the minimum.  Falls back to bumping the
/// dimension with the lowest current level.
static RefineFn make_chebyshev_refine_fn(ChebyshevRefinementState& state) {
    return [&state](size_t worst_dim, const ErrorBins& /*error_bins*/,
                    std::vector<double>& moneyness,
                    std::vector<double>& tau,
                    std::vector<double>& vol,
                    std::vector<double>& rate) -> bool
    {
        std::array<size_t*, 4> levels = {
            &state.m_level, &state.tau_level,
            &state.sigma_level, &state.rate_level
        };
        std::array<std::vector<double>*, 4> grids = {
            &moneyness, &tau, &vol, &rate
        };
        std::array<double, 4> lo = {
            state.m_lo, state.tau_lo, state.sigma_lo, state.rate_lo
        };
        std::array<double, 4> hi = {
            state.m_hi, state.tau_hi, state.sigma_hi, state.rate_hi
        };

        // Find minimum CC level across all dimensions
        size_t min_level = state.max_level;
        for (size_t d = 0; d < 4; ++d) {
            min_level = std::min(min_level, *levels[d]);
        }

        // Try worst_dim first if it isn't too far ahead
        constexpr size_t kMaxSpread = 2;
        size_t dim = worst_dim;
        if (*levels[dim] >= state.max_level ||
            *levels[dim] > min_level + kMaxSpread) {
            // Fall back: find lowest-level dimension that can be bumped
            dim = 4;  // sentinel
            size_t lowest = state.max_level;
            for (size_t d = 0; d < 4; ++d) {
                if (*levels[d] < state.max_level && *levels[d] < lowest) {
                    lowest = *levels[d];
                    dim = d;
                }
            }
            if (dim == 4) return false;  // All maxed out
        }

        (*levels[dim])++;
        *grids[dim] = cc_level_nodes(*levels[dim], lo[dim], hi[dim]);
        return true;
    };
}

/// Create a RefineFn for segmented Chebyshev CC-level refinement.
/// Tau refinement generates per-segment CC-level nodes instead of a single
/// range.  Uses the same balanced strategy as the vanilla refine function.
static RefineFn make_segmented_chebyshev_refine_fn(
    ChebyshevRefinementState& state)
{
    return [&state](size_t worst_dim, const ErrorBins&,
                    std::vector<double>& moneyness,
                    std::vector<double>& tau,
                    std::vector<double>& vol,
                    std::vector<double>& rate) -> bool
    {
        std::array<size_t*, 4> levels = {
            &state.m_level, &state.tau_level,
            &state.sigma_level, &state.rate_level
        };

        // Find minimum CC level
        size_t min_level = state.max_level;
        for (size_t d = 0; d < 4; ++d) {
            min_level = std::min(min_level, *levels[d]);
        }

        // Balanced dimension selection (same as vanilla)
        constexpr size_t kMaxSpread = 2;
        size_t dim = worst_dim;
        if (*levels[dim] >= state.max_level ||
            *levels[dim] > min_level + kMaxSpread) {
            dim = 4;
            size_t lowest = state.max_level;
            for (size_t d = 0; d < 4; ++d) {
                if (*levels[d] < state.max_level && *levels[d] < lowest) {
                    lowest = *levels[d];
                    dim = d;
                }
            }
            if (dim == 4) return false;
        }

        (*levels[dim])++;

        switch (dim) {
        case 0:
            moneyness = cc_level_nodes(state.m_level, state.m_lo, state.m_hi);
            break;
        case 1: {
            // Per-segment CC-level tau nodes
            tau.clear();
            for (size_t s = 0; s + 1 < state.seg_boundaries.size(); ++s) {
                if (state.seg_is_gap[s]) continue;
                double seg_lo = state.seg_boundaries[s];
                double seg_hi = state.seg_boundaries[s + 1];
                for (double t : cc_level_nodes(state.tau_level, seg_lo, seg_hi))
                    tau.push_back(t);
            }
            std::sort(tau.begin(), tau.end());
            tau.erase(std::unique(tau.begin(), tau.end(),
                [](double a, double b) { return std::abs(a - b) < 1e-10; }),
                tau.end());
            break;
        }
        case 2:
            vol = cc_level_nodes(state.sigma_level, state.sigma_lo, state.sigma_hi);
            break;
        case 3:
            rate = cc_level_nodes(state.rate_level, state.rate_lo, state.rate_hi);
            break;
        }
        return true;
    };
}

}  // anonymous namespace

// ============================================================================
// Free function implementations
// ============================================================================

std::expected<ChebyshevAdaptiveResult, PriceTableError>
build_adaptive_chebyshev(
    const AdaptiveGridParams& params,
    const OptionGrid& chain, OptionType type)
{
    auto domain = extract_chain_domain(chain);
    if (!domain.has_value()) {
        return std::unexpected(domain.error());
    }
    auto ctx = std::move(*domain);
    ctx.option_type = type;

    // Initial CC levels for each dimension
    constexpr size_t kInitMLevel = 5;      // 33 nodes
    constexpr size_t kInitTauLevel = 3;    // 9 nodes
    constexpr size_t kInitSigmaLevel = 2;  // 5 nodes
    constexpr size_t kInitRateLevel = 1;   // 3 nodes

    // Frozen headroom computed from initial CC node counts
    auto hfn = [](double lo, double hi, size_t n) {
        return 3.0 * (hi - lo)
             / static_cast<double>(std::max(n, size_t{4}) - 1);
    };
    double hm = hfn(ctx.min_moneyness, ctx.max_moneyness,
                     (1u << kInitMLevel) + 1);
    double ht = hfn(ctx.min_tau, ctx.max_tau,
                     (1u << kInitTauLevel) + 1);
    double hs = hfn(ctx.min_vol, ctx.max_vol,
                     (1u << kInitSigmaLevel) + 1);
    double hr = hfn(ctx.min_rate, ctx.max_rate,
                     (1u << kInitRateLevel) + 1);

    ChebyshevRefinementState state{
        .m_level = kInitMLevel, .tau_level = kInitTauLevel,
        .sigma_level = kInitSigmaLevel, .rate_level = kInitRateLevel,
        .max_level = 7,
        .m_lo = ctx.min_moneyness - hm,
        .m_hi = ctx.max_moneyness + hm,
        .tau_lo = std::max(ctx.min_tau - ht, 1e-4),
        .tau_hi = ctx.max_tau + ht,
        .sigma_lo = std::max(ctx.min_vol - hs, 0.01),
        .sigma_hi = ctx.max_vol + hs,
        .rate_lo = std::max(ctx.min_rate - hr, -0.05),
        .rate_hi = ctx.max_rate + hr,
    };

    // Keep ctx bounds at the original chain domain (NOT the extended
    // Chebyshev domain).  ctx bounds control validation LHS sampling and
    // error-bin normalization -- both should target the market-relevant
    // region.  The CGL/CC grid nodes extend beyond ctx via state.*_lo/hi,
    // and initial.exact=true prevents seed_grid() from clipping them.

    ChebyshevPDECache pde_cache;
    ChebyshevBuildConfig build_cfg{
        .K_ref = chain.spot,
        .option_type = type,
        .dividend_yield = chain.dividend_yield,
    };

    std::shared_ptr<ChebyshevRawSurface> last_surface;
    auto build_fn = make_chebyshev_build_fn(pde_cache, build_cfg, last_surface);
    auto refine_fn = make_chebyshev_refine_fn(state);
    auto validate_fn = make_validate_fn(chain.dividend_yield, type);

    auto compute_error_fn = make_fd_vega_error_fn(params, validate_fn, type);

    // Seed initial grids: CC-level nodes for all dimensions (nested)
    InitialGrids initial;
    initial.moneyness = cc_level_nodes(state.m_level, state.m_lo, state.m_hi);
    initial.tau = cc_level_nodes(state.tau_level, state.tau_lo, state.tau_hi);
    initial.vol = cc_level_nodes(state.sigma_level, state.sigma_lo, state.sigma_hi);
    initial.rate = cc_level_nodes(state.rate_level, state.rate_lo, state.rate_hi);
    initial.exact = true;  // Preserve CC node placement

    auto grid_result = run_refinement(
        params, build_fn, validate_fn, refine_fn, ctx,
        compute_error_fn, initial);
    if (!grid_result.has_value()) {
        return std::unexpected(grid_result.error());
    }

    auto& grids = grid_result.value();

    // Build final Chebyshev surface from the converged grids.
    // The side-channel last_surface captures the typed ChebyshevRawSurface.
    auto final_handle = build_fn(grids.moneyness, grids.tau,
                                 grids.vol, grids.rate);
    if (!final_handle.has_value()) {
        return std::unexpected(final_handle.error());
    }

    ChebyshevAdaptiveResult result;
    result.surface = std::move(last_surface);
    result.iterations = std::move(grids.iterations);
    result.achieved_max_error = grids.achieved_max_error;
    result.achieved_avg_error = grids.achieved_avg_error;
    result.target_met = grids.target_met;
    result.total_pde_solves = pde_cache.total_pde_solves();

    return result;
}

std::expected<ChebyshevSegmentedAdaptiveResult, PriceTableError>
build_adaptive_chebyshev_segmented(
    const AdaptiveGridParams& params,
    const SegmentedAdaptiveConfig& config,
    const IVGrid& domain)
{
    // 1. Determine K_refs
    std::vector<double> K_refs = config.kref_config.K_refs;
    if (K_refs.empty()) {
        const int count = config.kref_config.K_ref_count;
        const double span = config.kref_config.K_ref_span;
        if (count < 1 || span <= 0.0) {
            return std::unexpected(PriceTableError{PriceTableErrorCode::InvalidConfig});
        }
        const double log_lo = std::log(1.0 - span);
        const double log_hi = std::log(1.0 + span);
        K_refs.reserve(static_cast<size_t>(count));
        if (count == 1) {
            K_refs.push_back(config.spot);
        } else {
            for (int i = 0; i < count; ++i) {
                double t = static_cast<double>(i) / static_cast<double>(count - 1);
                K_refs.push_back(config.spot * std::exp(log_lo + t * (log_hi - log_lo)));
            }
        }
    }
    std::sort(K_refs.begin(), K_refs.end());

    // 2. Domain setup
    if (domain.moneyness.empty() || domain.vol.empty() || domain.rate.empty()) {
        return std::unexpected(PriceTableError{PriceTableErrorCode::InvalidConfig});
    }

    double min_m = domain.moneyness.front();
    double max_m = domain.moneyness.back();

    double total_div = total_discrete_dividends(config.discrete_dividends, config.maturity);
    double ref_min = K_refs.front();
    double expansion = (ref_min > 0.0) ? total_div / ref_min : 0.0;
    if (expansion > 0.0) {
        double m_min_money = std::exp(min_m);
        double expanded = std::max(m_min_money - expansion, 0.01);
        min_m = std::log(expanded);
    }

    double min_vol = domain.vol.front();
    double max_vol = domain.vol.back();
    double min_rate = domain.rate.front();
    double max_rate = domain.rate.back();

    expand_domain_bounds(min_m, max_m, 0.10);
    expand_domain_bounds(min_vol, max_vol, 0.10, kMinPositive);
    expand_domain_bounds(min_rate, max_rate, 0.04);

    double min_tau = std::min(0.01, config.maturity * 0.5);
    double max_tau = config.maturity;
    expand_domain_bounds(min_tau, max_tau, 0.1, kMinPositive);
    max_tau = std::min(max_tau, config.maturity);

    // Initial CC levels for segmented path
    constexpr size_t kInitMLevel = 5;      // 33 nodes
    constexpr size_t kInitTauLevel = 3;    // 9 nodes per segment
    constexpr size_t kInitSigmaLevel = 2;  // 5 nodes
    constexpr size_t kInitRateLevel = 1;   // 3 nodes

    // Chebyshev headroom from initial CC node counts
    auto hfn = [](double lo, double hi, size_t n) {
        return 3.0 * (hi - lo)
             / static_cast<double>(std::max(n, size_t{4}) - 1);
    };
    double hm = hfn(min_m, max_m, (1u << kInitMLevel) + 1);
    double ht = hfn(min_tau, max_tau, (1u << kInitTauLevel) + 1);
    double hs = hfn(min_vol, max_vol, (1u << kInitSigmaLevel) + 1);
    double hr = hfn(min_rate, max_rate, (1u << kInitRateLevel) + 1);

    // 3. Compute segment boundaries
    auto [seg_bounds, seg_is_gap] = compute_segment_boundaries(
        config.discrete_dividends, config.maturity, min_tau, max_tau);

    // 4. Adaptive refinement at probe K_ref = spot
    ChebyshevRefinementState state{
        .m_level = kInitMLevel, .tau_level = kInitTauLevel,
        .sigma_level = kInitSigmaLevel, .rate_level = kInitRateLevel,
        .max_level = 7,
        .m_lo = min_m - hm,
        .m_hi = max_m + hm,
        .tau_lo = std::max(min_tau - ht, 1e-4),
        .tau_hi = max_tau + ht,
        .sigma_lo = std::max(min_vol - hs, 0.01),
        .sigma_hi = max_vol + hs,
        .rate_lo = std::max(min_rate - hr, -0.05),
        .rate_hi = max_rate + hr,
        .seg_boundaries = seg_bounds,
        .seg_is_gap = seg_is_gap,
    };

    ChebyshevPDECache pde_cache;
    SegmentedChebyshevBuildConfig build_cfg{
        .K_ref = config.spot,
        .option_type = config.option_type,
        .dividend_yield = config.dividend_yield,
        .discrete_dividends = config.discrete_dividends,
        .seg_boundaries = seg_bounds,
        .seg_is_gap = seg_is_gap,
    };

    auto build_fn = make_segmented_chebyshev_build_fn(pde_cache, build_cfg, state);
    auto refine_fn = make_segmented_chebyshev_refine_fn(state);
    auto validate_fn = make_validate_fn(
        config.dividend_yield, config.option_type, config.discrete_dividends);
    auto compute_error_fn = make_fd_vega_error_fn(
        params, validate_fn, config.option_type);

    // Seed initial tau: union of per-segment CC-level nodes (skip gap segments)
    InitialGrids initial;
    initial.moneyness = cc_level_nodes(state.m_level, state.m_lo, state.m_hi);
    initial.tau.clear();
    for (size_t s = 0; s + 1 < seg_bounds.size(); ++s) {
        if (seg_is_gap[s]) continue;
        double seg_lo = seg_bounds[s];
        double seg_hi = seg_bounds[s + 1];
        for (double t : cc_level_nodes(state.tau_level, seg_lo, seg_hi))
            initial.tau.push_back(t);
    }
    if (initial.tau.empty()) {
        return std::unexpected(
            PriceTableError(PriceTableErrorCode::InvalidConfig));
    }
    std::sort(initial.tau.begin(), initial.tau.end());
    initial.tau.erase(std::unique(initial.tau.begin(), initial.tau.end(),
        [](double a, double b) { return std::abs(a - b) < 1e-10; }),
        initial.tau.end());
    initial.vol = cc_level_nodes(state.sigma_level, state.sigma_lo, state.sigma_hi);
    initial.rate = cc_level_nodes(state.rate_level, state.rate_lo, state.rate_hi);
    initial.exact = true;

    RefinementContext ctx{
        .spot = config.spot,
        .dividend_yield = config.dividend_yield,
        .option_type = config.option_type,
        .min_moneyness = min_m, .max_moneyness = max_m,
        .min_tau = min_tau, .max_tau = max_tau,
        .min_vol = min_vol, .max_vol = max_vol,
        .min_rate = min_rate, .max_rate = max_rate,
    };

    auto grid_result = run_refinement(
        params, build_fn, validate_fn, refine_fn, ctx,
        compute_error_fn, initial);
    if (!grid_result.has_value()) {
        return std::unexpected(grid_result.error());
    }
    auto& grids = grid_result.value();

    // 5. Build all K_refs with final grid sizes
    std::vector<std::function<double(double, double, double, double, double)>> kref_fns;
    size_t total_solves = pde_cache.total_pde_solves();
    kref_fns.reserve(K_refs.size());

    for (double k_ref : K_refs) {
        ChebyshevPDECache kref_cache;
        SegmentedChebyshevBuildConfig kref_cfg{
            .K_ref = k_ref,
            .option_type = config.option_type,
            .dividend_yield = config.dividend_yield,
            .discrete_dividends = config.discrete_dividends,
            .seg_boundaries = seg_bounds,
            .seg_is_gap = seg_is_gap,
        };

        auto kref_build_fn = make_segmented_chebyshev_build_fn(
            kref_cache, kref_cfg, state);
        auto surface = kref_build_fn(grids.moneyness, grids.tau,
                                     grids.vol, grids.rate);
        if (!surface.has_value()) {
            return std::unexpected(surface.error());
        }
        total_solves += kref_cache.total_pde_solves();
        kref_fns.push_back(std::move(surface->price));
    }

    // 6. Assemble multi-K_ref
    ChebyshevSegmentedAdaptiveResult result;

    if (K_refs.size() == 1) {
        result.price_fn = std::move(kref_fns[0]);
    } else {
        auto fns = std::make_shared<std::vector<
            std::function<double(double, double, double, double, double)>>>(
            std::move(kref_fns));
        auto split = std::make_shared<MultiKRefSplit>(K_refs);

        result.price_fn = [fns, split](
            double spot, double strike, double tau,
            double sigma, double rate) -> double
        {
            auto br = split->bracket(spot, strike, tau, sigma, rate);
            double combined = 0.0;
            for (size_t i = 0; i < br.count; ++i) {
                auto idx = br.entries[i].index;
                auto [ls, lk, lt, lv, lr] = split->to_local(
                    idx, spot, strike, tau, sigma, rate);
                double raw = (*fns)[idx](ls, lk, lt, lv, lr);
                double norm = split->normalize(idx, strike, raw);
                combined += br.entries[i].weight * norm;
            }
            return split->denormalize(combined, spot, strike, tau, sigma, rate);
        };
    }

    result.iterations = std::move(grids.iterations);
    result.achieved_max_error = grids.achieved_max_error;
    result.achieved_avg_error = grids.achieved_avg_error;
    result.target_met = grids.target_met;
    result.total_pde_solves = total_solves;

    return result;
}

}  // namespace mango
