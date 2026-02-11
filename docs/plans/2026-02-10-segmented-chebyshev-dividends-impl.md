# Segmented Chebyshev Discrete Dividend Support — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add `build_segmented_chebyshev()` to `AdaptiveGridBuilder` — a Chebyshev-tensor
analog of the existing B-spline `build_segmented()` that supports discrete dividends via
tau-segmented V/K_ref storage (no EEP decomposition).

**Architecture:** Per-segment 4D Chebyshev tensors store V/K_ref directly, assembled with
existing `TauSegmentSplit` + `MultiKRefSplit`. A single PDE solve per (sigma, rate) pair
covers all segments — the PDE solver handles dividends internally during backward
time-stepping, and snapshots are distributed to segments.

**Tech Stack:** C++23, Bazel, GoogleTest, existing Chebyshev/split infrastructure

**Key insight:** No IC chaining between segments. The PDE solver handles discrete dividend
jumps internally. A single `BatchAmericanOptionSolver` solve from 0 to full maturity with
`set_snapshot_times()` produces correct V/K_ref at all tau values. Segment boundaries are
placed between dividend dates so CGL nodes never straddle a dividend kink.

---

### Task 1: Add ChebyshevSegmentedLeaf type alias

**Files:**
- Modify: `src/option/table/chebyshev/chebyshev_surface.hpp`
- Modify: `src/option/table/chebyshev/BUILD.bazel`

**Step 1: Add the include and type alias**

In `src/option/table/chebyshev/chebyshev_surface.hpp`, add after the existing aliases (after line 24):

```cpp
#include "mango/option/table/eep/identity_eep.hpp"

/// Leaf for segmented Chebyshev surfaces (V/K_ref, no EEP decomposition).
/// Used with TauSegmentSplit for discrete dividend support.
using ChebyshevSegmentedLeaf = EEPSurfaceAdapter<
    ChebyshevInterpolant<4, RawTensor<4>>,
    StandardTransform4D, IdentityEEP>;
```

Note: the `#include` for `identity_eep.hpp` goes after the existing includes at the top of
the file (after line 10).

**Step 2: Add BUILD dependency**

In `src/option/table/chebyshev/BUILD.bazel`, add `"//src/option/table:identity_eep"` to
the `chebyshev_surface` target's `deps` list (after line 10):

```python
    deps = [
        "//src/math/chebyshev:chebyshev_interpolant",
        "//src/math/chebyshev:raw_tensor",
        "//src/math/chebyshev:tucker_tensor",
        "//src/option/table:bounded_surface",
        "//src/option/table:analytical_eep",
        "//src/option/table:identity_eep",       # <-- NEW
        "//src/option/table:eep_surface_adapter",
        "//src/option/table:standard_transform_4d",
    ],
```

**Step 3: Verify it compiles**

Run: `bazel build //src/option/table/chebyshev:chebyshev_surface`
Expected: BUILD SUCCESS

**Step 4: Commit**

```bash
git add src/option/table/chebyshev/chebyshev_surface.hpp \
        src/option/table/chebyshev/BUILD.bazel
git commit -m "Add ChebyshevSegmentedLeaf type alias for dividend support"
```

---

### Task 2: Add compute_segment_boundaries() helper

**Files:**
- Modify: `src/option/table/adaptive_grid_builder.cpp` (anonymous namespace, ~30 lines)

**Step 1: Write the failing test**

In `tests/adaptive_grid_builder_test.cc`, add at the end (before the closing namespace):

```cpp
// ============================================================================
// Segment boundary computation (exposed via test-only header or tested
// indirectly through build_segmented_chebyshev; this test exercises the
// helper directly by duplicating its logic in test — keeps the helper static)
// ============================================================================

// Tests are written against the public API (build_segmented_chebyshev)
// which exercises compute_segment_boundaries internally.
// Direct unit test of the static helper is not possible without exposing it.
// See Task 4 integration test instead.
```

Actually, `compute_segment_boundaries` will be `static` in the anonymous namespace, so it
can't be unit-tested directly. We test it indirectly through the integration test in Task 5.

**Step 2: Implement compute_segment_boundaries**

In `src/option/table/adaptive_grid_builder.cpp`, add in the anonymous namespace (after
`total_discrete_dividends` around line 105):

```cpp
/// Compute tau-space segment boundaries from dividend schedule.
/// Returns sorted boundaries: {tau_min, tau_split_1 - kInset, tau_split_1 + kInset, ...}
/// where splits occur at tau_split = maturity - dividend.calendar_time.
/// Dividends outside (tau_min, tau_max) are ignored.
/// If no dividends fall inside range, returns {tau_min, tau_max} (single segment).
static std::vector<double> compute_segment_boundaries(
    const std::vector<Dividend>& dividends, double maturity,
    double tau_min, double tau_max)
{
    constexpr double kInset = 5e-4;  // gap around dividend in tau-space

    // Collect tau-space split points from dividends
    std::vector<double> splits;
    for (const auto& div : dividends) {
        if (div.amount <= 0.0) continue;
        double tau_split = maturity - div.calendar_time;
        if (tau_split > tau_min + 2 * kInset && tau_split < tau_max - 2 * kInset) {
            splits.push_back(tau_split);
        }
    }
    std::sort(splits.begin(), splits.end());

    // Build boundaries: tau_min, (split-inset, split+inset)..., tau_max
    std::vector<double> bounds;
    bounds.push_back(tau_min);
    for (double sp : splits) {
        bounds.push_back(sp - kInset);
        bounds.push_back(sp + kInset);
    }
    bounds.push_back(tau_max);

    return bounds;
}
```

**Step 3: Verify it compiles**

Run: `bazel build //src/option/table:adaptive_grid_builder`
Expected: BUILD SUCCESS

**Step 4: Commit**

```bash
git add src/option/table/adaptive_grid_builder.cpp
git commit -m "Add compute_segment_boundaries helper for tau segmentation"
```

---

### Task 3: Add SegmentedChebyshevBuildConfig and build_fn

This is the core implementation: the build function that creates per-segment Chebyshev
tensors from cached PDE solutions, and the refine function variant for segmented tau.

**Files:**
- Modify: `src/option/table/adaptive_grid_builder.cpp` (anonymous namespace, ~200 lines)

**Step 1: Add SegmentedChebyshevBuildConfig**

In `src/option/table/adaptive_grid_builder.cpp`, add after `ChebyshevRefinementState`
(after line 388):

```cpp
/// Config for segmented Chebyshev build (discrete dividends, no EEP)
struct SegmentedChebyshevBuildConfig {
    double K_ref;
    OptionType option_type;
    double dividend_yield;
    std::vector<Dividend> discrete_dividends;
    std::vector<double> seg_boundaries;  // from compute_segment_boundaries()
};
```

**Step 2: Add seg_boundaries to ChebyshevRefinementState**

In `ChebyshevRefinementState` (line 379), add a new field:

```cpp
struct ChebyshevRefinementState {
    size_t sigma_level = 2;
    size_t rate_level = 1;
    size_t num_m = 40;
    size_t num_tau = 15;
    size_t max_level = 6;
    double m_lo, m_hi, tau_lo, tau_hi;
    double sigma_lo, sigma_hi, rate_lo, rate_hi;
    std::vector<double> seg_boundaries;  // empty = vanilla (no segmentation)
};
```

**Step 3: Implement make_segmented_chebyshev_build_fn**

Add after `make_chebyshev_build_fn` (after line 543):

```cpp
/// Create a BuildFn for segmented Chebyshev surfaces (discrete dividends).
/// Stores V/K_ref directly (no EEP subtraction).  Per-segment Chebyshev tensors
/// are assembled into a TauSegmentSplit SplitSurface.
static BuildFn make_segmented_chebyshev_build_fn(
    PDESliceCache& cache,
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
        // Tau change invalidates all cached slices
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

        // 2. Figure out which merged tau indices belong to each segment
        const auto& seg = config.seg_boundaries;
        const size_t n_seg = seg.size() - 1;

        // Map each tau_node to a segment index
        std::vector<std::vector<size_t>> seg_tau_indices(n_seg);
        for (size_t ti = 0; ti < tau_nodes.size(); ++ti) {
            double t = tau_nodes[ti];
            // Find segment: last seg where t >= seg[s]
            size_t s = 0;
            for (size_t k = 0; k < n_seg; ++k) {
                if (t >= seg[k] && t <= seg[k + 1]) {
                    s = k;
                    break;
                }
            }
            seg_tau_indices[s].push_back(ti);
        }

        // 3. Build per-segment Chebyshev tensors storing V/K_ref directly
        const size_t Nm = m_nodes.size();
        const size_t Ns = sigma_nodes.size();
        const size_t Nr = rate_nodes.size();

        std::vector<ChebyshevSegmentedLeaf> leaves;
        leaves.reserve(n_seg);

        for (size_t s = 0; s < n_seg; ++s) {
            const auto& tau_idx = seg_tau_indices[s];
            const size_t Nt_seg = tau_idx.size();
            if (Nt_seg == 0) {
                // Empty segment — build a trivial tensor
                Domain<4> domain{
                    .lo = {m_nodes.front(), seg[s], sigma_nodes.front(), rate_nodes.front()},
                    .hi = {m_nodes.back(), seg[s+1], sigma_nodes.back(), rate_nodes.back()},
                };
                std::array<size_t, 4> num_pts = {2, 2, 2, 2};
                std::vector<double> zeros(16, 0.0);
                auto interp = ChebyshevInterpolant<4, RawTensor<4>>::
                    build_from_values(std::span<const double>(zeros), domain, num_pts);
                leaves.emplace_back(std::move(interp), StandardTransform4D{},
                                    IdentityEEP{}, config.K_ref);
                continue;
            }

            // Per-segment local tau nodes (relative to segment start)
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
                        auto* spline = cache.get_slice(sigma, rate, tau_idx[jt]);
                        if (!spline) continue;
                        for (size_t mi = 0; mi < Nm; ++mi) {
                            double m = m_nodes[mi];
                            // V/K_ref directly — spline stores V/K_ref from PDE
                            double v_over_k = spline->eval(m);
                            size_t flat = mi * (Nt_seg * Ns * Nr)
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
                build_from_values(std::span<const double>(values), domain, num_pts);

            leaves.emplace_back(std::move(interp), StandardTransform4D{},
                                IdentityEEP{}, config.K_ref);
        }

        // 4. Assemble TauSegmentSplit
        std::vector<double> tau_start(n_seg), tau_end(n_seg);
        std::vector<double> tau_min_v(n_seg), tau_max_v(n_seg);
        for (size_t s = 0; s < n_seg; ++s) {
            tau_start[s] = seg[s];
            tau_end[s] = seg[s + 1];
            tau_min_v[s] = 0.0;
            tau_max_v[s] = seg[s + 1] - seg[s];
        }

        TauSegmentSplit split(std::move(tau_start), std::move(tau_end),
                              std::move(tau_min_v), std::move(tau_max_v),
                              config.K_ref);

        using ChebTauSeg = SplitSurface<ChebyshevSegmentedLeaf, TauSegmentSplit>;
        auto surface = std::make_shared<ChebTauSeg>(
            std::move(leaves), std::move(split));

        return SurfaceHandle{
            .price = [surface](double spot, double strike, double tau,
                               double sigma, double rate) {
                return surface->price(spot, strike, tau, sigma, rate);
            },
            .pde_solves = new_solves,
        };
    };
}
```

**Step 4: Implement make_segmented_chebyshev_refine_fn**

Add after `make_chebyshev_refine_fn` (after line 594). This is a variant that generates
per-segment CGL tau nodes instead of a single CGL range.

```cpp
/// Create a RefineFn for segmented Chebyshev. Tau refinement generates
/// per-segment CGL nodes (union of per-segment grids).
static RefineFn make_segmented_chebyshev_refine_fn(
    ChebyshevRefinementState& state)
{
    return [&state](size_t worst_dim, const ErrorBins& /*error_bins*/,
                    std::vector<double>& moneyness,
                    std::vector<double>& tau,
                    std::vector<double>& vol,
                    std::vector<double>& rate) -> bool
    {
        for (size_t attempt = 0; attempt < 4; ++attempt) {
            size_t dim = (worst_dim + attempt) % 4;
            switch (dim) {
            case 0: {  // moneyness: grow CGL count
                size_t new_n = std::min(
                    static_cast<size_t>(state.num_m * 1.3), size_t{80});
                if (new_n <= state.num_m) break;
                state.num_m = new_n;
                moneyness = chebyshev_nodes(new_n, state.m_lo, state.m_hi);
                return true;
            }
            case 1: {  // tau: grow per-segment CGL count
                size_t new_n = std::min(
                    static_cast<size_t>(state.num_tau * 1.3), size_t{30});
                if (new_n <= state.num_tau) break;
                state.num_tau = new_n;
                tau.clear();
                for (size_t s = 0; s + 1 < state.seg_boundaries.size(); ++s) {
                    double lo = state.seg_boundaries[s];
                    double hi = state.seg_boundaries[s + 1];
                    for (double t : chebyshev_nodes(new_n, lo, hi))
                        tau.push_back(t);
                }
                std::sort(tau.begin(), tau.end());
                // Remove near-duplicates (nodes at segment boundary ± epsilon)
                tau.erase(std::unique(tau.begin(), tau.end(),
                    [](double a, double b) { return std::abs(a - b) < 1e-10; }),
                    tau.end());
                return true;
            }
            case 2: {  // sigma: bump CC level
                if (state.sigma_level >= state.max_level) break;
                state.sigma_level++;
                vol = cc_level_nodes(
                    state.sigma_level, state.sigma_lo, state.sigma_hi);
                return true;
            }
            case 3: {  // rate: bump CC level
                if (state.rate_level >= state.max_level) break;
                state.rate_level++;
                rate = cc_level_nodes(
                    state.rate_level, state.rate_lo, state.rate_hi);
                return true;
            }
            }
        }
        return false;
    };
}
```

**Step 5: Add required includes**

At top of `adaptive_grid_builder.cpp` (around lines 1-24), add if not already present:

```cpp
#include "mango/option/table/splits/tau_segment.hpp"
#include "mango/option/table/splits/multi_kref.hpp"
#include "mango/option/table/split_surface.hpp"
#include "mango/option/table/eep/identity_eep.hpp"
```

**Step 6: Add BUILD dependencies**

In `src/option/table/BUILD.bazel`, add to the `adaptive_grid_builder` target's `deps`
(after line 273):

```python
        ":identity_eep",
        ":split_surface",
        ":tau_segment_split",
        ":multi_kref_split",
```

**Step 7: Verify it compiles**

Run: `bazel build //src/option/table:adaptive_grid_builder`
Expected: BUILD SUCCESS

**Step 8: Commit**

```bash
git add src/option/table/adaptive_grid_builder.cpp \
        src/option/table/BUILD.bazel
git commit -m "Add segmented Chebyshev build_fn and refine_fn for dividends"
```

---

### Task 4: Add build_segmented_chebyshev() entry point

**Files:**
- Modify: `src/option/table/adaptive_grid_builder.hpp` (add declaration)
- Modify: `src/option/table/adaptive_grid_builder.cpp` (add implementation, ~100 lines)

**Step 1: Add declaration**

In `src/option/table/adaptive_grid_builder.hpp`, add after `build_chebyshev` declaration
(after line 97):

```cpp
    /// Build segmented Chebyshev surface with discrete dividend support.
    /// Uses TauSegmentSplit for tau segmentation at dividend dates and
    /// MultiKRefSplit for multi-K_ref interpolation.
    /// Stores V/K_ref directly (no EEP decomposition).
    ///
    /// @param config Segmented config with spot, dividends, maturity, K_refs
    /// @param domain IV grid providing domain bounds
    /// @return AdaptiveResult with price_fn, or error
    [[nodiscard]] std::expected<AdaptiveResult, PriceTableError>
    build_segmented_chebyshev(const SegmentedAdaptiveConfig& config,
                              const IVGrid& domain);
```

**Step 2: Implement build_segmented_chebyshev**

In `src/option/table/adaptive_grid_builder.cpp`, add before the closing `}  // namespace mango`
(before line 1743):

```cpp
std::expected<AdaptiveResult, PriceTableError>
AdaptiveGridBuilder::build_segmented_chebyshev(
    const SegmentedAdaptiveConfig& config,
    const IVGrid& domain)
{
    // 1. Determine K_refs (same logic as build_segmented, lines 1522-1542)
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

    // Expand for discrete dividend spot shifts
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

    // Chebyshev headroom
    auto hfn = [](double lo, double hi, size_t n) {
        return 3.0 * (hi - lo)
             / static_cast<double>(std::max(n, size_t{4}) - 1);
    };
    double hm = hfn(min_m, max_m, 40);
    double ht = hfn(min_tau, max_tau, 15);
    double hs = hfn(min_vol, max_vol, 15);
    double hr = hfn(min_rate, max_rate, 9);

    // 3. Compute segment boundaries
    auto seg_bounds = compute_segment_boundaries(
        config.discrete_dividends, config.maturity, min_tau, max_tau);

    // 4. Probe K_ref = spot: adaptive refinement
    ChebyshevRefinementState state{
        .sigma_level = 2, .rate_level = 1,
        .num_m = 40, .num_tau = 10,
        .max_level = 6,
        .m_lo = min_m - hm,
        .m_hi = max_m + hm,
        .tau_lo = std::max(min_tau - ht, 1e-4),
        .tau_hi = max_tau + ht,
        .sigma_lo = std::max(min_vol - hs, 0.01),
        .sigma_hi = max_vol + hs,
        .rate_lo = std::max(min_rate - hr, -0.05),
        .rate_hi = max_rate + hr,
        .seg_boundaries = seg_bounds,
    };

    PDESliceCache pde_cache;
    SegmentedChebyshevBuildConfig build_cfg{
        .K_ref = config.spot,
        .option_type = config.option_type,
        .dividend_yield = config.dividend_yield,
        .discrete_dividends = config.discrete_dividends,
        .seg_boundaries = seg_bounds,
    };

    auto build_fn = make_segmented_chebyshev_build_fn(pde_cache, build_cfg, state);
    auto refine_fn = make_segmented_chebyshev_refine_fn(state);
    auto validate_fn = make_validate_fn(
        config.dividend_yield, config.option_type, config.discrete_dividends);
    auto compute_error_fn = make_bs_vega_error_fn(params_);

    // Seed initial tau grid: union of per-segment CGL nodes
    InitialGrids initial;
    initial.moneyness = chebyshev_nodes(state.num_m, state.m_lo, state.m_hi);
    initial.tau.clear();
    for (size_t s = 0; s + 1 < seg_bounds.size(); ++s) {
        for (double t : chebyshev_nodes(state.num_tau, seg_bounds[s], seg_bounds[s + 1]))
            initial.tau.push_back(t);
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
        params_, build_fn, validate_fn, refine_fn, ctx,
        compute_error_fn, initial);
    if (!grid_result.has_value()) {
        return std::unexpected(grid_result.error());
    }
    auto& grids = grid_result.value();

    // 5. Build all K_refs with final grid sizes
    using ChebTauSeg = SplitSurface<ChebyshevSegmentedLeaf, TauSegmentSplit>;

    std::vector<ChebTauSeg> kref_surfaces;
    size_t total_solves = pde_cache.total_pde_solves();

    for (double k_ref : K_refs) {
        PDESliceCache kref_cache;
        SegmentedChebyshevBuildConfig kref_cfg{
            .K_ref = k_ref,
            .option_type = config.option_type,
            .dividend_yield = config.dividend_yield,
            .discrete_dividends = config.discrete_dividends,
            .seg_boundaries = seg_bounds,
        };

        auto kref_build_fn = make_segmented_chebyshev_build_fn(
            kref_cache, kref_cfg, state);
        auto surface = kref_build_fn(grids.moneyness, grids.tau,
                                     grids.vol, grids.rate);
        if (!surface.has_value()) {
            return std::unexpected(surface.error());
        }
        total_solves += kref_cache.total_pde_solves();

        // Extract the ChebTauSeg from the SurfaceHandle.
        // The build_fn returns a lambda over shared_ptr<ChebTauSeg>,
        // so we rebuild the surface inline here.

        // Actually — the build_fn already built the surface internally.
        // We need to store the ChebTauSeg, not just a lambda.
        // Build it again directly (same code, no caching overhead since
        // all PDE results are freshly cached).

        // Simpler approach: just collect SurfaceHandle price lambdas
        // and wrap with MultiKRefSplit at the price_fn level.
        kref_surfaces.push_back({}); // placeholder — see below
    }

    // Alternative approach: collect price_fn per K_ref and combine
    // through MultiKRefSplit manually at the lambda level.
    // This avoids extracting the concrete ChebTauSeg from SurfaceHandle.

    std::vector<std::function<double(double, double, double, double, double)>> kref_fns;
    total_solves = pde_cache.total_pde_solves();

    kref_surfaces.clear();
    kref_fns.reserve(K_refs.size());

    for (double k_ref : K_refs) {
        PDESliceCache kref_cache;
        SegmentedChebyshevBuildConfig kref_cfg{
            .K_ref = k_ref,
            .option_type = config.option_type,
            .dividend_yield = config.dividend_yield,
            .discrete_dividends = config.discrete_dividends,
            .seg_boundaries = seg_bounds,
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

    // 6. Assemble multi-K_ref via MultiKRefSplit
    AdaptiveResult result;

    if (K_refs.size() == 1) {
        result.price_fn = std::move(kref_fns[0]);
    } else {
        auto k_refs_copy = K_refs;
        auto fns = std::make_shared<std::vector<
            std::function<double(double, double, double, double, double)>>>(
            std::move(kref_fns));
        auto split = std::make_shared<MultiKRefSplit>(std::move(k_refs_copy));

        result.price_fn = [fns, split, K_refs](
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
```

**Step 3: Verify it compiles**

Run: `bazel build //src/option/table:adaptive_grid_builder`
Expected: BUILD SUCCESS

**Step 4: Run existing tests to check for regressions**

Run: `bazel test //tests:adaptive_grid_builder_test`
Expected: All existing tests pass

**Step 5: Commit**

```bash
git add src/option/table/adaptive_grid_builder.hpp \
        src/option/table/adaptive_grid_builder.cpp
git commit -m "Add build_segmented_chebyshev entry point for dividend support"
```

---

### Task 5: Add benchmark section

**Files:**
- Modify: `benchmarks/interp_iv_safety.cc` (~70 lines)

**Step 1: Add run_chebyshev_dividends function**

In `benchmarks/interp_iv_safety.cc`, add after `run_chebyshev_adaptive` (after line 701):

```cpp
// ============================================================================
// Chebyshev Adaptive — Discrete Dividends (segmented, no EEP)
// ============================================================================

static std::array<ErrorTable, kNV>
run_chebyshev_dividends(const PriceGrid& prices) {
    std::printf("\n================================================================\n");
    std::printf("Chebyshev Adaptive — Discrete Dividends (segmented)\n");
    std::printf("================================================================\n\n");

    AdaptiveGridParams params;
    params.target_iv_error = 5e-4;  // 5 bps
    params.max_iter = 6;

    SegmentedAdaptiveConfig config{
        .spot = kSpot,
        .option_type = OptionType::PUT,
        .dividend_yield = kDivYield,
        .discrete_dividends = make_div_schedule(1.0),  // 1-year maturity
        .maturity = 1.0,
        .kref_config = {.K_refs = {80.0, 100.0, 120.0}},
    };

    IVGrid domain{
        .moneyness = {std::log(kSpot / 120.0), std::log(kSpot / 115.0),
                      std::log(kSpot / 110.0), std::log(kSpot / 105.0),
                      std::log(kSpot / 100.0), std::log(kSpot / 95.0),
                      std::log(kSpot / 90.0), std::log(kSpot / 85.0),
                      std::log(kSpot / 80.0)},
        .vol = {0.10, 0.15, 0.20, 0.30, 0.50},
        .rate = {0.03, 0.05, 0.07},
    };

    std::printf("--- Building segmented Chebyshev surface (target=%.1f bps)...\n",
                params.target_iv_error * 1e4);

    AdaptiveGridBuilder builder(params);
    auto result = builder.build_segmented_chebyshev(config, domain);
    if (!result.has_value()) {
        std::fprintf(stderr, "Chebyshev dividend build failed\n");
        std::array<ErrorTable, kNV> empty{};
        return empty;
    }

    std::printf("  Iterations: %zu, PDE solves: %zu, target_met: %s\n",
                result->iterations.size(),
                result->total_pde_solves,
                result->target_met ? "yes" : "no");
    for (const auto& it : result->iterations) {
        std::printf("  iter %zu: grid [%zu, %zu, %zu, %zu] "
                    "max_err=%.1f bps avg_err=%.1f bps\n",
                    it.iteration,
                    it.grid_sizes[0], it.grid_sizes[1],
                    it.grid_sizes[2], it.grid_sizes[3],
                    it.max_error * 1e4, it.avg_error * 1e4);
    }

    std::array<ErrorTable, kNV> all_errors{};
    std::printf("--- Computing Chebyshev dividend IV errors...\n");
    for (size_t vi = 0; vi < kNV; ++vi) {
        char title[128];
        std::snprintf(title, sizeof(title),
                      "Cheb Dividend IV Error (bps) — σ=%.0f%%",
                      kVols[vi] * 100);
        all_errors[vi] = compute_errors_from_price_fn(
            prices, result->price_fn, vi);
        print_heatmap(title, all_errors[vi]);
    }
    return all_errors;
}
```

**Step 2: Wire into main()**

In `main()` (around line 766), add after the chebyshev adaptive section:

```cpp
    // Chebyshev adaptive dividends (segmented)
    auto cheb_div_errors = run_chebyshev_dividends(div_prices);
```

Add to the TV/K comparison section — add a new comparison block for dividends
(after the vanilla TV/K block around line 780):

```cpp
    std::printf("\n================================================================\n");
    std::printf("TV/K Filtered Comparison — discrete dividends\n");
    std::printf("================================================================\n");

    for (size_t vi = 0; vi < kNV; ++vi) {
        AlgoErrors algos[] = {
            {"B-spline(div)", &div_errors[vi]},
            {"Cheb(div)", &cheb_div_errors[vi]},
        };
        print_tvk_comparison(div_prices, vi, algos);
    }
```

**Step 3: Verify benchmark compiles**

Run: `bazel build //benchmarks:interp_iv_safety`
Expected: BUILD SUCCESS

**Step 4: Run the benchmark**

Run: `bazel-bin/benchmarks/interp_iv_safety`

Expected: See dividend heatmaps with sub-10 bps errors from Chebyshev path.
Note: this step is for validation only — errors may be high on first attempt
and require debugging.

**Step 5: Commit**

```bash
git add benchmarks/interp_iv_safety.cc
git commit -m "Add Chebyshev dividend benchmark to interp_iv_safety"
```

---

### Task 6: Full test suite and build verification

**Files:**
- None (verification only)

**Step 1: Run all tests**

Run: `bazel test //...`
Expected: 120+ tests pass, zero failures

**Step 2: Build all benchmarks**

Run: `bazel build //benchmarks/...`
Expected: BUILD SUCCESS

**Step 3: Build Python bindings**

Run: `bazel build //src/python:mango_option`
Expected: BUILD SUCCESS

**Step 4: Commit if any fixups were needed**

```bash
# Only if fixups were required
git add <files>
git commit -m "Fix build issues found during verification"
```

---

## Verification Checklist

1. `bazel build //src/option/table:adaptive_grid_builder` — compiles with new code
2. `bazel build //benchmarks:interp_iv_safety` — benchmark compiles
3. `bazel test //...` — no regressions (120+ tests)
4. `bazel build //benchmarks/...` — all benchmarks compile
5. `bazel build //src/python:mango_option` — Python bindings compile
6. `bazel-bin/benchmarks/interp_iv_safety` — dividend heatmaps print
7. Compare Chebyshev dividend errors vs B-spline dividend errors

## Notes for implementer

- **domain.moneyness is already log(S/K)** in this context — `IVGrid` stores log-moneyness
  when passed through the internal adaptive path. The user-facing API converts S/K → ln(S/K)
  at the boundary in `interpolated_iv_solver.cpp`, but here we build `IVGrid` directly with
  log-moneyness values.

- **PDE solves spot = strike = K_ref** — the PDE solver returns V/K_ref naturally when
  spot = K_ref. The CubicSpline in PDESliceCache stores spatial values indexed by
  log-moneyness x = ln(S/K_ref).

- **TauSegmentSplit.to_local** replaces strike with K_ref and computes local_tau = tau - tau_start.
  The normalize step multiplies by K_ref (converting V/K_ref back to V).
  **MultiKRefSplit.to_local** replaces strike with k_refs[i], and normalize divides by k_refs[i].
  The denormalize step multiplies by strike. Combined: V/K_ref × K_ref / K_ref × strike = V.

- **IdentityEEP.scale** returns `strike / K_ref`. In `EEPSurfaceAdapter.price()`:
  `eep_val * scale(strike, K_ref) + european_price(...)` = `(V/K_ref) * (strike/K_ref) + 0`.
  Wait — this gives V × strike/K_ref², not V. But when wrapped in TauSegmentSplit,
  the `normalize` step also multiplies by K_ref. So the chain is:
  raw = `(V/K_ref) * (strike/K_ref)` from EEPSurfaceAdapter, then normalize = raw × K_ref
  = V × strike / K_ref. Then MultiKRefSplit.normalize divides by k_refs[i] and
  denormalize multiplies by strike. This doesn't look right.

  **IMPORTANT**: Double-check the IdentityEEP.scale interaction with TauSegmentSplit.normalize.
  The existing B-spline segmented path uses IdentityEEP + TauSegmentSplit and works, so the
  chain must be correct. Read the existing B-spline segmented code path to verify before
  worrying. The `ChebyshevSegmentedLeaf`'s `.price()` call chain is:
  1. `to_local` sets strike = K_ref
  2. `EEPSurfaceAdapter.price(spot, K_ref, local_tau, sigma, rate)` →
     `raw * scale(K_ref, K_ref) + 0` = `raw * 1.0` = raw (V/K_ref)
  3. `TauSegmentSplit.normalize(i, strike, raw)` = `raw * K_ref_` (where K_ref_ = config K_ref)
     = V/K_ref × K_ref = V ✓

  So when `to_local` sets strike = K_ref, `IdentityEEP.scale(K_ref, K_ref) = 1.0`. Correct.
