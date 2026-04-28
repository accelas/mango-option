# Chebyshev Segmented Typed Composition — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace the type-erased lambda chain in the Chebyshev segmented path with typed `SplitSurface` composition, eliminating `FDVegaLeaf` and gaining analytical vega.

**Architecture:** Extract a `build_chebyshev_segmented_pieces()` function that returns typed `ChebyshevSegmentedLeaf` vectors + `TauSegmentSplit`. Compose with `SplitSurface<..., MultiKRefSplit>` and wrap in `PriceTable`. Add the new surface to `AnyIVSolver`'s variant. Run both old and new paths side-by-side in an equivalence test before deleting the old path.

**Tech Stack:** C++23, Bazel, GoogleTest. Key files: `chebyshev_surface.hpp`, `chebyshev_adaptive.hpp`/`.cpp`, `interpolated_iv_solver.hpp`/`.cpp`, `adaptive_refinement.hpp`/`.cpp`.

**Design doc:** `docs/plans/2026-02-11-chebyshev-segmented-typed-composition.md`

---

### Task 1: Add type aliases to `chebyshev_surface.hpp`

**Files:**
- Modify: `src/option/table/chebyshev/chebyshev_surface.hpp`
- Modify: `src/option/table/chebyshev/BUILD.bazel`

**Step 1: Add includes and type aliases**

In `src/option/table/chebyshev/chebyshev_surface.hpp`, add includes for the split infrastructure and new aliases after the existing `ChebyshevSegmentedLeaf`:

```cpp
#include "mango/option/table/split_surface.hpp"
#include "mango/option/table/splits/tau_segment.hpp"
#include "mango/option/table/splits/multi_kref.hpp"
```

After line 31 (`ChebyshevSegmentedLeaf` definition), add:

```cpp
/// Tau-segmented Chebyshev surface (one leaf per inter-dividend interval)
using ChebyshevTauSegmented = SplitSurface<ChebyshevSegmentedLeaf, TauSegmentSplit>;

/// Multi-K_ref blended segmented Chebyshev surface
using ChebyshevMultiKRefInner = SplitSurface<ChebyshevTauSegmented, MultiKRefSplit>;

/// Multi-K_ref segmented Chebyshev price table (final queryable surface)
using ChebyshevMultiKRefSurface = PriceTable<ChebyshevMultiKRefInner>;
```

**Step 2: Update BUILD.bazel deps**

In `src/option/table/chebyshev/BUILD.bazel`, add deps to `chebyshev_surface` target:

```python
    deps = [
        "//src/math/chebyshev:chebyshev_interpolant",
        "//src/math/chebyshev:raw_tensor",
        "//src/math/chebyshev:tucker_tensor",
        "//src/option/table:price_table",
        "//src/option/table:analytical_eep",
        "//src/option/table:eep_layer",
        "//src/option/table:transform_leaf",
        "//src/option/table:standard_transform_4d",
        "//src/option/table:split_surface",       # NEW
        "//src/option/table:tau_segment_split",    # NEW
        "//src/option/table:multi_kref_split",     # NEW
    ],
```

**Step 3: Build to verify**

Run: `bazel build //src/option/table/chebyshev:chebyshev_surface`
Expected: SUCCESS

**Step 4: Commit**

```bash
git add src/option/table/chebyshev/chebyshev_surface.hpp src/option/table/chebyshev/BUILD.bazel
git commit -m "Add typed Chebyshev segmented surface aliases"
```

---

### Task 2: Add `make_tau_split_from_segments` helper

**Files:**
- Modify: `src/option/table/adaptive_refinement.hpp`
- Modify: `src/option/table/adaptive_refinement.cpp`

**Step 1: Write the failing test**

In `tests/adaptive_grid_builder_test.cc`, add at the bottom (before the closing namespace or last test):

```cpp
// ===========================================================================
// Tests for make_tau_split_from_segments
// ===========================================================================

TEST(MakeTauSplitTest, SingleDividendAbsorbsGap) {
    // Boundaries: [0.01, 0.4995, 0.5005, 1.0]
    // is_gap:     [false, true,   false]
    // Expected: 2 real segments, gap midpoint = 0.5 is boundary
    std::vector<double> bounds = {0.01, 0.4995, 0.5005, 1.0};
    std::vector<bool> is_gap = {false, true, false};

    auto split = make_tau_split_from_segments(bounds, is_gap, 100.0);

    // Tau in left real segment → bracket returns single entry
    auto br_left = split.bracket(100.0, 100.0, 0.3, 0.2, 0.05);
    EXPECT_EQ(br_left.count, 1u);
    EXPECT_EQ(br_left.entries[0].index, 0u);

    // Tau in right real segment
    auto br_right = split.bracket(100.0, 100.0, 0.7, 0.2, 0.05);
    EXPECT_EQ(br_right.count, 1u);
    EXPECT_EQ(br_right.entries[0].index, 1u);

    // Tau inside the gap (left half → routes to seg 0)
    auto br_gap_left = split.bracket(100.0, 100.0, 0.4999, 0.2, 0.05);
    EXPECT_EQ(br_gap_left.count, 1u);
    EXPECT_EQ(br_gap_left.entries[0].index, 0u);

    // Tau inside the gap (right half → routes to seg 1)
    auto br_gap_right = split.bracket(100.0, 100.0, 0.5001, 0.2, 0.05);
    EXPECT_EQ(br_gap_right.count, 1u);
    EXPECT_EQ(br_gap_right.entries[0].index, 1u);
}

TEST(MakeTauSplitTest, TwoDividendsTwoGaps) {
    // Boundaries: [0.01, 0.2495, 0.2505, 0.4995, 0.5005, 1.0]
    // is_gap:     [false, true,   false,  true,   false]
    // Expected: 3 real segments
    std::vector<double> bounds = {0.01, 0.2495, 0.2505, 0.4995, 0.5005, 1.0};
    std::vector<bool> is_gap = {false, true, false, true, false};

    auto split = make_tau_split_from_segments(bounds, is_gap, 100.0);

    auto br0 = split.bracket(100.0, 100.0, 0.15, 0.2, 0.05);
    EXPECT_EQ(br0.entries[0].index, 0u);

    auto br1 = split.bracket(100.0, 100.0, 0.375, 0.2, 0.05);
    EXPECT_EQ(br1.entries[0].index, 1u);

    auto br2 = split.bracket(100.0, 100.0, 0.75, 0.2, 0.05);
    EXPECT_EQ(br2.entries[0].index, 2u);
}

TEST(MakeTauSplitTest, NoGaps) {
    // No dividends → single real segment, no gaps
    std::vector<double> bounds = {0.01, 1.0};
    std::vector<bool> is_gap = {false};

    auto split = make_tau_split_from_segments(bounds, is_gap, 100.0);

    auto br = split.bracket(100.0, 100.0, 0.5, 0.2, 0.05);
    EXPECT_EQ(br.count, 1u);
    EXPECT_EQ(br.entries[0].index, 0u);
}
```

**Step 2: Run test to verify it fails**

Run: `bazel test //tests:adaptive_grid_builder_test --test_output=all --test_filter='*MakeTauSplit*'`
Expected: FAIL — `make_tau_split_from_segments` not declared.

**Step 3: Add declaration to header**

In `src/option/table/adaptive_refinement.hpp`, add include and declaration. After the `compute_segment_boundaries` declaration (line 220), add:

```cpp
/// Collapse gap segments into adjacent real segments for TauSegmentSplit.
/// Each real segment's range extends to the midpoint of its adjacent gap.
/// Only real segments are kept; gaps are absorbed.
TauSegmentSplit make_tau_split_from_segments(
    const std::vector<double>& bounds,
    const std::vector<bool>& is_gap,
    double K_ref);
```

Add include at top of `adaptive_refinement.hpp`:

```cpp
#include "mango/option/table/splits/tau_segment.hpp"
```

Update BUILD deps for `adaptive_refinement` target — add `//src/option/table:tau_segment_split` if not already present.

**Step 4: Implement in `adaptive_refinement.cpp`**

At the end of `src/option/table/adaptive_refinement.cpp` (after `compute_segment_boundaries`), add:

```cpp
TauSegmentSplit make_tau_split_from_segments(
    const std::vector<double>& bounds,
    const std::vector<bool>& is_gap,
    double K_ref)
{
    const size_t n_seg = is_gap.size();

    // Collect real segment ranges, absorbing adjacent gaps to their midpoints
    std::vector<double> tau_start, tau_end, tau_min, tau_max;

    for (size_t s = 0; s < n_seg; ++s) {
        if (is_gap[s]) continue;

        double start = bounds[s];
        double end = bounds[s + 1];

        // Absorb gap to the left (if s > 0 and left neighbor is a gap)
        if (s > 0 && is_gap[s - 1]) {
            double gap_lo = bounds[s - 1];
            double gap_hi = bounds[s];
            start = (gap_lo + gap_hi) * 0.5;
        }

        // Absorb gap to the right (if s + 1 < n_seg and right neighbor is a gap)
        if (s + 1 < n_seg && is_gap[s + 1]) {
            double gap_lo = bounds[s + 1];
            double gap_hi = bounds[s + 2];
            end = (gap_lo + gap_hi) * 0.5;
        }

        tau_start.push_back(start);
        tau_end.push_back(end);
        // Local tau: [0, seg_width]. TauSegmentSplit clamps to [tau_min, tau_max].
        tau_min.push_back(0.0);
        tau_max.push_back(bounds[s + 1] - bounds[s]);
    }

    return TauSegmentSplit(
        std::move(tau_start), std::move(tau_end),
        std::move(tau_min), std::move(tau_max), K_ref);
}
```

**Step 5: Build adaptive_refinement and run test**

Run: `bazel test //tests:adaptive_grid_builder_test --test_output=all --test_filter='*MakeTauSplit*'`
Expected: PASS — all 3 tests green.

**Step 6: Commit**

```bash
git add src/option/table/adaptive_refinement.hpp src/option/table/adaptive_refinement.cpp tests/adaptive_grid_builder_test.cc
git commit -m "Add make_tau_split_from_segments helper"
```

---

### Task 3: Add `build_chebyshev_segmented_pieces` function

**Files:**
- Modify: `src/option/table/chebyshev/chebyshev_adaptive.hpp`
- Modify: `src/option/table/chebyshev/chebyshev_adaptive.cpp`

**Step 1: Add declaration to header**

In `src/option/table/chebyshev/chebyshev_adaptive.hpp`, add after the `build_adaptive_chebyshev_segmented` declaration:

```cpp
/// Per-K_ref typed pieces for assembling a ChebyshevMultiKRefSurface.
struct ChebyshevSegmentedPieces {
    std::vector<ChebyshevSegmentedLeaf> leaves;  ///< One leaf per real segment
    TauSegmentSplit tau_split;                    ///< Gap-absorbed tau routing
};

/// Build typed Chebyshev segmented pieces from converged grids.
/// Each leaf stores V/K_ref (no EEP decomposition).
/// The TauSegmentSplit absorbs gap segments at construction time.
[[nodiscard]] std::expected<ChebyshevSegmentedPieces, PriceTableError>
build_chebyshev_segmented_pieces(
    double K_ref,
    OptionType option_type,
    double dividend_yield,
    const std::vector<Dividend>& discrete_dividends,
    const std::vector<double>& seg_bounds,
    const std::vector<bool>& seg_is_gap,
    std::span<const double> m_nodes,
    std::span<const double> tau_nodes,
    std::span<const double> sigma_nodes,
    std::span<const double> rate_nodes);
```

Add the needed includes at the top:

```cpp
#include "mango/option/table/splits/tau_segment.hpp"
#include <span>
```

**Step 2: Extract implementation from existing lambda**

In `src/option/table/chebyshev/chebyshev_adaptive.cpp`, add the new function after the anonymous namespace (after line 552). This extracts the per-K_ref logic from `make_segmented_chebyshev_build_fn` (lines 269-355 of the lambda) into a standalone function that also calls `make_tau_split_from_segments`.

```cpp
std::expected<ChebyshevSegmentedPieces, PriceTableError>
build_chebyshev_segmented_pieces(
    double K_ref,
    OptionType option_type,
    double dividend_yield,
    const std::vector<Dividend>& discrete_dividends,
    const std::vector<double>& seg_bounds,
    const std::vector<bool>& seg_is_gap,
    std::span<const double> m_nodes,
    std::span<const double> tau_nodes,
    std::span<const double> sigma_nodes,
    std::span<const double> rate_nodes)
{
    // 1. PDE solves for this K_ref
    ChebyshevPDECache cache;
    auto missing = cache.missing_pairs(sigma_nodes, rate_nodes);

    if (!missing.empty()) {
        std::vector<PricingParams> batch;
        batch.reserve(missing.size());
        for (auto [si, ri] : missing) {
            PricingParams p(
                OptionSpec{.spot = K_ref, .strike = K_ref,
                           .maturity = tau_nodes.back() * 1.01,
                           .rate = rate_nodes[ri],
                           .dividend_yield = dividend_yield,
                           .option_type = option_type},
                sigma_nodes[si]);
            p.discrete_dividends = discrete_dividends;
            batch.push_back(std::move(p));
        }

        BatchAmericanOptionSolver solver;
        solver.set_grid_accuracy(
            make_grid_accuracy(GridAccuracyProfile::Ultra));
        std::vector<double> tau_vec(tau_nodes.begin(), tau_nodes.end());
        solver.set_snapshot_times(std::span<const double>(tau_vec));
        auto batch_result = solver.solve_batch(
            std::span<const PricingParams>(batch), true);

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
    }

    // 2. Map tau nodes to segments
    const size_t n_seg = seg_bounds.size() - 1;
    std::vector<std::vector<size_t>> seg_tau_indices(n_seg);
    for (size_t ti = 0; ti < tau_nodes.size(); ++ti) {
        double t = tau_nodes[ti];
        size_t s = 0;
        for (size_t k = 0; k < n_seg; ++k) {
            if (seg_is_gap[k]) continue;
            if (t >= seg_bounds[k] && t <= seg_bounds[k + 1]) {
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

    // Track only real (non-gap) segment count for validation
    size_t real_seg_count = 0;

    for (size_t s = 0; s < n_seg; ++s) {
        if (seg_is_gap[s]) continue;  // Skip gap segments entirely

        real_seg_count++;
        const auto& tau_idx = seg_tau_indices[s];
        const size_t Nt_seg = tau_idx.size();

        if (Nt_seg == 0) {
            // Degenerate: no tau nodes in this real segment (very short segment)
            Domain<4> domain{
                .lo = {m_nodes.front(), seg_bounds[s],
                       sigma_nodes.front(), rate_nodes.front()},
                .hi = {m_nodes.back(), seg_bounds[s + 1],
                       sigma_nodes.back(), rate_nodes.back()},
            };
            std::array<size_t, 4> num_pts = {2, 2, 2, 2};
            std::vector<double> zeros(16, 0.0);
            auto interp = ChebyshevInterpolant<4, RawTensor<4>>::
                build_from_values(std::span<const double>(zeros),
                                  domain, num_pts);
            leaves.emplace_back(std::move(interp), StandardTransform4D{}, K_ref);
            continue;
        }

        std::vector<double> local_tau(Nt_seg);
        for (size_t j = 0; j < Nt_seg; ++j) {
            local_tau[j] = tau_nodes[tau_idx[j]] - seg_bounds[s];
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
        leaves.emplace_back(std::move(interp), StandardTransform4D{}, K_ref);
    }

    // 4. Build TauSegmentSplit (gap-absorbed)
    auto tau_split = make_tau_split_from_segments(seg_bounds, seg_is_gap, K_ref);

    return ChebyshevSegmentedPieces{
        .leaves = std::move(leaves),
        .tau_split = std::move(tau_split),
    };
}
```

Add include at top of `.cpp`:

```cpp
#include "mango/option/table/adaptive_refinement.hpp"  // for make_tau_split_from_segments
```

(Already included — verify.)

**Step 3: Build to verify compilation**

Run: `bazel build //src/option/table/chebyshev:chebyshev_adaptive`
Expected: SUCCESS

**Step 4: Commit**

```bash
git add src/option/table/chebyshev/chebyshev_adaptive.hpp src/option/table/chebyshev/chebyshev_adaptive.cpp
git commit -m "Add build_chebyshev_segmented_pieces function"
```

---

### Task 4: Add `build_adaptive_chebyshev_segmented_typed` (new path alongside old)

**Files:**
- Modify: `src/option/table/chebyshev/chebyshev_adaptive.hpp`
- Modify: `src/option/table/chebyshev/chebyshev_adaptive.cpp`

**Step 1: Add new result type and function declaration**

In `chebyshev_adaptive.hpp`, add after `ChebyshevSegmentedPieces`:

```cpp
/// Result of typed adaptive segmented Chebyshev surface construction.
struct ChebyshevSegmentedTypedResult {
    ChebyshevMultiKRefSurface surface;
    std::vector<IterationStats> iterations;
    double achieved_max_error = 0.0;
    double achieved_avg_error = 0.0;
    bool target_met = false;
    size_t total_pde_solves = 0;
};

/// Build typed segmented Chebyshev surface with discrete dividend support.
/// Returns a fully typed ChebyshevMultiKRefSurface (analytical vega).
[[nodiscard]] std::expected<ChebyshevSegmentedTypedResult, PriceTableError>
build_adaptive_chebyshev_segmented_typed(const AdaptiveGridParams& params,
                                         const SegmentedAdaptiveConfig& config,
                                         const IVGrid& domain);
```

**Step 2: Implement by copying `build_adaptive_chebyshev_segmented` and modifying the tail**

In `chebyshev_adaptive.cpp`, add the new function after `build_adaptive_chebyshev_segmented`. The body is identical through the adaptive refinement loop (lines 660-812). The tail (lines 815-877) changes to use `build_chebyshev_segmented_pieces` and typed composition.

The new function reuses the existing `make_segmented_chebyshev_build_fn` and `make_segmented_chebyshev_refine_fn` for the refinement loop (unchanged). Only the final assembly after `run_refinement` changes:

Replace the per-K_ref lambda collection + manual MultiKRefSplit blending with:

```cpp
    // 5. Build all K_refs with final grid sizes → typed pieces
    std::vector<ChebyshevTauSegmented> kref_surfaces;
    size_t total_solves = pde_cache.total_pde_solves();

    for (double k_ref : K_refs) {
        auto pieces = build_chebyshev_segmented_pieces(
            k_ref, config.option_type, config.dividend_yield,
            config.discrete_dividends, seg_bounds, seg_is_gap,
            grids.moneyness, grids.tau, grids.vol, grids.rate);
        if (!pieces.has_value()) {
            return std::unexpected(pieces.error());
        }
        kref_surfaces.emplace_back(
            std::move(pieces->leaves), std::move(pieces->tau_split));
    }

    // 6. Compose with MultiKRefSplit → PriceTable
    ChebyshevMultiKRefInner inner(
        std::move(kref_surfaces), MultiKRefSplit(K_refs));

    SurfaceBounds bounds{
        .m_min = min_m, .m_max = max_m,
        .tau_min = min_tau, .tau_max = max_tau,
        .sigma_min = min_vol, .sigma_max = max_vol,
        .rate_min = min_rate, .rate_max = max_rate,
    };

    ChebyshevMultiKRefSurface surface(
        std::move(inner), bounds,
        config.option_type, config.dividend_yield);

    ChebyshevSegmentedTypedResult result;
    result.surface = std::move(surface);
    result.iterations = std::move(grids.iterations);
    result.achieved_max_error = grids.achieved_max_error;
    result.achieved_avg_error = grids.achieved_avg_error;
    result.target_met = grids.target_met;
    result.total_pde_solves = total_solves;

    return result;
```

Note: the domain bound variables (`min_m`, `max_m`, etc.) are computed earlier in the same function — they are locals, not lost. Copy the identical domain-setup code from `build_adaptive_chebyshev_segmented` into the new function.

**Step 3: Build to verify compilation**

Run: `bazel build //src/option/table/chebyshev:chebyshev_adaptive`
Expected: SUCCESS

**Step 4: Commit**

```bash
git add src/option/table/chebyshev/chebyshev_adaptive.hpp src/option/table/chebyshev/chebyshev_adaptive.cpp
git commit -m "Add build_adaptive_chebyshev_segmented_typed"
```

---

### Task 5: Wire new surface into `InterpolatedIVSolver` and `AnyIVSolver`

**Files:**
- Modify: `src/option/interpolated_iv_solver.cpp`

**Step 1: Add solver instantiation**

Near line 44, after the existing `template class` declarations, add:

```cpp
template class InterpolatedIVSolver<ChebyshevMultiKRefSurface>;
```

Add the include at the top of the file:

```cpp
#include "mango/option/table/chebyshev/chebyshev_surface.hpp"
```

(May already be included — verify. The file currently pulls in `chebyshev_surface.hpp` transitively through `chebyshev_adaptive.hpp`.)

**Step 2: Add to SolverVariant**

In `AnyIVSolver::Impl` (line 184), add the new entry to the variant. Keep the old `ChebyshevSegmentedSurface` entry for now:

```cpp
    using SolverVariant = std::variant<
        InterpolatedIVSolver<BSplinePriceTable>,
        InterpolatedIVSolver<BSplineMultiKRefSurface>,
        InterpolatedIVSolver<ChebyshevSurface>,
        InterpolatedIVSolver<ChebyshevRawSurface>,
        InterpolatedIVSolver<ChebyshevSegmentedSurface>,    // OLD — keep for now
        InterpolatedIVSolver<ChebyshevMultiKRefSurface>,    // NEW
        InterpolatedIVSolver<BSpline3DPriceTable>,
        InterpolatedIVSolver<Chebyshev3DPriceTable>
    >;
```

**Step 3: Build to verify**

Run: `bazel build //src/option:interpolated_iv_solver`
Expected: SUCCESS

**Step 4: Commit**

```bash
git add src/option/interpolated_iv_solver.cpp
git commit -m "Add ChebyshevMultiKRefSurface to solver variant"
```

---

### Task 6: Equivalence test — old vs new path

**Files:**
- Modify: `tests/adaptive_grid_builder_test.cc`

This is the critical validation step. Build both the old (type-erased `price_fn`) and new (typed `ChebyshevMultiKRefSurface`) paths from identical config and compare outputs.

**Step 1: Write the equivalence test**

Add to `tests/adaptive_grid_builder_test.cc`:

```cpp
// ===========================================================================
// Equivalence: typed ChebyshevMultiKRefSurface vs type-erased price_fn
// ===========================================================================

TEST(ChebyshevSegmentedEquivalence, PriceMatchesOldPath) {
    AdaptiveGridParams params;
    params.target_iv_error = 0.005;
    params.max_iter = 2;
    params.validation_samples = 8;

    SegmentedAdaptiveConfig seg_config{
        .spot = 100.0,
        .option_type = OptionType::PUT,
        .dividend_yield = 0.02,
        .discrete_dividends = {Dividend{.calendar_time = 0.5, .amount = 2.0}},
        .maturity = 1.0,
        .kref_config = {.K_refs = {80.0, 100.0, 120.0}},
    };

    auto m_domain = to_log_m({0.8, 0.9, 1.0, 1.1, 1.2});
    std::vector<double> v_domain = {0.10, 0.20, 0.30};
    std::vector<double> r_domain = {0.03, 0.05};
    IVGrid grid{m_domain, v_domain, r_domain};

    // Build old path (type-erased price_fn)
    auto old_result = build_adaptive_chebyshev_segmented(
        params, seg_config, grid);
    ASSERT_TRUE(old_result.has_value()) << "Old path failed";

    // Build new path (typed surface)
    auto new_result = build_adaptive_chebyshev_segmented_typed(
        params, seg_config, grid);
    ASSERT_TRUE(new_result.has_value()) << "New path failed";

    // Compare prices across a grid of query points
    std::vector<double> spots = {85.0, 95.0, 100.0, 110.0};
    std::vector<double> strikes = {90.0, 100.0, 110.0};
    std::vector<double> taus = {0.1, 0.3, 0.49, 0.51, 0.7, 0.9};
    std::vector<double> sigmas = {0.15, 0.25};
    std::vector<double> rates = {0.04};

    size_t mismatch_count = 0;
    double max_abs_diff = 0.0;

    for (double spot : spots) {
        for (double K : strikes) {
            for (double tau : taus) {
                for (double sigma : sigmas) {
                    for (double rate : rates) {
                        double p_old = old_result->price_fn(
                            spot, K, tau, sigma, rate);
                        double p_new = new_result->surface.price(
                            spot, K, tau, sigma, rate);

                        double diff = std::abs(p_new - p_old);
                        max_abs_diff = std::max(max_abs_diff, diff);

                        // Tight tolerance: same PDE data, same interpolant
                        // Only difference is gap routing (midpoint vs nearest-side)
                        // which affects a ~1ms band
                        if (diff > 1e-8) {
                            mismatch_count++;
                        }
                    }
                }
            }
        }
    }

    EXPECT_EQ(mismatch_count, 0u)
        << "Price mismatch count: " << mismatch_count
        << ", max abs diff: " << max_abs_diff;
}

TEST(ChebyshevSegmentedEquivalence, VegaReasonable) {
    // New path provides analytical vega; verify it's sensible
    AdaptiveGridParams params;
    params.target_iv_error = 0.005;
    params.max_iter = 2;
    params.validation_samples = 8;

    SegmentedAdaptiveConfig seg_config{
        .spot = 100.0,
        .option_type = OptionType::PUT,
        .dividend_yield = 0.02,
        .discrete_dividends = {Dividend{.calendar_time = 0.5, .amount = 2.0}},
        .maturity = 1.0,
        .kref_config = {.K_refs = {80.0, 100.0, 120.0}},
    };

    auto m_domain = to_log_m({0.8, 0.9, 1.0, 1.1, 1.2});
    IVGrid grid{m_domain, {0.10, 0.20, 0.30}, {0.03, 0.05}};

    auto result = build_adaptive_chebyshev_segmented_typed(
        params, seg_config, grid);
    ASSERT_TRUE(result.has_value());

    // ATM put: vega should be positive and finite
    double vega = result->surface.vega(100.0, 100.0, 0.5, 0.20, 0.05);
    EXPECT_TRUE(std::isfinite(vega));
    EXPECT_GT(vega, 0.0);

    // Compare analytical vega vs FD vega (central diff)
    double eps = 1e-4;
    double p_up = result->surface.price(100.0, 100.0, 0.5, 0.20 + eps, 0.05);
    double p_dn = result->surface.price(100.0, 100.0, 0.5, 0.20 - eps, 0.05);
    double fd_vega = (p_up - p_dn) / (2.0 * eps);

    // Analytical should agree with FD within 1%
    double rel_diff = std::abs(vega - fd_vega) / std::max(std::abs(vega), 1e-6);
    EXPECT_LT(rel_diff, 0.01)
        << "Analytical vega=" << vega << " vs FD vega=" << fd_vega;
}
```

**Step 2: Run tests**

Run: `bazel test //tests:adaptive_grid_builder_test --test_output=all --test_filter='*ChebyshevSegmentedEquivalence*'`
Expected: PASS — prices match within 1e-8, vega within 1% of FD.

**Important:** If the price test fails with small but nonzero diffs, the cause is likely the gap routing change (midpoint boundary vs. nearest-side). Investigate:
- Check whether diffs only occur at tau values near dividend dates (within the gap)
- If so, the tolerance may need loosening for those specific points, or the gap midpoint boundary may need adjustment

**Step 3: Commit**

```bash
git add tests/adaptive_grid_builder_test.cc
git commit -m "Add equivalence test: typed vs type-erased Chebyshev segmented"
```

---

### Task 7: Add `build_chebyshev_segmented_manual` (non-adaptive path)

**Files:**
- Modify: `src/option/table/chebyshev/chebyshev_adaptive.hpp`
- Modify: `src/option/table/chebyshev/chebyshev_adaptive.cpp`

**Step 1: Write test**

Add to `tests/adaptive_grid_builder_test.cc`:

```cpp
TEST(ChebyshevSegmentedManual, BasicPricing) {
    SegmentedAdaptiveConfig seg_config{
        .spot = 100.0,
        .option_type = OptionType::PUT,
        .dividend_yield = 0.02,
        .discrete_dividends = {Dividend{.calendar_time = 0.5, .amount = 2.0}},
        .maturity = 1.0,
        .kref_config = {.K_refs = {80.0, 100.0, 120.0}},
    };

    auto m_domain = to_log_m({0.8, 0.9, 1.0, 1.1, 1.2});
    IVGrid grid{m_domain, {0.10, 0.20, 0.30}, {0.03, 0.05}};

    auto result = build_chebyshev_segmented_manual(
        seg_config, grid);
    ASSERT_TRUE(result.has_value()) << "Manual build failed";

    // ATM put: price should be positive and finite
    double p = result->price(100.0, 100.0, 0.5, 0.20, 0.05);
    EXPECT_TRUE(std::isfinite(p));
    EXPECT_GT(p, 0.0);

    // Vega should be positive
    double v = result->vega(100.0, 100.0, 0.5, 0.20, 0.05);
    EXPECT_TRUE(std::isfinite(v));
    EXPECT_GT(v, 0.0);
}
```

**Step 2: Run test to verify it fails**

Run: `bazel test //tests:adaptive_grid_builder_test --test_output=all --test_filter='*ChebyshevSegmentedManual*'`
Expected: FAIL — `build_chebyshev_segmented_manual` not declared.

**Step 3: Add declaration**

In `chebyshev_adaptive.hpp`:

```cpp
/// Build typed segmented Chebyshev surface from explicit CC levels (no adaptive refinement).
/// Used for benchmarking with fixed grid sizes.
[[nodiscard]] std::expected<ChebyshevMultiKRefSurface, PriceTableError>
build_chebyshev_segmented_manual(
    const SegmentedAdaptiveConfig& config,
    const IVGrid& domain,
    std::array<size_t, 4> cc_levels = {5, 3, 2, 1});
```

**Step 4: Implement**

In `chebyshev_adaptive.cpp`, add after `build_adaptive_chebyshev_segmented_typed`:

```cpp
std::expected<ChebyshevMultiKRefSurface, PriceTableError>
build_chebyshev_segmented_manual(
    const SegmentedAdaptiveConfig& config,
    const IVGrid& domain,
    std::array<size_t, 4> cc_levels)
{
    // 1. Domain setup (same as adaptive path)
    if (domain.moneyness.empty() || domain.vol.empty() || domain.rate.empty()) {
        return std::unexpected(PriceTableError{PriceTableErrorCode::InvalidConfig});
    }

    double min_m = domain.moneyness.front();
    double max_m = domain.moneyness.back();

    double total_div = total_discrete_dividends(config.discrete_dividends, config.maturity);
    double ref_min = config.kref_config.K_refs.empty()
        ? config.spot : config.kref_config.K_refs.front();
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

    // Headroom
    auto hfn = [](double lo, double hi, size_t n) {
        return 3.0 * (hi - lo)
             / static_cast<double>(std::max(n, size_t{4}) - 1);
    };
    double hm = hfn(min_m, max_m, (1u << cc_levels[0]) + 1);
    double ht = hfn(min_tau, max_tau, (1u << cc_levels[1]) + 1);
    double hs = hfn(min_vol, max_vol, (1u << cc_levels[2]) + 1);
    double hr = hfn(min_rate, max_rate, (1u << cc_levels[3]) + 1);

    double m_lo = min_m - hm, m_hi = max_m + hm;
    double tau_lo = std::max(min_tau - ht, 1e-4), tau_hi = max_tau + ht;
    double sigma_lo = std::max(min_vol - hs, 0.01), sigma_hi = max_vol + hs;
    double rate_lo = std::max(min_rate - hr, -0.05), rate_hi = max_rate + hr;

    // 2. Segment boundaries
    auto [seg_bounds, seg_is_gap] = compute_segment_boundaries(
        config.discrete_dividends, config.maturity, min_tau, max_tau);

    // 3. Generate CGL grids at fixed CC levels
    auto m_nodes = cc_level_nodes(cc_levels[0], m_lo, m_hi);

    std::vector<double> tau_nodes;
    for (size_t s = 0; s + 1 < seg_bounds.size(); ++s) {
        if (seg_is_gap[s]) continue;
        for (double t : cc_level_nodes(cc_levels[1], seg_bounds[s], seg_bounds[s + 1]))
            tau_nodes.push_back(t);
    }
    std::sort(tau_nodes.begin(), tau_nodes.end());
    tau_nodes.erase(std::unique(tau_nodes.begin(), tau_nodes.end(),
        [](double a, double b) { return std::abs(a - b) < 1e-10; }),
        tau_nodes.end());

    auto sigma_nodes = cc_level_nodes(cc_levels[2], sigma_lo, sigma_hi);
    auto rate_nodes = cc_level_nodes(cc_levels[3], rate_lo, rate_hi);

    if (tau_nodes.empty()) {
        return std::unexpected(PriceTableError(PriceTableErrorCode::InvalidConfig));
    }

    // 4. Determine K_refs
    std::vector<double> K_refs = config.kref_config.K_refs;
    if (K_refs.empty()) {
        K_refs.push_back(config.spot);
    }
    std::sort(K_refs.begin(), K_refs.end());

    // 5. Build per-K_ref typed pieces
    std::vector<ChebyshevTauSegmented> kref_surfaces;
    for (double k_ref : K_refs) {
        auto pieces = build_chebyshev_segmented_pieces(
            k_ref, config.option_type, config.dividend_yield,
            config.discrete_dividends, seg_bounds, seg_is_gap,
            m_nodes, tau_nodes, sigma_nodes, rate_nodes);
        if (!pieces.has_value()) {
            return std::unexpected(pieces.error());
        }
        kref_surfaces.emplace_back(
            std::move(pieces->leaves), std::move(pieces->tau_split));
    }

    // 6. Compose
    ChebyshevMultiKRefInner inner(
        std::move(kref_surfaces), MultiKRefSplit(K_refs));

    SurfaceBounds bounds{
        .m_min = min_m, .m_max = max_m,
        .tau_min = min_tau, .tau_max = max_tau,
        .sigma_min = min_vol, .sigma_max = max_vol,
        .rate_min = min_rate, .rate_max = max_rate,
    };

    return ChebyshevMultiKRefSurface(
        std::move(inner), bounds,
        config.option_type, config.dividend_yield);
}
```

Note: `kMinPositive` is defined in the anonymous namespace in `chebyshev_adaptive.cpp` — either move it above the new function or re-declare. Since the new function is outside the anonymous namespace, define a local constant or use `1e-6` directly.

**Step 5: Run test**

Run: `bazel test //tests:adaptive_grid_builder_test --test_output=all --test_filter='*ChebyshevSegmentedManual*'`
Expected: PASS

**Step 6: Commit**

```bash
git add src/option/table/chebyshev/chebyshev_adaptive.hpp src/option/table/chebyshev/chebyshev_adaptive.cpp tests/adaptive_grid_builder_test.cc
git commit -m "Add build_chebyshev_segmented_manual for non-adaptive path"
```

---

### Task 8: Wire factory to use new typed path

**Files:**
- Modify: `src/option/interpolated_iv_solver.cpp`

**Step 1: Modify `build_chebyshev_segmented` factory function**

Replace the body of `build_chebyshev_segmented` (lines 448-499) to use the new typed path. The function currently constructs `FDVegaLeaf` + `ChebyshevSegmentedSurface`. Change it to call `build_adaptive_chebyshev_segmented_typed` or `build_chebyshev_segmented_manual` and construct `InterpolatedIVSolver<ChebyshevMultiKRefSurface>`.

```cpp
static std::expected<AnyIVSolver, ValidationError>
build_chebyshev_segmented(const IVSolverFactoryConfig& config,
                          const ChebyshevBackend& /* backend */,
                          const DiscreteDividendConfig& divs) {
    auto log_m = to_log_moneyness(config.grid.moneyness);
    if (!log_m.has_value()) {
        return std::unexpected(log_m.error());
    }

    SegmentedAdaptiveConfig seg_config{
        .spot = config.spot,
        .option_type = config.option_type,
        .dividend_yield = config.dividend_yield,
        .discrete_dividends = divs.discrete_dividends,
        .maturity = divs.maturity,
        .kref_config = divs.kref_config,
    };

    IVGrid log_grid{std::move(*log_m), config.grid.vol, config.grid.rate};

    if (config.adaptive.has_value()) {
        auto result = build_adaptive_chebyshev_segmented_typed(
            *config.adaptive, seg_config, log_grid);
        if (!result.has_value()) {
            return std::unexpected(ValidationError{
                ValidationErrorCode::InvalidGridSize, 0.0});
        }

        auto solver = InterpolatedIVSolver<ChebyshevMultiKRefSurface>::create(
            std::move(result->surface), config.solver_config);
        if (!solver.has_value()) {
            return std::unexpected(solver.error());
        }
        return make_any_solver(std::move(*solver));
    } else {
        auto surface = build_chebyshev_segmented_manual(
            seg_config, log_grid);
        if (!surface.has_value()) {
            return std::unexpected(ValidationError{
                ValidationErrorCode::InvalidGridSize, 0.0});
        }

        auto solver = InterpolatedIVSolver<ChebyshevMultiKRefSurface>::create(
            std::move(*surface), config.solver_config);
        if (!solver.has_value()) {
            return std::unexpected(solver.error());
        }
        return make_any_solver(std::move(*solver));
    }
}
```

**Step 2: Run the existing factory test**

Run: `bazel test //tests:iv_solver_factory_test --test_output=all --test_filter='*Chebyshev*segmented*'`
Expected: PASS — the factory test exercises the end-to-end path including IV solve.

**Step 3: Run all tests**

Run: `bazel test //tests:adaptive_grid_builder_test //tests:iv_solver_factory_test --test_output=errors`
Expected: ALL PASS

**Step 4: Commit**

```bash
git add src/option/interpolated_iv_solver.cpp
git commit -m "Wire factory to use typed ChebyshevMultiKRefSurface"
```

---

### Task 9: Full CI check

**Step 1: Run all tests**

Run: `bazel test //...`
Expected: ALL PASS (116+ tests)

**Step 2: Build benchmarks and Python bindings**

Run: `bazel build //benchmarks/... && bazel build //src/python:mango_option`
Expected: SUCCESS

**Step 3: Commit (squash fixups if needed)**

If any fixes were needed, commit them. Otherwise this is a verification step only.

---

### Task 10: Clean up old path (separate PR)

**Do this only after Task 9 passes in CI.**

**Files:**
- Modify: `src/option/interpolated_iv_solver.cpp`
- Modify: `src/option/table/chebyshev/chebyshev_adaptive.hpp`
- Modify: `src/option/table/chebyshev/chebyshev_adaptive.cpp`

**Step 1: Remove `FDVegaLeaf` and old surface type**

In `src/option/interpolated_iv_solver.cpp`:
- Delete `FDVegaLeaf` class (lines 52-75)
- Delete `using ChebyshevSegmentedSurface = PriceTable<FDVegaLeaf>` (line 77)
- Delete `template class InterpolatedIVSolver<ChebyshevSegmentedSurface>` (line 78)
- Remove `InterpolatedIVSolver<ChebyshevSegmentedSurface>` from `SolverVariant`

**Step 2: Remove old `ChebyshevSegmentedAdaptiveResult` and `build_adaptive_chebyshev_segmented`**

In `chebyshev_adaptive.hpp`:
- Delete `ChebyshevSegmentedAdaptiveResult` struct (the one with `price_fn`)
- Delete `build_adaptive_chebyshev_segmented` declaration

In `chebyshev_adaptive.cpp`:
- Delete `build_adaptive_chebyshev_segmented` implementation
- Rename `build_adaptive_chebyshev_segmented_typed` → `build_adaptive_chebyshev_segmented`
- Rename `ChebyshevSegmentedTypedResult` → `ChebyshevSegmentedAdaptiveResult`

**Step 3: Update all callers**

Search for remaining references to the old names:
- `tests/adaptive_grid_builder_test.cc`: update calls to use new name
- Remove the equivalence test (or keep as regression comparing against FDM)

**Step 4: Run all tests**

Run: `bazel test //...`
Expected: ALL PASS

**Step 5: Build benchmarks and Python bindings**

Run: `bazel build //benchmarks/... && bazel build //src/python:mango_option`
Expected: SUCCESS

**Step 6: Commit**

```bash
git add -A
git commit -m "Remove FDVegaLeaf and old type-erased Chebyshev segmented path"
```
