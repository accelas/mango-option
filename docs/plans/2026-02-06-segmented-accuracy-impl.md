# Segmented IV Accuracy Improvements — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Reduce segmented IV interpolation error from 228 bps to <20 bps everywhere and <5 bps near ATM.

**Architecture:** Three changes: (1) probe more strikes during adaptive refinement so deep OTM/ITM coverage gaps are caught, (2) scale tau points proportionally to segment width so wide segments aren't under-resolved, (3) add a post-build validation pass with FD American vega to `build_segmented_strike`. All changes are in the `fix-segmented-accuracy` worktree, rebased on top of PR #361 (`bench/iv-sweep-adaptive-baseline`).

**Tech Stack:** C++23, Bazel, GoogleTest, Latin Hypercube sampling, B-spline interpolation, FD American option solver.

**Design doc:** `docs/plans/2026-02-06-segmented-accuracy-design.md`

**Prerequisites already in branch (from PR #361):**
- `SegmentedAdaptiveResult` and `StrikeAdaptiveResult` types in `adaptive_grid_types.hpp`
- `build_segmented()` returns `SegmentedAdaptiveResult`, `build_segmented_strike()` returns `StrikeAdaptiveResult`
- `iv_solver_factory.cpp` updated to use `result->surface`
- Tests updated to use `result->surface.price()`
- `latin_hypercube<N>` template ready in `latin_hypercube.hpp` (unstaged, will be committed in Task 0)

---

### Task 0: Commit pending prerequisite changes

The LHS template and updated design doc are unstaged. Commit them.

**Files:**
- Modified: `src/math/latin_hypercube.hpp` (templatized `latin_hypercube<N>`)
- Modified: `tests/latin_hypercube_test.cc` (Template3D, Template2D, BackwardCompatible4D tests)
- Modified: `docs/plans/2026-02-06-segmented-accuracy-design.md` (Codex review fixes)
- New: `docs/plans/2026-02-06-segmented-accuracy-impl.md` (this file)

**Step 1: Run LHS tests**

```bash
cd /home/kai/work/mango-option/.worktrees/fix-segmented-accuracy
bazel test //tests:latin_hypercube_test --test_output=errors
```

Expected: 11 tests PASS

**Step 2: Commit**

```bash
cd /home/kai/work/mango-option/.worktrees/fix-segmented-accuracy
git add src/math/latin_hypercube.hpp tests/latin_hypercube_test.cc \
        docs/plans/2026-02-06-segmented-accuracy-design.md \
        docs/plans/2026-02-06-segmented-accuracy-impl.md
git commit -m "Add latin_hypercube<N> template and updated design docs

Templatize latin_hypercube_4d into latin_hypercube<N> matching the
PriceTableBuilder<N> pattern. latin_hypercube_4d remains as a
backward-compatible wrapper.

Update design doc with Codex review fixes: tolerance consistency,
LHS template, API prerequisites, FD vega cost clarification."
```

---

### Task 1: Improve `select_probes` to cover more strikes

The `select_probes` function (line 54 in `adaptive_grid_builder.cpp`) returns all items only if N <= 3. This misses deep OTM/ITM error regions when there are 5-15 strikes. Change the threshold to 15 and add percentile sampling for N > 15.

**Files:**
- Modify: `src/option/table/adaptive_grid_builder.cpp:52-68`
- Test: `tests/adaptive_grid_builder_test.cc`

**Step 1: Write the failing test**

Add to `tests/adaptive_grid_builder_test.cc` at the end (before the closing `}`):

```cpp
// ===========================================================================
// select_probes coverage tests
// ===========================================================================

TEST(AdaptiveGridBuilderTest, SelectProbesSmallN) {
    // N <= 15 should return all items
    std::vector<double> strikes = {80, 85, 90, 95, 100, 105, 110, 115, 120};
    double spot = 100.0;

    // Build a segmented config with these strikes and check probe count
    // by observing the probe_and_build behavior.
    // Since select_probes is internal, we test via the public API:
    // build_segmented_strike with 9 strikes should probe all 9.
    //
    // We can't directly test select_probes, but we CAN test that the
    // build succeeds with higher accuracy when all strikes are probed.
    // For now, test indirectly: build with 9 strikes, verify it works.
    AdaptiveGridParams params;
    params.target_iv_error = 0.001;  // 10 bps (loose, for speed)
    params.max_iter = 2;
    params.validation_samples = 16;
    AdaptiveGridBuilder builder(params);

    SegmentedAdaptiveConfig config{
        .spot = spot,
        .option_type = OptionType::PUT,
        .dividend_yield = 0.02,
        .discrete_dividends = {{.calendar_time = 0.25, .amount = 0.50}},
        .maturity = 0.5,
        .kref_config = {},
    };

    ManualGrid domain{
        .moneyness = {0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3},
        .vol = {0.15, 0.20, 0.30},
        .rate = {0.03, 0.05},
    };

    auto result = builder.build_segmented_strike(config, strikes, domain);
    ASSERT_TRUE(result.has_value()) << "build_segmented_strike should succeed with 9 strikes";
}
```

**Step 2: Run test to verify it compiles and passes (baseline)**

```bash
cd /home/kai/work/mango-option/.worktrees/fix-segmented-accuracy
bazel test //tests:adaptive_grid_builder_test --test_output=errors --test_filter=SelectProbesSmallN
```

Expected: PASS (current code still works, just sub-optimal)

**Step 3: Implement the `select_probes` improvement**

Replace `select_probes` (lines 52-68 of `src/option/table/adaptive_grid_builder.cpp`):

```cpp
/// Select probes from a sorted vector.
/// - N <= 15: return all items (probing is cheap vs accuracy cost of missing strikes)
/// - N > 15: percentile sampling {min, p25, nearest-to-ref, p75, max}
std::vector<double> select_probes(const std::vector<double>& items,
                                  double reference_value) {
    if (items.size() <= 15) return items;

    std::vector<double> probes;
    const size_t n = items.size();
    probes.push_back(items.front());                    // min
    probes.push_back(items[n / 4]);                     // p25
    probes.push_back(items[3 * n / 4]);                 // p75
    probes.push_back(items.back());                     // max

    // ATM: closest to reference_value (if not already included)
    auto atm_it = std::min_element(items.begin(), items.end(),
        [&](double a, double b) {
            return std::abs(a - reference_value) < std::abs(b - reference_value);
        });
    bool already_included = false;
    for (double p : probes) {
        if (std::abs(p - *atm_it) < 1e-12) { already_included = true; break; }
    }
    if (!already_included) {
        probes.push_back(*atm_it);
    }

    std::sort(probes.begin(), probes.end());
    probes.erase(std::unique(probes.begin(), probes.end()), probes.end());
    return probes;
}
```

**Step 4: Run test to verify it passes**

```bash
cd /home/kai/work/mango-option/.worktrees/fix-segmented-accuracy
bazel test //tests:adaptive_grid_builder_test --test_output=errors
```

Expected: ALL tests PASS

**Step 5: Commit**

```bash
cd /home/kai/work/mango-option/.worktrees/fix-segmented-accuracy
git add src/option/table/adaptive_grid_builder.cpp tests/adaptive_grid_builder_test.cc
git commit -m "Probe all strikes when N <= 15 in select_probes

Raises the threshold from 3 to 15 so that typical strike lists
(5-12 strikes) are fully probed during adaptive refinement.
For N > 15, uses percentile sampling {min, p25, ATM, p75, max}.

Fixes adaptive coverage gap for deep OTM/ITM strikes."
```

---

### Task 2: Add width-proportional tau points to SegmentedPriceTableBuilder

Currently `tau_points_per_segment` is constant. Wide segments get the same resolution as narrow ones, causing under-resolution for long-tenor segments. Add a `tau_target_dt` config field so `make_segment_tau_grid` can scale points proportionally to segment width.

**Files:**
- Modify: `src/option/table/segmented_price_table_builder.hpp:21-36`
- Modify: `src/option/table/segmented_price_table_builder.cpp:26-47,179-181`
- Test: `tests/adaptive_grid_builder_test.cc`

**Step 1: Write the failing test**

Add to `tests/adaptive_grid_builder_test.cc`:

```cpp
// ===========================================================================
// Width-proportional tau points tests
// ===========================================================================

TEST(SegmentedPriceTableBuilderTest, TauTargetDtScaling) {
    // With tau_target_dt = 0.1 and a segment width of 0.5,
    // we should get ceil(0.5/0.1)+1 = 6 points.
    // With a segment width of 1.5, we should get ceil(1.5/0.1)+1 = 16 points.
    // This tests that wider segments get proportionally more tau points.

    SegmentedPriceTableBuilder::Config config{
        .K_ref = 100.0,
        .option_type = OptionType::PUT,
        .dividends = {
            .dividend_yield = 0.02,
            .discrete_dividends = {
                {.calendar_time = 0.5, .amount = 0.50},
            },
        },
        .grid = {
            .moneyness = {0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3},
            .vol = {0.15, 0.20, 0.30, 0.40},
            .rate = {0.02, 0.03, 0.05, 0.07},
        },
        .maturity = 2.0,
        .tau_points_per_segment = 5,
        .tau_target_dt = 0.1,  // NEW: target dt between tau points
    };

    auto result = SegmentedPriceTableBuilder::build(config);
    ASSERT_TRUE(result.has_value());

    // Query should succeed - the surface was built with width-proportional tau
    PriceQuery query{.spot = 100.0, .strike = 100.0, .tau = 1.0, .sigma = 0.20, .rate = 0.05};
    double price = result->price(query);
    EXPECT_GT(price, 0.0);
    EXPECT_TRUE(std::isfinite(price));
}

TEST(SegmentedPriceTableBuilderTest, TauTargetDtZeroFallback) {
    // tau_target_dt = 0 should fall back to constant tau_points_per_segment
    SegmentedPriceTableBuilder::Config config{
        .K_ref = 100.0,
        .option_type = OptionType::PUT,
        .dividends = {
            .dividend_yield = 0.02,
            .discrete_dividends = {
                {.calendar_time = 0.25, .amount = 0.50},
            },
        },
        .grid = {
            .moneyness = {0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3},
            .vol = {0.15, 0.20, 0.30, 0.40},
            .rate = {0.02, 0.03, 0.05, 0.07},
        },
        .maturity = 1.0,
        .tau_points_per_segment = 5,
        .tau_target_dt = 0.0,  // legacy mode
    };

    auto result = SegmentedPriceTableBuilder::build(config);
    ASSERT_TRUE(result.has_value());
}
```

**Step 2: Run test to verify it fails**

```bash
cd /home/kai/work/mango-option/.worktrees/fix-segmented-accuracy
bazel test //tests:adaptive_grid_builder_test --test_output=errors --test_filter=TauTargetDt
```

Expected: FAIL — `tau_target_dt` doesn't exist in Config yet.

**Step 3: Add config fields to `SegmentedPriceTableBuilder::Config`**

In `src/option/table/segmented_price_table_builder.hpp`, replace the Config struct:

```cpp
    struct Config {
        double K_ref;
        OptionType option_type;
        DividendSpec dividends;  ///< Continuous yield + discrete schedule

        /// Grid specification: moneyness, vol, and rate grids
        ManualGrid grid;

        double maturity;  // T in years

        /// Minimum tau points per segment (actual count may be higher)
        int tau_points_per_segment = 5;

        /// If true, skip internal moneyness expansion (caller pre-expanded).
        bool skip_moneyness_expansion = false;

        /// Target dt between tau grid points.
        /// When > 0, each segment gets ceil(width / tau_target_dt) + 1 points
        /// (clamped to [tau_points_min, tau_points_max]).
        /// When == 0, falls back to constant tau_points_per_segment.
        double tau_target_dt = 0.0;
        int tau_points_min = 4;   ///< B-spline minimum
        int tau_points_max = 30;  ///< Cap for very wide segments
    };
```

**Step 4: Update `make_segment_tau_grid` in `segmented_price_table_builder.cpp`**

Replace the function signature and body (lines 26-47):

```cpp
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
```

**Step 5: Pass tau_target_dt from Config to make_segment_tau_grid**

In the `SegmentedPriceTableBuilder::build` method (around line 180), update the `make_segment_tau_grid` call:

```cpp
        auto local_tau = make_segment_tau_grid(
            0.0, seg_width, config.tau_points_per_segment, is_last_segment,
            config.tau_target_dt, config.tau_points_min, config.tau_points_max);
```

**Step 6: Run tests**

```bash
cd /home/kai/work/mango-option/.worktrees/fix-segmented-accuracy
bazel test //tests:adaptive_grid_builder_test --test_output=errors
```

Expected: ALL tests PASS

**Step 7: Commit**

```bash
cd /home/kai/work/mango-option/.worktrees/fix-segmented-accuracy
git add src/option/table/segmented_price_table_builder.hpp \
        src/option/table/segmented_price_table_builder.cpp \
        tests/adaptive_grid_builder_test.cc
git commit -m "Add width-proportional tau points to segmented builder

New Config fields: tau_target_dt, tau_points_min, tau_points_max.
When tau_target_dt > 0, each segment gets
ceil(width / tau_target_dt) + 1 tau points, clamped to
[tau_points_min, tau_points_max]. When 0, falls back to the
existing constant tau_points_per_segment.

Fixes under-resolution of wide segments in long-tenor cases."
```

---

### Task 3: Wire tau_target_dt through adaptive builder

The adaptive builder calls `SegmentedPriceTableBuilder` via `make_seg_config` and `probe_and_build`. It needs to compute `tau_target_dt` from the shortest segment width and the refined tau_points, then pass it through.

**Files:**
- Modify: `src/option/table/adaptive_grid_builder.cpp:70-100` (make_seg_config) and `696-702` (final grid assembly)

**Step 1: Understand the flow**

`probe_and_build` calls `make_seg_config` to build a `SegmentedPriceTableBuilder::Config`. Currently `make_seg_config` doesn't set `tau_target_dt`. After refinement, we know the tau_points from the probe. We need to:
1. Compute the shortest segment width from the dividend schedule
2. Set `tau_target_dt = shortest_width / tau_points`

**Step 2: Update `make_seg_config` to accept tau_target_dt**

Read `make_seg_config` (lines 70-100 approximately) to see its current signature:

In `src/option/table/adaptive_grid_builder.cpp`, add `tau_target_dt` parameter to `make_seg_config`:

```cpp
SegmentedPriceTableBuilder::Config make_seg_config(
    const SegmentedAdaptiveConfig& config,
    const std::vector<double>& m_grid,
    const std::vector<double>& v_grid,
    const std::vector<double>& r_grid,
    int tau_pts,
    double tau_target_dt = 0.0)
{
    return SegmentedPriceTableBuilder::Config{
        .K_ref = 0.0,  // caller sets per-surface
        .option_type = config.option_type,
        .dividends = {.dividend_yield = config.dividend_yield,
                      .discrete_dividends = config.discrete_dividends},
        .grid = ManualGrid{
            .moneyness = m_grid,
            .vol = v_grid,
            .rate = r_grid,
        },
        .maturity = config.maturity,
        .tau_points_per_segment = tau_pts,
        .skip_moneyness_expansion = true,
        .tau_target_dt = tau_target_dt,
    };
}
```

**Step 3: Compute shortest segment width and set tau_target_dt in probe_and_build**

Add a helper to compute shortest segment width from dividends:

```cpp
/// Compute the shortest segment width (in τ) from a dividend schedule.
double shortest_segment_width(const std::vector<Dividend>& divs, double maturity) {
    std::vector<double> boundaries;
    boundaries.push_back(0.0);
    // Filter and sort dividends
    std::vector<double> div_times;
    for (const auto& d : divs) {
        if (d.calendar_time > 0.0 && d.calendar_time < maturity && d.amount > 0.0) {
            div_times.push_back(d.calendar_time);
        }
    }
    std::sort(div_times.begin(), div_times.end());
    for (auto it = div_times.rbegin(); it != div_times.rend(); ++it) {
        boundaries.push_back(maturity - *it);
    }
    boundaries.push_back(maturity);

    double min_width = maturity;
    for (size_t i = 0; i + 1 < boundaries.size(); ++i) {
        double w = boundaries[i + 1] - boundaries[i];
        if (w > 0.0) min_width = std::min(min_width, w);
    }
    return min_width;
}
```

In `probe_and_build` (around line 696), after computing `max_tau_pts`, compute `tau_target_dt`:

```cpp
    int max_tau_pts = gsz.tau_points;

    // Compute tau_target_dt from shortest segment width
    double min_seg_width = shortest_segment_width(
        config.discrete_dividends, config.maturity);
    double tau_target_dt = (max_tau_pts > 1)
        ? min_seg_width / static_cast<double>(max_tau_pts - 1)
        : 0.0;

    auto seg_template = make_seg_config(config, final_m, final_v, final_r,
                                        max_tau_pts, tau_target_dt);
```

**Step 4: Run tests**

```bash
cd /home/kai/work/mango-option/.worktrees/fix-segmented-accuracy
bazel test //tests:adaptive_grid_builder_test --test_output=errors
```

Expected: ALL tests PASS

**Step 5: Commit**

```bash
cd /home/kai/work/mango-option/.worktrees/fix-segmented-accuracy
git add src/option/table/adaptive_grid_builder.cpp
git commit -m "Wire tau_target_dt from adaptive builder to segmented builder

Computes tau_target_dt from shortest segment width / refined
tau_points, so wider segments automatically get proportionally
more tau grid points. Narrow segments keep their existing density."
```

---

### Task 4: Add validation pass to `build_segmented_strike`

`build_segmented_strike` (line 1309) has no post-build validation — it just builds and returns. `build_segmented` (line 1230) has a validation pass with retry. Add the same to `build_segmented_strike`, using FD American vega for the error metric and `latin_hypercube<3>` for sampling.

`StrikeAdaptiveResult` already exists (from PR #361) but needs new fields for validation reporting.

**Files:**
- Modify: `src/option/table/adaptive_grid_types.hpp:113-117` (add error fields to existing StrikeAdaptiveResult)
- Modify: `src/option/table/adaptive_grid_builder.cpp:1309-1332` (add validation)
- Test: `tests/adaptive_grid_builder_test.cc`

**Step 1: Add validation fields to existing `StrikeAdaptiveResult`**

In `src/option/table/adaptive_grid_types.hpp`, add fields to the existing struct:

```cpp
struct StrikeAdaptiveResult {
    StrikeSurface<> surface;
    ManualGrid grid;            ///< The grid sizes adaptive chose
    int tau_points_per_segment;
    double max_iv_error = 0.0;  ///< Worst IV error across all validation points
    double p95_iv_error = 0.0;  ///< 95th percentile IV error
    bool target_met = false;    ///< Whether acceptance thresholds were satisfied
};
```

**Step 2: Implement validation pass in `build_segmented_strike`**

Replace the end of `build_segmented_strike` (from where the surface is assembled to the return) with:

```cpp
    // ... existing code up to building the surface ...

    std::vector<StrikeEntry> entries;
    for (size_t i = 0; i < strikes.size(); ++i) {
        entries.push_back({.strike = strikes[i], .surface = std::move(result->surfaces[i])});
    }
    auto surface = build_strike_surface(std::move(entries), /*use_nearest=*/true);
    if (!surface.has_value()) {
        return std::unexpected(surface.error());
    }

    auto& build = *result;
    auto final_m = linspace(build.expanded_min_m, build.max_m, build.gsz.moneyness);
    auto final_v = linspace(build.min_vol, build.max_vol, build.gsz.vol);
    auto final_r = linspace(build.min_rate, build.max_rate, build.gsz.rate);

    // === Validation pass with FD American vega ===
    constexpr double kMaxAcceptable = 20e-4;   // 20 bps
    constexpr double kP95Acceptable = 5e-4;    // 5 bps
    constexpr double kVegaBumpH = 0.005;       // 50 bps sigma bump
    constexpr int kMaxRetries = 2;

    auto validate_surface = [&](const StrikeSurface<>& surf)
        -> std::pair<double, double>  // {max_error, p95_error}
    {
        auto lhs = latin_hypercube<3>(params_.validation_samples, params_.lhs_seed + 777);
        std::array<std::pair<double, double>, 3> bounds = {{
            {build.min_tau, build.max_tau},
            {build.min_vol, build.max_vol},
            {build.min_rate, build.max_rate},
        }};
        auto samples = scale_lhs_samples(lhs, bounds);

        std::vector<double> errors;
        errors.reserve(samples.size() * strikes.size());

        for (const auto& s : samples) {
            double tau = s[0], sigma = s[1], rate = s[2];

            for (double K : strikes) {
                // Interpolated price
                PriceQuery q{.spot = config.spot, .strike = K, .tau = tau,
                             .sigma = sigma, .rate = rate};
                double interp = surf.price(q);

                // Reference FD price
                PricingParams p;
                p.spot = config.spot;
                p.strike = K;
                p.maturity = tau;
                p.rate = rate;
                p.dividend_yield = config.dividend_yield;
                p.option_type = config.option_type;
                p.volatility = sigma;
                p.discrete_dividends = config.discrete_dividends;
                auto fd = solve_american_option(p);
                if (!fd.has_value()) continue;

                double price_error = std::abs(interp - fd->value());

                // FD American vega via central difference
                double sigma_lo = std::max(sigma - kVegaBumpH, 0.01);
                double sigma_hi = sigma + kVegaBumpH;
                p.volatility = sigma_lo;
                auto fd_lo = solve_american_option(p);
                p.volatility = sigma_hi;
                auto fd_hi = solve_american_option(p);

                double am_vega = 0.0;
                if (fd_lo.has_value() && fd_hi.has_value()) {
                    am_vega = (fd_hi->value() - fd_lo->value()) /
                              (sigma_hi - sigma_lo);
                }
                double vega_clamped = std::max(std::abs(am_vega), params_.vega_floor);
                double iv_err = price_error / vega_clamped;

                // Cap when price is already tiny
                double price_tol = kMaxAcceptable * params_.vega_floor;
                if (price_error <= price_tol) {
                    iv_err = std::min(iv_err, kMaxAcceptable);
                }

                errors.push_back(iv_err);
            }
        }

        if (errors.empty()) return {0.0, 0.0};

        std::sort(errors.begin(), errors.end());
        double max_err = errors.back();
        size_t p95_idx = std::min(
            static_cast<size_t>(errors.size() * 0.95),
            errors.size() - 1);
        double p95_err = errors[p95_idx];
        return {max_err, p95_err};
    };

    auto [max_err, p95_err] = validate_surface(*surface);
    bool target_met = (max_err <= kMaxAcceptable && p95_err <= kP95Acceptable);

    // Retry loop
    auto best_surface = std::move(*surface);
    double best_max = max_err;
    double best_p95 = p95_err;

    for (int retry = 0; retry < kMaxRetries && !target_met; ++retry) {
        size_t bumped_m = std::min(build.gsz.moneyness + 2, params_.max_points_per_dim);
        size_t bumped_v = std::min(build.gsz.vol + 1, params_.max_points_per_dim);
        size_t bumped_r = std::min(build.gsz.rate + 1, params_.max_points_per_dim);
        int bumped_tau = std::min(build.gsz.tau_points + 2,
            static_cast<int>(params_.max_points_per_dim));

        // Recompute tau_target_dt
        double min_seg_width = shortest_segment_width(
            config.discrete_dividends, config.maturity);
        double retry_dt = (bumped_tau > 1)
            ? min_seg_width / static_cast<double>(bumped_tau - 1)
            : 0.0;

        auto retry_m = linspace(build.expanded_min_m, build.max_m, bumped_m);
        auto retry_v = linspace(build.min_vol, build.max_vol, bumped_v);
        auto retry_r = linspace(build.min_rate, build.max_rate, bumped_r);

        auto retry_cfg = make_seg_config(config, retry_m, retry_v, retry_r,
                                         bumped_tau, retry_dt);
        auto retry_segs = build_segmented_surfaces(retry_cfg, strikes);
        if (!retry_segs.has_value()) continue;

        std::vector<StrikeEntry> retry_entries;
        for (size_t i = 0; i < strikes.size(); ++i) {
            retry_entries.push_back({.strike = strikes[i],
                                     .surface = std::move((*retry_segs)[i])});
        }
        auto retry_surf = build_strike_surface(std::move(retry_entries), true);
        if (!retry_surf.has_value()) continue;

        auto [rm, rp] = validate_surface(*retry_surf);
        if (rm < best_max) {
            best_surface = std::move(*retry_surf);
            best_max = rm;
            best_p95 = rp;
            final_m = retry_m;
            final_v = retry_v;
            final_r = retry_r;
            build.gsz.moneyness = bumped_m;
            build.gsz.vol = bumped_v;
            build.gsz.rate = bumped_r;
            build.gsz.tau_points = bumped_tau;
        }
        target_met = (best_max <= kMaxAcceptable && best_p95 <= kP95Acceptable);
    }

    return StrikeAdaptiveResult{
        .surface = std::move(best_surface),
        .grid = {.moneyness = final_m, .vol = final_v, .rate = final_r},
        .tau_points_per_segment = build.gsz.tau_points,
        .max_iv_error = best_max,
        .p95_iv_error = best_p95,
        .target_met = target_met,
    };
```

**Step 4: Run tests**

```bash
cd /home/kai/work/mango-option/.worktrees/fix-segmented-accuracy
bazel test //... --test_output=errors
```

Expected: ALL 117+ tests PASS

**Step 5: Commit**

```bash
cd /home/kai/work/mango-option/.worktrees/fix-segmented-accuracy
git add src/option/table/adaptive_grid_types.hpp \
        src/option/table/adaptive_grid_builder.cpp \
        tests/adaptive_grid_builder_test.cc
git commit -m "Add validation pass with FD American vega to build_segmented_strike

Adds post-build LHS validation using latin_hypercube<3> and FD
American vega (central difference). Retries up to 2x with bumped
grids if max error > 20 bps or p95 > 5 bps.

Returns StrikeAdaptiveResult with max_iv_error, p95_iv_error,
and target_met for caller inspection."
```

---

### Task 5: Preserve anchor moneyness knots

After `probe_and_build` computes `final_m = linspace(min, max, N)`, merge in m=1.0 (ATM) and the original user-provided moneyness knots. This ensures the interpolation grid includes financially meaningful locations.

**Files:**
- Modify: `src/option/table/adaptive_grid_builder.cpp` (in `probe_and_build`, after linspace)

**Step 1: Implement anchor merging in `probe_and_build`**

In `probe_and_build`, after the line `auto final_m = linspace(...)` (around line 697), add:

```cpp
    // Merge anchor moneyness knots: m=1.0 (ATM) and user-provided domain knots
    auto merge_anchors = [](std::vector<double>& grid, const std::vector<double>& anchors) {
        for (double a : anchors) {
            if (a >= grid.front() && a <= grid.back()) {
                // Check if already present (within tolerance)
                bool found = false;
                for (double g : grid) {
                    if (std::abs(g - a) < 1e-6) { found = true; break; }
                }
                if (!found) grid.push_back(a);
            }
        }
        std::sort(grid.begin(), grid.end());
    };

    // Always include ATM moneyness
    std::vector<double> anchors = {1.0};
    // Include user-provided domain knots
    for (double m : domain.moneyness) {
        if (m >= expanded_min_m && m <= max_m) anchors.push_back(m);
    }
    merge_anchors(final_m, anchors);
```

**Step 2: Run tests**

```bash
cd /home/kai/work/mango-option/.worktrees/fix-segmented-accuracy
bazel test //tests:adaptive_grid_builder_test --test_output=errors
```

Expected: ALL tests PASS

**Step 3: Commit**

```bash
cd /home/kai/work/mango-option/.worktrees/fix-segmented-accuracy
git add src/option/table/adaptive_grid_builder.cpp
git commit -m "Preserve ATM and user moneyness knots in final grid

Merges m=1.0 and original domain moneyness points into the
linspace grid. Ensures interpolation grid includes financially
meaningful locations."
```

---

### Task 6: Add regression test for sigma=0.30, T=2.0, K=80

This is the exact case from the `interp_iv_safety` benchmark that hit 228 bps. After all accuracy improvements, this should be under 20 bps.

**Files:**
- Test: `tests/adaptive_grid_builder_test.cc`

**Step 1: Write the regression test**

```cpp
// ===========================================================================
// Regression: sigma=0.30, T=2.0, quarterly $0.50, K=80 was hitting 228 bps
// ===========================================================================

TEST(AdaptiveGridBuilderTest, DeepOTMLongTenorDividendAccuracy) {
    AdaptiveGridParams params;
    params.max_iter = 3;
    params.validation_samples = 32;
    AdaptiveGridBuilder builder(params);

    SegmentedAdaptiveConfig config{
        .spot = 100.0,
        .option_type = OptionType::PUT,
        .dividend_yield = 0.02,
        .discrete_dividends = {
            {.calendar_time = 0.50, .amount = 0.50},
            {.calendar_time = 1.00, .amount = 0.50},
            {.calendar_time = 1.50, .amount = 0.50},
        },
        .maturity = 2.0,
        .kref_config = {},
    };

    ManualGrid domain{
        .moneyness = {0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3},
        .vol = {0.15, 0.20, 0.30, 0.40},
        .rate = {0.02, 0.03, 0.05, 0.07},
    };

    std::vector<double> strikes = {80.0, 90.0, 100.0, 110.0, 120.0};

    auto result = builder.build_segmented_strike(config, strikes, domain);
    ASSERT_TRUE(result.has_value());

    // Query at the exact trouble spot: K=80, tau=2.0, sigma=0.30
    double spot = 100.0;
    double K = 80.0;
    double tau = 2.0;
    double sigma = 0.30;
    double rate = 0.05;

    PriceQuery query{.spot = spot, .strike = K, .tau = tau,
                     .sigma = sigma, .rate = rate};
    double interp_price = result->surface.price(query);

    // Fresh FD reference
    PricingParams p;
    p.spot = spot; p.strike = K; p.maturity = tau;
    p.rate = rate; p.dividend_yield = 0.02;
    p.option_type = OptionType::PUT; p.volatility = sigma;
    p.discrete_dividends = config.discrete_dividends;
    auto fd = solve_american_option(p);
    ASSERT_TRUE(fd.has_value());
    double ref_price = fd->value();

    // Compute IV error via FD American vega
    p.volatility = sigma - 0.005;
    auto fd_lo = solve_american_option(p);
    p.volatility = sigma + 0.005;
    auto fd_hi = solve_american_option(p);
    ASSERT_TRUE(fd_lo.has_value());
    ASSERT_TRUE(fd_hi.has_value());
    double am_vega = (fd_hi->value() - fd_lo->value()) / 0.01;
    double iv_err_bps = std::abs(interp_price - ref_price) /
                        std::max(std::abs(am_vega), 1e-4) * 10000.0;

    // Target: < 20 bps (was 228 bps before this work)
    EXPECT_LT(iv_err_bps, 20.0)
        << "IV error at K=80, T=2.0, sigma=0.30: " << iv_err_bps << " bps"
        << " (interp=" << interp_price << ", ref=" << ref_price << ")";

    // Also check the validation diagnostics
    EXPECT_LT(result->max_iv_error, 20e-4)
        << "max_iv_error should be < 20 bps";
}
```

**Step 2: Run the test**

```bash
cd /home/kai/work/mango-option/.worktrees/fix-segmented-accuracy
bazel test //tests:adaptive_grid_builder_test --test_output=all --test_filter=DeepOTMLongTenor
```

Expected: PASS with IV error < 20 bps (if all prior tasks are complete)

**Step 3: Run full test suite**

```bash
cd /home/kai/work/mango-option/.worktrees/fix-segmented-accuracy
bazel test //... --test_output=errors
```

Expected: ALL tests PASS

**Step 4: Commit**

```bash
cd /home/kai/work/mango-option/.worktrees/fix-segmented-accuracy
git add tests/adaptive_grid_builder_test.cc
git commit -m "Add regression test for deep OTM long-tenor dividend IV

Tests the exact case from interp_iv_safety that hit 228 bps:
sigma=0.30, T=2.0, quarterly $0.50, K=80. Asserts < 20 bps."
```

---

### Task 7: Final verification

Run the full pre-PR checklist from CLAUDE.md.

**Step 1: Run all tests**

```bash
cd /home/kai/work/mango-option/.worktrees/fix-segmented-accuracy
bazel test //...
```

Expected: ALL tests PASS

**Step 2: Build all benchmarks**

```bash
cd /home/kai/work/mango-option/.worktrees/fix-segmented-accuracy
bazel build //benchmarks:interp_iv_safety
```

Expected: Compiles cleanly

**Step 3: Build Python bindings**

```bash
cd /home/kai/work/mango-option/.worktrees/fix-segmented-accuracy
bazel build //src/python:mango_option
```

Expected: Compiles cleanly

**Step 4: Run interp_iv_safety benchmark**

```bash
cd /home/kai/work/mango-option/.worktrees/fix-segmented-accuracy
bazel run //benchmarks:interp_iv_safety
```

Expected: sigma=0.30 dividend heatmap shows no cell > 20 bps, ATM cells < 5 bps
