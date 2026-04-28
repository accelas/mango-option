# Log-Moneyness Axes Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Store `axes.grids[0]` as log-moneyness natively, eliminating 6 redundant `log()` transforms and 2 chain rule corrections.

**Architecture:** Change the coordinate system of axis 0 from moneyness (S/K) to log-moneyness (ln(S/K)) throughout the price table subsystem. The log transform moves to API boundaries (factory methods and AmericanPriceSurface), while the internal B-spline pipeline becomes a direct pass-through. Chain rule corrections for derivatives move from PriceTableSurface to AmericanPriceSurface where physical context (spot, strike) is available.

**Tech Stack:** C++23, Bazel, GoogleTest

**Design doc:** `docs/plans/2026-02-07-log-moneyness-axes-design.md`

**Worktree:** `/home/kai/work/mango-option/.worktrees/log-moneyness` (branch: `feature/log-moneyness-axes`)

---

### Task 1: Simplify PriceTableSurface — remove log transforms and chain rules

This is the core change. PriceTableSurface becomes a thin pass-through to the B-spline on axis 0.

**Files:**
- Modify: `src/option/table/price_table_surface.cpp:19-142`
- Modify: `src/option/table/price_table_surface.hpp:38-61`
- Modify: `src/option/table/price_table_metadata.hpp:20-30`

**Step 1: Update PriceTableMetadata doc comments**

In `src/option/table/price_table_metadata.hpp`, change the doc comments and field descriptions to reflect log-moneyness:

```cpp
/// Metadata for price table surface
///
/// Stores reference strike, dividend information, log-moneyness bounds,
/// and discrete dividend schedule.
///
/// The bounds (m_min, m_max) store the log-moneyness range: ln(S/K).
struct PriceTableMetadata {
    double K_ref = 0.0;                                     ///< Reference strike price
    DividendSpec dividends;                                  ///< Continuous yield + discrete schedule
    double m_min = 0.0;                                     ///< Minimum log-moneyness ln(S/K)
    double m_max = 0.0;                                     ///< Maximum log-moneyness ln(S/K)
    SurfaceContent content = SurfaceContent::EarlyExercisePremium;  ///< What tensor stores
};
```

**Step 2: Simplify PriceTableSurface::build()**

In `src/option/table/price_table_surface.cpp`, replace lines 20-86. The key changes:
- Remove the comment about "storing original moneyness bounds before transforming"
- Set `metadata.m_min`/`.m_max` directly from `axes.grids[0]` (already log-moneyness)
- Delete the log-transform loop (lines 40-48)
- Delete `internal_axes` — use `axes` directly
- Pass `axes` (not `internal_axes`) to BSplineND creation

Replace lines 20-86 with:

```cpp
template <size_t N>
std::expected<std::shared_ptr<const PriceTableSurface<N>>, PriceTableError>
PriceTableSurface<N>::build(
    PriceTableAxes<N> axes,
    std::vector<double> coeffs,
    PriceTableMetadata metadata)
{
    // Validate axes
    if (auto valid = axes.validate(); !valid.has_value()) {
        return std::unexpected(PriceTableError{PriceTableErrorCode::InvalidConfig});
    }

    // Store log-moneyness bounds in metadata
    if constexpr (N >= 1) {
        if (!axes.grids[0].empty()) {
            metadata.m_min = axes.grids[0].front();
            metadata.m_max = axes.grids[0].back();
        }
    }

    // Check coefficient size matches axes
    size_t expected_size = axes.total_points();
    if (coeffs.size() != expected_size) {
        return std::unexpected(PriceTableError{
            PriceTableErrorCode::FittingFailed, 0, coeffs.size()});
    }

    // Create knot sequences for clamped cubic B-splines
    typename BSplineND<double, N>::KnotArray knots;
    for (size_t dim = 0; dim < N; ++dim) {
        knots[dim] = clamped_knots_cubic(axes.grids[dim]);
    }

    // Create BSplineND with log-moneyness grid
    typename BSplineND<double, N>::GridArray grids_copy;
    for (size_t dim = 0; dim < N; ++dim) {
        grids_copy[dim] = axes.grids[dim];
    }

    auto spline_result = BSplineND<double, N>::create(
        std::move(grids_copy),
        std::move(knots),
        std::move(coeffs));

    if (!spline_result.has_value()) {
        return std::unexpected(PriceTableError{PriceTableErrorCode::SurfaceBuildFailed});
    }

    auto spline = std::make_unique<BSplineND<double, N>>(std::move(spline_result.value()));

    auto surface = std::shared_ptr<const PriceTableSurface<N>>(
        new PriceTableSurface<N>(std::move(axes), std::move(metadata), std::move(spline)));

    return surface;
}
```

**Step 3: Simplify value(), partial(), second_partial()**

Replace lines 88-142 with direct pass-through:

```cpp
template <size_t N>
double PriceTableSurface<N>::value(const std::array<double, N>& coords) const {
    return spline_->eval(coords);
}

template <size_t N>
double PriceTableSurface<N>::partial(size_t axis, const std::array<double, N>& coords) const {
    return spline_->eval_partial(axis, coords);
}

template <size_t N>
double PriceTableSurface<N>::second_partial(size_t axis, const std::array<double, N>& coords) const {
    return spline_->eval_second_partial(axis, coords);
}
```

**Step 4: Update PriceTableSurface doc comments**

In `src/option/table/price_table_surface.hpp`, update the doc comments for `value()`, `partial()`, and `second_partial()` to reflect that coords[0] is log-moneyness:

```cpp
    /// Evaluate price at query point
    ///
    /// @param coords N-dimensional coordinates (axis 0 = log-moneyness)
    /// @return Interpolated value (clamped at boundaries)
    [[nodiscard]] double value(const std::array<double, N>& coords) const;

    /// Partial derivative along specified axis
    ///
    /// @param axis Axis index (0 to N-1)
    /// @param coords N-dimensional coordinates (axis 0 = log-moneyness)
    /// @return Partial derivative (axis 0 = ∂f/∂x where x = ln(S/K))
    [[nodiscard]] double partial(size_t axis, const std::array<double, N>& coords) const;

    /// Second partial derivative along specified axis
    ///
    /// @param axis Axis index (0 to N-1)
    /// @param coords N-dimensional coordinates (axis 0 = log-moneyness)
    /// @return Second partial derivative (axis 0 = ∂²f/∂x² where x = ln(S/K))
    [[nodiscard]] double second_partial(size_t axis, const std::array<double, N>& coords) const;
```

**Step 5: Build to verify compilation**

Run: `cd /home/kai/work/mango-option/.worktrees/log-moneyness && bazel build //src/option/...`
Expected: Compiles (tests may fail — downstream callers not yet updated)

**Step 6: Commit**

```bash
git add src/option/table/price_table_surface.cpp src/option/table/price_table_surface.hpp src/option/table/price_table_metadata.hpp
git commit -m "Simplify PriceTableSurface for log-moneyness axis

Axis 0 now expects log-moneyness coordinates directly.
Removes 3 log() transforms and 2 chain rule corrections.
Part of #373."
```

---

### Task 2: Update PriceTableBuilder — remove log transforms in extract_tensor and fit_coeffs

**Files:**
- Modify: `src/option/table/price_table_builder.cpp:54-57,81-82,212-213,376-381,444,453,457,498-512,573,579-581,594,598,646-651,653`
- Modify: `src/option/table/price_table_builder.hpp:95-107`

**Step 1: Update from_vectors() — accept log-moneyness**

In `price_table_builder.cpp`, update the `from_vectors()` method (lines 559-614):

- Rename parameter `moneyness` → `log_moneyness`
- Remove positivity check for moneyness (log values can be negative)
- Change axis name from `"moneyness"` to `"log_moneyness"`

```cpp
template <>
PriceTableBuilder<4>::Setup
PriceTableBuilder<4>::from_vectors(
    std::vector<double> log_moneyness,
    std::vector<double> maturity,
    std::vector<double> volatility,
    std::vector<double> rate,
    double K_ref,
    PDEGridSpec pde_grid,
    OptionType type,
    double dividend_yield,
    double max_failure_rate)
{
    // Sort and dedupe
    log_moneyness = sort_and_dedupe(std::move(log_moneyness));
    maturity = sort_and_dedupe(std::move(maturity));
    volatility = sort_and_dedupe(std::move(volatility));
    rate = sort_and_dedupe(std::move(rate));

    // Validate (no positivity check for log_moneyness — can be negative)
    if (!maturity.empty() && maturity.front() < 0.0) {
        return std::unexpected(PriceTableError{PriceTableErrorCode::NonPositiveValue, 1});
    }
    if (!volatility.empty() && volatility.front() <= 0.0) {
        return std::unexpected(PriceTableError{PriceTableErrorCode::NonPositiveValue, 2});
    }
    if (K_ref <= 0.0) {
        return std::unexpected(PriceTableError{PriceTableErrorCode::NonPositiveValue, 4});
    }

    // Build axes
    PriceTableAxes<4> axes;
    axes.grids[0] = std::move(log_moneyness);
    axes.grids[1] = std::move(maturity);
    axes.grids[2] = std::move(volatility);
    axes.grids[3] = std::move(rate);
    axes.names = {"log_moneyness", "maturity", "volatility", "rate"};

    // Build config
    PriceTableConfig config;
    config.option_type = type;
    config.K_ref = K_ref;
    config.pde_grid = std::move(pde_grid);
    config.dividends.dividend_yield = dividend_yield;
    config.max_failure_rate = max_failure_rate;

    // Validate config
    if (auto err = validate_config(config); err.has_value()) {
        return std::unexpected(PriceTableError{PriceTableErrorCode::InvalidConfig});
    }

    return std::make_pair(PriceTableBuilder<4>(config), std::move(axes));
}
```

Update the declaration in `price_table_builder.hpp` (lines 88-115):

```cpp
    /// @param log_moneyness Log-moneyness values (ln(S/K), any sign)
    ...
    static Setup
    from_vectors(
        std::vector<double> log_moneyness,
        ...
```

**Step 2: Update from_strikes() — compute log(spot/K)**

In `price_table_builder.cpp`, update `from_strikes()` (lines 616-664):

```cpp
    // Compute log-moneyness = log(spot/strike)
    auto log_moneyness = strikes
        | std::views::transform([spot](double K) { return std::log(spot / K); })
        | std::ranges::to<std::vector>();
    std::ranges::sort(log_moneyness);

    return from_vectors(
        std::move(log_moneyness),
        ...
```

**Step 3: Remove positive moneyness check in build()**

In `price_table_builder.cpp`, lines 54-57, remove or update the check:

```cpp
    // (Delete the positive moneyness check — log-moneyness can be negative)
```

**Step 4: Update PDE domain coverage check**

In `price_table_builder.cpp`, lines 81-82, axes.grids[0] is already log-moneyness:

```cpp
        const double x_min_requested = axes.grids[0].front();
        const double x_max_requested = axes.grids[0].back();
```

**Step 5: Update ensure_moneyness_coverage()**

In `price_table_builder.cpp`, lines 212-213, axes.grids[0] is already log-moneyness:

```cpp
    const double log_m_min = axes.grids[0].front();
    const double log_m_max = axes.grids[0].back();
```

**Step 6: Remove log_moneyness precomputation in extract_tensor()**

In `price_table_builder.cpp`, delete lines 376-381 (the `log_moneyness` vector) and update line 444 to use `axes.grids[0][i]` directly:

```cpp
                        double normalized_price = spline.eval(axes.grids[0][i]);
```

Also update the EEP computation on line 453-457 — `m = axes.grids[0][i]` is now log-moneyness, so spot computation changes:

```cpp
                            double x = axes.grids[0][i];  // log-moneyness
                            double tau = axes.grids[1][j];
                            double sigma = axes.grids[2][σ_idx];
                            double rate = axes.grids[3][r_idx];
                            double spot = std::exp(x) * K_ref;
```

**Step 7: Remove log transform in fit_coeffs()**

In `price_table_builder.cpp`, lines 498-512, delete the moneyness-to-log transform loop. Just copy grids directly:

```cpp
    std::array<std::vector<double>, N> grids;
    for (size_t i = 0; i < N; ++i) {
        grids[i] = axes.grids[i];
    }
```

**Step 8: Build to verify compilation**

Run: `cd /home/kai/work/mango-option/.worktrees/log-moneyness && bazel build //src/option/...`

**Step 9: Commit**

```bash
git add src/option/table/price_table_builder.cpp src/option/table/price_table_builder.hpp
git commit -m "Accept log-moneyness in PriceTableBuilder factories

from_vectors() takes log-moneyness directly.
from_strikes() computes log(spot/K) at the boundary.
Removes 3 remaining log() transforms in builder pipeline.
Part of #373."
```

---

### Task 3: Update AmericanPriceSurface — pass log-moneyness, apply chain rules here

**Files:**
- Modify: `src/option/table/american_price_surface.cpp:54-168`
- Modify: `src/option/table/american_price_surface.hpp:27-55`

**Step 1: Update price()**

In `american_price_surface.cpp`, update the `price()` method (lines 54-67):

```cpp
double AmericanPriceSurface::price(double spot, double strike, double tau,
                                   double sigma, double rate) const {
    if (surface_->metadata().content == SurfaceContent::RawPrice) {
        assert(strike == K_ref_ && "RawPrice surfaces require strike == K_ref");
        double x = std::log(spot / K_ref_);
        return surface_->value({x, tau, sigma, rate});
    }
    double x = std::log(spot / strike);
    double eep = surface_->value({x, tau, sigma, rate});
    auto eu = EuropeanOptionSolver(
        OptionSpec{.spot = spot, .strike = strike, .maturity = tau,
            .rate = rate, .dividend_yield = dividend_yield_, .option_type = type_}, sigma).solve().value();
    return eep * (strike / K_ref_) + eu.value();
}
```

**Step 2: Update delta() — apply chain rule**

In `american_price_surface.cpp`, update `delta()` (lines 69-79):

```cpp
double AmericanPriceSurface::delta(double spot, double strike, double tau,
                                   double sigma, double rate) const {
    double x = std::log(spot / strike);
    // partial(0, ...) returns ∂EEP/∂x where x = ln(S/K)
    // Need ∂EEP/∂S = (∂EEP/∂x) * (∂x/∂S) = (∂EEP/∂x) * (1/S)
    // delta_eep = (K/K_ref) * (∂EEP/∂x) * (1/S) = (1/K_ref) * (∂EEP/∂x) * (strike/spot)
    double dEdx = surface_->partial(0, {x, tau, sigma, rate});
    double eep_delta = (1.0 / K_ref_) * dEdx * (strike / spot);
    auto eu = EuropeanOptionSolver(
        OptionSpec{.spot = spot, .strike = strike, .maturity = tau,
            .rate = rate, .dividend_yield = dividend_yield_, .option_type = type_}, sigma).solve().value();
    return eep_delta + eu.delta();
}
```

**Step 3: Update gamma() — apply second-order chain rule**

In `american_price_surface.cpp`, update `gamma()` (lines 81-91):

```cpp
double AmericanPriceSurface::gamma(double spot, double strike, double tau,
                                   double sigma, double rate) const {
    double x = std::log(spot / strike);
    // ∂²V/∂S² from log-moneyness derivatives:
    // ∂V/∂S = (K/K_ref) * (∂EEP/∂x) / S
    // ∂²V/∂S² = (K/K_ref) * [(∂²EEP/∂x² - ∂EEP/∂x)] / S²
    double dEdx = surface_->partial(0, {x, tau, sigma, rate});
    double d2Edx2 = surface_->second_partial(0, {x, tau, sigma, rate});
    double eep_gamma = (strike / K_ref_) * (d2Edx2 - dEdx) / (spot * spot);
    auto eu = EuropeanOptionSolver(
        OptionSpec{.spot = spot, .strike = strike, .maturity = tau,
            .rate = rate, .dividend_yield = dividend_yield_, .option_type = type_}, sigma).solve().value();
    return eep_gamma + eu.gamma();
}
```

**Step 4: Update vega() and theta() — pass log-moneyness**

In `american_price_surface.cpp`, update `vega()` (lines 93-108) and `theta()` (lines 110-122):

For vega:
```cpp
    double x = std::log(spot / strike);
    double eep_vega = (strike / K_ref_) * surface_->partial(2, {x, tau, sigma, rate});
```

For theta:
```cpp
    double x = std::log(spot / strike);
    double eep_dtau = (strike / K_ref_) * surface_->partial(1, {x, tau, sigma, rate});
```

**Step 5: Update doc comments in header**

In `american_price_surface.hpp`, update the doc comments for delta and gamma to reflect the new chain rule location.

**Step 6: Build and run AmericanPriceSurface tests**

Run: `cd /home/kai/work/mango-option/.worktrees/log-moneyness && bazel test //tests:american_price_surface_test --test_output=all`
Expected: Tests should pass (prices/greeks unchanged, chain rule is mathematically equivalent)

**Step 7: Commit**

```bash
git add src/option/table/american_price_surface.cpp src/option/table/american_price_surface.hpp
git commit -m "Move chain rule from PriceTableSurface to AmericanPriceSurface

AmericanPriceSurface now passes log(spot/strike) and applies
chain rule corrections for delta and gamma locally.
Part of #373."
```

---

### Task 4: Update InterpolatedIVSolver — log-moneyness bounds checking

**Files:**
- Modify: `src/option/interpolated_iv_solver.hpp:126-137`
- Modify: `src/option/interpolated_iv_solver.cpp:87-102,197-205,232-237,257-262`

**Step 1: Update is_in_bounds() in header**

In `interpolated_iv_solver.hpp`, lines 126-137, compare `log(m)` against bounds:

```cpp
    bool is_in_bounds(const IVQuery& query, double vol) const {
        const double x = std::log(query.spot / query.strike);
        double rate_value = get_zero_rate(query.rate, query.maturity);
        return x >= m_range_.first && x <= m_range_.second &&
               query.maturity >= tau_range_.first && query.maturity <= tau_range_.second &&
               vol >= sigma_range_.first && vol <= sigma_range_.second &&
               rate_value >= r_range_.first && rate_value <= r_range_.second;
    }
```

**Step 2: Update extract_bounds() in .cpp**

In `interpolated_iv_solver.cpp`, lines 93-102, convert moneyness to log-moneyness:

```cpp
GridBounds extract_bounds(const IVGrid& grid) {
    // Convert moneyness bounds to log-moneyness
    auto minmax_m = std::minmax_element(grid.moneyness.begin(), grid.moneyness.end());
    auto minmax_v = std::minmax_element(grid.vol.begin(), grid.vol.end());
    auto minmax_r = std::minmax_element(grid.rate.begin(), grid.rate.end());
    return {
        .m_min = std::log(*minmax_m.first), .m_max = std::log(*minmax_m.second),
        .sigma_min = *minmax_v.first, .sigma_max = *minmax_v.second,
        .rate_min = *minmax_r.first, .rate_max = *minmax_r.second,
    };
}
```

**Step 3: Update build_standard() manual grid path**

In `interpolated_iv_solver.cpp`, lines 202-204, convert moneyness to log-moneyness before passing to `from_vectors`:

```cpp
    // Convert moneyness to log-moneyness for from_vectors
    std::vector<double> log_m;
    log_m.reserve(config.grid.moneyness.size());
    for (double m : config.grid.moneyness) {
        log_m.push_back(std::log(m));
    }
    auto setup = PriceTableBuilder<4>::from_vectors(
        std::move(log_m), path.maturity_grid, config.grid.vol, config.grid.rate,
        config.spot, GridAccuracyParams{}, config.option_type,
        config.dividend_yield);
```

**Step 4: Build and run IV solver tests**

Run: `cd /home/kai/work/mango-option/.worktrees/log-moneyness && bazel test //tests:interpolated_iv_solver_test --test_output=all`

**Step 5: Commit**

```bash
git add src/option/interpolated_iv_solver.hpp src/option/interpolated_iv_solver.cpp
git commit -m "Use log-moneyness bounds in InterpolatedIVSolver

Bounds checking now compares log(S/K) against log-space range.
Factory converts IVGrid moneyness to log-moneyness at boundary.
Part of #373."
```

---

### Task 5: Update grid estimator and adaptive builder

**Files:**
- Modify: `src/option/table/price_table_grid_estimator.hpp:178-296`
- Modify: `src/option/table/adaptive_grid_builder.cpp` (m_min/m_max writes, moneyness grid construction)
- Modify: `src/option/table/segmented_price_table_builder.cpp:100,139,145`

**Step 1: Update estimate_grid_for_price_table()**

In `price_table_grid_estimator.hpp`, lines 189-230. The function currently takes moneyness bounds and generates a log-uniform grid (which produces moneyness values). Change to take log-moneyness bounds and produce a uniform grid in log-space:

```cpp
inline PriceTableGridEstimate<4> estimate_grid_for_price_table(
    double log_m_min, double log_m_max,
    double tau_min, double tau_max,
    double sigma_min, double sigma_max,
    double r_min, double r_max,
    const PriceTableGridAccuracyParams<4>& params = {})
{
    // ... (point allocation unchanged) ...

    // Log-moneyness: uniform in log-space (already our native coordinate)
    estimate.grids[0] = detail::uniform_grid(log_m_min, log_m_max, n_m);

    // ... (rest unchanged) ...
}
```

**Step 2: Update estimate_grid_from_grid_bounds()**

In `price_table_grid_estimator.hpp`, lines 250-296. Compute log-moneyness bounds instead of moneyness:

```cpp
    // Compute log-moneyness bounds from strikes
    double log_m_min = std::log(spot / *std::max_element(strikes.begin(), strikes.end()));
    double log_m_max = std::log(spot / *std::min_element(strikes.begin(), strikes.end()));
    if (log_m_min > log_m_max) std::swap(log_m_min, log_m_max);

    // Add padding in log-space
    double pad = 0.01 * (log_m_max - log_m_min);
    log_m_min -= pad;
    log_m_max += pad;
```

**Step 3: Update adaptive_grid_builder.cpp metadata writes**

At lines 969-970, the adaptive builder writes `m_grid.front()`/`.back()` to metadata. After the change, `m_grid` contains log-moneyness values, so the writes remain correct. However, any code that computes `m_grid` from `spot/strike` must now compute `log(spot/strike)`. Audit the `build()` and `build_segmented()` methods for these patterns:

- Line 760-767: `m = chain.spot / strike` → `m = std::log(chain.spot / strike)`
- Lines 1075-1077: `initial_grids.moneyness.push_back(chain.spot / strike)` → `std::log(chain.spot / strike)`
- Lines 584-586: `min_m = domain.moneyness.front()` — if `domain.moneyness` (from IVGrid) is still in moneyness, need to log-convert at the boundary

**Step 4: Update SegmentedPriceTableBuilder**

In `segmented_price_table_builder.cpp`:
- Line 100: `config.grid.moneyness.size() < 4` — IVGrid still uses moneyness (user-facing), so this check stays
- Line 139: `expanded_m_grid = config.grid.moneyness` — must convert to log-moneyness
- Line 145: `m_min_expanded = config.grid.moneyness.front() - total_div / K_ref` — expansion logic in moneyness space, then convert to log

The segmented builder uses IVGrid moneyness values and must convert to log-moneyness before passing to `from_vectors()`.

**Step 5: Build and run all table tests**

Run: `cd /home/kai/work/mango-option/.worktrees/log-moneyness && bazel test //tests:price_table_builder_test //tests:segmented_price_table_builder_test //tests:adaptive_grid_builder_test --test_output=all`

**Step 6: Commit**

```bash
git add src/option/table/price_table_grid_estimator.hpp src/option/table/adaptive_grid_builder.cpp src/option/table/segmented_price_table_builder.cpp
git commit -m "Produce log-moneyness grids in estimator and builders

Grid estimator generates uniform log-moneyness grids.
Adaptive and segmented builders convert at IVGrid boundary.
Part of #373."
```

---

### Task 6: Stub out PriceTableWorkspace persistence

**Files:**
- Modify: `src/option/table/price_table_workspace.cpp`
- Modify: `src/option/table/price_table_workspace.hpp:45-46`

**Step 1: Update doc comments**

In `price_table_workspace.hpp`, update `m_min`/`m_max` parameter docs:

```cpp
    /// @param m_min Minimum log-moneyness ln(S/K)
    /// @param m_max Maximum log-moneyness ln(S/K)
```

And the accessor docs:

```cpp
    /// Log-moneyness bounds
    double m_min() const { return m_min_; }
    double m_max() const { return m_max_; }
```

**Step 2: Stub save() and load()**

In `price_table_workspace.cpp`, replace `save()` body with:

```cpp
    return std::unexpected(std::string("Persistence temporarily disabled (issue #373)"));
```

Replace `load()` body with:

```cpp
    return std::unexpected(LoadError::UNSUPPORTED_VERSION);
```

Keep `create()` functional — it is used by tests and Python bindings for in-memory operation.

**Step 3: Build**

Run: `cd /home/kai/work/mango-option/.worktrees/log-moneyness && bazel build //src/option/...`

**Step 4: Commit**

```bash
git add src/option/table/price_table_workspace.cpp src/option/table/price_table_workspace.hpp
git commit -m "Stub PriceTableWorkspace persistence for log-moneyness

Save/load disabled pending schema redesign.
In-memory create() still functional.
Part of #373."
```

---

### Task 7: Update Python bindings

**Files:**
- Modify: `src/python/mango_bindings.cpp`

**Step 1: Update workspace creation parameter names and docs**

Find the `create_workspace` wrapper and update parameter names from `m_min`/`m_max` to reflect log-moneyness semantics. Update docstrings.

**Step 2: Update PriceTableMetadata binding**

Update the `.def_readwrite("m_min", ...)` docstring to say "Minimum log-moneyness ln(S/K)".

**Step 3: Build Python bindings**

Run: `cd /home/kai/work/mango-option/.worktrees/log-moneyness && bazel build //src/python:mango_option`

**Step 4: Commit**

```bash
git add src/python/mango_bindings.cpp
git commit -m "Update Python bindings for log-moneyness metadata

Parameter docs reflect log-moneyness semantics.
Part of #373."
```

---

### Task 8: Update tests

**Files:**
- Modify: `tests/price_table_surface_test.cc`
- Modify: `tests/american_price_surface_test.cc`
- Modify: `tests/interpolated_iv_solver_test.cc`
- Modify: `tests/price_table_workspace_test.cc`
- Modify: `tests/test_bindings.py`

**Step 1: Update price_table_surface_test.cc**

Any test that constructs `PriceTableAxes` with moneyness values and passes to `PriceTableSurface::build()` must use log-moneyness instead. Any test that calls `value()`, `partial()`, or `second_partial()` with moneyness coordinates must pass log-moneyness. m_min/m_max assertions must use log values.

**Step 2: Update american_price_surface_test.cc**

Tests calling `m_min()`/`m_max()` must expect log-moneyness values. End-to-end price/delta/gamma values should remain the same (the math is equivalent).

**Step 3: Update interpolated_iv_solver_test.cc**

Bounds assertions update to log-moneyness. IV solve results should remain the same.

**Step 4: Disable workspace persistence tests**

In `price_table_workspace_test.cc`, disable save/load tests (mark as `DISABLED_`). Keep in-memory `create()` tests, updating m_min/m_max values to log-space.

In `tests/test_bindings.py`, disable workspace save/load tests.

**Step 5: Run full test suite**

Run: `cd /home/kai/work/mango-option/.worktrees/log-moneyness && bazel test //... --test_output=errors`
Expected: All 116 tests pass

**Step 6: Commit**

```bash
git add tests/
git commit -m "Update tests for log-moneyness axes

Coordinates and bounds assertions use log-space values.
End-to-end prices, deltas, gammas unchanged.
Workspace persistence tests disabled pending redesign.
Part of #373."
```

---

### Task 9: Final verification and cleanup

**Step 1: Run full test suite**

Run: `cd /home/kai/work/mango-option/.worktrees/log-moneyness && bazel test //...`
Expected: All tests pass

**Step 2: Build benchmarks**

Run: `cd /home/kai/work/mango-option/.worktrees/log-moneyness && bazel build //benchmarks/...`

**Step 3: Build Python bindings**

Run: `cd /home/kai/work/mango-option/.worktrees/log-moneyness && bazel build //src/python:mango_option`

**Step 4: Commit any remaining fixes**

If any tests or builds fail, fix and commit.
