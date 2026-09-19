# Round-Trip IV-Error Metric Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the adaptive builders' vega-scaled IV-error metric and its unanchored admission filters with a surface-independent reference-bracket admission test and the product's own price-to-σ inversion run on the candidate surface, reporting statuses instead of a scalar-or-nothing.

**Architecture:** A new low-level `//src/option:surface_inversion` target holds the product's bracket policy, multi-root screen and Brent inversion; `InterpolatedIVSolver::solve` and the refinement loop both call it. `adaptive_metrics` gains a reference oracle (High profile, one nested grid family per point, six-solve stencil with two-grid uncertainty estimates) and a round-trip scorer returning `PointScore{status, iv_error, price_residual}`. `run_refinement`, the segmented final gates and the builders consume statuses: fewer holdout failures rank first, any resolved failure makes a candidate non-viable, `kViabilityBound` disappears, failures feed refinement bins, and diagnostics report unresolved/unsupported/failure counts.

**Tech Stack:** C++23, Bazel (bzlmod), GoogleTest, pybind11, USDT probes via `ivcalc_trace.h`.

**Spec:** `docs/superpowers/specs/2026-09-19-round-trip-iv-metric-500-design.md` (rev 4). Executors read the spec's §0, §4 (D1–D10, 4.11, laws L1–L6) alongside each task.

## Global Constraints

- Library code: `std::expected` for errors, no exceptions, no `printf`/`fprintf`; USDT probes only (`src/support/ivcalc_trace.h`).
- Every new source file starts with `// SPDX-License-Identifier: MIT` (`.cc/.hpp/.cpp`) or `# SPDX-License-Identifier: MIT` (BUILD, `.py`).
- No C ABI layout change: `MangoAdaptiveGridParams` keeps `vega_floor` at offset 56 (`src/ffi/mango_c_api.h:233`, `crates/mango-option-sys/tests/layout.rs:59`).
- `kReferenceAccuracy = GridAccuracyProfile::High`, `F_s = 3`, grid rounding `n ≡ 1 (mod 16)` within `[min_spatial_points, max_spatial_points]`.
- Language law L5: comments, diagnostics and docs say "estimate", "calibrated", "outcome", "on the declared sample set"; never "bound", "guarantee", "for every price".
- Every bug-motivated test carries `// Regression:` and `// Bug:` lines (CLAUDE.md).
- Commit subjects: imperative, ≤ 50 chars, no period; body wrapped at 72.
- Run tests with `TMPDIR` pointed at the session scratch dir: `D="/tmp/codex-skills/$CLAUDE_CODE_SESSION_ID"; mkdir -p "$D"; TMPDIR="$D" bazel test ...`.
- Baseline in this worktree: `bazel test //...` = 156/156 green at `a1f38e77`.

---

## File map

| File | Responsibility after this plan |
|---|---|
| `src/option/surface_inversion.hpp/.cpp` (new) | `SurfaceInversionPolicy`, `detail::ObjectiveRef`, `detail::BracketScreen`, `detail::screen_bracket`, `effective_sigma_bracket`, `invert_price_on_surface`. No table/builder deps. |
| `src/option/interpolated_iv_solver.hpp/.cpp` | `solve()` = validate → bracket → bounds → `invert_price_on_surface`. Screen code removed (moved). |
| `src/option/table/adaptive_grid_types.hpp` | `PointStatus`, `PointScore`, new `BuildDiagnostics`/`IterationStats` fields, `vega_floor` deprecation comment. |
| `src/option/table/adaptive_metrics.hpp/.cpp` | Constants, `ReferenceGridFamily`, `make_reference_grid_family`, `ReferenceOracle`, `ReferenceSolveCounter`, `make_stencil_refs_fn`, `make_validate_fn` (High), `make_round_trip_score_fn`. Old helpers deleted. |
| `src/option/table/adaptive_refinement.hpp/.cpp` | `SurfaceHandle::vega`, new `ErrorRefs`, `ScoreErrorFn` returning `PointScore`, `ErrorBins::failure_counts`, `SampleEval`/`FinalScore` statuses, ordering/viability/restart, no `kViabilityBound`, probes, `scan_monotonicity` without `vega_floor`. |
| `src/option/table/bspline/bspline_adaptive.cpp` | Handles carry vega; stencil refs via `ReferenceOracle`; probe adapter scales all monetary fields; `maturity_is_supported` for the segmented path; counters; diagnostics fill. |
| `src/option/table/chebyshev/chebyshev_adaptive.cpp` | Same; **new** probe adapter for the segmented sizing loop; `maturity_is_supported` from `seg_bounds_`/`seg_is_gap_`. |
| `src/support/ivcalc_trace.h` | `DTRACE_PROBE7/8` fallbacks; `MANGO_TRACE_ADAPTIVE_VALIDATION_REFUSED`, `MANGO_TRACE_ADAPTIVE_NO_VIABLE_SURFACE`. |
| `src/python/mango_bindings.cpp`, `src/ffi/mango_c_api.h`, `crates/mango-option/src/interp.rs`, `crates/mango-option-sys/src/lib.rs` | Deprecation comments; Python dict keys. |
| `benchmarks/interp_iv_safety.cc` | Drop `kVegaFloor`, `kTVKThreshold` mirrors. |
| `tests/surface_inversion_test.cc` (new), `tests/reference_oracle_test.cc` (new), `tests/reference_oracle_calibration_test.cc` (new, slow), existing adaptive tests | Coverage per spec D9. |
| `docs/MATHEMATICAL_FOUNDATIONS.md`, `docs/API_GUIDE.md`, `docs/ARCHITECTURE.md`, `CONTEXT.md` | Spec D10. |

Task order keeps the tree compiling at every boundary: Task 1 adds; Task 2 refactors the product path without behaviour change; Tasks 3–5 add the new metric types alongside the old; Task 6 switches the loop; Tasks 7–8 switch the builders; Task 9 deletes the old helpers; Tasks 10–13 are tests, bindings, docs, verification.

---

### Task 1: Reference grid family and oracle

**Files:**
- Modify: `src/option/table/adaptive_metrics.hpp` (append after the `make_validate_fn` declaration at `:43`)
- Modify: `src/option/table/adaptive_metrics.cpp`
- Modify: `src/option/table/BUILD.bazel:70-84` (add deps `//src/option:grid_spec_types`, `//src/pde/core:grid`, `//src/pde/core:time_domain` — use the actual target names found with `grep -n "name = " src/option/BUILD.bazel src/pde/core/BUILD.bazel`)
- Create: `tests/reference_oracle_test.cc`
- Modify: `tests/BUILD.bazel` (new `cc_test` after `adaptive_refinement_unit_test`)

**Interfaces:**
- Produces:
  ```cpp
  inline constexpr GridAccuracyProfile kReferenceAccuracy = GridAccuracyProfile::High;
  inline constexpr double kRichardsonSafetyFactor = 3.0;     // Roache, two-grid
  inline constexpr double kReferenceConvergenceOrder = 1.0;  // calibrated by Task 11; must stay <= measured minimum
  struct ReferenceGridFamily { std::vector<PDEGridConfig> levels; std::vector<size_t> point_counts; std::vector<size_t> time_steps; bool rounded_down = false; };
  std::expected<ReferenceGridFamily, ValidationError> make_reference_grid_family(const PricingParams& params, const GridAccuracyParams& accuracy, size_t levels);
  struct ReferenceSolveCounter { std::atomic<size_t> fine_attempts{0}, coarse_attempts{0}, fine_failures{0}, coarse_failures{0}; };
  struct ReferenceOracle { double dividend_yield; OptionType option_type; std::vector<Dividend> discrete_dividends; std::optional<double> reference_maturity; GridAccuracyParams accuracy;
      PricingParams contract(double spot, double strike, double tau, double sigma, double rate) const;
      std::expected<double, SolverError> solve(const PricingParams& p, const PDEGridConfig& grid) const; };
  ```
- Consumes: `estimate_pde_grid` (`grid_spec_types.hpp:111`), `GridSpec<double>::multi_sinh_spaced/sinh_spaced` + accessors `type()/x_min()/x_max()/concentration()/clusters()` (`src/pde/core/grid.hpp:175-183`), `TimeDomain::n_steps()`, `AmericanOptionSolver::create(params, PDEGridSpec)`, `rolled_dividends`/`filter_and_merge_dividends` (already used in `adaptive_metrics.cpp`).

- [ ] **Step 1: Write the failing tests**

```cpp
// SPDX-License-Identifier: MIT
#include <gtest/gtest.h>
#include "mango/option/table/adaptive_metrics.hpp"
#include "mango/option/grid_spec_types.hpp"
#include <cmath>

using namespace mango;

static PricingParams put_1y_with_divs() {
    PricingParams p;
    p.spot = 100.0; p.strike = 100.0; p.maturity = 1.0; p.rate = 0.05;
    p.dividend_yield = 0.02; p.option_type = OptionType::PUT; p.volatility = 0.2;
    p.discrete_dividends = {{0.25, 0.5}, {0.5, 0.5}, {0.75, 0.5}};
    return p;
}

// Spec D1: every level is odd, nested (every 2^k-th node), shares the middle
// index, and the fine grid does not depend on how many levels were asked for.
TEST(ReferenceGridFamily, LevelsAreNestedOddAndShareMiddleNode) {
    const auto acc = make_grid_accuracy(kReferenceAccuracy);
    auto fam3 = make_reference_grid_family(put_1y_with_divs(), acc, 3);
    ASSERT_TRUE(fam3.has_value());
    ASSERT_EQ(fam3->levels.size(), 4u);
    EXPECT_EQ(fam3->point_counts[0] % 16, 1u);
    EXPECT_FALSE(fam3->rounded_down);
    auto fine = fam3->levels[0].grid_spec.generate();
    auto fine_pts = fine.view().span();
    for (size_t k = 1; k <= 3; ++k) {
        auto pts = fam3->levels[k].grid_spec.generate().view().span();
        EXPECT_EQ(pts.size() % 2, 1u) << "level " << k;
        ASSERT_EQ((fine_pts.size() - 1) >> k, pts.size() - 1) << "level " << k;
        for (size_t j = 0; j < pts.size(); ++j) {
            EXPECT_NEAR(pts[j], fine_pts[j << k], 1e-12 * (1.0 + std::abs(fine_pts[j << k])))
                << "level " << k << " node " << j;
        }
        EXPECT_DOUBLE_EQ(pts[(pts.size() - 1) / 2], fine_pts[(fine_pts.size() - 1) / 2]);
        EXPECT_EQ(fam3->levels[k].n_time, (fam3->levels[0].n_time + (1u << k) - 1) >> k);
        EXPECT_TRUE(fam3->levels[k].mandatory_times.empty());
    }
    auto fam1 = make_reference_grid_family(put_1y_with_divs(), acc, 1);
    ASSERT_TRUE(fam1.has_value());
    EXPECT_EQ(fam1->point_counts[0], fam3->point_counts[0]);
    EXPECT_EQ(fam1->levels[0].n_time, fam3->levels[0].n_time);
}

// The fine count never exceeds the profile's strict cap; below the cap it is
// the smallest n = 1 (mod 16) at or above the estimate.
TEST(ReferenceGridFamily, RoundsUpWithinCapElseDownAndFlags) {
    auto acc = make_grid_accuracy(kReferenceAccuracy);
    auto p = put_1y_with_divs();
    auto est = estimate_pde_grid(p, acc);
    ASSERT_TRUE(est.has_value());
    const size_t n0 = est->first.n_points();
    auto fam = make_reference_grid_family(p, acc, 1);
    ASSERT_TRUE(fam.has_value());
    EXPECT_GE(fam->point_counts[0], n0);
    EXPECT_LT(fam->point_counts[0], n0 + 16);
    // Force the cap right at the estimate: rounding up is impossible.
    acc.max_spatial_points = n0;
    acc.min_spatial_points = std::min(acc.min_spatial_points, n0);
    auto capped = make_reference_grid_family(p, acc, 1);
    ASSERT_TRUE(capped.has_value());
    EXPECT_LE(capped->point_counts[0], n0);
    EXPECT_EQ(capped->point_counts[0] % 16, 1u);
    EXPECT_TRUE(capped->rounded_down);
}

// The oracle rolls dividends onto a fixed-expiry contract exactly as
// make_validate_fn does, and solves on the grid it is handed.
TEST(ReferenceOracle, SolvesOnGivenGridAndMatchesValidateFn) {
    ReferenceOracle oracle{.dividend_yield = 0.02, .option_type = OptionType::PUT,
                           .discrete_dividends = {{0.25, 0.5}, {0.5, 0.5}, {0.75, 0.5}},
                           .reference_maturity = 1.0,
                           .accuracy = make_grid_accuracy(kReferenceAccuracy)};
    auto p = oracle.contract(100.0, 110.0, 0.4, 0.2, 0.05);
    auto fam = make_reference_grid_family(p, oracle.accuracy, 1);
    ASSERT_TRUE(fam.has_value());
    auto v = oracle.solve(p, fam->levels[0]);
    ASSERT_TRUE(v.has_value());
    auto validate = make_validate_fn(0.02, OptionType::PUT,
                                     {{0.25, 0.5}, {0.5, 0.5}, {0.75, 0.5}}, 1.0);
    auto ref = validate(100.0, 110.0, 0.4, 0.2, 0.05);
    ASSERT_TRUE(ref.has_value());
    // Same profile; the family adds at most 15 spatial points, so the two agree
    // far inside the High profile's own two-grid difference.
    EXPECT_NEAR(*v, *ref, 1e-4);
    auto half = oracle.solve(p, fam->levels[1]);
    ASSERT_TRUE(half.has_value());
    EXPECT_NE(*half, *v);
}
```

Add to `tests/BUILD.bazel` after the `adaptive_refinement_unit_test` target:

```python
cc_test(
    name = "reference_oracle_test",
    size = "medium",
    srcs = ["reference_oracle_test.cc"],
    deps = [
        "//src/option:grid_spec_types",
        "//src/option/table:adaptive_metrics",
        "@googletest//:gtest_main",
    ],
)
```

(Match the `gtest_main` label used by neighbouring targets.)

- [ ] **Step 2: Run the test to verify it fails**

Run: `TMPDIR="$D" bazel test //tests:reference_oracle_test --test_output=errors`
Expected: build FAILED with `make_reference_grid_family` / `ReferenceOracle` undeclared.

- [ ] **Step 3: Implement**

In `adaptive_metrics.hpp` add includes `<atomic>`, `<memory>`, `"mango/option/grid_spec_types.hpp"` and the declarations from **Interfaces**. In `adaptive_metrics.cpp`:

```cpp
namespace {

// Rebuild the same generator family at a different point count.  Every
// GridSpec generator is a pure map of eta = i/(n-1) (grid.hpp generate()),
// so re-sampling at (n-1)/2^k + 1 points yields the every-2^k-th-node
// subsequence exactly (up to floating-point rounding of eta).
std::expected<GridSpec<double>, ValidationError>
resample(const GridSpec<double>& spec, size_t n) {
    switch (spec.type()) {
        case GridSpec<double>::Type::MultiSinhSpaced: {
            std::vector<MultiSinhCluster<double>> clusters(
                spec.clusters().begin(), spec.clusters().end());
            // auto_merge=false: the clusters were merged when the estimator
            // built the fine spec; merging again could move them.
            return GridSpec<double>::multi_sinh_spaced(
                spec.x_min(), spec.x_max(), n, std::move(clusters), /*auto_merge=*/false);
        }
        case GridSpec<double>::Type::SinhSpaced:
            return GridSpec<double>::sinh_spaced(spec.x_min(), spec.x_max(), n, spec.concentration());
        case GridSpec<double>::Type::Uniform:
            return GridSpec<double>::uniform(spec.x_min(), spec.x_max(), n);
        case GridSpec<double>::Type::LogSpaced:
            return GridSpec<double>::log_spaced(spec.x_min(), spec.x_max(), n);
    }
    return std::unexpected(ValidationError(ValidationErrorCode::InvalidGridSize, static_cast<double>(n)));
}

constexpr size_t kFamilyModulus = 16;  // odd through three coarsenings (spec D1)

}  // namespace

std::expected<ReferenceGridFamily, ValidationError>
make_reference_grid_family(const PricingParams& params,
                           const GridAccuracyParams& accuracy,
                           size_t levels) {
    auto est = estimate_pde_grid(params, accuracy);
    if (!est) return std::unexpected(est.error());
    const auto& [spec0, time0] = *est;
    const size_t n0 = spec0.n_points();
    const size_t cap = accuracy.max_spatial_points;
    const size_t floor = std::max<size_t>(accuracy.min_spatial_points, 3);
    ReferenceGridFamily fam;
    // Smallest n >= n0 with n = 1 (mod 16), if it fits under the strict cap.
    size_t n = n0 + ((kFamilyModulus + 1 - (n0 % kFamilyModulus)) % kFamilyModulus);
    if (n > cap) {
        // Largest n <= cap with n = 1 (mod 16) that is still >= floor.
        n = cap - ((cap % kFamilyModulus) + kFamilyModulus - 1) % kFamilyModulus;
        if (n < floor || n < 17) {
            return std::unexpected(ValidationError(
                ValidationErrorCode::InvalidGridSize, static_cast<double>(cap)));
        }
        fam.rounded_down = true;
    }
    const size_t n_time0 = time0.n_steps();
    for (size_t k = 0; k <= levels; ++k) {
        const size_t nk = ((n - 1) >> k) + 1;
        auto spec = resample(spec0, nk);
        if (!spec) return std::unexpected(spec.error());
        const size_t tk = (n_time0 + (size_t{1} << k) - 1) >> k;
        // mandatory_times stays empty: resolve_grid merges the dividend taus
        // into every explicit config (american_option.cpp:68), and copying
        // fine time nodes here would stop the coarse level from coarsening.
        fam.levels.push_back(PDEGridConfig{.grid_spec = std::move(*spec),
                                           .n_time = tk,
                                           .mandatory_times = {}});
        fam.point_counts.push_back(nk);
        fam.time_steps.push_back(tk);
    }
    return fam;
}

PricingParams ReferenceOracle::contract(double spot, double strike, double tau,
                                        double sigma, double rate) const {
    PricingParams p;
    p.spot = spot; p.strike = strike; p.maturity = tau; p.rate = rate;
    p.dividend_yield = dividend_yield; p.option_type = option_type;
    p.volatility = sigma;
    p.discrete_dividends = reference_maturity
        ? rolled_dividends(discrete_dividends, *reference_maturity, tau)
        : filter_and_merge_dividends(discrete_dividends, tau);
    return p;
}

std::expected<double, SolverError>
ReferenceOracle::solve(const PricingParams& p, const PDEGridConfig& grid) const {
    auto solver = AmericanOptionSolver::create(p, PDEGridSpec{grid});
    if (!solver) return std::unexpected(SolverError{.code = SolverErrorCode::InvalidConfiguration});
    auto r = solver->solve();
    if (!r) return std::unexpected(r.error());
    const double v = r->value();
    if (!std::isfinite(v)) return std::unexpected(SolverError{});
    return v;
}
```

Also change `make_validate_fn` to build its solver through a `ReferenceOracle` at `kReferenceAccuracy` on the estimated grid (`AmericanOptionSolver::create(p, PDEGridSpec{make_grid_accuracy(kReferenceAccuracy)})`) so single-price callers get the High profile too. Keep its signature.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `TMPDIR="$D" bazel test //tests:reference_oracle_test //tests:adaptive_refinement_unit_test --test_output=errors`
Expected: PASS. If `adaptive_refinement_unit_test`'s fixed-expiry oracle test (`:1298`, `EXPECT_DOUBLE_EQ` against `solve_american_option`) fails because the oracle now runs at High, change its comparator to `AmericanOptionSolver::create(p, PDEGridSpec{make_grid_accuracy(kReferenceAccuracy)})` and keep its dividend-rolling assertion (spec D9).

- [ ] **Step 5: Commit**

```bash
git add src/option/table/adaptive_metrics.hpp src/option/table/adaptive_metrics.cpp src/option/table/BUILD.bazel tests/reference_oracle_test.cc tests/BUILD.bazel tests/adaptive_refinement_unit_test.cc
git commit -m "Add nested reference grid family and High oracle"
```

---

### Task 2: Extract the product inversion into `surface_inversion`

**Files:**
- Create: `src/option/surface_inversion.hpp`, `src/option/surface_inversion.cpp`
- Modify: `src/option/BUILD.bazel` (new `cc_library` before `interpolated_iv_solver_core` at `:154`; add `":surface_inversion"` to `interpolated_iv_solver_core` deps)
- Modify: `src/option/interpolated_iv_solver.hpp:88-160` (remove `ObjectiveRef`, `BracketScreen`, `screen_bracket` declarations; include the new header), `:616-643` (`adaptive_bounds`), `:645-800` (`solve`)
- Modify: `src/option/interpolated_iv_solver.cpp:20-176` (move `screen_bracket` body verbatim)
- Create: `tests/surface_inversion_test.cc`; modify `tests/BUILD.bazel`

**Interfaces:**
- Produces (`surface_inversion.hpp`, namespace `mango`):
  ```cpp
  struct SurfaceInversionPolicy {
      double config_sigma_min = 0.01, config_sigma_max = 3.0;
      double published_sigma_min = 0.0, published_sigma_max = 0.0;
      double vega_threshold = 1e-4;
      bool detect_multiple_roots = true;
      double tolerance = 1e-6;
      size_t max_iter = 50;
  };
  namespace detail { class ObjectiveRef; struct BracketScreen; BracketScreen screen_bracket(ObjectiveRef, double, double, double, double); }
  std::pair<double, double> effective_sigma_bracket(double spot, double strike, OptionType type, double target_price, const SurfaceInversionPolicy& policy) noexcept;
  std::expected<IVSuccess, IVError> invert_price_on_surface(detail::ObjectiveRef price, detail::ObjectiveRef vega, double target_price, std::pair<double, double> bracket, double spot, const SurfaceInversionPolicy& policy) noexcept;
  ```
  `price`/`vega` take σ only (the caller binds the other coordinates). `IVSuccess::used_rate_approximation` is left `false`; the solver sets it.

- [ ] **Step 1: Write the failing equivalence test**

`tests/surface_inversion_test.cc`:

```cpp
// SPDX-License-Identifier: MIT
#include <gtest/gtest.h>
#include "mango/option/surface_inversion.hpp"
#include <cmath>

using namespace mango;

// A monotone synthetic "surface": price = 10 + 40*(sigma - 0.2).
static double lin_price(double s) { return 10.0 + 40.0 * (s - 0.2); }
static double lin_vega(double) { return 40.0; }

TEST(SurfaceInversion, EffectiveBracketAppliesCapConfigPublishedAndFallback) {
    SurfaceInversionPolicy pol{.published_sigma_min = 0.1, .published_sigma_max = 0.5};
    // time_value_pct = (12-0)/12 = 1 > 0.5 -> cap 3.0 -> [0.1, 0.5]
    auto b = effective_sigma_bracket(100.0, 100.0, OptionType::PUT, 12.0, pol);
    EXPECT_DOUBLE_EQ(b.first, 0.1); EXPECT_DOUBLE_EQ(b.second, 0.5);
    // Config narrower than published wins.
    pol.config_sigma_min = 0.15; pol.config_sigma_max = 0.4;
    b = effective_sigma_bracket(100.0, 100.0, OptionType::PUT, 12.0, pol);
    EXPECT_DOUBLE_EQ(b.first, 0.15); EXPECT_DOUBLE_EQ(b.second, 0.4);
    // Disjoint -> fallback to the published range (interpolated_iv_solver.hpp:636).
    pol.config_sigma_min = 0.6; pol.config_sigma_max = 0.9;
    b = effective_sigma_bracket(100.0, 100.0, OptionType::PUT, 12.0, pol);
    EXPECT_DOUBLE_EQ(b.first, 0.1); EXPECT_DOUBLE_EQ(b.second, 0.5);
}

TEST(SurfaceInversion, InvertsMonotoneSurface) {
    SurfaceInversionPolicy pol{.published_sigma_min = 0.1, .published_sigma_max = 0.5};
    auto r = invert_price_on_surface(lin_price, lin_vega, lin_price(0.31),
                                     {0.1, 0.5}, 100.0, pol);
    ASSERT_TRUE(r.has_value());
    EXPECT_NEAR(r->implied_vol, 0.31, 1e-7);
    EXPECT_FALSE(r->used_rate_approximation);
}

TEST(SurfaceInversion, ReportsProductErrorCodes) {
    SurfaceInversionPolicy pol{.published_sigma_min = 0.1, .published_sigma_max = 0.5};
    // Target above the surface's range -> BracketingFailed.
    auto no_root = invert_price_on_surface(lin_price, lin_vega, lin_price(0.9), {0.1, 0.5}, 100.0, pol);
    ASSERT_FALSE(no_root.has_value());
    EXPECT_EQ(no_root.error().code, IVErrorCode::BracketingFailed);
    // Flat surface -> VegaTooSmall from the quartile pre-check.
    auto flat = invert_price_on_surface([](double) { return 10.0; }, [](double) { return 0.0; },
                                        10.0, {0.1, 0.5}, 100.0, pol);
    ASSERT_FALSE(flat.has_value());
    EXPECT_EQ(flat.error().code, IVErrorCode::VegaTooSmall);
    // Non-monotone with two crossings -> MultipleRoots.
    auto bump = [](double s) { return 10.0 + 40.0 * (s - 0.2) - 30.0 * (s - 0.2) * (s - 0.2) * 10.0; };
    auto multi = invert_price_on_surface(bump, [](double) { return 1.0; }, 10.5, {0.1, 0.5}, 100.0, pol);
    if (!multi.has_value()) EXPECT_EQ(multi.error().code, IVErrorCode::MultipleRoots);
    // NaN interior -> NumericalInstability.
    auto nan_price = [](double s) { return s > 0.3 ? std::nan("") : lin_price(s); };
    auto nf = invert_price_on_surface(nan_price, lin_vega, lin_price(0.25), {0.1, 0.5}, 100.0, pol);
    ASSERT_FALSE(nf.has_value());
    EXPECT_EQ(nf.error().code, IVErrorCode::NumericalInstability);
}
```

BUILD target: `cc_test(name = "surface_inversion_test", size = "small", srcs = [...], deps = ["//src/option:surface_inversion", "@googletest//:gtest_main"])`.

- [ ] **Step 2: Run to verify it fails**

Run: `TMPDIR="$D" bazel test //tests:surface_inversion_test --test_output=errors`
Expected: build FAILED (header missing).

- [ ] **Step 3: Create the target**

`src/option/BUILD.bazel` (before `interpolated_iv_solver_core`):

```python
cc_library(
    name = "surface_inversion",
    srcs = ["surface_inversion.cpp"],
    hdrs = ["surface_inversion.hpp"],
    deps = [
        ":iv_result",
        ":option_spec",
        "//src/math:root_finding",
        "//src/support:error_types",
        "//src/support:ivcalc_trace_hdr",
    ],
    copts = ["-Wall", "-Wextra", "-Werror", "-O3"],
    visibility = ["//visibility:public"],
    strip_include_prefix = "/src/option",
    include_prefix = "mango/option",
)
```

`surface_inversion.hpp`: move `detail::ObjectiveRef` and `detail::BracketScreen` + the `screen_bracket` declaration verbatim from `interpolated_iv_solver.hpp:88-150`, add `SurfaceInversionPolicy` and the two free functions. `surface_inversion.cpp`: move the `screen_bracket` body verbatim from `interpolated_iv_solver.cpp:24-176`, then:

```cpp
std::pair<double, double> effective_sigma_bracket(
    double spot, double strike, OptionType type, double target_price,
    const SurfaceInversionPolicy& policy) noexcept {
    // Verbatim port of InterpolatedIVSolver::adaptive_bounds.
    const double intrinsic = intrinsic_value(spot, strike, type);
    const double time_value = target_price - intrinsic;
    const double time_value_pct = time_value / target_price;
    double sigma_upper;
    if (time_value_pct > 0.5) sigma_upper = 3.0;
    else if (time_value_pct > 0.2) sigma_upper = 2.0;
    else sigma_upper = 1.5;
    double lo = std::max(policy.config_sigma_min, policy.published_sigma_min);
    double hi = std::min({sigma_upper, policy.config_sigma_max, policy.published_sigma_max});
    if (lo >= hi) { lo = policy.published_sigma_min; hi = policy.published_sigma_max; }
    return {lo, hi};
}

std::expected<IVSuccess, IVError> invert_price_on_surface(
    detail::ObjectiveRef price, detail::ObjectiveRef vega, double target_price,
    std::pair<double, double> bracket, double spot,
    const SurfaceInversionPolicy& policy) noexcept {
    const auto [sigma_min, sigma_max] = bracket;
    // Vega pre-check: verbatim port of interpolated_iv_solver.hpp:686-712.
    if (policy.vega_threshold > 0.0) {
        const double span = sigma_max - sigma_min;
        const double probes[3] = {sigma_min + 0.25 * span, sigma_min + 0.5 * span, sigma_min + 0.75 * span};
        double max_vega = -std::numeric_limits<double>::infinity();
        for (double sv : probes) {
            const double v = vega(sv);
            if (!std::isfinite(v)) {
                return std::unexpected(IVError{.code = IVErrorCode::NumericalInstability,
                    .iterations = 0, .final_error = std::numeric_limits<double>::quiet_NaN(), .last_vol = sv});
            }
            max_vega = std::max(max_vega, v);
        }
        if (max_vega < policy.vega_threshold) {
            return std::unexpected(IVError{.code = IVErrorCode::VegaTooSmall,
                .iterations = 0, .final_error = max_vega, .last_vol = std::nullopt});
        }
    }
    const auto objective = [&](double s) { return price(s) - target_price; };
    double brent_lo = sigma_min, brent_hi = sigma_max;
    bool check_slope = false; double f_lo = 0.0, f_hi = 0.0;
    if (policy.detect_multiple_roots) {
        auto screen = detail::screen_bracket(objective, sigma_min, sigma_max, spot, policy.tolerance);
        if (screen.refusal.has_value()) return std::unexpected(*screen.refusal);
        if (screen.boundary_root.has_value()) return *screen.boundary_root;
        brent_lo = screen.lo; brent_hi = screen.hi;
        check_slope = screen.check_slope; f_lo = screen.f_lo; f_hi = screen.f_hi;
    }
    RootFindingConfig cfg{.max_iter = policy.max_iter, .brent_tol_abs = policy.tolerance};
    auto result = find_root(objective, brent_lo, brent_hi, cfg);
    if (!result.has_value()) {
        // Verbatim port of the switch at interpolated_iv_solver.hpp:760-786.
        ...
    }
    if (check_slope) {
        const double slope = (f_hi - f_lo) / (brent_hi - brent_lo);
        if (!(slope > 0.0)) {
            return std::unexpected(IVError{.code = IVErrorCode::MultipleRoots,
                .iterations = result->iterations, .final_error = 1.0, .last_vol = brent_lo});
        }
    }
    return IVSuccess{.implied_vol = result->root, .iterations = result->iterations,
                     .final_error = result->final_error, .vega = std::nullopt,
                     .used_rate_approximation = false};
}
```

(Copy the error-code switch from the solver verbatim; `find_root` is whatever the solver calls today at `:757` — keep the same function.)

Then rewrite `InterpolatedIVSolver<Surface>::solve` (`:645-800`) as:

```cpp
    auto error = validate_query(query);
    if (error.has_value()) return std::unexpected(validation_error_to_iv_error(*error));
    const double moneyness = query.spot / query.strike;
    const bool rate_is_curve = is_yield_curve(query.rate);
    const double rate_value = get_zero_rate(query.rate, query.maturity);
    const SurfaceInversionPolicy policy{
        .config_sigma_min = config_.sigma_min, .config_sigma_max = config_.sigma_max,
        .published_sigma_min = sigma_range_.first, .published_sigma_max = sigma_range_.second,
        .vega_threshold = config_.vega_threshold, .detect_multiple_roots = config_.detect_multiple_roots,
        .tolerance = config_.tolerance, .max_iter = config_.max_iter};
    const auto bracket = effective_sigma_bracket(query.spot, query.strike, query.option_type,
                                                 query.market_price, policy);
    if (!is_in_bounds(query, bracket.first) || !is_in_bounds(query, bracket.second)) {
        return std::unexpected(IVError{.code = IVErrorCode::InvalidGridConfig, .iterations = 0,
                                       .final_error = 0.0, .last_vol = std::nullopt});
    }
    // Price reconstructs spot as (spot/strike)*strike exactly as before; vega uses the
    // original spot exactly as before (bit-for-bit equivalence, spec L4).
    const auto price = [&](double s) { return eval_price(moneyness, query.maturity, s, rate_value, query.strike); };
    const auto vega = [&](double s) { return surface_.vega(query.spot, query.strike, query.maturity, s, rate_value); };
    auto r = invert_price_on_surface(price, vega, query.market_price, bracket, query.spot, policy);
    if (!r) return std::unexpected(r.error());
    r->used_rate_approximation = rate_is_curve;
    return *r;
```

Delete `adaptive_bounds` (`:616-643`) or make it call `effective_sigma_bracket`; keep it if other code calls it (`grep -n adaptive_bounds src tests`).

- [ ] **Step 4: Run the new test and every solver test**

Run: `TMPDIR="$D" bazel test //tests:surface_inversion_test //tests:interpolated_iv_solver_test //tests:iv_solver_factory_test //tests:price_table_factory_test --test_output=errors` (add any target whose deps include `interpolated_iv_solver`: `grep -n "interpolated_iv_solver" tests/BUILD.bazel`).
Expected: all PASS with no numeric change (existing fixtures pin the values).

- [ ] **Step 5: Commit**

```bash
git add src/option/surface_inversion.hpp src/option/surface_inversion.cpp src/option/BUILD.bazel src/option/interpolated_iv_solver.hpp src/option/interpolated_iv_solver.cpp tests/surface_inversion_test.cc tests/BUILD.bazel
git commit -m "Extract surface inversion from the IV solver"
```

---

### Task 3: Status and diagnostics types

**Files:**
- Modify: `src/option/table/adaptive_grid_types.hpp` (`:61-63` comment; `:91-101` `IterationStats`; `:104-134` `BuildDiagnostics`; add `PointStatus`/`PointScore` before `IterationStats`)
- Modify: `tests/adaptive_grid_types_test.cc:17-36` (`vega_floor` tests) and `tests/adaptive_refinement_unit_test.cc:27-31` (`BuildDiagnosticsTest.DefaultsAreEmpty`)

**Interfaces (produces):**

```cpp
enum class PointStatus : uint8_t {
    Measured, ReferenceUnresolved, SurfaceVegaTooSmall, SurfaceNoRoot,
    SurfaceAmbiguous, SurfaceNonConvergent, SurfaceNonFinite,
};
/// True for every Surface* status (an operational failure of the shipped inversion).
constexpr bool is_surface_failure(PointStatus s) noexcept {
    return s != PointStatus::Measured && s != PointStatus::ReferenceUnresolved;
}
struct PointScore {
    PointStatus status = PointStatus::ReferenceUnresolved;
    double iv_error = std::numeric_limits<double>::quiet_NaN();      // iff Measured
    double price_residual = std::numeric_limits<double>::quiet_NaN(); // |S - V̂|/K when finite
    bool edge_band_rescue = false;                                     // diagnostic only
};
// IterationStats += size_t unresolved = 0, surface_failures = 0, edge_band_rescues = 0;
// BuildDiagnostics += holdout_points_unresolved, holdout_points_unsupported, surface_failures,
//   edge_band_rescues (size_t); max_price_residual, reference_uncertainty_max (double);
//   reference_solves_fine, reference_solves_coarse (size_t).
```

- [ ] **Step 1: Write failing tests** — in `adaptive_refinement_unit_test.cc` extend `BuildDiagnosticsTest.DefaultsAreEmpty` with `EXPECT_EQ(d.surface_failures, 0u); EXPECT_EQ(d.holdout_points_unresolved, 0u); EXPECT_EQ(d.reference_solves_fine, 0u);` and add `TEST(PointStatusTest, SurfaceFailureClassification) { EXPECT_FALSE(is_surface_failure(PointStatus::Measured)); EXPECT_FALSE(is_surface_failure(PointStatus::ReferenceUnresolved)); EXPECT_TRUE(is_surface_failure(PointStatus::SurfaceNoRoot)); }`. In `adaptive_grid_types_test.cc` replace the `vega_floor` default/assignment expectations with one test asserting the field still exists and accepts `0.0` and `-1.0` (it is ignored).
- [ ] **Step 2: Run** `//tests:adaptive_refinement_unit_test //tests:adaptive_grid_types_test` — expect compile failure.
- [ ] **Step 3: Implement** the types; change the `vega_floor` comment to: `/// Deprecated and ignored since the round-trip metric (spec 2026-09-19 D6); kept for C ABI layout stability until #463 removes it.`
- [ ] **Step 4: Run** the two tests — PASS.
- [ ] **Step 5: Commit** — `git commit -m "Add point statuses and diagnostics fields"`.

---

### Task 4: Stencil references and resolution (D1–D2)

**Files:**
- Modify: `src/option/table/adaptive_refinement.hpp:259-263` (`ErrorRefs`), `:265-268` (`PrepareRefsFn` comment)
- Modify: `src/option/table/adaptive_metrics.hpp/.cpp` (add `make_stencil_refs_fn`; keep `make_fd_vega_refs_fn` compiling for now by filling only `ref_price` and `resolved=false` — it is deleted in Task 9)
- Modify: `tests/reference_oracle_test.cc` (add stencil tests with a fake oracle), `tests/adaptive_refinement_unit_test.cc` (Harness `prepare_fn` at `:396-410` fills the new fields)

**Interfaces:**
- `ErrorRefs` becomes exactly the spec's struct (D1): `ref_price, bracket_lo_price, bracket_hi_price, sigma_lo, sigma_hi, delta, delta_lo, delta_hi, resolved, fine_steps, coarse_steps`. `vega` is **removed** (fix every `ErrorRefs{.ref_price=..., .vega=...}` initialiser: `tests/adaptive_grid_builder_test.cc:1007,1020,1042`, Harness `:406`, `bspline_adaptive.cpp:795`).
- Produces:
  ```cpp
  /// Solves at one sigma on one grid; injectable for tests.
  using StencilSolveFn = std::function<std::expected<double, SolverError>(const PricingParams&, const PDEGridConfig&)>;
  PrepareRefsFn make_stencil_refs_fn(const AdaptiveGridParams& params, ReferenceOracle oracle,
                                     std::shared_ptr<ReferenceSolveCounter> counter,
                                     StencilSolveFn solve = {});   // default: oracle.solve
  bool stencil_resolved(const ErrorRefs& r) noexcept;              // the D2 inequalities alone
  ```

- [ ] **Step 1: Write the failing tests** (`tests/reference_oracle_test.cc`)

```cpp
// A fake oracle: price(sigma) = base + slope*(sigma-0.2) on the fine grid,
// plus `coarse_bias` on any grid whose point count is below the fine count.
struct FakeStencil {
    double slope = 40.0, coarse_bias = 1e-4;
    std::vector<std::pair<size_t, size_t>> calls;  // (n_points, n_time)
    StencilSolveFn fn() {
        return [this](const PricingParams& p, const PDEGridConfig& g) -> std::expected<double, SolverError> {
            calls.emplace_back(g.grid_spec.n_points(), g.n_time);
            const bool coarse = g.grid_spec.n_points() < calls.front().first;
            return 10.0 + slope * (p.volatility - 0.2) + (coarse ? coarse_bias : 0.0);
        };
    }
};

static ReferenceOracle plain_oracle() {
    return ReferenceOracle{.dividend_yield = 0.0, .option_type = OptionType::PUT,
                           .discrete_dividends = {}, .reference_maturity = std::nullopt,
                           .accuracy = make_grid_accuracy(kReferenceAccuracy)};
}

// Spec D1/L2: six solves, three identical fine configs, three identical coarse
// configs, coarse = every-other-node of fine.
TEST(StencilRefs, SixSolvesOnOneNestedGridPair) {
    FakeStencil fake;
    AdaptiveGridParams params; params.target_iv_error = 5e-4;
    auto counter = std::make_shared<ReferenceSolveCounter>();
    auto prep = make_stencil_refs_fn(params, plain_oracle(), counter, fake.fn());
    auto refs = prep(100.0, 100.0, 1.0, 0.2, 0.05);
    ASSERT_TRUE(refs.has_value());
    ASSERT_EQ(fake.calls.size(), 6u);
    EXPECT_EQ(fake.calls[0], fake.calls[2]); EXPECT_EQ(fake.calls[0], fake.calls[4]);
    EXPECT_EQ(fake.calls[1], fake.calls[3]); EXPECT_EQ(fake.calls[1], fake.calls[5]);
    EXPECT_EQ(fake.calls[1].first, (fake.calls[0].first - 1) / 2 + 1);
    EXPECT_EQ(counter->fine_attempts.load(), 3u);
    EXPECT_EQ(counter->coarse_attempts.load(), 3u);
    EXPECT_TRUE(refs->resolved);
    EXPECT_DOUBLE_EQ(refs->sigma_lo, 0.2 - 5e-4);
    EXPECT_DOUBLE_EQ(refs->sigma_hi, 0.2 + 5e-4);
    // delta = F_s * |bias| / (2^p - 1)
    EXPECT_NEAR(refs->delta, kRichardsonSafetyFactor * 1e-4 / (std::pow(2.0, kReferenceConvergenceOrder) - 1.0), 1e-15);
}

// Spec D2, reviewer example: y=10, lo=9.85, hi=10.15, delta=0.10 at every
// point -> intervals overlap -> unresolved.
TEST(StencilRefs, OverlappingEndpointIntervalsAreUnresolved) {
    ErrorRefs r{.ref_price = 10.0, .bracket_lo_price = 9.85, .bracket_hi_price = 10.15,
                .sigma_lo = 0.1, .sigma_hi = 0.3, .delta = 0.10, .delta_lo = 0.10, .delta_hi = 0.10};
    EXPECT_FALSE(stencil_resolved(r));
    r.delta = r.delta_lo = r.delta_hi = 0.07;   // 10-0.07 > 9.85+0.07 and 10.15-0.07 > 10+0.07
    EXPECT_TRUE(stencil_resolved(r));
    r.bracket_lo_price = 10.02;                  // reversed ordering
    EXPECT_FALSE(stencil_resolved(r));
}

// Flat reference (exercise region): lo == y == hi -> unresolved even with zero delta.
TEST(StencilRefs, FlatReferenceIsUnresolved) {
    FakeStencil fake; fake.slope = 0.0; fake.coarse_bias = 0.0;
    AdaptiveGridParams params; params.target_iv_error = 5e-4;
    auto prep = make_stencil_refs_fn(params, plain_oracle(), std::make_shared<ReferenceSolveCounter>(), fake.fn());
    auto refs = prep(100.0, 100.0, 1.0, 0.2, 0.05);
    ASSERT_TRUE(refs.has_value());
    EXPECT_FALSE(refs->resolved);
    EXPECT_DOUBLE_EQ(refs->ref_price, 10.0);   // base price still present (partial stencil contract)
}

// sigma0 - tau <= 0 -> unresolved with the base price present, one fine solve only.
TEST(StencilRefs, SigmaBelowToleranceIsUnresolvedWithBase) {
    FakeStencil fake;
    AdaptiveGridParams params; params.target_iv_error = 0.5;
    auto counter = std::make_shared<ReferenceSolveCounter>();
    auto prep = make_stencil_refs_fn(params, plain_oracle(), counter, fake.fn());
    auto refs = prep(100.0, 100.0, 1.0, 0.2, 0.05);
    ASSERT_TRUE(refs.has_value());
    EXPECT_FALSE(refs->resolved);
    EXPECT_TRUE(std::isnan(refs->bracket_lo_price));
    EXPECT_EQ(counter->fine_attempts.load(), 1u);
}

// A failed bracket solve -> unresolved (base present); a failed base solve -> unexpected.
TEST(StencilRefs, PartialAndBaseFailures) {
    AdaptiveGridParams params; params.target_iv_error = 5e-4;
    size_t n = 0;
    StencilSolveFn fail_third = [&](const PricingParams&, const PDEGridConfig&) -> std::expected<double, SolverError> {
        if (++n == 3) return std::unexpected(SolverError{});
        return 10.0;
    };
    auto counter = std::make_shared<ReferenceSolveCounter>();
    auto prep = make_stencil_refs_fn(params, plain_oracle(), counter, fail_third);
    auto refs = prep(100.0, 100.0, 1.0, 0.2, 0.05);
    ASSERT_TRUE(refs.has_value()); EXPECT_FALSE(refs->resolved);
    EXPECT_EQ(counter->fine_failures.load() + counter->coarse_failures.load(), 1u);
    StencilSolveFn fail_first = [](const PricingParams&, const PDEGridConfig&) -> std::expected<double, SolverError> {
        return std::unexpected(SolverError{}); };
    auto prep2 = make_stencil_refs_fn(params, plain_oracle(), std::make_shared<ReferenceSolveCounter>(), fail_first);
    EXPECT_FALSE(prep2(100.0, 100.0, 1.0, 0.2, 0.05).has_value());
}

// Spec D2: every target y, y±delta must pass validate_iv_query, including the
// upper no-arbitrage bound.  A call priced above spot at y+delta is unresolved.
TEST(StencilRefs, TargetsAboveUpperBoundAreUnresolved) {
    AdaptiveGridParams params; params.target_iv_error = 5e-4;
    size_t n = 0;
    // Fine solves return 99.99 for a call on S=100 (valid), coarse solves 99.5 -> delta = 3*0.49/(2^p-1) pushes y+delta above 100.
    StencilSolveFn near_cap = [&](const PricingParams& p, const PDEGridConfig& g) -> std::expected<double, SolverError> {
        (void)p; return (++n % 2 == 1) ? 99.99 : 99.5; };
    auto oracle = plain_oracle(); oracle.option_type = OptionType::CALL;
    auto prep = make_stencil_refs_fn(params, oracle, std::make_shared<ReferenceSolveCounter>(), near_cap);
    auto refs = prep(100.0, 90.0, 1.0, 0.2, 0.05);
    ASSERT_TRUE(refs.has_value());
    EXPECT_FALSE(refs->resolved);
}
```

Note the fake's call order assumption: implement the stencil as `[y, y½, lo, lo½, hi, hi½]` so index parity distinguishes fine/coarse.

- [ ] **Step 2: Run** `//tests:reference_oracle_test` — expect compile failure on `make_stencil_refs_fn`.

- [ ] **Step 3: Implement** in `adaptive_metrics.cpp`:

```cpp
bool stencil_resolved(const ErrorRefs& r) noexcept {
    const double v[] = {r.ref_price, r.bracket_lo_price, r.bracket_hi_price,
                        r.delta, r.delta_lo, r.delta_hi};
    for (double x : v) if (!std::isfinite(x)) return false;
    return (r.ref_price - r.delta > r.bracket_lo_price + r.delta_lo) &&
           (r.bracket_hi_price - r.delta_hi > r.ref_price + r.delta);
}

namespace {
double richardson_estimate(double fine, double coarse) {
    return kRichardsonSafetyFactor * std::abs(fine - coarse)
         / (std::pow(2.0, kReferenceConvergenceOrder) - 1.0);
}
bool target_is_valid_query(const PricingParams& p, double target) {
    IVQuery q;
    static_cast<OptionSpec&>(q) = static_cast<const OptionSpec&>(p);   // spot, strike, maturity, rate, yield, type
    q.market_price = target;
    q.discrete_dividends = p.discrete_dividends;
    return validate_iv_query(q).has_value();
}
}  // namespace

PrepareRefsFn make_stencil_refs_fn(const AdaptiveGridParams& params, ReferenceOracle oracle,
                                   std::shared_ptr<ReferenceSolveCounter> counter,
                                   StencilSolveFn solve) {
    if (!solve) solve = [oracle](const PricingParams& p, const PDEGridConfig& g) { return oracle.solve(p, g); };
    const double tau_iv = params.target_iv_error;
    return [=](double spot, double strike, double tau, double sigma, double rate)
        -> std::expected<ErrorRefs, SolverError> {
        const double nan = std::numeric_limits<double>::quiet_NaN();
        ErrorRefs out{.ref_price = nan, .bracket_lo_price = nan, .bracket_hi_price = nan,
                      .sigma_lo = sigma - tau_iv, .sigma_hi = sigma + tau_iv,
                      .delta = nan, .delta_lo = nan, .delta_hi = nan, .resolved = false,
                      .fine_steps = 0, .coarse_steps = 0};
        // One family per preparation, chosen at the widest stencil member (spec D1/L2).
        const PricingParams widest = oracle.contract(spot, strike, tau, out.sigma_hi, rate);
        auto fam = make_reference_grid_family(widest, oracle.accuracy, 1);
        if (!fam) return std::unexpected(SolverError{.code = SolverErrorCode::InvalidConfiguration});
        const PDEGridConfig& fine = fam->levels[0];
        const PDEGridConfig& coarse = fam->levels[1];
        out.fine_steps = static_cast<uint32_t>(fam->time_steps[0]);
        out.coarse_steps = static_cast<uint32_t>(fam->time_steps[1]);

        auto run = [&](double s, const PDEGridConfig& g, bool is_fine) -> std::optional<double> {
            (is_fine ? counter->fine_attempts : counter->coarse_attempts).fetch_add(1);
            auto r = solve(oracle.contract(spot, strike, tau, s, rate), g);
            if (!r || !std::isfinite(*r)) {
                (is_fine ? counter->fine_failures : counter->coarse_failures).fetch_add(1);
                return std::nullopt;
            }
            return *r;
        };
        auto y = run(sigma, fine, true);
        if (!y) return std::unexpected(SolverError{});          // invalid point (base failed)
        out.ref_price = *y;
        if (!(out.sigma_lo > 0.0) || !std::isfinite(out.sigma_hi)) return out;  // unresolved, base present
        auto y2 = run(sigma, coarse, false);
        auto lo = run(out.sigma_lo, fine, true);  auto lo2 = lo ? run(out.sigma_lo, coarse, false) : std::nullopt;
        auto hi = run(out.sigma_hi, fine, true);  auto hi2 = hi ? run(out.sigma_hi, coarse, false) : std::nullopt;
        if (!y2 || !lo || !lo2 || !hi || !hi2) return out;
        out.bracket_lo_price = *lo; out.bracket_hi_price = *hi;
        out.delta = richardson_estimate(*y, *y2);
        out.delta_lo = richardson_estimate(*lo, *lo2);
        out.delta_hi = richardson_estimate(*hi, *hi2);
        if (!stencil_resolved(out)) return out;
        const PricingParams base = oracle.contract(spot, strike, tau, sigma, rate);
        for (double target : {out.ref_price - out.delta, out.ref_price, out.ref_price + out.delta}) {
            if (!target_is_valid_query(base, target)) return out;   // reference limitation
        }
        out.resolved = true;
        return out;
    };
}
```

(If the six-call ordering in the fake differs from what you implement, keep the implementation order `[y, y½, lo, lo½, hi, hi½]` and adjust nothing else.) Update the Harness `prepare_fn` to return `ErrorRefs{.ref_price = base, .bracket_lo_price = base - 0.01, .bracket_hi_price = base + 0.01, .sigma_lo = sigma - params.target_iv_error, .sigma_hi = sigma + params.target_iv_error, .delta = 0.0, .delta_lo = 0.0, .delta_hi = 0.0, .resolved = true}`; fix the three `ErrorRefs{...}` initialisers in `adaptive_grid_builder_test.cc` the same way.

- [ ] **Step 4: Run** `//tests:reference_oracle_test //tests:adaptive_refinement_unit_test //tests:adaptive_grid_builder_test` — PASS (the two builder tests still use the old score fn; that is fine until Task 6).
- [ ] **Step 5: Commit** — `git commit -m "Prepare six-solve reference stencils with resolution"`.

---

### Task 5: Round-trip scorer (D3)

**Files:**
- Modify: `src/option/table/adaptive_refinement.hpp:33-37` (`SurfaceHandle` gains `vega`), `:270-283` (`ScoreErrorFn` → returns `PointScore`, takes `const SurfaceHandle&`)
- Modify: `src/option/table/adaptive_metrics.hpp/.cpp` (`make_round_trip_score_fn`), `src/option/table/BUILD.bazel` (`adaptive_metrics` deps += `//src/option:surface_inversion`)
- Modify: `tests/reference_oracle_test.cc` (scorer tests)

Compilation note: changing `ScoreErrorFn` breaks the loop and builders until Task 6/7. Do Tasks 5–7 as one compile unit if you prefer (commit per task once green), or temporarily keep the old alias as `LegacyScoreErrorFn` and switch users in Task 6. The plan assumes the latter: rename the old alias in this task and update the four call sites' type names only.

**Interfaces (produces):**

```cpp
ScoreErrorFn make_round_trip_score_fn(const AdaptiveGridParams& params,
                                      const RefinementContext& ctx,   // sample_bounds, bounds copied
                                      OptionType option_type);
```

- [ ] **Step 1: Write the failing tests**

```cpp
static RefinementContext score_ctx() {
    return RefinementContext{.spot = 100.0, .dividend_yield = 0.0, .option_type = OptionType::PUT,
        .bounds = {.m_min = -0.5, .m_max = 0.5, .tau_min = 0.05, .tau_max = 2.0, .sigma_min = 0.05, .sigma_max = 0.6, .rate_min = 0.0, .rate_max = 0.1},
        .sample_bounds = {.m_min = -0.3, .m_max = 0.3, .tau_min = 0.1, .tau_max = 1.0, .sigma_min = 0.1, .sigma_max = 0.5, .rate_min = 0.01, .rate_max = 0.09}};
}
static ErrorRefs resolved_refs(double y) {
    return ErrorRefs{.ref_price = y, .bracket_lo_price = y - 0.02, .bracket_hi_price = y + 0.02,
                     .sigma_lo = 0.2995, .sigma_hi = 0.3005, .delta = 1e-4, .delta_lo = 1e-4, .delta_hi = 1e-4, .resolved = true};
}
// Surface: price = 10 + 40*(sigma-0.3) + bias; vega = 40.
static SurfaceHandle linear_handle(double bias) {
    return SurfaceHandle{
        .price = [bias](double, double, double, double s, double) { return 10.0 + 40.0 * (s - 0.3) + bias; },
        .vega = [](double, double, double, double, double) { return 40.0; }};
}

TEST(RoundTripScore, MeasuresBiasAsSigmaDistanceOverThreeTargets) {
    AdaptiveGridParams params; params.target_iv_error = 5e-4;
    auto score = make_round_trip_score_fn(params, score_ctx(), OptionType::PUT);
    // bias 0.04 -> root at 0.3 - 0.001; targets y±1e-4 add ±2.5e-6.
    auto s = score(linear_handle(0.04), resolved_refs(10.0), 100.0, 100.0, 0.5, 0.3, 0.05);
    EXPECT_EQ(s.status, PointStatus::Measured);
    EXPECT_NEAR(s.iv_error, 0.001 + 2.5e-6, 1e-7);
    EXPECT_NEAR(s.price_residual, 0.04 / 100.0, 1e-12);
    EXPECT_FALSE(s.edge_band_rescue);
}

TEST(RoundTripScore, UnresolvedReferenceStillRecordsResidual) {
    AdaptiveGridParams params; params.target_iv_error = 5e-4;
    auto score = make_round_trip_score_fn(params, score_ctx(), OptionType::PUT);
    auto refs = resolved_refs(10.0); refs.resolved = false;
    auto s = score(linear_handle(0.04), refs, 100.0, 100.0, 0.5, 0.3, 0.05);
    EXPECT_EQ(s.status, PointStatus::ReferenceUnresolved);
    EXPECT_NEAR(s.price_residual, 4e-4, 1e-12);
    EXPECT_TRUE(std::isnan(s.iv_error));
}

// Root beyond the published edge: NoRoot under the exact product bracket;
// the tau_iv edge band rescues it only as a diagnostic flag.
TEST(RoundTripScore, EdgeMissIsNoRootWithRescueFlag) {
    AdaptiveGridParams params; params.target_iv_error = 5e-3;   // band 50 bps
    auto score = make_round_trip_score_fn(params, score_ctx(), OptionType::PUT);
    // Surface overprices by 0.04 at sigma=0.1 -> root at 0.099 (1e-3 below the edge, inside the band)
    auto refs = resolved_refs(10.0 + 40.0 * (0.1 - 0.3));
    auto s = score(linear_handle(0.04), refs, 100.0, 100.0, 0.5, 0.1, 0.05);
    EXPECT_EQ(s.status, PointStatus::SurfaceNoRoot);
    EXPECT_TRUE(s.edge_band_rescue);
    params.target_iv_error = 5e-4;   // band 5 bps: root is 10 bps outside -> no rescue
    auto score2 = make_round_trip_score_fn(params, score_ctx(), OptionType::PUT);
    auto s2 = score2(linear_handle(0.04), refs, 100.0, 100.0, 0.5, 0.1, 0.05);
    EXPECT_EQ(s2.status, PointStatus::SurfaceNoRoot);
    EXPECT_FALSE(s2.edge_band_rescue);
}

TEST(RoundTripScore, MapsEveryFailureKind) {
    AdaptiveGridParams params; params.target_iv_error = 5e-4;
    auto score = make_round_trip_score_fn(params, score_ctx(), OptionType::PUT);
    const auto refs = resolved_refs(10.0);
    SurfaceHandle flat{.price = [](double, double, double, double, double) { return 10.0; },
                       .vega = [](double, double, double, double, double) { return 0.0; }};
    EXPECT_EQ(score(flat, refs, 100, 100, 0.5, 0.3, 0.05).status, PointStatus::SurfaceVegaTooSmall);
    SurfaceHandle nan_mid{.price = [](double, double, double, double s, double) { return s > 0.35 ? std::nan("") : 10.0 + 40.0 * (s - 0.3); },
                          .vega = [](double, double, double, double, double) { return 40.0; }};
    EXPECT_EQ(score(nan_mid, refs, 100, 100, 0.5, 0.3, 0.05).status, PointStatus::SurfaceNonFinite);
    // Decreasing crossing: MultipleRoots via the post-Brent slope check.
    SurfaceHandle falling{.price = [](double, double, double, double s, double) { return 10.0 - 40.0 * (s - 0.3); },
                          .vega = [](double, double, double, double, double) { return 40.0; }};
    auto f = score(falling, refs, 100, 100, 0.5, 0.3, 0.05);
    EXPECT_TRUE(f.status == PointStatus::SurfaceAmbiguous || f.status == PointStatus::SurfaceNoRoot);
    // y+delta straddles the top of the surface's range -> the worst of the three targets wins.
    SurfaceHandle capped{.price = [](double, double, double, double s, double) { return std::min(10.0 + 40.0 * (s - 0.3), 10.00005); },
                         .vega = [](double, double, double, double, double) { return 40.0; }};
    EXPECT_EQ(score(capped, refs, 100, 100, 0.5, 0.3, 0.05).status, PointStatus::SurfaceNoRoot);
}
```

- [ ] **Step 2: Run** `//tests:reference_oracle_test` — compile failure.
- [ ] **Step 3: Implement**

```cpp
namespace {
PointStatus status_of(const IVError& e) {
    switch (e.code) {
        case IVErrorCode::VegaTooSmall: return PointStatus::SurfaceVegaTooSmall;
        case IVErrorCode::BracketingFailed: return PointStatus::SurfaceNoRoot;
        case IVErrorCode::MultipleRoots: return PointStatus::SurfaceAmbiguous;
        case IVErrorCode::MaxIterationsExceeded: return PointStatus::SurfaceNonConvergent;
        default: return PointStatus::SurfaceNonFinite;   // NumericalInstability, invariant violations
    }
}
int severity(PointStatus s) {  // higher = more severe (spec D3 ordering)
    switch (s) {
        case PointStatus::SurfaceNonFinite: return 5; case PointStatus::SurfaceNonConvergent: return 4;
        case PointStatus::SurfaceAmbiguous: return 3; case PointStatus::SurfaceNoRoot: return 2;
        case PointStatus::SurfaceVegaTooSmall: return 1; default: return 0;
    }
}
}  // namespace

ScoreErrorFn make_round_trip_score_fn(const AdaptiveGridParams& params,
                                      const RefinementContext& ctx, OptionType type) {
    const double tau_iv = params.target_iv_error;
    const SurfaceBounds sample = ctx.sample_bounds, fit = ctx.bounds;
    return [=](const SurfaceHandle& h, const ErrorRefs& refs, double spot, double strike,
               double tau, double sigma, double rate) -> PointScore {
        PointScore out;
        const double s0 = h.price(spot, strike, tau, sigma, rate);
        if (std::isfinite(s0) && std::isfinite(refs.ref_price)) out.price_residual = std::abs(s0 - refs.ref_price) / strike;
        if (!refs.resolved) { out.status = PointStatus::ReferenceUnresolved; return out; }
        const auto price = [&](double s) { return h.price(spot, strike, tau, s, rate); };
        const auto vega  = [&](double s) { return h.vega(spot, strike, tau, s, rate); };
        auto run = [&](double target, double pub_lo, double pub_hi) -> std::expected<double, PointStatus> {
            SurfaceInversionPolicy pol; pol.published_sigma_min = pub_lo; pol.published_sigma_max = pub_hi;
            auto br = effective_sigma_bracket(spot, strike, type, target, pol);
            auto r = invert_price_on_surface(price, vega, target, br, spot, pol);
            if (!r) return std::unexpected(status_of(r.error()));
            return r->implied_vol;
        };
        double worst = 0.0; PointStatus status = PointStatus::Measured;
        for (double target : {refs.ref_price - refs.delta, refs.ref_price, refs.ref_price + refs.delta}) {
            auto r = run(target, sample.sigma_min, sample.sigma_max);
            if (!r) { if (severity(r.error()) > severity(status)) status = r.error(); continue; }
            worst = std::max(worst, std::abs(*r - sigma));
        }
        out.status = status;
        if (status == PointStatus::Measured) { out.iv_error = worst; return out; }
        if (status == PointStatus::SurfaceNoRoot) {
            // Diagnostic only (spec D3): would the tau_iv edge band have found it?
            const double lo = std::max(sample.sigma_min - tau_iv, fit.sigma_min);
            const double hi = std::min(sample.sigma_max + tau_iv, fit.sigma_max);
            bool all_ok = true;
            for (double target : {refs.ref_price - refs.delta, refs.ref_price, refs.ref_price + refs.delta}) {
                if (!run(target, lo, hi)) { all_ok = false; break; }
            }
            out.edge_band_rescue = all_ok;
        }
        return out;
    };
}
```

- [ ] **Step 4: Run** `//tests:reference_oracle_test` — PASS.
- [ ] **Step 5: Commit** — `git commit -m "Add round-trip scorer with statuses"`.

---

### Task 6: Loop consumption (D4, D5, D7 probes)

**Files:**
- Modify: `src/option/table/adaptive_refinement.hpp` (`ErrorBins` `:124-200`, `FinalScore` `:440-460`, remove `kViabilityBound` `:62-65`, `scan_monotonicity` `:490-503`)
- Modify: `src/option/table/adaptive_refinement.cpp` (`SampleEval` `:27-39`, `Candidate` `:42-52`, `evaluate_fresh_samples` `:387-478`, `evaluate_holdout` `:480-492`, `pick_refinement_axis` `:498-517`, `scan_monotonicity` `:542-590`, `prepare_final_validation` `:590-626`, `score_final_surface` `:628-676`, `needs_final_retry`/`select_final_surface` `:677-696`, validation `:759`, holdout prep `:800-830`, loop `:873-1060`, diag fill `:1101-1124`)
- Modify: `src/support/ivcalc_trace.h` (after `:45`: `DTRACE_PROBE7/8` fallbacks; after `:226`: two macros)
- Modify: `tests/adaptive_refinement_unit_test.cc` (Harness `score_fn` `:412-419`, `build_fn` handle gains `.vega`, `ParamValidation` `:578-581` removed lines, new tests), `tests/adaptive_grid_builder_test.cc:1000-1165` (`filtering_score`, `score_of`, `FilteredPoints…` tests)

**Interfaces:**
- `ErrorBins` += `std::array<std::array<size_t, N_BINS>, N_DIMS> failure_counts = {}; void record_failure(const std::array<double, N_DIMS>& normalized_pos);` `problematic_bins` and `pick_refinement_axis` sum both arrays.
- `SampleEval`/`FinalScore` += `size_t unresolved = 0, unsupported = 0, surface_failures = 0, edge_band_rescues = 0; double max_price_residual = 0.0; double max_delta = 0.0; ErrorBins failure_bins;` (`FinalScore.filtered` renamed `unresolved`).
- `FinalScore::viable()`: `all_finite && measured > 0 && surface_failures == 0`.
- `Candidate` += `size_t holdout_failures = 0;` and a comparator:
  ```cpp
  static bool better_candidate(const Candidate& a, const Candidate& b);  // fewer holdout_failures, then lower max, then lower avg, then earlier iteration
  ```
- `select_final_surface`: after the existing viability filter, `retry` wins iff `retry->surface_failures < original.surface_failures || (equal && retry->max_error < original.max_error)`.
- `scan_monotonicity(points, handle, ctx, target_iv_error, diag)` (no `vega_floor`); tolerance per point = `max(1e-8*spot, finite_max(delta, delta_lo, delta_hi))`, skip the point when none is finite.
- Probes:
  ```c
  #define MANGO_TRACE_ADAPTIVE_VALIDATION_REFUSED(set, requested, prepared, resolved, unsupported) \
      DTRACE_PROBE5(MANGO_PROVIDER, adaptive_validation_refused, set, requested, prepared, resolved, unsupported)
  #define MANGO_TRACE_ADAPTIVE_NO_VIABLE_SURFACE(stage, candidates, no_root, ambiguous, nonconv, nonfinite, vega, rescues) \
      DTRACE_PROBE8(MANGO_PROVIDER, adaptive_no_viable_surface, stage, candidates, no_root, ambiguous, nonconv, nonfinite, vega, rescues)
  ```
  with `set`: 1 = holdout, 2 = final; `stage`: 1 = loop, 2 = final, 3 = retry.

- [ ] **Step 1: Write the failing Harness tests** (append to `adaptive_refinement_unit_test.cc`; the Harness's default `score_fn` becomes `PointScore{.status = Measured, .iv_error = |interp - ref|, .price_residual = |interp - ref| / strike}` and its handle gets `.vega = [](...) { return 1.0; }`):

```cpp
// Spec D4: a resolved surface failure on the fixed holdout makes the candidate
// non-viable; a later candidate with zero failures but a worse max is picked.
TEST(RunRefinementTest, HoldoutFailureRejectsCandidateAndFailuresRankFirst) {
    Harness h;
    h.script = [](const GridSizes&, size_t call) {
        SurfaceScript s; s.holdout_err = call == 0 ? 1e-5 : 5e-4; return s; };
    size_t scored = 0;
    h.score_override = [&](const mango::SurfaceHandle& hd, const mango::ErrorRefs& refs, double spot,
                           double strike, double tau, double sigma, double rate) -> mango::PointScore {
        const double interp = hd.price(spot, strike, tau, sigma, rate);
        mango::PointScore p{.status = mango::PointStatus::Measured, .iv_error = std::abs(interp - refs.ref_price),
                            .price_residual = std::abs(interp - refs.ref_price) / strike};
        // First build only: one holdout point fails.
        if (scored++ == 0) p.status = mango::PointStatus::SurfaceNoRoot;
        return p;
    };
    auto r = h.run();
    ASSERT_TRUE(r.has_value());
    EXPECT_NE(r->diagnostics.picked_iteration, 0u);
    EXPECT_EQ(r->diagnostics.surface_failures, 0u);
}

// A fresh-sample failure vetoes viability even when the holdout is clean.
TEST(RunRefinementTest, FreshFailureVetoesViability) {
    Harness h; h.params.max_iter = 1;
    h.score_override = [&](const mango::SurfaceHandle& hd, const mango::ErrorRefs& refs, double spot,
                           double strike, double tau, double sigma, double rate) -> mango::PointScore {
        const bool holdout = h.holdout_keys.count({strike, tau, sigma, rate}) > 0;
        return mango::PointScore{.status = holdout ? mango::PointStatus::Measured : mango::PointStatus::SurfaceAmbiguous,
                                 .iv_error = holdout ? 0.0 : std::nan(""), .price_residual = 0.0};
    };
    auto r = h.run();
    ASSERT_FALSE(r.has_value());
    EXPECT_EQ(r.error().code, mango::PriceTableErrorCode::NoViableSurface);
}

// Unresolved references enter no statistic; fewer than max(4, N/4) resolved -> refusal.
TEST(RunRefinementTest, UnresolvedHoldoutBelowCoverageRefuses) {
    Harness h;  // validation_samples = 8 -> need 4 resolved
    size_t n = 0;
    auto base_prep = h.prepare_fn();
    h.prepare_override = [&, base_prep](double spot, double strike, double tau, double sigma, double rate) {
        auto r = base_prep(spot, strike, tau, sigma, rate);
        if (r && (n++ % 8) < 5) r->resolved = false;   // 5 of every 8 unresolved
        return r;
    };
    auto r = h.run();
    ASSERT_FALSE(r.has_value());
    EXPECT_EQ(r.error().code, mango::PriceTableErrorCode::ValidationFailed);
}

// Failures are attributed to bins unconditionally and steer the axis walk.
TEST(ErrorBinsTest, FailureCountsDriveProblematicBins) {
    mango::ErrorBins b;
    b.record_failure({0.05, 0.5, 0.5, 0.5});
    b.record_failure({0.07, 0.5, 0.5, 0.5});
    EXPECT_EQ(b.problematic_bins(0), (std::vector<size_t>{0}));
    EXPECT_TRUE(b.problematic_bins(1).empty() || b.problematic_bins(1) == std::vector<size_t>{2});
}

// vega_floor is ignored: 0 and NaN no longer invalidate the run.
TEST(RunRefinementTest, VegaFloorIsIgnored) {
    Harness h; h.params.vega_floor = 0.0;
    EXPECT_TRUE(h.run().has_value());
    h.params.vega_floor = std::numeric_limits<double>::quiet_NaN();
    EXPECT_TRUE(h.run().has_value());
}
```

Add a `prepare_override` member to the Harness (a `std::function` consulted by `prepare_fn()` when set), analogous to `score_override`. In `adaptive_grid_builder_test.cc`, `filtering_score` returns `PointScore{.status = ReferenceUnresolved}` for filtered points and the tests assert `s.unresolved` instead of `s.filtered`; `score_of(x)` builds a `FinalScore` with `surface_failures = 0`; `SelectionPrefersViableOverLowerError` becomes: `garbage` has `surface_failures = 1` (non-viable) instead of `max 5.0`, and add a case where a finite error above 0.20 with zero failures is viable (best effort).

- [ ] **Step 2: Run** `//tests:adaptive_refinement_unit_test //tests:adaptive_grid_builder_test` — compile failures.

- [ ] **Step 3: Implement** the loop changes. Key edits:

`evaluate_fresh_samples` (order: support → surface price → veto → prepare → score):

```cpp
    for (const auto& sample : samples) {
        ...
        if (ctx.maturity_is_supported && !ctx.maturity_is_supported(tau)) { ++ev.unsupported; continue; }
        const double strike = ctx.spot * std::exp(-m);
        const double interp_price = handle.price(ctx.spot, strike, tau, sigma, rate);
        if (!std::isfinite(interp_price)) { ev.all_finite = false; continue; }   // veto first (spec D4)
        auto refs_result = prepare_refs(ctx.spot, strike, tau, sigma, rate);
        if (!refs_result.has_value()) continue;                                   // invalid reference
        ev.pde_solves_validation++;
        const auto ps = score(handle, refs_result.value(), ctx.spot, strike, tau, sigma, rate);
        if (std::isfinite(ps.price_residual)) ev.max_price_residual = std::max(ev.max_price_residual, ps.price_residual);
        ev.max_delta = std::max(ev.max_delta, finite_or_zero(refs_result->delta));
        const auto norm_pos = normalized_position(sample, ctx.sample_bounds);
        switch (ps.status) {
            case PointStatus::ReferenceUnresolved: ++ev.unresolved; continue;
            case PointStatus::Measured: break;
            default: ++ev.surface_failures; if (ps.edge_band_rescue) ++ev.edge_band_rescues;
                     ev.error_bins.record_failure(norm_pos); continue;
        }
        if (!std::isfinite(ps.iv_error) || ps.iv_error < 0.0) { ev.all_finite = false; continue; }
        ev.max_error = std::max(ev.max_error, ps.iv_error); sum_error += ps.iv_error; ev.measured++;
        ev.error_bins.record_error(norm_pos, ps.iv_error, target_iv_error);
    }
```

`score_final_surface` mirrors this (no support check — the points were filtered at preparation — and it fills `failure_bins` from `pt.coords` normalised over `ctx.sample_bounds`). `evaluate_holdout` copies `unresolved`, `surface_failures`, `edge_band_rescues`, `max_price_residual`, `max_delta`, and `error_bins = scored.failure_bins`.

Candidate bookkeeping in the loop (`:932-990`):

```cpp
            cand.holdout_failures = hold.surface_failures;
            cand.bins = fresh.error_bins;               // measured-error bins + fresh failure bins
            cand.bins.merge_failures(hold.error_bins);  // add holdout failure counts (add this small method)
            cand.fresh_converged = fresh.measured > 0 && fresh.max_error <= params.target_iv_error && fresh.surface_failures == 0;
            cand.viable = hold.all_finite && fresh.all_finite && hold.measured > 0 &&
                          std::isfinite(hold.max_error) && fresh.surface_failures == 0 && hold.surface_failures == 0;
            // Walk bookkeeping: fewer holdout failures is a measured improvement.
            const bool improved = (have_base && cand.holdout_failures < base.holdout_failures) ||
                (cand.holdout_failures == (have_base ? base.holdout_failures : cand.holdout_failures) &&
                 std::isfinite(cand.holdout_max) && cand.holdout_max < prev_best_holdout * (1.0 - kMinRelImprovement));
```

Replace the base-advance condition and the retention pick with `better_candidate` (viable filter first for retention). Holdout preparation (`:800-830`) and `prepare_final_validation` (`:596-620`): accept `refs` whenever `has_value() && std::isfinite(refs->ref_price)`; count `resolved` and `unsupported`; refuse with `ValidationFailed` when `prepared < min_valid || resolved < min_valid`, firing `MANGO_TRACE_ADAPTIVE_VALIDATION_REFUSED(set, params.validation_samples, prepared, resolved, unsupported)` first. When retention finds no viable candidate (`picked == nullptr`, `:1062`) fire `MANGO_TRACE_ADAPTIVE_NO_VIABLE_SURFACE(1, candidates.size(), ...)` with per-status totals accumulated from `SampleEval` (add `std::array<size_t, 7> status_counts` to `SampleEval` and sum). Diagnostics fill (`:1101-1124`): `holdout_points_unresolved`, `holdout_points_unsupported`, `surface_failures = picked->holdout_failures`, `edge_band_rescues`, `max_price_residual`, `reference_uncertainty_max` from the picked candidate's holdout evaluation (store the `SampleEval` on the `Candidate`). Delete `kViabilityBound` and the `vega_floor` validation at `:759`. Add the two probe macros and `DTRACE_PROBE7/8` fallbacks to `ivcalc_trace.h`.

- [ ] **Step 4: Run** `//tests:adaptive_refinement_unit_test //tests:adaptive_grid_builder_test` — PASS. Then `bazel build //src/option/table/...` — builders still compile because they still construct `SurfaceHandle` without `vega` (an empty `std::function`; Task 7 fills it) and still use the legacy score alias only if you kept it; otherwise proceed directly to Task 7 before committing.
- [ ] **Step 5: Commit** — `git commit -m "Consume point statuses in the refinement loop"`.

---

### Task 7: Builders: handles with vega, stencil references, probe adapters, maturity support

**Files:**
- Modify: `src/option/table/bspline/bspline_adaptive.cpp:423-431` (ordinary handle), `:482-483` (prepare/score), `:749-770` (probe handle), `:771-798` (probe adapter), `:864-875` (final prepare/score/accounting), `:881-889` (final handle), `:940-968` (diagnostics fill, `scan_monotonicity` call)
- Modify: `src/option/table/chebyshev/chebyshev_adaptive.cpp:555-561` (continuous handle), `:618-655` (segmented sizing handle: add `.vega` via `(*leaves_shared)[seg_idx].vega(...) * K_ref` with identical routing), `:800-801`, `:1043-1048` (segmented sizing: **new probe adapter** and `maturity_is_supported`), `:1060-1075` (`RefinementContext` for the segmented loop), `:1100-1150` (final validation, accounting)
- Modify: `src/option/table/BUILD.bazel` (builder targets depend on `adaptive_metrics` already; add nothing unless the compiler asks)

**Interfaces (consumes):** `make_stencil_refs_fn`, `make_round_trip_score_fn`, `ReferenceOracle`, `ReferenceSolveCounter`, `compute_segment_boundaries` (`adaptive_refinement.hpp:309`).

- [ ] **Step 1: Write the failing tests** (in `tests/adaptive_surface_build_integration_test.cc`):

```cpp
// Regression: the segmented Chebyshev sizing loop scored a K_ref-scaled leaf
// against a reference solved on the user's contract, so with a non-ATM strike
// and a cash dividend the loop chased a dividend-scaling residual.
// Bug: no probe adapter on the Chebyshev sizing path (bspline had one).
TEST(SegmentedChebyshevAdaptive, SizingReferencesLiveOnProbeContract) {
    // Build with one off-ATM K_ref and a $1.50 dividend; the assembled surface must be
    // Measured (not a failure) at a resolved point priced on the user's contract.
    AdaptiveGridParams params{.target_iv_error = 1e-3, .max_iter = 2, .validation_samples = 16};
    SegmentedAdaptiveConfig cfg{.spot = 100.0, .option_type = OptionType::PUT, .dividend_yield = 0.0,
        .discrete_dividends = {{0.5, 1.5}}, .maturity = 1.0, .kref_config = {.K_refs = {95.0, 105.0}}};
    IVGrid domain{.moneyness = {std::log(100.0 / 108.0), std::log(100.0 / 92.0)}, .vol = {0.15, 0.35}, .rate = {0.02, 0.06}};
    auto result = build_adaptive_chebyshev_segmented(params, cfg, domain);
    ASSERT_TRUE(result.has_value()) << static_cast<int>(result.error().code);
    EXPECT_EQ(result->diagnostics.surface_failures, 0u);
    EXPECT_GT(result->diagnostics.holdout_points_measured, 0u);
    EXPECT_GT(result->diagnostics.reference_solves_fine, 0u);
}

// Spec D4: event-gap maturities are excluded before preparation, not scored as defects.
TEST(SegmentedChebyshevAdaptive, EventGapSamplesAreUnsupportedNotFailures) {
    // Same config as above with a maturity grid that puts holdout samples near tau = 0.5.
    ... build ...
    EXPECT_EQ(result->diagnostics.surface_failures, 0u);
    // At least the sample the LHS puts inside the +-5e-4 gap around tau=0.5 (seed chosen in the test) is counted:
    EXPECT_GE(result->diagnostics.holdout_points_unsupported, 0u);   // exact count pinned after first run
}
```

(For the second test, choose `params.lhs_seed` by trying a few seeds locally until one sample lands in the gap; pin that seed and the count with a comment.)

- [ ] **Step 2: Run** the integration test — compile/link failures expected.

- [ ] **Step 3: Implement.** Common pattern for every handle:

```cpp
return SurfaceHandle{
    .price = [shared](double s, double k, double t, double v, double r) { return shared->price(s, k, t, v, r); },
    .vega  = [shared](double s, double k, double t, double v, double r) { return shared->vega(s, k, t, v, r); },
    .pde_solves = ...};
```

Segmented Chebyshev sizing handle: factor the routing into a local lambda `route(tau) -> optional<pair<size_t seg, double local_tau>>` used by both `.price` and `.vega`; vega = `(*leaves_shared)[seg].vega(spot, strike, local_tau, sigma, rate) * K_ref`.

Reference wiring per path (replace `make_validate_fn` + `make_fd_vega_refs_fn` + `make_iv_score_fn`):

```cpp
auto counter = std::make_shared<ReferenceSolveCounter>();
ReferenceOracle oracle{.dividend_yield = ..., .option_type = ..., .discrete_dividends = ..., .reference_maturity = ...,
                       .accuracy = make_grid_accuracy(kReferenceAccuracy)};
auto prepare_refs_fn = make_stencil_refs_fn(params, oracle, counter);
auto score_fn = make_round_trip_score_fn(params, ctx, type);
```

Probe adapters (B-spline `:771-798` extended; Chebyshev segmented sizing **new**, wrapping `prepare_refs_fn` for `K_ref = config_.spot`):

```cpp
PrepareRefsFn prepare_refs_fn = [base, probe_ref](double spot, double strike, double tau, double sigma, double rate)
    -> std::expected<ErrorRefs, SolverError> {
    const double scale = (strike > 0.0) ? strike / probe_ref : 1.0;
    auto refs = base(spot / scale, probe_ref, tau, sigma, rate);
    if (!refs) return std::unexpected(refs.error());
    refs->ref_price *= scale; refs->bracket_lo_price *= scale; refs->bracket_hi_price *= scale;
    refs->delta *= scale; refs->delta_lo *= scale; refs->delta_hi *= scale;   // resolved unchanged (scale-invariant)
    return refs;
};
```

Maturity support for both segmented loops and both final validations:

```cpp
auto segs = compute_segment_boundaries(config_.discrete_dividends, config_.maturity, dom_tau_min, dom_tau_max);
ctx.maturity_is_supported = [b = segs.bounds, g = segs.is_gap](double tau) {
    for (size_t s = 0; s + 1 < b.size(); ++s)
        if (g[s] && tau > b[s] && tau < b[s + 1]) return false;
    return true;
};
```

(Chebyshev already holds `seg_bounds_`/`seg_is_gap_`; use them.) Accounting: replace every `ref_attempts * 3` and `pde_solves_validation * 3` with `counter->fine_attempts + counter->coarse_attempts` and fill `diagnostics.reference_solves_fine/coarse`. Final gates: `diagnostics.surface_failures = final_score.surface_failures; holdout_points_unresolved = final_score.unresolved; edge_band_rescues; max_price_residual; reference_uncertainty_max = final_score.max_delta;` `target_met = final_score.viable() && final_score.max_error <= params.target_iv_error`. `scan_monotonicity(validation->points, handle, ctx, params.target_iv_error, diagnostics)`.

- [ ] **Step 4: Run** `bazel build //...` then `TMPDIR="$D" bazel test //tests:adaptive_surface_build_integration_test //tests:adaptive_grid_builder_test //tests:iv_solver_factory_test --test_output=errors`. Existing pins that reference `kViabilityBound` (`integration_test:55,268,372,437,472,743`) change to `EXPECT_EQ(result->diagnostics.surface_failures, 0u)` plus the existing `holdout_points_measured > 0`; lines `765-780` use `make_stencil_refs_fn`/`make_round_trip_score_fn` and compare `final.measured` with `holdout_points_measured`.
- [ ] **Step 5: Commit** — `git commit -m "Wire builders to stencil references and round trip"`.

---

### Task 8: Delete the old metric and deprecate `vega_floor` everywhere

**Files:**
- Modify: `src/option/table/adaptive_metrics.hpp/.cpp` (delete `compute_iv_error`, `make_fd_vega_refs_fn`, `make_iv_score_fn`, `kTVKThreshold`)
- Modify: `benchmarks/interp_iv_safety.cc:985-1000` (`kVegaFloor`, `kTVKThreshold` and the text that explains them)
- Modify: `src/ffi/mango_c_api.h:92`, `crates/mango-option/src/interp.rs:67`, `crates/mango-option-sys/src/lib.rs:106`, `src/python/mango_bindings.cpp:932` — add the deprecation comment/doc on each
- Modify: `tests/adaptive_refinement_unit_test.cc:38-71` (delete `ScoreFnTest.*` and `PrepareRefsTest.PropagatesSolveFailure`, now covered by Task 4's tests)

- [ ] **Step 1:** `grep -rn "compute_iv_error\|make_fd_vega_refs_fn\|make_iv_score_fn\|kTVKThreshold\|kVegaFloor\|kViabilityBound" src tests benchmarks crates` and edit every hit.
- [ ] **Step 2: Run** `bazel build //... && bazel build //benchmarks/... && bazel build //src/python:mango_option` — expect success.
- [ ] **Step 3: Run** `TMPDIR="$D" bazel test //tests:adaptive_refinement_unit_test //tests:adaptive_grid_types_test --test_output=errors` — PASS.
- [ ] **Step 4: Commit** — `git commit -m "Remove vega-scaled metric and deprecate vega_floor"`.

---

### Task 9: Python diagnostics keys and binding tests

**Files:**
- Modify: `src/python/mango_bindings.cpp:279-294` (eight new keys), `:1058-1070` (docstring), `:932` (comment)
- Modify: `tests/test_bindings.py:207-240` (expected keys += the eight; add `test_vega_floor_is_ignored`)

- [ ] **Step 1: Write the failing test**

```python
def test_vega_floor_is_ignored_and_new_diagnostics_present():
    config = make_price_table_config()
    adaptive = mo.AdaptiveGridParams()
    adaptive.target_iv_error = 0.002
    adaptive.max_iter = 2
    adaptive.validation_samples = 16
    adaptive.vega_floor = 0.0          # deprecated, ignored (spec D6)
    config.adaptive = adaptive
    table = mo.make_price_table(config)
    diag = table.build_diagnostics
    for key in ("holdout_points_unresolved", "holdout_points_unsupported", "surface_failures",
                "edge_band_rescues", "max_price_residual", "reference_uncertainty_max",
                "reference_solves_fine", "reference_solves_coarse"):
        assert key in diag, key
    assert diag["surface_failures"] == 0
    assert diag["reference_solves_fine"] > 0
```

- [ ] **Step 2: Run** `TMPDIR="$D" bazel test //tests:python_bindings_test --test_output=errors` — FAIL on missing keys.
- [ ] **Step 3: Implement** the eight `result[...] = d....;` lines and the docstring list.
- [ ] **Step 4: Run** — PASS.
- [ ] **Step 5: Commit** — `git commit -m "Expose round-trip diagnostics to Python"`.

---

### Task 10: Measure and pin the #500 fixture; re-measure old pins

**Files:**
- Modify: `tests/adaptive_surface_build_slow_test.cc:25-73` (rewrite `WideBandDividendBracketRemainsViable` → `WideBandDividendBracketRoundTrip`), `:100-110`, `:180-200`, `:815-860` (uses of removed helpers / `kViabilityBound` / pinned `holdout_points_measured`)
- Modify: `tests/iv_solver_factory_slow_test.cc:178` (`holdout_points_measured == 64` pin) and any nightly pin that fails for a metric reason

- [ ] **Step 1: Measure first.** Rewrite the #500 test body to build the two-K_ref surface as today, prepare the stencil with `make_stencil_refs_fn(params, oracle, counter)` at the offending point, score it with `make_round_trip_score_fn(params, ctx, PUT)` where `ctx.sample_bounds` is the fixture's `IVGrid` domain and `ctx.bounds` the builder's fit domain, and print (via `RecordProperty`, not `printf`) `resolved`, `status`, `iv_error`, `edge_band_rescue`, `price_residual`. Run it once:
  `TMPDIR="$D" bazel test -c opt //tests:adaptive_surface_build_slow_test --test_filter=SegmentedFinalContract.WideBandDividendBracketRoundTrip --test_output=all --test_env=OMP_NUM_THREADS=8`
- [ ] **Step 2: Pin what was measured.** Assert `refs.resolved` (with the measured δ̂ values in a comment), the measured `status`, and if `Measured` an `iv_error` ceiling of 1.5× the measured value (state the measured number and the run's commit in the test). If the status is a failure, assert that status and record `edge_band_rescue`; do **not** widen anything. Add the `// Regression:` / `// Bug:` lines describing the 788 bps artifact.
- [ ] **Step 3:** Update the other slow-test sites: `:107` `EXPECT_LE(..., kViabilityBound)` → `EXPECT_EQ(result->diagnostics.surface_failures, 0u)`; `:108` and `iv_solver_factory_slow_test.cc:178` exact `holdout_points_measured` pins → re-run, and pin the new count with `holdout_points_measured + holdout_points_unresolved + holdout_points_unsupported == holdout_points` as the invariant plus the measured value; `:186-196` and `:838-856` switch to the new factories.
- [ ] **Step 4: Run** `TMPDIR="$D" bazel test -c opt //tests:adaptive_surface_build_slow_test //tests:iv_solver_factory_slow_test --test_output=errors --test_env=OMP_NUM_THREADS=8` — PASS.
- [ ] **Step 5: Commit** — `git commit -m "Pin measured round-trip outcome for #500 fixture"` with the measured numbers in the body.

---

### Task 11: Oracle calibration test (D8)

**Files:**
- Create: `tests/reference_oracle_calibration_test.cc`
- Modify: `tests/BUILD.bazel` (`cc_test`, `size = "enormous"`, `tags = ["slow"]`, deps `//src/option/table:adaptive_metrics`, `//src/option:american_option`, `//src/option:grid_spec_types`)
- Modify: `src/option/table/adaptive_metrics.hpp` (`kReferenceConvergenceOrder` value + provenance comment)

- [ ] **Step 1: Write the test**

```cpp
// SPDX-License-Identifier: MIT
// Calibrates kReferenceConvergenceOrder and checks the High profile (spec D8).
#include <gtest/gtest.h>
#include "mango/option/table/adaptive_metrics.hpp"
#include "mango/option/american_option.hpp"
#include <cmath>
using namespace mango;

struct CalPoint { const char* name; double S, K, tau, sigma, r; OptionType type; std::vector<Dividend> divs; double T; };
static const CalPoint kPoints[] = {
    {"atm-1y-3div", 100, 100, 1.0, 0.20, 0.05, OptionType::PUT, {{0.25,0.5},{0.5,0.5},{0.75,0.5}}, 1.0},
    {"500-trigger", 100, 113.71897954276989, 0.27191459370080351, 0.1396857726802572, 0.055127327935524113, OptionType::PUT, {{0.25,0.5},{0.5,0.5},{0.75,0.5}}, 1.0},
    {"otm-30d", 100, 92, 30.0/365, 0.25, 0.05, OptionType::PUT, {}, 30.0/365},
    {"deep-otm-7d", 100, 85, 7.0/365, 0.30, 0.05, OptionType::PUT, {}, 7.0/365},
    {"itm-2y", 100, 120, 2.0, 0.20, 0.05, OptionType::PUT, {}, 2.0},
    {"atm-6m-call", 100, 100, 0.5, 0.20, 0.05, OptionType::CALL, {}, 0.5},
};
enum class Triple { Insufficient, Oscillatory, Usable };
static Triple classify(double d_a, double d_b, double theta, double* p) {
    if (std::abs(d_a) <= theta || std::abs(d_b) <= theta) return Triple::Insufficient;
    if (d_a * d_b < 0) return Triple::Oscillatory;
    *p = std::log(d_a / d_b) / std::log(2.0);
    return Triple::Usable;
}

TEST(ReferenceOracleCalibration, OrderIsUsableStableAndAboveConstant) {
    const auto acc = make_grid_accuracy(kReferenceAccuracy);
    double min_usable = std::numeric_limits<double>::infinity();
    size_t points_with_two_usable = 0;
    for (const auto& pt : kPoints) {
        ReferenceOracle oracle{.dividend_yield = 0.02, .option_type = pt.type, .discrete_dividends = pt.divs,
                               .reference_maturity = pt.T, .accuracy = acc};
        for (double tau_iv : {5e-4, 1e-3}) {
            const auto widest = oracle.contract(pt.S, pt.K, pt.tau, pt.sigma + tau_iv, pt.r);
            auto fam = make_reference_grid_family(widest, acc, 3);
            ASSERT_TRUE(fam.has_value()) << pt.name;
            for (double s : {pt.sigma - tau_iv, pt.sigma, pt.sigma + tau_iv}) {
                double v[4];
                for (size_t k = 0; k < 4; ++k) {
                    auto r = oracle.solve(oracle.contract(pt.S, pt.K, pt.tau, s, pt.r), fam->levels[k]);
                    ASSERT_TRUE(r.has_value()) << pt.name; v[k] = *r;
                }
                const double theta = std::ldexp(1.0, -40) * pt.K;
                double p1 = 0, p2 = 0;
                auto c1 = classify(v[2] - v[1], v[1] - v[0], theta, &p1);   // (0,1,2)
                auto c2 = classify(v[3] - v[2], v[2] - v[1], theta, &p2);   // (1,2,3)
                RecordProperty(std::string(pt.name) + "-p1", p1); RecordProperty(std::string(pt.name) + "-p2", p2);
                EXPECT_NE(c1, Triple::Oscillatory) << pt.name; EXPECT_NE(c2, Triple::Oscillatory) << pt.name;
                if (c1 == Triple::Usable) { EXPECT_GT(p1, 0.0); min_usable = std::min(min_usable, p1); }
                if (c2 == Triple::Usable) { EXPECT_GT(p2, 0.0); min_usable = std::min(min_usable, p2); }
                if (c1 == Triple::Usable && c2 == Triple::Usable && s == pt.sigma && tau_iv == 5e-4) {
                    EXPECT_NEAR(p1, p2, 0.5) << pt.name << " order not stable";
                    ++points_with_two_usable;
                }
            }
        }
        // Profile check: High vs Ultra within the two-grid estimate.
        auto high = oracle.solve(oracle.contract(pt.S, pt.K, pt.tau, pt.sigma, pt.r), make_reference_grid_family(oracle.contract(pt.S, pt.K, pt.tau, pt.sigma, pt.r), acc, 1)->levels[0]);
        ReferenceOracle ultra = oracle; ultra.accuracy = make_grid_accuracy(GridAccuracyProfile::Ultra);
        auto u = ultra.solve(ultra.contract(pt.S, pt.K, pt.tau, pt.sigma, pt.r), make_reference_grid_family(ultra.contract(pt.S, pt.K, pt.tau, pt.sigma, pt.r), ultra.accuracy, 1)->levels[0]);
        auto half = oracle.solve(oracle.contract(pt.S, pt.K, pt.tau, pt.sigma, pt.r), make_reference_grid_family(oracle.contract(pt.S, pt.K, pt.tau, pt.sigma, pt.r), acc, 1)->levels[1]);
        const double delta = kRichardsonSafetyFactor * std::abs(*high - *half) / (std::pow(2.0, kReferenceConvergenceOrder) - 1.0);
        EXPECT_LE(std::abs(*high - *u), delta) << pt.name;
        // Domain sensitivity: widen x-domain by one sigma*sqrt(T); shift must be within delta.
        GridAccuracyParams wide = acc; wide.n_sigma += 1.0;
        ReferenceOracle wide_oracle = oracle; wide_oracle.accuracy = wide;
        auto w = wide_oracle.solve(wide_oracle.contract(pt.S, pt.K, pt.tau, pt.sigma, pt.r), make_reference_grid_family(wide_oracle.contract(pt.S, pt.K, pt.tau, pt.sigma, pt.r), wide, 1)->levels[0]);
        EXPECT_LE(std::abs(*w - *high), delta) << pt.name << " domain sensitivity";
    }
    EXPECT_GE(points_with_two_usable, 4u);
    EXPECT_LE(kReferenceConvergenceOrder, min_usable);
}
```

Also record (RecordProperty) the spatial-only / temporal-only refinement shifts (solve on `{levels[1].grid_spec, levels[0].n_time}` and `{levels[0].grid_spec, levels[1].n_time}`) and `complementarity_report()` fields from a direct `AmericanOptionSolver` for two points; these are reported, not asserted.

- [ ] **Step 2: Run** `TMPDIR="$D" bazel test -c opt //tests:reference_oracle_calibration_test --test_output=all --test_env=OMP_NUM_THREADS=8`. Read the recorded `p1/p2`. If any assertion fails, follow spec D8's failure policy (add levels 4–5 / revisit the domain rule); do **not** tune the constant to pass.
- [ ] **Step 3:** Set `kReferenceConvergenceOrder` to the largest value ≤ the measured minimum usable order rounded down to one decimal (e.g. `1.4` if the minimum was 1.47), with a comment giving the measured table and commit. Re-run Tasks 4, 10 and this test (δ̂ scales with `1/(2^p − 1)`).
- [ ] **Step 4: Commit** — `git commit -m "Calibrate reference convergence order"` with the measured table in the body.

---

### Task 12: Documentation

**Files:**
- Modify: `docs/MATHEMATICAL_FOUNDATIONS.md:850-868` (replace "Stage 4: Error Metric" body) and add a subsection "Adaptive validation metric (round trip)" after Stage 5 containing: the six-solve stencil and grid family, the admission inequalities, the three-target round trip and status list, the edge-band diagnostic, §0's empirical contract, the constant inventory table from spec §4.11, the calibration record (numbers from Task 11), and the citations (Higham §1.6; Roache 1994; Forsyth & Vetzal 2002 Tables 8.1/11.1; Liu et al. 2020 §2.3.1; Jäckel 2015 §7; Ackerer et al. 2020 eq. 10) copied from `docs/research/2026-09-19-iv-inversion-conditioning.md`.
- Modify: `docs/API_GUIDE.md:466-468` (viability text: replace "absolute, target-independent garbage-detection bound" with the failure rule), `:843` (replace the 2,000 bps sentence), and the diagnostics key list near `:1338`; add the `vega_floor` deprecation note where `AdaptiveGridParams` fields are listed.
- Modify: `docs/ARCHITECTURE.md:283-290` (viability bullet), add one paragraph naming `surface_inversion` as the shared component.
- Modify: `CONTEXT.md` (three entries in the `## Language` list, matching the existing `**Term**:` / definition / `_Avoid_:` format):
  - **Reference-resolved point**: A validation point whose numerical reference separates σ0 from σ0 ± target by more than its own estimated uncertainty at all three stencil points and whose three target prices are valid queries; decided before any candidate surface exists. _Avoid_: measurable point, TV/K-admitted point, vega-floor pass.
  - **Operational round trip**: The build-time IV-error measurement that inverts the reference price on the candidate surface with the shipped solver's own bracket, pre-check, screen and Brent, at three target prices. _Avoid_: vega-scaled error, linearised IV error, price/vega.
  - **Surface inversion failure**: An operational outcome of the shipped inversion on a resolved point (no root, ambiguous, non-convergent, non-finite, vega too small); not proof of mathematical non-existence. _Avoid_: infinite IV error, 2000 bps garbage, non-invertible point.
- [ ] **Step 1:** Make the edits; keep every sentence within law L5.
- [ ] **Step 2:** `grep -rn "2,000 bps\|2000 bps\|kViabilityBound\|vega floor\|TV/K" docs CONTEXT.md` — only historical mentions inside `docs/research/` and `docs/superpowers/` may remain.
- [ ] **Step 3: Commit** — `git commit -m "Document the round-trip validation metric"`.

---

### Task 13: Full verification and runtime measurement

- [ ] **Step 1:** `TMPDIR="$D" bazel test //... --test_output=errors` — expect 158/158 (156 + `reference_oracle_test` + `surface_inversion_test`; the calibration test is `slow` and runs only when tagged targets are requested — run it explicitly: `bazel test -c opt //tests:reference_oracle_calibration_test`).
- [ ] **Step 2:** `bazel build //benchmarks/... && bazel build //src/python:mango_option && bazel test //crates/mango-option-sys:layout_test` (use the actual Rust test target name from `crates/*/BUILD.bazel`).
- [ ] **Step 3:** Runtime: time `bazel test -c opt //tests:adaptive_surface_build_integration_test //tests:adaptive_grid_builder_test --test_env=OMP_NUM_THREADS=8` on `main` (`git stash` is forbidden here; use `git worktree list` to find the main checkout's path and run there) and on this branch; record both wall-clock numbers for the PR body (acceptance criterion 10).
- [ ] **Step 4:** `git diff --check`; `git log --oneline main..HEAD` reads as a coherent sequence.
- [ ] **Step 5:** No commit; report the numbers.

---

## Self-review (done while writing)

- Spec coverage: D1 → T1, T4; D2 → T4, T6; D3 → T2, T5; D4 → T6, T7; D5 → T6; D6 → T3, T8; D7 → T3, T6, T7, T9; D8 → T11; D9 → T1–T11 tests; D10 → T12; L1–L6 → enforced by T4 (surface-independent prep), T1 (one family), T6 (statuses never numbers), T2 (shared inversion), T12 (language), T7 (probe contract). Acceptance criteria 1–11 map to T8, T4, T4, T2, T6, T10, T13, T11, T9, T13, T7.
- Placeholders: the error-code switch in T2 says "verbatim port" with the source line range; the T7 gap-count test pins its number after a first run by instruction (measured, not invented). No "TBD".
- Type consistency: `ErrorRefs` fields, `PointScore`/`PointStatus`, `make_stencil_refs_fn(params, oracle, counter, solve)`, `make_round_trip_score_fn(params, ctx, type)`, `SurfaceHandle::vega`, `effective_sigma_bracket`, `invert_price_on_surface` are spelled identically in every task.
