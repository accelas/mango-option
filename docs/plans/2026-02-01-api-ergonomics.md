# API Ergonomics Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Fix 6 medium/low priority API ergonomics issues (#301–#306) to reduce surprise, eliminate silent misconfiguration, and unify naming.

**Architecture:** Three phases ordered by dependency. Phase 1 fixes self-contained issues with no cross-cutting impact. Phase 2 unifies config naming (touches many files but mechanically). Phase 3 tackles config struct proliferation, which depends on Phase 2's naming being settled first.

**Tech Stack:** C++23, Bazel, GoogleTest, pybind11

**Closes:** #301, #302, #303, #304, #305, #306

---

## Phase 1 — Self-Contained Fixes (no cross-cutting changes)

Issues #301, #303, #306 are independent. Each touches a small set of files with no naming or structural overlap.

---

### Task 1: Move `solve_american_option_auto` to `american_option.hpp` (#301)

The simplest convenience API lives in `american_option_batch.hpp` instead of `american_option.hpp`. Users reading the primary header never discover it. It also discards the estimated `TimeDomain` and re-estimates inside `solve()`.

**Files:**
- Modify: `src/option/american_option.hpp` (add function before closing `}`)
- Modify: `src/option/american_option_batch.hpp:260-301` (remove function)
- Test: `tests/american_option_test.cc`

**Step 1: Write a test that includes only `american_option.hpp` and calls `solve_american_option_auto`**

```cpp
// In tests/american_option_test.cc (or a new section)
TEST(AmericanOptionTest, SolveAutoFromPrimaryHeader) {
    // Verifies #301: function is accessible from american_option.hpp
    mango::PricingParams params{
        mango::OptionSpec{.spot = 100.0, .strike = 100.0, .maturity = 1.0,
                          .rate = 0.05, .dividend_yield = 0.02, .type = mango::OptionType::PUT},
        0.20};
    auto result = mango::solve_american_option_auto(params);
    ASSERT_TRUE(result.has_value());
    EXPECT_NEAR(result->value_at(100.0), 6.35, 0.5);  // Rough sanity check
}
```

**Step 2: Run test — expect compile error** (function not found via `american_option.hpp`)

```bash
bazel test //tests:american_option_test --test_output=all
```

**Step 3: Move the function**

Cut `solve_american_option_auto` (lines 260–301) from `american_option_batch.hpp` and paste it at the end of `american_option.hpp` (before closing `}`). The function already depends only on symbols in `american_option.hpp` and `pde_workspace.hpp` (both already included).

Also fix the wasted-work bug: pass the estimated grid into `create()` so the solver doesn't re-estimate:

```cpp
inline std::expected<AmericanOptionResult, SolverError> solve_american_option_auto(
    const PricingParams& params)
{
    auto [grid_spec, time_domain] = estimate_grid_for_option(params);

    size_t n = grid_spec.n_points();
    std::pmr::vector<double> buffer(PDEWorkspace::required_size(n),
                                     std::pmr::get_default_resource());

    auto workspace_result = PDEWorkspace::from_buffer(buffer, n);
    if (!workspace_result.has_value()) {
        return std::unexpected(SolverError{
            .code = SolverErrorCode::InvalidConfiguration,
            .iterations = 0
        });
    }

    // Pass grid to avoid double-estimation
    auto solver_result = AmericanOptionSolver::create(
        params, workspace_result.value(), ExplicitPDEGrid{.grid_spec = grid_spec, .n_time = time_domain.n_steps()});
    if (!solver_result) {
        return std::unexpected(SolverError{
            .code = SolverErrorCode::InvalidConfiguration,
            .iterations = 0
        });
    }
    return solver_result.value().solve();
}
```

**Step 4: Run tests**

```bash
bazel test //tests:american_option_test --test_output=errors
bazel test //tests/... --test_output=errors
```

**Step 5: Commit**

```bash
git add src/option/american_option.hpp src/option/american_option_batch.hpp tests/american_option_test.cc
git commit -m "Move solve_american_option_auto to american_option.hpp (#301)

Relocates convenience API from batch header to primary header so
users discover it naturally. Passes estimated grid into create()
to avoid double-estimation of grid parameters."
```

---

### Task 2: Default `PDEGridSpec` to `GridAccuracyParams` in `PriceTableBuilder` (#303)

`PriceTableBuilder::from_vectors` defaults `pde_grid` to `ExplicitPDEGrid{}` (101 points, fixed [-3,3] domain). This is almost never what users want. `GridAccuracyParams{}` auto-estimates based on option parameters.

**Files:**
- Modify: `src/option/table/price_table_builder.hpp:120,148,167,185` (change 4 default args)
- Modify: `src/option/table/price_table_config.hpp:17` (change default)
- Test: `tests/price_table_builder_test.cc` or existing integration tests

**Step 1: Write a test verifying the default uses auto-estimation**

```cpp
TEST(PriceTableBuilderTest, DefaultGridUsesAutoEstimation) {
    // #303: from_vectors should default to GridAccuracyParams, not ExplicitPDEGrid
    std::vector<double> m = {0.8, 0.9, 1.0, 1.1, 1.2};
    std::vector<double> tau = {0.25, 0.5, 1.0};
    std::vector<double> vol = {0.15, 0.20, 0.30};
    std::vector<double> rate = {0.02, 0.05};

    // Call without pde_grid argument — should use GridAccuracyParams
    auto result = mango::PriceTableBuilder<4>::from_vectors(
        m, tau, vol, rate, 100.0,
        mango::GridAccuracyParams{},  // This is what default should already be
        mango::OptionType::PUT, 0.0);
    ASSERT_TRUE(result.has_value());
}
```

Note: This test passes today since callers already pass `GridAccuracyParams{}` explicitly. The change ensures callers who omit the argument get `GridAccuracyParams{}` instead of `ExplicitPDEGrid{}`.

**Step 2: Change the 4 default arguments in `price_table_builder.hpp`**

Lines 120, 148, 167, 185: change `PDEGridSpec pde_grid = ExplicitPDEGrid{}` to `PDEGridSpec pde_grid = GridAccuracyParams{}`.

Also in `price_table_config.hpp:17`: change `PDEGridSpec pde_grid;` to `PDEGridSpec pde_grid = GridAccuracyParams{};` (variant default-constructs to first alternative which is `ExplicitPDEGrid`; we want `GridAccuracyParams`).

**Step 3: Run all tests**

```bash
bazel test //tests/... --test_output=errors
bazel build //benchmarks/...
```

**Step 4: Commit**

```bash
git add src/option/table/price_table_builder.hpp src/option/table/price_table_config.hpp
git commit -m "Default PDEGridSpec to GridAccuracyParams (#303)

ExplicitPDEGrid{} produces a fixed 101-point grid that ignores
option parameters. GridAccuracyParams{} auto-estimates based on
volatility and maturity, producing correct grids by default."
```

---

### Task 3: Validate workspace/grid at `create()` time (#306)

`AmericanOptionSolver::create()` only validates pricing params. Grid estimation and workspace size check happen at `solve()` time. A user can get a successful `create()` that always fails at `solve()`.

**Files:**
- Modify: `src/option/american_option.cpp:24-39` (factory method)
- Modify: `src/option/american_option.hpp` (update declaration if signature changes)
- Test: `tests/american_option_test.cc`

**Step 1: Write a test that create() rejects workspace/grid mismatch**

```cpp
TEST(AmericanOptionTest, CreateRejectsMismatchedWorkspace) {
    // #306: Mismatch should fail at create(), not at solve()
    mango::PricingParams params{
        mango::OptionSpec{.spot = 100.0, .strike = 100.0, .maturity = 1.0,
                          .rate = 0.05, .type = mango::OptionType::PUT},
        0.20};

    // Create a workspace that is deliberately too small (10 points)
    std::pmr::vector<double> buffer(mango::PDEWorkspace::required_size(10));
    auto ws = mango::PDEWorkspace::from_buffer(buffer, 10);
    ASSERT_TRUE(ws.has_value());

    // create() should fail because auto-estimated grid needs ~100+ points
    auto result = mango::AmericanOptionSolver::create(params, ws.value());
    EXPECT_FALSE(result.has_value());
}
```

**Step 2: Run test — expect failure** (create() currently succeeds, solve() would fail)

```bash
bazel test //tests:american_option_test --test_output=all --test_filter="*CreateRejectsMismatchedWorkspace*"
```

**Step 3: Move grid estimation and workspace validation into `create()`**

In `american_option.cpp`, `create()`:

```cpp
std::expected<AmericanOptionSolver, ValidationError>
AmericanOptionSolver::create(
    const PricingParams& params,
    PDEWorkspace workspace,
    std::optional<PDEGridSpec> grid,
    std::optional<std::span<const double>> snapshot_times) noexcept
{
    auto validation = validate_pricing_params(params);
    if (!validation) {
        return std::unexpected(validation.error());
    }

    // Resolve grid now (not at solve time) so we can validate workspace size
    auto [grid_spec, time_domain] = grid.has_value()
        ? std::visit(overloaded{
            [&](const GridAccuracyParams& acc) {
                return estimate_grid_for_option(params, acc);
            },
            [&](const ExplicitPDEGrid& eg) {
                auto td = eg.mandatory_times.empty()
                    ? TimeDomain::from_n_steps(0.0, params.maturity, eg.n_time)
                    : TimeDomain::with_mandatory_points(0.0, params.maturity,
                        params.maturity / static_cast<double>(eg.n_time),
                        eg.mandatory_times);
                return std::make_pair(eg.grid_spec, td);
            }
        }, *grid)
        : estimate_grid_for_option(params);

    // Validate workspace size matches grid — fail early, not at solve()
    if (workspace.size() != grid_spec.n_points()) {
        return std::unexpected(ValidationError{
            ValidationErrorCode::InvalidGridSize,
            static_cast<double>(grid_spec.n_points()),
            workspace.size()});
    }

    return AmericanOptionSolver(params, workspace,
        std::make_pair(grid_spec, time_domain), snapshot_times);
}
```

This means the constructor should accept a pre-resolved `std::pair<GridSpec<double>, TimeDomain>` instead of `std::optional<PDEGridSpec>`. Update the private constructor signature accordingly, and remove the grid resolution and workspace validation from `solve()`.

**Step 4: Run all tests**

```bash
bazel test //tests/... --test_output=errors
```

**Step 5: Commit**

```bash
git add src/option/american_option.cpp src/option/american_option.hpp tests/american_option_test.cc
git commit -m "Validate workspace/grid match at create() time (#306)

Grid estimation now happens in create() instead of solve(), so
workspace size mismatch is caught immediately. Users no longer
get a successful create() that always fails at solve()."
```

---

## Phase 2 — Unify Config Naming (#305)

Normalize field names across all config structs. Mechanical find-and-replace, but must touch many files.

---

### Task 4: Unify `max_iterations` naming and type (#305)

`RootFindingConfig` uses `max_iter` (`size_t`). `IVSolverInterpolatedConfig` uses `max_iterations` (`int`). Standardize on `max_iter` / `size_t` everywhere, since `RootFindingConfig` is the lower-level primitive.

**Files:**
- Modify: `src/option/iv_solver_interpolated.hpp:69` (rename `max_iterations` → `max_iter`, change `int` → `size_t`)
- Modify: `src/option/iv_solver_interpolated.hpp` (update usage site ~line 337)
- Modify: `src/python/mango_bindings.cpp` (update binding if exposed)
- Test: Existing tests — verify no breakage

**Step 1: Search for all uses of `max_iterations`**

```bash
grep -rn "max_iterations" src/ tests/ benchmarks/
```

**Step 2: Rename `max_iterations` to `max_iter` and change type to `size_t`**

In `IVSolverInterpolatedConfig`:
```cpp
size_t max_iter = 50;  ///< Maximum Newton iterations
```

Update the usage in `iv_solver_interpolated.hpp` (~line 337) to remove the cast:
```cpp
// Before: .max_iter = static_cast<size_t>(std::max(0, config_.max_iterations)),
// After:  .max_iter = config_.max_iter,
```

Update all callers in tests/benchmarks that set `.max_iterations = ...` to `.max_iter = ...`.

**Step 3: Run all tests and build benchmarks**

```bash
bazel test //tests/... --test_output=errors
bazel build //benchmarks/...
bazel build //src/python:mango_option
```

**Step 4: Commit**

```bash
git commit -m "Unify iteration limit naming to max_iter (#305)

Rename IVSolverInterpolatedConfig::max_iterations to max_iter
and change type from int to size_t, matching RootFindingConfig."
```

---

### Task 5: Unify `option_type` vs `type` field naming (#305 continued)

`OptionSpec` uses `.type` while `IVSolverConfig`, `PriceTableConfig`, and builder configs use `.option_type`. Standardize on `.option_type` everywhere (more explicit, less ambiguous).

**Files:**
- Modify: `src/option/option_spec.hpp` (`OptionSpec::type` → `OptionSpec::option_type`)
- Modify: All files using `.type` on `OptionSpec`, `PricingParams`, or `IVQuery` (tests, benchmarks, bindings)
- This is a large mechanical rename — use grep to find all sites

**Step 1: Find all `.type =` usages on OptionSpec-derived structs**

```bash
grep -rn "\.type\s*=" src/ tests/ benchmarks/ | grep -v "option_type" | grep -v "//.*type"
```

**Step 2: Rename `OptionSpec::type` to `OptionSpec::option_type`**

In `option_spec.hpp`:
```cpp
OptionType option_type = OptionType::PUT;  ///< Call or put
```

Then fix every `.type =` → `.option_type =` across all files. Also update any reads like `params.type` → `params.option_type`.

**Step 3: Run all tests, benchmarks, and Python bindings**

```bash
bazel test //tests/... --test_output=errors
bazel build //benchmarks/...
bazel build //src/python:mango_option
```

**Step 4: Commit**

```bash
git commit -m "Rename OptionSpec::type to option_type (#305)

Unifies naming: all config structs now use option_type consistently
instead of mixing type and option_type."
```

---

### Task 6: Unify `dividends` vs `discrete_dividends` naming (#305 continued)

Field is named `dividends` in builder configs but `discrete_dividends` in `IVSolverConfig`, `PriceTableConfig`, and `PriceTableMetadata`. Standardize on `discrete_dividends` (more explicit about what it holds).

**Files:**
- Modify: `src/option/table/segmented_multi_kref_builder.hpp:25` (`dividends` → `discrete_dividends`)
- Modify: `src/option/table/segmented_price_table_builder.hpp:23` (`dividends` → `discrete_dividends`)
- Modify: `src/option/table/segmented_price_surface.hpp` (if it has `dividends` field)
- Modify: all callers in `src/option/iv_solver_factory.cpp`, tests, etc.
- Test: Existing tests

**Step 1: Find all usages**

```bash
grep -rn "\.dividends\b" src/ tests/ benchmarks/ | grep -v "discrete_dividends" | grep -v "dividend_yield"
```

**Step 2: Rename all `dividends` → `discrete_dividends` in builder configs**

**Step 3: Run all tests**

```bash
bazel test //tests/... --test_output=errors
bazel build //benchmarks/...
bazel build //src/python:mango_option
```

**Step 4: Commit**

```bash
git commit -m "Unify dividend field naming to discrete_dividends (#305)

Renames dividends to discrete_dividends in builder configs,
matching IVSolverConfig and PriceTableConfig convention."
```

---

## Phase 3 — Structural Improvements (#302, #304)

These tasks change config struct layout. They depend on Phase 2's naming being settled.

---

### Task 7: Split `IVSolverConfig` maturity ambiguity with variant (#302)

`IVSolverConfig` has both `maturity` (scalar, segmented path) and `maturity_grid` (vector, standard path). Only one is used depending on `discrete_dividends`. Replace with a variant that makes the two paths explicit.

**Files:**
- Modify: `src/option/iv_solver_factory.hpp:26-38` (restructure config)
- Modify: `src/option/iv_solver_factory.cpp` (update factory logic)
- Modify: `src/python/mango_bindings.cpp` (if exposed)
- Modify: All callers (tests, benchmarks)
- Test: `tests/iv_solver_factory_test.cc`

**Step 1: Write test for clear error on misconfiguration**

```cpp
TEST(IVSolverFactoryTest, RejectsMismatchedDividendPath) {
    // #302: Providing maturity_grid with discrete_dividends should be an error
    // (maturity_grid is for the standard path; segmented path uses maturity)
    mango::IVSolverConfig config{
        .option_type = mango::OptionType::PUT,
        .spot = 100.0,
        .discrete_dividends = {mango::Dividend{0.5, 2.0}},
        .moneyness_grid = {0.8, 0.9, 1.0, 1.1, 1.2},
        .vol_grid = {0.15, 0.20, 0.30},
        .rate_grid = {0.02, 0.05},
    };
    // Must provide maturity (not maturity_grid) when using dividends
    // This test verifies the new config structure enforces this
}
```

**Step 2: Restructure IVSolverConfig**

Replace the two fields with a variant:

```cpp
/// Standard path config (no discrete dividends)
struct StandardIVPath {
    std::vector<double> maturity_grid;
};

/// Segmented path config (discrete dividends)
struct SegmentedIVPath {
    double maturity;
    std::vector<Dividend> discrete_dividends;
    MultiKRefConfig kref_config;
};

struct IVSolverConfig {
    OptionType option_type = OptionType::PUT;
    double spot = 100.0;
    double dividend_yield = 0.0;
    std::vector<double> moneyness_grid;
    std::vector<double> vol_grid;
    std::vector<double> rate_grid;
    IVSolverInterpolatedConfig solver_config;

    /// Path selection: standard (no dividends) or segmented (with dividends)
    std::variant<StandardIVPath, SegmentedIVPath> path;
};
```

**Step 3: Update factory implementation** to use `std::visit` on `config.path`.

**Step 4: Update all callers**

Standard path callers:
```cpp
IVSolverConfig config{
    .option_type = OptionType::PUT,
    .spot = 100.0,
    .moneyness_grid = {...},
    .vol_grid = {...},
    .rate_grid = {...},
    .path = StandardIVPath{.maturity_grid = {0.25, 0.5, 1.0}},
};
```

Segmented path callers:
```cpp
IVSolverConfig config{
    .option_type = OptionType::PUT,
    .spot = 100.0,
    .dividend_yield = 0.02,
    .moneyness_grid = {...},
    .vol_grid = {...},
    .rate_grid = {...},
    .path = SegmentedIVPath{
        .maturity = 1.0,
        .discrete_dividends = {{0.5, 2.0}},
        .kref_config = {.K_refs = {80.0, 100.0, 120.0}},
    },
};
```

**Step 5: Run all tests**

```bash
bazel test //tests/... --test_output=errors
bazel build //benchmarks/...
bazel build //src/python:mango_option
```

**Step 6: Commit**

```bash
git commit -m "Split IVSolverConfig into path variants (#302)

Replace ambiguous maturity/maturity_grid fields with a
std::variant<StandardIVPath, SegmentedIVPath> that makes the
two code paths explicit. Setting the wrong one is now a compile
error instead of a silent misconfiguration."
```

---

### Task 8: Extract shared `DividendSpec` to reduce config duplication (#304)

`option_type`, `dividend_yield`, and `discrete_dividends` appear in 6+ config structs. Extract a shared `DividendSpec` and embed it.

**Files:**
- Modify: `src/option/option_spec.hpp` (add `DividendSpec` struct)
- Modify: `src/option/iv_solver_factory.hpp` (embed `DividendSpec`)
- Modify: `src/option/table/price_table_config.hpp` (embed `DividendSpec`)
- Modify: `src/option/table/segmented_multi_kref_builder.hpp` (embed `DividendSpec`)
- Modify: `src/option/table/segmented_price_table_builder.hpp` (embed `DividendSpec`)
- Modify: `src/option/table/price_table_metadata.hpp` (embed `DividendSpec`)
- Modify: All callers
- Test: Existing tests

**Step 1: Define `DividendSpec`**

```cpp
/// Shared dividend specification embedded in config structs
struct DividendSpec {
    double dividend_yield = 0.0;
    std::vector<Dividend> discrete_dividends;
};
```

Note: `option_type` stays separate since some structs don't have it (e.g., `PriceTableMetadata`).

**Step 2: Embed in config structs**

Example for `IVSolverConfig` (post-Task 7):
```cpp
struct IVSolverConfig {
    OptionType option_type = OptionType::PUT;
    double spot = 100.0;
    DividendSpec dividends;              // replaces dividend_yield + discrete_dividends
    std::vector<double> moneyness_grid;
    // ...
};
```

For `SegmentedMultiKRefBuilder::Config`:
```cpp
struct Config {
    double spot;
    OptionType option_type;
    DividendSpec dividends;   // replaces dividend_yield + discrete_dividends
    // ...
};
```

**Step 3: Update all callers** — change `.dividend_yield = 0.02, .discrete_dividends = {...}` to `.dividends = {.dividend_yield = 0.02, .discrete_dividends = {...}}`.

**Step 4: Run all tests**

```bash
bazel test //tests/... --test_output=errors
bazel build //benchmarks/...
bazel build //src/python:mango_option
```

**Step 5: Commit**

```bash
git commit -m "Extract DividendSpec to reduce config duplication (#304)

Replaces repeated dividend_yield + discrete_dividends fields
across 6 config structs with a shared DividendSpec struct.
Values can no longer drift between configs."
```

---

## Summary

| Phase | Task | Issue | Description |
|-------|------|-------|-------------|
| 1 | 1 | #301 | Move `solve_american_option_auto` to primary header |
| 1 | 2 | #303 | Default `PDEGridSpec` to `GridAccuracyParams` |
| 1 | 3 | #306 | Validate workspace/grid at `create()` time |
| 2 | 4 | #305 | Unify `max_iter` naming and type |
| 2 | 5 | #305 | Unify `type` → `option_type` naming |
| 2 | 6 | #305 | Unify `dividends` → `discrete_dividends` naming |
| 3 | 7 | #302 | Split `IVSolverConfig` with path variant |
| 3 | 8 | #304 | Extract shared `DividendSpec` |
