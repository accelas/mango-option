# API Cleanup Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Unify grid specification, simplify configuration, and clean up the public API surface across the mango-option library.

**Architecture:** Seven sequential breaking changes, each compiling independently. Move shared types to a lightweight header, simplify configs via `std::variant`, privatize internals, and rename for clarity. Each task is one commit.

**Tech Stack:** C++23, Bazel, GoogleTest, pybind11

**Design doc:** `docs/plans/2026-02-01-api-cleanup-design.md`

---

### Task 1: Create `grid_spec_types.hpp` and Relocate Types

Move `GridAccuracyParams`, `GridAccuracyProfile`, `grid_accuracy_profile()` from `american_option.hpp` and `ExplicitPDEGrid`/`PDEGridSpec` from `price_table_config.hpp` into a new shared header. Change `ExplicitPDEGrid` default from uniform to sinh-spaced. Add `mandatory_times` field.

**Files:**
- Create: `src/option/grid_spec_types.hpp`
- Create: `src/option/grid_spec_types.cpp` (for `grid_accuracy_profile()` definition)
- Modify: `src/option/american_option.hpp:39-101` — remove `GridAccuracyParams`, `GridAccuracyProfile`, `grid_accuracy_profile()`; add `#include "mango/option/grid_spec_types.hpp"`
- Modify: `src/option/table/price_table_config.hpp:4-22` — remove `ExplicitPDEGrid` and `PDEGridSpec`; replace `#include "mango/option/american_option.hpp"` with `#include "mango/option/grid_spec_types.hpp"` and `#include "mango/option/option_spec.hpp"`
- Modify: `src/option/BUILD.bazel` — add `grid_spec_types` cc_library target; update deps for `american_option` and others
- Modify: `src/option/table/BUILD.bazel:58-66` — update `price_table_config` deps: replace `/src/option:american_option` with `/src/option:grid_spec_types` and `/src/option:option_spec`

**Step 1: Create the new header**

Write `src/option/grid_spec_types.hpp`:

```cpp
// SPDX-License-Identifier: MIT
#pragma once
#include "mango/pde/core/grid.hpp"
#include <variant>
#include <vector>

namespace mango {

enum class GridAccuracyProfile { Low, Medium, High, Ultra };

struct GridAccuracyParams {
    double n_sigma = 5.0;
    double alpha = 2.0;
    double tol = 1e-2;
    double c_t = 0.75;
    size_t min_spatial_points = 100;
    size_t max_spatial_points = 1200;
    size_t max_time_steps = 5000;
};

struct ExplicitPDEGrid {
    GridSpec<double> grid_spec = GridSpec<double>::sinh_spaced(-3.0, 3.0, 101, 2.0).value();
    size_t n_time = 1000;
    std::vector<double> mandatory_times = {};
};

using PDEGridSpec = std::variant<ExplicitPDEGrid, GridAccuracyParams>;

GridAccuracyParams grid_accuracy_profile(GridAccuracyProfile profile);

}  // namespace mango
```

Create `src/option/grid_spec_types.cpp` — move the `grid_accuracy_profile()` implementation from `american_option.hpp:72-101` (it's currently `inline` in the header — move to .cpp).

**Step 2: Update BUILD.bazel**

Add to `src/option/BUILD.bazel`:
```python
cc_library(
    name = "grid_spec_types",
    srcs = ["grid_spec_types.cpp"],
    hdrs = ["grid_spec_types.hpp"],
    deps = [
        "/src/pde/core:grid",
    ],
    visibility = ["/visibility:public"],
)
```

Update `american_option` target deps to add `:grid_spec_types`.

Update `src/option/table/BUILD.bazel` `price_table_config` target:
- Remove dep: `/src/option:american_option`
- Add deps: `/src/option:grid_spec_types`, `/src/option:option_spec`

**Step 3: Update `american_option.hpp`**

- Add `#include "mango/option/grid_spec_types.hpp"` to includes
- Remove lines 39-101 (`GridAccuracyParams` struct, `GridAccuracyProfile` enum, `grid_accuracy_profile()` function) — they now live in the new header
- Keep `estimate_grid_for_option()` and `compute_global_grid_for_batch()` here (they depend on `PricingParams` which is in `option_spec.hpp`)

**Step 4: Update `price_table_config.hpp`**

Replace:
```cpp
#include "mango/option/american_option.hpp"
```
With:
```cpp
#include "mango/option/grid_spec_types.hpp"
#include "mango/option/option_spec.hpp"
```

Remove the `ExplicitPDEGrid` struct (lines 15-19) and `PDEGridSpec` alias (line 22) — now in `grid_spec_types.hpp`.

**Step 5: Build and test**

Run: `bazel build //...`
Expected: Build succeeds. Some tests may need re-baselining if they relied on default uniform grid.

Run: `bazel test //...`
Expected: All tests pass. If any fail due to the sinh-spaced default change, update their expected values.

**Step 6: Commit**

```bash
git add src/option/grid_spec_types.hpp src/option/grid_spec_types.cpp \
        src/option/american_option.hpp src/option/table/price_table_config.hpp \
        src/option/BUILD.bazel src/option/table/BUILD.bazel
git commit -m "Extract grid_spec_types.hpp; default to sinh-spaced grid"
```

---

### Task 2: Simplify IVSolverFDMConfig

Replace 9 fields with 3 using the shared `PDEGridSpec` variant.

**Files:**
- Modify: `src/option/iv_solver_fdm.hpp:47-96` — rewrite `IVSolverFDMConfig`
- Modify: `src/option/iv_solver_fdm.cpp:86,143-152,198-200` — replace manual grid logic with `std::visit`; delete dead validators
- Modify: `src/python/mango_bindings.cpp:182-191` — update config binding fields
- Modify: `tests/iv_solver_test.cc:166-215` — update 4 manual grid tests to use `ExplicitPDEGrid`

**Step 1: Rewrite `IVSolverFDMConfig`**

In `src/option/iv_solver_fdm.hpp`, replace lines 47-96 with:

```cpp
struct IVSolverFDMConfig {
    /// Root-finding configuration (Brent's method)
    RootFindingConfig root_config;

    /// Minimum batch size for OpenMP parallelization
    size_t batch_parallel_threshold = 4;

    /// PDE grid specification: auto-estimate (default) or explicit
    PDEGridSpec grid = GridAccuracyParams{};
};
```

Add `#include "mango/option/grid_spec_types.hpp"` if not already included transitively.

**Step 2: Update `iv_solver_fdm.cpp`**

Replace the manual grid construction logic (lines 86, 143-152) with a `std::visit` dispatcher:

```cpp
auto [grid_spec, time_domain] = std::visit(overloaded{
    [&](const GridAccuracyParams& acc) {
        return estimate_grid_for_option(option_params, acc);
    },
    [&](const ExplicitPDEGrid& eg) {
        auto td = eg.mandatory_times.empty()
            ? TimeDomain::from_n_steps(0.0, query.maturity, eg.n_time)
            : TimeDomain::with_mandatory_points(0.0, query.maturity,
                query.maturity / static_cast<double>(eg.n_time), eg.mandatory_times);
        return std::make_pair(eg.grid_spec, td);
    }
}, config_.grid);
```

Delete the `validate_grid_params()` function or its `use_manual_grid` branch — it's dead code.

**Step 3: Update test manual grid tests**

In `tests/iv_solver_test.cc`, replace lines 166-215 (4 tests that set `use_manual_grid = true`). Example for the first:

```cpp
// Test 12: Zero spatial grid points validation
TEST_F(IVSolverTest, ZeroGridSpaceValidation) {
    auto bad_grid = GridSpec<double>::sinh_spaced(-3.0, 3.0, 0, 2.0);
    // GridSpec::sinh_spaced with 0 points should fail validation
    ASSERT_FALSE(bad_grid.has_value());
    // If GridSpec itself rejects 0, the test validates at GridSpec level.
    // Otherwise, set grid with n_space=3 (minimum) and verify solve works:
    config.grid = ExplicitPDEGrid{
        GridSpec<double>::sinh_spaced(-3.0, 3.0, 3, 2.0).value(), 100};
    IVSolverFDM solver(config);
    auto result = solver.solve_impl(query);
    // Minimal grid should still produce a result (possibly less accurate)
    ASSERT_TRUE(result.has_value());
}
```

Adapt each test to use `ExplicitPDEGrid` with appropriate invalid/edge-case values. The error code expectations (`InvalidGridConfig`) should be preserved — verify the validation path still triggers.

**Step 4: Update python bindings**

In `src/python/mango_bindings.cpp:182-191`, replace the config field bindings:

```cpp
py::class_<mango::IVSolverFDMConfig>(m, "IVSolverFDMConfig")
    .def(py::init<>())
    .def_readwrite("root_config", &mango::IVSolverFDMConfig::root_config)
    .def_readwrite("batch_parallel_threshold", &mango::IVSolverFDMConfig::batch_parallel_threshold);
    // Note: PDEGridSpec variant binding requires separate handling
```

**Step 5: Build and test**

Run: `bazel test //...`
Expected: All tests pass.

**Step 6: Commit**

```bash
git add src/option/iv_solver_fdm.hpp src/option/iv_solver_fdm.cpp \
        src/python/mango_bindings.cpp tests/iv_solver_test.cc
git commit -m "Simplify IVSolverFDMConfig: 9 fields to 3 via PDEGridSpec"
```

---

### Task 3: Unify Grid in AmericanOptionSolver and Batch Solver; Remove Dead Types

Replace `std::optional<std::pair<GridSpec<double>, TimeDomain>>` with `std::optional<PDEGridSpec>` in both single and batch solvers. Remove dead types.

**Files:**
- Modify: `src/option/american_option.hpp:32,263-267,280-283` — change `create()` signature, remove `AmericanOptionParams` alias
- Modify: `src/option/american_option.cpp:29,46` — update constructor and create() parameter
- Modify: `src/option/american_option_batch.hpp:201,209,230,237,257` — change grid parameter type
- Modify: `src/option/american_option_batch.cpp:73,185,294,326,368,447,454,512-541,575` — update all grid usage
- Modify: `src/option/iv_solver_fdm.cpp:198-200` — update solver construction
- Modify: `src/option/option_spec.hpp:201-208` — remove `OptionSolverGrid`
- Modify: `src/python/mango_bindings.cpp:273-285,300,348,447` — update `AmericanOptionParams` → `PricingParams`, update constructor call to `create()`
- Modify: 14 test files that reference `AmericanOptionParams` — mechanical rename to `PricingParams`

**Step 1: Remove dead types**

In `src/option/option_spec.hpp`, delete `OptionSolverGrid` (lines 201-208).

In `src/option/american_option.hpp`, delete line 32: `using AmericanOptionParams = PricingParams;`

**Step 2: Update `AmericanOptionSolver::create()` signature**

In `src/option/american_option.hpp:263-267`, change:

```cpp
static std::expected<AmericanOptionSolver, ValidationError>
create(const PricingParams& params,
       PDEWorkspace workspace,
       std::optional<PDEGridSpec> grid = std::nullopt,
       std::optional<std::span<const double>> snapshot_times = std::nullopt) noexcept;
```

Update constructor signature at lines 280-283 to match (same parameter change).

**Step 3: Update `create()` implementation**

In `src/option/american_option.cpp`, resolve the `PDEGridSpec` variant:

```cpp
auto [grid_spec, time_domain] = grid.has_value()
    ? std::visit(overloaded{
        [&](const GridAccuracyParams& acc) {
            return estimate_grid_for_option(params, acc);
        },
        [&](const ExplicitPDEGrid& eg) {
            auto td = eg.mandatory_times.empty()
                ? TimeDomain::from_n_steps(0.0, params.maturity, eg.n_time)
                : TimeDomain::with_mandatory_points(0.0, params.maturity,
                    params.maturity / static_cast<double>(eg.n_time), eg.mandatory_times);
            return std::make_pair(eg.grid_spec, td);
        }
    }, *grid)
    : estimate_grid_for_option(params);
```

**Step 4: Update batch solver**

Apply same `PDEGridSpec` change to `src/option/american_option_batch.hpp` and `.cpp`. The batch solver's `solve_batch()` method and internal helpers change from `std::optional<std::pair<GridSpec<double>, TimeDomain>>` to `std::optional<PDEGridSpec>`.

**Step 5: Rename `AmericanOptionParams` → `PricingParams` everywhere**

Mechanical find-and-replace in all 14 test files, `src/python/mango_bindings.cpp`, and internal `.cpp` files. The type is identical — just the alias name changes.

**Step 6: Build and test**

Run: `bazel test //...`
Expected: All tests pass.

**Step 7: Commit**

```bash
git add src/option/american_option.hpp src/option/american_option.cpp \
        src/option/american_option_batch.hpp src/option/american_option_batch.cpp \
        src/option/iv_solver_fdm.cpp src/option/option_spec.hpp \
        src/python/mango_bindings.cpp tests/
git commit -m "Unify grid spec in solvers; remove AmericanOptionParams alias and OptionSolverGrid"
```

---

### Task 4: Privatize AmericanOptionSolver Constructor

**Files:**
- Modify: `src/option/american_option.hpp:280-283` — move constructor to `private:`
- Modify: `src/option/iv_solver_fdm.cpp:198-200` — use `create()` with `std::expected` propagation
- Modify: `src/option/american_option_batch.cpp:575` — use `create()` with `std::expected` propagation
- Modify: `src/python/mango_bindings.cpp:348` — use `create()`
- Modify: test/benchmark files that call constructor directly — use `create(...).value()`

**Step 1: Move constructor to private**

In `src/option/american_option.hpp`, move the constructor declaration from `public:` to `private:`.

**Step 2: Update production callers**

In `src/option/iv_solver_fdm.cpp:198-200`, replace:
```cpp
AmericanOptionSolver solver(option_params, pde_workspace_result.value(),
                            std::nullopt, custom_grid_config);
```
With:
```cpp
auto solver_result = AmericanOptionSolver::create(option_params, pde_workspace_result.value(), grid);
if (!solver_result) {
    return std::unexpected(convert_to_iv_error(solver_result.error()));
}
auto& solver = solver_result.value();
```

In `src/option/american_option_batch.cpp:575`, apply same pattern with `SolverError` propagation.

**Step 3: Update test callers**

Find all test files constructing `AmericanOptionSolver` directly. Replace with `create(...).value()`. Common pattern:

```cpp
// Before:
AmericanOptionSolver solver(params, workspace);
// After:
auto solver = AmericanOptionSolver::create(params, workspace).value();
```

**Step 4: Build and test**

Run: `bazel test //...`
Expected: All tests pass.

**Step 5: Commit**

```bash
git commit -am "Privatize AmericanOptionSolver constructor; enforce create() factory"
```

---

### Task 5: Rename `solve_impl` → `solve`

**Files:**
- Modify: `src/option/iv_solver_fdm.hpp:186,196` — rename declarations
- Modify: `src/option/iv_solver_fdm.cpp` — rename definitions
- Modify: `src/option/iv_solver_interpolated.hpp:109,117` — rename declarations
- Modify: `src/option/iv_solver_interpolated.cpp` — rename instantiation (if method names are in template)
- Modify: `src/option/iv_solver_factory.cpp:21,27` — rename calls in `std::visit`
- Modify: `src/python/mango_bindings.cpp:259,1009` — rename in bindings
- Modify: 10 test files — mechanical rename `solve_impl` → `solve`, `solve_batch_impl` → `solve_batch`

**Step 1: Rename in headers**

In `src/option/iv_solver_fdm.hpp`:
- Line 186: `solve_impl` → `solve`
- Line 196: `solve_batch_impl` → `solve_batch`

In `src/option/iv_solver_interpolated.hpp`:
- Line 109: `solve_impl` → `solve`
- Line 117: `solve_batch_impl` → `solve_batch`
- Update template implementations at lines 269-387

**Step 2: Rename in implementations**

In `src/option/iv_solver_fdm.cpp`: rename method definitions.

In `src/option/iv_solver_factory.cpp:21,27`:
```cpp
return std::visit([&](const auto& solver) {
    return solver.solve(query);        // was: solve_impl
}, solver_);
```

**Step 3: Rename in python bindings**

In `src/python/mango_bindings.cpp:259`: `.def("solve", ...)` (was `"solve_impl"`)
In `src/python/mango_bindings.cpp:1009`: same change for interpolated solver

**Step 4: Rename in tests**

Mechanical `solve_impl` → `solve` and `solve_batch_impl` → `solve_batch` in 10 test files:
- `tests/iv_solver_test.cc`
- `tests/iv_solver_expected_test.cc`
- `tests/iv_solver_interpolated_test.cc`
- `tests/iv_solver_integration_test.cc`
- `tests/iv_solver_property_test.cc`
- `tests/production_config_integration_test.cc`
- `tests/quantlib_accuracy_batch_test.cc`
- `tests/quantlib_validation_framework.hpp`
- `tests/real_market_data_test.cc`
- `tests/test_bindings.py`

**Step 5: Build and test**

Run: `bazel test //...`
Expected: All tests pass.

**Step 6: Commit**

```bash
git commit -am "Rename solve_impl to solve on IV solvers"
```

---

### Task 6: Clean Up PriceTableBuilder Internal API

Remove 11 public forwarding methods (`_for_testing` and `_internal`). Replace with `friend` access for production consumers and a test access helper.

**Files:**
- Modify: `src/option/table/price_table_builder.hpp:199-294` — delete forwarding methods, add `friend` declarations
- Modify: `src/option/table/segmented_price_table_builder.cpp:326,371,379,389` — change `_for_testing` → direct private method calls
- Modify: `src/option/table/adaptive_grid_builder.cpp:169,291,301-303,311` — change `_internal` → direct private method calls
- Create: `tests/price_table_builder_test_access.hpp` — inline test access helper
- Modify: `tests/price_table_builder_test.cc:56,85,119-120,154-156,243,291,342,382` — use test access helper
- Modify: `tests/price_table_builder_custom_grid_test.cc:29,39,131`
- Modify: `tests/price_table_builder_custom_grid_advanced_test.cc:29,99,182`
- Modify: `tests/price_table_builder_custom_grid_diagnosis_test.cc:30`
- Modify: `tests/price_table_builder_root_cause_test.cc:44,67`

**Step 1: Modify `price_table_builder.hpp`**

Delete lines 199-294 (all `_for_testing` and `_internal` methods).

Add friend declarations after the private methods section:

```cpp
    friend class AdaptiveGridBuilder;
    friend class SegmentedPriceTableBuilder;
    template <size_t M> friend struct PriceTableBuilderAccess;
```

**Step 2: Create test access helper**

Write `tests/price_table_builder_test_access.hpp`:

```cpp
// SPDX-License-Identifier: MIT
#pragma once
#include "mango/option/table/price_table_builder.hpp"

namespace mango::testing {

template <size_t N>
struct PriceTableBuilderAccess {
    static inline auto make_batch(const PriceTableBuilder<N>& b, const PriceTableAxes<N>& a) {
        return b.make_batch(a);
    }
    static inline auto solve_batch(const PriceTableBuilder<N>& b,
                                    const std::vector<PricingParams>& batch,
                                    const PriceTableAxes<N>& a) {
        return b.solve_batch(batch, a);
    }
    static inline auto extract_tensor(const PriceTableBuilder<N>& b,
                                       const BatchAmericanOptionResult& batch,
                                       const PriceTableAxes<N>& a) {
        return b.extract_tensor(batch, a);
    }
    static inline auto fit_coeffs(const PriceTableBuilder<N>& b,
                                   const PriceTensor<N>& tensor,
                                   const PriceTableAxes<N>& a) {
        return b.fit_coeffs(tensor, a);
    }
    static inline auto find_nearest_valid_neighbor(const PriceTableBuilder<N>& b,
                                                    size_t s_idx, size_t r_idx,
                                                    size_t Ns, size_t Nr,
                                                    const std::vector<bool>& valid) {
        return b.find_nearest_valid_neighbor(s_idx, r_idx, Ns, Nr, valid);
    }
    static inline auto repair_failed_slices(const PriceTableBuilder<N>& b,
                                             PriceTensor<N>& tensor,
                                             const std::vector<size_t>& failed_pde,
                                             const std::vector<std::tuple<size_t, size_t, size_t>>& failed_spline,
                                             const PriceTableAxes<N>& a) {
        return b.repair_failed_slices(tensor, failed_pde, failed_spline, a);
    }
};

}  // namespace mango::testing
```

**Step 3: Update production callers**

In `src/option/table/segmented_price_table_builder.cpp`, replace:
- `builder.make_batch_for_testing(axes)` → `builder.make_batch(axes)`
- `builder.extract_tensor_for_testing(...)` → `builder.extract_tensor(...)`
- `builder.repair_failed_slices_for_testing(...)` → `builder.repair_failed_slices(...)`
- `builder.fit_coeffs_for_testing(...)` → `builder.fit_coeffs(...)`

In `src/option/table/adaptive_grid_builder.cpp`, replace:
- `builder.make_batch_internal(axes)` → `builder.make_batch(axes)`
- `builder.extract_tensor_internal(...)` → `builder.extract_tensor(...)`
- `builder.repair_failed_slices_internal(...)` → `builder.repair_failed_slices(...)`
- `builder.fit_coeffs_internal(...)` → `builder.fit_coeffs(...)`

**Step 4: Update test files**

Add `#include "tests/price_table_builder_test_access.hpp"` and alias:
```cpp
using Access = mango::testing::PriceTableBuilderAccess<4>;
```

Replace all `builder.make_batch_for_testing(axes)` with `Access::make_batch(builder, axes)`. Same pattern for all other methods.

**Step 5: Build and test**

Run: `bazel test //...`
Expected: All tests pass.

**Step 6: Commit**

```bash
git commit -am "Clean up PriceTableBuilder: remove public _for_testing/_internal methods"
```

---

### Task 7: Rename OptionChain → OptionGrid

**Files:**
- Rename: `src/option/option_chain.hpp` → `src/option/option_grid.hpp`
- Modify: renamed file — `struct OptionChain` → `struct OptionGrid`
- Modify: `src/option/BUILD.bazel:12-15` — rename target and header
- Modify: `src/option/table/price_table_builder.hpp:11` — update include; rename `from_chain` → `from_grid`, `from_chain_auto` → `from_grid_auto`, `from_chain_auto_profile` → `from_grid_auto_profile`
- Modify: `src/option/table/price_table_builder.cpp` — rename factory implementations
- Modify: `src/option/table/adaptive_grid_builder.hpp:8` — update include
- Modify: `src/python/mango_bindings.cpp:15,96-111,930` — update include, binding name, function names
- Modify: `tests/option_chain_test.cc` → rename to `tests/option_grid_test.cc`, update all `OptionChain` → `OptionGrid`
- Modify: `tests/price_table_builder_factories_test.cc` — update `OptionChain` → `OptionGrid`
- Modify: `tests/adaptive_grid_builder_test.cc`, `tests/adaptive_grid_builder_integration_test.cc` — update if they reference `OptionChain`
- Modify: `tests/BUILD.bazel` — update test target names and deps

**Step 1: Rename file and type**

```bash
git mv src/option/option_chain.hpp src/option/option_grid.hpp
```

In the renamed file, change `struct OptionChain` to `struct OptionGrid`.

**Step 2: Update BUILD target**

In `src/option/BUILD.bazel:12-15`, rename:
```python
cc_library(
    name = "option_grid",  # was: option_chain
    hdrs = ["option_grid.hpp"],  # was: option_chain.hpp
    visibility = ["/visibility:public"],
)
```

Update all deps referencing `:option_chain` → `:option_grid`.

**Step 3: Update includes and references**

In `src/option/table/price_table_builder.hpp:11`:
- `#include "mango/option/option_chain.hpp"` → `#include "mango/option/option_grid.hpp"`
- Rename factory methods: `from_chain` → `from_grid`, `from_chain_auto` → `from_grid_auto`, `from_chain_auto_profile` → `from_grid_auto_profile`
- Rename parameter types: `const OptionChain&` → `const OptionGrid&`

Same include/type updates in `adaptive_grid_builder.hpp`, `price_table_builder.cpp`, `mango_bindings.cpp`.

**Step 4: Update tests**

Rename test file: `git mv tests/option_chain_test.cc tests/option_grid_test.cc`

In all test files, replace `OptionChain` → `OptionGrid` and `option_chain` includes.

Update `tests/BUILD.bazel` target names.

**Step 5: Build and test**

Run: `bazel test //...`
Expected: All tests pass.

**Step 6: Commit**

```bash
git commit -am "Rename OptionChain to OptionGrid; rename factory methods"
```

---

## Pre-merge Checklist

After all 7 tasks:

- [ ] `bazel test //...` — all tests pass
- [ ] `bazel build //benchmarks/...` — benchmarks compile
- [ ] `bazel build //src/python:mango_option` — python bindings compile
- [ ] `tests/test_bindings.py` passes
- [ ] Update `docs/API_GUIDE.md` examples if they reference changed API
- [ ] Update `CLAUDE.md` code examples if they reference changed API
