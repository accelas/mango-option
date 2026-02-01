# API Cleanup Design

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Address architectural issues found in code review — unify grid specification, simplify configuration, clean up public API surface.

**Architecture:** Breaking changes across 7 sections, ordered by dependency so each step compiles independently. All changes are to public API types and method signatures.

**Tech Stack:** C++23, Bazel, GoogleTest

**Baseline note:** Section 1 changes the default PDE grid from uniform to sinh-spaced. This shifts numerical outputs slightly. Tests and benchmarks using default grids may need re-baselining.

---

## Section 1: Create `grid_spec_types.hpp` and Relocate Types

Move `GridAccuracyParams` and `GridAccuracyProfile` from `american_option.hpp`, and `ExplicitPDEGrid`/`PDEGridSpec` from `price_table_config.hpp`, into a new lightweight shared header.

**Files:**
- Create: `src/option/grid_spec_types.hpp`
- Modify: `src/option/american_option.hpp` — remove `GridAccuracyParams` and `GridAccuracyProfile` definitions, add `#include "src/option/grid_spec_types.hpp"`
- Modify: `src/option/table/price_table_config.hpp` — remove `ExplicitPDEGrid` and `PDEGridSpec`, add include, replace heavy `american_option.hpp` include with `option_spec.hpp`
- Modify: `BUILD.bazel` — add new header to appropriate `cc_library` target
- Modify: any file that includes `american_option.hpp` solely for `GridAccuracyParams` — switch to lighter include

**New header contents:**

```cpp
// SPDX-License-Identifier: MIT
#pragma once
#include "src/pde/core/grid.hpp"
#include "src/pde/core/time_domain.hpp"
#include <variant>

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
    std::vector<double> mandatory_times = {};  // e.g., dividend dates
};

using PDEGridSpec = std::variant<ExplicitPDEGrid, GridAccuracyParams>;

GridAccuracyParams grid_accuracy_profile(GridAccuracyProfile profile);

}  // namespace mango
```

**Key changes:**
- `ExplicitPDEGrid` default changes from `uniform(-3.0, 3.0, 101)` to `sinh_spaced(-3.0, 3.0, 101, 2.0)`. Matches documentation recommendations.
- `ExplicitPDEGrid` gains a `mandatory_times` field to support discrete dividend time-stepping (needed by Section 3 — without this, callers cannot specify mandatory time points for dividend dates).

**`price_table_config.hpp` after:**
```cpp
#include "src/option/grid_spec_types.hpp"  // PDEGridSpec, ExplicitPDEGrid, GridAccuracyParams
#include "src/option/option_spec.hpp"      // OptionType (was: american_option.hpp — too heavy)
```

This fixes the heavy transitive include: `price_table_config.hpp` no longer pulls in the PDE solver chain.

---

## Section 2: Unify Grid Specification in IVSolverFDMConfig

Replace 9 fields with 3 using the shared `PDEGridSpec` type.

**Files:**
- Modify: `src/option/iv_solver_fdm.hpp`
- Modify: `src/option/iv_solver_fdm.cpp` — resolve variant in solve path; delete dead manual-grid validation code
- Modify: `src/python/mango_bindings.cpp` — update binding for new config fields
- Modify: test files that set manual grid fields

**Before:**
```cpp
struct IVSolverFDMConfig {
    RootFindingConfig root_config;
    size_t batch_parallel_threshold = 4;
    bool use_manual_grid = false;
    size_t grid_n_space = 101;
    size_t grid_n_time = 1000;
    double grid_x_min = -3.0;
    double grid_x_max = 3.0;
    double grid_alpha = 2.0;
    GridAccuracyParams grid_accuracy;
};
```

**After:**
```cpp
struct IVSolverFDMConfig {
    RootFindingConfig root_config;
    size_t batch_parallel_threshold = 4;
    PDEGridSpec grid = GridAccuracyParams{};
};
```

**Migration for callers:**
- `config.use_manual_grid = true; config.grid_n_space = 51;` becomes `config.grid = ExplicitPDEGrid{GridSpec<double>::sinh_spaced(-3.0, 3.0, 51, 2.0).value(), 100};`
- Default (auto-estimation) is unchanged — `GridAccuracyParams{}` is the default

**Implementation:** `IVSolverFDM` uses `std::visit` on `config_.grid`:
- `GridAccuracyParams` → call `estimate_grid_for_option()`
- `ExplicitPDEGrid` → construct `GridSpec` and `TimeDomain` from explicit values, incorporating `mandatory_times`

**Cleanup:** Delete dead `use_manual_grid`-specific validation logic in `iv_solver_fdm.cpp`.

---

## Section 3: Unify Grid Specification in AmericanOptionSolver and Batch Solver

Replace `std::optional<std::pair<GridSpec<double>, TimeDomain>>` with `std::optional<PDEGridSpec>` in both the single and batch solvers.

**Files:**
- Modify: `src/option/american_option.hpp`
- Modify: `src/option/american_option.cpp`
- Modify: `src/option/american_option_batch.hpp` — same grid type change
- Modify: `src/option/american_option_batch.cpp`
- Modify: `src/option/iv_solver_fdm.cpp` — internal solver construction
- Modify: `src/python/mango_bindings.cpp` — update bindings
- Modify: test/benchmark files that pass custom grids

**Before:**
```cpp
static std::expected<AmericanOptionSolver, ValidationError>
create(const PricingParams& params, PDEWorkspace workspace,
       std::optional<std::span<const double>> snapshot_times = std::nullopt,
       std::optional<std::pair<GridSpec<double>, TimeDomain>> custom_grid_config = std::nullopt);
```

**After:**
```cpp
static std::expected<AmericanOptionSolver, ValidationError>
create(const PricingParams& params, PDEWorkspace workspace,
       std::optional<PDEGridSpec> grid = std::nullopt,
       std::optional<std::span<const double>> snapshot_times = std::nullopt);
```

Note: parameter order changes — `grid` before `snapshot_times` since it's used more often.

**Resolution logic in `create()`:**
- `std::nullopt` → call `estimate_grid_for_option(params)`
- `GridAccuracyParams` → call `estimate_grid_for_option(params, accuracy)`
- `ExplicitPDEGrid` → construct `GridSpec` and `TimeDomain` from explicit values; if `mandatory_times` is non-empty, use `TimeDomain::with_mandatory_points()`

**Dead code removal:**
- Remove `AmericanOptionParams` alias (`american_option.hpp`). Update all 10 files that reference it to use `PricingParams` directly.
- Remove `OptionSolverGrid` struct (`option_spec.hpp:201-208`). It is unused — only defined and referenced in an archived plan doc.

---

## Section 4: Privatize AmericanOptionSolver Constructor

**Files:**
- Modify: `src/option/american_option.hpp` — move constructor to `private:`
- Modify: `src/option/iv_solver_fdm.cpp` — production caller, must propagate `std::expected` (not `.value()`)
- Modify: `src/option/american_option_batch.cpp` — production caller, must propagate `std::expected`
- Modify: `src/python/mango_bindings.cpp` — switch to `create()`
- Modify: test/benchmark files that call constructor directly — switch to `create(...).value()`

**Before:**
```cpp
class AmericanOptionSolver {
public:
    static std::expected<AmericanOptionSolver, ValidationError> create(...);
    AmericanOptionSolver(...);  // deprecated but public
```

**After:**
```cpp
class AmericanOptionSolver {
public:
    static std::expected<AmericanOptionSolver, ValidationError> create(...);

private:
    AmericanOptionSolver(...);
```

**Migration:**
- **Production code** (`iv_solver_fdm.cpp`, `american_option_batch.cpp`): propagate `std::expected`, do not call `.value()`. Use `auto solver = AmericanOptionSolver::create(...); if (!solver) return std::unexpected(convert_error(solver.error()));`
- **Tests/benchmarks**: `auto solver = AmericanOptionSolver::create(params, workspace).value();` is acceptable since test failure on bad params is desired.

---

## Section 5: Rename `solve_impl` → `solve`

Drop the `_impl` suffix from public methods on both IV solvers.

**Files:**
- Modify: `src/option/iv_solver_fdm.hpp` — rename methods + update doc comments
- Modify: `src/option/iv_solver_fdm.cpp`
- Modify: `src/option/iv_solver_interpolated.hpp` — rename methods + update doc comments
- Modify: `src/option/iv_solver_interpolated.cpp`
- Modify: `src/option/iv_solver_factory.cpp` — calls `solve_impl` internally
- Modify: `src/python/mango_bindings.cpp` — update bindings
- Modify: all callers in tests/benchmarks (19 files each, mechanical rename)
- Modify: doc comments in headers that reference `solve_impl`

**Renames:**
- `IVSolverFDM::solve_impl()` → `IVSolverFDM::solve()`
- `IVSolverFDM::solve_batch_impl()` → `IVSolverFDM::solve_batch()`
- `IVSolverInterpolated::solve_impl()` → `IVSolverInterpolated::solve()`
- `IVSolverInterpolated::solve_batch_impl()` → `IVSolverInterpolated::solve_batch()`

The type-erased `IVSolver` wrapper already uses `solve()` / `solve_batch()` — this makes concrete solvers match.

---

## Section 6: Clean Up PriceTableBuilder Internal API

Remove the duplicated `_for_testing` / `_internal` methods. Both are thin wrappers around the same private methods. Replace with a single set of private methods accessed via `friend`.

**Files:**
- Modify: `src/option/table/price_table_builder.hpp` — remove 11 public forwarding methods, add friend declarations
- Modify: `src/option/table/segmented_price_table_builder.cpp` — change `_for_testing` calls to direct private method calls (via friend)
- Modify: `src/option/table/adaptive_grid_builder.cpp` — change `_internal` calls to direct private method calls (via friend); update `AmericanOptionParams` references to `PricingParams`
- Create: `tests/price_table_builder_test_access.hpp` — inline test access helper
- Modify: ~5 test files — change `_for_testing` calls to use test access helper

**Before (11 public forwarding methods):**
```cpp
// 6 _for_testing methods (public)
make_batch_for_testing(), solve_batch_for_testing(), extract_tensor_for_testing(),
fit_coeffs_for_testing(), find_nearest_valid_neighbor_for_testing(),
repair_failed_slices_for_testing()

// 5 _internal methods (public)
make_batch_internal(), solve_batch_internal(), extract_tensor_internal(),
fit_coeffs_internal(), repair_failed_slices_internal()
```

**After:**
```cpp
class PriceTableBuilder {
public:
    // Only real API: constructor, build(), set_*(), factories

private:
    // Pipeline steps (single set, already exist as private methods)
    std::vector<PricingParams> make_batch(const PriceTableAxes<N>& axes) const;
    BatchAmericanOptionResult solve_batch(const std::vector<PricingParams>& batch,
                                          const PriceTableAxes<N>& axes) const;
    std::expected<ExtractionResult<N>, PriceTableError> extract_tensor(...) const;
    std::expected<FitResult, PriceTableError> fit_coeffs(...) const;
    std::optional<std::pair<size_t, size_t>> find_nearest_valid_neighbor(...) const;
    std::expected<RepairStats, PriceTableError> repair_failed_slices(...) const;

    friend class AdaptiveGridBuilder;
    friend class SegmentedPriceTableBuilder;

    // Test access: declared in tests/price_table_builder_test_access.hpp
    template <size_t M> friend struct PriceTableBuilderAccess;
};
```

**Test access helper** (`tests/price_table_builder_test_access.hpp`):
```cpp
// SPDX-License-Identifier: MIT
#pragma once
#include "src/option/table/price_table_builder.hpp"

namespace mango::testing {

template <size_t N>
struct PriceTableBuilderAccess {
    static inline auto make_batch(const PriceTableBuilder<N>& b, const PriceTableAxes<N>& a) {
        return b.make_batch(a);
    }
    // ... inline forwarding methods for each pipeline step
};

}  // namespace mango::testing
```

All methods are `inline` to avoid ODR violations.

Test files include this helper and call `PriceTableBuilderAccess<4>::make_batch(builder, axes)` instead of `builder.make_batch_for_testing(axes)`.

---

## Section 7: Rename OptionChain → OptionGrid

**Files:**
- Rename: `src/option/option_chain.hpp` → `src/option/option_grid.hpp`
- Modify: renamed file — `struct OptionChain` → `struct OptionGrid`
- Modify: `src/option/table/price_table_builder.hpp` — `from_chain()` → `from_grid()`, `from_chain_auto()` → `from_grid_auto()`, `from_chain_auto_profile()` → `from_grid_auto_profile()`
- Modify: `src/python/mango_bindings.cpp` — update binding name
- Modify: `BUILD.bazel` — update header references
- Modify: ~8 files that include `option_chain.hpp` or reference `OptionChain`

`mango::simple::OptionChain` stays unchanged — it genuinely represents a market option chain.

---

## Implementation Order

Each step compiles and tests pass before the next:

1. **Section 1** — Create `grid_spec_types.hpp`, relocate types, fix default, add `mandatory_times`
2. **Section 2** — Simplify `IVSolverFDMConfig`
3. **Section 3** — Unify grid in `AmericanOptionSolver` and batch solver; remove dead types
4. **Section 4** — Privatize constructor
5. **Section 5** — Rename `solve_impl` → `solve`
6. **Section 6** — Clean up `PriceTableBuilder` internal API
7. **Section 7** — Rename `OptionChain` → `OptionGrid`

One commit per section. Each section is independently valuable — if the work needs to stop partway, any prefix of this sequence is a valid stopping point.

---

## Python Bindings Migration Checklist

All sections that change public C++ API must also update `src/python/mango_bindings.cpp`:

- [ ] Section 2: Remove manual grid fields from `IVSolverFDMConfig` binding, add `PDEGridSpec` variant binding
- [ ] Section 3: Update `AmericanOptionSolver` binding to use `PDEGridSpec`; remove `AmericanOptionParams` binding alias
- [ ] Section 4: Switch binding from constructor to `create()` factory
- [ ] Section 5: Rename `solve_impl` → `solve` in all IV solver bindings
- [ ] Section 7: Rename `OptionChain` → `OptionGrid` in bindings
