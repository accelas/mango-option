# Hide PDEWorkspace from Public API — Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Hide `PDEWorkspace` from the public API of `AmericanOptionSolver` by moving its allocation into `solve()` via a thread-local `std::pmr::monotonic_buffer_resource`, then move the workspace headers into a Bazel-restricted `internal/` package.

**Architecture:** A `thread_local std::array<std::byte, ~270 KB>` per OS thread backs a `monotonic_buffer_resource` that is constructed fresh inside each `solve()` call. The resource has `std::pmr::new_delete_resource()` as its upstream, so grids exceeding `TLS_RESERVE_N=2048` transparently fall back to heap and are auto-freed when the arena destructs at scope exit. `AmericanOptionSolver` no longer holds `PDEWorkspace` as a member; the public header has zero PDE-internal includes.

**Tech Stack:** C++23 (`std::pmr`, `std::expected`, `std::span`, `std::variant`, `std::in_place_type`), Bazel + Bzlmod, GoogleTest, OpenMP.

**Spec:** [`docs/plans/2026-04-28-hide-pde-workspace-design.md`](2026-04-28-hide-pde-workspace-design.md). Read this before starting — it covers the design rationale, file-layout decisions, and risk analysis that this plan does not repeat.

---

## Conventions

- All test runs use `bazel test --test_output=all //tests:<target>`. Replace `--test_output=all` with `--test_output=errors` for batch runs.
- All builds use `bazel build //...`. CI parity check: `bazel build //... && bazel test //... && bazel build //benchmarks/... && bazel build //src/python:mango_option`.
- Each step ends with running tests; commit only after green.
- All commit messages follow the seven-rule format from CLAUDE.md (imperative mood, ≤50 char subject, body explains why).
- Per CLAUDE.md: when you find a bug during this work, add a regression test for it.

---

## Chunk 1: Prerequisites and Solver Restructure

This chunk lands the foundational change: `AmericanOptionSolver` no longer holds `PDEWorkspace`, and a new public auto API exists. The existing explicit-workspace API is kept temporarily so all current call sites continue to build. Subsequent chunks migrate them.

### Task 1: Make `PDEWorkspace::required_size` constexpr

**Files:**
- Modify: `src/pde/core/pde_workspace.hpp:46-60`

The pseudocode in the spec relies on `TLS_RESERVE_BYTES = required_size(TLS_RESERVE_N) * sizeof(double)` being a constant expression. The current `required_size` is a non-constexpr static; the body is pure arithmetic, so making it `constexpr` is a one-token edit.

- [ ] **Step 1: Add `constexpr` to the function signature**

```cpp
// src/pde/core/pde_workspace.hpp:41-43 — already constexpr
static constexpr size_t pad_to_simd(size_t n) {
    return ((n + SIMD_WIDTH - 1) / SIMD_WIDTH) * SIMD_WIDTH;
}

// src/pde/core/pde_workspace.hpp:46 — add constexpr
static constexpr size_t required_size(size_t n) {
    size_t n_padded = pad_to_simd(n);
    size_t n_minus_1_padded = pad_to_simd(n - 1);

    size_t regular_n = 12 * n_padded;
    size_t arrays_n_minus_1 = 3 * n_minus_1_padded;
    size_t tridiag = pad_to_simd(2 * n);

    return regular_n + arrays_n_minus_1 + tridiag;
}
```

- [ ] **Step 2: Verify it builds**

```bash
bazel build //src/pde/core:pde_workspace
```
Expected: build success.

- [ ] **Step 3: Add a constexpr smoke test**

In `tests/pde_workspace_test.cc`, add:
```cpp
TEST(PDEWorkspaceTest, RequiredSizeIsConstexpr) {
    constexpr size_t s = PDEWorkspace::required_size(1024);
    static_assert(s > 0, "required_size must be usable in constant expressions");
    EXPECT_GT(s, 0u);
}
```

Run: `bazel test //tests:pde_workspace_test --test_output=all`
Expected: PASS, including the new test.

- [ ] **Step 4: Commit**

```bash
git add src/pde/core/pde_workspace.hpp tests/pde_workspace_test.cc
git commit -m "$(cat <<'EOF'
Make PDEWorkspace::required_size constexpr

Required so that callers can use the result in compile-time
contexts (e.g., as a std::array template argument). Function body
is pure integer arithmetic — constexpr-ization adds no cost.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 2: Restructure solver — add auto API and PMR arena in `solve()`

**Files:**
- Modify: `src/option/american_option.hpp`
- Modify: `src/option/american_option.cpp`
- Test: `tests/american_option_auto_api_test.cc` (new)

This is the foundational change. Both API forms exist after this task: the existing `create(params, workspace, grid_spec)` (workspace stored as `optional<PDEWorkspace> explicit_workspace_`) and the new `create(params, grid_spec)`. `solve()` uses `explicit_workspace_` when set, otherwise constructs a `monotonic_buffer_resource` over `tls_storage` for the duration of the call.

**Design note on the optional:** `std::optional<T>` requires a complete `T` (unlike `std::unique_ptr<T>`). Therefore `american_option.hpp` keeps `#include "mango/pde/core/pde_workspace.hpp"` until the explicit form is deleted in Task 10. The "hide the include from the public header" goal is reached at Task 13, not Task 2.

- [ ] **Step 1: Write the failing parity test**

Create `tests/american_option_auto_api_test.cc`:
```cpp
// SPDX-License-Identifier: MIT
#include <gtest/gtest.h>
#include "mango/option/american_option.hpp"
#include "mango/pde/core/pde_workspace.hpp"
#include <memory_resource>
#include <vector>

namespace {

mango::PricingParams make_params() {
    return mango::PricingParams(
        mango::OptionSpec{
            .spot = 100.0, .strike = 100.0, .maturity = 1.0,
            .rate = 0.05, .dividend_yield = 0.02,
            .option_type = mango::OptionType::PUT},
        0.20);
}

TEST(AmericanOptionAutoAPITest, MatchesExplicitFormToMachineEpsilon) {
    auto params = make_params();

    // Explicit form (existing API)
    auto [grid_spec, time_domain] = mango::estimate_pde_grid(params);
    size_t n = grid_spec.n_points();
    std::pmr::vector<double> buffer(
        mango::PDEWorkspace::required_size(n),
        std::pmr::get_default_resource());
    auto ws = mango::PDEWorkspace::from_buffer(buffer, n).value();
    auto explicit_solver = mango::AmericanOptionSolver::create(params, ws).value();
    auto explicit_result = explicit_solver.solve();
    ASSERT_TRUE(explicit_result.has_value());

    // New auto form (no workspace param)
    auto auto_solver = mango::AmericanOptionSolver::create(params).value();
    auto auto_result = auto_solver.solve();
    ASSERT_TRUE(auto_result.has_value());

    // Pricing parity to machine epsilon — covers spot, delta, and an
    // off-spot value so the spatial-operator pointer-aliasing concern
    // (variant init must not move solver objects) is exercised.
    EXPECT_DOUBLE_EQ(
        explicit_result->value_at(params.spot),
        auto_result->value_at(params.spot));
    EXPECT_DOUBLE_EQ(
        explicit_result->delta(),
        auto_result->delta());
    EXPECT_DOUBLE_EQ(
        explicit_result->value_at(params.spot * 1.1),
        auto_result->value_at(params.spot * 1.1));
}

}  // namespace
```

Add a corresponding entry to `tests/BUILD.bazel`:
```python
cc_test(
    name = "american_option_auto_api_test",
    size = "small",
    srcs = ["american_option_auto_api_test.cc"],
    deps = [
        "//src/option:american_option",
        "//src/pde/core:pde_workspace",
        "@com_google_googletest//:gtest_main",
    ],
)
```

- [ ] **Step 2: Run test to verify it fails (no overload yet)**

```bash
bazel test //tests:american_option_auto_api_test --test_output=all
```
Expected: COMPILE FAIL (`AmericanOptionSolver::create` has no overload taking just `params`).

- [ ] **Step 3: Update private state — replace `workspace_` with `optional<PDEWorkspace>`**

In `src/option/american_option.hpp`:
- Keep `#include "mango/pde/core/pde_workspace.hpp"` (line 19) — required for `std::optional<PDEWorkspace>` since `optional` requires complete types. The include is removed in Task 13 when the header relocates.
- Replace `PDEWorkspace workspace_;` (line 95) with:
  ```cpp
  // Set when constructed via the legacy explicit-workspace API path;
  // empty when constructed via the auto API (solve() uses tls_storage).
  // The legacy path is removed in Task 10; this field then becomes
  // unconditionally empty and is deleted alongside it.
  std::optional<PDEWorkspace> explicit_workspace_;
  ```
- Adjust the private constructor's workspace parameter from `PDEWorkspace workspace` to `std::optional<PDEWorkspace> workspace`. The legacy `create(params, workspace, ...)` factory wraps its incoming `PDEWorkspace` into the optional via `std::optional<PDEWorkspace>{std::move(workspace)}`.

Special members can stay implicitly defaulted because `PDEWorkspace` is now a complete type at the class definition (the include is preserved).

- [ ] **Step 4: Add the thread-local arena and TLS_RESERVE constants**

In `src/option/american_option.cpp`, add to the include block at the top:
```cpp
#include "mango/pde/core/american_pde_workspace.hpp"
#include <array>
#include <cstddef>
#include <memory_resource>
#include <span>
```

Add the anonymous namespace near the top of the .cpp (after the existing includes, before the first function definition):
```cpp
namespace {
    // Sized to cover GridAccuracyParams::max_spatial_points (1200 by
    // default) without heap fallback. Per-thread footprint:
    // ~270 KB (33920 doubles × 8 bytes).
    constexpr size_t TLS_RESERVE_N = 2048;
    constexpr size_t TLS_RESERVE_BYTES =
        mango::PDEWorkspace::required_size(TLS_RESERVE_N) * sizeof(double);

    alignas(64) thread_local std::array<std::byte, TLS_RESERVE_BYTES> tls_storage;
}
```

- [ ] **Step 5: Move `solve_american_option` out of the public header**

In `src/option/american_option.hpp` lines 121-173: delete the inline definition; keep only a declaration:
```cpp
/// Solve a single American option with automatic grid determination.
/// Convenience wrapper around AmericanOptionSolver::create + solve.
std::expected<AmericanOptionResult, SolverError>
solve_american_option(const PricingParams& params);
```

In `src/option/american_option.cpp`, define it with explicit error mapping (`ValidationError` → `SolverErrorCode::InvalidConfiguration`):
```cpp
std::expected<AmericanOptionResult, SolverError>
solve_american_option(const PricingParams& params) {
    auto solver_result = AmericanOptionSolver::create(params);
    if (!solver_result.has_value()) {
        return std::unexpected(SolverError{
            .code = SolverErrorCode::InvalidConfiguration,
            .iterations = 0
        });
    }
    return solver_result->solve();
}
```

- [ ] **Step 6: Rewrite `solve()` with arena-when-empty / explicit-when-set logic**

Replace `AmericanOptionSolver::solve()` (currently at american_option.cpp:348-394) with:

```cpp
std::expected<AmericanOptionResult, SolverError>
AmericanOptionSolver::solve() {
    auto& [grid_spec, time_domain] = grid_config_;
    size_t n = grid_spec.n_points();

    // Pick workspace source.
    // - Legacy explicit path: caller provided a PDEWorkspace at create-time;
    //   it lives in explicit_workspace_ for the solver's lifetime.
    // - Auto path: build a fresh PMR arena over tls_storage; the
    //   AmericanPDEWorkspace lives only for this solve() invocation.
    std::optional<std::pmr::monotonic_buffer_resource> arena;
    std::optional<AmericanPDEWorkspace> arena_ws;
    PDEWorkspace* ws_ptr = nullptr;

    if (explicit_workspace_.has_value()) {
        ws_ptr = &*explicit_workspace_;
    } else {
        // Construct the resource in place so the bump pointer starts at
        // offset 0 of tls_storage. Reconstruct (rather than release()) on
        // each solve to avoid release-discipline bugs.
        arena.emplace(
            tls_storage.data(), tls_storage.size(),
            std::pmr::new_delete_resource());

        size_t bytes_needed = PDEWorkspace::required_size(n) * sizeof(double);
        auto* raw_bytes = static_cast<std::byte*>(
            arena->allocate(bytes_needed, 64));

        // Route through AmericanPDEWorkspace::from_bytes which calls
        // start_array_lifetime<double> internally — preserves
        // aliasing-safety (do NOT cast raw bytes directly to double*).
        // Construct in place via emplace to avoid moving the workspace
        // (its spans point into raw_bytes; moves are unsafe even though
        // spans are POD, because semantics require the constructed
        // object to remain at the storage address).
        auto from_bytes_result = AmericanPDEWorkspace::from_bytes(
            std::span<std::byte>(raw_bytes, bytes_needed), n);
        if (!from_bytes_result.has_value()) {
            return std::unexpected(SolverError{
                .code = SolverErrorCode::InvalidConfiguration,
                .iterations = 0
            });
        }
        arena_ws.emplace(std::move(*from_bytes_result));
        ws_ptr = &arena_ws->workspace();
    }

    // === Below: faithful relocation of american_option.cpp:351-394 ===
    auto grid_result = Grid<double>::create(
        grid_spec, time_domain,
        snapshot_times_.empty()
            ? std::span<const double>()
            : std::span<const double>(snapshot_times_));
    if (!grid_result.has_value()) {
        return std::unexpected(SolverError{
            .code = SolverErrorCode::InvalidConfiguration,
            .iterations = 0,
            .residual = grid_result.error().value
        });
    }
    auto grid = grid_result.value();

    // dx initialization
    auto dx_span = ws_ptr->dx();
    auto grid_points = grid->x();
    for (size_t i = 0; i < grid_points.size() - 1; ++i) {
        dx_span[i] = grid_points[i + 1] - grid_points[i];
    }

    // Variant construction with std::in_place_type — required because
    // AmericanPutSolver::spatial_op_ holds a non-owning pointer into
    // workspace_local_ (spatial_operator.hpp:170). Constructing the
    // solver as a temporary and moving it into the variant would
    // leave spatial_op_ pointing at the moved-from object's member.
    AmericanSolverVariant solver = (params_.option_type == OptionType::PUT)
        ? AmericanSolverVariant{
              std::in_place_type<AmericanPutSolver>,
              params_, grid, *ws_ptr}
        : AmericanSolverVariant{
              std::in_place_type<AmericanCallSolver>,
              params_, grid, *ws_ptr};

    // init_dividends() must come AFTER variant construction so callbacks
    // capture the post-construction address of dividend_spline_.
    auto solve_result = std::visit([&](auto& pde_solver) {
        pde_solver.init_dividends();
        if (custom_ic_) {
            pde_solver.initialize(*custom_ic_);
        } else {
            pde_solver.initialize(
                std::remove_reference_t<decltype(pde_solver)>::payoff);
        }
        pde_solver.set_config(trbdf2_config_);
        return pde_solver.solve();
    }, solver);

    if (!solve_result.has_value()) {
        return std::unexpected(solve_result.error());
    }
    return AmericanOptionResult(grid, params_);
}
```

- [ ] **Step 7: Add new `create(params, grid_spec, snapshot_times)` overload**

In `src/option/american_option.hpp`, after the existing `create` declaration:
```cpp
/// Create solver with auto-managed scratch buffer.
/// Internally uses a thread-local PMR arena (~270 KB per thread)
/// for typical grid sizes; falls back to heap for n > 2048.
static std::expected<AmericanOptionSolver, ValidationError>
create(const PricingParams& params,
       std::optional<PDEGridSpec> grid = std::nullopt,
       std::optional<std::span<const double>> snapshot_times = std::nullopt);
```

In `src/option/american_option.cpp`, add the definition. Resolve the grid to a `std::pair<GridSpec<double>, TimeDomain>` in both branches (matching the type of `grid_config_`):

```cpp
std::expected<AmericanOptionSolver, ValidationError>
AmericanOptionSolver::create(const PricingParams& params,
                             std::optional<PDEGridSpec> grid,
                             std::optional<std::span<const double>> snapshot_times) {
    // Resolve grid → (GridSpec<double>, TimeDomain) for grid_config_.
    // - If the caller gave a PDEGridSpec, use its fields directly.
    // - Otherwise auto-estimate.
    std::pair<GridSpec<double>, TimeDomain> resolved_grid_config;
    if (grid.has_value()) {
        // PDEGridSpec carries grid + n_time + mandatory_times; extract
        // GridSpec and reconstruct TimeDomain. (Mirror whatever the
        // existing explicit-form create does internally — extract into
        // a private helper if it grows beyond 5 lines.)
        resolved_grid_config = std::make_pair(
            grid->grid_spec,
            TimeDomain::from_n_steps(params.maturity, grid->n_time, grid->mandatory_times));
    } else {
        resolved_grid_config = estimate_pde_grid(params);
    }

    // (Same validation logic as the explicit-form create — extract to a
    // private static helper if non-trivial. For brevity here, assume any
    // params validation produces a ValidationError as today.)

    return AmericanOptionSolver(
        params, std::nullopt, resolved_grid_config, snapshot_times);
}
```

(If the existing explicit-form `create` does grid resolution slightly differently, mirror its logic. Refactor any duplication into a private static `resolve_grid_config(params, grid)` helper.)

- [ ] **Step 8: Run parity test**

```bash
bazel test //tests:american_option_auto_api_test --test_output=all
```
Expected: PASS — auto and explicit forms produce identical pricing, delta, and off-spot value to machine epsilon.

- [ ] **Step 9: Run American-pricing test set**

```bash
bazel test //tests:american_option_test //tests:iv_solver_test \
           //tests:american_option_batch_test //tests:real_market_data_test \
           //tests:quantlib_accuracy_test //tests:american_pde_workspace_test \
           --test_output=errors
```
Expected: all PASS.

- [ ] **Step 10: Run the full test suite**

```bash
bazel test //... --test_output=errors
```
Expected: 130 tests PASS.

- [ ] **Step 11: Commit**

```bash
git add src/option/american_option.hpp src/option/american_option.cpp \
        tests/american_option_auto_api_test.cc tests/BUILD.bazel
git commit -m "$(cat <<'EOF'
Restructure AmericanOptionSolver around thread-local PMR arena

solve() now builds a std::pmr::monotonic_buffer_resource over a
270 KB thread_local std::array per call. The arena destructs at
scope exit, auto-freeing any heap fallback for grids beyond
TLS_RESERVE_N=2048.

The new public create(params, grid_spec) overload constructs the
solver with no explicit workspace; solve() uses the arena. The
legacy create(params, workspace, ...) overload is preserved by
storing the caller's workspace in an optional<PDEWorkspace> that
solve() prefers when set. Both forms produce identical pricing,
delta, and off-spot values per the new parity test.

Variant construction uses std::in_place_type to avoid moving
solver objects whose spatial_op_ holds pointers into
workspace_local_ — same lifetime concern that already drove the
post-construction init_dividends() call.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Chunk 2: Migrate Public-API Call Sites

This chunk migrates all callers of `AmericanOptionSolver::create(params, workspace, ...)` to the auto form `create(params, ...)`. The old form still exists at the end of this chunk.

The migrations are mechanical and follow the same pattern. Each migration deletes the workspace allocation and the workspace param. Group them into commits by directory for clean review.

> **Task numbering note:** The original plan had a separate Task 3 for "move workspace allocation into solve()", but it was merged into Task 2 above to avoid an unworkable intermediate state (`std::optional<incomplete>` does not compile). Tasks 4–14 retain their original numbers.

### Task 4: Migrate `iv_solver.cpp`

**Files:**
- Modify: `src/option/iv_solver.cpp:80-130`

The existing code already does manual `thread_local` buffer management — replace with a single call to the auto API.

- [ ] **Step 1: Identify the block to replace**

Read `src/option/iv_solver.cpp:80-130`. The pattern is:
```cpp
thread_local std::vector<double> workspace_buffer;
// ... resize, from_buffer ...
auto solver = AmericanOptionSolver::create(option_params, workspace_result.value(), ...);
```

- [ ] **Step 2: Replace with auto API call**

Delete the `thread_local` buffer + `from_buffer` lines. Replace the `create` call with:
```cpp
auto solver = AmericanOptionSolver::create(option_params, PDEGridSpec{explicit_grid});
```
Drop the `#include "mango/pde/core/pde_workspace.hpp"` if no longer used.

- [ ] **Step 3: Run iv_solver tests**

```bash
bazel test //tests:iv_solver_test --test_output=errors
```
Expected: PASS.

- [ ] **Step 4: Commit**

```bash
git add src/option/iv_solver.cpp
git commit -m "$(cat <<'EOF'
Migrate iv_solver to auto-managed PDEWorkspace API

Replaces manual thread_local buffer management with the new
AmericanOptionSolver::create(params, grid_spec) overload. Behavior
preserved — the auto API uses the same thread_local arena strategy
internally.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 5: Migrate Python bindings

**Files:**
- Modify: `src/python/mango_bindings.cpp:355-365`

- [ ] **Step 1: Replace explicit workspace allocation with auto API**

Find the block at line 355 that does `std::vector<double> buffer(...)` + `from_buffer` + explicit `create`. Replace with a single auto-API call. Drop the `pde_workspace.hpp` include if no longer needed.

- [ ] **Step 2: Build Python bindings**

```bash
bazel build //src/python:mango_option
```
Expected: success.

- [ ] **Step 3: Commit**

```bash
git add src/python/mango_bindings.cpp
git commit -m "Migrate Python bindings to auto PDEWorkspace API"
```

---

### Task 6: Migrate test files (excluding white-box tests)

**Files:**
- Modify (each, mechanically — find/replace the `pmr::vector + from_buffer + create(params, ws)` pattern with `create(params)`):
  - `tests/real_market_data_test.cc` (lines 43-44, 70-71, 176-177)
  - `tests/normalized_solver_regression_test.cc` (lines 105-106)
  - `tests/american_option_test.cc` (lines 339-340)
  - `tests/quantlib_accuracy_test.cc` (lines 115, 174, 232, 255)
  - `tests/quantlib_accuracy_batch_test.cc` (lines 232-234, 267-269)
  - `tests/quantlib_validation_framework.hpp` (lines 149-151)
  - `tests/discrete_dividend_accuracy_test.cc` (lines 76-77)
  - `tests/custom_grid_example_test.cc` (8 sites, lines 46-48, 63-67, 82-86, 106-110, 136-138, 162-166, 180-181, 191-192)
  - `tests/eep_integration_test.cc` (lines 76, 80)
  - `tests/american_option_gamma_oscillation_test.cc` (lines 34-35, 41)

These files are *not* white-box tests — they exercise the public solver API only. White-box tests (which directly construct PDESolver, SpatialOperator, or PDEWorkspace) are migrated separately in Chunk 3.

- [ ] **Step 1: Apply the migration to each file**

For each file, locate the pattern:
```cpp
std::pmr::vector<double> buffer(PDEWorkspace::required_size(n), &pool);
auto ws = PDEWorkspace::from_buffer(buffer, n).value();
auto solver = AmericanOptionSolver::create(params, ws, [grid_spec]).value();
```
Replace with:
```cpp
auto solver = AmericanOptionSolver::create(params, [PDEGridSpec{grid_spec}]).value();
```
The `grid_spec` argument is optional; preserve it where the original passed one.

Also remove `#include "mango/pde/core/pde_workspace.hpp"` from each file's includes (it should no longer be needed after the migration).

- [ ] **Step 2: Build and test each file as you migrate**

For each file `tests/foo.cc` that has a corresponding `tests:foo` Bazel target:
```bash
bazel test //tests:foo --test_output=errors
```
Expected: PASS.

After all are done:
```bash
bazel test //tests:... --test_output=errors
```

- [ ] **Step 3: Commit (one commit per directory or one commit total — your choice)**

```bash
git add tests/
git commit -m "$(cat <<'EOF'
Migrate test suite to auto PDEWorkspace API

Removes explicit PDEWorkspace allocation from 10 test files that
exercise only the public AmericanOptionSolver API. Behavior
preserved; tests still cover the same scenarios with simpler
setup. White-box tests (pde_solver_test, etc.) are migrated
separately along with their move into tests/internal/.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 7: Migrate benchmarks

**Files:**
- Modify (mechanically, same pattern as Task 6):
  - `benchmarks/quantlib_accuracy.cc` (lines 125, 255, 308, 393, 625, 692)
  - `benchmarks/quantlib_performance.cc` (lines 94, 149, 204, 261)
  - `benchmarks/readme_benchmarks.cc` (lines 192, 249)
  - `benchmarks/interpolation_greek_accuracy.cc` (lines 145, 162)
  - `benchmarks/latency_sweep.cc` (lines 113, 200)
  - `benchmarks/real_data_benchmark.cc` (lines 187, 237)
  - `benchmarks/quantlib_mesh_comparison.cc` (lines 118, 356)
  - `benchmarks/component_performance.cc` (lines 108-118, 139-149, 170-180, 205-215, 337-351, 400-414)
  - `benchmarks/grid_sweep.cc` (lines 67-68)
  - `benchmarks/iv_interpolation_sweep.cc` (line 77)

- [ ] **Step 1: Apply the migration to each benchmark**

Same find/replace pattern as Task 6.

- [ ] **Step 2: Build all benchmarks**

```bash
bazel build //benchmarks/... 2>&1 | tail -20
```
Expected: build success for all targets.

- [ ] **Step 3: Run a representative perf benchmark to confirm no regression**

```bash
bazel run -c opt //benchmarks:iv_benchmark 2>&1 | tail -30
```
Compare median µs to the value before migration. Expected: within 2%.

If you do not have a baseline run from before the chunk started: run the benchmark on `main` first, save the output, then return to your branch and re-run. A 2% difference in either direction is acceptable; larger swings indicate something to investigate.

- [ ] **Step 4: Commit**

```bash
git add benchmarks/
git commit -m "Migrate benchmarks to auto PDEWorkspace API"
```

---

### Task 8: Migrate the batch hot loop

**Files:**
- Modify: `src/option/american_option_batch.cpp:444-595`

This is the largest mechanical simplification in the refactor. The current batch path manages a per-thread `ThreadWorkspaceBuffer` with explicit size pre-computation and a heap fallback. After the migration, each contract just calls `AmericanOptionSolver::create(params[i], solver_grid_spec)` — the thread-local arena inside `solve()` handles buffer reuse and oversize fallback automatically.

- [ ] **Step 1: Read the current loop**

Read `src/option/american_option_batch.cpp:440-600` to understand the structure. Note these sections:
- Lines 444-466: pre-loop workspace size computation (deletes)
- Lines 469-486: thread setup including `ThreadWorkspaceBuffer` (the buffer object deletes)
- Lines 491-573: per-iteration workspace plumbing including heap fallback (deletes)
- Line 583: `AmericanOptionSolver::create(params[i], *workspace_ptr, solver_grid_spec)` (becomes auto form)

- [ ] **Step 2: Replace the workspace machinery**

Delete the workspace_size_bytes computation, the `ThreadWorkspaceBuffer buffer(...)` declaration, all the `workspace_ptr` / `workspace_storage` / `heap_buffer_storage` plumbing, and the heap-fallback branches. The remaining loop body should be approximately:

```cpp
MANGO_PRAGMA_PARALLEL
{
    std::shared_ptr<Grid<double>> thread_grid;
    if (use_shared_grid && shared_grid.has_value()) {
        auto [grid_spec, time_domain] = shared_grid.value();
        auto grid_result = Grid<double>::create(grid_spec, time_domain);
        if (grid_result.has_value()) thread_grid = grid_result.value();
    }

    MANGO_PRAGMA_FOR_STATIC
    for (size_t i = 0; i < params.size(); ++i) {
        // Resolve grid for this contract
        std::optional<PDEGridSpec> solver_grid_spec;
        if (use_shared_grid) {
            solver_grid_spec = PDEGridConfig{
                shared_grid->first,
                shared_grid->second.n_steps(), {}};
        } else {
            auto [grid_spec, time_domain] = resolved_custom_grid.has_value()
                ? resolved_custom_grid.value()
                : estimate_pde_grid(params[i], grid_accuracy_);
            solver_grid_spec = PDEGridConfig{grid_spec, time_domain.n_steps(), {}};
        }

        auto solver_result = AmericanOptionSolver::create(params[i], solver_grid_spec);
        if (!solver_result) {
            results[i] = std::unexpected(SolverError{
                .code = SolverErrorCode::InvalidConfiguration,
                .iterations = 0});
            MANGO_PRAGMA_ATOMIC ++failed_count;
            continue;
        }
        auto& solver = solver_result.value();

        if (!snapshot_times_.empty())
            solver.set_snapshot_times(std::span{snapshot_times_});
        // ... rest of solver setup + .solve() call unchanged
    }
}
```

The `ThreadWorkspaceBuffer` import / forward declaration at the top of the file may also become unused — drop it if so.

- [ ] **Step 3: Run batch tests**

```bash
bazel test //tests:american_option_batch_test //tests:american_option_batch_workspace_test --test_output=errors
```
Expected: PASS.

- [ ] **Step 4: Run batch benchmark for perf parity**

```bash
bazel run -c opt //benchmarks:american_option_batch_benchmark 2>&1 | tail -30
```
Compare to baseline. Expected: within 2% — the thread-local arena gives the same buffer-reuse property as the old `ThreadWorkspaceBuffer`.

- [ ] **Step 5: Commit**

```bash
git add src/option/american_option_batch.cpp
git commit -m "$(cat <<'EOF'
Simplify batch loop to use auto PDEWorkspace API

Removes ~120 lines of explicit workspace plumbing
(ThreadWorkspaceBuffer, AmericanPDEWorkspace::from_bytes,
heap-fallback branches, pre-loop size computation). Each contract
now calls AmericanOptionSolver::create(params, grid_spec); the
thread-local arena inside solve() handles buffer reuse and
oversize fallback automatically. Per-thread memory footprint is
unchanged (~270 KB).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 9: Resolve `american_option_new_api_test.cc` (prerequisite for chunk 3)

**Files:**
- Decide: `tests/american_option_new_api_test.cc` — rewrite each test to use the auto API, or delete the file entirely.

Per the spec: this file actively tests the explicit-workspace `create(params, workspace, ...)` form which is deleted in chunk 3. The file's name ("new_api") refers to a *previously-new* API that is now legacy.

Recommendation: **delete the file**. Its 5 test cases (`SolveWithPDEWorkspace`, `SolveWithSnapshots`, `CallOptionWithNewAPI`, `ValueAtInterpolation`, `GreeksComputation`) all overlap with coverage in `american_option_test.cc` and the new `american_option_auto_api_test.cc` from Task 2.

- [ ] **Step 1: Verify coverage overlap**

```bash
grep -l "ValueAtInterpolation\|GreeksComputation\|Snapshots" tests/*.cc
```
If at least one other test file covers each named scenario, deletion is safe.

- [ ] **Step 2: Delete file and BUILD entry**

```bash
git rm tests/american_option_new_api_test.cc
```
Edit `tests/BUILD.bazel` to remove the `cc_test` block named `american_option_new_api_test`.

- [ ] **Step 3: Verify test suite still builds and passes**

```bash
bazel test //tests:... --test_output=errors
```
Expected: all PASS.

- [ ] **Step 4: Commit**

```bash
git add tests/BUILD.bazel
git commit -m "$(cat <<'EOF'
Remove obsolete american_option_new_api_test

The file tested the explicit-workspace create(params, workspace,
...) overload which is removed in the next commit. All scenarios
it covered (snapshots, calls, value_at, Greeks) are exercised by
american_option_test and american_option_auto_api_test.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Chunk 3: Cleanup and Header Move

This chunk removes the now-unused explicit-workspace `create` overload, moves white-box tests to `tests/internal/`, moves the workspace headers to `src/pde/internal/`, and verifies the boundary holds.

### Task 10: Delete the explicit-workspace `create` overload

**Files:**
- Modify: `src/option/american_option.hpp` (remove old `create` declaration)
- Modify: `src/option/american_option.cpp` (remove old `create` definition + `explicit_workspace_` field handling)

After Task 9, no callers of the explicit form remain in production, tests, or benchmarks. Delete the API.

- [ ] **Step 1: Verify no callers remain**

```bash
grep -rn "AmericanOptionSolver::create.*workspace" src tests benchmarks
```
Expected: zero hits after chunk 2 + Task 9 are complete.

White-box tests (`pde_solver_test.cc`, `spatial_operator_jacobian_test.cc`, `pde_workspace_test.cc`, `temporal_event_test.cc`, `obstacle_test.cc`, `pde_solver_snapshot_test.cc`, `american_pde_workspace_test.cc`, `american_option_batch_workspace_test.cc`) construct `PDESolver`, `SpatialOperator`, and `PDEWorkspace` directly — they do *not* call `AmericanOptionSolver::create(params, workspace, ...)`. Verified by grep: none of those files call the public `AmericanOptionSolver::create` factory at all. Task 10 therefore does not break any white-box test.

- [ ] **Step 2: Remove the explicit-form declaration from the header**

In `src/option/american_option.hpp`: delete the `create(const PricingParams&, PDEWorkspace, ...)` overload declaration. Also delete the `explicit_workspace_` private member and any related ctor parameter — the solver only uses the thread-local arena now.

- [ ] **Step 3: Remove the explicit-form definition from the .cpp**

In `src/option/american_option.cpp`: delete `AmericanOptionSolver::create(const PricingParams&, PDEWorkspace, ...)` (currently around line 309). Simplify `solve()` to drop the `if (explicit_workspace_)` branch — only the arena path remains.

- [ ] **Step 4: Run full test suite**

```bash
bazel test //... --test_output=errors
```
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add src/option/american_option.hpp src/option/american_option.cpp
git commit -m "$(cat <<'EOF'
Remove explicit-workspace AmericanOptionSolver::create overload

All public callers (and the batch hot loop) were migrated to the
auto API in earlier commits. White-box tests construct PDESolver
and PDEWorkspace directly via internal headers, so they are
unaffected. The solver no longer holds an optional<PDEWorkspace>
member — solve() always uses the thread-local arena.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 11: Move white-box tests to `tests/internal/`

**Files:**
- Create: `tests/internal/BUILD.bazel`
- Move:
  - `tests/pde_workspace_test.cc` → `tests/internal/pde_workspace_test.cc`
  - `tests/american_pde_workspace_test.cc` → `tests/internal/american_pde_workspace_test.cc`
  - `tests/american_option_batch_workspace_test.cc` → `tests/internal/american_option_batch_workspace_test.cc`
  - `tests/pde_solver_test.cc` → `tests/internal/pde_solver_test.cc`
  - `tests/temporal_event_test.cc` → `tests/internal/temporal_event_test.cc`
  - `tests/spatial_operator_jacobian_test.cc` → `tests/internal/spatial_operator_jacobian_test.cc`
  - `tests/pde_solver_snapshot_test.cc` → `tests/internal/pde_solver_snapshot_test.cc`
  - `tests/obstacle_test.cc` → `tests/internal/obstacle_test.cc`

- [ ] **Step 1: Create the new test package**

```bash
mkdir -p tests/internal
```

Write `tests/internal/BUILD.bazel` with all 8 `cc_test` entries. The deps come from copying each test's existing `cc_test` block in `tests/BUILD.bazel`. Each test gets the same deps it has today; the only difference is the BUILD file location.

Concretely, before writing the new file, capture the existing dep lists:
```bash
grep -A8 "name = \"pde_workspace_test\"" tests/BUILD.bazel
grep -A8 "name = \"american_pde_workspace_test\"" tests/BUILD.bazel
grep -A8 "name = \"american_option_batch_workspace_test\"" tests/BUILD.bazel
grep -A8 "name = \"pde_solver_test\"" tests/BUILD.bazel
grep -A8 "name = \"temporal_event_test\"" tests/BUILD.bazel
grep -A8 "name = \"spatial_operator_jacobian_test\"" tests/BUILD.bazel
grep -A8 "name = \"pde_solver_snapshot_test\"" tests/BUILD.bazel
grep -A8 "name = \"obstacle_test\"" tests/BUILD.bazel
```

Then write `tests/internal/BUILD.bazel`. Skeleton (deps must be filled in from the grep output above — each test's existing dep list, plus the workspace dep stays as `//src/pde/core:pde_workspace` for now, swapped in Task 13):

```python
# SPDX-License-Identifier: MIT
load("@rules_cc//cc:defs.bzl", "cc_test")

package(default_visibility = ["//visibility:private"])

# White-box tests for PDE solver internals. These exercise types
# (PDEWorkspace, PDESolver, SpatialOperator) that move to
# //src/pde/internal:workspace in Task 13. For now, the deps still
# reference //src/pde/core: targets — the swap happens atomically
# in Task 13's commit.

cc_test(
    name = "pde_workspace_test",
    size = "small",
    srcs = ["pde_workspace_test.cc"],
    deps = [<copy from tests/BUILD.bazel>],
)

cc_test(
    name = "american_pde_workspace_test",
    size = "small",
    srcs = ["american_pde_workspace_test.cc"],
    deps = [<copy from tests/BUILD.bazel>],
)

cc_test(
    name = "american_option_batch_workspace_test",
    size = "small",
    srcs = ["american_option_batch_workspace_test.cc"],
    deps = [<copy from tests/BUILD.bazel>],
)

cc_test(
    name = "pde_solver_test",
    size = "small",
    srcs = ["pde_solver_test.cc"],
    deps = [<copy from tests/BUILD.bazel — note: this likely
             needs //src/pde/core:pde_solver and operator deps>],
)

cc_test(
    name = "temporal_event_test",
    size = "small",
    srcs = ["temporal_event_test.cc"],
    deps = [<copy from tests/BUILD.bazel — needs operator deps>],
)

cc_test(
    name = "spatial_operator_jacobian_test",
    size = "small",
    srcs = ["spatial_operator_jacobian_test.cc"],
    deps = [<copy from tests/BUILD.bazel — needs spatial_operator
             and operator_factory deps>],
)

cc_test(
    name = "pde_solver_snapshot_test",
    size = "small",
    srcs = ["pde_solver_snapshot_test.cc"],
    deps = [<copy from tests/BUILD.bazel — needs operator deps>],
)

cc_test(
    name = "obstacle_test",
    size = "small",
    srcs = ["obstacle_test.cc"],
    deps = [<copy from tests/BUILD.bazel — needs operator_factory dep>],
)
```

Important: the `.cc` files moved in Step 2 below keep their existing
`#include "mango/pde/core/..."` paths. They get rewritten to
`mango/pde/internal/...` atomically in Task 13. This intermediate
state builds because the headers are still at `src/pde/core/` until
Task 13 moves them.

- [ ] **Step 2: Physically move the files**

```bash
git mv tests/pde_workspace_test.cc tests/internal/pde_workspace_test.cc
git mv tests/american_pde_workspace_test.cc tests/internal/american_pde_workspace_test.cc
# ... repeat for all 8 files
```

- [ ] **Step 3: Remove the old `cc_test` entries from `tests/BUILD.bazel`**

For each moved file, delete its `cc_test` block in `tests/BUILD.bazel`.

- [ ] **Step 4: Verify all 8 new test targets build and pass**

```bash
bazel test //tests/internal/... --test_output=errors
```
Expected: all 8 PASS.

- [ ] **Step 5: Verify the rest of the test suite still passes**

```bash
bazel test //tests:... --test_output=errors
```
Expected: PASS (with 8 fewer test files).

- [ ] **Step 6: Commit**

```bash
git add tests/internal/ tests/BUILD.bazel
git commit -m "$(cat <<'EOF'
Move white-box PDE solver tests to tests/internal/

These tests directly construct PDEWorkspace, PDESolver, and
SpatialOperator — types that are about to be moved into
src/pde/internal/. Their new location matches their
internal-API consumer status. Test contents are unchanged; only
the build location changes.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 12: Delete dead test file

**Files:**
- Delete: `tests/american_option_solver_test.cc` and its BUILD entry

Per the spec, this file references `PDEWorkspace::create(grid_spec, mr)` — an API that no longer exists. Verified earlier with `bazel build //tests:american_option_solver_test` returning compile errors.

- [ ] **Step 1: Verify the file does not currently build**

```bash
bazel build //tests:american_option_solver_test 2>&1 | tail -3
```
Expected: BUILD FAIL with errors about removed API. (Confirmed before chunk 1; re-verifying.)

- [ ] **Step 2: Delete file and BUILD entry**

```bash
git rm tests/american_option_solver_test.cc
```
Edit `tests/BUILD.bazel` to remove the `american_option_solver_test` block.

- [ ] **Step 3: Verify suite still builds**

```bash
bazel build //... 2>&1 | tail -5
```
Expected: success.

- [ ] **Step 4: Commit**

```bash
git add tests/BUILD.bazel
git commit -m "$(cat <<'EOF'
Remove dead american_option_solver_test

References removed PDEWorkspace::create(grid_spec, mr) API; has
not built for some time. No live references in CI or other tests.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 13: Move workspace headers to `src/pde/internal/` (atomic commit)

**Files:**
- Move:
  - `src/pde/core/pde_workspace.hpp` → `src/pde/internal/pde_workspace.hpp`
  - `src/pde/core/american_pde_workspace.hpp` → `src/pde/internal/american_pde_workspace.hpp`
  - `src/pde/core/pde_solver.hpp` → `src/pde/internal/pde_solver.hpp`
  - `src/pde/operators/spatial_operator.hpp` → `src/pde/internal/spatial_operator.hpp`
  - `src/pde/operators/operator_factory.hpp` → `src/pde/internal/operator_factory.hpp`
- Create: `src/pde/internal/BUILD.bazel`
- Modify: every file that includes any of the above (must be one atomic commit)
- Modify: `src/option/BUILD.bazel` to use `implementation_deps`

This task **must be a single commit**. All file moves, all `#include` updates, and all BUILD changes happen together — intermediate states do not build.

- [ ] **Step 1: Create the new internal package**

```bash
mkdir -p src/pde/internal
```

Capture the existing deps before writing the new file:
```bash
grep -A12 'name = "pde_workspace"' src/pde/core/BUILD.bazel
grep -A12 'name = "american_pde_workspace"' src/pde/core/BUILD.bazel
grep -A12 'name = "pde_solver"' src/pde/core/BUILD.bazel
grep -A12 'name = "spatial_operator"' src/pde/operators/BUILD.bazel
grep -A12 'name = "operator_factory"' src/pde/operators/BUILD.bazel
```

Take the union of all `deps` lists from those 5 grep outputs (deduped) and use it as the `deps` for the new combined target.

Then write `src/pde/internal/BUILD.bazel`:
```python
# SPDX-License-Identifier: MIT
load("@rules_cc//cc:defs.bzl", "cc_library")

package(default_visibility = ["//visibility:private"])

cc_library(
    name = "workspace",
    hdrs = [
        "pde_workspace.hpp",
        "american_pde_workspace.hpp",
        "pde_solver.hpp",
        "spatial_operator.hpp",
        "operator_factory.hpp",
    ],
    visibility = [
        "//src/option:__pkg__",
        "//tests/internal:__pkg__",
        "//benchmarks:__pkg__",
    ],
    deps = [
        # Filled in from the grep output above. At minimum (verify
        # against the existing BUILD files):
        "//src/math:tridiagonal_matrix_view",
        "//src/math:thomas_solver",
        "//src/pde/core:pde_grid",       # if pde_solver depends on it
        "//src/pde/core:boundary_conditions",
        "//src/pde/operators:black_scholes_pde",
        "//src/pde/operators:laplacian_pde",
        "//src/pde/operators:centered_difference",
        "//src/support:error_types",
        "//src/support:lifetime",
        "//src/support:parallel",
    ],
)
```

Verify the dep list builds cleanly with:
```bash
bazel build //src/pde/internal:workspace 2>&1 | tail -10
```
Any "undeclared inclusion" errors point to missing deps — add them.

- [ ] **Step 2: Move the 5 headers**

```bash
git mv src/pde/core/pde_workspace.hpp src/pde/internal/pde_workspace.hpp
git mv src/pde/core/american_pde_workspace.hpp src/pde/internal/american_pde_workspace.hpp
git mv src/pde/core/pde_solver.hpp src/pde/internal/pde_solver.hpp
git mv src/pde/operators/spatial_operator.hpp src/pde/internal/spatial_operator.hpp
git mv src/pde/operators/operator_factory.hpp src/pde/internal/operator_factory.hpp
```

- [ ] **Step 3: Update all `#include` paths**

For every consumer (now exclusively `src/option/american_option.cpp`, `src/option/american_option_batch.cpp`, `tests/internal/*.cc`, and `benchmarks/*.cc`), update includes:

```cpp
// Before:
#include "mango/pde/core/pde_workspace.hpp"
#include "mango/pde/core/american_pde_workspace.hpp"
#include "mango/pde/core/pde_solver.hpp"
#include "mango/pde/operators/spatial_operator.hpp"
#include "mango/pde/operators/operator_factory.hpp"

// After:
#include "mango/pde/internal/pde_workspace.hpp"
#include "mango/pde/internal/american_pde_workspace.hpp"
#include "mango/pde/internal/pde_solver.hpp"
#include "mango/pde/internal/spatial_operator.hpp"
#include "mango/pde/internal/operator_factory.hpp"
```

Find them all with: `grep -rln "mango/pde/core/pde_workspace\|mango/pde/core/american_pde_workspace\|mango/pde/core/pde_solver\|mango/pde/operators/spatial_operator\|mango/pde/operators/operator_factory" src tests benchmarks`

- [ ] **Step 4: Update `src/option/BUILD.bazel` to use `implementation_deps`**

In the `cc_library` block for `american_option`, replace the dep on the (now-deleted) `//src/pde/core:pde_workspace` with:
```python
implementation_deps = [
    "//src/pde/internal:workspace",
],
```
Verify the rest of `deps` doesn't transitively pull the workspace headers.

- [ ] **Step 5: Remove the old workspace targets from `src/pde/core/BUILD.bazel` and `src/pde/operators/BUILD.bazel`**

Delete the `cc_library` blocks that exposed the moved headers. Other targets in those packages (e.g., `black_scholes_pde`) stay.

- [ ] **Step 6: Update `tests/internal/BUILD.bazel` to depend on the new target**

Change `"//src/pde/core:pde_workspace"` (used as a placeholder in Task 11) to `"//src/pde/internal:workspace"` for every `cc_test` in `tests/internal/`.

- [ ] **Step 7: Build everything**

```bash
bazel build //... 2>&1 | tail -10
```
Expected: success. If any non-internal target fails because it tries to include a moved header, that's a missed call site — fix the include and the BUILD dep.

- [ ] **Step 8: Run the full test suite**

```bash
bazel test //... --test_output=errors
```
Expected: 130 tests PASS.

- [ ] **Step 9: Build benchmarks and Python bindings (CI parity)**

```bash
bazel build //benchmarks/... && bazel build //src/python:mango_option
```
Expected: success.

- [ ] **Step 10: Commit (one atomic commit)**

The single commit must include: the 5 moved headers, the new `src/pde/internal/BUILD.bazel`, the deletion of header entries from `src/pde/core/BUILD.bazel` and `src/pde/operators/BUILD.bazel`, the include-path rewrites in `src/option/american_option.cpp`, `src/option/american_option_batch.cpp`, all `tests/internal/*.cc` files, all `benchmarks/*.cc` files, and the `src/option:american_option` BUILD update to use `implementation_deps`.

```bash
git add -A src/pde/ \
        src/option/BUILD.bazel \
        src/option/american_option.cpp \
        src/option/american_option_batch.cpp \
        tests/internal/ \
        benchmarks/
git commit -m "$(cat <<'EOF'
Move PDE workspace headers to src/pde/internal/

Restricts visibility of PDEWorkspace, AmericanPDEWorkspace,
PDESolver, SpatialOperator, and create_spatial_operator to
//src/option, //tests/internal, and //benchmarks. The public
//src/option:american_option target now depends on
//src/pde/internal:workspace via implementation_deps, preventing
transitive header exposure to library consumers.

Atomic move: all 5 headers, every #include, and the BUILD changes
land in a single commit so intermediate states build cleanly.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 14: Verify no leaks

- [ ] **Step 1: Grep for stale references**

The pattern uses `\b` word boundaries so it catches `SpatialOperator`,
`SpatialOperator&`, `SpatialOperator*`, `unique_ptr<SpatialOperator>`,
etc., not just `SpatialOperator(`:

```bash
grep -rnE "\b(PDEWorkspace|AmericanPDEWorkspace|PDESolver|SpatialOperator|create_spatial_operator|from_buffer|from_buffer_and_grid|from_bytes|required_size|required_bytes)\b" \
    src tests benchmarks \
    | grep -v "src/pde/internal/" \
    | grep -v "src/option/american_option\.cpp" \
    | grep -v "src/option/american_option_batch\.cpp" \
    | grep -v "tests/internal/" \
    | grep -v "benchmarks/" \
    | grep -v "src/math/bspline/" \
    | grep -v "src/math/cubic_spline" \
    | grep -v "BSplineCollocationWorkspace"
```

Expected: zero hits. The `bspline` and `cubic_spline` exclusions
filter out the unrelated B-spline workspace (a separate type that
also has its own `from_bytes`/`required_bytes`).

If hits remain, each is a missed migration — fix before declaring
the refactor complete.

- [ ] **Step 2: Confirm public-header isolation**

```bash
grep -n "pde_workspace\|pde_solver\|spatial_operator\|operator_factory" \
    src/option/american_option.hpp
```

Expected: zero hits. The public header includes none of the now-internal types.

- [ ] **Step 3: Confirm forward declaration is clean**

```bash
grep -n "class PDEWorkspace" src/option/american_option.hpp
```

Expected: zero hits (Task 10 removed the `optional<PDEWorkspace>` member, so the forward declaration also goes).

- [ ] **Step 4: Run the full CI matrix locally**

```bash
bazel test //... && bazel build //benchmarks/... && bazel build //src/python:mango_option
```
Expected: all green.

- [ ] **Step 5: Run the perf benchmarks one more time, compare to pre-refactor baseline**

Before starting Chunk 1, capture a baseline by checking out `main` and saving:
```bash
git checkout main
bazel run -c opt //benchmarks:iv_benchmark > /tmp/iv_baseline.txt
bazel run -c opt //benchmarks:american_option_benchmark > /tmp/aopt_baseline.txt
bazel run -c opt //benchmarks:american_option_batch_benchmark > /tmp/batch_baseline.txt
git checkout <feature-branch>
```

After Task 13:
```bash
bazel run -c opt //benchmarks:iv_benchmark > /tmp/iv_new.txt
bazel run -c opt //benchmarks:american_option_benchmark > /tmp/aopt_new.txt
bazel run -c opt //benchmarks:american_option_batch_benchmark > /tmp/batch_new.txt
diff /tmp/iv_baseline.txt /tmp/iv_new.txt   # eyeball median µs columns
```

Expected: each median within 2% of baseline. If a benchmark regresses by more than 2%, investigate — likely cause is the heap-fallback path engaging unexpectedly (verify `n` stays under `TLS_RESERVE_N=2048`).

- [ ] **Step 6: (Optional) Run with ThreadSanitizer to catch hidden races**

```bash
bazel test --config=tsan //tests/... //tests/internal/... --test_output=errors
```
(Only if a `tsan` config exists in this project. If not, skip — `solve()` does not introduce any cross-thread sharing of the new arena.)

- [ ] **Step 7: Final cleanup commit (if anything trivial slipped through)**

```bash
git status
```

If clean, no commit needed.

---

## Summary of expected outcome

After this plan completes:
- Public header `src/option/american_option.hpp` has no PDE-internal includes.
- `PDEWorkspace`, `AmericanPDEWorkspace`, `PDESolver`, `SpatialOperator`, `create_spatial_operator` live in `src/pde/internal/`, accessible only to `//src/option`, `//tests/internal`, `//benchmarks`.
- Public users construct solvers via `AmericanOptionSolver::create(params, grid_spec=auto)` or `solve_american_option(params)` — no workspace ceremony.
- Per-thread memory footprint: ~270 KB for the `tls_storage` array, allocated lazily on first `solve()` per thread.
- Batch hot loop is ~120 lines shorter; the workspace-management complexity disappears.
- 130 existing tests pass unchanged (after locality and migration fixups). Two new tests (`american_option_auto_api_test` parity test + `OversizeFallsBackToHeap`) exercise the new code paths.
- `am_option_solver_test.cc` and `am_option_new_api_test.cc` deleted (dead/obsolete).

The refactor is reversible at any point up to Task 13 (the atomic header move). After Task 13, reverting requires backing out the BUILD changes plus the include updates, which is mechanical but touches many files — prefer to verify thoroughly before merging Task 13.
