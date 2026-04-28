# Hide PDEWorkspace from Public API — Design

**Date:** 2026-04-28
**Status:** Draft (rev 3 — fixed lifetime/UB issues from Codex review,
TLS_RESERVE_N raised to 2048 to cover default `max_spatial_points`,
`implementation_deps` for transitive-exposure prevention,
`std::in_place_type` for variant safety)

## Problem

`PDEWorkspace` is a 16-field byte-buffer-backed scratch struct used by the
American option PDE solver. It currently appears in every public
constructor, factory, operator method, and is held by value as a member
of public classes:

- `AmericanOptionSolver::create(params, workspace, grid_spec)` — public factory
  (american_option.cpp:309-330)
- `AmericanOptionSolver::workspace_` — value member of the public class
  (american_option.hpp:95)
- `PDESolver<Derived>::workspace_` — value member of the CRTP base
  (pde_solver.hpp:229)
- `create_spatial_operator(pde, grid, workspace)` — public factory
  (operator_factory.hpp:14, 22)
- `SpatialOperator(pde, grid, workspace)` — public ctor (spatial_operator.hpp:46)
- `solve_american_option(params)` — convenience function defined inline
  in the public header (american_option.hpp:131-173) that itself
  allocates a workspace and calls the explicit form
- `PDEWorkspace::from_buffer(buffer, n)` and `from_buffer_and_grid(...)`
  — required by every direct caller

Because the workspace is a value member of `AmericanOptionSolver`, the
public header `american_option.hpp` must include `pde_workspace.hpp`.
Same for `pde_solver.hpp`. Hiding the type therefore requires
restructuring how the workspace is owned, not just where the header
lives.

For most callers (single-shot pricing, IV solving, Python bindings,
tests) the workspace ceremony is pure overhead — they don't need
control over the buffer's allocation strategy. The pattern was designed
for the batch hot loop (`american_option_batch.cpp`), which legitimately
needs zero per-contract allocation by reusing one buffer per OpenMP
thread across all contracts on that thread.

The contrasting reference point is `BSplineCollocationWorkspace` — a
similarly-shaped byte-buffer scratch struct used by the B-spline fitter.
It is allocated locally inside `BSplineNDSeparable::fit_axis` and never
appears in the public `BSplineNDInterpolator::evaluate()` API. No
consumer of B-spline interpolation has ever needed to know it exists.

## Goal

Make `PDEWorkspace` an internal implementation detail. Public users see
only `AmericanOptionSolver::create(params, grid_spec=auto)`. The batch
hot loop uses the same public API — no separate explicit-workspace form
remains. Buffer reuse across calls is preserved via a thread-local
PMR arena inside `solve()`.

## Non-Goals

- No changes to `PDEWorkspace`'s field layout (16 fields, SIMD padding,
  `from_buffer` slicing logic).
- No changes to `BSplineCollocationWorkspace` (already at the right level).
- No changes to `RawTensor::contract` per-call vectors (separate concern).
- No changes to PDE solver mathematics (TR-BDF2, Newton iteration, etc.).

## Architecture

```
PUBLIC API (mango::)
  AmericanOptionSolver
    │ holds: params, grid_config, snapshot_times, trbdf2_config,
    │        custom_ic   — NO workspace member
    │
  AmericanOptionSolver::create(params, grid_spec=auto)
  solve_american_option(params)
    │
    │ public header has zero PDE-internal includes
    │
    ▼
INTERNAL (.cpp file)
  AmericanOptionSolver::solve()
    │ constructs std::pmr::monotonic_buffer_resource over
    │ thread_local std::array<std::byte, TLS_RESERVE_BYTES>
    │ (heap fallback via new_delete_resource for oversize n)
    │ allocates 64-byte-aligned PDEWorkspace from arena
    │ runs variant<AmericanPutSolver, AmericanCallSolver>
    │ arena destructs at end of solve(), releases any heap fallback
    │
    ▼
INTERNAL TYPES (namespace unchanged — see below)
  PDEWorkspace, AmericanPDEWorkspace      [in mango::]
  PDESolver<Derived>                      [in mango::]
  SpatialOperator<PDE, T>                 [in mango::operators::]
  create_spatial_operator(...)            [in mango::operators::]
    │ headers moved from src/pde/core/ and src/pde/operators/
    │ to src/pde/internal/
    │ Bazel visibility restricted to //src/option:__pkg__,
    │ //tests/internal:__pkg__, //benchmarks:__pkg__
    │ Combined with implementation_deps to prevent transitive exposure
```

**Namespace decision:** types keep their current namespaces
(`mango::`, `mango::operators::`). Renaming to `mango::internal::`
would multiply the diff with mechanical updates and create churn for
no functional benefit — the boundary is enforced by file location +
Bazel visibility, not by namespace. Internal-ness is a property of
the build graph, not the symbol name. Any future external consumer
who tries to use these types will fail at the Bazel layer regardless
of namespace.

## File Layout Changes

| Current path | New path |
|---|---|
| `src/pde/core/pde_workspace.hpp` | `src/pde/internal/pde_workspace.hpp` |
| `src/pde/core/american_pde_workspace.hpp` | `src/pde/internal/american_pde_workspace.hpp` |
| `src/pde/core/pde_solver.hpp` | `src/pde/internal/pde_solver.hpp` |
| `src/pde/operators/spatial_operator.hpp` | `src/pde/internal/spatial_operator.hpp` |
| `src/pde/operators/operator_factory.hpp` | `src/pde/internal/operator_factory.hpp` |

`PDESolver` is moved because it holds `PDEWorkspace workspace_;` as a
value member (pde_solver.hpp:229) and is only used as a CRTP base by
the private `AmericanPutSolver` / `AmericanCallSolver` variants in
`american_option.cpp`. Public users never construct it directly.

`SpatialOperator` and `create_spatial_operator` are similarly used only
by `american_option.cpp` and white-box tests.

Bazel: new target `//src/pde/internal:workspace`. Direct `visibility`
prevents unauthorized direct deps but does *not* prevent transitive
exposure (a public target that depends on `:workspace` could still
re-export the headers via its public include path). Use
`implementation_deps` for the link, not `deps`:

```python
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
    deps = ["//src/math:tridiagonal_matrix_view", ...],
)

# In //src/option:american_option:
cc_library(
    name = "american_option",
    hdrs = ["american_option.hpp"],          # public, no internal includes
    srcs = ["american_option.cpp", ...],
    deps = [...public deps...],
    implementation_deps = [
        "//src/pde/internal:workspace",       # private — not transitively exposed
    ],
    visibility = ["//visibility:public"],
)
```

`implementation_deps` (Bazel ≥ 6.0) ensures that consumers of
`//src/option:american_option` cannot transitively `#include` headers
from `//src/pde/internal:workspace`. Combined with the visibility
restriction, this gives belt-and-braces enforcement.

If the project's Bazel version doesn't support `implementation_deps`,
the alternative is to split into a private impl target whose public
target is the only thing exposed, and only the impl target depends on
`:workspace`.

## API Changes

### Public

```cpp
// src/option/american_option.hpp
// (no #include of pde_workspace.hpp, pde_solver.hpp, etc.)

class AmericanOptionSolver {
public:
    /// Create a solver with auto-managed scratch buffer.
    /// Internally uses a thread-local PMR arena (~270 KB per thread)
    /// for typical grid sizes; falls back to heap for oversize.
    static std::expected<AmericanOptionSolver, ValidationError>
    create(const PricingParams& params,
           std::optional<PDEGridSpec> grid = std::nullopt,
           std::optional<std::span<const double>> snapshot_times = std::nullopt);

    void set_snapshot_times(std::span<const double> times);
    void set_trbdf2_config(const TRBDF2Config& config);
    void set_initial_condition(InitialCondition ic);

    std::expected<AmericanOptionResult, SolverError> solve();

private:
    // No PDEWorkspace member. solve() constructs the arena locally.
    PricingParams params_;
    std::pair<GridSpec<double>, TimeDomain> grid_config_;
    std::vector<double> snapshot_times_;
    TRBDF2Config trbdf2_config_;
    std::optional<InitialCondition> custom_ic_;

    AmericanOptionSolver(...);  // private ctor, defined in .cpp
};

/// Single-call convenience function.
std::expected<AmericanOptionResult, SolverError>
solve_american_option(const PricingParams& params);
```

### Internal

No new internal API. The batch path uses the public `create(params, grid)`
API and benefits from the same thread-local arena automatically — each
OpenMP thread has its own `tls_storage`, reused across all contracts on
that thread.

### Implementation of `solve()`

The new `solve()` is a faithful relocation of the existing body
(american_option.cpp:348-394) with workspace allocation moved from
member initialization into a fresh arena per call. All other steps —
grid creation, dx initialization, variant construction, dividend init
ordering (which depends on the *post-move* address of
`dividend_spline_`), `set_config`, custom IC handling, payoff
initialization, result construction — are preserved verbatim.

**Prerequisites for the pseudocode below:**

- `PDEWorkspace::required_size(size_t)` must be marked `static constexpr`
  (it is currently a non-constexpr static at pde_workspace.hpp:46).
  The body is pure arithmetic — making it `constexpr` is a one-token
  change. Required so `TLS_RESERVE_BYTES` can be a constant expression
  for the `std::array` template argument.
- The variant alternative must be constructed *in-place* via
  `std::in_place_type<...>` rather than from a temporary, because
  `AmericanPutSolver::spatial_op_` holds a non-owning pointer into
  `workspace_local_` (spatial_operator.hpp:170). Constructing the
  solver as a temporary and moving it into the variant would leave
  `spatial_op_` pointing at the moved-from object's `workspace_local_`.
  This is the same lifetime concern that already drives the post-move
  `init_dividends()` call.
- After `arena.allocate(bytes, alignment)` returns raw storage,
  `start_array_lifetime<double>(ptr, count)` must be called before
  forming a `std::span<double>` — `allocate` provides aligned storage,
  not constructed objects. The existing
  `AmericanPDEWorkspace::from_bytes` (american_pde_workspace.hpp:67)
  already follows this pattern; we route through it rather than
  bypassing it.

```cpp
// src/option/american_option.cpp

#include "mango/pde/internal/pde_workspace.hpp"
#include "mango/pde/internal/american_pde_workspace.hpp"
#include "mango/support/lifetime.hpp"  // for start_array_lifetime
#include <array>
#include <cstddef>
#include <memory_resource>
#include <span>
// ... other internal includes

namespace {
    // Sized to cover GridAccuracyParams::max_spatial_points (1200 by
    // default at grid_spec_types.hpp:48) without heap fallback.
    // Per-thread footprint: ~270 KB (33920 doubles × 8 bytes for n=2048).
    // 16-thread OpenMP pool: ~4.3 MB total, no growth.
    constexpr size_t TLS_RESERVE_N = 2048;
    constexpr size_t TLS_RESERVE_BYTES =
        PDEWorkspace::required_size(TLS_RESERVE_N) * sizeof(double);

    // alignas on the array element type (std::byte); std::array layout
    // guarantees the data starts at the array's address, so this gives
    // a 64-byte-aligned starting address.
    alignas(64) thread_local std::array<std::byte, TLS_RESERVE_BYTES> tls_storage;
}

std::expected<AmericanOptionResult, SolverError>
AmericanOptionSolver::solve() {
    // Fresh arena per solve(). Reconstructing the resource resets the
    // bump pointer to offset 0 of tls_storage — no release() needed.
    // Heap fallback (via new_delete_resource) is auto-freed when arena
    // destructs at scope exit.
    std::pmr::monotonic_buffer_resource arena{
        tls_storage.data(), tls_storage.size(),
        std::pmr::new_delete_resource()
    };

    auto& [grid_spec, time_domain] = grid_config_;
    size_t n = grid_spec.n_points();

    // Allocate aligned bytes from arena, then route through
    // AmericanPDEWorkspace::from_bytes which calls start_array_lifetime
    // internally — preserving the existing aliasing-safe pattern.
    size_t bytes_needed =
        PDEWorkspace::required_size(n) * sizeof(double);
    auto* raw_bytes = static_cast<std::byte*>(
        arena.allocate(bytes_needed, 64));
    auto american_ws_result = AmericanPDEWorkspace::from_bytes(
        std::span<std::byte>(raw_bytes, bytes_needed), n);
    if (!american_ws_result.has_value()) {
        return std::unexpected(SolverError{
            .code = SolverErrorCode::InvalidConfiguration,
            .iterations = 0
        });
    }
    auto& ws = american_ws_result->workspace();

    // Grid construction (relocates american_option.cpp:351-362)
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

    // dx initialization from grid spacing (american_option.cpp:364-369)
    auto dx_span = ws.dx();
    auto grid_points = grid->x();
    for (size_t i = 0; i < grid_points.size() - 1; ++i) {
        dx_span[i] = grid_points[i + 1] - grid_points[i];
    }

    // Variant construction with in_place_type to avoid moving a
    // solver object whose spatial_op_ holds a pointer into
    // workspace_local_. This matches the safety contract that
    // already protects dividend_spline_.
    AmericanSolverVariant solver = (params_.option_type == OptionType::PUT)
        ? AmericanSolverVariant{
              std::in_place_type<AmericanPutSolver>,
              params_, grid, ws}
        : AmericanSolverVariant{
              std::in_place_type<AmericanCallSolver>,
              params_, grid, ws};

    // init_dividends() must come AFTER variant construction so
    // callbacks capture the post-construction address of
    // `dividend_spline_`. Preserves the order from
    // american_option.cpp:376-387.
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
    // arena destructs here; any heap-fallback allocation is freed
}
```

`solve_american_option(params)` requires explicit error mapping
because `create()` returns `expected<_, ValidationError>` while
`solve()` returns `expected<_, SolverError>`. Cannot use a single
`and_then`. Concretely:

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

Moves out of `american_option.hpp` into `american_option.cpp`.

## Migration Order

Each step is independently mergeable and leaves the codebase in a
working state.

### Step 1: Restructure `AmericanOptionSolver` and `solve()`

- Move `solve_american_option` definition out of `american_option.hpp`
  into `american_option.cpp`.
- Replace `PDEWorkspace workspace_;` member with the PMR-arena-based
  solve() implementation shown above.
- Add new public `create(params, grid_spec, snapshot_times)` overload.
- Keep the old `create(params, workspace, grid_spec, snapshot_times)`
  for now, but route it through the same code path:
  the workspace argument is no longer stored on the solver; instead,
  callers of the explicit form get a deprecation note that the
  workspace will be replaced by the thread-local arena. (This keeps
  white-box tests building during transition.)
- All existing tests pass without modification.
- Add new unit test verifying the public auto API produces identical
  results to the explicit form to machine epsilon.

### Step 2: Migrate public-API call sites to the auto form

All of these allocate a one-shot workspace today and discard it after
the solve. Switch each to call `AmericanOptionSolver::create(params)`
or `solve_american_option(params)` (no workspace param). Behavior is
preserved.

| File | Line(s) |
|---|---|
| `src/option/iv_solver.cpp` | 86-120 (already manual `thread_local`) |
| `src/python/mango_bindings.cpp` | 358-360 |
| `tests/real_market_data_test.cc` | 43-44, 70-71, 176-177 |
| `tests/normalized_solver_regression_test.cc` | 105-106 |
| `tests/american_option_test.cc` | 339-340 |
| `tests/quantlib_accuracy_test.cc` | 115, 174, 232, 255 (multiple) |
| `tests/quantlib_accuracy_batch_test.cc` | 232-234, 267-269 |
| `tests/quantlib_validation_framework.hpp` | 149-151 |
| `tests/discrete_dividend_accuracy_test.cc` | 76-77 |
| `tests/custom_grid_example_test.cc` | 46-48, 63-67, 82-86, 106-110, 136-138, 162-166, 180-181, 191-192 |
| `tests/eep_integration_test.cc` | 76, 80 |
| `tests/american_option_gamma_oscillation_test.cc` | 34-35, 41 |
| `tests/american_option_new_api_test.cc` | All 5 test cases — actively tests the explicit-workspace `create(params, workspace, ...)` form. Either rewrite each to use the auto API, or delete the file (its name suggests it was meant to test a *new* API at the time, which is no longer new). Decision documented in step 7. |
| `benchmarks/quantlib_accuracy.cc` | 125, 255, 308, 393, 625, 692 |
| `benchmarks/quantlib_performance.cc` | 94, 149, 204, 261 |
| `benchmarks/readme_benchmarks.cc` | 192, 249 |
| `benchmarks/interpolation_greek_accuracy.cc` | 145, 162 |
| `benchmarks/latency_sweep.cc` | 113, 200 (two distinct workspace/create paths) |
| `benchmarks/real_data_benchmark.cc` | 187, 237 |
| `benchmarks/quantlib_mesh_comparison.cc` | 118, 356 |
| `benchmarks/component_performance.cc` | 108-118, 139-149, 170-180, 205-215, 337-351, 400-414 |
| `benchmarks/grid_sweep.cc` | 67-68 |
| `benchmarks/iv_interpolation_sweep.cc` | 77 (direct `PDEWorkspace` allocation + `AmericanOptionSolver::create` — *not* indirect via `IVSolver` as an earlier draft incorrectly assumed) |

Plus: `tests/obstacle_test.cc:75, 137` — has a *second* `create_spatial_operator` call site at line 137 in addition to line 75 (covered by step 5, not step 2). The full step-5 enumeration of white-box tests is unchanged but obstacle_test has 2 sites, not 1.

After this step, the explicit-workspace form has no public users.

### Step 3: Migrate batch path to public API

`src/option/american_option_batch.cpp` (lines 444-595): replace the
entire `ThreadWorkspaceBuffer` / `AmericanPDEWorkspace::from_bytes` /
heap-fallback machinery with a straight call to
`AmericanOptionSolver::create(params[i], solver_grid_spec)` per
contract. Each OpenMP thread gets its own `tls_storage`, reused across
contracts on that thread — exact same property as today's
`ThreadWorkspaceBuffer`. Net deletion: ~120 lines.

The pre-loop workspace size estimation (lines 444-466) is also deleted
— no longer needed.

### Step 4: Delete the old explicit-workspace `create` API

- **Pre-requirement**: `tests/american_option_new_api_test.cc` must
  be migrated or deleted before this step. It currently uses the
  old API in 5 test cases and is listed in `tests/BUILD.bazel`.
  Verify with `grep -c "AmericanOptionSolver::create" tests/american_option_new_api_test.cc`
  reports zero non-auto-form calls before proceeding.
- Remove `AmericanOptionSolver::create(params, workspace, grid_spec, snapshot_times)`
  from `american_option.hpp` and its definition at
  `american_option.cpp:309-330`.
- Remove the (now-unused) old constructor.
- All callers were migrated in steps 2-3.

### Step 5: Move white-box tests to `tests/internal/`

These tests legitimately exercise internal types (workspace as a test
parameter, direct calls to `create_spatial_operator`, etc.). Move them
to a new `tests/internal/` directory with visibility on
`//src/pde/internal:workspace`:

| Test file | Why internal |
|---|---|
| `tests/pde_workspace_test.cc` | Tests `PDEWorkspace` directly |
| `tests/american_pde_workspace_test.cc` | Tests `AmericanPDEWorkspace` directly |
| `tests/american_option_batch_workspace_test.cc` | Tests batch workspace handling |
| `tests/pde_solver_test.cc` | Helper takes `PDEWorkspace` (lines 20, 66, 129) |
| `tests/temporal_event_test.cc` | Helper takes `PDEWorkspace` (lines 25, 50, 73, 76) |
| `tests/spatial_operator_jacobian_test.cc` | Calls `create_spatial_operator` directly |
| `tests/pde_solver_snapshot_test.cc` | Calls `create_spatial_operator` directly (lines 71, 114) |
| `tests/obstacle_test.cc` | Calls `create_spatial_operator` directly (lines 75, 137) |

Test code is unchanged; only build location and visibility change.

### Step 6: Move workspace headers to `internal/` (single atomic commit)

This step must be done as one commit — the header move + all
`#include` updates + Bazel changes happen together, otherwise
intermediate states won't build.

- Physically move the five headers per the File Layout Changes table.
- Update all `#include` paths simultaneously:
  - `src/option/american_option.cpp` (workspace + operator + pde_solver)
  - `src/option/american_option_batch.cpp`
  - All `tests/internal/*.cc` files moved in step 5
- Remove `#include "mango/pde/core/pde_workspace.hpp"` from
  `src/option/american_option.hpp` (line 19) — no longer needed
  after step 1's restructure.
- Remove `#include "mango/pde/core/pde_solver.hpp"` from
  `src/option/american_option.hpp` (line 10) — also no longer needed.
- Update Bazel BUILD files: create `//src/pde/internal:workspace`
  with restricted visibility; delete the old `//src/pde/core` and
  `//src/pde/operators` workspace-related targets.
- Bazel visibility enforces that no public-visibility target can
  include these headers.

### Step 7: Delete dead/obsolete tests

- `tests/american_option_solver_test.cc` — does not currently build
  (references removed `PDEWorkspace::create(grid_spec, mr)` API).
  Verified via `bazel build //tests:american_option_solver_test`.
  Remove the file and its BUILD entry.

(`tests/american_option_new_api_test.cc` migration is a *prerequisite*
for step 4, not a step-7 audit — see step 4 pre-requirement. Choose
"rewrite for auto API" or "delete entirely" before reaching step 4.)

### Step 8: Verify no leaks

- Grep for `PDEWorkspace`, `from_buffer`, `from_bytes`,
  `from_buffer_and_grid`, `required_size`, `required_bytes`,
  `create_spatial_operator`, `SpatialOperator(` in all files outside
  `src/pde/internal/`, `src/option/american_option*.cpp`,
  `tests/internal/`, and `benchmarks/`. Should be zero hits.
- Bazel build with restricted visibility fails if any public target
  still depends on `//src/pde/internal:workspace`.
- `bazel test //...` passes (130 tests).

## Error Handling

- `arena.allocate(bytes, 64)` for `n ≤ 2048` returns memory inside
  `tls_storage` — never fails.
- For `n > 2048`, `arena.allocate` falls back to `new_delete_resource`,
  which may throw `std::bad_alloc`. Same exposure as today's
  `pmr::vector<double> buffer(...)` calls; behavior preserved.
- `PDEWorkspace::from_buffer` returns `std::expected`, but after
  arena allocation the buffer is guaranteed correctly sized, so
  `.value()` cannot fail. Local invariant.
- All other error paths (validation, Newton non-convergence, etc.)
  are unchanged.

## Testing

### Existing tests (must pass unchanged after migration)

- All 130 tests in `bazel test //...`.
- Specifically verify: `american_option_test`, `iv_solver_test`,
  `american_option_batch_test`, `real_market_data_test`,
  `quantlib_accuracy_test`, all white-box tests after step 5 move.

### New tests (added in step 1)

`tests/american_option_arena_test.cc`:

1. `PublicCreateMatchesExplicitForm`: during the transition window
   (after step 1, before step 4), call public `create(params)` and
   the explicit form with the same inputs; verify identical pricing
   result to machine epsilon. After step 4 (explicit form deleted),
   the test checks `create(params)` against a known-good pricing
   regression value.
2. `OversizeFallsBackToHeap`: call with `n > TLS_RESERVE_N` (e.g.,
   `n = 4096`); verify result is numerically correct (the heap-fallback
   path produces identical math to the in-array path). No
   allocation-counting required — correctness is the contract.

Note: explicit allocation-counting tests are *not* added. They would
require a test-only seam to inject a custom `memory_resource` into
`solve()`, which the design deliberately omits to keep the public API
clean. Production behavior is verified by:
- `OversizeFallsBackToHeap` (correctness of fallback path)
- Existing benchmarks (perf regression detection — see below)

A `-fsanitize=thread` run of `bazel test //...` on the migrated
codebase catches data races without a dedicated concurrency test.

### Performance verification

Run before and after the full migration:

- `benchmarks:iv_benchmark` — single-shot IV path (most sensitive
  to per-call overhead).
- `benchmarks:american_option_benchmark` — single-shot pricing.
- `benchmarks:american_option_batch_benchmark` — batch hot loop.

Pass criterion: no benchmark regresses by more than 2%. The new
implementation does an extra `monotonic_buffer_resource` construction
(stack allocation, ~constant time) and `arena.allocate` (pointer bump)
per `solve()` — both negligible compared to the PDE solve itself
(~0.3ms ATM, ~9-19ms off-ATM per CLAUDE.md).

## Risks and Mitigations

| Risk | Mitigation |
|---|---|
| `tls_storage` (~270 KB/thread at TLS_RESERVE_N=2048) consumes memory even for processes that never solve | Acceptable — same order as today's `ThreadWorkspaceBuffer` allocations. 16-thread OpenMP pool: ~4.3 MB total. Users who care can compile-time-tune `TLS_RESERVE_N` down. |
| Tight IV-solver loop with grids beyond `TLS_RESERVE_N` would pay heap alloc per `solve()` | `TLS_RESERVE_N=2048` covers `GridAccuracyParams::max_spatial_points=1200` (default at grid_spec_types.hpp:48) plus headroom. If a future user configures `max_spatial_points > 2048`, switch to a grow-only `thread_local std::vector<std::byte>` fallback (separate change, simple swap). |
| `solve()` is implicitly non-reentrant (recursive call would invalidate the arena) | Today's solver has no recursion. Document the contract in a comment on `solve()`. Add a debug-only sentinel (`thread_local bool solving_`) that asserts on reentry under `-DDEBUG`. |
| Hidden global mutable state (`tls_storage`) makes tests harder to instrument for allocation counts | Accepted: the design deliberately omits a test-only API seam to keep the surface clean. Allocation behavior is verified indirectly through correctness tests (parity with the explicit form during transition + oversize fallback test) and through benchmark perf comparison. No allocation-counting tests are added. |
| White-box tests blocked by step 4 (old API deletion) | Verified by grep that white-box tests (`pde_solver_test`, `spatial_operator_jacobian_test`, `pde_workspace_test`, etc.) construct `PDESolver` / `SpatialOperator` / `PDEWorkspace` directly without going through `AmericanOptionSolver::create(params, workspace, ...)`. Step 4 deletes only the latter, leaving lower-level internal-API tests unaffected. |
| Batch path benchmarks regress because `tls_storage` is per-thread but batch's existing `ThreadWorkspaceBuffer` was identical — should be same | Confirm with `american_option_batch_benchmark` before/after. If regression observed, root-cause (likely cache effects from changed allocation pattern). |
| `pmr::monotonic_buffer_resource` construction overhead per solve() | Stack-allocated, no syscalls, no atomics. Measured constant ~tens of ns. Negligible vs. ~0.3ms+ solve. |
| Removing `PDEWorkspace::workspace_` member changes `sizeof(AmericanOptionSolver)` — could break ABI | This library has no stable ABI guarantee. Python bindings rebuild against new headers. No external consumers tracked. |
| `solve_american_option`'s current per-call allocation pattern (american_option.hpp:139) changes to thread-local reuse | Strict improvement: zero alloc after first call per thread vs. one alloc per call. No correctness change. Document in API comment. |

## Open Questions

None. All design points settled in pre-spec discussion.

## References

- Discussion thread: conversation 2026-04-28
- Related code:
  - `src/pde/core/pde_workspace.hpp` — current workspace
  - `src/pde/core/pde_solver.hpp:229` — workspace member of CRTP base
  - `src/option/american_option.hpp:95` — workspace member of public class
  - `src/option/american_option.cpp:309-330` — current explicit-workspace create
  - `src/option/american_option.cpp:372-374` — variant construction (relocates into solve())
  - `src/option/american_option_batch.cpp:444-595` — batch hot loop (simplifies)
  - `src/option/iv_solver.cpp:86-120` — current manual `thread_local` pattern
  - `src/math/bspline/bspline_collocation_workspace.hpp` — analogous workspace already kept private
  - `src/math/bspline/bspline_nd_separable.hpp:322-329` — example of locally-allocated workspace
