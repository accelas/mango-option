# Public API Rename Design

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Remove implementation details from public API names.

**Approach:** Mechanical rename, split into independent batches by symbol. Archive docs (`docs/archive/`, `docs/plans/`) are left unchanged — they reflect the API at the time of writing.

## Rename Table

| # | Current | New | Rationale |
|---|---|---|---|
| 1 | `solve_american_option_auto` | `solve_american_option` | "_auto" is implementation detail |
| 2 | `estimate_grid_for_option` | `estimate_pde_grid` | "for_option" redundant; "pde" distinguishes from price table grid estimators |
| 3 | `compute_global_grid_for_batch` | `estimate_batch_pde_grid` | Consistent with `estimate_pde_grid`, conveys union-grid intent |
| 4 | `IVSolverFDM` | `IVSolver` | FDM is implementation detail; this is the primary solver |
| 5 | `IVSolverFDMConfig` | `IVSolverConfig` | Follows class rename |
| 6 | `IVSolverConfig` (factory, iv_solver_factory.hpp) | `IVSolverFactoryConfig` | Avoids collision with #5; "Factory" matches `make_*` pattern |
| 7 | `IVSolver` (factory wrapper, iv_solver_factory.hpp) | `AnyIVSolver` | Conveys type-erasure; avoids collision with #4 |
| 8 | `make_iv_solver` | `make_interpolated_iv_solver` | Clarifies it builds interpolated solvers, not FDM |
| 9 | `IVSolverInterpolated<S>` | `InterpolatedIVSolver<S>` | Adjective-first reads naturally |
| 10 | `IVSolverInterpolatedConfig` | `InterpolatedIVSolverConfig` | Follows class rename |
| 11 | `IVSolverInterpolatedStandard` | `DefaultInterpolatedIVSolver` | Keep as alias with clear name; used in Python bindings and vol_surface |
| 12 | `ExplicitPDEGrid` | `PDEGridConfig` | "Config" clarifies it's a configuration, not a grid object |
| 13 | `grid_accuracy_profile()` (grid_spec_types.hpp) | `make_grid_accuracy()` | Consistent with `make_*` factory pattern |
| 14 | `grid_accuracy_profile()` (price_table_grid_estimator.hpp) | `make_price_table_grid_accuracy()` | Rename both overloads together for consistency |

### Header file renames

| Current | New | Rationale |
|---|---|---|
| `iv_solver_fdm.hpp` / `.cpp` | `iv_solver.hpp` / `.cpp` | Match class rename |
| `iv_solver_interpolated.hpp` / `.cpp` | `interpolated_iv_solver.hpp` / `.cpp` | Match class rename |

Update all `#include` paths accordingly.

## Scope

**In scope:** All `.hpp`, `.cpp`, `.cc`, `.py` source files, active docs (README.md, CLAUDE.md, API_GUIDE.md, ARCHITECTURE.md, PYTHON_GUIDE.md, MATHEMATICAL_FOUNDATIONS.md, PERF_ANALYSIS.md), Python bindings, BUILD.bazel files (for header renames).

**Out of scope:** `docs/archive/`, `docs/plans/` (historical).

## Ordering

Renames #4–#8 must land in the same commit to avoid `IVSolver` / `IVSolverConfig` name collisions.

Recommended batches:
1. **Grid config:** `ExplicitPDEGrid` → `PDEGridConfig`, `grid_accuracy_profile` → `make_grid_accuracy` / `make_price_table_grid_accuracy`
2. **Grid functions:** `estimate_grid_for_option` → `estimate_pde_grid`, `compute_global_grid_for_batch` → `estimate_batch_pde_grid`
3. **IV solver core:** `IVSolverFDM` → `IVSolver`, `IVSolverFDMConfig` → `IVSolverConfig`, `IVSolverConfig` (factory) → `IVSolverFactoryConfig`, `IVSolver` (factory) → `AnyIVSolver`, `make_iv_solver` → `make_interpolated_iv_solver`, header renames
4. **IV solver interpolated:** `IVSolverInterpolated` → `InterpolatedIVSolver`, `IVSolverInterpolatedConfig` → `InterpolatedIVSolverConfig`, `IVSolverInterpolatedStandard` → `DefaultInterpolatedIVSolver`, header renames
5. **Convenience:** `solve_american_option_auto` → `solve_american_option`

## Verification

After each batch:
- `bazel test //...` — all tests pass
- `bazel build //benchmarks/...` — benchmarks compile
- `bazel build //src/python:mango_option` — Python bindings compile
- Grep for old names in source files to confirm none remain
