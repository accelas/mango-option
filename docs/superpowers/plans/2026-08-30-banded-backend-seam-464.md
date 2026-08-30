# Banded-Backend Seam (#464) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make LAPACK an implementation detail of the banded-solver layer: a `BandedSolverBackend` concept + `LapackBandedBackend` policy type absorb all six `LAPACKE_*` call sites and the banded-layout knowledge out of `bspline_collocation.hpp`, with bit-for-bit identical output.

**Architecture:** New backend-agnostic header (concept + relocated `BandedResult`) and a LAPACK backend header (double-only statics wrapping pack/dgbtrf/dgbtrs/dgbcon). `BSplineCollocation1D` and `BSplineCollocationFactorization` gain a defaulted, concept-constrained `Backend` template parameter; every fit path is re-expressed on backend primitives in the exact existing LAPACK call order. The `//tests:bspline_bit_identity_test` goldens (pinned in #457) prove nothing changed numerically.

**Tech Stack:** C++23 concepts, Bazel, GoogleTest, LAPACKE, mdspan (`lapack_banded_layout`).

**Spec:** `docs/superpowers/specs/2026-08-30-banded-backend-seam-464-design.md`

## Global Constraints

- Every new source file starts with `// SPDX-License-Identifier: MIT` (`#` form in BUILD files).
- Compile clean under `-Wall -Wextra` (CI adds `-Werror`). No printf in library code; no `<functional>` in these headers.
- **Bit-for-bit output identity**: `//tests:bspline_bit_identity_test` must pass unchanged after every task — it is the acceptance proof.
- `fit()`/`fit_with_buffer()`/`fit_with_workspace()` must issue backend primitives in the exact existing order `pack → factorize(dgbtrf) → copy RHS → solve(dgbtrs) → residual → norm → condition(dgbcon)`. NEVER implement them as public `factorize()` + `solve_factored()` (that reorders dgbcon before dgbtrs).
- `LapackBandedBackend` members are plain `double` functions, NOT templates with a body `static_assert` — the concept must be false for `float`.
- Never commit `MODULE.bazel.lock` churn (`git checkout -- MODULE.bazel.lock` before staging).
- Existing template spellings (`BSplineCollocation1D<double>`, `BSplineCollocationFactorization<double>`) must keep compiling via defaulted parameters.

---

### Task 1: Backend-agnostic header — concept + BandedResult relocation

**Files:**
- Create: `src/math/banded_solver_backend.hpp`
- Modify: `src/math/banded_matrix_solver.hpp` (delete the `BandedResult` definition at lines 33–59; add `#include "mango/math/banded_solver_backend.hpp"`)
- Modify: `src/math/BUILD.bazel` (new target; `banded_matrix_solver` gains the dep)

**Interfaces:**
- Produces: `mango::BandedResult<T>` (moved verbatim), `mango::BandedSolverBackend<B, T>` concept — consumed by Tasks 2–3 exactly as written in the spec's "New header `src/math/banded_solver_backend.hpp`" section (copy the concept from the spec verbatim).

- [ ] **Step 1: Create the header.** SPDX line; `#pragma once`; includes `<concepts>`, `<cstddef>`, `<optional>`, `<span>`, `<string_view>`. Move the `BandedResult` struct (banded_matrix_solver.hpp lines 33–59) verbatim, then the `BandedSolverBackend` concept exactly as in the spec (the `std::floating_point<T> && requires(...)` form with `factor_storage_size`, `pivot_storage_size`, `pack`, `factorize`, `solve`, `condition`, and `typename B::pivot_type`).
- [ ] **Step 2: Update `banded_matrix_solver.hpp`** — delete its `BandedResult` definition, add the include (keep everything else, including `<lapacke.h>`).
- [ ] **Step 3: BUILD** — in `src/math/BUILD.bazel` add:

```python
cc_library(
    name = "banded_solver_backend",
    hdrs = ["banded_solver_backend.hpp"],
    visibility = ["//visibility:public"],
    strip_include_prefix = "/src/math",
    include_prefix = "mango/math",
)
```

and add `":banded_solver_backend"` to `banded_matrix_solver`'s deps.
- [ ] **Step 4: Verify** — `bazel test //tests:bspline_banded_solver_test //tests:bspline_collocation_1d_test //tests:bspline_bit_identity_test --test_output=errors` all PASS; `bazel build //src/...` clean.
- [ ] **Step 5: Commit** — `git commit -m "Extract BandedResult and BandedSolverBackend concept"`.

---

### Task 2: LapackBandedBackend

**Files:**
- Create: `src/math/lapack_banded_backend.hpp`
- Create: `tests/lapack_banded_backend_test.cc`
- Modify: `src/math/BUILD.bazel`, `tests/BUILD.bazel`

**Interfaces:**
- Consumes: Task 1's header.
- Produces: `mango::LapackBandedBackend` with exactly these members (Task 3 consumes them verbatim):
  - `using pivot_type = lapack_int;`
  - `static constexpr std::size_t factor_storage_size(std::size_t n, std::size_t bandwidth)` → `(3 * (bandwidth - 1) + 1) * n`
  - `static constexpr std::size_t pivot_storage_size(std::size_t n, std::size_t)` → `n`
  - `static void pack(std::span<const double> band_rows, std::span<const int> col_start, std::size_t n, std::size_t bandwidth, std::span<double> factors)`
  - `static BandedResult<double> factorize(std::span<double> factors, std::span<lapack_int> pivots, std::size_t n, std::size_t bandwidth)`
  - `static BandedResult<double> solve(std::span<const double> factors, std::span<const lapack_int> pivots, std::span<double> x, std::size_t n, std::size_t bandwidth)`
  - `static double condition(std::span<const double> factors, std::span<const lapack_int> pivots, double norm, std::size_t n, std::size_t bandwidth)`

- [ ] **Step 1: Write the failing test** (`tests/lapack_banded_backend_test.cc`, SPDX + regression header citing #464):

```cpp
// Regression: LAPACK calls and banded-layout knowledge leaked above the
// solver layer (issue #464)
// Bug: bspline_collocation.hpp held six direct LAPACKE_* call sites plus
// layout math; the backend concept makes the library swappable.
static_assert(mango::BandedSolverBackend<mango::LapackBandedBackend, double>);
static_assert(!mango::BandedSolverBackend<mango::LapackBandedBackend, float>);
static_assert(!mango::BandedSolverBackend<int, double>);

TEST(LapackBandedBackendTest, PackMatchesLapackBandedLayout) {
    // 5-point cubic band (bandwidth 4): row i covers cols [col_start[i], ...)
    constexpr std::size_t n = 5, bw = 4;
    std::vector<double> band_rows(n * bw);
    std::iota(band_rows.begin(), band_rows.end(), 1.0);   // distinct values
    std::vector<int> col_start{0, 0, 0, 1, 1};
    std::vector<double> factors(
        mango::LapackBandedBackend::factor_storage_size(n, bw), -7.0);
    mango::LapackBandedBackend::pack(band_rows, col_start, n, bw, factors);
    // Expected offsets from the documented LAPACK formula
    // (kl+ku+i-j) + j*ldab with kl=ku=3, ldab=10:
    for (std::size_t i = 0; i < n; ++i) {
        const std::size_t j_end = std::min(static_cast<std::size_t>(col_start[i]) + bw, n);
        for (std::size_t j = col_start[i]; j < j_end; ++j) {
            const std::size_t off = (6 + i - j) + j * 10;
            EXPECT_EQ(factors[off], band_rows[i * bw + (j - col_start[i])])
                << "i=" << i << " j=" << j;
        }
    }
    // Everything outside the band was zeroed, not left at the sentinel
    EXPECT_EQ(std::count(factors.begin(), factors.end(), -7.0), 0);
}

TEST(LapackBandedBackendTest, FactorizeSolveRoundTrip) {
    // Identity-band system: diagonal-only band solves x = b exactly
    constexpr std::size_t n = 6, bw = 4;
    std::vector<double> band_rows(n * bw, 0.0);
    std::vector<int> col_start(n);
    for (std::size_t i = 0; i < n; ++i) {
        col_start[i] = static_cast<int>(i > 2 ? i - 2 : 0);
        band_rows[i * bw + (i - col_start[i])] = 2.0;   // diag = 2
    }
    std::vector<double> factors(mango::LapackBandedBackend::factor_storage_size(n, bw));
    std::vector<lapack_int> pivots(mango::LapackBandedBackend::pivot_storage_size(n, bw));
    mango::LapackBandedBackend::pack(band_rows, col_start, n, bw, factors);
    ASSERT_TRUE(mango::LapackBandedBackend::factorize(factors, pivots, n, bw).ok());
    std::vector<double> x{2.0, 4.0, 6.0, 8.0, 10.0, 12.0};
    ASSERT_TRUE(mango::LapackBandedBackend::solve(factors, pivots, x, n, bw).ok());
    for (std::size_t i = 0; i < n; ++i) EXPECT_DOUBLE_EQ(x[i], (i + 1) * 1.0);
    EXPECT_GT(mango::LapackBandedBackend::condition(factors, pivots, 2.0, n, bw), 0.0);
}
```

Includes: `<gtest/gtest.h>`, `<algorithm>`, `<numeric>`, `<vector>`, `"mango/math/lapack_banded_backend.hpp"`.

- [ ] **Step 2: BUILD for the test** (fails to build yet — header missing):

```python
cc_test(
    name = "lapack_banded_backend_test",
    size = "small",
    srcs = ["lapack_banded_backend_test.cc"],
    deps = [
        "//src/math:lapack_banded_backend",
        "@googletest//:gtest_main",
    ],
    copts = ["-Wall", "-Wextra"],
)
```

- [ ] **Step 3: Implement the header.** SPDX; includes `"mango/math/banded_solver_backend.hpp"`, `"mango/math/lapack_banded_layout.hpp"`, `<experimental/mdspan>`, `<algorithm>`, `<cstddef>`, `<limits>`, `<span>`, `<lapacke.h>`. Members per the Interfaces block. Bodies transplant the current logic verbatim:
  - `pack`: zero `factors`, then the `fill_lapack_band` loop from `bspline_collocation.hpp` (mdspan over `lapack_banded_layout::mapping` with `dextents<std::size_t, 2>{n, n}`, `kl = ku = bandwidth − 1`), reading `band_rows[i * bandwidth + (j − col_start[i])]`.
  - `factorize`: `LAPACKE_dgbtrf(LAPACK_COL_MAJOR, n, n, kl, ku, factors.data(), ldab, pivots.data())`; `info < 0` → `error_result("LAPACKE_dgbtrf: invalid argument")`, `info > 0` → `error_result("Matrix is singular")`.
  - `solve`: `LAPACKE_dgbtrs(LAPACK_COL_MAJOR, 'N', n, kl, ku, 1, factors.data(), ldab, pivots.data(), x.data(), n)`; errors `"LAPACKE_dgbtrs: invalid argument"` / `"LAPACKE_dgbtrs: zero pivot"`.
  - `condition`: return `infinity` if `norm == 0`; `LAPACKE_dgbcon(..., '1', ...)`; `info != 0 || rcond == 0` → infinity; else `1.0 / rcond`.
  - All `lapack_int` casts happen here (`static_cast<lapack_int>(n)` etc.); `ldab = 3 * (bandwidth − 1) + 1` computed locally.
- [ ] **Step 4: BUILD for the library:**

```python
cc_library(
    name = "lapack_banded_backend",
    hdrs = ["lapack_banded_backend.hpp"],
    deps = [
        ":banded_solver_backend",
        ":lapack_banded_layout",
        "@mdspan//:mdspan",
    ],
    linkopts = ["-llapacke"],
    visibility = ["//visibility:public"],
    strip_include_prefix = "/src/math",
    include_prefix = "mango/math",
)
```

- [ ] **Step 5: Run** — `bazel test //tests:lapack_banded_backend_test --test_output=errors` PASSES.
- [ ] **Step 6: Commit** — `git commit -m "Add LapackBandedBackend policy type"`.

---

### Task 3: Thread Backend through the collocation layer

**Files:**
- Modify: `src/math/bspline/bspline_collocation.hpp` (the bulk)
- Modify: `src/math/bspline/bspline_collocation_workspace.hpp` (pivot vocabulary)
- Modify: `src/math/bspline/BUILD.bazel`

**Interfaces:**
- Consumes: Tasks 1–2 exactly as produced.
- Produces: `BSplineCollocation1D<T, Bandwidth = 4, Backend = LapackBandedBackend> requires BandedSolverBackend<Backend, T>`; `BSplineCollocationFactorization<T, Backend = LapackBandedBackend>` with `std::vector<typename Backend::pivot_type> pivots`; `factorize()` → `std::expected<BSplineCollocationFactorization<T, Backend>, InterpolationError>`; `solve_factored(const BSplineCollocationFactorization<T, Backend>&, ...)`.

- [ ] **Step 1: Retarget the templates.**
  - `BSplineCollocationFactorization`: add `typename Backend = LapackBandedBackend` param + `requires BandedSolverBackend<T ordering: (Backend, T)>` clause; `pivots` becomes `std::vector<typename Backend::pivot_type>`; doc comment notes the representation is backend-defined.
  - `BSplineCollocation1D`: `template<std::floating_point T, std::size_t Bandwidth = 4, typename Backend = LapackBandedBackend> requires BandedSolverBackend<Backend, T>`.
- [ ] **Step 2: Re-express the factor-once API.**
  - `factorize()`: sizes via `Backend::factor_storage_size(n_, BANDWIDTH)` / `Backend::pivot_storage_size(n_, BANDWIDTH)`; `Backend::pack(band_values_, band_col_start_, n_, BANDWIDTH, fact.lu)` (note: `band_col_start_` is `std::vector<int>` — passes as `std::span<const int>`); `Backend::factorize(...)` failure → `InterpolationError{FittingFailed, n_}`; `compute_matrix_norm1()`; `fact.condition_estimate = Backend::condition(...)`. Sequence identical to current code.
  - `solve_factored()`: replace the two hand-computed size checks with `fact.lu.size() != Backend::factor_storage_size(n_, BANDWIDTH)` and `fact.pivots.size() != Backend::pivot_storage_size(n_, BANDWIDTH)` (same `BufferSizeMismatch` payloads: `fact.lu.size()` / `fact.pivots.size()`); keep the values/coeffs size checks, uintptr_t aliasing check, and NaN/Inf loop untouched; `Backend::solve` failure → `FittingFailed`; residual/tolerance unchanged.
- [ ] **Step 3: Re-express `fit()` and `fit_with_buffer()` on backend primitives** — replacing the `BandedMatrix`/`BandedLUWorkspace` path, in this exact order (Global Constraint):

```cpp
// fit_with_buffer core (fit() delegates identically into a local vector):
std::vector<T> factors(Backend::factor_storage_size(n_, BANDWIDTH));
std::vector<typename Backend::pivot_type> pivots(Backend::pivot_storage_size(n_, BANDWIDTH));
Backend::pack(std::span<const T>{band_values_}, std::span<const int>{band_col_start_},
              n_, BANDWIDTH, std::span<T>{factors});
if (!Backend::factorize(factors, pivots, n_, BANDWIDTH).ok()) {
    return std::unexpected(InterpolationError{InterpolationErrorCode::FittingFailed, n_});
}
std::copy(values.begin(), values.end(), coeffs_out.begin());
if (!Backend::solve(factors, pivots, coeffs_out, n_, BANDWIDTH).ok()) {
    return std::unexpected(InterpolationError{InterpolationErrorCode::FittingFailed, n_});
}
const T max_residual = compute_residual_from_span(coeffs_out, values);
// tolerance check unchanged, then:
const T norm_A = compute_matrix_norm1();
const T cond_est = Backend::condition(factors, pivots, norm_A, n_, BANDWIDTH);
```

`fit()` keeps its `std::vector<T> coeffs(n_)` + moves it into the result; both keep their existing validation preambles and error payloads verbatim.
- [ ] **Step 4: Re-express `fit_with_workspace()`** with the member constraint `requires std::same_as<Backend, LapackBandedBackend>` (trailing requires clause on the method): `Backend::pack` into `ws.band_storage()` (replacing `build_collocation_matrix_to_workspace`; keep that method as a one-line `Backend::pack` wrapper since it's called there only), copy band→`ws.lapack_storage()`, `Backend::factorize(ws.lapack_storage(), ws.pivots(), n_, BANDWIDTH)`, copy values→`ws.coeffs()`, `Backend::solve(ws.lapack_storage(), ws.pivots(), ws.coeffs(), n_, BANDWIDTH)`, residual from `ws.coeffs()`, `compute_matrix_norm1()`, `Backend::condition(ws.lapack_storage(), ws.pivots(), norm, n_, BANDWIDTH)`. Same error mappings as today.
- [ ] **Step 5: Delete the absorbed privates** — `fill_lapack_band`, `factorize_banded_workspace`, `solve_banded_workspace`, `estimate_banded_condition_workspace`, `estimate_condition_from`. Keep `compute_residual_from_span`, `compute_residual`, `compute_matrix_norm1`, `build_collocation_matrix`.
- [ ] **Step 6: Include + BUILD hygiene.**
  - `bspline_collocation.hpp`: drop `<lapacke.h>`, `"mango/math/lapack_banded_layout.hpp"`, `"mango/math/banded_matrix_solver.hpp"`; add `"mango/math/banded_solver_backend.hpp"`, `"mango/math/lapack_banded_backend.hpp"`.
  - `bspline_collocation_workspace.hpp`: replace `#include <lapacke.h>` with `"mango/math/lapack_banded_backend.hpp"` and every raw `lapack_int` with `LapackBandedBackend::pivot_type` (or a local `using pivot_type = LapackBandedBackend::pivot_type;`).
  - `src/math/bspline/BUILD.bazel`: `bspline_collocation` deps swap `"//src/math:banded_matrix_solver"` → `"//src/math:banded_solver_backend"` + `"//src/math:lapack_banded_backend"`, and DELETE its `linkopts = ["-llapacke"]`; `bspline_collocation_workspace` adds `"//src/math:lapack_banded_backend"`.
- [ ] **Step 7: Leak check** — `grep -c "LAPACKE_\|lapacke.h" src/math/bspline/bspline_collocation.hpp` prints 0.
- [ ] **Step 8: Run the proof** — `bazel test //tests:bspline_bit_identity_test //tests:bspline_collocation_1d_test //tests:bspline_fit_with_workspace_test //tests:bspline_collocation_workspace_test //tests:bspline_fitter_4d_separable_test //tests:lapack_banded_backend_test --test_output=errors` — ALL PASS (goldens bit-identical). `bazel build //src/...` clean.
- [ ] **Step 9: Commit** — `git commit -m "Route B-spline collocation through the banded backend concept"` (body: closes the six-call-site LAPACK leak; fixes #464).

---

### Task 4: Full verification gate

- [ ] **Step 1:** `bazel test //...` — green (137-test baseline).
- [ ] **Step 2:** `bazel build //benchmarks/...`, `bazel build //src/python:mango_option`, `bazel build //crates/mango-option:mango_option` — all succeed.
- [ ] **Step 3:** Discard `MODULE.bazel.lock` churn; commit stragglers if any. Branch ready for holistic review + PR.
