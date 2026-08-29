# Parallel-Safety Fixes: Issues #435 and #436 — Design

**Date:** 2026-08-29
**Issues:** #435 (OpenMP data race + per-slice LU refactorization in B-spline
collocation fitting), #436 (AmericanOptionResult documents const methods as
thread-safe but lazily mutates without synchronization)
**Branch:** `fix/435-436-parallel-safety` (off main 393db1e2)

## Problem statements

### Issue #436 — AmericanOptionResult lazy mutation under a thread-safety promise

`src/option/american_option_result.hpp:35` documents "Const methods are
thread-safe (read-only access)", but the const accessors lazily mutate:

- `value()` / `value_at()` / `delta()` / `theta()` call `ensure_spline()`,
  which builds a `mutable CubicSpline<double> spline_` guarded only by a
  plain `mutable bool spline_built_`.
- `gamma()` calls `ensure_operator()`, which lazily constructs
  `mutable std::unique_ptr<CenteredDifference<double>> operator_`.

Two threads calling `value()` concurrently both observe
`spline_built_ == false` and both run `spline_.build()` on the same object:
a data race (UB), torn coefficients, or a crash — under exactly the usage
the doc invites.

### Issue #435 — shared-solver mutation inside the OpenMP fit loop

`BSplineNDSeparable::fit_axis` (`src/math/bspline/bspline_nd_separable.hpp:352`)
calls `solvers_[Axis]->fit_with_workspace(...)` from every OpenMP thread on a
**shared** `BSplineCollocation1D` instance. Two defects:

1. **Data race (UB):** `fit_with_workspace` →
   `build_collocation_matrix_to_workspace(ws)` → `build_collocation_matrix()`,
   which writes the shared members `band_values_` / `band_col_start_` while
   other threads concurrently read them (`compute_residual_from_span`,
   `compute_matrix_norm1`). All threads write identical values, so it is
   benign on x86 in practice, but it is UB per the C++ memory model and a
   TSan finding.
2. **Redundant work:** for every slice along an axis the code rebuilds the
   identical collocation matrix, re-runs `dgbtrf`, and re-runs `dgbcon` —
   all functions of the grid alone. A 50×30×20×10 table fitting axis 0
   performs ~6,000 redundant builds + factorizations + condition estimates
   (each with LAPACKE-internal heap allocation, inside the parallel region)
   where one per axis suffices. `compute_matrix_norm1` also heap-allocates
   per slice.

## Decisions

Settled with the user at triage (2026-08-29):

1. **Packaging: one branch / one PR for both issues.** Two independent Major
   code-review bugs with no file overlap; one review cycle covers both
   (precedent: the #441 latent-traps batch, PR #456).
2. **#436 fix: eager construction.** Build the spline and the gamma operator
   in the constructor; delete `mutable`, the lazy flags, and `ensure_*()`.
   Rationale: cost is one O(n) tridiagonal solve plus one `CenteredDifference`
   construction (microseconds) against the ≥0.3ms PDE solve that necessarily
   precedes every result; it keeps the documented thread-safety promise with
   zero synchronization machinery. Alternatives rejected: `std::call_once`
   (a bare `std::once_flag` member would break the class's movability —
   `solve_american_option` returns the result through `std::expected` — so it
   needs a `unique_ptr<once_flag>` or hand-written moves, i.e. more code for
   the same guarantee) and downgrading the doc (leaves the multi-threaded
   read usage broken).
3. **#435 fix: factor once per axis, solve per slice** (the fix suggested in
   the issue): build the collocation matrix and LU-factorize once, before
   the parallel region; threads run only `dgbtrs` + the residual check
   against shared read-only state. Made structurally race-proof by giving
   the per-slice call a `const` receiver and a `const` factorization — the
   compiler enforces what the OpenMP contract needs.

## Design

### Part 1 — AmericanOptionResult (#436)

`src/option/american_option_result.{hpp,cpp}`:

- Constructor gains two lines: `build_spline(spline_, grid_->solution())`
  and `operator_ = std::make_unique<operators::CenteredDifference<double>>(grid_->spacing())`.
- Members lose `mutable`: `CubicSpline<double> spline_;`
  `std::unique_ptr<operators::CenteredDifference<double>> operator_;`.
  `spline_built_` is deleted. (`operator_` stays a `unique_ptr` value —
  cheapest way to keep the defaulted moves.)
- `ensure_spline()` / `ensure_operator()` are deleted; call sites drop the
  calls. `theta()`'s local `prev_spline` is already function-local and
  stays as is.
- The thread-safety doc comment stays, now true. Add a line noting the
  spline and gamma operator are built at construction.

No public API change. Behavior change: construction cost rises by O(n)
(microseconds, dwarfed by the PDE solve that produces the grid); accessors
lose their first-call spike. `assert(grid_)` already guards the null case.

**Thread-safety contract (design-review round 1):** the object is not
*deep*-const — it shares a caller-supplied `shared_ptr<Grid<double>>`,
`grid()` hands that mutable pointer back out, and `value_at`'s boundary
branch, `gamma()`, and `theta()`'s previous-step half read live grid
storage. The documented guarantee is therefore: *const methods are
thread-safe provided no thread mutates the underlying grid concurrently*
(the doc comment says exactly that). A second recorded consequence of
eager construction: the spline snapshots `grid_->solution()` at
construction time, so mutating the grid between construction and first
access no longer changes `value_at`/`delta` output the way it did under
lazy init. Both facts go in the class doc comment. Deep constness would
need an ownership redesign — out of scope.

### Part 2 — BSplineCollocation1D (#435)

`src/math/bspline/bspline_collocation.hpp` (header-only class):

**2a. Build the collocation matrix once, in the constructor.**
The private constructor already sizes `band_values_` / `band_col_start_`;
it now also calls `build_collocation_matrix()`. The matrix depends only on
the grid, so members become logically immutable after construction. All
`fit*` paths drop their `build_collocation_matrix()` calls, and
`build_collocation_matrix_to_workspace` degenerates to a `const` copy of
the prebuilt band into the workspace's LAPACK layout. This alone removes
the UB: no shared-state writes remain anywhere in the fit paths, so even
the legacy `fit_with_workspace` becomes safe to call concurrently
(each thread's workspace is private).

**2b. Explicit factorization object + const per-slice solve.**

```cpp
/// LU factorization of the collocation matrix (grid-dependent only).
/// Produced once per axis; read-only during parallel solves.
template<std::floating_point T>
struct BSplineCollocationFactorization {
    std::vector<T> lu;                  ///< LDAB×n LAPACK banded LU factors
    std::vector<lapack_int> pivots;     ///< n pivot indices from dgbtrf
    T condition_estimate;               ///< From dgbcon (1-norm), once
};

/// Factorize the collocation matrix (dgbtrf + dgbcon). O(n).
[[nodiscard]] std::expected<BSplineCollocationFactorization<T>, InterpolationError>
factorize() const;

/// Solve B·c = values against a prebuilt factorization (dgbtrs) and
/// verify the residual. Writes coefficients to coeffs_out. Thread-safe:
/// no solver or factorization state is written.
[[nodiscard]] std::expected<T /*max_residual*/, InterpolationError>
solve_factored(const BSplineCollocationFactorization<T>& fact,
               std::span<const T> values,
               std::span<T> coeffs_out,
               const BSplineCollocationConfig<T>& config = {}) const;
```

- `factorize()` copies the prebuilt band into a local LDAB×n buffer, runs
  `LAPACKE_dgbtrf`, computes `compute_matrix_norm1()` (local accumulator,
  no member writes), runs `LAPACKE_dgbcon`, and returns the owning struct.
  Errors map to `InterpolationErrorCode::FittingFailed` exactly as today.
- `solve_factored()` validates sizes and NaN/Inf (same checks and error
  codes as `fit_with_workspace`), copies `values` into `coeffs_out`
  (`dgbtrs` solves in place), runs `LAPACKE_dgbtrs` against
  `fact.lu`/`fact.pivots`, computes the residual with the existing
  `compute_residual_from_span` (now `const`-clean reads of the prebuilt
  band), and enforces `config.tolerance`. Returns the max residual.
- **Factorization validity invariants (design-review round 1):** the
  aggregate is decoupled from its producing solver, so `solve_factored()`
  must reject malformed input before any LAPACK call:
  `fact.lu.size() != LDAB * n_` or `fact.pivots.size() != n_` returns
  `InterpolationErrorCode::BufferSizeMismatch`. This blocks
  default-constructed or truncated factorizations (out-of-bounds LAPACK
  access). A *same-sized* factorization from a different solver with the
  same n is not detectable at this layer and remains caller
  responsibility, stated in the doc comment (the sole production caller,
  `fit_axis`, creates and consumes the pair locally). Full provenance
  coupling (opaque type keyed to the solver) was considered and rejected
  as machinery the one caller doesn't need.
- `fit()` / `fit_with_buffer()` / `fit_with_workspace()` remain, with
  unchanged signatures and semantics minus the redundant rebuilds; they
  are kept because tests and `thread_workspace.hpp` docs reference them.

**2c. `BSplineNDSeparable::fit_axis` restructure**
(`src/math/bspline/bspline_nd_separable.hpp`):

```cpp
auto fact = solvers_[Axis]->factorize();       // once, before the parallel region
if (!fact) {
    failed[Axis] = n_slices;                    // whole axis fails: one matrix
    max_residuals[Axis] = conditions[Axis] = T{0};
    return;
}
MANGO_PRAGMA_PARALLEL
{
    std::vector<T> slice_buffer(n_axis), coeff_buffer(n_axis);
    // per-thread residual max + failed count as today
    MANGO_PRAGMA_FOR
    for (slices) {
        extract slice → slice_buffer;
        auto r = solvers_[Axis]->solve_factored(*fact, slice_buffer,
                                                coeff_buffer, {.tolerance = tolerance});
        r ? (track residual, write back coeff_buffer) : ++local_failed;
    }
    // critical-section reduction unchanged
}
conditions[Axis] = fact->condition_estimate;    // one estimate, not a per-slice max
```

- `ThreadWorkspaceBuffer` + `BSplineCollocationWorkspace` disappear from
  this loop (two small per-thread `std::vector`s replace them; allocation
  once per thread, same as the old buffer).
- Per-slice LAPACKE heap allocation (`dgbtrf`/`dgbcon` internals) is gone.
- **Semantics change (deliberate):** a singular collocation matrix now
  fails the whole axis up front (`failed[Axis] = n_slices`) instead of
  failing each slice identically inside the loop — same observable
  outcome for callers, which only test `failed > 0`.
- `conditions[Axis]` was previously the max over per-slice estimates of
  the *same matrix* (identical values); one estimate is the same number.

**Numerical identity:** the per-slice LAPACK call sequence on each slice's
data is unchanged (`dgbtrf` on the identical matrix produces identical LU
factors; `dgbtrs` then sees bit-identical inputs), so fitted coefficients
are expected bit-for-bit identical to today's output. Because no existing
test pins coefficient *contents* (the 4D separable test checks success and
count only, and table-level comparisons carry tolerances), the claim is
**proved by a golden test written against the current implementation
before the refactor** (see Testing item 0) rather than asserted.

## Error handling

- `factorize()`: `FittingFailed{n}` on dgbtrf failure (matches today's
  mapping); dgbcon failure degrades to `condition_estimate = inf` exactly
  as `estimate_banded_condition_workspace` does now.
- `solve_factored()`: `ValueSizeMismatch` / `BufferSizeMismatch` /
  `NaNInput` / `InfInput` / `FittingFailed`(residual or dgbtrs) — the same
  codes the current fit paths produce for the same conditions.
- No new error codes anywhere.

## Testing

Per CLAUDE.md, every bug gets a regression test with the standard header
comment naming the bug.

0. **Bit-identity golden (written FIRST, against the unmodified code):**
   a new test fits a small fixed nonuniform 3D grid via
   `BSplineNDSeparable` and a 1D grid via `fit()`, and pins **every**
   resulting coefficient, compared as bit patterns
   (`std::bit_cast<uint64_t>` — exact even for signed zero) against
   golden values hardcoded from a run of the *current* implementation.
   Committed green before any #435 refactor commit; the refactor must
   keep it green, which proves the bit-for-bit claim against a true
   pre-change baseline. (Round 2: a sampling would let untested
   coefficients drift; `EXPECT_EQ` on doubles misses −0.0 vs +0.0.)
1. **#436 regression** (`tests/american_option_test.cc`): compute expected
   values (`value_at`/`delta`/`gamma`/`theta`) from a *separate* result
   solved with identical params, then construct a **fresh** result and
   race the accessors' *first* calls from ~8 threads; assert exact
   equality with the expected values. The two lazy resources are
   independent (`ensure_spline` vs `ensure_operator`), so each gets its
   own synchronized first-call phase: a reusable `std::barrier` is
   re-armed immediately before the spline phase (`value_at` as every
   thread's first call) and again before the operator phase — run on a
   *second* fresh result with `gamma()` as every thread's first call, so
   the operator race isn't serialized by threads arriving late from the
   spline phase. On the old code each phase concurrently runs its lazy
   build — the actual race; warming the caches first would make the
   threaded phase read-only and prove nothing.
2. **#435 equivalence** (`tests/bspline_collocation_test.cc` or the
   workspace test file): `factorize()` + `solve_factored()` must reproduce
   `fit()` coefficients bit-for-bit on a nonuniform grid; error paths
   (value-size mismatch, coeffs-size mismatch, NaN/Inf input) return the
   same codes as the fit paths.
3. **#435 malformed-factorization guard:** `solve_factored()` with a
   default-constructed factorization, and with truncated `lu` / `pivots`,
   returns `BufferSizeMismatch` and never reaches LAPACK.
4. **#435 concurrent solve:** one shared solver + one shared factorization,
   ~8 threads each solving a different slice's values concurrently; results
   equal the serial answers exactly.
5. **Existing coverage:** `bspline_fitter_4d_separable_test.cc` and the
   table-builder suites must pass unchanged (success/shape and
   tolerance-level checks; the exactness burden is carried by item 0).

TSan is not in CI; the structural fix (const receiver + const
factorization, matrix built in the constructor) is the guarantee, and the
concurrency tests document the contract.

## Acceptance criteria

- `bazel test //...` green.
- No `mutable` members remain in `AmericanOptionResult`; no member writes
  remain in any `BSplineCollocation1D` method after construction.
- `fit_axis` performs exactly one `dgbtrf` and one `dgbcon` per axis.
- Existing fitted-value tests pass without tolerance changes.

## Out of scope

- Multi-RHS batching (`dgbtrs` with `nrhs > 1` to solve many slices per
  call) — a further optimization on top of this structure; file as a
  follow-up if profiling justifies it.
- A TSan CI job (belongs with the #444 test-gap work).
- The B-spline surface/table layers above `BSplineNDSeparable` — untouched.
