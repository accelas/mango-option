# Banded-Solver Backend Seam (Issue #464) — Design

**Date:** 2026-08-30
**Issue:** #464 — hide the banded-solver backend behind a concept (LAPACK as
implementation detail, Eigen-ready)
**Branch:** `fix/464-banded-backend-seam` (off main 8b507fec)
**History:** drafted as an amendment to PR #457's spec, split out at the
owner's direction; one Codex design-review round already ran against the
amendment — its two [P1]s (concept syntax; missing pivot-sizing contract)
and three [P2]s are folded in below.

## Problem

LAPACK is meant to be an implementation detail of the banded-solver layer,
and a planned experiment replaces it with Eigen for banded fitting (a
worktree for it already exists: `.worktrees/migrate-lapack-eigen`). After
PR #457, `src/math/bspline/bspline_collocation.hpp` carries six direct
`LAPACKE_*` call sites — three on the legacy workspace path
(`factorize_banded_workspace`, `solve_banded_workspace`,
`estimate_banded_condition_workspace`) and three from the factor-once API
(`factorize()`, `solve_factored()`, `estimate_condition_from()`) — plus
direct knowledge of the LAPACK banded layout (`fill_lapack_band`,
`lapack_banded_layout.hpp`) and an `<lapacke.h>` include. Every additional
call site raises the cost of the backend swap.

## Decisions (user-settled 2026-08-30)

1. **Concept-based seam**, not just free-function centralization: the
   collocation solver is templated on a backend policy type constrained by
   a C++20 concept. Rationale: an Eigen backend differs not only in calls
   but in factor-storage layout and pivot representation, so the seam must
   let the backend own all three.
2. **Backends are stateless tag types with static members.** No dynamic
   dispatch anywhere (user constraint: nothing `std::function`-shaped in
   hot paths; `<functional>` stays out of these headers).
3. **LAPACK remains the only shipped backend**; the Eigen experiment
   happens later against the seam. Nothing in this change may alter
   numerical output: the `//tests:bspline_bit_identity_test` goldens
   pinned in #457 are the acceptance proof, at zero tolerance.

## Design

### New header `src/math/banded_solver_backend.hpp` (backend-agnostic)

No LAPACK includes. `BandedResult<T>` **moves here** (it is
backend-vocabulary, not LAPACK-vocabulary); `banded_matrix_solver.hpp`
includes this header so its own API and users are unaffected.

```cpp
/// A banded linear-solver backend: a stateless policy type owning its
/// factor-storage layout, its pivot representation, its sizing, and every
/// call into the underlying linear-algebra library.
///
/// `band_rows` is the solver's neutral band form: n×bandwidth row-major
/// values plus a per-row first-column index (exactly what
/// BSplineCollocation1D::build_collocation_matrix produces).
template<typename B, typename T>
concept BandedSolverBackend =
    std::floating_point<T> &&
    requires(std::span<const T> band_rows, std::span<const int> col_start,
             std::span<T> factors, std::span<typename B::pivot_type> pivots,
             std::span<const T> cfactors,
             std::span<const typename B::pivot_type> cpivots,
             std::span<T> x, T norm, std::size_t n, std::size_t bandwidth)
{
    typename B::pivot_type;
    { B::factor_storage_size(n, bandwidth) } -> std::convertible_to<std::size_t>;
    { B::pivot_storage_size(n, bandwidth) } -> std::convertible_to<std::size_t>;
    { B::pack(band_rows, col_start, n, bandwidth, factors) };
    { B::factorize(factors, pivots, n, bandwidth) } -> std::same_as<BandedResult<T>>;
    { B::solve(cfactors, cpivots, x, n, bandwidth) } -> std::same_as<BandedResult<T>>;
    { B::condition(cfactors, cpivots, norm, n, bandwidth) } -> std::convertible_to<T>;
};
```

- `pivot_storage_size` exists because "pivots are n entries" is a LAPACK
  fact, not a universal one (review [P1]): owning factorizations size
  their pivot vector through the backend, never by assuming `n`.
- `pack` writes the backend's factor-storage layout from the neutral band;
  the layout is thereby private to the backend.
- `factorize` factors **in place** in `factors` (which `pack` filled);
  `solve` solves in place in `x` (caller pre-loads the RHS); `condition`
  returns infinity on failure or zero norm.

### New header `src/math/lapack_banded_backend.hpp`

```cpp
struct LapackBandedBackend {
    using pivot_type = lapack_int;

    static constexpr std::size_t factor_storage_size(std::size_t n, std::size_t bandwidth)
    { return (3 * (bandwidth - 1) + 1) * n; }        // ldab*n, kl=ku=bandwidth-1
    static constexpr std::size_t pivot_storage_size(std::size_t n, std::size_t)
    { return n; }

    static void pack(std::span<const double> band_rows, std::span<const int> col_start,
                     std::size_t n, std::size_t bandwidth, std::span<double> factors);
    // zero factors; for each row i and j in [col_start[i],
    // min(col_start[i]+bandwidth, n)):
    //   factors[lapack_banded_layout offset (i,j)] = band_rows[i*bandwidth + (j-col_start[i])]
    // — byte-identical to the former fill_lapack_band / BandedMatrix fill.

    static BandedResult<double> factorize(std::span<double> factors,
                                          std::span<lapack_int> pivots,
                                          std::size_t n, std::size_t bandwidth);   // dgbtrf
    static BandedResult<double> solve(std::span<const double> factors,
                                      std::span<const lapack_int> pivots,
                                      std::span<double> x, std::size_t n,
                                      std::size_t bandwidth);                      // dgbtrs
    static double condition(std::span<const double> factors,
                            std::span<const lapack_int> pivots,
                            double norm, std::size_t n, std::size_t bandwidth);    // dgbcon
};
```

**The members are plain `double` functions, not templates** (round 2
[P2] fold): a requires-expression never instantiates bodies, so a
templated member with a body-level `static_assert(same_as<T, double>)`
would make `BandedSolverBackend<LapackBandedBackend, float>` *true* and
admit `BSplineCollocation1D<float>` only to explode at call time. With
concrete `double` signatures the concept check fails naturally for any
other `T` — the constraint tells the truth, and the old `static_assert`s
disappear. `pack` uses
`lapack_banded_layout::mapping` over `dextents<std::size_t, 2>{n, n}` with
`kl = ku = bandwidth − 1` — the same single source of layout truth as
today. Error strings and info-code handling are copied verbatim from the
current call sites ("LAPACKE_dgbtrf: invalid argument", "Matrix is
singular", "LAPACKE_dgbtrs: invalid argument"/"zero pivot"). This header
and `banded_matrix_solver.hpp` become the only `<lapacke.h>` includes on
the fitting path. LAPACK size params are `static_cast<lapack_int>` inside
the backend.

### Threading through `bspline_collocation.hpp`

```cpp
template<std::floating_point T, std::size_t Bandwidth = 4,
         typename Backend = LapackBandedBackend>
    requires BandedSolverBackend<Backend, T>
class BSplineCollocation1D;

template<std::floating_point T, typename Backend = LapackBandedBackend>
    requires BandedSolverBackend<Backend, T>
struct BSplineCollocationFactorization {
    std::vector<T> lu;                              ///< backend factor storage
    std::vector<typename Backend::pivot_type> pivots;
    T condition_estimate;
};
```

(Review [P1] fold: `typename Backend`, constrained via a `requires`
clause; `std::floating_point<T>` with angle brackets in the concept. All
existing `BSplineCollocation1D<double>` /
`BSplineCollocationFactorization<double>` spellings compile unchanged via
the defaults.)

- **The Backend parameter threads through the factor-once API** (round 2
  [P1] fold — without this a custom backend would silently get the LAPACK
  factorization type and its pivot representation):

  ```cpp
  [[nodiscard]] std::expected<BSplineCollocationFactorization<T, Backend>,
                              InterpolationError>
  factorize() const;

  [[nodiscard]] std::expected<T, InterpolationError> solve_factored(
      const BSplineCollocationFactorization<T, Backend>& fact,
      std::span<const T> values, std::span<T> coeffs_out,
      const BSplineCollocationConfig<T>& config = {}) const;
  ```

- `factorize()` sizes `fact.lu` by `Backend::factor_storage_size(n_,
  BANDWIDTH)` and `fact.pivots` by `Backend::pivot_storage_size(n_,
  BANDWIDTH)`; body becomes `Backend::pack` → `Backend::factorize` →
  `compute_matrix_norm1()` → `Backend::condition`. (Sequence unchanged
  from the merged code, so bits are unchanged.)
- `solve_factored()` validates sizes against the two backend size
  functions (same `BufferSizeMismatch` payloads and ordering as today,
  aliasing check unchanged), then `Backend::solve` + residual.
- `fit()` / `fit_with_buffer()` are re-expressed **directly on backend
  primitives in the existing LAPACK call order**:
  `pack → factorize (dgbtrf) → copy RHS → solve (dgbtrs) → residual →
  norm → condition (dgbcon)` — using a local factor buffer + pivot vector.
  **Not** as public `factorize()` followed by `solve_factored()`: public
  `factorize()` runs `condition` (dgbcon) immediately, which would reorder
  the LAPACK sequence to dgbtrf→dgbcon→dgbtrs and void the bit-identity
  claim (review [P2] — load-bearing). This retires the
  `BandedMatrix`/`BandedLUWorkspace` usage (and the
  `banded_matrix_solver.hpp` include) from this header; that legacy API
  itself stays for its other users (tests).
- `fit_with_workspace()` keeps its byte-identical flow via the backend:
  `Backend::pack` into `ws.band_storage()`, copy to `ws.lapack_storage()`,
  `Backend::factorize` on it with `ws.pivots()`, copy RHS into
  `ws.coeffs()`, `Backend::solve`, residual, norm, `Backend::condition`.
  It is constrained with a **member `requires` clause**, not a body
  `static_assert`:

  ```cpp
  [[nodiscard]] std::expected<BSplineCollocationResult<T>, InterpolationError>
  fit_with_workspace(std::span<const T> values,
                     BSplineCollocationWorkspace<T, BANDWIDTH>& ws,
                     const BSplineCollocationConfig<T>& config = {}) const
      requires std::same_as<Backend, LapackBandedBackend>;
  ```

  (Review [P2]: a deferred `static_assert` would make `requires`-based
  detection see the method as available on non-LAPACK backends and fail
  only at body instantiation; the member constraint states the truth —
  the workspace hardcodes LAPACK's LDAB layout and pivot regions.)
- The private helpers `fill_lapack_band`, `factorize_banded_workspace`,
  `solve_banded_workspace`, `estimate_banded_condition_workspace`, and
  `estimate_condition_from` are deleted (absorbed by the backend).
  `compute_matrix_norm1` and `compute_residual_from_span` stay (neutral
  math on the internal band). `build_collocation_matrix_to_workspace`
  becomes a thin `Backend::pack` call.
- Includes dropped from `bspline_collocation.hpp`: `<lapacke.h>`,
  `lapack_banded_layout.hpp`, `banded_matrix_solver.hpp`. Includes added:
  `banded_solver_backend.hpp`, `lapack_banded_backend.hpp` (named as the
  default template argument).

### `bspline_collocation_workspace.hpp`

Stays LAPACK-specific by design (its `required_bytes`/`LDAB` byte layout
is a contract with pre-allocating callers). It swaps its raw `lapack_int`
mentions for `LapackBandedBackend::pivot_type` and gets LAPACK vocabulary
through `lapack_banded_backend.hpp` instead of `<lapacke.h>` (review
[P2]). Its `lapack_banded_layout.hpp` include stays.

### Untouched

- `BSplineNDSeparable` — uses the default backend; no signature change.
- `banded_matrix_solver.hpp` — keeps `BandedMatrix`/`BandedLUWorkspace`
  and the workspace-based free functions for its users (tests
  `bspline_banded_solver_test`, `mdspan_integration_e2e_test`); gains the
  `banded_solver_backend.hpp` include for `BandedResult`.
- `lapack_banded_layout.hpp` and its test — now consumed by the LAPACK
  backend and the workspace.

### Bazel

`src/math/BUILD.bazel`: new `cc_library` targets `banded_solver_backend`
(hdr only) and `lapack_banded_backend` (deps: `banded_solver_backend`,
`lapack_banded_layout`; **carries `linkopts = ["-llapacke"]`** so every
direct consumer links correctly — there is no reusable LAPACKE dep target
today, `banded_matrix_solver` carries the same linkopt itself, round 2
[P2]). `banded_matrix_solver` gains dep `banded_solver_backend` and keeps
its linkopt; `//src/math/bspline:bspline_collocation` swaps
`banded_matrix_solver` for the two new targets **and drops its own
`-llapacke` linkopt** (implementation-specific linking belongs to the
backend target); `bspline_collocation_workspace` adds
`lapack_banded_backend`.

## Consumer audit (from the design review — verified against the tree)

- `BandedResult` users: banded solver + collocation only; preserved via
  the include shuffle.
- `BandedMatrix`/`BandedLUWorkspace` external users: tests only; API kept.
- `fit_with_buffer`: no in-tree callers outside its definition (kept
  regardless, re-expressed).
- `BSplineCollocationFactorization<double>` in
  `tests/bspline_collocation_1d_test.cc` and `BSplineCollocation1D<double>`
  spellings everywhere: source-compatible via defaulted params.

## Testing

1. **Goldens are the proof:** `//tests:bspline_bit_identity_test` (every
   coefficient, bit-pattern-exact, pinned pre-refactor in #457) must pass
   unchanged, plus the full suite (`bazel test //...`, 137-test baseline).
2. **Concept enforcement test** (compile-time, in the dedicated
   `tests/lapack_banded_backend_test.cc` — the plan moved items 2–3 out of
   `bspline_collocation_1d_test.cc` into this backend-scoped file):
   `static_assert(mango::BandedSolverBackend<mango::LapackBandedBackend, double>);`
   plus two negative checks:
   `static_assert(!mango::BandedSolverBackend<int, double>);` and — the
   honest-double-constraint guard from round 2 —
   `static_assert(!mango::BandedSolverBackend<mango::LapackBandedBackend, float>);`.
3. **Backend-unit regression** (new tests in the same file, with the
   standard regression header citing #464): `LapackBandedBackend::pack`
   output equals the former layout byte-for-byte for a small fixed band
   (golden literals), and `factorize`+`solve` round-trips a known system.
4. **No-LAPACK-leak guard:** a grep-style assertion in the PR (not a unit
   test): `grep -c "LAPACKE_" src/math/bspline/bspline_collocation.hpp`
   must be 0. Recorded in the PR body; CI-proof is the compile after the
   include drop.
5. Existing `bspline_fit_with_workspace_test`,
   `bspline_collocation_workspace_test`, `bspline_banded_solver_test`,
   `lapack_banded_layout_test`, 4D separable + table suites: unchanged and
   green.

## Acceptance criteria

- Zero `LAPACKE_*` references and no `<lapacke.h>` /
  `lapack_banded_layout.hpp` / `banded_matrix_solver.hpp` includes in
  `bspline_collocation.hpp`.
- `BandedSolverBackend` concept compiles and gates the class; all existing
  template spellings compile unchanged.
- Bit-identity goldens and full suite green; benchmarks + Python + Rust
  builds green.

## Out of scope

- The Eigen backend itself (the experiment lives in
  `.worktrees/migrate-lapack-eigen`; this seam only has to make it
  possible).
- Migrating `banded_matrix_solver.hpp`'s legacy `BandedMatrix` API or its
  test users onto the backend.
- Backend-parameterizing `BSplineCollocationWorkspace`'s byte layout or
  `BSplineNDSeparable`.
