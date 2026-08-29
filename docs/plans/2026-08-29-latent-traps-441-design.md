# Latent trap fixes (issue #441) — design

**Issue:** [#441](https://github.com/accelas/mango-option/issues/441) — eleven latent
traps found in the July 2026 core review. None is triggered by a production call
path today; each is a footgun for future callers.

**Scope:** fix items 1–3 and 5–11 in one batch PR of small, independent fixes,
each with a regression test. Item 4 (Neumann Jacobian boundary rows degenerate
to identity) is split into its own issue: it requires genuine numerical design
(how to differentiate the spatial operator at a Neumann boundary), it has no
in-repo Neumann user, and its `lower[n-2]` sub-item was already fixed on main
by PR #445 (issue #433). The new issue is filed when this PR opens,
referencing #441.

All line references below are against main @ `b87a6b65` (the branch point of
`fix/441-latent-traps`).

## Decisions

Brainstorm Q&A (2026-08-29), each settled with the user:

1. **Scope** — options: (a) 10 items, split out #4; (b) all 11 items; (c)
   smaller subset. **Chosen: (a).** Rationale: #4 is the only item needing real
   numerical design and has zero current users; the issue itself says "split
   out any item when picked up."
2. **Item 1, `TimeDomain(t_start, t_end, dt)`** — options: (a) recompute dt
   after ceil'ing n_steps; (b) remove the constructor; (c) keep dt, clamp the
   last point. **Chosen: (a) recompute dt.** Matches how
   `with_mandatory_points` already routes its ceil through `from_n_steps`
   (time_domain.hpp:73–76); dt becomes "at most the requested dt" and the grid
   lands on `t_end` within floating-point rounding.
3. **Item 6, `CubicSpline2D::eval` extrapolation** — options: (a) clamp the
   code to match the documented nearest-boundary contract; (b) re-document the
   cubic extrapolation. **Chosen: (a) clamp.** Docs are the contract;
   nearest-boundary is the safe behavior for pricing surfaces. Low risk: no
   production consumer found — only its own test and one benchmark
   (`benchmarks/cubic_spline_template_vs_hardcoded.cc`).
4. **Item 7, factory error collapsing** — options: (a) one new catch-all code;
   (b) full 1:1 `PriceTableErrorCode` → `ValidationErrorCode` mapping; (c)
   restructure to propagate the underlying code. **Chosen: (a).** Stops the
   `InvalidGridSize` lie with minimal enum growth; per-code granularity
   deferred until someone needs it. *Adjusted in design review round 1:* the
   catch-all is named `PriceTableBuildFailed` (not `SurfaceBuildFailed`) —
   the default arm also covers config, serialization, and allocation
   failures, so a surface-specific name would be a new lie.
5. **Item 8, root-finding tolerances** — options: (a) docs-only fix; (b) also
   split Brent's x-tolerance into a new field; (c) implement relative
   tolerance as documented. **Initially chosen: (a) docs-only.** *Revised at
   the plan go/no-go:* the user accepted the design reviewer's twice-made
   case and switched to **(b)** — add `std::optional<double> brent_tol_x`
   used for bracket-width and safeguard comparisons with
   `value_or(brent_tol_abs)` fallback, so existing callers stay bit-for-bit
   identical; `tolerance`'s doc fix (absolute, not relative) is unchanged.
6. **Item 3 sub-choice** (resolved from code evidence, not asked): **both**
   types get **re-pointing user-defined copy ops**. *Adjusted in design
   review round 2:* the original plan deleted `BSplineND`'s copies on the
   premise that all uses go through `shared_ptr<const BSplineND>`; that
   premise is false — `BSplineND::create` returns
   `std::expected<BSplineND, …>` (a value-type API) and tests/benchmarks use
   it as a value, so deletion would be a public API break beyond this batch's
   charter. `NonUniformSpacing` additionally sits in a `std::variant` inside
   the value-type `Grid` (grid.hpp:635), so it must stay copyable.
7. **Item 8 reviewer disagreement (resolved):** design review rounds 2 and 3
   argued for an optional `brent_tol_x` field (`value_or(brent_tol_abs)`
   fallback) on the grounds that docs alone leave the scale-dependent unit
   mixing in place — rescaling the objective changes effective root-location
   accuracy. Initially held to the brainstorm's docs-only choice and
   recorded as a disagreement; at the plan go/no-go the user accepted the
   reviewer's variant. See decision 5.

## Fixes

### 1. `TimeDomain(t_start, t_end, dt)` overshoots `t_end`

`src/pde/core/time_domain.hpp:22–27`: `n_steps_ = ceil((t_end−t_start)/dt)`
while `dt_` keeps the requested value, so `time_points()` (`t_start + i*dt`)
ends past `t_end` whenever the span is not an integer multiple of dt —
e.g. `(0, 1, 0.3)` integrates to t = 1.2.

**Fix:** after computing `n_steps_ = max<size_t>(1, ceil((t_end−t_start)/dt))`,
set `dt_ = (t_end − t_start) / n_steps_` — identical semantics to routing
through `from_n_steps`. Update the constructor doc: dt is a *maximum* step;
the actual step is shrunk so the grid lands on `t_end` within floating-point
rounding.

**Preconditions (documented, not validated):** finite inputs,
`t_end > t_start`, `dt > 0`. Invalid inputs (reversed bounds, dt ≤ 0,
non-finite values, extreme `span/dt` overflowing the float-to-`size_t`
conversion) remain unsupported — same as today, and same as
`from_n_steps(…, 0)`'s division by zero. Adding validation to `TimeDomain`
is explicitly out of scope for this batch; the constructor doc states the
preconditions.

**Test:** `TimeDomain(0.0, 1.0, 0.3)` → `n_steps() == 4`, `dt() == 0.25`,
`time_points().back()` equal to `1.0` within `1e-12` (tolerance, not
bit-exact — `t_start + n_steps·dt` need not reproduce arbitrary `t_end`
bit-for-bit); exact-multiple case `(0, 1, 0.25)` unchanged.

### 2. `PDESolver` cannot be re-run (events dropped)

`src/pde/internal/pde_solver.hpp:233`: `next_event_idx_ = 0` is only ever
incremented (`:250`, `:273`); neither `solve()` (`:114`) nor `initialize()`
(`:97`) resets it. A second `solve()` on the same instance starts with the
index past the end and silently skips every temporal event (dividends
dropped).

**Fix:** reset `next_event_idx_ = 0;` in `initialize()`, where a new run's
solution state begins — cursor reset and state reset stay coupled. (Resetting
in `solve()` instead would replay events if a caller mistakenly re-solves the
already-final state without re-initializing.) Document the lifecycle on both
methods: the reuse contract is `initialize(ic); solve();` per run, and
`solve()` without a fresh `initialize()` is unsupported.

**Test:** with a temporal event that visibly changes the solution and counts
its invocations: `initialize(ic); solve();` capture the result; then
`initialize(ic); solve();` on the same instance and assert (a) the second
result equals the first and (b) the event fired exactly twice — today it is
skipped on the second run, so equality alone would be an indirect signal.

### 3. Implicit copies leave mdspan views dangling

Two value types hold an mdspan over their own vector and rely on
compiler-generated copies, which alias the *source's* buffer:

Both types get user-defined **re-pointing copy operations** and explicitly
defaulted moves (safe: `std::vector` move preserves `data()`, so the view
copied from the source points into the buffer now owned by the target). All
special-member declarations public. Copy assignment is strong-exception-safe
via copy-construct-then-move-assign, so a throwing allocation never leaves
size metadata inconsistent with the buffer.

- `src/math/bspline/bspline_nd.hpp:271` — `coeffs_view_` over `coeffs_`:
  copy the members, then `coeffs_view_ = create_coeffs_view(coeffs_.data(),
  dims_);`. (`BSplineND` is a value type — `create` returns
  `std::expected<BSplineND, …>` and tests/benchmarks hold it by value — so
  copyability must be preserved, not deleted.)
- `src/pde/core/grid.hpp:547` — `NonUniformSpacing::sections_view_` over
  `precomputed`: copy the members, then `sections_view_ =
  SectionView(precomputed.data(), 5, n − 2);`. Keeps `Grid`/`SpacingVariant`
  copyable.

**Tests:** for each type: copy-construct *and* copy-assign inside a scope,
destroy the source, then assert outside the scope that the view aliases the
copy's own buffer (`sections_view_.data_handle() == precomputed.data()` for
`NonUniformSpacing`) and values read correctly; for `BSplineND`, evaluation
after the source is destroyed exercises `coeffs_view_`. Move tests likewise
construct the moved-to object, end the source's lifetime, and then evaluate.

**Moved-from contract (documented):** after a move, the source's view may
alias the destination's buffer; a moved-from object supports only
destruction and reassignment, never evaluation.

### 5. Thomas solve with n==0 returns misleading error

`src/math/thomas_solver.hpp:100` checks `lower.size() != n − 1` before the
`n == 0` trivial-success at `:117`; with `n == 0`, `n − 1` wraps to
`SIZE_MAX`, so the empty system returns "Lower diagonal size must be n-1" and
the success path is dead code. `solve_thomas_projected` (the obstacle
variant) duplicates the bug (`:336` guard vs `:356` trivial case).

**Fix:** in both functions, compute
`const size_t offdiag_size = (n == 0) ? 0 : n - 1;`, validate all companion
span sizes against it (so a malformed call — empty `diag` with nonempty
`lower`/`upper`/`rhs`/`psi`/`solution` — is still rejected), then return
`ok_result()` for `n == 0` after validation passes.

**Test:** both solvers with all-empty spans → success; empty `diag` with a
nonempty companion span → error, not success.

### 6. `CubicSpline2D::eval` ignores its documented extrapolation contract

Docs (`src/math/cubic_spline_solver.hpp:440`) promise "extrapolation uses
nearest boundary value," but the implementation evaluates the boundary cubic
at the out-of-range offset (1D `eval` `:185–195` clamps only the interval
index, not `dx`), so out-of-range queries diverge cubically.

**Fix:** clamp in `CubicSpline2D::eval` only — clamp `x_eval` to
`[x_.front(), x_.back()]` and `y_eval` to `[y_.front(), y_.back()]` before
evaluating. The 1D `CubicSpline::eval` keeps its current (documented at
`:183`: "or extrapolated if outside domain") behavior — changing the 1D class
would silently alter every 1D consumer, out of scope for a latent-trap batch.
Risk is low: `CubicSpline2D` has no production consumer — only its own test
and `benchmarks/cubic_spline_template_vs_hardcoded.cc`.

**Test:** build a 2D spline on data whose boundary cubic has nonzero slope;
query outside the grid on each side *and* at corners (both coordinates
outside); assert the result equals the boundary evaluation
(`eval(x_max, y)` etc.), not the diverged cubic value. Update the existing
extrapolation tests in `tests/cubic_spline_2d_test.cc`, whose comments
currently describe "natural spline extrapolation" — they pin the old
behavior and must flip to the clamped contract.

### 7. Factory collapses distinct build failures into `InvalidGridSize`

`src/option/price_table_factory.cpp:90–105`: `to_validation_error` maps
`NonPositiveValue → InvalidBounds` and
`InsufficientGridPoints`/`GridNotSorted → InvalidGridSize`, but the
`default:` arm sends every other `PriceTableErrorCode` (`InvalidConfig`,
`EmptyBatch`, `ExtractionFailed`, `RepairFailed`, `FittingFailed`,
`SurfaceBuildFailed`, `SerializationFailed`, `ArenaAllocationFailed`,
`TensorCreationFailed`) to `InvalidGridSize` too. Five more sites hardcode
`ValidationError{ValidationErrorCode::InvalidGridSize, 0.0}` for
surface-construction failures (`:297`, `:332`, `:535`, `:541`, `:558`).
Diagnostics are destroyed: a fitting failure surfaces as a grid-size error.

**Fix:**
- Add `ValidationErrorCode::PriceTableBuildFailed` (appended at the end of
  the enum, preserving existing ordinals). The name is deliberately generic:
  the default arm covers config, extraction, repair, fitting, surface,
  serialization, allocation, and tensor failures, so a surface-specific name
  would just be a different lie.
- Map the `default:` arm and the five hardcoded sites to
  `PriceTableBuildFailed` (payload `value`/`index` preserved where
  available).
- **Update every `ValidationErrorCode` conversion switch.** Three conversion
  switches require semantic updates; two are exhaustive with an
  *uninitialized* `code` local and no `default:`, so omitting the new value
  is UB, not just a warning. (Python's error-message formatting also
  switches on selected values, but its `default:` is safe and that path
  cannot receive `PriceTableBuildFailed` — no change needed.)
  - `convert_to_iv_error(const ValidationError&)`
    (`error_types.hpp:262`): add `PriceTableBuildFailed →
    IVErrorCode::InvalidGridConfig` — the same bucket the collapsed
    `InvalidGridSize` lands in today, so IV-level observable behavior is
    unchanged.
  - `convert_to_price_table_error(const ValidationError&)`
    (`error_types.hpp:348`): add `PriceTableBuildFailed →
    PriceTableErrorCode::SurfaceBuildFailed` (closest generic build-failure
    code on the round trip).
  - `validation_error_to_iv_error` (`iv_result.hpp:55`) has a `default:` (→
    `ArbitrageViolation`); add an explicit `PriceTableBuildFailed →
    IVErrorCode::InvalidGridConfig` case rather than letting a build failure
    masquerade as an arbitrage violation.
- Prefer explicit grouped `case` labels over keeping a `default:` arm, so a
  future `PriceTableErrorCode` value triggers a compiler warning instead of
  being silently swallowed by the catch-all.
- **Route the five hardcoded sites through the same mapper**: each site
  constructs a `PriceTableError` (converting an `InterpolationError` via the
  existing `convert_to_price_table_error` where that is the failure in hand)
  and calls the single `to_validation_error` overload — no overload
  proliferation on the mapper. This way the direct unit test pins the five
  sites' code path too, and available `InterpolationError` payloads
  (index/grid-size) are preserved instead of returning `{…, 0.0}`
  unconditionally.
- For testability, declare `to_validation_error` in `namespace mango::detail`
  in a small internal header (e.g.
  `src/option/detail/price_table_error_mapping.hpp`, commented as internal
  and unstable — not part of the public API). **Bazel:** the header must not
  be exported through the public factory target's `hdrs`; give it a small
  library target (or `hdrs` entry) with visibility restricted to
  `//src/option` and `//tests`, so reverse dependencies of the factory never
  see it.

**ADR / bindings note:** ADR 0001 (Python API parity centers on reusable
price tables) makes factory error plumbing part of the Python surface;
Python does not bind the enum — it puts `static_cast<int>(error.code)` on
the exception's `code` attribute (`mango_bindings.cpp:50`) — so appending
preserves all existing numeric values and only adds a new observable code. No persisted format (Parquet
schema) serializes `ValidationErrorCode`, so no artifact change.

**Interaction with PR #454:** the open PR #454 extends this same switch
(adds `NoViableSurface`/`AdaptiveValidationFailed` arms). Whichever merges
second resolves a small, mechanical conflict in `to_validation_error` and the
`ValidationErrorCode` enum; the changes are semantically independent (454
adds specific arms; this change only retargets the fallback). Rebase rule:
already-merged enum values keep their ordinals; whichever change lands second
appends its new values after the first's.

**Test:** direct unit test of `mango::detail::to_validation_error`: each
still-collapsed `PriceTableErrorCode` maps to `PriceTableBuildFailed`, and
the existing specific arms are unchanged. Conversion tests for the new value
through `convert_to_iv_error`, `convert_to_price_table_error`, and
`validation_error_to_iv_error`.

### 8. Root-finding tolerance docs lie

`src/math/root_finding.hpp:24–25` documents `tolerance` as "relative
convergence tolerance," but Newton uses it as an absolute residual bound
(`std::abs(fx) < config.tolerance`). `brent_tol_abs` (`:31`) is applied both
to f-values (`std::abs(fb) < brent_tol_abs`) and to x-interval widths.

**Fix:** two parts, both behavior-preserving for existing callers:

- Re-document `tolerance` as Newton's absolute residual tolerance |f(x)|
  (it was documented as relative).
- Add `std::optional<double> brent_tol_x` to `RootFindingConfig`, declared
  *after* `brent_tol_abs` (designated-initializer order compatibility). In
  `find_root_brent`, `const double tol_x =
  config.brent_tol_x.value_or(config.brent_tol_abs);` is used for the
  x-distance comparisons — the `|b − a|` stopping test and safeguard
  conditions 4 (`|b − c| < tol`) and 5 (`|c − d| < tol`) — while
  `brent_tol_abs` keeps the f-value roles (endpoint root checks and the
  `|f(b)|` stopping test). Unset (the default), behavior is bit-for-bit
  identical. Document both fields' units.

**Test:** a fallback-compatibility test (unset `brent_tol_x` reproduces the
current root/iteration count on a known problem) and a scale-sensitivity
regression showing a tight `brent_tol_x` with a loose `brent_tol_abs` on a
rescaled objective locates the root to x-accuracy that the single shared
knob could not.

### 9. `BSplineCollocation1D` Bandwidth parameter is a stack-overflow trap

`src/math/bspline/bspline_collocation.hpp:74`: `template<std::floating_point
T, size_t Bandwidth = 4>`, but `T basis[BANDWIDTH]` (`:439`) is passed to
`cubic_basis_nonuniform`, which unconditionally writes 4 entries
(`bspline_basis.hpp:174–177`). Any instantiation with `Bandwidth < 4` is a
stack buffer overflow; only 4 is ever instantiated.

**Fix:** `static_assert(Bandwidth == 4, ...)` in `BSplineCollocation1D`
(the type that actually calls the 4-entry cubic basis routine), and
`static_assert(Bandwidth > 0, ...)` in `BSplineCollocationWorkspace` — the
workspace is a public type instantiated directly by tests and used
independently by `BSplineNDSeparable`, and its band-layout math is generic;
its only independent hazard is the `Bandwidth − 1` underflow at 0, so
restricting it to exactly 4 would break valid generic instantiations. The
collocation class's parameter doc changes from "degree + 1" generality to
*reserved*: only 4 (cubic) is supported until the basis evaluation is
generalized.

**Test:** compile-time; no runtime test. (The static_asserts *are* the
regression guard.)

### 10. Workspace pivots typed `int`, not `lapack_int`

`src/math/bspline/bspline_collocation_workspace.hpp` sizes, aligns, and
slices the pivot block as `int` (`:30`, `:54`, `:68–70`, `:111–114`, `:165`)
but hands `pivots().data()` to `LAPACKE_dgbtrf`/`LAPACKE_dgbtrs`, which take
`lapack_int*`. Under an ILP64 LAPACK build this breaks at compile time.
Sibling `banded_matrix_solver.hpp:235` already uses
`std::vector<lapack_int>`.

**Fix:** switch the pivot block to `lapack_int` throughout the workspace
(span type, `sizeof`/`alignof` in layout math, `start_array_lifetime`),
mirroring `banded_matrix_solver.hpp`. Include the same LAPACK header that
provides `lapack_int` there. **Also update the existing workspace tests**
(`tests/bspline_collocation_workspace_test.cc`), which hardcode 400 pivot
bytes and use `sizeof(int)` in pointer arithmetic — those assumptions fail
under exactly the ILP64 configuration this fix targets; replace with
`sizeof(lapack_int)`.

**Test:** in the workspace test, `using Pivot = typename
decltype(ws.pivots())::element_type;
static_assert(std::same_as<Pivot, lapack_int>);` behavior under LP64
unchanged (existing tests cover). **Known limitation, accepted:** under the
repo's LP64 CI (`lapack_int == int`) this assertion passes even without the
fix, so there is no effective automated regression guard for the type
mismatch; a real guard needs an ILP64 compile job, which is out of scope for
this batch. The fix itself plus `sizeof(lapack_int)`-correct arithmetic is
the deliverable.

### 11. Newton `ConvergenceFailure` reports residual ≈ 0

`src/pde/internal/pde_solver.hpp:803`: the last statement of each Newton
iteration copies `u` into `newton_u_old`; the failure return then computes
`.residual = compute_step_delta_error(u, newton_u_old)` — a comparison of a
buffer with itself, so every convergence failure reports residual 0. The
genuine step delta was computed at `:796` and discarded.

**Fix:** track the most recent step-delta error in a local
(`double last_error = std::numeric_limits<double>::infinity();` updated each
iteration) and report it in the `ConvergenceFailure` return. The
`LinearSolveFailure` path (reports infinity) is untouched. Document
`SolverError::residual` as a **code-dependent diagnostic** — the same field
carries a grid-validation value elsewhere (`american_option.cpp:389`),
infinity for linear-solve failures, and, only for `ConvergenceFailure`, the
solution-norm-normalized RMS step delta from `compute_step_delta_error`
(not a PDE residual).

**Test:** force a convergence failure with `max_iter == 1` (not 0, which
would trivially report the infinity seed) on a deterministic fixture whose
first Newton update is nonzero, with a tolerance tight enough that one
iteration cannot converge; assert `std::isfinite(residual) &&
residual > 0`.

## Test strategy

Each behavior change carries a regression test in the CLAUDE.md format
(`// Regression:` / `// Bug:` comment, named test). Item 9
(compile-time assert) has no runtime test; item 10's runtime assertion is
acknowledged ineffective under LP64 CI (see item 10). Existing suites
touched: `time_domain_test`, `tests/internal/pde_solver_test.cc` (its
`TestPDESolver` fixture fits the reuse and convergence tests),
`thomas_solver_test`, `cubic_spline_2d_test`, plus the b-spline
workspace/collocation and factory tests. Full gate: `bazel test //...`,
`bazel build //benchmarks/...`, `bazel build //src/python:mango_option`.

## Out of scope

- Item 4 (Neumann Jacobian boundary rows) — new issue filed at PR time.
- Relative-tolerance semantics for `tolerance` (rejected in brainstorm;
  revisit only if a caller needs it).
- Per-code error taxonomy beyond `PriceTableBuildFailed` (rejected in
  brainstorm).
- Any change to 1D `CubicSpline::eval` extrapolation.
- Input validation in `TimeDomain` (both the dt constructor and
  `from_n_steps(…, 0)`'s division by zero) — preconditions documented
  instead; enforcing them is a separate decision.
- An ILP64 LAPACK CI job (the only effective automated guard for item 10's
  regression class).
