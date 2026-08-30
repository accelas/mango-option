# Fix AmericanOptionResult Greeks: gamma stencil range, theta step size (issue #438)

**Issue:** https://github.com/accelas/mango-option/issues/438
**Status:** Spec for design review

## Problem

Two confirmed correctness defects in `AmericanOptionResult` Greeks
(`src/option/american_option_result.cpp`), found in code review.

### 1. Gamma skips interior node `n-2`

```cpp
operator_->compute_first_derivative(solution, std::span(dv_dx), 1, n - 2);
operator_->compute_second_derivative(solution, std::span(d2v_dx2), 1, n - 2);
```

`CenteredDifference` loops `for (i = start; i < end; ++i)` — the end bound is
exclusive (verified in `src/pde/operators/centered_difference.hpp:49,88`; the
interior convention elsewhere is `[1, n-1)`, e.g. `spatial_operator.hpp`). With
`end = n - 2`, interior node `n-2` is never written and stays `0.0`. When the
query spot falls in the last interior interval, `find_grid_index` returns
`i_right = n-2`, and the linear interpolation blends against a spurious zero —
biasing gamma toward 0 by up to the full local value. The stencil at `i = n-2`
reads `u[n-1]`, which is in range, so extending the bound is safe.

### 2. Theta divides by the average dt, not the actual last step

```cpp
double dt = grid_->dt();                            // average dt
double theta_normalized = (v_prev - v_current) / dt;
```

`grid_->solution_prev()` is exactly one **actual** final time step away from
`grid_->solution()`, i.e. `dt_at(n_steps - 1)`. `Grid::dt()` forwards
`TimeDomain::dt()`, which for non-uniform grids (the discrete-dividend case:
`with_mandatory_points` sets `dt_ = span / n_steps` as an average) differs from
the final step size. Theta comes out scaled by `dt_avg / dt_last` — off by tens
of percent for realistic dividend schedules.

## Fix

Both fixes are local to `src/option/american_option_result.cpp`. No interface
changes anywhere.

### Gamma

Change both derivative range ends from `n - 2` to `n - 1` in `gamma()`:

```cpp
operator_->compute_first_derivative(solution, std::span(dv_dx), 1, n - 1);
operator_->compute_second_derivative(solution, std::span(d2v_dx2), 1, n - 1);
```

### Theta

Replace the average dt with the actual final step size. `Grid::time()`
(`src/pde/core/grid.hpp:917`) already exposes the `TimeDomain`, and
`TimeDomain::dt_at(step)` (`time_domain.hpp:140`) returns the true step for
non-uniform grids and the constant dt for uniform ones — so the issue's
contingency ("threading it into the result if needed") does not arise:

```cpp
const auto& time = grid_->time();
double dt = time.dt_at(time.n_steps() - 1);
double theta_normalized = (v_prev - v_current) / dt;
```

For uniform time grids `dt_at(n_steps - 1) == dt()`, so behavior there is
bit-identical.

**Known limitation (documented, not fixed here):** when `n_time == 1`, the
American solver's unconditional Rannacher startup leaves `solution_prev` at the
midpoint substep — only `dt/2` behind the final solution — so theta is ~2×
off. This is pre-existing (the old average-dt divisor is the same full `dt`
there) and unchanged by this fix; theta's doc comment gains a note that it
assumes the final step is a plain TR-BDF2 step (`n_steps >= 2`, true for every
production grid config). No validation is added — minimal-diff scope.

## Regression tests

New tests in `tests/american_option_result_test.cc` (the existing home of
`AmericanOptionResult` behavior tests), in the repo's regression-test format
(`// Regression:` / `// Bug:` comments).

Additionally, `tests/american_option_gamma_oscillation_test.cc:48` duplicates
the defective `[1, n-2)` range in its own stencil computation and then examines
node `n-2` — it reads a never-written zero. That range is corrected to
`[1, n-1)` in the same commit (in scope: it is the same bug, copy-pasted).

### Gamma: interpolation touching node `n-2`

Node `n-1` is a boundary node and stays zero even after the fix (the stencil
range is interior-only), so gamma in the *last* interval `(x[n-2], x[n-1])` is
deliberately tapered — that interval is NOT the test target. The interval where
the bug is cleanly visible is `(x[n-3], x[n-2])`: pre-fix, `i_right = n-2`
blends against a spurious zero; post-fix both ends are genuine values.

Two assertions:

1. **Exact-node guard (guaranteed RED):** choose spot so `x_spot` lands exactly
   on node `n-2` (`find_grid_index` returns `{n-2, n-2}` via its 1e-14 snap).
   Pre-fix gamma is exactly 0.0 there; assert it is nonzero and matches an FD
   reference.
2. **FD cross-check:** with spot strictly inside `(x[n-3], x[n-2])`, assert
   gamma matches a central-difference reference computed from `value_at`:

```
gamma_fd = (V(S+h) - 2 V(S) + V(S-h)) / h²,  h small enough that S±h stays
           inside (exp(x[n-3])·K, exp(x[n-1])·K)  — no boundary clamping
EXPECT_NEAR(result->gamma(), gamma_fd, tol)
```

`tol` is calibrated to the FD reference's truncation error (loose, e.g. 15%
relative or an absolute floor), not machine precision — the point is catching
the zero-blended value.

### Theta: non-uniform time grid (discrete dividend)

Price an option whose `PricingParams` carries a discrete dividend at a
calendar time that forces `with_mandatory_points` to produce a non-uniform
time grid where `dt_at(n_steps-1)` differs measurably from the average `dt()`
(assert this precondition inside the test so it cannot silently pass
vacuously). A dividend with calendar time shortly after valuation (τ-event
near t_end) makes the final segment's step size differ from the average by
tens of percent — e.g. maturity 1.0, dt ≈ 0.09, dividend at calendar 0.05 →
last segment [0.95, 1.0] steps at 0.05 vs average ≈ 0.091.

Compare `result->theta()` against a **calendar** FD reference: advancing the
valuation date by `h` shortens time-to-maturity to `T−h` AND brings every
future dividend `h` closer — `Dividend::calendar_time` is measured from the
valuation date, and the solver places the jump at `τ = maturity −
calendar_time`. The bumped solve must therefore use maturity `T−h` **and**
each dividend's `calendar_time` reduced by `h`; bumping maturity alone moves
the dividend's τ-coordinate and measures a different sensitivity:

```
theta_fd = (V_{T-h, divs-h}(S) - V_{T, divs}(S)) / h    (two full solves)
EXPECT_NEAR(result->theta(), theta_fd, tol)
```

Sign convention check: the solver marches τ from 0 to T, `solution_prev` is at
τ = T − dt_last, so `(v_prev − v_current)/dt = −∂V/∂τ = ∂V/∂t` — the existing
convention, unchanged by this fix.

Uniform-grid non-regression: an existing uniform-grid theta value must be
unchanged (dt_at == dt there); covered by the existing greek tests continuing
to pass.

## Decisions

- **Minimal correctness diff.** The issue's perf nit (gamma heap-allocates two
  `n`-vectors and computes all interior nodes per call when only 2 are needed)
  is explicitly out of scope — it is labeled a nit in the issue, and shrinking
  the computation is an optimization with no behavior change, separable later.
- **No interface changes.** `Grid::time()` + `TimeDomain::dt_at` already
  provide the last actual step; nothing is threaded through
  `AmericanOptionResult` or `Grid`.
- **Straight-to-plan triage.** The issue supplies concrete fixes, exact
  file/line targets, and the regression-test outline; no interface boundary was
  open, so no brainstorm was held. This spec is the reviewed artifact.
- **Blast radius.** Gamma changes only affect queries whose interpolation
  touches node `n-2` (near the right grid edge); theta changes only affect
  non-uniform time grids. Uniform-time ATM paths — everything the existing
  test suite pins — are numerically identical.
- **`n_time == 1` Rannacher edge documented, not fixed.** See the theta fix
  section. Pre-existing, unreachable by production grid configs, and the fix
  neither improves nor worsens it.
- **`american_option_gamma_oscillation_test.cc` range corrected in scope.** It
  copy-pastes the same `[1, n-2)` defect and reads node `n-2`; same bug, same
  commit.

## Design-review history

Round 1 (Codex): confirmed both production edits correct for multi-step
solves; three folds — (a) theta FD oracle must also shift dividend
calendar_times by `h` (dividends anchor to the valuation date), (b) gamma test
retargeted to `(x[n-3], x[n-2])` / exact node `n-2` since boundary node `n-1`
stays zero by design, (c) `n_time == 1` Rannacher midpoint edge documented as
a known pre-existing limitation.
