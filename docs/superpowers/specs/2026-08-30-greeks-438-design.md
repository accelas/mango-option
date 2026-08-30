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

## Regression tests

New tests in `tests/american_option_test.cc` (the existing home of
`AmericanOptionResult` behavior tests), in the repo's regression-test format
(`// Regression:` / `// Bug:` comments).

### Gamma: spot in the last interior interval

Solve an ATM-ish put on a custom narrow grid chosen so
`x_spot = ln(S/K)` lands strictly inside the last interior interval
`(x[n-2], x[n-1])` — e.g. solve with an explicit `GridSpec` whose x-range is
asymmetric around 0. Assert the returned gamma matches a central-difference
reference computed from `value_at`:

```
gamma_fd = (V(S+h) - 2 V(S) + V(S-h)) / h²,  h = 0.5% of S
EXPECT_NEAR(result->gamma(), gamma_fd, tol)
```

Under the bug, the interpolation blends with `d2v_dx2[n-2] == 0`, so gamma is
biased low by construction; the test fails RED before the fix. `tol` is
calibrated to the FD reference's own truncation error (loose, e.g. 15% relative
or an absolute floor), not to machine precision — the point is catching a
zero-blended value, which is off by ~alpha × 100%.

Also add a cheap exactness guard that pins the convention: for a solved grid,
gamma computed at a spot exactly on node `n-2` must be nonzero (under the bug it
is exactly 0 when `i_left == i_right == n-2`).

### Theta: non-uniform time grid (discrete dividend)

Price an option whose `PricingParams` carries a discrete dividend at a
calendar time that forces `with_mandatory_points` to produce a non-uniform
time grid where `dt_at(n_steps-1)` differs measurably from the average `dt()`
(assert this precondition inside the test so it cannot silently pass vacuously).
Compare `result->theta()` against a bumped-maturity FD reference:

```
theta_fd = (V_{T-h}(S) - V_T(S)) / h        (two full solves, h small)
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
