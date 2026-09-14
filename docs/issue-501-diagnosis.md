# Issue 501: segmented B-spline dividend placement

Diagnosed on 2026-09-13 against `f510c1ab`, using optimized builds and
`OMP_NUM_THREADS=1`. [Issue #501](https://github.com/accelas/mango-option/issues/501)
contains the original configuration and observations.

This records the initial validation-only fix in PR #503. The event-sided
snapshot follow-up below supersedes its gap-admission behavior.

## Confirmed cause and change

The segmented surface deliberately returns NaN inside omitted dividend
neighborhoods, extending 0.0005 years either side of an event. The adaptive
refiner sampled the rectangular maturity range without these exclusions.
An otherwise valid surface was rejected whenever a fresh or fixed holdout
sample landed in a gap. Moving the dividend changes which fixed samples
hit gaps, explaining the nonmonotonic viability pattern.

For a 14-day maturity with a dividend on day 10, the debugger found three
nonfinite holdout prices at tau 0.0114473, 0.0108616, and 0.0104966. All
three fall inside the excluded neighborhood of event tau 4/365. Five
refinement candidates were rejected this way. Their underlying segmented
builds succeeded. After admission was applied, all eight probe candidates
were finite and viable; the best probe error was approximately 0.00144.

`RefinementContext` now accepts a fixed maturity-admission predicate.
The B-spline builder derives it from the same segment boundaries and
`TauSegmentSplit` used by its surface builder. Fresh samples, cached probe
holdouts, and final assembly/retry references exclude unsupported times
before evaluating prices or solving references. Exclusions do not inflate
failed-reference diagnostics or PDE counts. Supported-time NaNs still
reject candidates, and too few supported references still fail validation.
Public gap queries continue to refuse; no neighboring time is substituted.

## Regression loop

```sh
bazel test -c opt //tests:adaptive_surface_build_integration_test \
  --test_filter='Issue501/*' --test_env=OMP_NUM_THREADS=1 --test_output=all
```

The three retained cases use (first dividend day, maturity days) of
(10, 30), (60, 180), and (75, 365), with quarterly $0.50 dividends and the
issue's grid. On the original builder all three failed with error code 13
(`NoViableSurface`); with the fix all three pass, individually in roughly
0.4–1.2 seconds. They also verify that event queries remain unsupported and
that supported times on both sides return finite prices.

Unit regressions verify that excluded times are never evaluated, supported
time NaNs still reject a candidate, excluded references are not counted as
failures, and an empty supported holdout cannot certify a surface.

## Full matrix replay

All 56 configurations were replayed before and after the change. The
calendar schedule was generated as `first + n * 91.25` days, strictly below
maturity. The original builder refused 16 cases; the corrected builder
refuses three. All 40 previously successful cases retained the same
reported error at the precision captured (six significant digits).

The following 13 cases changed from refusal to successful construction.
Errors below are reported holdout IV-error estimates in basis points;
a successful build does not imply that the requested 10 bps target was met.

| First dividend (days) | Maturity (days) | Before | After (bps) |
| --- | --- | --- | --- |
| 10 | 30 | NoViableSurface | 376.19 |
| 10 | 60 | NoViableSurface | 162.91 |
| 10 | 90 | NoViableSurface | 129.97 |
| 10 | 180 | NoViableSurface | 81.41 |
| 10 | 365 | NoViableSurface | 45.91 |
| 10 | 730 | NoViableSurface | 36.37 |
| 20 | 60 | NoViableSurface | 162.91 |
| 20 | 730 | NoViableSurface | 36.08 |
| 30 | 60 | NoViableSurface | 162.69 |
| 30 | 730 | NoViableSurface | 36.06 |
| 45 | 730 | NoViableSurface | 36.39 |
| 60 | 180 | NoViableSurface | 80.92 |
| 75 | 365 | NoViableSurface | 40.08 |

## Remaining numerical limitations

The three remaining refusals are caused by finite final-surface scores
above the existing 0.20 viability ceiling, not by excluded-time NaNs.
Debugger inspection of both assembly candidates gave:

| First dividend (days) | Maturity (days) | Original max error | Retry max error |
| --- | --- | --- | --- |
| 10 | 14 | 29.87299 | 29.30071 |
| 20 | 30 | 5.31150 | 5.35013 |
| 45 | 60 | 0.28103 | 0.26196 |

These are absolute IV-error estimates, not basis points. Every final score
had `all_finite=true` and zero skipped nonfinite evaluations. The fix keeps
these safety refusals. Their numerical accuracy requires separate work;
this diagnosis does not establish the cause of that pricing error.
Dividend-free short-maturity errors reported in the issue are unchanged.

One reproduction detail differs from the issue's table: with the specified
strict `(0, T)` filter, first=30d and T=30d contains no dividend and builds
at 329.43 bps on both versions. A one-year schedule consisting of day 5 plus
calendar 0.25, 0.5, and 0.75 year dividends also built on the original code,
so it was not retained as a regression test.

## Validation

All five relevant Bazel targets pass in optimized mode with
`--test_env=OMP_NUM_THREADS=1` (156 test cases):

- `//tests:adaptive_refinement_unit_test`
- `//tests:adaptive_grid_builder_test`
- `//tests:adaptive_surface_build_integration_test`
- `//tests:segmented_price_table_builder_test`
- `//tests:iv_solver_factory_test`

The subsequent pre-PR `bazel test //...` run passed all 156 test targets,
including the slow accuracy suites, with `--jobs=8 --local_test_jobs=4
--test_env=OMP_NUM_THREADS=2`. An unrelated ignored `.cache` directory of
research copies was temporarily excluded from Bazel package discovery;
the local exclusion was restored before committing.

Temporary matrix harnesses and debugger scripts were removed; no debug
instrumentation remains in source or tests.

The Python binding, benchmark wildcard, seven explicit CI benchmark targets,
and CI debug-mode adaptive logic test also passed.

## Event-sided snapshot follow-up (2026-09-14)

The B-spline builder now fits exact dividend boundaries using two raw rows
per event: the state immediately before the backward jump, and the ordinary
state after the jump and exercise/boundary projections. The smaller-tau leaf
owns equality, giving an exact-event query the post-dividend calendar value.
No fixed exclusion width or minimum event spacing is used by this builder.
Chebyshev retains its existing gap-based representation.

The PDE equations, TR-BDF2/Rannacher stages, and jump/projection operations
are unchanged. Optional snapshot capture copies existing states; call/put
regressions verify bit-identical final solutions, previous solutions, and
ordinary snapshots with capture enabled and disabled. A separate truncated
solve checks the first post-dividend calendar state independently. Extra
storage is allocated only for the requested event-side rows.

A former-gap regression at tau 0.4999 failed before this change (4 ms) and
now prices tau 0.4999, 0.5, and 0.5001 against independent FDM references.
Further tests cover sub-hour distances from either endpoint, sub-hour
separation between events, merged same-date dividends, adjacent floating-
point times on either side of an event, public IV solving, and persistence.
The tight near-expiry price checks use the High grid profile; supporting
the time itself does not establish the accuracy of a coarse grid.

The 56-case replay still builds 53 cases. The same (first dividend day,
maturity days) pairs (10, 14), (20, 30), and (45, 60) still return
`NoViableSurface`; this representation change does not resolve their
numerical accuracy. All formerly buildable cases remain buildable, and
validation now samples the full time domain, including the old gaps.

All 146 non-nightly targets passed with the CI compilation mode and
`OMP_NUM_THREADS=8`. An attempted full optimized suite exposed an existing
`NDEBUG` restriction on test-only builder accessors, so the full suite was
run in its supported CI mode; targeted optimized event tests also passed.

The three high-accuracy segmented sampling/Greek tests and three affected
slow B-spline adaptive tests also passed in optimized mode. The temporary
56-case matrix harness was removed after recording its results.

Before updating PR #503 with the complete implementation, all 156 repository
test targets passed, including nightly and Rust coverage, with
`OMP_NUM_THREADS=8`. The Python binding, new benchmark harnesses, explicit CI
benchmark targets, and debug-mode adaptive logic test also passed. The accuracy
harness was cleaned up to avoid a GCC initializer warning and reproduced both
stored evaluation CSVs byte-for-byte.
