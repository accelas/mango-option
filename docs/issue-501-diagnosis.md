# Issue 501: segmented B-spline dividend placement

Diagnosed on 2026-09-13 against `f510c1ab`, using optimized builds and
`OMP_NUM_THREADS=1`. [Issue #501](https://github.com/accelas/mango-option/issues/501)
contains the original configuration and observations.

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
