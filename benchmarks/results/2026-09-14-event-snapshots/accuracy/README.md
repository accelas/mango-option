# Matched before/after accuracy check

Measured 2026-09-14 against the same baseline and candidate as the
[performance comparison](../README.md). This check uses identical query
coordinates and one shared independent reference dataset, rather than each
builder's own holdout score.

The change preserves the existing level of accuracy on the common domain;
it is not a general accuracy improvement. The newly supported dividend
neighborhoods have accuracy comparable to neighboring supported times.
Substantial pre-existing short-remaining-life errors and IV bracketing
failures remain.

## Method and scope

- Generated **279 unique reference points** once using the baseline direct
  FDM solver. Each point is solved separately from the table construction.
- Reference prices use both **High** and **Ultra** grid profiles. Their
  maximum disagreement is **$0.00009167**, with median **$0.00000641**.
  Agreement between two resolutions is a sensitivity check, not a rigorous
  bound on the unknown exact price.
- Evaluated **342 table/query pairs per revision**: two adaptive tables,
  plus a manual single-reference table evaluated on the ATM-strike subset
  of the one-year references. All three tables build on both revisions.
- Adaptive configurations match the performance experiment: spot 100, PUT,
  yield 2%, reference strikes 90–110 in steps of 2.5, target IV error 10 bps;
  one-year expiry with quarterly $0.50 dividends, and 30-day expiry with a
  $0.50 dividend on day 10. The manual one-year table has 60 moneyness sites
  and five time sites per segment, with the same quarterly schedule.
- Query strikes are 96.25, 100, and 103.75: two midpoints between reference
  strikes and one anchor. Volatilities are 12.5%, 22.5%, and 27.5%, and the
  rate is 4%; these are off the supplied seed knots.
- One-year common interior times are 0.03, 0.10, 0.33, 0.60, 0.90, and 1.0
  years of remaining life. Short-table interior times are 1, 7, 14, 25, and
  30 days. Around every event, queries are made at offsets ±0.0006 years
  (outside the old gap), ±0.0001 years (inside it), and exactly at the event.
- Every reference uses the correctly rolled dividend schedule. At an exact
  event the dividend has elapsed: the post-dividend calendar convention.
- Price errors are absolute dollars against Ultra references and include
  all queries with finite table prices, even when IV is not meaningful.
- IV error is **actual recovery error**: feed the Ultra price to the real
  `InterpolatedIVSolver`, then compute `abs(recovered_sigma - known_sigma)
  * 10000`. It is not the builder's linearized price-error/vega estimate.
- For IV-error summaries, eligibility depends only on the shared reference:
  time value / strike >= 1e-4 and positive Ultra FD vega >= 1e-4. Vega uses
  symmetric 1% volatility bumps. Eight short-table points have negligible
  time value or vega and are excluded from IV statistics, not price errors.
  Failures on eligible queries are counted separately, never as zero error.
- Tables use default PDE accuracy; the reference solves use finer grids.
  The samples are deterministic probes, not a representative market sample.

## Previously supported queries: price error

These are common interior and outside-gap edge queries. Prices are finite
at every one in both revisions.

| Table | Queries | Median before / after | P95 before / after | Max before / after |
| --- | ---: | ---: | ---: | ---: |
| Manual 1Y, single reference strike | 36 | $0.006415 / $0.006435 | $0.155234 / $0.155292 | $0.275453 / $0.275547 |
| Adaptive 1Y, multiple reference strikes | 108 | $0.008393 / $0.008477 | $0.034562 / $0.034545 | $0.047253 / $0.047172 |
| Adaptive 30d, multiple reference strikes | 63 | $0.028701 / $0.028670 | $0.055060 / $0.055056 | $0.139519 / $0.139645 |

The largest pointwise change in table price is $0.000542. The maximum
absolute error for the manual table is larger than the adaptive table's;
its five uniform time sites per segment are a coarse configuration.

## Previously supported queries: IV recovery error

All errors below are **basis points of absolute volatility**. These
statistics cover successful inversions on reference-eligible queries.
The success/failure sets are identical before and after.

| Table | Successes / eligible (both) | Median before / after | P95 before / after | Max before / after |
| --- | ---: | ---: | ---: | ---: |
| Manual 1Y, single reference strike | 35 / 36 | 2.45 / 2.45 | 82.61 / 82.61 | 382.43 / 382.58 |
| Adaptive 1Y, multiple reference strikes | 107 / 108 | 3.35 / 3.30 | 51.18 / 50.70 | 126.55 / 126.42 |
| Adaptive 30d, multiple reference strikes | 54 / 55 | 42.85 / 42.82 | 178.12 / 178.12 | 721.70 / 722.55 |

Each table has one unchanged `BracketingFailed` query among the eligible
common set. Among the eight reference-ineligible short-table points,
three additional raw IV refusals also remain unchanged: zero price,
near-intrinsic quote rounding, and a bracketing failure. The raw CSVs retain
all of them. These are query-level outcomes, not table-build refusals.

The maximum pointwise change in recovered IV on the eligible common set is
0.85 bps. This small before/after difference should not obscure the absolute
error: for the 30-day table queried with **one day remaining**, ATM,
known volatility **22.5%**, the shared reference price is **$0.46719136**.
The table prices it at $0.35261925 before and $0.35251622 after, and recovers
**29.71698% before versus 29.72551% after**. The approximately 7.2 volatility
point error is already present in the baseline.

For successful eligible IV solves, changing the input reference price from
Ultra to High changes recovered IV by at most 0.374 bps, far smaller than
these worst-case approximation errors. This is another reference-sensitivity
check, not a mathematical error bound.

## Newly supported dividend neighborhoods

The baseline refuses all **135** of these table/query pairs. The candidate
returns finite prices and successfully recovers IV for all 135. There is no
baseline error number to compare against a NaN refusal.

| Table | New queries | Max price error | Median IV error | P95 IV error | Max IV error |
| --- | ---: | ---: | ---: | ---: | ---: |
| Manual 1Y, single reference strike | 27 | $0.013330 | 2.53 bps | 6.59 bps | 6.75 bps |
| Adaptive 1Y, multiple reference strikes | 81 | $0.034783 | 3.29 bps | 16.21 bps | 22.68 bps |
| Adaptive 30d, multiple reference strikes | 27 | $0.054809 | 46.32 bps | 121.71 bps | 122.64 bps |

For comparison, the candidate's just-outside-gap maximum IV errors are
6.74 bps (manual), 22.73 bps (adaptive one-year), and 122.88 bps (adaptive
30-day). The newly covered intervals therefore look like neighboring
supported times, not a new accuracy cliff. The adaptive configurations still
do not uniformly meet their requested 10 bps target.

## Reproduce and inspect

The common [harness](../../../event_snapshot_accuracy.cc) compiles against
both source revisions. Copy it and its BUILD target into both source trees,
then build each with `bazel build -c opt //benchmarks:event_snapshot_accuracy`.
Generate references only once, then pass the same file to both executables:

```sh
OMP_NUM_THREADS=8 OMP_PROC_BIND=false taskset -c 8-15 \
  /path/to/before/bazel-bin/benchmarks/event_snapshot_accuracy --references > references.csv
/path/to/before/bazel-bin/benchmarks/event_snapshot_accuracy --evaluate references.csv > before.csv
/path/to/after/bazel-bin/benchmarks/event_snapshot_accuracy --evaluate references.csv > after.csv
python3 summarize.py
```

Adjust the CPU mask for another machine. Put CSV files beside the summary
script. [environment.json](https://github.com/accelas/mango-option/blob/6ced24a01579b449d441d3bad7c2c9cd5b2a0d06/benchmarks/results/2026-09-14-event-snapshots/accuracy/environment.json) records binary, harness, and
reference hashes. [summary.json](https://github.com/accelas/mango-option/blob/6ced24a01579b449d441d3bad7c2c9cd5b2a0d06/benchmarks/results/2026-09-14-event-snapshots/accuracy/summary.json) includes counts, failures,
pointwise movement, and reference sensitivity; [summarize.py](summarize.py)
recomputes it from [references.csv](https://github.com/accelas/mango-option/blob/6ced24a01579b449d441d3bad7c2c9cd5b2a0d06/benchmarks/results/2026-09-14-event-snapshots/accuracy/references.csv), [before.csv](https://github.com/accelas/mango-option/blob/6ced24a01579b449d441d3bad7c2c9cd5b2a0d06/benchmarks/results/2026-09-14-event-snapshots/accuracy/before.csv),
and [after.csv](https://github.com/accelas/mango-option/blob/6ced24a01579b449d441d3bad7c2c9cd5b2a0d06/benchmarks/results/2026-09-14-event-snapshots/accuracy/after.csv). No production solver code was changed for this check.

Before publishing the PR update, the manual-fixture initializer was rewritten
to avoid a GCC 14 warning. Both revisions then reproduced the stored evaluation
CSVs byte-for-byte. The reference generator and its data were unchanged;
`environment.json` retains the original reference-generation fingerprints.

Recorded data links point to the immutable measurement commit; generated CSV/JSON/log files are excluded from the final PR diff.
