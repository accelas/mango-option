# Event-sided snapshot performance comparison

Measured 2026-09-14 on AMD Ryzen 9 9955HX, GCC 14.2.0, optimized Bazel
builds. Baseline: `8ef8f5829b42d043f73c8109a9b4ee4da5cb8bd2`, which already
contains PR #503's validation-domain fix. Candidate: the uncommitted
event-sided snapshot implementation on `feature/dividend-event-snapshots`.
The production patch hash, common harness hash, timed binary hashes, and
machine details are in [environment.json](https://github.com/accelas/mango-option/blob/6ced24a01579b449d441d3bad7c2c9cd5b2a0d06/benchmarks/results/2026-09-14-event-snapshots/environment.json).

The measurements show no material performance regression in these cases.
Single-thread dividend-table construction is 3–6.5% faster. Ordinary PDE
and query medians change by less than 0.7%. Eight-core construction results
are mostly small differences; the three-dividend manual build improves
4.5% consistently across all six pairs. The larger event coverage comes
without a measurable capture-on/capture-off timing penalty in the fixed-grid
control. These are fixture results, not a guarantee for all workloads.

## Method

- Created separate source copies for baseline and candidate; copied exactly
  the same benchmark harness and target into both. No production sources
  were swapped in the working directory.
- Built `//benchmarks:event_snapshot_impact` and `//benchmarks:greek_latency`
  with `-c opt --jobs=8`. Benchmark translation units use the repository's
  `-O3 -march=native` convention; dependency flags are identical in both.
- Six independent paired runs, alternating before/after and after/before.
  Runs were serial, after compilation finished. Each benchmark uses at
  least 0.25 seconds of measurement and 0.05 seconds of warmup. An adaptive
  build longer than that is measured as one complete iteration per run.
- Reported numbers are medians of six wall-clock per-operation estimates.
  They are warm measurements, not cold-process startup or latency percentiles.
- Single-thread runs use CPU 8. Parallel construction uses eight workers
  within physical CPUs 8–15, `OMP_PROC_BIND=false`, `OMP_DYNAMIC=FALSE`,
  and `OMP_WAIT_POLICY=PASSIVE`. Worker affinity was inspected explicitly.
- Initial parallel runs with automatic OpenMP binding were discarded after
  inspection showed all workers confined to CPU 8. They are excluded from
  every result below. Single-thread runs were unaffected.
- CPU frequency scaling remained enabled. Small changes around 1% should
  be treated cautiously; run ranges and paired deltas are in
  [summary.json](https://github.com/accelas/mango-option/blob/6ced24a01579b449d441d3bad7c2c9cd5b2a0d06/benchmarks/results/2026-09-14-event-snapshots/summary.json). All [raw JSON results and logs](https://github.com/accelas/mango-option/tree/6ced24a01579b449d441d3bad7c2c9cd5b2a0d06/benchmarks/results/2026-09-14-event-snapshots/raw/) are archived in the linked measurement commit; generated output is excluded from this PR’s final diff.

The existing `greek_latency` benchmark supplies the price/Greek cases.
`component_performance` hit a system Arrow/zlib link error in both isolated
copies; its two direct-pricing cases were reproduced in the focused harness
without that unrelated factory dependency. No linker workaround or Arrow
setting difference is present in the measured builds.

## One-thread results

| Operation | Before | After | Change |
| --- | ---: | ---: | ---: |
| Direct PDE: ATM put, no cash dividends | 0.4547 ms | 0.4545 ms | -0.03% |
| Direct PDE: ATM call, three cash dividends | 0.5516 ms | 0.5525 ms | +0.16% |
| Manual table: no dividends | 15.7229 ms | 15.7613 ms | +0.24% |
| Manual table: one dividend | 17.1720 ms | 16.6310 ms | -3.15% |
| Manual table: three dividends | 18.9102 ms | 17.6844 ms | -6.48% |
| Adaptive table: 1Y, quarterly dividends | 1597.8369 ms | 1536.4950 ms | -3.84% |
| Adaptive table: 30d, dividend on day 10 | 398.5491 ms | 375.8650 ms | -5.69% |
| Fixed-grid solve: capture disabled | 7.1968 ms | 7.1909 ms | -0.08% |
| Segmented price query | 0.2513 µs | 0.2519 µs | +0.24% |
| Segmented vega query | 0.4645 µs | 0.4675 µs | +0.65% |
| Segmented price + five Greeks | 2.8270 µs | 2.8260 µs | -0.04% |

## Eight-thread construction results

| Operation | Before | After | Change |
| --- | ---: | ---: | ---: |
| Manual table: no dividends | 15.162 ms | 15.136 ms | -0.17% |
| Manual table: one dividend | 2.803 ms | 2.763 ms | -1.44% |
| Manual table: three dividends | 3.474 ms | 3.317 ms | -4.53% |
| Adaptive table: 1Y, quarterly dividends | 805.850 ms | 795.685 ms | -1.26% |
| Adaptive table: 30d, dividend on day 10 | 190.393 ms | 188.959 ms | -0.75% |

The no-dividend build uses a different, normalized batch path, so its poor
scaling should not be interpreted as a dividend-capture regression; before
and after are effectively unchanged. The short adaptive eight-thread case
has overlapping before/after run ranges (188.94–192.31 ms versus
186.73–193.89 ms); its small median improvement is not compelling evidence
of a speedup. The three-dividend manual case improves in every pair
(3.6–5.5%).

## Capture and memory cost

The fixed-grid control uses 201 spatial points, 1,008 actual time steps,
21 ordinary snapshots, and three dividend events. Only capture is toggled
within the candidate; the PDE grid is identical.

- Capture disabled: **7.1909 ms**.
- Capture enabled: **7.1899 ms** (-0.014%, indistinguishable at this measurement resolution).
- Capturing three extra rows adds **4,824 bytes (4.71 KiB)** of sample
  payload in this control, excluding metadata and allocator overhead.
- `sizeof(Grid<double>)` grows from **264 to 352 bytes**, an **88-byte**
  object-size increase even when capture is disabled.
- In the manual table fixtures, total raw sample rows remain **80 / 160 /
  320** for zero / one / three dividends. Event times replace two separated
  ordinary samples with one ordinary and one event-side sample, so table
  construction does not double snapshot payload. These are accounting
  observations, not peak-RSS measurements.

All repeated direct-pricing control outputs agree exactly across versions.
The capture-enabled and capture-disabled fixed-grid prices also agree exactly.

## Why construction improves

After timing, debugger inspection of the actual shared PDE grids found:

| Manual build | Before time steps | After time steps | Spatial points | PDE solves |
| --- | ---: | ---: | ---: | ---: |
| One dividend | 249 | 240 | 101 | 16 |
| Three dividends | 262 | 240 | 101 | 16 |

The old inset endpoints create extra mandatory intervals around each event.
Exact shared event boundaries remove those extra time steps. This explains
the observed benefit while the controlled capture cost is negligible.
The inspection used separately rebuilt symbol-bearing binaries, and no
measurements under the debugger enter the timing results.
[Accounting data](https://github.com/accelas/mango-option/blob/6ced24a01579b449d441d3bad7c2c9cd5b2a0d06/benchmarks/results/2026-09-14-event-snapshots/grid-accounting.json) and the [inspection script](inspect_grid.gdb)
are included.

## Accuracy and work counters

Both adaptive configurations build successfully in every run, with the
same reported refinement iteration counts and sample-point counts.
The reported holdout IV-error estimates are:

| Adaptive fixture | Before | After | Refinement iterations |
| --- | ---: | ---: | ---: |
| 1Y, quarterly $0.50 dividends | 46.2824 bps | 45.3553 bps | 8 / 8 |
| 30d, $0.50 dividend on day 10 | 376.1868 bps | 375.8254 bps | 5 / 5 |

Both miss the requested 10 bps target, as before. These are each build's
own reported holdout scores; the candidate includes the former gaps, so
they are not an independent common-query accuracy comparison. The aggregate reported
PDE-work counters change from 1,216 to 1,217 and 1,151 to 1,154 as validation
now includes the old event neighborhoods. Manual construction still uses
16 fixed-expiry PDE solves per reference strike. The prior 56-placement
replay remains 53 successful builds and the same three accuracy refusals;
these performance measurements do not establish that those numerical
limitations are resolved. Query timings use times supported in both
versions, not comparisons between a baseline NaN refusal and a real price.

## Reproduce

Build the two source revisions in separate directories with the identical
[focused harness](../../event_snapshot_impact.cc) and its BUILD target copied
to both. The baseline API intentionally lacks the capture-enabled control,
which the runner excludes for that revision.

```sh
bazel build -c opt //benchmarks:event_snapshot_impact //benchmarks:greek_latency --jobs=8
python3 benchmarks/results/2026-09-14-event-snapshots/run_comparison.py \
  --before /path/to/before/bazel-bin/benchmarks \
  --after /path/to/after/bazel-bin/benchmarks \
  --out /tmp/event-snapshot-results
```

CPU masks in [run_comparison.py](run_comparison.py) are specific to this
machine; choose equivalent physical cores on another host. Binaries should
be rebuilt without the debugger flags before timing.

## Follow-up: independent accuracy comparison

A subsequent [matched-query accuracy check](accuracy/README.md) uses one
shared High/Ultra FDM reference dataset and actual IV recovery for both
revisions. It confirms nearly unchanged common-domain accuracy, successful
IV solving throughout the tested former gaps, and substantial pre-existing
short-remaining-life approximation errors. These findings are more
informative about accuracy than the builders' own holdout scores above.

Raw output can be regenerated with the scripts below and is ignored by Git.
