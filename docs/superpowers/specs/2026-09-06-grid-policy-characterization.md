# Grid policy characterization (#487)

The required regressions test grid ownership and exact numerical constraints
through grid estimators, direct/batch pricing, IV, and the B-spline builder's
public raw-tensor transform. They check actual-grid routing, preserved explicit
coordinates/clustering, automatic coverage, strict odd spatial ceilings, and
typed rejection of impossible point intervals. Exact explicit-grid samples
agree with direct pricing before the table fit.

The separate exploratory population below assesses default accuracy/cost. It
is not a uniform error certificate and is not the epic's final supported cohort.
No default was tuned using this population.

## Reproduction

```sh
bazel build -c opt //benchmarks:grid_policy_characterization
OMP_NUM_THREADS=1 bazel-bin/benchmarks/grid_policy_characterization > grid-policy.csv
OMP_NUM_THREADS=1 bazel-bin/benchmarks/grid_policy_characterization batch > grid-routing.csv
```

The executable contains the fixed population and reports every failed solve.
Each of seven scenarios is run for puts and calls at log-moneyness -1, 0, 1:

- sigma .01, maturity .03;
- sigma .05, maturity .5;
- sigma .20, maturity .5;
- sigma .50, maturity 2 (resolved width exceeds 5.8);
- sigma .10, maturity .2501, cash dividend 1.5 at calendar offset .25;
- sigma .10, maturity .2499, cash event removed;
- sigma .20, maturity 1, cash dividend 1.5 at calendar offset .25.

Spot/strike anchor is 100; rate .05; continuous yield zero on no-cash scenarios
and .02 on the cash/event-removed scenarios. Event-removed and near-expiry
contracts are separate characterization inputs, not a calendar-time roll of
one fixed-expiry table. Invalid expired events must be removed from the input.

References use clearance 8 and fixed 2001/4001 spatial points, with time ceilings
8000/16000 and c_t=.25. Report the reference difference alongside each candidate
error; the largest observed reference change is .00149 in the .01-sigma
short-maturity case. This limits interpretation of submill errors there.

## Results

All 56 candidates solved. Errors below are maximum absolute quote-unit errors
over the 21 evaluation points per type. Costs sum the seven solves per type;
they are local measurements, not deployment latency promises.

| Type | Policy | Maximum error | Scenarios over .01 | Sum solve time |
|---|---|---:|---:|---:|
| Put | clearance1 | .48512 | 2/7 | 36.3 ms |
| Put | default clearance 3 | .02782 | 1/7 | 44.6 ms |
| Put | clearance6 | .00797 | 0/7 | 86.5 ms |
| Put | geometry-derived alpha | .09060 | 1/7 | 146.4 ms |
| Call | clearance1 | .02068 | 1/7 | 36.9 ms |
| Call | default clearance 3 | .02068 | 1/7 | 46.1 ms |
| Call | clearance6 | .00924 | 0/7 | 84.7 ms |
| Call | geometry-derived alpha | .02068 | 1/7 | 144.3 ms |

The width 7.071 put solves with 143 points / 96 steps and .00849 error (reference
change .000102). Width alone cannot justify rejecting it. Geometry-derived
alpha uses the actual resolved half-width in diffusion lengths. It increases
short-low-sigma time steps from 39 to 1439, and worsens the near-expiry dividend
put from .02782 to .09060. Clearance 6 improves the sampled boundary errors but
nearly doubles summed solve cost and does not establish a uniform guarantee.
Retain the fixed alpha and clearance 3 defaults pending broader acceptance
measurements. A caller requiring .01 needs measured accuracy/refusal; default
grid heuristics alone do not establish it.

The low-short scenario hits the spatial ceiling: maximum 1200 now yields 1199
rather than 1201. Ultra maximum 5000 similarly yields 4999 rather than 5001.
CSV rows include width, actual points/steps, cap hits, solve time, maximum error,
reference change, and failure status. Mandatory event insertion may add time
steps; this change makes spatial ceilings strict, not temporal ceilings.

## Explicit-grid compatibility

An old test asked for 21 uniform points but depended on the builder silently
substituting 101 points. Honoring 21 reveals a raw B-spline deep-OTM undershoot
of -1.52105e-7 normalized. That coverage test now explicitly requests automatic
estimation with 101 points and keeps its price assertions. A separate regression
requires exact manual-grid fidelity. Coarse manual grids can expose approximation
errors previously hidden by replacement; the later physical-price certification
and measured-accuracy gates must assess/refuse inadequate final surfaces.
