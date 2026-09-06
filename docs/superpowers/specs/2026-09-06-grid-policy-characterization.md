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
OMP_NUM_THREADS=1 bazel-bin/benchmarks/grid_policy_characterization constraints > grid-constraints.csv
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

## Matched ceiling comparison

The same benchmark source was compiled with `-c opt` against `003bf126` and
the completed grid-policy sources (`a470bbab`). Both direct populations contain
56 successful candidates. Only the low-short scenarios change spatial count:
1201 to 1199 under the 1200 ceiling. Fixed-alpha time steps change from 40 to 39;
geometry-alpha steps from 1441 to 1439. The largest change in measured maximum
error is 1.776585e-5 quote units. Other scenarios' prices are unchanged.

## Numerical cutoffs and normalized reuse

The initial implementation applied the historical width-5.8 routing heuristic
to the resolved grid. A matched batch with coverage [-3,3] then performed 20
original-contract solves instead of one normalized solve, increasing latency
from roughly .9 ms to 18 ms with identical points, time steps and prices.
That experiment does not justify a categorical cutoff. Width, spacing and
margin cutoffs have all been removed from reuse eligibility.

On a fixed log-moneyness grid, contracts in an eligible group have the same
sigma, rate curve, continuous yield, maturity and option type, with no cash
dividends. Their normalized value V/K satisfies the same PDE, normalized
payoff, obstacle and boundary conditions. Spot only selects where the resolved
solution is evaluated. Changing quote units cannot introduce an independent
width, spacing or clearance restriction on reuse. Il'in fitting and the
solver's existing admissibility checks apply equally to either solve.

Public regressions now require shared reuse on an explicit width-six grid and
on the full 20-contract automatic wide-coverage population, for both calls and
puts. Every quote agrees exactly with an independent regular solve on the same
grid. Additional regressions cover coarse explicit cells, narrow margins,
ordinary automatic inputs, impossible point intervals and PDE admissibility
failures against regular-route behavior. Coverage, explicit coordinates and
spatial ceilings remain unchanged.

The complete numeric-cutoff removal was measured with 20 contracts (spot 100,
strikes 90 through 109, maturity .5, sigma .2, rate .05) on two exact manual
uniform grids. Both have 101 points and 200 time steps: the coarse grid spans
[-3,3], the narrow-margin grid [-.15,.15]. Each timing averages 10 batches with
one OpenMP thread. The regular route is selected through its public debug
switch; the reuse route solves the same configurations. These measurements
compare equivalent numerical results, not accuracy against a finer PDE.

| Type | Grid | Regular solves | Reused solves | Regular batch | Reused batch | Maximum price difference |
|---|---|---:|---:|---:|---:|---:|
| Put | coarse | 20 | 1 | 31.18 ms | 1.44 ms | 0 |
| Put | narrow margins | 20 | 1 | 26.35 ms | 1.38 ms | 0 |
| Call | coarse | 20 | 1 | 27.84 ms | 1.81 ms | 0 |
| Call | narrow margins | 20 | 1 | 25.82 ms | 1.27 ms | 0 |

The separate before/after width-cutoff experiment compared `003bf126` with
`3e6e915f` on the same 20-contract population. Automatic grids used 101 points /
85 steps; wide coverage used 343 / 60; the fine explicit override used 401 / 1000.
Wide coverage returned to one solve and roughly .9–1.1 ms; the fine explicit
override with coarse automatic settings improved from 20 solves /294–297 ms
to one solve /14–16 ms. Ordinary automatic batches retained one solve at
roughly .4–.5 ms. All 120 quote comparisons retained exactly the same reported
maximum errors (all below .006 against High-profile direct references).
Timings were measured on a shared host; small timing differences do not support
claims about a performance regression or improvement. PDE reuse counts and
price equivalence are the decisive evidence for retiring numerical cutoffs.
