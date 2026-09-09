# Requested domains and numerical support

The continuous adaptive builders now retain the caller's moneyness, maturity,
volatility, and rate bounds for validation and publication. Numerical padding
does not expand those bounds. Both builders retain their historical numerical
minimum spreads and backend-specific headroom in the separate fit domain.
Supplied maturity seeds and explicit refinement ceilings remain unchanged.

## Reproduced problem

The frozen Phase-A CS-PUT request supplies maturities `{1, 2, 4, 7}/365`.
Previously, `extract_chain_domain` applied a minimum half-year spread to the
measurement domain. Validation consequently sampled `[1e-6, .500001]`, with
numerical maturity seeds ending in a gap from `7/365` to `.500001`.

In a matched `bazel -c opt` reproduction, the unchanged public request refused
with `NoViableSurface` in 78.6 seconds.
All eight candidate fits succeeded, all 64 holdout references were valid, and
62 holdout points were measured with finite values. No viable candidate was
discarded: every holdout maximum exceeded the unchanged `.20` viability bound.
The final candidate's worst point had maturity `.35866133499386504`, outside
the requested week. Its price was `22.099721440662226`, versus a cached direct
reference `11.390548803310853` with vega `23.411831227373987`; the IV-scaled
error was `.45742567223147446`. The fitted early-exercise premium contributed
`10.835942946718394` of that price. A saved-payload replay reproduces this
rejection in milliseconds without PDE solves.

After domain separation, the same default request builds and saves in
31.6 seconds, publishes exactly `1/365` through `7/365`, and rejects a
quarter-year query. Numerical support still extends to `.500001`; it is
not advertised as the caller's domain. The reported maximum IV error is
`.010185945693351841`, above the requested `2e-5`, with `target_met=false`.
This corrects the domain refusal; it does not complete accuracy tuning or
claim the requested target is met. No numerical threshold was relaxed.

## Singleton axes

Existing callers supply a single strike, volatility, rate, or maturity.
Such an axis retains its single requested coordinate while numerical grids
remain nondegenerate. Validation samples it at that coordinate. Error-bin
normalization avoids division by zero, refinement skips axes with no requested
interval, and an improvement does not reactivate those fixed axes.

Price-table serialization accepts finite ordered bounds with equality and
preserves them exactly. The coefficient grids retain their original numerical
support. Generic format-4 I/O already carries these scalar endpoints and their
checksum, so no schema field changes. Older typed loaders that require strict
inequality reject point-domain payloads rather than interpreting a wider
scope. The IV solver still validates its volatility bracket separately and
rejects a zero-width bracket; a point-domain price table is not an IV curve.

## Evidence and scope

Tests cover all four narrow requested axes, the public default CS-PUT build,
preserved maturity seeds and ceilings, fixed-axis sampling/refinement, existing
B-spline singleton use, continuous Chebyshev singleton use, and a public
all-point price-table save/load with IV-bracket refusal. The adaptive loop's
all-filtered and nonfinite-data rejection criteria remain in force.

The historical raw-node PDE-coverage fixture now declares its former effective
measurement ranges explicitly, retaining the original seeds, query classes,
and `1e-5`/`.2` price criteria. Its original narrow request is a separate
regression: all eight fixed holdout references have negligible time value,
so a vacuous zero maximum must still return `NoViableSurface`. This fixture
alignment changes neither frozen cohort nor numerical acceptance criteria.

The detailed baseline, per-iteration traces, exact spline payloads, replay
source, and after-fix measurements are retained under
`.cache/483-research/459-cs-put-diagnosis`. Diagnostic instrumentation is absent
from the production source. The frozen Phase-A manifest and the independently
running reference census were not modified. Continuous-domain certification
and final Gate-7 accuracy selection remain separate work.
