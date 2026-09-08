# Epic #483: agent handoffs

Status: approved for implementation, 2026-09-06, by the user invoking
implement-spec after the grilling decisions. The policies below govern
the implementation task graph.

Read [the shared contract](contract.md) before any task. It is the single
source of truth for the agreed product semantics, acceptance policy, and
scope. Each task then supplies its own work sequence and completion criteria.

## Handoffs

| Issue | Gate | Task | Prerequisites |
|---|---|---|---|
| #485 | 1 | [Correct the fixed-expiry dividend timeline](485-timeline.md) | merged #484; reviewed math-fix baseline #489 |
| #488 | 1 | [Sample B-spline segments from an end-to-end solve](488-sampling.md) | #485 |
| #463, factory half | 2 | [Honor continuous Chebyshev adaptive configuration](463-factory.md) | #480/#484; independent of #485/#488 |
| #458 | 3 | [Stabilize clustered cubic fitting](458-fitting.md) | #488 |
| #460 | 4 | [Correct reference-strike semantics and selection](460-reference-strikes.md) | #488; sparse-reference characterization belongs here |
| #461 | 5 | [Preserve adaptive positions under ceilings](461-knots.md) | #458 and #460 |
| #459 | 6 | [Certify surfaces and enforce IV identifiability](459-certification.md) | #458, #460, #461 |
| #462 | 7 | [Enforce accuracy policy and retune the required cohort](462-acceptance.md) | #459 and completed public factory routing, including #463 factory half |
| #463, ABI half | 8 | [Publish diagnostics and the settled contract](463-abi.md) | #459 and #462 |
| #486 | non-blocking | [Characterize and correct manual Chebyshev defaults](486-defaults.md) | #485 for a meaningful measurement |
| #487 | non-blocking | [Resolve grid policy on the grid actually used](487-grid-policy.md) | #484 and #489 baseline; no new core-chain dependency |

Issue #480 is complete. Issue #443's interpolation work is superseded;
its remaining gamma allocation work belongs to #470, outside this packet.
Treat the two halves of #463 as separate agent tasks and separate PRs.
Closing #463 requires both halves.

The dependency graph governs scheduling. Independent investigation and RED
preparation may run concurrently. A behavior-changing PR waits for its actual
prerequisites. Keep behavior-changing gates in separate PRs. Final tuning
uses the completed public factory routes because its cohort exercises those
routes, not a substitute internal builder.

## Superseded decisions

The docs-only branch fix/485-488-segmented-solve contains a rev6 design and
per-maturity implementation plan. Their chain-maturity D0 decision and the
per-dividend-bearing-maturity solve architecture are superseded here.
They are historical evidence, not an implementation recipe for this packet.

The final choice retains the existing K_ref/temporal split composition:
fixed-expiry lifecycle semantics for dated cash dividends, and shared
cross-expiry reuse for time-homogeneous pricing. It adds no engine wall clock,
expiry-bank module, fleet-size input, or automatic maturity-cohort feature.

The draft's unapproved 0.5x error / 0.5x build-time / 0.1x query-time ratios
and 30-second / five-minute ceilings are not acceptance requirements.
Accuracy gates come first; comparative promotion follows measured query
latency. Build time and memory remain reported quantities.

## Empirical work is part of the task

An agent may choose contract-preserving private implementation details.
Characterization steps have specified inputs, metrics, and permitted next
steps. They may conclude that a reported premise is stale. They do not
authorize lowering agreed targets, changing the numerical model, expanding
scope, silently narrowing domains, or exceeding explicit resource ceilings.

If a required result remains impossible under the agreed contract, deliver
the smallest reproduction, the measured limitation, and the precise
additional decision needed. Do not close the issue as fixed or silently
choose a different contract.

## Evidence and publication

Implementation baseline reviewed in this session: 003bf126, PR #489.
Refresh actual issue/PR states before starting; green CI is not evidence
that a prerequisite has merged. An explicitly stacked branch must declare
its parent PR.

Each issue PR states its gate, prerequisites, RED/GREEN evidence, measured
accuracy and refusals, and applicable performance results. Preserve unrelated
working-tree changes.

GitHub issue bodies and the older design branch have not been edited by
this packet. These handoffs supersede conflicting historical suggestions.
