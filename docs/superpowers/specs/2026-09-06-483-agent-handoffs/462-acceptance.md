# #462 — Enforce measured accuracy and retune final behavior

Gate 7. Read [contract.md](contract.md).
Depends on #459 and the completed public factory routes, including #463's
factory half. Reuse #460's sparse-reference characterization.

## Outcome

Build success, price-only usability, IV measurability, and best-effort output
have explicit meanings. Final configurations are selected on a fixed
supported population with honest accuracy/refusal/cost reporting.

## Sequence

1. Declare and version the required cohort and independent-reference method
   before tuning. Start with existing documented configurations; add both
   option types, supported rate regimes, low vol, short positive maturity,
   temporal/event sides, off-node moneyness, off-K_ref strikes, and independent
   spot movement. Use fixed-expiry lifecycle semantics on dated tables.
   Done when every case has a model, domain, expected measurability/status,
   and reference convergence check.
2. Keep exploratory stress cases separate. Count failed builds, failed
   queries, filtered IV measurements, and successful measurements for every
   algorithm. Done when RMS cannot improve merely by dropping cases and an
   all-filtered IV population cannot appear accurate.
3. Enforce requested build policy: strict targets by default, explicit
   best-effort only within hard viability/certification. Price-only builds
   may succeed with IV unmeasured; IV-targeted builds may not claim that
   target. Expose measured/filtered evidence for planned accuracy reporting.
4. Use agreed defaults: price 0.01 quote units; IV 2e-5 decimal volatility
   (0.2 absolute-IV bp). Correct the misleading 2-bp comments. Gate on max
   absolute measured error; report RMS and per-stratum maxima.
5. Fix benchmark contract/reference mismatches before tuning: current dividend
   sweeps mix maturity-scaled calendars and fixed-table calendars, and some
   strikes coincide exactly with K_refs. Use public factory/price/Greek/IV
   methods and matched contracts rather than duplicate Greek formulas.
6. Tune configurations satisfying the criteria. After accuracy passes,
   prioritize trading-hour query latency. Report build time, memory, PDE
   counts, K_ref density, proof work, refusals, and achieved errors together.
   Promote a backend only on measured comparable evidence.

## Completion

Every required pricing case passes; every independently IV-measurable
required case passes; expected non-identifiable cases have the correct
status. The benchmark process exits unsuccessfully for required-cohort
violations. Publish the selected configurations and reference population,
with regressions pinning the final behavior. Preserve tighter existing tests.

Do not inherit unapproved backend ratios or deployment deadlines. Respect
explicit resource limits; changes to defaults must be justified and measured.
If the agreed default target cannot be achieved, deliver the limitation
rather than quietly multiplying it by ten.

Inspect adaptive_grid_types, adaptive_metrics, adaptive_refinement,
price_table_factory, interp_iv_safety, IV sweep benchmarks, and the existing
accuracy/Greek tests. The separate accuracy-sweep audit from PR #489 is
evidence of coverage gaps, not a substitute for this task's final cohort.
