# Reference-strike evaluator checkpoint

This checkpoint adds a shared build-time evaluator and adapter. Public pricing
routes are not connected to them yet. The selector is a reference-approximation
policy; whole-table strict/best-effort publication remains a separate gate.

The evaluator compares direct fixed-expiry contracts with the positive,
linear-in-strike blend evaluated at `S_i=S*K_i/K`. It records ideal blending,
fitting, and composed price errors separately. Its IV channel is a
**price-error/reference-vega proxy**, not an actual inverted-IV error. The
final acceptance adapter must retain that evidence kind.

The physical population is declared before measuring candidates. Original
moneyness/strike corners and four non-dyadic interior strike fractions are
retained. Interior strikes also cross both moneyness tails and volatility/rate
endpoints. An explicit reference list adds representable interiors of every
served reference gap, so matching a fixed list of probe strikes cannot evade
measurement. Maturity probes remain inside actual admitted intervals, on both
sides of payments, with exact supplied ratio endpoints. Unsupported quote
corners are configuration errors, not filtered IV observations.

No-future-cash queries and exact reference-strike queries have structurally
zero ideal price perturbation. This identity does not fabricate an IV
observation or establish the accuracy of a fitted table. Error summaries
partition all requested rows into measured, structurally exact, filtered,
unresolved, refused, and untested counts. Numerical statistics cover measured
rows only; absent measurements have absent statistics.

Direct references use independent space, time, and domain ladders with seven
unique solves, at most 2,049 spatial points/2,048 nominal time steps in the
first round and one 4,097/4,096 retry. Domain extensions retain interior sinh
coordinates. The private evaluator ceiling is 8,192 solves, including failed
attempts; exhausting it leaves explicit incomplete evidence. Price allowances
must resolve 0.001 quote units or one tenth of a tighter requested target.
Proxy allowances similarly resolve 2e-6 decimal volatility or one tenth of a
tighter target. TV/K below 1e-4 or qualified small vega filters only the IV
channel. These are empirical convergence checks, not uniform mathematical
error bounds.

Vega and the ideal blend residual are assessed directly on common mesh
ladders. Noncontracting sequences remain unresolved. A qualified witness can
reject a sparse candidate without finishing its population, with remaining
rows recorded as untested. A failed composed-rescue attempt may likewise stop
on a qualified total-error witness when prior ideal inadequacy is established.
Successful composed rescue requires complete applicable measured evidence;
ideal structural identities cannot substitute for composed measurements.

## Evidence and limits

The original 50-row pilot is retained in the research logs. Individually
propagated price allowances cost 1,862 solves/318.08 seconds. Direct derivative
qualification alone cost 1,876/406.62 seconds and left more rows unresolved.
Structural identities, common residual checks and central-first witness
search reduced two-reference rejection to 21 solves/1.97 seconds. All 50
original rows remained accounted for; no passing subset replaced them.

That original population was insufficient for off-reference tail assessment.
The current population supplements it with crossed tail/volatility/rate
strata and explicit-gap probes. The original dense results are not an
adequacy claim: 17 and 33 references both retained unresolved observations.
A separate controlled audit confirms that an alternating, contracting coarse
residual can become noncontracting on retry; accepting all alternating
sequences would be unjustified.

The module tests cover structural identities without fabricated IV evidence
and an independently reproduced roughly three-cent sparse-reference error.
The public factory RED and a draft measured manual adapter are preserved in
the owner's work in progress. Remaining work includes complete public routing,
measured ordinary-case density/cost, and final fixture/binding alignment.
