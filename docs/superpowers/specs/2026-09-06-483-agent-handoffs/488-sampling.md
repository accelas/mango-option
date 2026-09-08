# #488 — End-to-end B-spline segmented samples

Gate 1. Read [contract.md](contract.md). Depends on #485.

## Outcome

B-spline segments are fitted from raw snapshots of the fixed-expiry PDE
solution. A fitted table is never the initial condition for the next dividend
segment. Existing temporal and reference-strike splits remain the query path.

## Sequence

1. Reproduce the issue's low-sigma case after #485's semantic correction.
   Capture a chained fitted initial condition and the corresponding raw
   end-to-end state for one K_ref/sigma/rate. Compare identical physical
   times and event sides. Done when any discrepancy is quantified or the
   proposed amplification mechanism is explicitly shown absent.
2. Add a public segmented pricing regression with low sigma and a query
   after a dividend crossing in backward time. Compare with a converged
   fixed-expiry direct oracle. Done when its RED result isolates sample
   chaining, rather than MultiKRef blending.
3. Run one fixed-expiry PDE per K_ref/sigma/rate, with exact snapshots and
   event-side semantics established by #485. Extract raw per-segment tensors,
   then use existing cubic fitting and split assembly. Done when no
   prev_spline/chained fitted-IC handoff remains in this path.
4. Remeasure documented adaptive configurations and attribute remaining
   errors/refusals to measured causes. Done when pins describe current
   behavior and stop attributing failures to removed chaining.

If the original cheap comparison shows no error amplification, record that
fact. The agreed raw-sample architecture still removes the intermediate fitted
handoff; do not claim an unmeasured accuracy improvement.

## Required behavior

Preserve cubic fitting and public configuration where it expresses the
selected contract. Default behavior is strict about failed raw rows; explicit
failure/repair allowance is counted by missing requested rows before repair,
not by a misleading count of solve jobs. Any permitted repair stays within
one temporal regime and is subjected to the same final validation.

Exact snapshots and valid per-segment node placement are mandatory. Preserve
raw sample identity when sharing a cohort grid. A concrete requested grid is
a constraint; changes to defaults follow the shared contract.

## Completion

Public pricing/Greek regressions, segmented builder/surface tests, PDE-cache
tests, and relevant factory/adaptive tests pass. Report sample count, PDE
solve count, cap hits, achieved errors, and build/query cost. A remaining
B-spline refusal may be pinned pending #458/#460; comparative promotion
belongs to final tuning after accuracy passes.

Inspect bspline_segmented_builder.cpp/.hpp, bspline_builder.cpp,
bspline_adaptive.cpp, and the #485 sampling/time helper. Remove only obsolete
chained-IC plumbing whose use is verified; preserve unrelated solver access.
