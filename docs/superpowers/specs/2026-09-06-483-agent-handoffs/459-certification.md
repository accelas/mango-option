# #459 — Certify physical prices and enforce IV identifiability

Gate 6. Read [contract.md](contract.md), especially sections 5–8.
Depends on #458, #460, and #461.

## Outcome

Public price-table factories and loading paths return sigma-nondecreasing
physical price surfaces or refuse. Public interpolated-IV construction
requires that certificate. A returned IV also satisfies a final-root
sensitivity check; legitimate exercise plateaus are not mistaken for
strictly invertible price curves.

## Phase A: certification and refusal

1. Inventory public construction/load/IV-construction paths across supported
   B-spline, Chebyshev, dimensionless, continuous, and segmented variants.
   State the model/domain each path represents. Done when there is no
   public uncertified IV-construction bypass.
2. Add RED cases: a negative-sigma pocket between scan points, a valid
   nondecreasing surface with a flat interval, a transformed/EEP surface,
   and an uncertified loaded/manual surface. Test public behavior.
3. Certify the final physical expression after transforms, EEP floor/add-back,
   and split composition. Use conservative polynomial/interval bounds or
   another justified proof method. A successful dense scan is not proof.
   Keep proof work bounded; distinguish witnessed nonmonotonicity from
   indeterminate certification or exhausted resources.
4. Enforce the certificate on public factories and loads; raw mathematical
   interpolants remain usable outside this financial guarantee.
5. Add final-root sensitivity admission. Preserve existing TV/K/reference-vega
   measurement filters and applicable query pre-checks. A healthy bracket
   probe does not make a poorly conditioned returned root trustworthy.
6. Once the public path is covered, remove the heuristic 17-point root scan
   and detect_multiple_roots configuration. Update callers/tests and the
   pending ABI handoff. Flat/non-identifiable IV outcomes retain explicit
   refusal semantics.

## Phase B: authorized shape-constrained fitting if needed

Measure Phase A on the corrected required cohort. If witnessed price-shape
violations prevent useful certified output, a separate fitting phase may
change the approximation method while meeting the same physical price/IV
criteria. If the limitation is conservative proof rather than an actual
violation, improve proof bounds/work allocation first.

Keep the requested domain and explicit resource constraints. Do not rename a
sampled heuristic a certificate or hide failures by narrowing the published
domain. Record which approximation changes, why it is needed, and its
price/Greek/IV/build/query consequences.

## Completion

All public paths are accounted for; uncertified construction/load is refused;
valid plateaus remain priceable; unidentifiable IV is refused; certified
identifiable examples solve correctly. False-positive/indeterminate failure
rates and proof cost are reported on the required/exploratory populations.
The applicable phase must serve the required workload before closing #459.

Persist only the agreed numerical metadata and compact historical accuracy
summary, tied to coefficients/model/domain. Recompute certification on load.
Historical measured accuracy and current proof status remain distinct.

Run surface/Greek, interpolated IV, factory, adaptive, serialization, and
Parquet tests. Coordinate ABI field removal with #463. This task does not
invent a per-query market/model uncertainty interval.
