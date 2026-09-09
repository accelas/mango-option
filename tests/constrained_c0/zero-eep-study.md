# Zero EEP plus physical intrinsic floor: frozen-output feasibility

This price-only calculation evaluates `max(European, intrinsic)` on the
unchanged 772 C0 CALL physical rows and original independent references.
It performs no PDE solve, table fit, IV inversion, or Greek/proof computation.
All 772 original reference hashes were verified before use.

| Population | Price measured / unresolved | Misses > .01 | Maximum error | RMS |
|---|---:|---:|---:|---:|
| All rows |757 /15|112|2.183371932|.181873160|
| Negative rates |195 /15|112|2.183371932|.358343266|
| Nonnegative rates |562 /0|0|3.02e-14|6.60e-15|

The worst row is `C0-CALL-8e3ffb3fa4819250e1c8`:
S=130,K=100,tau=2,sigma=.3,r=-.05,q=0. European value is 31.3803055204 and
intrinsic is 30. The independently qualified American reference is
33.5636774524±.0000947999. The missing premium is 2.183371932.

Of the 112 misses, 106 occur when European value exceeds intrinsic; 6 occur on
the intrinsic branch. 75 misses are at ratios [.95,1.05], 20 above 1.05, and 17
below .95. 89 have sigma>.1 and 23 sigma<=.1. 15 are at maturities<=30 days,
56 at 30 days<tau<=1 year, and 41 above 1 year. These are reporting strata only;
none changes eligibility or removes a row.

71 price misses have independently measurable reference IV, and 41 have
unresolved reference IV. None of the 153 legitimately IV-filtered rows misses
the price target. These are reference-eligibility counts, not measured IV
errors: no IV was inverted in this study. The full 555/153/64 independent
measurable/filtered/unresolved IV partition remains unchanged.

`max(European,intrinsic)` is the lower approximation obtained by retaining the
European alternative and immediate exercise. The substantial remaining gap
therefore distinguishes genuine early-exercise modeling from the old fit's
overshoot and below-intrinsic defects. Structural floors and the exact
nonnegative-rate branch do not eliminate the need to model that premium.
This result does not promote a new composition or claim a full Greek or
machine-arithmetic sigma certificate.

The analytic calculation uses the ordinary binary64 Black-Scholes formula
with `erfc` on finite ordinary-scale rows. Its 562 exact analytic controls agree
with the independent 50/100-digit references within 3.02e-14. The closest
measured error to the .01 classification threshold is .000707515 away; this
check is not sensitive to those rounding differences. The largest positive
signed discrepancy is 2.80e-11; all material errors are underestimates.

Artifacts are in main-repo `.cache/483-research/459-zero-eep-floor/`:
`rows.json` retains every physical request/reference status and price outcome;
`summary.json` contains complete group counts and worst rows. Reproduce using
`PYTHONDONTWRITEBYTECODE=1 python3 tests/constrained_c0/zero_eep_feasibility.py /absolute/main/repo`.
No source reference record, target, or required-population denominator changes.
