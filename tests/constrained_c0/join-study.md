# C0 near-zero-rate exploratory study

The fixed 21-row diagnostic completed in 122.803 seconds under its 600-second
research execution limit. 20 prices qualified and one remains unresolved.
The existing 772 primary reference records are unchanged; every original hash
was checked. No second candidate was assembled or published.

The manifest was frozen before observations, digest
`40235a4fd53d5cb165dbca04434c3b70779186d769b76e7a48d7841f60fab626`.
Three anchors were replayed at
[-.00625,-.001,-.0001,-.00001,-.000001,-.0000001,0]:

- ATM: S=K=100, tau=30/365, sigma=.5;
- ITM_LOW: S=130,K=100,tau=2,sigma=.05;
- ITM_MID: S=130,K=100,tau=2,sigma=.3.

The reference owner supplied centered worker 056fa48f and the frozen 523834a4
qualification driver. Its three independent space/time/domain ladders,
three-round cap, .04 sigma bump fraction, .001 price budget, and 2e-6
vega-scaled IV budget were preserved. Analytic r=0 values use the independent
50/100-digit implementation; negative-rate American calls use direct FDE.
No new reference floor or Greek acceptance tolerance was introduced.

## Complete evidence counts

| Channel | Outcomes |
|---|---|
| Price |20 qualified; 1 oracle-unresolved |
| Independent IV eligibility |14 measurable; 4 time-value-filtered; 2 both-filtered; 1 unresolved |
| Point-left rho |Not qualified by these endpoint secants |
| New candidate price/actual IV/Greeks/proof |Not run; basis rejected before construction |

The unresolved row is ITM_LOW at r=-1e-5. Its third-round price is numerically
near 30, but the spatial sequence is noncontracting/oscillating under the
unchanged qualification policy. Neither its apparent proximity to intrinsic
nor the small differences was used to waive that policy.

The study issued 2013 unique worker requests: 2010 direct FDE and 3 analytic.
No worker response failed. Sum of completed worker durations is 237.137 seconds
across two workers, with reported PDE work 6,892,985,738. This is reference
qualification cost, not candidate build or query latency. Complete cases and
each expensive sample are checkpointed in the study directory.

## Rate sensitivity does not supply a broad quadratic transition width

Backward secants use the already declared rates and r=0 anchor. Their numerical
uncertainty is the sum of independently qualified endpoint price uncertainties
divided by h. They are interval-average rate sensitivities, not point rho.

For ITM_LOW the h=1e-6 secant is 60.3392±2.2754; at h=1e-7 it is 71.3569±22.2012.
The analytic right rho is 199.9762. These observations are consistent with a
very slow approach to the pointwise C1 limit; they do not establish a finite
width where a quadratic factor is accurate. The missing h=1e-5 price prevents
an unbroken convergence series. ATM and ITM_MID smallest-h secants become
numerically unresolved after amplification of endpoint uncertainty. All
values, sides, and uncertainties are retained in `summary.json`.

## Structural obstruction for the proposed knot at -.0125

This conclusion applies specifically to the declared composition
**European + max(0, raw EEP), without a physical intrinsic floor**.

With one double knot at -.0125 and a C1 zero join at 0, the rate spline is cubic
on each side of -.0125. Values E1..E4 at the four already sampled negative rates
[-.05,-.0375,-.025,-.0125] determine the raw midpoint premium:

`E(-.00625) = (-2 E1 + 9 E2 - 18 E3 + 35 E4) / 48`.

The formula follows from cubic differentiation on the left and Hermite
interpolation to zero value/derivative on the right. An independent Cox-de
Boor basis calculation agrees with those rational weights to 1.11e-16.

At ITM_LOW, interpolation of the existing construction controls gives physical
price 29.6878268283 at r=-.00625. The independently qualified reference is
29.99999999999±2.21689e-11: an error of -.31217317.

This is not merely an unfortunate unconstrained least-squares optimum. Under
the q=0 common discounted-stock coupling, every discounted stopping payoff
is nondecreasing in r. Thus for negative r,
`intrinsic <= American(r) <= American(0) = European(0)`.
For this anchor the true price at each of the four source rates is therefore
in [30,30.000200023]. Allowing .01 error at every source quote and applying the
exact rate identity yields the midpoint price range
[29.6744067035,29.7013400674]. This does not intersect its qualified target
range [29.99,30.01]. All corresponding raw premiums are positive, so the
existing zero-EEP floor does not alter the identity. The arithmetic uses
ordinary binary64 European add-back values with roughly .289 separation; this
is a capacity diagnostic, not an outward-rounded sigma certificate.

The source controls are existing construction points, separate from the 772
primary validation denominator. The witness excludes simultaneously accurate
pricing of these five valid physical quotes by this basis/composition. It is
not a claim that a particular full 772-row candidate was built or evaluated.

An additional **physical intrinsic floor** would raise the deficient midpoint
to 30 and evade this particular obstruction. That would be a different
composition, requiring matching query/proof behavior and complete measured
accuracy/Greek evidence. It would not resolve the separate OTM negative-vega
witness automatically. No such floor was added or implicitly assumed here.

The proposed no-intrinsic-floor basis is therefore unsupported at this last-cell
width. In accordance with the bounded experiment, it was rejected before a
second full fit. A next construction needs a separately declared narrower or
richer negative-rate basis, or an explicitly measured intrinsic-floor
composition. It must retain the 772 rows, all unresolved evidence, original
price/actual-IV criteria, and whole-domain physical proof after other-axis fit.

## Reproduction and artifacts

Main-repo artifacts are `.cache/483-research/459-rate-join-21/`: frozen manifest,
complete cases, SQLite sample cache, metadata/library digests, cost and status
summaries, rate secants, basis identity check, and capacity bounds. The first
failed candidate remains separately preserved in 459-constrained-c0/.

`join_diagnostic.py /absolute/main/repo` runs only the frozen study with the
approved existing worker/driver; it refuses to overwrite a completed study.
The original run used external `timeout --signal=TERM 600s`, OMP=2, and two
reference workers. `summarize_join.py /absolute/main/repo` replays the offline
summary and capacity arithmetic without a PDE solve. Research limits are not
user resource ceilings, and no timeout occurred in this study.
