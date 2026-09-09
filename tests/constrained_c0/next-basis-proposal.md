# Next basis proposal after the stopped C0 candidate

No next sample panel or fitted table has been run. Existing 772 required rows,
reference qualification, published domain, targets, and resource caps remain.

## Separate findings

The first active-set implementation falsely treated redundant tight constraints
as a terminal failure. An independent three-variable KKT example demonstrates
the bug and its repair. All four seam tests pass; final feasibility still
checks all original inequalities. The original candidate attempt and its
133 failed/227 solved block statuses remain immutable. It has not been rerun.

Independently, the original two-degree negative-rate subspace has inadequate
construction-sample capacity: inside the published domain, the best raw-sample
least-squares residual has RMS .0829467 over 6720 samples and maximum 1.030757.
The worst block at ratio 1.3,tau 2 has RMS .363682. The worst row is sigma .05,
r=-.0125. The full 7200 samples include 480 points on numerical support outside
the published ratio range; their errors are reported separately.

The nearest sampled negative rate dominates: its inside-domain error is
largest at low sigma and long maturity. Therefore the first piecewise
refinement belongs near the analytic-rate join, rather than allocating more
positive-rate sample planes.

## Smallest representation refinement supported by the capacity audit

A predeclared algebraic audit used the same existing samples, with no PDE
rerun, candidate assembly, or validation fit. Add one interior knot at -.0125:

| Interior multiplicity | Free negative-rate coefficients | Inside raw sample RMS | Max |
|---|---:|---:|---:|
| None |2|.0829467|1.030757|
| One (C2) |3|.0123901|.1353795|
| Two (C1) |4|1.54e-15|1.95e-14|

All variants retain the double rate knot at 0 and an exactly zero positive
branch. The last row can interpolate four negative sample planes; its tiny
residual is expected algebra, not evidence of financial accuracy or shape.
Of these audited alternatives, the smallest refinement that removes the
demonstrated sample-capacity defect is a double knot at -.0125: eight stored rate coefficients, still
well within the existing 160-point cap, with four trailing coefficients zero.
It preserves C1 rate continuity and the existing payload/evaluation/proof type.

Do not build that candidate yet merely because its nodal residual vanishes.
There are no construction samples between -.0125 and 0. An additional
near-boundary sample/Greek qualification panel is needed to decide the last
interval's width and whether another negative knot is necessary. This is an
identified evidence gap, not a license to drop rows or widen tolerances.

## Mathematical status of the zero derivative trace

The zero value trace is supported by the exact cash-free CALL identity at
q=0,r>=0. Healy's Proposition 1 establishes the corresponding no-early-exercise
region and explains why it fails at negative rates
([primary paper](https://arxiv.org/pdf/2109.15157)). That proposition alone does
not establish rate-derivative matching.

The following is our mathematical inference, reviewed with the proof owner,
conditional on the standard early-exercise-premium representation for this
nondegenerate finite-horizon diffusion. Fix finite positive S,K,sigma,T and
q=0. Under the common discounted-stock coupling,
`exp(-r*t) (S_t^r-K)+ = (X_t-K*exp(-r*t))+`, with X independent of r.
The payoff difference at every stopping time <=T is bounded by
`K max_{t<=T}|exp(-r*t)-1|`, giving continuity of the American value in r.
For each fixed t<T, the limiting state is finite positive and European time
value at r=0 is strictly positive. Thus it cannot be an exercise state for
sufficiently small negative r. The exercise indicator tends to zero.

For q=0 and r<0 the premium representation is
`EEP(r)=(-r) K integral_0^T exp(-r*t) P_r(exercise at t) dt`.
Bounded dominated convergence then gives `EEP(r)/(-r) -> 0` as r approaches
zero from below. Consequently the pointwise left rate derivative of EEP is
zero. The terminal time has zero integral measure; sigma=0/T=0 are excluded from
the strict-time-value argument. This yields **o(|r|), not O(r²)**. It does not
justify a fixed-width quadratic rate factor, nor establish a useful uniform
numerical transition width over the entire domain.

The frozen row S=130,K=100,sigma .05,T=2 has analytic time value approximately
.000200023 and right rho 199.976. Their ratio is approximately 1.0e-6. This is
only a linearized intrinsic-crossing scale, not an exercise-boundary oracle,
but it makes the gap from the nearest sampled rate -.0125 material. At still
shorter, deep-ITM points, time value can be much smaller. A broad last rate
cell may satisfy the correct limiting derivative and still be inaccurate.

## Bounded next evidence step before selecting a new fit

Recommend an exploratory join panel, kept separate from the 772 immutable
required rows, at three already declared physical anchors:

- S=K=100, tau=30/365, sigma .5 (the observed nonzero price trace);
- S=130,K=100,tau 2,sigma .05 (thin deep-ITM transition and rho concern);
- S=130,K=100,tau 2,sigma .3 (same physical location with ordinary time value).

At each anchor use the fixed rate schedule
`[-.00625,-.001,-.0001,-.00001,-.000001,-.0000001,0]`, 21 physical rows total.
This is a proposed finite diagnostic panel, not an executed qualification or
new final-cohort denominator. Measure independently qualified prices, and
one-sided/centered rate sensitivities with explicit convergence and uncertainty;
retain failures/filtered/unresolved rows. Existing actual-IV measurability is
independent of fitted tables. No inferred derivative is silently promoted to
an independently measured Greek.

This panel would determine whether the minimal double knot at -.0125 needs a
second near-zero knot before another candidate is frozen. No knot search or
solver retry follows automatically from a miss. Any later construction still
requires whole physical sigma proof after all other-axis fits, all 772 required
price/actual-IV/Greek outcomes, matched optimized build/query timing, and the
same original acceptance criteria and explicit work caps.
