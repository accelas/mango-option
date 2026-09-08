# Cubic collocation fitting

Generated cubic knots preserve the supplied data sites. Interior knots follow
sites `x[2]` through `x[n-3]`; when there are at least two interior knots, the
outer two lie at the midpoints of `(x[1], x[2])` and `(x[n-3], x[n-2])`.
Endpoints retain multiplicity four. Four sites define a single cubic; five
sites use the single interior knot `x[2]`.

This is a data-aligned cubic construction with midpoint endpoint supports,
not the canonical not-a-knot condition. It replaces proportional placement
across the entire data-site index range, which developed nearly dependent
basis columns on large grids. Canonical not-a-knot and averaged knots both
stabilized the captured system, but exceeded an existing coarse log-grid
approximation tolerance. Midpoint endpoint supports preserve that tolerance.
No data sites move, interpolation remains cubic, and explicitly supplied or
persisted knot vectors retain their exact values.

Issue #458's corrected dividend-sample reproduction is pinned in
`tests/data/bspline_458_dividend_axis.txt`: 281 moneyness sites, historically
expanded from 120 requested sites, axis 0 / slice 0 after successful rate, volatility and
time fits. The old matrix's estimated condition was about 1e21 and its
absolute residual 0.942 at tolerance 1e-6 despite successful LAPACK
factorization and solve. On the same sites/RHS, the selected construction's
dense control has condition about 43.5 and residual below 1e-17.

The captured fixture retains that history. The current raw sampler consumes
supplied interpolation axes exactly; its end-to-end regressions explicitly
provide 118, 160, 188, and 281 sites and check the actual sample counts.

Regressions also cover cubic polynomial reproduction between nodes, actual
raw dividend builds through the former fitting cliff, and public prices and
Greeks. At a cubic knot, a central second difference may have a first-order
error because the third derivative jumps. The Greek consistency reference
uses Richardson cancellation with a second-scale convergence check, keeping
its previous numerical tolerance.

The fit tolerance remains an absolute collocation residual tolerance (1e-9
for the 1D fitter, 1e-6 per axis for separable fitting). It is not a bound on
interpolation error between nodes or a monotonicity certificate. Strongly
nonuniform sites can remain ill-conditioned; the condition estimate remains
part of the diagnostics. Shape constraints and final physical-price
certification belong to #459.

Nonfinite data sites fail before constructing a matrix. Nonfinite solved
values cannot pass the residual gate. Numerical failures retain the existing
`FittingFailed` category, with `message` identifying factorization, solve,
or residual rejection; residual errors also preserve `max_residual`.
The separable fitter stops at the first failed axis in its existing reverse
axis order, retaining that error and the lowest failing slice index
deterministically across threads. It does not fit later axes from a partially
failed tensor. `grid_size` identifies the failed axis length and `index`
identifies the axis; the message carries the slice index.
