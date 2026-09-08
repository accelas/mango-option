# Private proof arithmetic preparation

This module encloses raw mathematical polynomials. It does **not** certify a
financial price table, activate IV refusal, or constrain any fitter. Its
Bazel visibility is limited to project sources and tests. Gate #459 must
compose the final physical expression and bind successful proof to immutable
coefficients, model, and domain before any public publication decision.

`Interval` imports each binary64 constant exactly and keeps directed MPFR
endpoints at 128 bits. Sign predicates use those retained endpoints. Directed
binary64 conversions are diagnostics, including when a nonzero bound exports
as signed zero. Invalid domains, nonfinite operands, and constants that cannot
be imported exactly under externally modified MPFR exponent settings cannot
become successful bounds. No hardware rounding-mode changes are needed.
Precision escalation is not implemented in this preparation slice; callers
receive an indeterminate result when the current arithmetic cannot resolve a
sign. MPFR allocation and arithmetic status are separate from mathematical
inclusion and are not a pricing uncertainty estimate.

`BernsteinTensor` holds coefficient enclosures over a unit box in row-major
order. The convex hull of its coefficients bounds the polynomial because
Bernstein basis functions are nonnegative and sum to one. Midpoint de
Casteljau subdivision uses enclosing operations at every step. A deterministic
depth-first proof returns one of:

- `Certified`: every visited leaf has a nonnegative coefficient lower bound.
- `NegativeWitness`: a reported unit sub-box has a strictly negative upper
  bound for the polynomial throughout that box.
- `Indeterminate`: work/depth limits or arithmetic uncertainty prevent proof.

A negative coarse coefficient alone is neither a witness nor a rejection of
the mathematical property. Limits apply to node visits and subdivision depth;
the representation also caps rank at 4, degree at 64, and total coefficients
at 4096. Depth is capped at 52 so witness box labels remain exact binary64
dyadics. The depth-first stack retains at most one pending sibling per level.
Zero work budget produces indeterminate, never vacuous certification.

`extract_cubic_bspline_cell` applies the Cox-de Boor recurrence to Bernstein
polynomials on one explicit stored-knot cell. It does not refit samples or
regenerate knots. The selected derivative first differences the original
stored coefficients and then extracts the degree-two derivative spline, so
constant coefficient lines retain exact zero even when another coordinate's
basis conversion is inexact. Derivative scaling includes the physical knot
width. Fully clamped Bernstein cells use an exact identity conversion.

The extractor validates finite, ordered, cubic-clamped knots, coefficient
shape, and the requested positive-width cell. Interior knot multiplicity above
three is refused: such a discontinuity would need a separate join-direction
proof. It currently validates the entire input on each extraction; a future
immutable validated payload can amortize that work. The function does not
certify all cells, grid clamping, right-endpoint tolerance snapping in the
existing B-spline evaluator, EEP/floors, dimensionless coordinates, Chebyshev
expressions, temporal routing, or reference-strike blends. Those are remaining
publication obligations, not implicitly certified extensions of this seam.

Tests use exact dyadic polynomial controls, including a negative derivative
pocket between 17 positive scan points; a square with a negative coarse
Bernstein coefficient; legitimate flat then rising intervals; nonuniform
cubic identities; all four tensor derivative axes; and witness sign retention
below binary64's smallest subnormal. Runtime financial evaluation is untouched.

The explicit opt-in Chebyshev modal representation and its restricted-box proof
support are described in
[CHEBYSHEV_POLYNOMIAL.md](../chebyshev/CHEBYSHEV_POLYNOMIAL.md). They share the
stored polynomial between values, Greeks, and proof; existing barycentric
financial evaluators are not switched by this preparation module.
