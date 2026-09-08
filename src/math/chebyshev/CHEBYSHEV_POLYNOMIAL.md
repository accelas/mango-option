# Explicit modal polynomial preparation

`mango::detail::ChebyshevPolynomial` is an explicit opt-in mathematical
representation. Existing `ChebyshevInterpolant`/`RawTensor` behavior and all
financial factories remain unchanged. The polynomial type owns its row-major
modal coefficients, shape, and physical coordinate bounds. Its mathematical
value is the tensor sum of `coefficient[i] * product(T_i(x_i))`; mode zero has
no additional half factor. Value, first partials, and second partials evaluate
that same expression and its analytical derivatives. Finite coordinates are
clamped independently to the physical domain; outside the differentiated axis,
partials are zero, with interior one-sided values at endpoints. Nonfinite or
wrong-rank queries yield NaN.

`from_coefficients` imports a polynomial directly. `from_cgl_values` performs
separable DCT-I conversion from samples ordered on ascending CGL tensor nodes.
The latter is an approximation/conversion step: long-double cosine matrix and
accumulation produce new stored binary64 modal coefficients. It does not claim
to preserve the rational function defined by rounded nodes with ideal
barycentric weights. It neither filters small coefficients nor reduces the
requested shape. Exactly constant sample lines are converted to an exactly
constant axis, an algebraic identity that preserves legitimate plateaus.

The current preparation capacity is rank 1–4, up to 257 coefficients per axis,
and up to 2^20 total coefficients. Unsupported capacities are refused; axes,
levels, and domains are never silently reduced. Zero-degree constant axes are
supported. The conversion matrix uses at most 257² long-double entries, and
one tensor-sized conversion scratch buffer. Query evaluation uses tensor basis
contraction without copying the coefficient tensor or storing separate Greek
payloads. The compiled evaluator uses the repository's normal optimized math
library pattern, with fast-math and contraction disabled.

The separate private `math/proof/chebyshev` seam only accepts this polynomial
type. It cannot be passed an old nodal/barycentric interpolant as if it were the
same expression. Derivative modal coefficients and physical coordinate scaling
are enclosed with retained MPFR128 endpoints. On restricted unit boxes, basis
ranges use `T_n(x)=cos(n*acos(x))`, with directed endpoints and all possible
interior cosine extrema included. This follows the [NIST DLMF identity](https://dlmf.nist.gov/18.5.E1)
and [MPFR directed mathematical functions](https://www.mpfr.org/mpfr-current/mpfr.html).
This avoids a high-degree monomial conversion whose rounding uncertainty could
otherwise dominate the proof.

A coarse bound is tried first. For eligible tensors, a Bernstein conversion
and the existing bounded subdivision kernel follow. This conversion is limited
to degree 64 per axis and 4096 effective coefficients. Only exactly zero modes
may be removed from proof scratch; the original polynomial payload and its
published shape remain unchanged. Higher-degree cases use deterministic
restricted-box subdivision directly in the Chebyshev representation. Both
paths respect the node/depth limits, keep signs at proof precision, and
separate a rigorous negative-box witness from indeterminate work/precision.
The 128-bit precision is fixed in this preparation; there is no silent
precision increase or claim that all valid polynomials can be certified.

The tests cover exact modal polynomial values/Greeks, conversion from nodal
inputs, tensor axes and clamping, exact flat directions, positive high-degree
partials, hidden negative pockets between 17 increasing scan prices, and an
oscillatory degree-256 negative witness through restricted intervals. An exact
rational checker independently verifies degree-256 value/partial enclosures
emitted by the manual characterization program. These finite tests supplement
the analytic inclusion argument; they do not replace it.

Run the arithmetic-only comparison with:

```sh
bazel run -c opt //benchmarks:chebyshev_polynomial_characterization > modal-report.txt
python3 benchmarks/check_chebyshev_proof_vectors.py modal-report.txt
```

The program reports conversion cost, coefficient payload, value/Greek errors,
query latency, and bounded proof work at 9/33/65/257 nodes and a 257×9×9×5
tensor. It compares the new polynomial and current barycentric representation
on identical nodal values. It does not benchmark or certify an American-option
financial cohort. In particular, small nodal errors can be amplified strongly
in high-degree second derivatives even when price reconstruction is excellent.

Before financial activation, preserve the explicit representation distinction
in query dispatch, certificate ownership, and saved payloads. Persistence needs
a modal-first-kind representation tag/version, coefficients with the stated
normalization, shape, and coordinate bounds. Existing nodal payloads cannot be
reinterpreted as modal coefficients. No serialization or public financial
construction path is changed here. EEP/floor/add-back, dimensionless physical
derivatives, split composition, final-root identifiability, required-cohort
accuracy, and load-time recertification remain separate Gate #459 obligations.
