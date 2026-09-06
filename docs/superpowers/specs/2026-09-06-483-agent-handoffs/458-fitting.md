# #458 — Stabilize clustered cubic fitting

Gate 3. Read [contract.md](contract.md). Depends on #488.

## Outcome

The corrected segmented path can cross the former roughly 160-node fitting
cliff with valid clustered data. Failure handling distinguishes invalid input,
factorization failure, and unacceptable numerical error.

## Characterize before choosing an algorithm

1. Run the documented discrete-dividend reproduction on unchained samples.
   Save the actual failing axis, knot vector, RHS, requested tolerances,
   factorization status, residual/backward error, and condition estimate.
   Done when a small deterministic numerical fixture reproduces the failure,
   or corrected samples demonstrably eliminate the historical cliff.
2. Compare controlled node counts below/at/above the former cliff. Separate
   floating near-duplicates, knot admissibility, factorization problems,
   residual scaling, and propagation of PDE/sample error.
3. Choose the smallest contract-preserving repair inside the collocation/
   banded-solver module. Pivoting, scaling, factorization, and residual
   implementation are agent choices justified by the captured fixture.
   Keep cubic interpolation and chosen valid refinement positions.

Smoothing or shape-constrained approximation belongs to #459's authorized
phase, not an undocumented substitute for this fitting repair. A numerical
tolerance change needs mathematical justification relative to its documented
meaning; an increase chosen merely to hide the reproduction is not a fix.

## RED/GREEN and completion

Pin the captured public fitter failure and an end-to-end corrected segmented
case. Invalid duplicate/nonfinite/ill-ordered inputs must still fail correctly.
Validate raw interpolation accuracy against the source rows and public price
accuracy against the independent oracle.

Complete when the valid fixture and former-cliff reproduction succeed within
their error requirements, existing banded/cubic tests pass, and condition/
failure behavior is documented. If the old failure disappears after #488,
pin the absence and report the revised diagnosis instead of manufacturing
an algorithm change.

Inspect src/math/bspline/bspline_collocation.hpp,
bspline_nd_separable.hpp, src/math/*banded*, and tests named
bspline_condition_number_stress, bspline_collocation_1d,
bspline_fitter_4d_separable, and lapack_banded.
