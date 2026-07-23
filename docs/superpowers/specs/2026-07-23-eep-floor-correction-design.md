# Exact EEP Nonnegative Projection Design

**Issue:** #434

**Status:** Approved

**Date:** 2026-07-23

## Context

Price-table construction stores the early exercise premium (EEP),

\[
\mathrm{EEP} = V_{\mathrm{American}} - V_{\mathrm{European}},
\]

and reconstructs an American price by interpolating the stored EEP and adding
the analytical European price.

The shared `eep_floor` helper currently applies a debiased softplus. Although
the bias correction makes zero map to zero, it subtracts
\(\log(2)/100 \approx 0.00693\) from ordinary positive EEP values. A separate
overflow guard switches abruptly to the identity when the raw EEP exceeds
`5.0`, creating a discontinuity there.

The current behavior therefore violates the intended EEP transform in two
ways:

- valid positive EEP values are systematically understated; and
- the transform jumps at the overflow-guard boundary.

The existing query path already clamps a negative interpolated EEP to zero.
The construction path should use the same mathematical constraint.

## Goals

- Preserve every finite nonnegative EEP exactly.
- Map every finite negative EEP to zero.
- Remove the discontinuity at EEP `5.0`.
- Keep build-time and query-time nonnegativity semantics consistent.
- Apply the correction to every backend through the shared EEP decomposition
  helper.
- Add tests that directly specify the transform rather than relying only on
  broad end-to-end price tolerances.

## Non-goals

- Denoising positive PDE residuals.
- Making the projection differentiable at zero.
- Changing interpolation algorithms or coordinate transforms.
- Changing public interfaces or call sites.
- Migrating, invalidating, or versioning previously persisted price tables.
- Repairing values already stored in existing price-table artifacts.

## Decision

Replace the debiased softplus with the exact projection onto the valid EEP
domain:

\[
P_{[0,\infty)}(x) = \max(0,x).
\]

Conceptually, `eep_floor` becomes:

```cpp
inline double eep_floor(double eep_raw) {
    return std::max(0.0, eep_raw);
}
```

The helper's contract covers finite inputs produced by the pricing pipeline:

- `x < 0` returns `0`;
- `x == 0` returns `0`; and
- `x > 0` returns `x` exactly.

Non-finite inputs remain outside this helper's contract. Existing upstream
pricing and table-validation boundaries remain responsible for finite data;
the hot path gains no exceptions or additional validation.

## Mathematical Rationale

The hard floor is the Euclidean projection of a scalar onto the closed convex
set \([0,\infty)\). It provides the properties required of a domain
constraint:

- identity on valid inputs;
- exact enforcement on invalid inputs;
- continuity and monotonicity;
- idempotence, \(P(P(x)) = P(x)\);
- non-expansiveness; and
- positive homogeneity, \(P(cx) = cP(x)\) for \(c \ge 0\).

A differentiable function cannot be identically zero immediately to the left
of zero and identically equal to `x` immediately to the right: its one-sided
derivatives would have to be both zero and one. Smooth approximations must
therefore alter valid EEP values or fail to enforce exact zero. That tradeoff
is inappropriate for a domain projection.

The existing debiased softplus is already non-differentiable at zero because
its output is wrapped in `max(0, ...)`. Replacing it with the exact projection
does not abandon smoothness that the current implementation actually
provides.

If positive numerical noise later requires suppression, it should be handled
as a separate, error-aware policy. The nonnegative projection must not also
serve as an implicit noise filter.

## Architecture and Data Flow

The architecture and interfaces do not change:

1. An EEP strategy computes the matching European price.
2. `eep_decompose` or `compute_eep` calculates
   `american_price - european_price`.
3. `eep_floor` projects the raw result onto \([0,\infty)\).
4. The projected dollar EEP is stored in the interpolation surface.
5. At query time, the leaf interpolates EEP, applies its existing nonnegative
   clamp, scales by `strike / K_ref`, and the EEP layer adds the analytical
   European component.

All B-spline, Chebyshev, dimensionless, adaptive, accessor-based, and per-point
paths continue to use the shared helper and inherit the correction without
signature changes. Removing the softplus also removes its sharpness constant,
transcendental operations, bias correction, and overflow branch.

## Documentation

Update current comments and mathematical/architecture documentation that
describe the transform as softplus-based or smooth. They should instead state
that EEP is projected exactly onto the nonnegative domain.

Historical design documents remain historical records and are not rewritten.

## Test Design

### Direct projection contract

Add focused tests for `eep_floor`:

- negative values, including small and large magnitudes, map exactly to zero;
- positive zero and negative zero produce zero;
- representative positive values such as `0.005`, `0.02`, `1.0`, values
  immediately below and above `5.0`, and a large value are returned exactly;
- samples around the former `5.0` guard demonstrate continuity and identity;
- applying the projection twice gives the same result; and
- representative positive scale factors commute with the projection.

Exact comparisons are appropriate for the identity cases because the selected
positive input is returned directly.

### Shared decomposition paths

Use a controlled EEP strategy and accessor to exercise both:

- bulk `eep_decompose`; and
- per-point `compute_eep`.

Verify that a small positive raw EEP survives unchanged, zero remains zero,
and a negative residual becomes zero. This proves that both public
decomposition paths retain the shared projection semantics.

### Integration regressions

Strengthen EEP/table integration coverage to include:

- calls and puts;
- short and long expiries;
- reconstructed prices no lower than their analytical European component; and
- at least one positive-EEP case whose expected interpolation error is small
  enough to use an absolute price tolerance below the former
  \(\log(2)/100\) bias.

The targeted absolute-tolerance regression must fail against the old
debiased-softplus implementation. Broad percentage tolerances may remain for
separate end-to-end checks but are not sufficient evidence for this defect.

### Regression verification

Run the focused EEP and price-table test targets, followed by the repository's
interpolation-IV safety benchmark. The benchmark must remain within its
established accuracy thresholds.

## Acceptance Criteria

- `eep_floor(x) == 0` for representative finite `x <= 0`.
- `eep_floor(x) == x` exactly for representative finite `x > 0`.
- No special behavior or discontinuity exists at EEP `5.0`.
- Bulk and per-point decomposition obey the same projection contract.
- Call/put and short/long-expiry integration coverage passes.
- A regression test demonstrably detects the former approximately `$0.00693`
  positive-EEP bias.
- Existing focused EEP and price-table tests pass.
- The interpolation-IV safety benchmark remains within its established
  thresholds.
- No serialization or public API changes are introduced.

## Alternatives Considered

### Plain softplus

Plain softplus is smooth and continuous but produces a positive EEP of
\(\log(2)/100 \approx 0.00693\) when the true EEP is zero. This affects cases
such as non-dividend American calls at any expiry, not only low-vega
short-dated options.

### Compact smooth transition

A piecewise smooth transition can be zero for negative inputs and become the
identity above a chosen threshold. It still understates valid small positive
EEP and introduces an arbitrary scale-sensitive threshold.

### Scale-aware softplus

Scaling sharpness with `K_ref` would make the approximation relatively
scale-aware, but it would retain a nonzero result at exact zero, require
additional plumbing, and remain an approximation where an exact projection is
available.
