# FD Re-Anchoring Design

## Problem

When building segmented price surfaces split at discrete dividend dates,
each segment's initial condition (IC) comes from evaluating the previous
segment's B-spline fitted surface. B-spline fitting error compounds
across segments:

- Segment 0: payoff IC → PDE solve → B-spline fit (error ε₀)
- Segment 1: B-spline₀(boundary) → PDE solve → B-spline fit (error ε₀ + ε₁)
- Segment 2: B-spline₁(boundary) → PDE solve → B-spline fit (error ε₀ + ε₁ + ε₂)

At K=80, T=2.0, σ=0.30 with quarterly $0.50 dividends (4 segments),
this reaches ~272 bps — far beyond the 20 bps acceptance threshold.
Grid density improvements cannot fix this; the error is structural.

## Approach: Raw Boundary Snapshots

Replace B-spline lookups in the chaining step with direct interpolation
from post-extraction tensor values (pre-B-spline-fit). No extra FD
solves needed.

After each segment's `extract_tensor` and `repair_failed_slices` steps,
the 4D tensor `[moneyness, tau, sigma, rate]` contains PDE-computed
values at all grid points. The slice at `tau = tau_max` holds the
segment's far boundary values. Store this 3D slice as a **boundary
snapshot** and use it for the next segment's IC instead of the B-spline
surface.

Note: `extract_tensor` already performs spline resampling from the PDE
spatial grid onto the moneyness grid. The snapshot captures these
resampled values — more accurate than a subsequent 4D B-spline fit, but
not raw PDE grid values.

### Re-anchored chain

- Segment 0: payoff IC → PDE solve → tensor → repair → snapshot₀ + B-spline fit
- Segment 1: interpolate snapshot₀ → PDE solve → tensor → repair → snapshot₁ + B-spline fit
- Segment 2: interpolate snapshot₁ → PDE solve → tensor → repair → snapshot₂ + B-spline fit

Each segment starts from repaired tensor values (PDE discretization
error only). B-spline fitting error no longer compounds across
segments. B-splines are still fitted for query-time use — unchanged.

## Data Structure

```cpp
struct BoundarySnapshot {
    std::vector<double> moneyness;  // m grid (shared across all σ,r)
    std::vector<double> values;     // flattened [m × σ × r], row-major
    size_t n_vol;
    size_t n_rate;

    /// Interpolate at shifted moneyness for a given (σ,r) index.
    /// Builds a cubic spline for the moneyness slice and evaluates.
    double interpolate(double m_adj, size_t vol_idx, size_t rate_idx) const;
};
```

**Spline caching:** The IC callback is invoked per spatial node in the
PDE grid. Building a cubic spline per call is wasteful. Instead, cache
one `CubicSpline` per `(vol_idx, rate_idx)` pair inside the callback
(or pre-build all N_σ × N_r splines when constructing the snapshot).
With typical grids of 6×6 = 36 pairs, this is negligible memory.

## Integration Point

Inside `SegmentedPriceTableBuilder::build()`:

### Segment 0 (EEP path)

Segment 0 currently uses `builder.build(axes)` which does not expose
the intermediate tensor. To capture a boundary snapshot, refactor
segment 0 to use the same manual path as chained segments:
`make_batch → solve_batch → extract_tensor → repair → snapshot → fit`.

This is a code-path change, not a behavioral change — the same PDE
solve, extraction, repair, and fitting steps happen in both paths.

### All segments (unified flow)

After `extract_tensor` and `repair_failed_slices` succeed for segment k:

1. Grab the `tau_max` slice from `extraction->tensor`.
2. If EEP content (segment 0): convert to V/K\_ref by adding the
   analytical European price at each grid point (see conversion
   section below).
3. If RawPrice content (segments 1+): use tensor values directly
   (already V/K\_ref).
4. Store as `BoundarySnapshot`.

The IC callback changes from:

```cpp
// OLD: B-spline lookup through AmericanPriceSurface
double raw = ic_ctx.prev->price(spot_adj, K_ref, prev_tau_end, sigma, rate);
```

to:

```cpp
// NEW: cubic interpolation from cached boundary spline
double raw = ic_ctx.snapshot->interpolate(m_adj, vol_idx, rate_idx);
```

`ChainedICContext` drops the `AmericanPriceSurface* prev` pointer and
gains a `const BoundarySnapshot* snapshot` instead. The `prev_is_eep`
flag is no longer needed since the snapshot stores V/K\_ref uniformly.

### Tensor slice extraction

The tensor layout is `[moneyness, tau, sigma, rate]` with sizes
`[N_m, N_tau, N_sigma, N_r]`. The `tau_max` slice index is
`N_tau - 1`. For flat index `(m, tau, sigma, r)`:

```
flat_idx = m * (N_tau * N_sigma * N_r) + tau * (N_sigma * N_r) + sigma * N_r + r
```

Snapshot extraction iterates `tau = N_tau - 1` and copies all
`(m, sigma, r)` values. The snapshot re-flattens as `[m, sigma, r]`:

```
snap_idx = m * (N_sigma * N_r) + sigma * N_r + r
```

Use `axes.grids[0]` for moneyness, `axes.grids[2]` for vol indices,
`axes.grids[3]` for rate indices to ensure correct mapping.

## Interpolation

Use the existing natural cubic spline (`CubicSpline` in
`src/math/cubic_spline_solver.hpp`) for 1D interpolation along the
moneyness dimension.

Rationale:
- PCHIP was previously implemented in this codebase and removed;
  natural cubic was the deliberate choice (PR #338).
- Dividend shifts are small (0.3–1.3 grid cells at typical densities).

Clamp `m_adj` to `[m_min, m_max]` before spline evaluation (flat
extrapolation at both ends). Deep ITM prices are dominated by intrinsic
value; deep OTM prices approach zero smoothly.

**Grid density guard:** The builder enforces >= 4 moneyness points, but
cubic spline quality degrades on very coarse grids. The adaptive grid
builder targets 60+ moneyness points. For manual grids below ~16
points, the snapshot interpolation may not improve over B-spline
chaining. No special fallback is needed — the code works correctly at
any density >= 4, just with reduced benefit.

## EEP-to-RawPrice Conversion

Segment 0 stores EEP (Early Exercise Premium) in its tensor. The
boundary snapshot needs V/K\_ref. Convert at each grid point:

```
snapshot[m, σ, r] = (eep_tensor[m, τ_max, σ, r] + european(m, τ_end, σ, r)) / K_ref
```

Both EEP and European price are in dollar units. Dividing their sum by
K\_ref produces the normalized V/K\_ref that chained segments expect as
IC.

The European price is Black-Scholes closed-form — exact and cheap.
Later segments (RawPrice) need no conversion; tensor values are already
V/K\_ref.

## Edge Cases

- **Large dividend shifts:** `m_adj` outside grid range → clamped to
  `[m_min, m_max]` (flat extrapolation at both ends).
- **Failed PDE solves:** `repair_failed_slices` fills in failed slices
  before snapshot extraction. If repair fails, the segment build
  returns an error — no snapshot is created.
- **Single-segment case (no dividends):** No chaining, no snapshot.
  Code path unchanged.

## Scope

### Changed

- `segmented_price_table_builder.cpp` — refactor segment 0 to use
  manual build path; capture boundary snapshot after repair; rewrite
  IC callback to read from snapshot.
- `segmented_price_table_builder.hpp` — add `BoundarySnapshot` struct.
- Test file — verify chaining error reduction.

Estimated scope: ~80-120 lines changed (segment 0 refactor is the
largest piece).

### Unchanged

- Query-time behavior (B-spline lookup at ~500ns, identical API).
- B-spline fitting (still happens for every segment, used at query
  time).
- `AmericanPriceSurface`, `SplicedSurface`, `StrikeSurface`.
- Adaptive grid builder (`probe_and_build`, `build_segmented_strike`).
- Validation pass from PR #362.

## Related

- PR #362: Segmented IV accuracy diagnostics (validation pass,
  tau\_target\_dt, select\_probes improvements).
- Issue #363: Numerical EEP for chained segments — addresses
  per-segment B-spline fitting quality (exercise boundary kink in
  RawPrice). Orthogonal to re-anchoring; would compound the benefit.
