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
from raw PDE-computed tensor values. No extra FD solves needed.

After each segment's `extract_tensor` step, the 4D tensor
`[moneyness, tau, sigma, rate]` already contains PDE-computed values at
all grid points. The slice at `tau = tau_max` holds the segment's far
boundary values. Store this 3D slice as a **boundary snapshot** and use
it for the next segment's IC instead of the B-spline surface.

### Re-anchored chain

- Segment 0: payoff IC → PDE solve → tensor → snapshot₀ + B-spline fit
- Segment 1: interpolate snapshot₀ → PDE solve → tensor → snapshot₁ + B-spline fit
- Segment 2: interpolate snapshot₁ → PDE solve → tensor → snapshot₂ + B-spline fit

Each segment starts from raw tensor values (PDE discretization error
only). B-spline fitting error no longer compounds across segments.
B-splines are still fitted for query-time use — unchanged.

## Data Structure

```cpp
struct BoundarySnapshot {
    std::vector<double> moneyness;  // m grid (shared across all σ,r)
    std::vector<double> values;     // flattened [m × σ × r], row-major
    size_t n_vol;
    size_t n_rate;

    /// Interpolate at shifted moneyness for a given (σ,r) index.
    double interpolate(double m_adj, size_t vol_idx, size_t rate_idx) const;
};
```

## Integration Point

Inside `SegmentedPriceTableBuilder::build()`, after `extract_tensor`
succeeds for segment k:

1. Grab the `tau_max` slice from `extraction->tensor`.
2. If EEP content (segment 0): convert to V/K\_ref by adding the
   analytical European price (Black-Scholes, closed-form) at each
   grid point.
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
// NEW: cubic interpolation from raw boundary values
double raw = ic_ctx.snapshot->interpolate(m_adj, vol_idx, rate_idx);
```

`ChainedICContext` drops the `AmericanPriceSurface* prev` pointer and
gains a `const BoundarySnapshot* snapshot` instead. The `prev_is_eep`
flag is no longer needed since the snapshot stores V/K\_ref uniformly.

## Interpolation

Use the existing natural cubic spline (`CubicSpline` in
`src/math/cubic_spline_solver.hpp`) for 1D interpolation along the
moneyness dimension. Rationale:

- PCHIP was previously implemented in this codebase and removed;
  natural cubic was the deliberate choice (PR #338).
- With 64+ moneyness points (grid spacing ~0.016), the exercise
  boundary kink is well-resolved.
- Dividend shifts are small (0.3–1.3 grid cells).

For extrapolation beyond the moneyness grid (large dividends pushing
`m_adj` below `m_min`), use flat extrapolation — return the boundary
value.

## EEP-to-RawPrice Conversion

Segment 0 stores EEP (Early Exercise Premium) in its tensor. The
boundary snapshot needs V/K\_ref. Convert at each grid point:

```
snapshot[m, σ, r] = eep_tensor[m, τ_max, σ, r] + european(m, τ_end, σ, r) / K_ref
```

The European price is Black-Scholes closed-form — exact and cheap.
Later segments (RawPrice) need no conversion.

## Edge Cases

- **Large dividend shifts:** `m_adj` below grid minimum → flat
  extrapolation (deep ITM prices dominated by intrinsic value).
- **Failed PDE solves:** Existing `repair_failed_slices` fills in
  failed slices before snapshot extraction. If repair fails, the
  segment build already returns an error.
- **Single-segment case (no dividends):** No chaining, no snapshot.
  Code path unchanged.

## Scope

### Changed

- `segmented_price_table_builder.cpp` — capture boundary snapshot
  after `extract_tensor`; rewrite IC callback to read from snapshot.
- `segmented_price_table_builder.hpp` — add `BoundarySnapshot` struct.
- Test file — verify chaining error reduction.

### Unchanged

- Query-time behavior (B-spline lookup at ~500ns, identical API).
- Segment 0 build (EEP with payoff IC).
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
