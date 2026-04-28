# Chebyshev 4D EEP: Rate-Axis Bottleneck and Node Budget Allocation

## Summary

Follow-up to the initial consultation on Chebyshev-Tucker 4D accuracy plateaus.
We implemented piecewise segments on the x-axis per the recommendation (hard dispatch,
not true spectral elements) and found no improvement. Axis ablation then showed that
increasing rate-axis nodes from 6 to 10 eliminates the long-maturity wing error,
though this also changes the extended domain (see confound note below). Tucker HOSVD
provides zero compression — the EEP function is full rank in all modes. Short
maturities (T < 60d) remain the primary unsolved problem.

We now seek guidance on optimal node budget allocation and whether to abandon Tucker.

## What we tried

### Piecewise Chebyshev on x-axis (no improvement observed)

Split x = ln(S/K) into 3 segments: ITM [−0.59, −0.10], ATM [−0.10, 0.15],
OTM [0.15, 0.45]. Each segment has 15 CGL nodes. Same 90 PDE batch, two-phase builder
(cache splines once, resample per segment).

Results (σ=30%, T≥60d, K=120):

| Method | T=60d | T=90d | T=180d | T=1y | T=2y |
|--------|-------|-------|--------|------|------|
| Global (40×15×15×6) | 64.6 | 46.9 | 24.0 | 23.3 | 29.4 |
| **Piecewise (3×15)** | **61.7** | **47.4** | **23.7** | **23.0** | **29.5** |

Wing errors are unchanged. Piecewise did solve 3 more probes (71/72 vs 68/72) but
the error magnitude is identical.

**Caveat:** This is not a true spectral-element method. Segments use hard dispatch
(nearest-segment lookup) with no overlap or blending at interfaces, so the
interpolant may be discontinuous at segment boundaries. The negative result
shows that x-axis resolution was not the bottleneck at these node counts, but
does not rule out piecewise methods in general.

### Axis ablation

Increased nodes on each axis individually while holding others constant:

| Experiment | PDE | T=180d K=120 | T=1y K=120 | Verdict |
|------------|-----|-------------|------------|---------|
| Baseline (40×15×15×6) | 90 | 24.0 | 23.3 | — |
| 60 x-nodes | 90 | 23.9 | 23.1 | No change |
| 25 tau-nodes | 90 | 25.1 | 23.8 | No change |
| 25 sigma-nodes | 150 | 26.9 | 22.0 | No change |
| **10 rate-nodes** | **150** | **3.4** | **1.4** | **Fixed** |
| All axes up (40×25×25×10) | 250 | 3.7 | 2.3 | Fixed |

In this test regime (fixed probe rate r = 0.05), only changing the rate axis
resolved the wing error. Individual improvements to x, tau, or sigma had zero
effect.

**Confound:** The headroom formula `3 × width / (n−1)` couples domain extent
to node count. Changing num_rate from 6 to 10 simultaneously changes degree
(5 → 9) and shrinks the extended domain ([−0.044, 0.154] → [−0.020, 0.130]).
The improvement could come from the higher degree, the tighter domain
concentrating more nodes in the region of interest, or both. A controlled
experiment holding the domain fixed while varying degree would isolate
the effect.

### Why rate was the bottleneck

The rate domain [0.01, 0.10] with headroom `3 × 0.09 / (6−1) = 0.054` per side
extends to [−0.044, 0.154] (width 0.198). With 6 CGL nodes (degree-5 polynomial),
the relative interpolation error is ~1e-3. For EEP ≈ 5 at ITM wings, that gives
~$0.005 price error → ~40 bps IV error.

With 10 nodes (degree 9), the relative error drops to ~1e-5, giving sub-bps accuracy.

The probe rate r = 0.05 is near the center of the extended domain (center = 0.055).
CGL nodes cluster at endpoints, so with only 6 nodes the interior has poor coverage.

### Tucker HOSVD: zero compression

| Config | Tucker ranks | Full tensor size |
|--------|-------------|-----------------|
| Global (40×15×15×6) | (35,15,15,6) | 54,000 |
| Piecewise seg 0 | (15,15,15,6) | 20,250 |
| Piecewise seg 1 | (15,15,15,6) | 20,250 |
| Piecewise seg 2 | (14,14,15,6) | 20,250 |

At ε=1e-8, Tucker retains full or near-full rank on every mode. The EEP function
is not low-rank in any mode. We added a `use_tucker=false` flag that skips HOSVD
entirely and stores the raw tensor — identical accuracy, avoids SVD overhead.

## Current best configurations

### 150 PDE (15σ × 10r) — best cost-accuracy tradeoff

```
              K=80    K=85    K=90    K=95    K=100   K=105   K=110   K=115   K=120
  T= 90d     2.6      3.4      3.1      3.0      0.1      2.6      4.1      8.3      2.8
  T=180d     3.3      2.9      2.3      0.7      0.7      0.3      3.2      3.6      3.4
  T=  1y     1.6      1.1      1.5      0.2      1.2      0.5      1.5      1.2      1.4
  T=  2y     1.4      1.1      1.2      0.3      0.0      0.1      2.3      0.8      1.1
```

T≥180d: 0.0–3.6 bps across all strikes. Comparable to B-spline (1044 PDE) at T≥1y
(both under 2.5 bps); at T=180d the Chebyshev is better (max 3.6 vs 14.2 bps)
though B-spline is slightly tighter at T=2y (max 1.1 vs 2.3 bps).

### 250 PDE (25σ × 10r) — higher sigma resolution

More sigma nodes improve T≥1y slightly but degrade T=90d at K=120 (12.5 vs 2.8 bps
for 150 PDE). The additional sigma nodes may introduce Runge-type oscillation at
short maturities where the EEP varies sharply in sigma.

```
              K=80    K=85    K=90    K=95    K=100   K=105   K=110   K=115   K=120
  T= 90d     2.8      3.5      2.9      2.8      0.1      1.8      2.6      5.0     12.5
  T=180d     3.5      3.0      2.1      0.6      0.5      1.3      2.3      3.6      3.7
  T=  1y     1.8      1.2      1.2      0.3      1.0      1.8      1.5      1.0      2.3
  T=  2y     1.2      1.0      1.3      0.4      0.4      0.4      1.2      1.8      1.8
```

### B-spline 4D reference (1044 PDE)

```
              K=80    K=85    K=90    K=95    K=100   K=105   K=110   K=115   K=120
  T= 90d     3.1      4.5      5.4      8.4      9.6     10.8      6.2      3.3     29.4
  T=180d     4.3      3.2      0.0      6.5     14.2     11.7      1.7      6.3      7.7
  T=  1y     1.6      1.0      1.1      0.5      1.2      1.1      0.3      0.8      1.3
  T=  2y     1.1      0.8      0.9      0.1      0.4      0.4      0.6      0.7      1.0
```

## Remaining problems

### 1. Short maturities (T < 60d) — the primary unsolved problem

T=7d–30d errors are 20–600+ bps with frequent Brent failures, especially at ITM
wings. This is structurally different from the long-maturity wing issue: at short
maturities the EEP changes rapidly near the free boundary, and no node-count
increase on any axis has addressed it. The B-spline adaptive grid handles this
better because it places extra knots near the exercise boundary; the Chebyshev
global polynomial has no equivalent mechanism. None of the experiments above
(piecewise x, axis ablation, rate-node increase) targeted or improved this regime.

### 2. σ=15% is harder than σ=30%

More Brent failures at low vol (49/72 vs 68/72). Low vol means smaller vega,
so price errors amplify more into IV errors. Also, the exercise boundary is
sharper at low vol, making the EEP function harder to approximate.

### 3. Storage scales with all axes

Without Tucker compression, storage is the raw tensor:
- 40×15×15×6 = 54,000 doubles = 422 KB
- 40×15×15×10 = 90,000 doubles = 703 KB
- 40×25×25×10 = 250,000 doubles = 1.95 MB

This is small in absolute terms, but grows as O(N₀ × N₁ × N₂ × N₃).

## Questions

1. **Is there a principled way to allocate the node budget across axes?**
   The axis ablation showed that per-axis degree matters more than total node count.
   Is there theory for choosing N_d per axis to equalize interpolation error
   contributions, given a fixed PDE budget N_sigma × N_rate?

2. **Should we drop Tucker entirely?** The EEP function is inherently non-separable
   in 4D. Tucker HOSVD adds build-time SVD cost and eval-time factor contractions
   for zero benefit. A plain tensor-product Chebyshev interpolant would be simpler
   and equally accurate. Are there functions in quantitative finance where Tucker
   actually helps, or is non-separability the norm for option pricing?

3. **What about the short-maturity regime?** T=7d–30d errors are 20–600+ bps.
   The B-spline adaptive approach handles this better because it places extra knots
   near the exercise boundary. Is there a Chebyshev-compatible strategy for
   short maturities — e.g., piecewise on the tau axis instead of x, or a
   coordinate transform specific to short maturities?

4. **Is there an alternative to headroom?** The 3 × width / (n−1) headroom
   formula extends each axis beyond the user domain. The polynomial interpolates
   on the extended domain (CGL nodes sit there), but the practical cost is that
   node budget is spent outside the region of interest. With 6 rate nodes, 2 sit
   entirely in the headroom zone, leaving only 4 to cover the user domain.
   Would clamped or "not-a-knot" boundary strategies work better than
   domain extension?

## Test setup

Same as previous consultation:
- American put options, S = 100, r = 5%, q = 0, K_ref = 100
- Strikes: 80, 85, 90, 95, 100, 105, 110, 115, 120
- Maturities: 7d, 14d, 30d, 60d, 90d, 180d, 1y, 2y
- Vols: σ = 15%, σ = 30%
- PDE solver: TR-BDF2, Ultra accuracy profile
- IV recovery: Brent's method on interpolated price

## Code

Branch `experiment/chebyshev-tensor`, worktree `.worktrees/chebyshev-tensor/`:

- `benchmarks/chebyshev_4d_eep_inner.hpp` — global + piecewise builders, `use_tucker` flag
- `benchmarks/interp_iv_safety.cc` — sections: `cheb4d`, `cheb4d-pw`, `cheb4d-diag`
- `src/option/table/dimensionless/chebyshev_tucker_4d.hpp` — `use_tucker` bypass
