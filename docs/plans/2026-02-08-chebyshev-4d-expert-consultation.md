# Chebyshev-Tucker 4D Approximation: Accuracy Plateau at Wings

## Summary

We approximate a 4D function (early exercise premium for American options) using
Chebyshev tensor-product interpolation with Tucker HOSVD compression. Interior
accuracy is excellent (2-5 bps), but errors grow monotonically toward the
moneyness wings, reaching 20-60+ bps at deep in-the-money strikes. We seek
advice on whether this is a fundamental limitation of the polynomial basis,
and what alternative approaches might break through.

## Context: What We're Building

We price American options by decomposing:

```
American(S, K, τ, σ, r) = EEP(ln(S/K), τ, σ, r) × (K/K_ref) + European(S, K, τ, σ, r)
```

where EEP (early exercise premium) is the non-negative excess of the American
price over the European price. European is computed exactly via Black-Scholes at
query time. The EEP is the function we approximate.

The end-to-end goal is implied volatility (IV) recovery: given a market price,
invert the price function for σ via Brent's method. The error metric is
|IV_interpolated − IV_reference| in basis points (1 bps = 0.01% vol).

### Production system (baseline): B-spline 4D

The production system uses a 4D B-spline surface on the same axes, built via
adaptive grid refinement. It achieves:

- **σ=30%**: 144 bps RMS (69/72 probes solved), interior T≥60d: 1-15 bps
- **σ=15%**: 866 bps RMS (57/72 solved), interior T≥90d: 1-20 bps
- **Cost**: 1044 PDE solves, 74s build time

### Experiment: Chebyshev-Tucker 4D

We replace the B-spline with Chebyshev tensor-product interpolation on
Chebyshev-Gauss-Lobatto (CGL) nodes, compressed via Tucker HOSVD. Current best:

- **σ=30%**: 156 bps RMS (68/72 solved), interior T≥60d: 2-21 bps
- **σ=15%**: 897 bps RMS (49/72 solved), interior T≥90d: 3-15 bps
- **Cost**: 90 PDE solves, 14s build time

The Chebyshev path uses **12× fewer PDE solves** and builds **5× faster**.
Interior accuracy matches B-spline. The problem is the wings.

## The Function Being Approximated

### EEP properties

EEP(x, τ, σ, r) where x = ln(S/K):

- **Non-negative** (enforced by a debiased softplus floor, see below)
- **Smooth** (C^∞ in the interior, smooth after softplus regularization)
- **Magnitude varies enormously across x**: For American puts at K_ref=100:
  - Deep OTM (x = +0.22, K=80): EEP ≈ 0
  - ATM (x = 0, K=100): EEP ≈ 0.5-3.0 (depends on τ, σ, r)
  - Deep ITM (x = −0.18, K=120): EEP ≈ 3-8
- **Monotonically decreasing in x** for puts (higher EEP for more ITM)
- **Increases with τ** (longer maturity → more exercise opportunity)
- **Increases with σ** (higher vol → more EEP, but complex interaction)
- **Non-separable in 4D** — Tucker HOSVD at ε=1e-8 yields full rank (40,15,15,6) on a 40×15×15×6 grid. Even ε=0 gives the same ranks, confirming the tensor doesn't compress.

### The softplus floor

Raw EEP can be slightly negative due to numerical noise. We regularize:

```
softplus(x) = log(1 + exp(100·x)) / 100
bias = log(2) / 100
eep = max(0, softplus(eep_raw) - bias)
```

This gives a smooth C^∞ function that's exactly 0 for large negative inputs.

## Approximation Method

### Grid and nodes

4 axes with CGL (Chebyshev-Gauss-Lobatto) nodes:

| Axis | Variable | User domain | Extended domain (with headroom) | Nodes |
|------|----------|-------------|---------------------------------|-------|
| 0 | x = ln(S/K) | [−0.50, 0.40] | [−0.569, 0.469] | 40 |
| 1 | τ (maturity) | [0.019, 2.0] | [0.0001, 2.424] | 15 |
| 2 | σ (volatility) | [0.05, 0.50] | [0.01, 0.596] | 15 |
| 3 | r (rate) | [0.01, 0.10] | [−0.004, 0.154] | 6 |

Headroom formula: 3 × domain_width / (n−1) per side, clamped to physical bounds.

Total tensor: 40 × 15 × 15 × 6 = 54,000 entries.
PDE cost: 15 × 6 = 90 solves (one per (σ, r) pair).

### Tucker HOSVD compression

Standard truncated HOSVD: unfold tensor along each of 4 modes, compute SVD,
truncate singular values below ε × σ_max, contract into compressed core.

At ε=1e-8, all ranks stay at full (40,15,15,6) — the function is not low-rank
in any mode. Even ε=0 gives the same result. Tucker compression provides no
benefit for this function.

### Evaluation

Barycentric Chebyshev interpolation per axis, contracted with the 4D core tensor:

1. Clamp query to domain bounds
2. For each axis d, compute barycentric weights → contracted vector of length R_d
3. Four-fold contraction with core: O(R0·R1·R2·R3)

Since ranks are full, this is equivalent to standard tensor-product Chebyshev
interpolation without Tucker.

### PDE sampling

For each (σ_j, r_k) pair, one PDE solve:
- Spot = Strike = K_ref (normalized PDE, returns V/K_ref)
- Maturity slightly beyond max τ node
- Snapshots at τ Chebyshev nodes
- Cubic spline resampling at x Chebyshev nodes
- EEP = K_ref × V/K_ref − European(x, τ, σ, r)
- Apply softplus floor

## Results: The Error Pattern

### σ=30% heatmap (Chebyshev 4D, 40×15 baseline)

```
              K=80    K=85    K=90    K=95    K=100   K=105   K=110   K=115   K=120
  T=  7d     ---     ---     5.6     1.2     0.8     37.2*    ---    636***  1025***
  T= 14d     ---     5.6     3.9     1.4     1.2     16.9*   20.5*  195***   243***
  T= 30d     2.2     1.1     2.7     4.1     2.8     10.0*    4.0    76.5**  263***
  T= 60d     2.1     3.1     3.1     4.6     3.3      7.8    20.4*   37.5*   64.6**
  T= 90d     3.6     4.0     3.2     4.5     3.3      6.5    13.6*   21.0*   46.9*
  T=180d     3.8     4.2     4.6     3.8     3.5      6.7    11.6*   18.5*   24.0*
  T=  1y     4.5     4.5     5.4     4.9     5.9      7.9    13.0*   17.4*   23.3*
  T=  2y     7.0     7.8     9.3    10.1*   12.0*    14.5*   19.1*   23.4*   29.4*

  RMS: 155.5 bps (68/72 solved)
```

Key observations:

1. **Monotonic OTM→ITM error gradient**: At every maturity, errors grow
   steadily from left (K=80, OTM) to right (K=120, ITM). For T≥60d:
   K=80 is 2-7 bps, K=120 is 24-65 bps.

2. **Short maturity × ITM wings are worst**: T=7d×K=120 = 1025 bps.
   These are also hard for B-spline (109 bps at same probe).

3. **Interior is competitive**: K=90-105 × T≥60d: 2-8 bps, matching
   B-spline's 1-11 bps in the same region.

4. **Error grows with τ at the wings**: K=120 goes from 65→47→24→23→29 bps
   as τ goes 60d→90d→180d→1y→2y. The U-shape at long maturities suggests
   the polynomial is oscillating.

### σ=30% heatmap (B-spline 4D, 1044 PDEs)

```
              K=80    K=85    K=90    K=95    K=100   K=105   K=110   K=115   K=120
  T=  7d    1155***  ---     22.1*    1.6     3.9     28.6*    ---    172**   109**
  T= 14d     30.7*   ---      3.3     1.7     2.2     14.1*   69.8**  53.6**  83.8**
  T= 30d      1.8    2.5      1.4     3.2     1.3      5.3    10.3*   45.0*  169**
  T= 60d      2.4    3.0      2.4     1.1     4.1      1.2     3.7    14.9*   14.3*
  T= 90d      3.1    4.5      5.4     8.4     9.6     10.8*    6.2     3.3    29.4*
  T=180d      4.3    3.2      0.0     6.5    14.2*    11.7*    1.7     6.3     7.7
  T=  1y      1.6    1.0      1.1     0.5     1.2      1.1     0.3     0.8     1.3
  T=  2y      1.1    0.8      0.9     0.1     0.4      0.4     0.6     0.7     1.0

  RMS: 143.8 bps (69/72 solved)
```

B-spline achieves 0.3-1.3 bps at T≥1y across all strikes — essentially flat.
Chebyshev reaches 5-29 bps in the same region. The B-spline's adaptive
refinement places extra knots where needed; Chebyshev has a fixed node layout.

## What We've Tried

### Node count scaling (helped, with diminishing returns)

| Config | σ=30% RMS | Interior (T≥60d, K=90-110) |
|--------|-----------|---------------------------|
| 10×10 | ~500 bps | 20-40 bps |
| 25×10 | ~300 bps | 5-15 bps |
| 40×15 | 156 bps | 2-8 bps |

Going from 10→40 x-nodes gave 3× improvement. Diminishing returns beyond 40.

### Tucker compression (no effect)

At ε=1e-8, ranks are full (40,15,15,6). Setting ε=0 gives identical results.
The EEP function is not low-rank in any mode. Tucker provides no compression
benefit for this function.

### Coordinate transforms (all hurt)

Implemented with individual on/off switches:

| Transform | Idea | σ=30% RMS | Verdict |
|-----------|------|-----------|---------|
| Baseline (none) | — | 156 bps | Best |
| sinh(α·u) on x, α=3 | Cluster nodes near ATM | 330 bps | **Worse** — starves wings |
| √τ coordinate | Cluster nodes at short maturities | 138 bps (68/72) | **Marginal** — new OTM failures |
| log(EEP + ε) | Smooth sharp transition | 1033 bps | **Catastrophic** — log(0+ε)=−23 |
| All three combined | — | 1033 bps | Worst |

**Why sinh_x failed**: The accuracy bottleneck is at the wings (large |x|), not
at ATM. Concentrating nodes near ATM removes nodes from exactly where they're
needed.

**Why log_eep failed**: For OTM puts, EEP ≈ 0 after the softplus floor.
log(0 + 1e-10) = −23. The Chebyshev polynomial must now approximate a function
ranging from −23 to +2, with a sharp transition near ATM. This is harder to
interpolate, not easier.

**Why sqrt_tau was marginal**: The EEP varies smoothly in τ for the maturities
of interest (7d-2y). The short-maturity regime where EEP changes rapidly is
exactly where EEP ≈ 0, so the IV error is dominated by vega ≈ 0, not by EEP
interpolation error.

## The Core Question

The Chebyshev interpolant produces excellent interior accuracy (2-5 bps) at
90 PDE solves vs B-spline's 1044. But the monotonic wing error (growing to
20-65 bps at K=115-120 for T≥60d) appears to be a fundamental property of
global polynomial interpolation: errors distribute according to the polynomial's
equioscillation properties, and the function's steep gradient at ITM puts pushes
those errors toward the wings.

**Is there a way to achieve B-spline-like flat error profiles (1-3 bps at all
strikes for T≥1y) using a Chebyshev-like approach with O(100) PDE solves?**

Specific questions:

1. **Is this the expected behavior for Chebyshev interpolation of a monotone
   function with a large dynamic range?** The EEP varies from 0 to ~8 across
   x ∈ [−0.5, 0.4]. A degree-39 polynomial should approximate well, but the
   end-to-end IV metric amplifies price errors at ITM strikes where vega is
   moderate but EEP is large.

2. **Would piecewise Chebyshev (spectral elements) help?** Split the x-axis
   into 2-3 segments, each with ~15 CGL nodes, with C¹ matching at segment
   boundaries. This would give local adaptivity while keeping the spectral
   convergence rate within each element. The cost is additional implementation
   complexity and deciding where to split.

3. **Would rational approximation (AAA, Padé, or rational Chebyshev) handle
   the wing behavior better?** The EEP has a transition from ~0 to large values
   that polynomial approximation handles poorly. Rational functions can
   represent such transitions with fewer degrees of freedom.

4. **Is there an algebraic reparameterization of EEP that would flatten the
   function for better polynomial approximation?** We tried log(EEP + ε) which
   failed due to the log-zero problem. Other ideas: EEP/K (normalize by strike),
   EEP/intrinsic_value, or something domain-specific.

5. **Is the non-separability (full Tucker rank) expected or a red flag?**
   The EEP function on a 40×15×15×6 grid has full rank in all modes. Does
   this indicate the function is genuinely high-rank in 4D, or could a
   different basis (e.g., orthogonal polynomials adapted to the weight
   function) expose hidden low-rank structure?

## Test Setup

- American put options on a non-dividend-paying stock
- S = 100, r = 5%, K_ref = 100
- Strikes: 80, 85, 90, 95, 100, 105, 110, 115, 120
- Maturities: 7d, 14d, 30d, 60d, 90d, 180d, 1y, 2y
- Vols tested: σ = 15%, σ = 30%
- IV reference: FDM (finite difference method) via PDE solver
- IV recovery: Brent's method on the interpolated price function
- PDE solver: TR-BDF2 with Newton iteration, Ultra accuracy profile

## Code Pointers

All code is on branch `experiment/chebyshev-tensor`:

- `src/option/table/dimensionless/chebyshev_tucker_4d.hpp` — interpolant class
  (CGL nodes, barycentric eval, domain clamping, finite-difference partial)
- `src/option/table/dimensionless/tucker_decomposition_4d.hpp` — HOSVD
  (mode unfold, truncated SVD per mode, sequential core contraction)
- `benchmarks/chebyshev_4d_eep_inner.hpp` — EEP adapter and builder
  (PDE sampling, softplus floor, coordinate transform switches)
- `benchmarks/interp_iv_safety.cc` — IV error benchmark
  (heatmap generation, Brent IV solve, B-spline comparison)
