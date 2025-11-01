# Implied Volatility Surface — Precomputation & Interpolation Design Guide

This document summarizes best practices and theoretical results for constructing a **fast implied volatility lookup** via **finite-difference precomputation** and **interpolation**.

---

## Overview

Goal: Compute implied volatility values efficiently at runtime by:
1. **Precomputing option prices** on a structured grid (via a finite-difference PDE solver).
2. **Storing prices (and optionally vegas)**.
3. **Interpolating** at query points for fast lookups.
4. Optionally **inverting price → implied volatility** using a fast root or interpolation method.

This approach replaces expensive per-query PDE solves with sub-microsecond interpolation.

---

## 1. Precomputation Grid Setup

### Coordinate Transformations
Use coordinate transforms to regularize behavior:
- **Log-moneyness**: \( x = \log(K/S) \)
- **Scaled time**: \( \tau = \sqrt{T} \)
- **Total variance (optional)**: \( w = \sigma^2 T \)

These reduce curvature in the surface and lower the number of grid nodes required.

### Grid Dimensions
Typical 2D grid: `(x, T)`
Typical 3D grid: `(x, T, σ)` (for precomputing price vs volatility)

| Accuracy Target | Grid Size (x × T) | Memory (price + vega) | Comment |
|------------------|------------------|-----------------------|----------|
| Low (~few bps)  | 50 × 30          | ~25 KB                | Coarse, fast precompute |
| Medium (~1 bp)  | 100 × 80         | ~128 KB               | Good default |
| High (<0.5 bp)  | 200 × 160        | ~512 KB               | High accuracy |
| 3D (x, T, σ)     | 100 × 80 × 40   | ~2.7 MB (price only)  | For direct inversion |

Memory scales as:

\[
\text{Memory} = 8 \times F \times \prod_i n_i \quad \text{bytes},
\]
where \( F \) = number of stored fields (price, vega, etc.).

---

## 2. Interpolation Scheme

### Recommended Method
**Tensor-product cubic spline interpolation**
- Order: 3
- Smoothness: \( C^2 \)
- Error order: \( O(h^4) \) per dimension
- Excellent balance between smoothness, speed, and local support.

### Alternatives
| Method | Pros | Cons |
|--------|------|------|
| Linear | Fast, simple | Large error (~O(h)) |
| Quadratic | Reasonable | Not C² continuous |
| Cubic spline | High accuracy, C² smooth | Slight setup cost |
| RBF (global) | Flexible, high smoothness | Expensive for large grids |
| Local RBF / polyharmonic | Smooth, scalable | More complex implementation |

### Implementation Notes
- Use **non-uniform grids** (denser near ATM and short maturities).
- Use **clamped or monotonic** extrapolation at boundaries.
- Precompute spline coefficients once after FD results are ready.

---

## 3. Theoretical Error Model

### 3.1 Finite-Difference (FD) Solver
For a second-order accurate scheme (e.g. Crank–Nicolson):

\[
E_{\text{FD}} = O(\Delta x^2) + O(\Delta t^2)
\]

Smooth regions achieve this order; near discontinuities (payoff kink, short T), expect first-order behavior locally.

---

### 3.2 Interpolation Error
For a cubic spline interpolant on uniform spacing \( h \):

\[
|P - P_{\text{interp}}| \le C \, h^4 \, \|P^{(4)}\|_\infty
\]

Multi-dimensional tensor spline:
\[
E_{\text{interp}} = O(\max_i h_i^4)
\]

---

### 3.3 Implied Volatility Inversion Error
Price error propagates through Vega:

\[
\Delta \sigma \approx \frac{\Delta P}{\text{Vega}}
\]

Combined asymptotic error:

\[
E_\sigma \approx \frac{C_{\text{FD}}(\Delta x^2 + \Delta t^2) + C_{\text{int}} h^4}{\text{Vega}}
\]

To achieve an implied vol error ≤ εₛ:
\[
E_{\text{FD}} + E_{\text{interp}} \le \varepsilon_\sigma \cdot \text{Vega}_{\min}
\]

---

## 4. Practical Guidelines

### 4.1 Target Error and Grid Sizing
Example:
Desired vol accuracy = 1 bp (1e-4)
Typical Vega ≈ 0.2
→ price tolerance ≈ \(2×10^{-5}\)

Assuming cubic interpolation dominates:
\[
h \lesssim (2×10^{-5})^{1/4} ≈ 0.12
\]

That is, grid spacing ≈ 0.1 in transformed coordinates is usually sufficient.

---

### 4.2 Adaptive Refinement Procedure
1. **Start coarse** (e.g. 41×31 grid).
2. **Compute reference "truth"** prices for a dense validation set.
3. **Interpolate** and compute absolute errors (price & vol).
4. **Refine locally** where error > target (split or add points).
5. Repeat until validation RMS error < tolerance.

This approach yields smaller total grids than uniform high-res meshes.

---

## 5. Numerical Stability Considerations

| Region | Issue | Remedy |
|--------|--------|--------|
| Near expiry (T→0) | Non-smooth payoff | Smooth or refine grid |
| Deep ITM/OTM | Vega small ⇒ amplified vol error | Limit inversion or add σ-axis |
| Extrapolation | Overshoot/undershoot | Use linear or clamped tails |
| Short maturities | Large curvature | Use √T transform |

---

## 6. Alternative Representations & Compression

If grid becomes too large:
- **Chebyshev tensor approximation** (global polynomial, compact coefficients)
- **Tensor decomposition (TT, CP, Tucker)** — effective for smooth low-rank surfaces
- **Local RBF or kNN interpolation** — scalable for sparse regions
- **Neural surrogate model** — learn (x, T) → price or vol mapping
- **Spline knot reduction** — adaptive control-point thinning

---

## 7. Recommended Workflow Summary

| Step | Action | Notes |
|------|---------|-------|
| 1 | Choose transformed coordinates (x, √T) | Stabilizes curvature |
| 2 | Run FD solver on coarse grid | Get prices, vegas |
| 3 | Build cubic spline interpolant | Precompute coefficients |
| 4 | Validate vs analytic (BS) prices | Measure error & refine |
| 5 | Store spline coefficients in memory | Optional: compress or quantize |
| 6 | Runtime: interpolate & invert price→vol | Fast, no PDE solve |

---

## 8. Typical Performance

| Grid | Interpolation | Query latency | Vol error |
|------|----------------|----------------|------------|
| 80×60 | Bicubic | ~1 µs | < 2 bp |
| 160×120 | Bicubic | ~3 µs | < 1 bp |
| 100×80×40 | Tricubic | ~5–10 µs | < 1 bp |

---

## 9. Summary

- **Cubic spline interpolation** offers ~4th-order accuracy and C² smoothness — ideal for implied vol tables.
- **Finite-difference precomputation** is second-order; ensure Δt and Δx small enough that interpolation dominates.
- **Non-uniform grid + coordinate transforms** minimize total grid size.
- **Adaptive refinement** ensures efficient memory use.
- For 1 bp vol accuracy, 2D grids of ~100×80 nodes are typically sufficient.
- If runtime inversion must be avoided, precompute price(x,T,σ) and perform 1D inversion along σ axis.

---

## 10. References

- Tavella & Randall (2000), *Pricing Financial Instruments*
- Hagan et al. (2002), *Managing Smile Risk*
- Press et al., *Numerical Recipes*
- De Boor (1978), *A Practical Guide to Splines*
- Judd (1998), *Numerical Methods in Economics*

---
