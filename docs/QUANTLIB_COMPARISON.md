# QuantLib Comparison: Mesh Convergence

Tracking mango-option FDM accuracy and speed vs QuantLib FdBlackScholesVanillaEngine.

**Test case:** ATM 1Y American put, S=K=100, sigma=0.20, r=0.05, q=0.02.
Grid scales are multiples of the auto-estimated base (101 spatial points).
Reference price from QuantLib at 2001x20000.

## Current Results

Jacobian stencil fix (#337) + equidistribution-optimal alpha (#336).

### Vanilla

| Scale | Nx | Nt | Mango (default) | Mango (opt-alpha) | QuantLib | Mango opt-alpha vs QL |
|-------|-----|------|-----------------|-------------------|----------|----------------------|
| 1x | 101 | 120 | 9.94e-3 / 0.34ms | 6.21e-3 / 0.34ms | 4.97e-3 / 0.42ms | 1.25x worse err, 20% faster |
| 2x | 203 | 240 | 2.99e-3 / 1.32ms | 2.05e-3 / 1.32ms | 2.05e-3 / 1.46ms | tied err, 10% faster |
| 4x | 405 | 480 | 996e-6 / 5.25ms | 762e-6 / 5.25ms | 909e-6 / 5.21ms | **1.19x better err**, tied speed |
| 8x | 809 | 960 | 362e-6 / 20.8ms | 304e-6 / 20.9ms | 417e-6 / 19.6ms | **1.37x better err**, 7% slower |

### Discrete Dividends (quarterly $0.50)

| Scale | Nx | Nt | Mango | QuantLib | Notes |
|-------|-----|------|-------|----------|-------|
| 1x | 101 | 96 | 326e-6 / 0.28ms | 4.72e-3 / 0.37ms | Mango much better at 1x |
| 2x | 203 | 192 | 2.64e-3 / 1.08ms | 264e-6 / 1.27ms | QuantLib better |
| 4x | 405 | 384 | 1.29e-3 / 4.24ms | 491e-6 / 4.41ms | QuantLib better |
| 8x | 809 | 768 | 1.05e-3 / 16.8ms | 429e-6 / 16.2ms | QuantLib better, mango plateaus |

Mango dividend convergence plateaus at ~1e-3 due to natural cubic spline BCs (#327).

## Convergence Order (vanilla, opt-alpha)

Empirical order from log-log slope (error vs grid size):

| Method | Order |
|--------|-------|
| Mango (default alpha) | ~1.5 |
| Mango (opt-alpha) | ~1.4 |
| QuantLib | ~1.1 |

## Changelog

### 2025-02-04: Jacobian stencil fix + CFL fix (#337)

**Changes:**
- Fixed first-derivative stencil mismatch in `assemble_jacobian()` on non-uniform grids
- Replaced approximate CFL formula `dx_min = dx_avg * exp(-alpha)` with actual grid minimum spacing
- Added regression tests for Jacobian/operator consistency

**Impact (vanilla, opt-alpha):**

| Scale | Before (err) | After (err) | Improvement |
|-------|-------------|-------------|-------------|
| 1x | 4.55e-3 | 6.21e-3 | worse (fewer time steps) |
| 2x | 1.23e-3 | 2.05e-3 | worse (fewer time steps) |
| 4x | 352e-6 | 762e-6 | worse (fewer time steps) |
| 8x | 100e-6 | 304e-6 | worse (fewer time steps) |

Note: accuracy appears worse because the CFL fix drastically reduced time steps (498 -> 120 at 1x).
The old CFL formula was over-allocating ~4x more time steps than needed, masking spatial error.
The Jacobian fix improves spatial accuracy at the same grid, but the time step reduction dominates.

### 2025-02-04: Equidistribution-optimal sinh alpha (#336)

**Changes:**
- Default alpha changed from 2.0 to `2*arcsinh(n_sigma/sqrt(2))` (~3.95 for n_sigma=5)
- 1.6-1.8x error reduction at zero speed cost

### 2025-02-04: Initial benchmark (#328)

Baseline comparison at default alpha=2.0.

## Open Issues

| Issue | Description | Expected Impact |
|-------|-------------|-----------------|
| #329 | ~~Jacobian stencil mismatch~~ | Fixed in #337 |
| #330 | Strike alignment on grid | High - kink-on-node matters at coarse grids |
| #331 | Monotone cubic interpolation for value_at() | High - linear interp dominates coarse-grid error |
| #332 | Upwind/hybrid advection scheme | Medium - monotonicity on coarse grids |
| #333 | Second cluster at exercise boundary | Medium - better resolution of free boundary |
| #334 | Robin BCs | Low - negligible at n_sigma=5 in log-moneyness |
| #335 | Graded time mesh near expiry | Low - Rannacher already handles this |
| #327 | Dividend convergence plateau | High - blocks dividend accuracy past ~1e-3 |

## How to Run

```bash
bazel run //benchmarks:quantlib_mesh_comparison -c opt
```

Requires `libquantlib-dev`.
