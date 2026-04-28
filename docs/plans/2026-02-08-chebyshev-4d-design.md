# 4D Chebyshev-Tucker EEP Surface

## Goal

Build a 4D Chebyshev-Tucker interpolant for the EEP surface, matching the
existing 4D B-spline standard surface architecture. Target similar interior
accuracy (1-10 bps) with better deep ITM/OTM coverage and lower PDE cost.

## Motivation

The 3D dimensionless Chebyshev experiment showed that Chebyshev-Tucker gives
3-6x better raw EEP accuracy than B-spline at the same PDE budget. However,
the 3D dimensionless parameterization (x, tau', ln_kappa) embeds sigma in two
coordinates, causing the query point to sweep across a wide domain during IV
solving. This limits end-to-end IV accuracy to ~312 bps RMS.

The 4D standard parameterization (ln(S/K), tau, sigma, rate) avoids this by
keeping sigma as its own axis. Newton iteration only moves along one dimension,
and each axis has a narrow, independent domain where Chebyshev's exponential
convergence should excel.

## Scope

Benchmark experiment only. Production IV factory integration
(`make_interpolated_iv_solver`) is a follow-up if the experiment succeeds.

## Architecture

### Axes

Same as 4D B-spline:

| Axis | Variable | Typical range |
|------|----------|---------------|
| 0 | ln(S/K) | [-0.50, 0.40] |
| 1 | tau | [0.019, 2.0] |
| 2 | sigma | [0.05, 0.50] |
| 3 | rate | [0.01, 0.10] |

### EEP Decomposition

```
American(S, K, tau, sigma, r) = (K/K_ref) * EEP(ln(S/K), tau, sigma, r)
                                + European(S, K, tau, sigma, r, q)
```

where q is the continuous dividend yield. The PDE solve uses q, and the
European component at query time requires q for accurate pricing. The
builder and inner class both accept and propagate dividend_yield.

EEP stored in Tucker-compressed Chebyshev tensor. European computed exactly
at query time via Black-Scholes with dividend yield.

### Tucker Compression

4D HOSVD: unfold tensor along each of 4 modes, SVD + truncate each, contract
into 4D core.

Storage: core (R0 x R1 x R2 x R3) + four factor matrices U_d(n_d, R_d).

Epsilon default: 1e-8. Benchmarks will sweep epsilon values (1e-6, 1e-8,
1e-10) to map the accuracy/compression tradeoff against IV error targets.

### Query (eval)

1. Clamp query to domain bounds (lesson from 3D experiment)
2. Barycentric Chebyshev interpolation of each factor column per axis
3. Contract with 4D core tensor
4. Cost: O(sum(n_d * R_d) + R0*R1*R2*R3)

### EEP Floor

Call production `EEPDecomposer::decompose()` from `eep_transform.cpp`
directly, rather than reimplementing. This includes the debiased softplus
with max(0, ...) post-clamp and overflow branch for large positive inputs.
Ensures benchmark results are directly comparable to the B-spline path.

### IV Solving

Newton-Raphson. Vega via `partial(2, coords)` -- finite difference along
sigma axis. Since sigma is its own axis, Newton only moves along one
dimension. No coordinate coupling.

**Boundary vega limitation:** Finite-difference partial with clamped eval
degenerates near domain edges (one-sided difference, biased toward zero).
For 4D, the sigma axis [0.05, 0.50] + headroom should keep typical Newton
iterates away from boundaries. If boundary vega proves problematic, the
fix is analytical Chebyshev differentiation (differentiate the barycentric
formula directly) -- left as future work.

## Grid Configurations

| Config | Grid | PDE solves | Tensor size |
|--------|------|-----------|-------------|
| Small | 8x8x10x5 | 50 | 4,000 |
| Standard | 10x10x15x6 | 90 | 9,000 |
| Large | 10x10x25x6 | 150 | 15,000 |

### Headroom

Per-axis headroom: `3 * domain_width / (n-1)` per side, clamped to valid
physical bounds:

| Axis | Floor | Rationale |
|------|-------|-----------|
| ln(S/K) | none | log-moneyness can be any real |
| tau | 1e-4 | maturity must be positive |
| sigma | 0.01 | volatility must be positive |
| rate | -0.05 | rates can be slightly negative (SOFR) |

Example: standard grid sigma axis [0.05, 0.50] with n=15:
headroom = 3 * 0.45 / 14 = 0.096. Extended: [max(0.05-0.096, 0.01),
0.50+0.096] = [0.01, 0.596]. Valid.

Example: standard grid tau axis [0.019, 2.0] with n=10:
headroom = 3 * 1.981 / 9 = 0.660. Extended: [max(0.019-0.660, 1e-4),
2.0+0.660] = [1e-4, 2.660]. Valid.

## Components

### New library files

1. `src/option/table/dimensionless/tucker_decomposition_4d.hpp`
   - `TuckerResult4D`: core, four factor matrices, four ranks
   - `tucker_hosvd_4d(tensor, {n0,n1,n2,n3}, epsilon)`

2. `src/option/table/dimensionless/chebyshev_tucker_4d.hpp`
   - `ChebyshevTucker4D` class
   - `build_from_values()`, `eval()`, `partial()`, `compressed_size()`, `ranks()`
   - Domain clamping on eval with per-axis floors

### New benchmark files

3. `benchmarks/chebyshev_4d_eep_inner.hpp`
   - `Chebyshev4DEEPInner` class: `price(PriceQuery)`, `vega(PriceQuery)`
   - `build_chebyshev_4d_eep(config)` builder
   - Accepts and propagates dividend_yield
   - Uses debiased softplus EEP floor

### Modified files

4. `benchmarks/interp_iv_safety.cc` -- add Chebyshev 4D section
5. `benchmarks/iv_interpolation_sweep.cc` -- add BM_Chebyshev4D_IV
6. `benchmarks/BUILD.bazel` -- add deps
7. `tests/chebyshev_tucker_4d_test.cc` -- interpolant tests

## Build Process

1. Generate Chebyshev nodes per axis (with headroom, clamped to valid bounds)
2. One PDE per (sigma, rate) pair: N_sigma * N_rate solves, with dividend_yield
3. Snapshot at each tau node
4. Resample spatial solution at each x=ln(S/K) node via cubic spline
5. EEP via production `EEPDecomposer::decompose()` (softplus + clamp + overflow)
6. Tucker 4D HOSVD compression
7. Package into ChebyshevTucker4D

## Success Criteria

1. Interior accuracy: match 4D B-spline ~1-10 bps at K=90-110, T>=60d
2. Deep ITM/OTM: more probes solved than B-spline's 68/72, AND newly
   solved probes must be <50 bps vs FDM reference IV (prevents clamp
   artifacts from inflating the solve count)
3. Build cost: 50-150 PDE solves (vs B-spline adaptive ~200-400)
4. Query speed: comparable ~3-5us per IV solve
