# Chebyshev-Tucker 3D Interpolation Experiment

## Goal

Benchmark Chebyshev polynomial interpolation with Tucker tensor decomposition
against B-spline interpolation for the dimensionless EEP function
`eep(x, τ', ln κ)`. Test whether Chebyshev breaks the B-spline accuracy wall
found during PR 386.

## Motivation

PR 386 found that B-spline interpolation on the 3D dimensionless surface hits
an accuracy plateau — adding more grid points no longer improves error. The
wide ln κ axis (~8 units) causes oscillation, requiring domain segmentation as
a workaround. Chebyshev interpolation's exponential convergence for smooth
functions should overcome this limitation.

## Design Decisions

- **Test function:** Dimensionless EEP directly (not analytic test functions).
  Ground truth via `reference_eep()` PDE solves at Ultra accuracy.
- **Tensor structure:** Low-rank Tucker decomposition from the start, testing
  the Glau et al. claim that parameter cross-interactions are weak.
- **SVD dependency:** Eigen (header-only) via Bazel. Experiment-only — not
  committed to production.
- **Scope:** Standalone proof-of-concept + head-to-head benchmark. No
  production integration, no derivatives, no adaptive refinement, no q > 0.

## Architecture

### Chebyshev Interpolation

Chebyshev-Gauss-Lobatto nodes on [-1, 1]:

    xⱼ = cos(jπ / N),  j = 0, ..., N

Mapped to each physical axis via affine transform. Barycentric interpolation
for O(N) evaluation without forming the polynomial explicitly:

    p(x) = Σⱼ (wⱼ/(x - xⱼ)) · fⱼ  /  Σⱼ (wⱼ/(x - xⱼ))

### Tucker Decomposition (HOSVD)

Full 3D tensor T of shape (N₀, N₁, N₂) decomposed as:

    T ≈ G ×₀ U₀ ×₁ U₁ ×₂ U₂

where Uₖ is Nₖ × Rₖ (orthogonal factor matrix) and G is R₀ × R₁ × R₂ (core
tensor). Ranks Rₖ chosen by singular value threshold (σᵢ/σ₀ < 1e-8).

Algorithm:
1. Mode-k unfolding of T into matrix (Nₖ, product of other dims)
2. SVD via Eigen's JacobiSVD, truncate to Rₖ
3. Core tensor: G = T ×₀ U₀ᵀ ×₁ U₁ᵀ ×₂ U₂ᵀ

### Query-Time Evaluation

1. Barycentric Chebyshev weights for each axis: O(N) per axis
2. Contract weights with factor matrices Uₖ: O(N × R) per axis
3. Contract with core tensor G: O(R³)
4. Total: O(3NR + R³) vs O(N³) full tensor

### File Layout

All under `src/option/table/dimensionless/` in the chebyshev-tensor worktree:

- `chebyshev_nodes.hpp` — Node generation + barycentric weights
- `tucker_decomposition.hpp` — HOSVD via Eigen SVD
- `chebyshev_tucker.hpp` — Combined interpolant: build + eval
- `tests/chebyshev_tucker_test.cc` — Accuracy tests against reference EEP
- `benchmarks/chebyshev_vs_bspline.cc` — Head-to-head comparison

## Benchmark Design

**Independent variable:** Nodes per axis N, swept from 6 to 30. Both methods
use the same N, so the same number of PDE solves.

**Measurements:**
- Max and mean absolute EEP error over 125 off-grid probes
- Tucker ranks achieved per mode
- Evaluation time per query (ns)

**Expected outcome:**
- B-spline error plateaus at some N (the "wall")
- Chebyshev-Tucker error continues dropping exponentially
- Tucker ranks stay low (R ≈ 5-10), confirming weak cross-interactions

## Out of Scope

- Production integration (replacing BSplineND<3>)
- Derivatives / vega via Chebyshev
- Adaptive node placement
- q > 0 handling
- Eigen in production builds
