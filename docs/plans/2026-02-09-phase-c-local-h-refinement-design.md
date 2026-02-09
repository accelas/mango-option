# Phase C: Local h-Refinement for Exercise Boundary

> Design specification for adaptive element splitting in (x, τ) with overlap
> blending and shape constraints. Targets the interpolation-structure ceiling
> identified by the Phase A noise-floor test.

**Context:** Phase A demonstrated that the Chebyshev 4D accuracy ceiling is 100%
interpolation structure, not PDE solver noise. Doubling PDE accuracy produces
identical IV errors. The bottleneck is the global polynomial's inability to
represent the EEP kink near the moving exercise boundary. Phase C addresses this
with local elements.

**Goal:** Reduce p95 IV error in the 60d-180d σ=15% regime from 979 bps to
<100 bps, and T<60d σ=30% from 168 bps to <80 bps, with no regression in
T>=1y σ=30% (≤2 bps).

---

## 1. Element Layout

Split the (x, τ) domain into 2 τ-bands × 3 x-elements = 6 elements.

### τ-bands

| Band | Range | Rationale |
|------|-------|-----------|
| Short | τ < 60d | Sharp near-ATM exercise boundary |
| Long | τ ≥ 60d | Smoother deeper-ITM boundary |

τ-band interface overlap: [55d, 65d]. Both bands evaluated and blended
smoothly in this zone.

### x-elements per τ-band

```
ITM element          Boundary element         OTM element
[x_min, x*-δ]       [x*-δ, x*+δ]            [x*+δ, x_max]
  coarse (15 nodes)    dense (25 nodes)         coarse (15 nodes)
```

- x\* = estimated exercise boundary center for this τ-band (Section 2).
- δ = half-width of boundary element, set from the 10th-90th percentile
  spread of x\* across (σ, r) values.
- Starting value: δ ≈ 0.15 (covers ~K=85 to K=115 in moneyness).

σ and r axes stay global (shared CC nodes across all elements), so the
PDE cache is fully reusable.

Adjacent x-elements overlap by `w_overlap = 2 * h_boundary`, where
`h_boundary = 2δ / (25 - 1)` is the boundary element's node spacing.

---

## 2. Boundary Detection from Cached PDE Slices

After populating the PDE cache (Phase A incremental CC-level refinement),
detect the exercise boundary x\*(τ, σ, r) for element break placement.

### Algorithm

For each τ-band, sample representative (σ, r, τ) triples from the cache:

1. Evaluate EEP at 200 uniformly-spaced x-points across the domain.
   EEP = spline.eval(x) · K\_ref − European(x, τ, σ, r).
2. Find the rightmost x where EEP > ε (ε = 0.001 · K\_ref). This is x\*
   for that slice.
3. Collect all x\* values across the representative triples.

Boundary placement:
- **Center:** median x\* across all sampled triples.
- **Half-width δ:** max(0.10, (p90 − p10) / 2 + margin), where margin ≈ 0.05.

Short-τ band: x\* closer to 0 (near ATM), smaller δ.
Long-τ band: x\* more negative (deeper ITM), larger δ.

Cost: negligible — spline evaluations and closed-form European, no new PDE
solves.

---

## 3. Overlap Blending

### Weight function

In overlap zone [a, b], define t = (x − a) / (b − a) ∈ [0, 1].

```
ψ(t) = exp(−1 / (1 − (2t−1)²))   for |2t−1| < 1,  else 0
Ψ(t) = ∫₀ᵗ ψ(s) ds / ∫₀¹ ψ(s) ds   (normalized CDF)

w_left(x) = 1 − Ψ(t)
w_right(x) = Ψ(t)
```

This gives C∞ transition. Ψ can be precomputed as a polynomial
approximation or lookup table.

### Query-time evaluation

1. Map x to the correct τ-band (short vs long).
   - If τ ∈ [55d, 65d]: evaluate both bands, blend in τ with the same
     bump weight.
2. Within τ-band, identify x-element(s):
   - Single element: evaluate that element's `ChebyshevTucker4D` directly.
   - Overlap zone: evaluate both adjacent elements, blend with weights.

### Derivatives

Since weights depend only on x (not σ), vega blends linearly:

```
vega_blended = w_left · vega_left + w_right · vega_right
```

Worst case (τ and x both in overlap): 4 element evaluations + 2D
blending. Each Chebyshev eval ≈ 200ns → total ≈ 1μs. Acceptable.

---

## 4. Shape Constraints

Enforced at element level during build, not at eval-time clipping.

### EEP non-negativity (EEP ≥ 0)

After building each element's Chebyshev tensor, evaluate on a dense check
grid (100 x-points × all τ, σ, r nodes). If any values are negative,
project by setting negatives to zero and re-fitting Chebyshev coefficients
via least-squares from the corrected values.

### Put-monotonicity (∂EEP/∂x ≤ 0)

For puts, EEP is non-increasing in x (deeper ITM = more exercise value).
Check monotonicity on the dense grid by finite differences. If violations
exceed tolerance (Δ > 0.001 · max\_EEP), replace violating values with
the monotone envelope (running max from right to left), then re-fit.

Both projections are per-element, run once at build time. The key benefit
is Newton/Brent stability: negative or non-monotone EEP causes IV solver
divergence (the `---` solve failures in heatmaps).

Constraints apply to the boundary and ITM elements. The OTM element has
EEP ≈ 0 everywhere, so constraints are trivially satisfied.

---

## 5. Integration with Incremental CC Cache

Sigma/rate CC-level refinement and PDE cache reuse work unchanged. Local
h-refinement operates only on (x, τ) — the free axes requiring zero new
PDE solves.

### Build pipeline

1. **Populate PDE cache** — same as Phase A. Incremental CC-level
   refinement on (σ, r). Cached slices are CubicSplines over the PDE
   solver's x-grid. No changes.

2. **Detect exercise boundary** — scan cached slices to find x\* per
   τ-band (Section 2). Determines element breaks.

3. **Build element tensors** — for each of the 6 elements, resample
   cached splines at that element's Chebyshev x-nodes via
   `cache.get_slice(σ, r, τ)->eval(x)`. No new PDE solves. Each element
   produces an independent `ChebyshevTucker4D`.

4. **Apply shape constraints** — per-element projection (Section 4).

5. **Package** — store the 6 element tensors, their x/τ bounds, and
   overlap regions in a `PiecewiseElementSet` struct.

### Cost model

- PDE cost: N\_σ × N\_r (unchanged from Phase A).
- European evaluations: 6 elements × ~15-25 x-nodes × 9 τ × 17 σ × 9 r
  ≈ 150k closed-form evals ≈ 15ms. Negligible vs PDE cost.
- Chebyshev fits: 6 independent `ChebyshevTucker4D::build_from_values()`
  calls. Fast.

---

## 6. Data Structures

```cpp
struct ElementConfig {
    double x_lo, x_hi;           // element x-bounds (extended with headroom)
    double tau_lo, tau_hi;        // element τ-bounds
    size_t num_x;                 // Chebyshev nodes in x (15 or 25)
    size_t num_tau;               // Chebyshev nodes in τ
    // σ, r shared across all elements
};

struct PiecewiseElementSet {
    std::vector<ChebyshevTucker4D> elements;  // 6 elements
    std::vector<ElementConfig> configs;        // per-element bounds/sizes

    // Overlap zones (precomputed)
    struct OverlapZone {
        size_t left_idx, right_idx;  // element indices
        double x_lo, x_hi;          // overlap x-range
    };
    std::vector<OverlapZone> x_overlaps;  // 2 per τ-band = 4 total

    double tau_blend_lo, tau_blend_hi;  // τ-band overlap [55d, 65d]
};
```

---

## 7. Success Gates

| Metric | Current | Target |
|--------|---------|--------|
| 60d-180d, σ=15% p95 | 979 bps | <100 bps |
| T<60d, σ=30% p95 | 168 bps | <80 bps |
| T>=1y, σ=30% p95 | 1.8 bps | ≤2 bps (no regression) |
| Solve success rate | 113/144 | ≥130/144 |
| C1 at interfaces | — | central diff matches to 1e-4 rel |
| EEP non-negativity | — | zero violations on dense check grid |
| Put-monotonicity | — | zero violations > 0.1% of max EEP |

---

## 8. Non-Goals

- No changes to sigma/rate CC-level refinement (Phase A, working).
- No boundary-aligned coordinates (can layer on later if A alone
  meets gates).
- No ROI-based adaptive loop (Phase B) — break placement is heuristic
  from cached slices.
- No τ < 7d support (exercise boundary too sharp for polynomial
  representation at any practical degree).
