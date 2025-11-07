# BSpline Interpolation Accelerator for IV Solver

**Date:** 2025-11-06  
**Status:** Design Phase  
**Goal:** Deliver <30 µs implied-volatility queries by replacing direct FDM pricing with tensor-product cubic B-spline interpolation built on precomputed PDE surfaces.

---

## Executive Summary

Finite-difference pricing with TR-BDF2 currently takes ~143 ms per IV query. The legacy cubic spline interpolation is 1D-only, can return negative prices, and does not supply smooth Greeks, so the modern IV solver still falls back to slow FDM paths.

This design introduces a separable tensor-product cubic B-spline pipeline:

- Run the existing PDE solver over a modest lattice of vol and rate values (20 × 10) to populate a 4D price grid.
- Fit cubic B-spline control points by solving width-4 banded systems per dimension, avoiding the infeasible dense 300k × 300k solve.
- Serve prices and Greeks via a clamped, derivative-consistent interpolation layer that keeps Newton’s method stable.
- Integrate a fast IV solver that uses the B-spline surface for both price and vega, targeting <30 µs per query (4,800× faster than FDM).

The plan stays within existing architectural boundaries (PDESolver, SnapshotCollector, IVSolver) and reuses our Thomas solver infrastructure by extending it to handle the cubic stencil band structure.

---

## Problem Statement

### Current Behaviour

- `PriceTableSnapshotCollector` assembles only 2D surfaces `V(m, τ)` for a single (σ, r).
- `SnapshotInterpolator` relies on natural cubic splines; overshoot can drive prices negative near boundaries, and derived Greeks are inconsistent with any post-hoc clamping.
- `IVSolver` must revert to on-demand TR-BDF2 solves when interpolated vegas are non-monotone or inconsistent.

### Root Causes

1. **Dimensionality gap:** No mechanism to aggregate PDE snapshots across volatility and rate axes, so higher-dimensional interpolation is unavailable.
2. **Control-point fitting infeasibility:** Treating the tensor-product basis as a dense global least-squares problem requires terabytes of memory.
3. **Derivative mismatch:** Clamping the price surface but differentiating the unclamped spline breaks Newton convergence.

---

## Design Overview

The proposed system introduces three cooperating layers:

```
 Overnight Batch
   └── PriceTable4DBuilder::add_slice(σ, r, snapshots)  ──► raw 4D grid
        └─ SeparableBSplineFitter::fit(...)            ──► control points
             └─ BSplineSurface4D::query_*()            ──► price / greeks
                  └─ IVSolverInterpolated::solve()     ──► implied volatility
```

Key properties:

- **Separable fitting:** Perform four passes of banded solves (one per axis) instead of a dense 4D solve.
- **Banded linear algebra:** Each 1D collocation system has bandwidth 4 (degree + 1). We extend the existing `ThomasWorkspace` to a generic narrow-band LU routine to retain O(n) behaviour.
- **Derivative-consistent clamping:** Default to a soft-plus style clamp that keeps the surface C¹ and avoids derivative discontinuities; optionally fall back to finite differences if stricter clamping is required.
- **Data reuse:** Snapshots are stored per (σ, r) slice, enabling rebuilds without rerunning PDEs.

---

## Detailed Design

### 1. Four-Dimensional Data Collection

- **Builder API:** Introduce `PriceTable4DBuilder` with grids for moneyness (nₘ ≈ 50), maturity (n_τ ≈ 30), volatility (n_σ ≈ 20), and rate (nᵣ ≈ 10). Optional dividend yield `q` is deferred but the builder reserves a hook for a fifth axis.
- **Public interface example:**
  ```cpp
  class PriceTable4DBuilder {
  public:
      explicit PriceTable4DBuilder(MultiGridBuffer grid, double K_ref);

      void add_slice(double sigma, double rate,
                     std::span<const Snapshot> snapshots);

      PriceTable4DSurface build_surface() const;

  private:
      MultiGridBuffer grid_;
      std::vector<double> prices_;  // flattened m × τ × σ × r
      double K_ref_;
  };
  ```
- **PDE batching:** For each `(σ_j, r_k)` pair, reuse `AmericanOptionSolver` to produce snapshots. `Snapshot::user_index` already encodes τ, and we map moneyness by evaluating `SnapshotInterpolator` at `S = m × K_ref`.
- **Storage:** Flattened array of size `nₘ · n_τ · n_σ · nᵣ`. Separate arrays for price, vega, and gamma can be captured if needed for validation; the fitter consumes only the price array.
- **Persistence:** Optionally write raw slices to disk (`.npz` or binary) for reproducibility.

### 2. 1D Cubic B-Spline Infrastructure

- **Basis evaluation:** Implement `BSplineBasis1D` (open uniform knot vector, degree 3) with `eval_basis(i, x)` and derivatives using Cox–de Boor recursion.
- **Banded collocation matrix:** For each grid point `x_p`, evaluate the four non-zero basis functions (`B_{i-3}`, …, `B_i`). Populate diagonals `a_{-3} … a_{0}` in compressed form.
- **Linear solver:** Extend `ThomasSolver` into `BandedLU` supporting half-bandwidth 3 (upper) / 3 (lower). Factorisation is O(n) and reused for all slices along a given axis.
  ```cpp
  class BandedLU {
  public:
      BandedLU(size_t n, size_t lower_bw, size_t upper_bw);

      void factorize(std::span<const double> lower_bands,
                     std::span<const double> diag,
                     std::span<const double> upper_bands);

      void solve(std::span<const double> rhs,
                 std::span<double> solution) const;
  };
  ```
- **Validation:** Unit tests on polynomials, exponentials, and consistency with SciPy/Boost references.

### 3. Separable Tensor-Product Fitter

- **Pass order:** `m (50 pts) → τ (30 pts) → σ (20 pts) → r (10 pts)` — widest dimension first keeps the largest stride contiguous and minimises temporary storage pressure. We can optionally benchmark all 24 permutations once at build-time and cache the fastest ordering, but the width-descending order is the default.
- Extracts 1D slices into contiguous buffers (can be views with stride-aware iterators to avoid copies).
- Solves the pre-factored banded system to update control points along that axis.
- Writes results back in-place.
- **Caching:** Precompute and cache LU factors for each axis once per builder invocation.
- **Convergence:** For separable inputs, the method reproduces the function exactly; for general inputs, the sequence converges in one sweep because we are effectively projecting onto tensor-product basis (de Boor Ch. 11).
- **Complexity:** O(n_total) operations, <1 s for the target grid on a modern CPU; memory footprint ~12 MB including temporaries.
  ```cpp
  class SeparableBSplineFitter4D {
  public:
      SeparableBSplineFitter4D(const AxisWorkspace& m_axis,
                               const AxisWorkspace& tau_axis,
                               const AxisWorkspace& sigma_axis,
                               const AxisWorkspace& rate_axis);

      void fit(std::span<const double> data,      // input grid values
               std::span<double> control_points); // output same shape
  };
  ```

### 4. Query Layer with Smooth Clamping

- **Surface representation:** `BSplineSurface4D` owns knot vectors and control points. Query evaluation computes at most 4⁴ = 256 basis products (SIMD-enabled).
- **Soft clamp:** Use a soft-plus transition near zero to preserve C¹ continuity:
  ```
  soft_clamp(V) = epsilon * log1p(exp((V - V_shift)/epsilon)) + offset
  ```
  where `epsilon ≈ 1e-4` and `V_shift` ensures `soft_clamp(0) = 0`. Analytical derivatives follow directly:
  ```
  d_soft_clamp/dV_raw = 1.0 / (1.0 + exp(-(V_raw - V_shift)/epsilon))
  d_soft_clamp/dσ = d_soft_clamp/dV_raw * (∂V_raw/∂σ)
  ```
  We apply the same sigmoid factor to derivatives with respect to other axes.
- **Fallback mode:** For validation or extremely tight no-overshoot requirements, enable a hard clamp with centred finite differences for vega/gamma (same interface, flagged in config).
- **API:**
  ```cpp
  struct PriceQueryResult {
      double price;
      double vega;
      double gamma;
  };

  PriceQueryResult query(double m, double tau, double sigma, double r,
                         QueryMode mode = QueryMode::kSmoothClamp) const;
  ```
- **Caching:** Cache basis values per dimension for the last few queries (LRU of size 4) to shave cycles off hot IV loops.
- **Epsilon guidance:** Default epsilon uses `epsilon = max(1e-5, 0.001 * V_typical)` where `V_typical` is the median price for the slice (≈$10 ATM → ε=0.01). Deep OTM regions (≈$0.10) naturally settle around ε≈1e-4. Expose overrides for infrastructure that wants tighter control.

### 5. IV Solver Integration

- **Solver:** `IVSolverInterpolated` runs damped Newton (`σ_{n+1} = σ_n - (V - V_market)/vega`, β=0.8). Uses query layer for price and vega, optionally gamma for diagnostics.
  ```cpp
  class IVSolverInterpolated {
  public:
      explicit IVSolverInterpolated(const PriceTable4DSurface& surface,
                                    IVSolverConfig config);

      IVResult solve(const MarketInputs& inputs) const;

  private:
      PriceTable4DSurface surface_;
      IVSolverConfig config_;
  };
  ```
- **Bounds & fallback:** Reject inputs outside precomputed lattice; caller switches to full FDM or extrapolation policy.
- **Glue:** `IVSolver` selects the interpolated path when a matching `PriceTableSurface` is loaded; otherwise the legacy path stays active.

### 6. Validation & Benchmarking

- **Unit tests:** Banded solver accuracy, tensor fitter on separable/analytic functions, smooth clamp derivative checks.
- **Integration tests:** Compare interpolated prices/Greeks vs direct PDE on dense checkpoints (ATM/OTM, short/long τ, low/high σ).
- **IV regression:** 1 000 random market scenarios; assert RMS IV error <0.5 bp and max error <1.5 bp.
- **Performance benchmarks:** Microbenchmarks for price/vega query (<2 µs / <5 µs targets) and full IV solves (<30 µs median).

---

## Implementation Plan

| Phase | Scope | Key Deliverables | Exit Criteria |
|-------|-------|------------------|---------------|
| **1. 1D B-spline foundation** (Week 1) | `BSplineBasis1D`, `BandedLU`, tests | New solver handles width-4 band systems; polynomial/exp fits pass | 50-point fit in <1 ms |
| **2. Separable fitter** (Week 2) | `SeparableBSplineFitter4D`, synthetic validation | One-pass fit reproduces separable functions; performance benchmark | 300k-grid fit <1 s |
| **3. 4D data builder** (Week 3) | `PriceTable4DBuilder`, PDE batching scripts | 200-run pipeline produces 4D table; raw slice persistence | Build time <10 s |
| **4. Query & clamp layer** (Week 4) | `BSplineSurface4D::query`, smooth clamp | Analytical derivatives validated; optional finite-diff path | Vega query <5 µs |
| **5. IV integration & validation** (Week 5) | `IVSolverInterpolated`, benchmarks, docs | IV RMS error <0.5 bp; 95% queries <30 µs | Benchmark + validation report |

---

## Success Metrics

- **Latency:** price <2 µs, vega <5 µs, IV solve <30 µs (95th percentile).  
- **Accuracy:** RMS IV error <0.5 bp; price RMS error <0.1%; gamma relative error <5%.  
- **Robustness:** No negative prices returned; Newton convergence rate >95% for in-bounds queries.  
- **Maintainability:** Code confined to new modules with unit/integration coverage; rebuildable surfaces from stored slices.

---

## Risks & Mitigations

- **Band solver stability:** Partial pivoting may be required for ill-conditioned grids. Mitigation: implement optional pivoting path, monitor condition numbers during fitting.
- **Clamp tuning:** Poor epsilon choice could distort near-zero prices. Mitigation: expose epsilon in config, validate across deep OTM/short τ cases.
- **PDE batch cost growth:** Larger lattices (adding dividend yield) multiply runtime. Mitigation: parallelise outer loops with existing threading infrastructure.
- **Memory footprint:** Control points plus caches may exceed expectations on very fine grids. Mitigation: document limits, allow streaming fit (axis-by-axis chunking) if needed.

---

## Open Questions

1. Should we persist the fitted control points in a portable format (e.g., Cap’n Proto) for deployment, or regenerate at startup?  
2. Do we need a first-class extrapolation policy for out-of-bounds queries (e.g., linear in σ), or is “fallback to FDM” acceptable?  
3. How aggressive should the basis-value caching be to balance memory and latency?  
4. When we extend to 5D (adding dividend yield), do we prioritise compressing the grid or further optimising the banded solver?

---

## Next Steps

1. Review and approve this design.  
2. Spin up a feature branch for Phase 1 foundations.  
3. Schedule PDE batch resource usage (overnight jobs) once the data builder lands.
