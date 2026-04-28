# Numerical EEP for Chained Segments

## Problem

The segmented price table builder stores chained segments (all except the last) as `RawPrice` — the full normalized American price `V/K_ref`. This value has a kink at the exercise boundary, which degrades B-spline fitting quality. The last segment avoids this by storing EEP (Early Exercise Premium = American − BSM European), which is smooth.

Chained segments cannot use analytical EEP because their initial condition comes from the previous segment's surface at the dividend boundary, not the standard payoff. No closed-form European exists for this IC.

Issue #363 proposed solving a second PDE without the exercise constraint to obtain a numerical European baseline. Issue #360 confirmed the exercise boundary kink is the dominant interpolation error source. The approach was closed as too expensive, but the cost was overestimated: Black-Scholes is linear, so Newton converges in one iteration, making the European solve the same cost as the American.

## Design

### Two surfaces per chained segment

For each chained segment, run two PDE solves per (σ, r) pair:

1. **American** — with obstacle (existing behavior)
2. **European** — same grid, same chained IC, obstacle projection disabled

Extract two tensors:

- **EEP**: `(V_american − V_european) / K_ref`, with softplus floor for non-negativity
- **European**: `V_european / K_ref`

Fit each tensor to a separate 4D B-spline surface. Both are smooth — neither has the exercise boundary kink.

At query time, `AmericanPriceSurface::price()` reconstructs. For `NumericalEEP` segments (strike pinned to `K_ref` by `SegmentedTransform`):

```
V = EEP_spline(m, τ, σ, r) + Eu_spline(m, τ, σ, r)
```

where `m = S/K_ref`. The `SegmentedTransform::normalize_value()` multiplies by `K_ref` to convert to dollar price, and the outer `StrikeSurfaceWrapper` handles scaling to arbitrary strikes.

IC chaining calls `prev->price()` as before. The two-surface reconstruction is transparent to callers.

### Runtime flag to disable obstacle projection

Add a `bool projection_enabled_ = true` flag to `AmericanOptionSolver`, exposed via:

```cpp
void set_projection_enabled(bool enabled);
```

Pass this flag through to the CRTP solver. The flag must disable projection at **all three sites** in the PDE solver:

1. **`initialize()`** (`pde_solver.hpp:103`) — calls `apply_obstacle()` after setting IC. Skip when projection disabled.
2. **`process_temporal_events()`** (`pde_solver.hpp:270`) — re-applies obstacle after dividend jumps. Skip when projection disabled.
3. **`solve_implicit_stage_projected()`** — the Brennan-Schwartz backward substitution. When projection disabled, skip `max(u, psi)` and deep-ITM locking (lines 632-651). Run plain Thomas instead.

The flag is a runtime `bool` checked at each site. The compile-time `HasObstacle` concept still routes to `solve_implicit_stage_projected()`, but the projected path degrades to plain Thomas when the flag is false.

### Boundary conditions for unconstrained solve

The current put solver uses American-style boundary conditions:

- **Left BC** (`american_option.cpp:163`): `g(t, x) = max(1 - exp(x), 0)` — intrinsic value
- **Right BC**: `g(t, x) = 0` — OTM, fine for both American and European

The left BC is the put payoff, which is correct for both American and European puts at the far-left boundary (deep ITM where `S → 0`, the option value converges to intrinsic regardless of exercise policy). No BC changes are needed for the unconstrained solve.

However, the validation test ("European PDE vs BSM") should not expect machine-precision agreement. The PDE uses a finite grid with these BCs, while BSM is exact. The test should verify **grid-convergence**: the PDE European matches BSM to within the PDE's truncation error (order of `dx²`), not to machine epsilon.

### New SurfaceContent type

Add a third `SurfaceContent` variant:

```cpp
enum class SurfaceContent : uint8_t {
    RawPrice = 0,
    EarlyExercisePremium = 1,
    NumericalEEP = 2,  ///< EEP computed by PDE subtraction, requires companion European surface
};
```

This is necessary because `SegmentedTransform` in `spliced_surface.hpp` dispatches on `content` to decide spot adjustment (EEP segments) vs strike pinning (RawPrice segments):

- `EarlyExercisePremium` → applies spot adjustment for future dividends, uses `m = S/K`
- `RawPrice` → pins strike to `K_ref`, uses `m = S/K_ref`

`NumericalEEP` needs its **own case** in `SegmentedTransform::to_local()`:

- **No spot adjustment.** Unlike analytical EEP (last segment), chained segments already incorporate boundary dividends via IC chaining. The spot adjustment in `compute_spot_adjustment()` would subtract the boundary dividend from spot, double-counting what the IC jump condition already handled. Chained segments span exactly the interval between two consecutive dividends, so no dividends fall within the segment — no adjustment is needed.
- **Pin strike to `K_ref`.** The chained segment PDE is solved at `Spot = Strike = K_ref`. The IC applies a cash dividend shift `V(S - D)` with absolute dollar amount `D`, which breaks strike homogeneity — the resulting surface is only valid at `K = K_ref`. This is the same constraint as `RawPrice`. The outer multi-`K_ref` composition (`StrikeSurfaceWrapper`) handles arbitrary strikes by interpolating across `K_ref` slices.
- **Convert τ to local time** (same as all paths).

In `normalize_value()`, `NumericalEEP` returns `raw * K_ref` (same as `RawPrice`), since `AmericanPriceSurface::price()` for `NumericalEEP` with pinned strike returns a normalized value.

Summary of `to_local()` behavior per content type:

| | Spot adjustment | Strike | `normalize_value()` |
|---|---|---|---|
| `EarlyExercisePremium` | Yes (future dividends) | Pass-through (`m = S/K`) | `raw` |
| `RawPrice` | No | Pin to `K_ref` | `raw * K_ref` |
| `NumericalEEP` | No | Pin to `K_ref` | `raw * K_ref` |

The separate enum value also lets `AmericanPriceSurface` distinguish analytical EEP (reconstruct with BSM) from numerical EEP (reconstruct with companion European surface).

### AmericanPriceSurface changes

Extend `AmericanPriceSurface` to hold an optional second `PriceTableSurface` (the European surface). The `create()` factory accepts an optional European surface pointer. When both surfaces are provided, `create()` validates that their axes and K_ref match.

When `content == NumericalEEP` and the European surface is present, the surface behaves like `RawPrice` at the `AmericanPriceSurface` level — strike is pinned to `K_ref`, and all queries use `m = S/K_ref`. The difference is internal: instead of evaluating one surface, it sums two.

```cpp
// NumericalEEP price (strike == K_ref, enforced by SegmentedTransform)
double m = spot / K_ref_;
return eep_surface_->value({m, tau, sigma, rate})
     + eu_surface_->value({m, tau, sigma, rate});
```

Greeks follow the same pattern — sum partials from both surfaces:

- **`delta()`**: `(1/K_ref) × (∂EEP/∂m + ∂Eu/∂m)`
- **`gamma()`**: `(1/(K_ref²)) × (∂²EEP/∂m² + ∂²Eu/∂m²)`
- **`vega()`**: `∂EEP/∂σ + ∂Eu/∂σ`
- **`theta()`**: `-(∂EEP/∂τ + ∂Eu/∂τ)`

No BSM evaluation needed at query time. The `normalize_value()` in `SegmentedTransform` handles the `× K_ref` denormalization, and `StrikeSurfaceWrapper` handles scaling to arbitrary strikes.

### Build pipeline changes

In `SegmentedPriceTableBuilder::build()`, for chained segments only, replace the current single-solve path (steps 5-10) with:

**Step 5a: American batch solve** (unchanged)
```
batch_result_am = batch_solver.solve_batch(batch_params, true, setup_callback, custom_grid);
```

**Step 5b: European batch solve** (new)
```
batch_result_eu = batch_solver.solve_batch(batch_params, true, eu_setup_callback, custom_grid);
```
Where `eu_setup_callback` wraps the existing `setup_callback` and additionally calls `solver.set_projection_enabled(false)`.

**Step 6: Failure mask reconciliation** (new)

Union the failed indices from both batch results. If either solve failed for a given (σ, r) pair, treat both as failed for that pair:
```
failed_union = batch_result_am.failed ∪ batch_result_eu.failed
```

**Step 7a: Extract American tensor** (existing `extract_tensor` with `SurfaceContent::RawPrice`)

Produces `tensor_am[m, τ, σ, r] = V_american / K_ref` (normalized price). Mark union-failed slices as NaN.

**Step 7b: Extract European tensor** (same call on European results)

Produces `tensor_eu[m, τ, σ, r] = V_european / K_ref`. Mark union-failed slices as NaN.

**Step 8: Compute EEP tensor** (new, pointwise)
```
eep_dollar = K_ref * (tensor_am - tensor_eu)       // convert to dollar EEP
eep_floored = softplus_floor(eep_dollar)            // existing debiased softplus (kSharpness=100, calibrated for dollars)
tensor_eep = eep_floored / K_ref                    // back to normalized
```

The softplus floor must operate in dollar space, not normalized space. The existing `kSharpness = 100.0` is calibrated for dollar-valued EEP (~$0 to ~$20). Applying it to normalized values (`/K_ref`, ~100x smaller) would change the smoothing scale and bias reconstructed prices.

**Step 9: Repair both tensors**

Run `repair_failed_slices()` on `tensor_eep` and `tensor_eu` independently.

**Step 10: Fit two B-spline surfaces**

`fit_coeffs(tensor_eep, axes)` → EEP coefficients
`fit_coeffs(tensor_eu, axes)` → European coefficients

**Step 11: Build AmericanPriceSurface**

Build with both surfaces, content set to `NumericalEEP`.

Note: steps 7a and 7b both need `SurfaceContent::RawPrice` extraction (normalized price, no BSM subtraction). Since `extract_tensor` reads `surface_content_` from builder state (set via `set_surface_content()`), the builder must call `set_surface_content(SurfaceContent::RawPrice)` before both extractions. The final `NumericalEEP` content is set only on the metadata when building the surface in step 11. The EEP subtraction happens in step 8, not inside `extract_tensor`. This avoids modifying `extract_tensor`'s interface.

### IC chaining normalization fix

`ChainedICContext` at `segmented_price_table_builder.cpp:248` uses `prev_is_eep` to decide whether to divide `price()` output by `K_ref`:

```cpp
.prev_is_eep = (prev->metadata().content == SurfaceContent::EarlyExercisePremium),
```

With `NumericalEEP`, `price()` returns a **normalized** value (strike pinned to `K_ref`, before `normalize_value()` denormalization). This matches `RawPrice` behavior — no division by `K_ref` needed. The original check is correct:

```cpp
.prev_is_eep = (prev->metadata().content == SurfaceContent::EarlyExercisePremium),
```

No change to chaining code. Only analytical `EarlyExercisePremium` returns dollar prices; both `RawPrice` and `NumericalEEP` return normalized values that the IC callback uses directly.

### Python bindings

- Add `NumericalEEP` to the `SurfaceContent` enum in `src/python/mango_bindings.cpp:701-703`.
- Update `AmericanPriceSurface` bindings to expose the new `create()` overload accepting an optional European surface.

### What stays unchanged

- **Last segment**: still uses analytical EEP with BSM subtraction
- **PriceTableBuilder core**: `from_vectors()`, `make_batch()`, `estimate_pde_grid()`, `fit_coeffs()`, `repair_failed_slices()` — all unchanged; we call the pipeline twice
- **InterpolatedIVSolver**: queries `price()` / `vega()` as before

## Cost

| Metric | Current (RawPrice) | Numerical EEP |
|--------|-------------------|---------------|
| PDE solves per (σ,r) per segment | 1 | 2 |
| Build time (20 pairs, one segment) | ~280 ms | ~560 ms (estimate; benchmark to confirm) |
| B-spline storage per segment | 1 surface | 2 surfaces |
| Query cost per segment | ~500 ns | ~1 μs (2 B-spline evals) |

Build time estimate is approximate — the second extraction, repair, and B-spline fit add overhead beyond the PDE solve. Benchmark after implementation to confirm.

## Testing

1. **European PDE grid convergence**: For the last segment (standard payoff IC), verify that the European PDE solve converges toward BSM as grid refines. At default accuracy, expect agreement within PDE truncation error (~`O(dx²)`), not machine precision.
2. **Round-trip accuracy**: Compare interpolated IV from numerical-EEP segments against FDM IV at several strikes. Target: < 10 bps error at scale=1 (vs current 20-90 bps for RawPrice).
3. **Scale=2 convergence**: Verify that grid refinement improves (or at least does not degrade) accuracy for chained segments.
4. **Greeks consistency**: Verify that `delta()`, `gamma()`, `vega()`, `theta()` from numerical-EEP segments match finite-difference estimates from `price()`.
5. **Softplus reconstruction gap**: Measure `|EEP_spline + Eu_spline - V_american_direct|` across the domain. The softplus floor introduces a small positive bias where true EEP ≈ 0. Document tolerated deviation.
6. **Regression**: Existing segmented IV tests must continue to pass.
