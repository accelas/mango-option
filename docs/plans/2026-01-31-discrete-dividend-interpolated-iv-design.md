# Discrete Dividend Support for Interpolated IV

## Problem

The interpolated IV path (`IVSolverInterpolated` → `AmericanPriceSurface` → `PriceTableBuilder`) supports only continuous dividend yield. Discrete dividends are stored in data structures but rejected at every computation stage. Equity options on dividend-paying stocks require discrete dividend handling for accurate interpolated IV.

## Approach: Maturity Segmentation with Backward Chaining + Multi-K_ref

For N discrete dividends, split the maturity axis into N+1 segments at dividend dates. Build each segment as an independent price surface. Chain segments backward in time, mirroring the PDE's backward-in-time structure.

Cash dividends break price homogeneity in (S, K): `P(λS, λK, D) ≠ λP(S, K, D)`. A single K_ref surface cannot be scaled to arbitrary strikes. To restore strike portability, build segmented surfaces at 2-3 reference strikes and interpolate in strike at query time.

Within each segment, no discrete dividends exist — only continuous yield — so the B-spline interpolation remains well-behaved. The last segment (closest to expiry) uses standard EEP decomposition. Earlier segments store raw American prices directly, since EEP decomposition requires a closed-form European price (which does not exist when the terminal condition is a chained surface rather than the standard payoff).

## Component Hierarchy

```
IVSolverInterpolated<SegmentedMultiKRefSurface>
  └── SegmentedMultiKRefSurface
        └── [K_ref = 0.8S] → SegmentedPriceSurface
        │     └── Segment N   (EEP, payoff IC)
        │     └── Segment N-1 (RawPrice, chained IC)
        │     └── ...
        │     └── Segment 0   (RawPrice, chained IC)
        └── [K_ref = 1.0S] → SegmentedPriceSurface
        │     └── ...
        └── [K_ref = 1.2S] → SegmentedPriceSurface
              └── ...
```

## Components

### `PriceSurface` concept

Defines the interface that `AmericanPriceSurface` and `SegmentedMultiKRefSurface` satisfy. These are the only two types the IV solver sees:

```cpp
template <typename S>
concept PriceSurface = requires(const S& s, double spot, double strike,
                                double tau, double sigma, double rate) {
    { s.price(spot, strike, tau, sigma, rate) } -> std::same_as<double>;
    { s.vega(spot, strike, tau, sigma, rate) } -> std::same_as<double>;
    { s.m_min() } -> std::convertible_to<double>;
    { s.m_max() } -> std::convertible_to<double>;
    { s.tau_min() } -> std::convertible_to<double>;
    { s.tau_max() } -> std::convertible_to<double>;
    { s.sigma_min() } -> std::convertible_to<double>;
    { s.sigma_max() } -> std::convertible_to<double>;
    { s.rate_min() } -> std::convertible_to<double>;
    { s.rate_max() } -> std::convertible_to<double>;
};
```

`AmericanPriceSurface` already provides `price()` and `vega()`. Add bounds accessors to formalize what the IV solver already reads from metadata.

### `SegmentedPriceSurface` (internal)

Owns N+1 `AmericanPriceSurface` segments plus the dividend schedule for a single K_ref. Internal to `SegmentedMultiKRefSurface` — not exposed to the solver or user API. Evaluates prices at its own K_ref only; strike portability is handled by the multi-K_ref layer above.

```cpp
class SegmentedPriceSurface {
    struct Segment {
        AmericanPriceSurface surface;
        double tau_start;  // segment start in global τ
        double tau_end;    // segment end in global τ
    };

    std::vector<Segment> segments_;
    std::vector<std::pair<double, double>> dividends_;  // (calendar time, amount)
    double K_ref_;
    double T_;  // expiry in calendar time
};
```

**Query algorithm** for `price(spot, K_ref, tau, sigma, rate)`:

1. Convert to calendar time: `t_query = T - tau`.
2. Find the segment containing `tau` (i.e., `tau_start < tau <= tau_end`).
3. Compute spot adjustment for dividends between query time and the segment's expiry-side boundary:
   ```
   t_boundary = T - segment.tau_start
   S_adj = spot - sum of D_i where t_query < t_i <= t_boundary
   ```
   Dividend at `t_query`: not subtracted (already ex-div). Dividend at boundary: subtracted.
4. Clamp `S_adj` to small positive ε if `S_adj <= 0`.
5. Convert to local segment time: `tau_local = tau - segment.tau_start`.
6. Return `segment.surface.price(S_adj, K_ref, tau_local, sigma, rate)`.

**Vega**: determined by segment type.
- **EEP segments** (last segment): use the existing analytic vega path via `AmericanPriceSurface::vega()`.
- **RawPrice segments**: central finite difference on `price()`:
  ```
  ε_σ = max(1e-4, 1e-4 * σ)
  vega ≈ (P(σ + ε_σ) - P(σ - ε_σ)) / (2 * ε_σ)
  ```

**Bounds**: `tau_min`/`tau_max` span the full maturity range. Moneyness, vol, and rate bounds come from the shared expanded grid.

#### Worked Example: One Dividend

Option expiry `T = 1.0` year. One dividend `D = 2.0` at `t_div = 0.5` (calendar time).

Segments (in τ = time-to-expiry):
- Segment 1 (last, closest to expiry): `τ ∈ [0, 0.5]`, local `τ_local ∈ [0, 0.5]`, EEP mode, payoff IC
- Segment 0 (earlier): `τ ∈ (0.5, 1.0]`, local `τ_local ∈ [0, 0.5]`, RawPrice mode, chained IC

Query: `price(S=100, K_ref=100, τ=0.8, σ=0.2, r=0.05)`
1. `t_query = 1.0 - 0.8 = 0.2` (calendar time)
2. `τ=0.8` falls in Segment 0 (`0.5 < 0.8 <= 1.0`)
3. `t_boundary = T - tau_start = 1.0 - 0.5 = 0.5`. Dividend at `t_div=0.5`: `t_query=0.2 < 0.5 <= 0.5` → subtracted. `S_adj = 100 - 2 = 98`
4. `tau_local = 0.8 - 0.5 = 0.3`
5. Return `segment_0.surface.price(98, 100, 0.3, 0.2, 0.05)`

Query: `price(S=100, K_ref=100, τ=0.3, σ=0.2, r=0.05)`
1. `t_query = 1.0 - 0.3 = 0.7`
2. `τ=0.3` falls in Segment 1 (`0 < 0.3 <= 0.5`)
3. `t_boundary = T - tau_start = 1.0 - 0.0 = 1.0`. No dividends where `0.7 < t_i <= 1.0` (dividend is at 0.5). `S_adj = 100`
4. `tau_local = 0.3 - 0.0 = 0.3`
5. Return `segment_1.surface.price(100, 100, 0.3, 0.2, 0.05)`

### `SegmentedMultiKRefSurface`

Wraps multiple `SegmentedPriceSurface` instances at different K_ref values. Provides strike portability via interpolation. Satisfies `PriceSurface`.

```cpp
class SegmentedMultiKRefSurface {
    struct Entry {
        double K_ref;
        SegmentedPriceSurface surface;
    };

    std::vector<Entry> entries_;  // sorted by K_ref

    // Cached bounds (intersection across all entries)
    double m_min_, m_max_;
    double tau_min_, tau_max_;
    double sigma_min_, sigma_max_;
    double rate_min_, rate_max_;
};
```

**Query algorithm** for `price(spot, strike, tau, sigma, rate)`:

1. Find the two entries with K_ref values bracketing `strike`. If outside range, clamp to nearest.
2. Compute interpolation weight: `w = (strike - K_ref_lo) / (K_ref_hi - K_ref_lo)`.
3. Evaluate each entry's surface at its own K_ref:
   ```
   P_lo = entry_lo.surface.price(spot, K_ref_lo, tau, sigma, rate)
   P_hi = entry_hi.surface.price(spot, K_ref_hi, tau, sigma, rate)
   ```
4. Interpolate: `P = (1 - w) * P_lo + w * P_hi`.

**Vega**: same interpolation applied to per-entry vega values.

**Bounds**: computed at construction time as intersection across all entries: `m_min = max(m_min_i)`, `m_max = min(m_max_i)`, etc. These bounds are used by both `price()` and `vega()` paths, and by `IVSolverInterpolated`'s `is_in_bounds` check. If the intersection is empty or too narrow, `create()` returns a validation error. The `SegmentedMultiKRefBuilder` logs a diagnostic warning when intersection shrinks the domain by more than 10% compared to any individual entry.

Moneyness grid expansion is applied per K_ref entry before building. The bounds intersection is computed after all entries are built, operating on already-expanded per-entry bounds.

### `SegmentedPriceTableBuilder`

Orchestrates backward-chained construction for a single K_ref. Delegates per-segment builds to `PriceTableBuilder`.

**Algorithm:**

1. Sort dividends by calendar time, convert to τ, define N+1 segment boundaries. Expand moneyness grid by `max(Dᵢ/K_ref)`, clamped so `m_min > 0`. Each segment uses local time `τ_local ∈ [0, Δτ_j]`.

2. **Last segment** (closest to expiry): standard `PriceTableBuilder` with payoff IC, EEP decomposition, wrapped in `AmericanPriceSurface` with `SurfaceContent::EarlyExercisePremium`.

3. **For j = N-1 ... 0** (backward): custom IC from `surface_{j+1}` evaluated at τ=0. Single PDE solve with American obstacle. Store raw American prices in B-spline (no EEP subtraction). Wrap in `AmericanPriceSurface` with `SurfaceContent::RawPrice`. Per-segment metadata must have `discrete_dividends = {}` (the dividend schedule lives only in `SegmentedPriceSurface`).

4. Assemble all segments + dividend schedule into `SegmentedPriceSurface`.

**Parallelism:** Segments build sequentially (each depends on the previous). Within each segment, the batch of (σ, r) solves runs in parallel as today.

### `SegmentedMultiKRefBuilder`

Top-level builder. Chooses K_ref values, builds a `SegmentedPriceSurface` for each, assembles into `SegmentedMultiKRefSurface`.

```cpp
struct MultiKRefConfig {
    std::vector<double> K_refs;    // explicit list, or empty for auto
    int K_ref_count = 3;           // used when K_refs is empty
    double K_ref_span = 0.2;       // ±20% around spot for auto mode
};
```

**Auto K_ref selection**: log-spaced around spot, e.g., `{0.8S, 1.0S, 1.2S}` for `K_ref_count=3, K_ref_span=0.2`.

Build loop:
```cpp
for (double K_ref : K_refs) {
    auto seg = SegmentedPriceTableBuilder::build(K_ref, grids, dividends, config);
    entries.push_back({K_ref, std::move(seg)});
}
return SegmentedMultiKRefSurface{std::move(entries)};
```

### `IVSolverInterpolated` templatization

The solver becomes `IVSolverInterpolated<Surface>` where `Surface` satisfies `PriceSurface`. Newton iteration logic, convergence, and error handling remain unchanged — the solver calls `surface_.price()` and `surface_.vega()`.

**Bounds handling**: the current solver reads bounds from `eep_surface().axes()` and `metadata` fields like `K_ref`. These are replaced by the `PriceSurface` concept accessors:
- `surface_.m_min()` / `surface_.m_max()` replace `axes[0]` range lookups
- `surface_.tau_min()` / `surface_.tau_max()` replace `axes[1]` range lookups
- `surface_.sigma_min()` / `surface_.sigma_max()` replace `axes[2]` range lookups
- `surface_.rate_min()` / `surface_.rate_max()` replace `axes[3]` range lookups
- The internal `K_ref_` field is removed from the solver. For `AmericanPriceSurface`, K_ref is embedded in the surface. For `SegmentedMultiKRefSurface`, K_ref is irrelevant (strike interpolation handles it).

**`create()` validation**: checks that bounds from the surface are valid (non-empty ranges, min < max). Returns `ValidationError` if bounds are degenerate.

**The template parameter is hidden from users.** A single factory function inspects the config and returns the appropriate solver:

```cpp
auto iv_solver = make_iv_solver(moneyness_grid, maturity_grid,
                                vol_grid, rate_grid, config);
auto result = iv_solver.solve(query);
```

Two paths:

1. **No discrete dividends**: factory builds a standard `AmericanPriceSurface` → `IVSolverInterpolated<AmericanPriceSurface>`.
2. **Discrete dividends**: factory builds via `SegmentedMultiKRefBuilder` → `IVSolverInterpolated<SegmentedMultiKRefSurface>`.

The user sees one factory and one solve interface. The `PriceSurface` concept erases the internal distinction. `SegmentedPriceSurface` is an implementation detail of `SegmentedMultiKRefSurface` and never appears in the public API.

## Changes to Existing Code

### `AmericanOptionSolver`
Add optional initial condition override. When provided, `solve()` passes it to `PDESolver::initialize()` instead of the static payoff function. The PDE solver already accepts any callable — this change is in `AmericanOptionSolver`, not in the PDE layer.

### `BatchAmericanOptionSolver`
Thread the optional IC through to individual solver instances. Concretely: add an optional IC field (same `std::function<void(span<const double>, span<double>)>` type) to the batch solver. In the batch solve loop, if an IC is set, pass it to each `AmericanOptionSolver` instance before calling `solve()`. The existing `SetupCallback` runs before solver construction and is not the right hook — the IC must be applied after the solver is constructed but before `solve()` calls `initialize()`.

### `PriceTableBuilder` (modified, not untouched)
Three additions to the existing builder:
- Accept optional IC via config or `build()` parameter. When provided, the IC is threaded through to `AmericanOptionSolver::solve()`.
- Accept optional `SurfaceContent` mode. When `RawPrice`, skip EEP subtraction during extraction.
- Allow `τ=0` in maturity grid when a custom IC is provided (relax existing `τ > 0` check via new `allow_tau_zero` flag in config).

These are additive changes — existing callers that don't set the new options see no behavior change.

### `AmericanPriceSurface`
Extend `create()` to accept `SurfaceContent::RawPrice` (currently rejects anything other than `EarlyExercisePremium`). Per-segment surfaces must have `metadata.discrete_dividends = {}`.

Handle `SurfaceContent::RawPrice` in pricing and Greeks. RawPrice surfaces are internal-only — used exclusively inside `SegmentedPriceSurface`, never by end users or the IV solver directly:
- `price()`: return spline value directly. No European addback, no `K/K_ref` scaling. Assert that `strike == K_ref` (programming error if violated — `SegmentedMultiKRefSurface` guarantees this by always passing the entry's K_ref).
- `vega()`: assert-fail / return NaN. RawPrice vega is never called directly — `SegmentedPriceSurface` computes FD vega at its layer instead of delegating to `AmericanPriceSurface::vega()`.

### `SurfaceContent` enum
Add `RawPrice` variant alongside `EarlyExercisePremium`.

### Bounds accessors on `AmericanPriceSurface`
Add `m_min()`, `m_max()`, `tau_min()`, `tau_max()`, `sigma_min()`, `sigma_max()`, `rate_min()`, `rate_max()` to satisfy the `PriceSurface` concept. These expose what the IV solver already reads from metadata/axes.

## Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Segmentation vs spot-adjustment at query time | Segmentation | Preserves accuracy; spot-adjustment approximation degrades for deep ITM and near-dividend options |
| Backward chaining vs independent segments | Backward chaining | Mirrors PDE structure; terminal condition of segment j depends on segment j+1 |
| New builder vs extending existing | New `SegmentedPriceTableBuilder` orchestrating modified `PriceTableBuilder` | Segmentation is orchestration; IC/RawPrice/τ=0 are additive changes to existing builder |
| EEP vs raw prices for earlier segments | Raw prices for earlier segments | EEP requires closed-form European price; custom IC breaks this assumption |
| Strike portability | Multi-K_ref with strike interpolation | Cash dividends break homogeneity; single K_ref cannot scale to arbitrary strikes; 2-3 K_ref surfaces with interpolation restores portability without 5D refactor |
| Concept vs type erasure for solver | Concept | Consistent with codebase style (CRTP, concepts); zero-cost abstraction |
| User API | Single factory, hidden template | Users pass a config with dividends; internal routing is invisible |
| τ=0 chaining | Allow τ=0 in builder when custom IC is provided | Avoids ε hack; injects exact terminal condition |
| Vega for RawPrice segments | Central finite differences, ε = max(1e-4, 1e-4*σ) | No analytic European component available; FD on the spline surface |

## Edge Cases

- **Dividend at or after expiry (`t_i >= T`)**: ignore. Filter before segmenting.
- **Dividend at or before valuation (`t_i <= 0`)**: ignore. Assume spot is already ex-div.
- **`S - D <= 0`**: clamp `S_adj` to small positive ε (`1e-12 * K_ref`). For puts, can also return intrinsic.
- **Near-zero segment length**: merge dividends closer than a minimum segment length. Sum amounts, use earliest date.
- **Two dividends on the same date**: sum amounts into a single dividend.
- **Moneyness grid expansion**: expand by `max(Dᵢ/K_ref)`. Validate `m_min > 0` after expansion; reject config if violated.

## Caveats

- **Raw price interpolation accuracy**: earlier segments store raw American prices, which have the early exercise kink. B-spline accuracy may be lower than EEP-based segments. Consider tighter grids for earlier segments. Monitor in testing.
- **Strike interpolation**: linear interpolation between 2-3 K_ref surfaces. Accuracy degrades for strikes far outside the K_ref range. Users can widen `K_ref_span` or provide explicit K_ref values for wider coverage.
- **Build cost**: 2-3x a single surface due to multi-K_ref. Each K_ref builds N+1 segments sequentially. Within each segment, batch (σ, r) solves are parallelized.
- **Moneyness normalization**: spot adjustment `S → S - D` translates to `m → m - D/K`. Each K_ref surface uses its own K_ref for normalization, so the multi-K_ref approach handles this correctly per-entry.
- **Bounds intersection**: `SegmentedMultiKRefSurface` uses the intersection of bounds across all K_ref entries. This may reduce the valid query domain compared to any single entry. Builder logs a warning when intersection shrinks domain by more than 10%.
