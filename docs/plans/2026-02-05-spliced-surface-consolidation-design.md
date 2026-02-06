# Spliced Surface Consolidation — Design Document

## Problem Statement

The codebase has three separate surface classes that follow the same pattern:

1. **PerMaturityPriceSurface** — splits by τ, linear interpolation
2. **SegmentedPriceSurface** — splits by τ (dividend ex-dates), discrete lookup
3. **SegmentedMultiKRefSurface** — splits by K_ref, bracket interpolation

Each class duplicates the same structure:
- Collection of lower-dimensional surfaces
- Strategy to find relevant slice(s)
- Coordinate transformation
- Value combination

This duplication causes:
- ~1200 lines of similar code
- Difficult composition (e.g., dividends + per-maturity)
- Inconsistent APIs between classes
- Maintenance burden

## Requirements

1. **Unified abstraction** — single template covering all three patterns
2. **Composable** — enable `MultiKRefSurface<SegmentedSurface<PerMaturitySurface>>`
3. **Performant** — maintain ~500ns query time
4. **Type-safe** — use C++23 concepts to enforce constraints
5. **Clean migration** — remove old classes entirely

## Design

### Core Abstraction

```cpp
template <PriceSurface Inner,
          SplitStrategy Split,
          SliceTransform Xform,
          CombineStrategy Combiner>
class SplicedSurface;
```

**Four pluggable components:**

| Component | Purpose | Examples |
|-----------|---------|----------|
| `Inner` | The surface type being wrapped | `PriceTableSurface3DAdapter`, `AmericanPriceSurfaceAdapter` |
| `Split` | Find slice indices + weights | `LinearBracket`, `SegmentLookup`, `KRefBracket` |
| `Xform` | Transform query coordinates | `IdentityTransform`, `SegmentedTransform`, `MaturityTransform` |
| `Combiner` | Blend sample values | `WeightedSum` |

### Canonical Query Type

```cpp
struct PriceQuery {
    double spot;
    double strike;
    double tau;
    double sigma;
    double rate;
};
```

All surfaces accept `PriceQuery`. Adapters wrap existing surfaces that use different signatures.

### Split Strategies

**LinearBracket** — 2 samples with lerp weights:
```cpp
// At tau=0.7 with grid [0.5, 1.0]:
// Returns [{idx=0, w=0.6}, {idx=1, w=0.4}]
```

**SegmentLookup** — 1 sample, discrete selection:
```cpp
// At tau=0.7 with segments [0,0.5], [0.5,1.0]:
// Returns [{idx=1, w=1.0}]
```

**KRefBracket** — 2 samples with linear strike interpolation:
```cpp
// At K=95 with K_refs [80, 100, 120]:
// Returns [{idx=0, w=0.25}, {idx=1, w=0.75}]
```

### Transforms

**IdentityTransform** — pass-through:
```cpp
PriceQuery to_local(size_t, const PriceQuery& q) { return q; }
double normalize_value(size_t, const PriceQuery&, double raw) { return raw; }
```

**SegmentedTransform** — dividend adjustment:
```cpp
PriceQuery to_local(size_t i, const PriceQuery& q) {
    PriceQuery out = q;
    out.tau = clamp(q.tau - tau_start[i], tau_min[i], tau_max[i]);
    if (content[i] == EEP) {
        out.spot = compute_spot_adjustment(q.spot, T - q.tau, T - tau_start[i]);
    }
    if (content[i] == RawPrice) {
        out.strike = K_ref;
    }
    return out;
}
```

**MaturityTransform** — EEP reconstruction:
```cpp
double normalize_value(size_t, const PriceQuery& q, double eep) {
    return eep + bs_price(q.spot, q.strike, q.tau, q.sigma, q.rate, div_yield, type);
}
```

**KRefTransform** — normalize by K_ref:
```cpp
double normalize_value(size_t i, const PriceQuery&, double raw) {
    return raw / k_refs[i];
}
```

### Type Aliases

```cpp
// Per-maturity: τ interpolation over 3D EEP surfaces
using PerMaturitySurface = SplicedSurface<
    PriceTableSurface3DAdapter,
    LinearBracket,
    MaturityTransform,
    WeightedSum>;

// Segmented: dividend segments with spot adjustment
template<PriceSurface Inner = AmericanPriceSurfaceAdapter>
using SegmentedSurface = SplicedSurface<
    Inner,
    SegmentLookup,
    SegmentedTransform,
    WeightedSum>;

// Multi-K_ref: strike bracket interpolation
template<PriceSurface Inner = SegmentedSurface<>>
using MultiKRefSurface = SplicedSurface<
    Inner,
    KRefBracket,
    KRefTransform,
    WeightedSum>;
```

### Composition Example

For a 2-year option with multiple maturities AND multiple dividends:

```cpp
// Per-maturity inside dividend segments
using ComposedSurface = SegmentedSurface<PerMaturitySurface>;

// Or with K_ref interpolation on top
using FullSurface = MultiKRefSurface<SegmentedSurface<PerMaturitySurface>>;
```

Query flow for `FullSurface`:
1. `KRefBracket` finds K_ref indices [lo, hi] and weights
2. For each K_ref slice:
   - `SegmentLookup` finds dividend segment for τ
   - `SegmentedTransform` adjusts spot, converts τ→τ_local
   - `LinearBracket` finds maturity indices and weights
   - `MaturityTransform` reconstructs EEP→full price
3. `WeightedSum` combines all samples

### Performance

**Query path:**
```
price(q):
  br = split.bracket(split.key(q))     // O(log n) or O(n) depending on strategy
  for (idx, w) in br:                   // 1-4 iterations
    local = xform.to_local(idx, q)      // O(1)
    raw = slice[idx].price(local)       // O(1) B-spline eval
    norm = xform.normalize(idx, q, raw) // O(1)
    samples.push(norm, w)
  return combine(samples)                // O(1)
```

**Expected:** ~500ns for single-level, ~1-2μs for composed surfaces.

### File Structure

**Before (10 files, ~1200 lines):**
```
per_maturity_price_surface.{hpp,cpp}
segmented_price_surface.{hpp,cpp}
segmented_multi_kref_surface.{hpp,cpp}
segmented_price_table_builder.{hpp,cpp}
segmented_multi_kref_builder.{hpp,cpp}
```

**After (3 files, ~600 lines):**
```
spliced_surface.hpp              # Template + strategies + transforms + adapters + aliases
spliced_surface_builder.hpp      # Builder function declarations
spliced_surface_builder.cpp      # Builder implementations
```

## Alternatives Considered

### 1. Keep separate classes, add composition layer

**Pros:** No breaking changes
**Cons:** Still duplicated code, composition awkward

### 2. Type erasure (virtual interface)

**Pros:** Runtime flexibility
**Cons:** vtable overhead, misses 500ns target

### 3. std::variant of inner types

**Pros:** No vtable
**Cons:** Combinatorial explosion, every query is a visit

**Chosen:** Templates with type aliases — zero overhead, natural composition, clean API.

## Migration Path

1. Add new components to `spliced_surface.hpp`
2. Add builder functions
3. Update consumers to use new types
4. Delete old classes
5. Consolidate tests

## Risks

1. **Template error messages** — mitigated by concepts and static_asserts
2. **Build time increase** — mitigated by keeping templates header-only
3. **API changes** — acceptable since we're ignoring compatibility

## Success Criteria

- [ ] All three patterns expressible as type aliases
- [ ] Composition works: `SegmentedSurface<PerMaturitySurface>`
- [ ] Query time ≤1μs for composed surfaces
- [ ] ~1200 lines removed
- [ ] All tests passing
