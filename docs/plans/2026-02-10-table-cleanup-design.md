# Price Table Cleanup Design

## Goal

Clean up `src/option/table/` to separate shared infrastructure from
B-spline-specific types, remove legacy naming, and establish a symmetric
structure between the B-spline and Chebyshev backends.

## Principles

- **Type system encodes backend, EEP strategy, transform, dimensionality.**
  No runtime enums duplicating what templates already express.
- **Interpolant owns its domain bounds.** No duplicate bounds in the wrapper.
- **`PriceTable<Inner>`** is the single top-level wrapper. Holds runtime
  metadata (option_type, K_ref, dividend_yield). The unit of querying and
  (eventually) persistence.
- **YAGNI on persistence.** This design enables a clean persistence story
  later but does not implement serialization.

## Renames

| Old | New | Reason |
|-----|-----|--------|
| `PriceTable<T>` | `PriceTable<T>` | Describes role, not implementation |
| `StandardLeaf` | `BSplineLeaf` | "Standard" is not descriptive |
| `StandardSurface` | `BSplinePriceTable` | Symmetric with `ChebyshevPriceTable` |
| `SegmentedLeaf` | `BSplineSegmentedLeaf` | Symmetric with `ChebyshevSegmentedLeaf` |
| `StandardTransform4D` | *(keep)* | Shared; "standard" describes the coordinate mapping |

## Removals

| Type | Reason |
|------|--------|
| `PriceTableMetadata` | Fields split: K_ref/dividends → `PriceTable`, bounds → interpolant |
| `SurfaceContent` enum | Redundant with type-level `AnalyticalEEP` vs `IdentityEEP` |

## Moves

| File | From | To | Notes |
|------|------|----|-------|
| `dividend_utils.hpp` | `table/` | `src/option/` | Applicable to FDM solver too |

## Merges into `bspline/`

| Old file | Merges into | Notes |
|----------|-------------|-------|
| `price_table_axes.hpp` | `bspline_surface.hpp` | Axes are part of the surface |
| `price_tensor.hpp` | `bspline_builder.cpp` | Build artifact, not public |
| `price_table_config.hpp` | `bspline_builder.hpp` | Builder config |
| `price_table_grid_estimator.hpp` | `bspline_builder.hpp` | Builder config |
| `recursion_helpers.hpp` | `bspline_builder.cpp` | Implementation detail |
| `slice_cache.hpp` | `bspline_slice_cache.hpp` | Rename with prefix |
| `error_attribution.hpp` | `bspline_slice_cache.hpp` | Same concern |
| `standard_surface.hpp/cpp` | `bspline_surface.hpp/cpp` | B-spline type aliases |

## `EEPDecomposer`

Currently takes `PriceTensor&` and `PriceTableAxes&` — both B-spline types
after this cleanup. Two options:

1. Move to `bspline/` (Chebyshev does inline EEP, same math).
2. Refactor `decompose()` to accept raw spans.

Decision deferred; either works.

## File layout after cleanup

```
src/option/
├── dividend_utils.hpp
└── table/
    ├── price_table.hpp                   # PriceTable<Inner>
    ├── price_surface_concept.hpp         # PriceSurface concept
    ├── surface_concepts.hpp              # SurfaceInterpolant, CoordinateTransform, EEPStrategy
    ├── eep_surface_adapter.hpp
    ├── split_surface.hpp
    ├── spliced_surface_builder.hpp/cpp
    ├── adaptive_grid_types.hpp
    ├── adaptive_grid_builder.hpp/cpp
    ├── eep/
    │   ├── analytical_eep.hpp
    │   ├── identity_eep.hpp
    │   └── eep_decomposer.hpp/cpp
    ├── splits/
    │   ├── tau_segment.hpp
    │   └── multi_kref.hpp
    ├── transforms/
    │   └── standard_4d.hpp
    ├── bspline/
    │   ├── bspline_surface.hpp/cpp       # Axes, surface, interpolant, type aliases
    │   ├── bspline_builder.hpp/cpp       # Builder + config + estimator + tensor
    │   ├── bspline_segmented_builder.hpp/cpp
    │   ├── bspline_workspace.hpp/cpp     # Persistence (disabled, fix later)
    │   └── bspline_slice_cache.hpp       # SliceCache + ErrorBins
    └── chebyshev/
        ├── chebyshev_surface.hpp         # Type aliases
        ├── chebyshev_table_builder.hpp/cpp
        └── pde_slice_cache.hpp
```

## Type alias pattern (both backends)

```cpp
// bspline/bspline_surface.hpp
using BSplineLeaf = EEPSurfaceAdapter<
    SharedBSplineInterp<4>, StandardTransform4D, AnalyticalEEP>;
using BSplinePriceTable = PriceTable<BSplineLeaf>;

using BSplineSegmentedLeaf = EEPSurfaceAdapter<
    SharedBSplineInterp<4>, StandardTransform4D, IdentityEEP>;

// chebyshev/chebyshev_surface.hpp
using ChebyshevLeaf = EEPSurfaceAdapter<
    ChebyshevInterpolant<4, TuckerTensor<4>>,
    StandardTransform4D, AnalyticalEEP>;
using ChebyshevPriceTable = PriceTable<ChebyshevLeaf>;

using ChebyshevSegmentedLeaf = EEPSurfaceAdapter<
    ChebyshevInterpolant<4, RawTensor<4>>,
    StandardTransform4D, IdentityEEP>;
```

## `PriceTable<Inner>` (was `PriceTable`)

```cpp
template <typename Inner>
class PriceTable {
public:
    // Runtime metadata
    OptionType option_type() const;
    double dividend_yield() const;
    double K_ref() const;

    // Delegates to inner interpolant
    double m_min() const;
    double m_max() const;
    double tau_min() const;
    // ...

    // Pricing
    double price(double spot, double strike, double tau,
                 double sigma, double rate) const;
    double vega(double spot, double strike, double tau,
                double sigma, double rate) const;

private:
    Inner inner_;
    OptionType option_type_;
    double dividend_yield_;
    double K_ref_;
};
```
