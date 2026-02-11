# Chebyshev Surface Integration Design

## Goal

Wire the generic `ChebyshevInterpolant<N, Storage>` into the pluggable
surface architecture so it can serve as a drop-in alternative to B-spline
for American option pricing and IV solving.

## Architecture

Three layers: adapter (type aliases), builder (PDE sampling + construction),
and benchmark (accuracy comparison against B-spline and FDM reference).

`ChebyshevInterpolant<4, TuckerTensor<4>>` already satisfies
`SurfaceInterpolant<T, 4>` directly — no wrapper needed. Compose with
the existing `EEPSurfaceAdapter`, `StandardTransform4D`, and
`AnalyticalEEP` to produce a full price surface.

## Type Aliases

```cpp
// Leaf: Chebyshev interpolant + EEP decomposition
using ChebyshevLeaf = EEPSurfaceAdapter<
    ChebyshevInterpolant<4, TuckerTensor<4>>,
    StandardTransform4D,
    AnalyticalEEP>;

// Full surface with bounds metadata
using ChebyshevSurface = PriceTable<ChebyshevLeaf>;
```

`ChebyshevSurface` satisfies `PriceSurface` and plugs directly into
`InterpolatedIVSolver<ChebyshevSurface>`.

## Builder

Samples PDE solutions at Chebyshev-Gauss-Lobatto nodes and constructs
the interpolant. Key differences from B-spline:

- Nodes are fixed (CGL points), not arbitrary grids
- No fitting step — values at CGL nodes are the representation
- Tucker compression replaces B-spline coefficient extraction

### Config

```cpp
struct ChebyshevTableConfig {
    std::array<size_t, 4> num_pts;   // CGL nodes per axis (m, tau, sigma, rate)
    Domain<4> domain;                // {lo, hi} per axis
    double K_ref;
    OptionType option_type;
    double dividend_yield;
    double tucker_epsilon;           // 0 = use RawTensor (no compression)
};
```

### Build Pipeline

1. Generate CGL nodes per axis from config
2. Create (sigma, rate) batch — one `PricingParams` per (sigma, rate) pair
3. Solve batch via `BatchAmericanOptionSolver` in chain mode:
   - Shared spatial grid across all (sigma, rate) pairs
   - Snapshot times = tau CGL nodes
4. For each (sigma, rate) result, build cubic spline from PDE spatial
   solution at each tau snapshot
5. For each (m, tau, sigma, rate) CGL node:
   - Evaluate American price via spline interpolation at log-moneyness m
   - Compute European price via Black-Scholes
   - EEP = American - European
6. Build `ChebyshevInterpolant<4, TuckerTensor<4>>` from EEP values

### Result

```cpp
struct ChebyshevTableResult {
    ChebyshevSurface surface;
    size_t n_pde_solves;
    double build_seconds;
};

std::expected<ChebyshevTableResult, PriceTableError>
build_chebyshev_table(const ChebyshevTableConfig& config);
```

## Benchmark

Add `run_chebyshev_4d()` to `benchmarks/interp_iv_safety.cc`:

1. Build `ChebyshevSurface` via `build_chebyshev_table()`
2. For each (vol, maturity, strike) test point:
   - Solve IV via Brent on `surface.price()`
   - Compare against FDM reference IV
   - Error in basis points
3. Print heatmap (same format as `run_vanilla()`)
4. Report compression ratio and per-query timing

Wire into main: `if (want("chebyshev-4d")) run_chebyshev_4d();`

Same 144 test points (2 vols x 8 maturities x 9 strikes) as the
B-spline vanilla benchmark for direct comparison.

## File Layout

```
src/option/table/chebyshev/
├── chebyshev_surface.hpp           # Type aliases
├── chebyshev_table_builder.hpp     # Config + build function declaration
├── chebyshev_table_builder.cpp     # Build implementation
└── BUILD.bazel

benchmarks/
└── interp_iv_safety.cc             # Add run_chebyshev_4d()

tests/
└── chebyshev_surface_test.cc       # Unit tests
```

## Tests

- Static assert: `ChebyshevSurface` satisfies `PriceSurface` concept
- `build_chebyshev_table()` succeeds with valid config
- Round-trip: polynomial EEP recovers exactly through surface
- IV solve: build surface, solve IV, verify against FDM within tolerance

## Not In Scope

- Discrete dividend support (future: per-maturity 3D tensors)
- Adaptive node placement
- RawTensor variant (Tucker only for now; RawTensor alias available if needed)
- Replacing B-spline in production IV solver factory
