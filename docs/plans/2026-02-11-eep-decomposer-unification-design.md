# Unify EEP Decomposer with EEPSurfaceAdapter

## Problem

The EEP lifecycle has two halves that are disconnected:

| Phase | Code | European source |
|---|---|---|
| Build-time decomposition | `analytical_eep_decompose()` in `eep/eep_decomposer.hpp` | Hardcodes `EuropeanOptionSolver` |
| Query-time reconstruction | `EEPSurfaceAdapter` via `AnalyticalEEP` | Calls `eep.european_price()` |

Both compute European prices via Black-Scholes, but independently. When
numerical EEP arrives (European from PDE, not Black-Scholes), the
decomposer must use the *same* European source as the query-time
strategy. Otherwise the decompose/reconstruct cycle won't round-trip.

The EEP strategy should be the single authority on European prices,
used at both build and query time.

## Design

### Core change

`analytical_eep_decompose(accessor, option_type, dividend_yield)` becomes
`eep_decompose(accessor, eep)` -- generic over both accessor AND EEP
strategy:

```cpp
template <EEPAccessor A, EEPStrategy EEP>
void eep_decompose(A&& accessor, const EEP& eep) {
    const size_t n = accessor.size();
    const double strike = accessor.strike();
    for (size_t i = 0; i < n; ++i) {
        double am = accessor.american_price(i);
        double eu = eep.european_price(accessor.spot(i), strike,
                        accessor.tau(i), accessor.sigma(i), accessor.rate(i));
        accessor.set_value(i, eep_floor(am - eu));
    }
}
```

`compute_eep()` gains an EEP overload:

```cpp
template <EEPStrategy EEP>
double compute_eep(double american_price, double spot, double strike,
                   double tau, double sigma, double rate, const EEP& eep) {
    return eep_floor(american_price
                     - eep.european_price(spot, strike, tau, sigma, rate));
}
```

The old signatures become wrappers that construct `AnalyticalEEP`
internally, for backward compatibility.

### File moves

| Before | After | Reason |
|---|---|---|
| `table/eep_surface_adapter.hpp` | `table/eep/eep_surface_adapter.hpp` | EEP component belongs with EEP files |
| `table/bspline/eep_decomposer.hpp` | `table/bspline/bspline_tensor_accessor.hpp` | Only contains `BSplineTensorAccessor`; old name is misleading |

### File structure after

```
src/option/table/eep/
    analytical_eep.hpp          # EEP strategy: BS European (unchanged)
    identity_eep.hpp            # EEP strategy: no decomposition (unchanged)
    eep_decomposer.hpp          # eep_decompose(accessor, eep) -- generic loop
    eep_surface_adapter.hpp     # query-time adapter (moved from parent)

src/option/table/bspline/
    bspline_tensor_accessor.hpp # BSplineTensorAccessor (renamed from eep_decomposer.hpp)
```

## Affected files

### eep/eep_decomposer.hpp (modify)

- Add `eep_decompose(A&&, const EEP&)` template
- Add `compute_eep(american, spot, strike, tau, sigma, rate, eep)` template
- Keep `eep_floor()` as-is
- Keep old `analytical_eep_decompose()` as thin wrapper (deprecated)
- Keep old `compute_eep()` as thin wrapper (deprecated)

### eep/eep_surface_adapter.hpp (move only)

- `git mv table/eep_surface_adapter.hpp table/eep/eep_surface_adapter.hpp`
- Update include path in 4 files:
  - `bspline/bspline_surface.hpp`
  - `chebyshev/chebyshev_surface.hpp`
  - `tests/surface_concepts_test.cc`
  - `table/BUILD.bazel`

### bspline/bspline_tensor_accessor.hpp (rename only)

- `git mv bspline/eep_decomposer.hpp bspline/bspline_tensor_accessor.hpp`
- Update include path in 15 files (tests, benchmarks, factory)
- Update Bazel target name

### Call site updates (~18 files)

All call sites of `analytical_eep_decompose(accessor, type, yield)`:
- Construct `AnalyticalEEP eep(type, yield)` once
- Call `eep_decompose(accessor, eep)`
- Where the same `eep` is later used for adapter construction, reuse it

Key files:
- `adaptive_grid_builder.cpp`
- `interpolated_iv_solver.cpp`
- `chebyshev/chebyshev_table_builder.cpp`
- 15 test/benchmark files

## What stays unchanged

- `EEPSurfaceAdapter` template -- already correct (parameterized on EEP)
- `AnalyticalEEP` -- no API change needed
- `IdentityEEP` -- no API change needed
- `EEPAccessor` concept -- unchanged
- `BSplineTensorAccessor` -- only file rename, no code change
- `ChebyshevSplineAccessor` -- no change (adapts different storage)
- `SurfaceInterpolant`, `CoordinateTransform` concepts -- unrelated

## Future: NumericalEEP

With this design, adding numerical EEP is:

1. Create `eep/numerical_eep.hpp` implementing `EEPStrategy`
2. Its `european_price()` uses PDE or other numerical method
3. Pass to `eep_decompose()` at build time
4. Pass to `EEPSurfaceAdapter` at query time
5. Round-trip is guaranteed: same strategy, same European computation
