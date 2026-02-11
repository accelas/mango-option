# Split adaptive_grid_builder into per-backend free functions

## Problem

`AdaptiveGridBuilder` is a god class (2296-line .cpp) that groups B-spline and
Chebyshev builders behind a single interface. The header leaks all backend
types through a single class, and results use `std::any` that consumers
immediately cast back to concrete types.

## Design

Replace the class with free functions per backend. Move implementations next
to their interpolant code. Extract shared refinement infrastructure into a
common module.

### File layout

```
src/option/table/
  adaptive_refinement.hpp        # Shared: run_refinement(), callbacks, error metrics
  adaptive_refinement.cpp        # ~600 lines shared infrastructure
  adaptive_grid_types.hpp        # AdaptiveGridParams, IterationStats, IVGrid, etc.

src/option/table/bspline/
  bspline_adaptive.hpp           # build_adaptive_bspline(), ..._segmented()
  bspline_adaptive.cpp           # ~600 lines: cached surface, probe_and_build, refine

src/option/table/chebyshev/
  chebyshev_adaptive.hpp         # build_adaptive_chebyshev(), ..._segmented()
  chebyshev_adaptive.cpp         # ~600 lines: chebyshev build/refine, segments

Deleted:
  adaptive_grid_builder.hpp
  adaptive_grid_builder.cpp
```

### Common API (`adaptive_refinement.hpp`)

Callback types used by all backends:

```cpp
using BuildFn = std::function<std::expected<SurfaceHandle, PriceTableError>(
    const std::vector<double>& m, const std::vector<double>& tau,
    const std::vector<double>& sigma, const std::vector<double>& rate)>;

using ValidateFn = std::function<std::expected<double, PriceTableError>(
    double spot, double strike, double tau, double sigma, double rate)>;

using RefineFn = std::function<bool(
    std::vector<double>& m, std::vector<double>& tau,
    std::vector<double>& sigma, std::vector<double>& rate,
    int worst_dim, const ErrorBins& bins)>;
```

Core refinement loop:

```cpp
std::expected<RefinementResult, PriceTableError>
run_refinement(const AdaptiveGridParams& params,
               BuildFn build_fn, ValidateFn validate_fn,
               RefineFn refine_fn, const RefinementContext& ctx,
               ComputeErrorFn compute_error, const InitialGrids& initial);
```

Shared helpers: `compute_iv_error()`, `make_validate_fn()`,
`make_bs_vega_error_fn()`, `select_probes()`, `seed_grid()`, `linspace()`,
`expand_domain_bounds()`, `extract_chain_domain()`, LHS sampling.

Internal types: `RefinementContext`, `RefinementResult`, `InitialGrids`,
`ErrorBins`, `SurfaceHandle`.

### B-spline backend (`bspline_adaptive.hpp`)

```cpp
struct BSplineAdaptiveResult {
    std::shared_ptr<const PriceTableSurface> surface;
    std::vector<IterationStats> iterations;
    double achieved_max_error = 0.0;
    double achieved_avg_error = 0.0;
    bool target_met = false;
    size_t total_pde_solves = 0;
};

struct BSplineSegmentedAdaptiveResult {
    BSplineMultiKRefInner surface;
    IVGrid grid;
    int tau_points_per_segment;
};

std::expected<BSplineAdaptiveResult, PriceTableError>
build_adaptive_bspline(const AdaptiveGridParams& params,
                       const OptionGrid& chain,
                       PDEGridSpec pde_grid, OptionType type);

std::expected<BSplineSegmentedAdaptiveResult, PriceTableError>
build_adaptive_bspline_segmented(const AdaptiveGridParams& params,
                                 const SegmentedAdaptiveConfig& config,
                                 const IVGrid& domain);
```

`SliceCache` becomes local to `bspline_adaptive.cpp`.

### Chebyshev backend (`chebyshev_adaptive.hpp`)

Same pattern. `ChebyshevAdaptiveResult` holds `ChebyshevRawSurface`.
`PDESliceCache` becomes local to `chebyshev_adaptive.cpp`.

### Consumer changes

`interpolated_iv_solver.cpp` replaces:

```cpp
AdaptiveGridBuilder builder(*config.adaptive);
auto result = builder.build(chain, accuracy, config.option_type);
auto surface = std::any_cast<...>(result->typed_surface);
```

with:

```cpp
auto result = build_adaptive_bspline(*config.adaptive, chain, pde_grid, type);
auto& surface = result->surface;  // already typed, no cast
```

### Cleanup

- `AdaptiveResult` and `SegmentedAdaptiveResult` deleted from
  `adaptive_grid_types.hpp` (replaced by typed results per backend).
- `adaptive_grid_builder.{hpp,cpp}` deleted.
- Tests and benchmarks update includes to backend-specific headers.
