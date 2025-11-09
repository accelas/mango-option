# Interpolation Table Files - Quick Reference

## Modern C++ Implementation (Primary - 2025)

### Core Builder
- **Header**: `/home/user/mango-iv/src/price_table_4d_builder.hpp` (155 lines)
  - Main public API for 4D price table building
  - Class: `PriceTable4DBuilder`
  - Result struct: `PriceTable4DResult`
  - Entry point: `PriceTable4DBuilder::precompute()`

- **Implementation**: `/home/user/mango-iv/src/price_table_4d_builder.cpp` (271 lines)
  - Grid validation and pre-computation orchestration
  - (σ, r) loop structure (lines 151-211)
  - B-spline fitting integration

### Data Collection Pipeline

- **Snapshot Data**: `/home/user/mango-iv/src/snapshot.hpp` (47 lines)
  - Struct: `Snapshot` - PDE solution at time t
  - Contains: V(x,t), derivatives, spatial grid

- **Snapshot Collector**: `/home/user/mango-iv/src/price_table_snapshot_collector.hpp` (236 lines)
  - Class: `PriceTableSnapshotCollector : SnapshotCollector`
  - Converts PDE solutions to (m, τ) price table
  - Handles delta, gamma, theta computation
  - Grid caching optimization (2-3× speedup)

- **Snapshot Interpolator**: `/home/user/mango-iv/src/snapshot_interpolator.hpp` (238 lines)
  - Class: `SnapshotInterpolator`
  - 1D cubic spline interpolation from PDE grid to moneyness
  - Epoch-based derivative caching (LRU 2-slot cache)
  - Fast rebuild when grid unchanged

### Fast Evaluation (Tensor Product)

- **B-spline 4D Evaluator**: `/home/user/mango-iv/src/bspline_4d.hpp` (150+ lines)
  - Class: `BSpline4D_FMA`
  - Evaluation: ~100-200ns per query
  - Clamped cubic B-splines with FMA optimization
  - Tensor-product structure for 4D

- **B-spline Basis Functions**: `/home/user/mango-iv/src/bspline_basis_1d.hpp` (60+ lines)
  - Cox-de Boor recursion for basis evaluation

- **B-spline Fitter**: `/home/user/mango-iv/src/bspline_fitter_4d.hpp` (120+ lines)
  - Class: `SeparableBSplineFitter4D`
  - Fits B-spline coefficients to 300K price points
  - Separable least-squares approach

- **B-spline Utilities**: `/home/user/mango-iv/src/bspline_utils.hpp` (80+ lines)
  - Knot vector generation
  - Utility functions for B-spline operations

### Tests

- **Integration Tests**: `/home/user/mango-iv/tests/integration_5d_price_table_test.cc`
  - Full end-to-end precomputation tests
  - Verifies prices, greeks, fitting accuracy

- **Snapshot Collector Tests**: `/home/user/mango-iv/tests/price_table_snapshot_collector_test.cc`
  - Collection and conversion tests

---

## Legacy C Implementation (Reference - Pre-2025)

### Complete C API

- **Header**: `/home/user/mango-iv/legacy/src/price_table.h` (706 lines)
  - Complete C API for price tables
  - Struct: `OptionPriceTable`
  - Functions: `price_table_create()`, `price_table_precompute()`, 
    `price_table_save()`, `price_table_load()`
  - Greeks queries: `price_table_interpolate_vega_4d()`, etc.

- **Implementation**: `/home/user/mango-iv/legacy/src/price_table.c` (2400+ lines)
  - Full implementation with save/load
  - Lines 2072-2160: `price_table_save()` implementation
  - Lines 2163-2300: `price_table_load()` implementation
  - Binary format with versioning (Version 4 current)

### Example and Tests

- **Example**: `/home/user/mango-iv/legacy/examples/example_precompute_table.c` (200+ lines)
  - Complete usage example
  - Grid creation, precomputation, saving, querying
  - Sample interpolation queries

- **Tests**: `/home/user/mango-iv/legacy/tests/`
  - `price_table_test.cc` - Basic functionality tests
  - `price_table_slow_test.cc` - Longer running tests
  - `interpolation_test.cc` - Interpolation accuracy tests

---

## Absolute File Paths for Copy-Paste

### Modern C++ Core Files
```
/home/user/mango-iv/src/price_table_4d_builder.hpp
/home/user/mango-iv/src/price_table_4d_builder.cpp
/home/user/mango-iv/src/price_table_snapshot_collector.hpp
/home/user/mango-iv/src/snapshot_interpolator.hpp
/home/user/mango-iv/src/snapshot.hpp
/home/user/mango-iv/src/bspline_4d.hpp
/home/user/mango-iv/src/bspline_fitter_4d.hpp
/home/user/mango-iv/src/bspline_basis_1d.hpp
/home/user/mango-iv/src/bspline_utils.hpp
```

### Legacy C Core Files
```
/home/user/mango-iv/legacy/src/price_table.h
/home/user/mango-iv/legacy/src/price_table.c
/home/user/mango-iv/legacy/examples/example_precompute_table.c
```

### Supporting Infrastructure
```
/home/user/mango-iv/src/american_option.hpp
/home/user/mango-iv/src/american_option.cpp
/home/user/mango-iv/src/cubic_spline_solver.hpp
```

### Documentation
```
/home/user/mango-iv/docs/code_reference_price_table.md
/home/user/mango-iv/docs/INTERPOLATION_TABLE_ANALYSIS.md
/home/user/mango-iv/docs/plans/2025-10-29-price-table-precomputation-design.md
/home/user/mango-iv/docs/plans/2025-10-29-price-table-precompute-implementation.md
/home/user/mango-iv/docs/architecture_analysis_price_table_4d.md
```

---

## Data Flow Diagram

```
Input: 4D Grids (m, τ, σ, r)
   |
   v
PriceTable4DBuilder::create()
   |
   +-- Validate grids (must be sorted, ≥4 points)
   |
   v
PriceTable4DBuilder::precompute()
   |
   +-- For each (σ, r) pair (200 iterations):
   |   |
   |   +-- AmericanOptionSolver::solve()
   |   |   |
   |   |   +-- FDM PDE Solve (550ms)
   |   |   |
   |   |   v
   |   |-- For each maturity τ (30 iterations):
   |   |   |
   |   |   +-- Snapshot @time t_j
   |   |   |   |
   |   |   |   v
   |   |   +-- PriceTableSnapshotCollector::collect()
   |   |       |
   |   |       +-- SnapshotInterpolator::eval()
   |   |       |   Converts x-space → m-space
   |   |       |
   |   |       +-- Transform prices
   |   |       |   V = K_ref × V_norm
   |   |       |
   |   |       +-- Transform deltas, gammas, thetas
   |   |       |
   |   |       v
   |   |   Populate prices_4d[m_i, τ_j, σ, r]
   |
   +-- After all solves (300K prices):
   |
   +-- SeparableBSplineFitter4D::fit()
   |   |
   |   +-- Fit B-splines (1D at a time)
   |   |
   |   v
   |-- Generate B-spline coefficients
   |
   v
Output: PriceTable4DResult
   ├── evaluator: BSpline4D_FMA (ready for ~150ns queries)
   ├── prices_4d: Raw 300K price array
   ├── n_pde_solves: 200
   ├── precompute_time: ~3 minutes
   └── fitting_stats: Residuals, condition numbers
```

---

## Key Classes and Methods

### PriceTable4DBuilder (Entry Point)
```cpp
class PriceTable4DBuilder {
    static PriceTable4DBuilder create(
        std::vector<double> moneyness,
        std::vector<double> maturity,
        std::vector<double> volatility,
        std::vector<double> rate,
        double K_ref
    );
    
    expected<PriceTable4DResult, std::string> precompute(
        OptionType option_type,
        const AmericanOptionGrid& grid_config,
        double dividend_yield = 0.0
    );
};
```

### PriceTableSnapshotCollector (Data Collection)
```cpp
class PriceTableSnapshotCollector : public SnapshotCollector {
    explicit PriceTableSnapshotCollector(
        const PriceTableSnapshotCollectorConfig& config
    );
    
    expected<void, std::string> collect_expected(
        const Snapshot& snapshot
    );
    
    std::span<const double> prices() const;
    std::span<const double> deltas() const;
    std::span<const double> gammas() const;
    std::span<const double> thetas() const;
};
```

### BSpline4D_FMA (Fast Evaluation)
```cpp
class BSpline4D_FMA {
    BSpline4D_FMA(
        std::span<const double> m_grid,
        std::span<const double> t_grid,
        std::span<const double> v_grid,
        std::span<const double> r_grid,
        std::span<const double> coeffs
    );
    
    double eval(double m, double tau, double sigma, double rate) const;
};
```

### OptionPriceTable (Legacy C)
```c
typedef struct {
    size_t n_moneyness, n_maturity, n_volatility, n_rate, n_dividend;
    double *prices, *vegas, *gammas, *thetas, *rhos;
} OptionPriceTable;

// Creation
OptionPriceTable* price_table_create(...);

// Pre-computation
int price_table_precompute(OptionPriceTable *table, 
                          const AmericanOptionGrid *grid);

// Queries
double price_table_interpolate_4d(const OptionPriceTable *table,
                                  double m, double tau, 
                                  double sigma, double rate);

// I/O
int price_table_save(const OptionPriceTable *table, const char *filename);
OptionPriceTable* price_table_load(const char *filename);
```

---

## Performance Benchmarks

### Pre-computation (300K grid points)
- Single thread: ~50 minutes
- 16 cores (OpenMP): ~3 minutes
- Throughput: 1,667 options/second

### Query Performance
- B-spline 4D eval: 100-200ns
- Legacy cubic interp: 400-500ns
- Greeks (delta/gamma): 5-10µs
- IV calculation: <30µs

### Memory
- Evaluator (B-splines): ~8.5 MB resident
- Table + Greeks (Legacy): ~12 MB

---

## Grid Specifications (Typical)

### Moneyness (50 points, log-spaced)
- Range: 0.7 to 1.3 (70% to 130% of strike)
- Represents: Deep ITM to OTM

### Maturity (30 points, linear)
- Range: 0.027 to 2.0 years (10 days to 2 years)
- Typical expiry schedule

### Volatility (20 points, linear)
- Range: 0.10 to 0.80 (10% to 80%)
- Market vol surface coverage

### Rate (10 points, linear)
- Range: 0.0 to 0.10 (0% to 10%)
- Reasonable rate environment

### Total: 50 × 30 × 20 × 10 = 300,000 grid points

---

## Binary Format Details (Legacy C)

### File Header (256 bytes)
```
Magic:           0x50545442 ("PTTB")
Version:         4
n_moneyness:     50
n_maturity:      30
n_volatility:    20
n_rate:          10
n_dividend:      0
type:            OPTION_PUT
exercise:        AMERICAN
underlying:      "SPX" (32 bytes)
generation_time: Unix timestamp
coord_system:    COORD_RAW | COORD_LOG_SQRT | COORD_LOG_VARIANCE
memory_layout:   LAYOUT_M_OUTER | LAYOUT_M_INNER | LAYOUT_BLOCKED
has_gammas:      1
has_thetas:      1
has_rhos:        1
padding:         117 bytes reserved
```

### Data Sections (in order)
1. Moneyness grid: 50 × 8 = 400 bytes
2. Maturity grid: 30 × 8 = 240 bytes
3. Volatility grid: 20 × 8 = 160 bytes
4. Rate grid: 10 × 8 = 80 bytes
5. Prices: 300K × 8 = 2.4 MB
6. Vegas: 300K × 8 = 2.4 MB (if present)
7. Gammas: 300K × 8 = 2.4 MB (if present)
8. Thetas: 300K × 8 = 2.4 MB (if present)
9. Rhos: 300K × 8 = 2.4 MB (if present)

**Total file size with all Greeks: ~12.1 MB**

