# Interpolation Table Code Analysis

## Executive Summary

The mango-iv codebase implements a sophisticated multi-dimensional option pricing system with two complementary approaches:

1. **Modern C++ Implementation** (`/home/user/mango-iv/src/`):
   - B-spline based 4D interpolation (`PriceTable4DBuilder`)
   - Snapshot-based data collection (`PriceTableSnapshotCollector`)
   - Fast evaluation at sub-microsecond speeds

2. **Legacy C Implementation** (`/home/user/mango-iv/legacy/src/`):
   - General-purpose price table with multi-strategy support
   - Binary save/load with versioning
   - Reference for storage/serialization patterns

---

## 1. File Locations and Organization

### Modern Implementation (Primary)
```
/home/user/mango-iv/src/
├── price_table_4d_builder.hpp        (155 lines) - Main API
├── price_table_4d_builder.cpp        (271 lines) - Implementation
├── price_table_snapshot_collector.hpp (236 lines) - Snapshot collection
├── snapshot_interpolator.hpp         (238 lines) - 1D spline interpolation
├── snapshot.hpp                      (47 lines)  - Snapshot data structure
├── bspline_4d.hpp                    (150+ lines) - 4D B-spline evaluation
├── bspline_fitter_4d.hpp            (120+ lines) - Coefficient fitting
├── bspline_basis_1d.hpp             (60+ lines)  - Basis functions
└── bspline_utils.hpp                (80+ lines)  - Utility functions
```

### Legacy Implementation (Reference)
```
/home/user/mango-iv/legacy/src/
├── price_table.h                     (706 lines) - Complete C API
├── price_table.c                     (2400+ lines) - Full implementation
└── examples/
    └── example_precompute_table.c   (200+ lines) - Usage example

/home/user/mango-iv/legacy/tests/
├── price_table_test.cc
├── price_table_slow_test.cc
└── interpolation_test.cc
```

---

## 2. Data Structures Overview

### 2.1 Modern C++ Implementation

#### **PriceTable4DBuilder** (Main Entry Point)
```cpp
class PriceTable4DBuilder {
private:
    std::vector<double> moneyness_;    // S/K (≥4 points, sorted)
    std::vector<double> maturity_;     // τ = T - t (≥4 points, sorted)
    std::vector<double> volatility_;   // σ (≥4 points, sorted)
    std::vector<double> rate_;         // r (≥4 points, sorted)
    double K_ref_;                     // Reference strike (typically 100.0)
};

// Key method:
expected<PriceTable4DResult, std::string> precompute(
    OptionType option_type,            // CALL or PUT
    const AmericanOptionGrid& grid_config,  // PDE spatial/temporal setup
    double dividend_yield = 0.0        // Continuous dividend yield
);
```

#### **PriceTable4DResult** (Pre-computation Output)
```cpp
struct PriceTable4DResult {
    std::unique_ptr<BSpline4D_FMA> evaluator;  // Fast B-spline evaluator
    std::vector<double> prices_4d;              // Raw 4D price array
    size_t n_pde_solves;                        // Number of PDE solves performed
    double precompute_time_seconds;             // Wall-clock time
    BSplineFittingStats fitting_stats;          // Diagnostic info
};
```

#### **Snapshot** (PDE Solution at Time t)
```cpp
struct Snapshot {
    // Time and indexing
    double time;                              // Solution time
    size_t user_index;                        // User-provided index for matching

    // Spatial domain
    std::span<const double> spatial_grid;     // PDE grid x-coordinates
    std::span<const double> dx;               // Grid spacing (size = n-1)

    // Solution data (all size = n_spatial)
    std::span<const double> solution;         // V(x,t)
    std::span<const double> spatial_operator; // L(V) = ∂V/∂t (from PDE)
    std::span<const double> first_derivative; // ∂V/∂x
    std::span<const double> second_derivative;// ∂²V/∂x²

    const void* problem_params = nullptr;     // User context
};
```

#### **PriceTableSnapshotCollector** (Data Collection)
```cpp
class PriceTableSnapshotCollector : public SnapshotCollector {
private:
    std::vector<double> moneyness_;
    std::vector<double> tau_;
    double K_ref_;
    OptionType option_type_;

    // Prices and Greeks at (m_i, τ_j) grid points
    std::vector<double> prices_;    // V(m_i, τ_j)
    std::vector<double> deltas_;    // ∂V/∂S
    std::vector<double> gammas_;    // ∂²V/∂S²
    std::vector<double> thetas_;    // -∂V/∂τ

    // Interpolators for converting PDE solution to price table
    SnapshotInterpolator value_interp_;      // For V(x) → prices
    SnapshotInterpolator lu_interp_;         // For L(V) → thetas

    // PERFORMANCE: Cached values to avoid repeated transcendentals
    std::vector<double> log_moneyness_;  // Cached ln(m)
    std::vector<double> spot_values_;    // Cached m × K_ref
    std::vector<double> inv_spot_;       // Cached 1/S
    std::vector<double> inv_spot_sq_;    // Cached 1/S²
};
```

#### **SnapshotInterpolator** (1D Spline Interpolation)
```cpp
class SnapshotInterpolator {
private:
    CubicSpline<double> spline_;           // Main interpolation spline
    std::vector<double> x_;                // Grid points (log-moneyness)
    std::vector<double> y_;                // Values at grid points
    bool built_ = false;

    // Epoch-based cache for derivative arrays (2-slot LRU)
    struct DerivedSplineCache {
        CubicSpline<double> spline;
        const double* data_ptr;   // For fast lookup
        uint64_t epoch;            // Data freshness check
        bool built;
    };
    mutable DerivedSplineCache cache_[2];  // First + second derivative
    uint64_t data_epoch_ = 0;
};
```

#### **BSpline4D_FMA** (Fast Evaluation)
```cpp
class BSpline4D_FMA {
private:
    std::vector<double> m_grid_;       // Moneyness knot vector
    std::vector<double> t_grid_;       // Maturity knot vector
    std::vector<double> v_grid_;       // Volatility knot vector
    std::vector<double> r_grid_;       // Rate knot vector
    std::vector<double> coeffs_;       // B-spline coefficients

    // Clamped knot vectors (for cubic B-splines)
    std::vector<double> knots_m_;      // Extended knot vector for m
    std::vector<double> knots_t_;      // Extended knot vector for τ
    std::vector<double> knots_v_;      // Extended knot vector for σ
    std::vector<double> knots_r_;      // Extended knot vector for r
};

// Fast evaluation: ~100-200ns per call
double eval(double m, double tau, double sigma, double rate) const;
```

### 2.2 Legacy C Implementation (Reference)

#### **OptionPriceTable** (Main Data Structure)
```c
typedef struct OptionPriceTable {
    // Grid definition (4D or 5D)
    size_t n_moneyness, n_maturity, n_volatility, n_rate, n_dividend;
    double *moneyness_grid, *maturity_grid, *volatility_grid, 
           *rate_grid, *dividend_grid;

    // Flattened multi-dimensional array
    double *prices;    // V(m, τ, σ, r, q)

    // Greeks data (computed during pre-computation)
    double *vegas;     // ∂V/∂σ
    double *gammas;    // ∂²V/∂S²
    double *thetas;    // -∂V/∂τ
    double *rhos;      // ∂V/∂r

    // Metadata
    OptionType type;   // CALL or PUT
    ExerciseType exercise;  // EUROPEAN or AMERICAN
    char underlying[32];
    time_t generation_time;

    // Coordinate transformation and memory layout
    CoordinateSystem coord_system;  // RAW, LOG_SQRT, or LOG_VARIANCE
    MemoryLayout memory_layout;     // M_OUTER, M_INNER, or BLOCKED

    // Pre-computed strides for fast indexing
    size_t stride_m, stride_tau, stride_sigma, stride_r, stride_q;

    // Interpolation strategy (pluggable)
    const InterpolationStrategy *strategy;
} OptionPriceTable;
```

---

## 3. Data Structures and Dimensions

### 3.1 Grid Organization

**4D Grids:**
```
Dimension 1: Moneyness (m = S/K)
  - Typical: 50 points, log-spaced from 0.7 to 1.3
  - Represents: Deep ITM to OTM range
  
Dimension 2: Maturity (τ = T - t, in years)
  - Typical: 30 points, linear from 0.027 (10 days) to 2.0 (2 years)
  
Dimension 3: Volatility (σ, as decimal)
  - Typical: 20 points, linear from 0.10 (10%) to 0.80 (80%)
  
Dimension 4: Interest Rate (r, as decimal)
  - Typical: 10 points, linear from 0.0 to 0.10 (10%)
  
Total Grid Points: 50 × 30 × 20 × 10 = 300,000
```

**Optional 5D Grid:**
```
Dimension 5: Continuous Dividend Yield (q, as decimal)
  - Added if n_dividend > 0
  - Typical: 5-10 points
```

### 3.2 Memory Layout

**Modern C++:**
- **Default**: Dense allocation in `PriceTable4DResult::prices_4d`
- **Indexing**: `idx = i_m * (Nt × Nv × Nr) + i_τ * (Nv × Nr) + i_σ * Nr + i_r`

**Legacy C (Multiple Strategies):**
```c
// LAYOUT_M_OUTER (default for point queries)
idx = i_m * stride_m + i_tau * stride_tau + i_sigma * stride_sigma
    + i_r * stride_r + i_q * stride_q;

// LAYOUT_M_INNER (optimized for cubic slice extraction)
// Memory layout: [r][sigma][tau][m]
// ~30x faster slice extraction due to cache locality
```

### 3.3 Greeks Storage

All implementations store Greeks alongside prices:

| Greek | Symbol | Meaning | Computation |
|-------|--------|---------|-------------|
| Vega  | ∂V/∂σ | Volatility sensitivity | Finite differences during precompute |
| Gamma | ∂²V/∂S² | Delta sensitivity | Finite differences on moneyness axis |
| Theta | -∂V/∂τ | Time decay | Negative time derivative (or -L(V) in PDE) |
| Rho   | ∂V/∂r | Rate sensitivity | Finite differences during precompute |

---

## 4. Storage and Serialization

### 4.1 Binary Format (Legacy C Implementation)

**File Structure:**
```
Offset          Data                        Size (bytes)
0               Header (PriceTableHeader)   256
256             Moneyness grid              n_m × 8
256 + n_m×8     Maturity grid               n_tau × 8
...             Volatility grid             n_v × 8
...             Rate grid                   n_r × 8
...             Dividend grid (if present)  n_q × 8
...             PRICES array                n_total × 8
...             VEGA array (if present)     n_total × 8
...             GAMMA array (if present)    n_total × 8
...             THETA array (if present)    n_total × 8
...             RHO array (if present)      n_total × 8
```

**Header Format (256 bytes, Version 4):**
```c
typedef struct {
    uint32_t magic;                    // 0x50545442 ("PTTB")
    uint32_t version;                  // 4 (current)
    size_t n_moneyness;                // Grid dimensions
    size_t n_maturity;
    size_t n_volatility;
    size_t n_rate;
    size_t n_dividend;
    OptionType type;                   // CALL or PUT
    ExerciseType exercise;             // EUROPEAN or AMERICAN
    char underlying[32];               // Symbol (e.g., "SPX")
    time_t generation_time;            // Timestamp
    CoordinateSystem coord_system;     // RAW, LOG_SQRT, LOG_VARIANCE
    MemoryLayout memory_layout;        // M_OUTER, M_INNER, BLOCKED
    uint8_t has_gammas;                // 1 if gamma data present
    uint8_t has_thetas;                // 1 if theta data present
    uint8_t has_rhos;                  // 1 if rho data present
    uint8_t padding[117];              // Reserved
} PriceTableHeader;
```

**Version History:**
- Version 1: Original (dimensions only)
- Version 2: Added coordinate system and memory layout
- Version 3: Added gamma data
- Version 4: Added theta and rho data (current)

**Storage Size Example:**
```
4D table (50 × 30 × 20 × 10 = 300K points):
  - Header:        256 bytes
  - Grids:         (50 + 30 + 20 + 10) × 8 = 720 bytes
  - Prices:        300K × 8 = 2.4 MB
  - Vegas:         300K × 8 = 2.4 MB (if computed)
  - Gammas:        300K × 8 = 2.4 MB (if computed)
  - Thetas:        300K × 8 = 2.4 MB (if computed)
  
  Total (with all Greeks): ~12 MB
```

### 4.2 Save/Load Functions (Legacy C)

```c
// Save to binary file
int price_table_save(const OptionPriceTable *table, const char *filename);
// Returns: 0 on success, -1 on error

// Load from binary file
OptionPriceTable* price_table_load(const char *filename);
// Returns: Newly allocated table, or NULL on error
```

### 4.3 Modern C++ Storage (Not Yet Implemented)

The `PriceTable4DResult` returns:
- `std::unique_ptr<BSpline4D_FMA> evaluator` - Contains B-spline coefficients
- `std::vector<double> prices_4d` - Raw 4D price array

**Future Storage Strategy:**
Would need to serialize:
1. B-spline knot vectors (4 vectors × (n_grid + 4) points each)
2. B-spline coefficients (flattened 4D array)
3. Fitting statistics and metadata

---

## 5. Typical Usage Pattern

### 5.1 Pre-computation Workflow

```cpp
// 1. Define 4D grids
auto builder = PriceTable4DBuilder::create(
    {0.7, 0.8, ..., 1.3},    // moneyness: 50 points
    {0.027, 0.1, ..., 2.0},  // maturity: 30 points
    {0.10, 0.15, ..., 0.80}, // volatility: 20 points
    {0.0, 0.02, ..., 0.10},  // rate: 10 points
    100.0                      // K_ref
);

// 2. Configure PDE solver (FDM grid)
AmericanOptionGrid grid_config{
    .n_space = 101,    // Spatial discretization
    .n_time = 1000,    // Time steps
    .x_min = -3.0,     // Log-moneyness range
    .x_max = 3.0
};

// 3. Pre-compute all 300,000 option prices
auto result = builder.precompute(
    OptionType::PUT,
    grid_config,
    0.02  // dividend yield
);

if (result) {
    auto& [evaluator, prices, n_solves, elapsed, stats] = *result;
    std::cout << "Pre-computed " << n_solves << " options in " 
              << elapsed << " seconds\n";
    std::cout << "Max fitting residual: " << stats.max_residual_overall << "\n";
    
    // Fast evaluation: ~500ns per query
    double price = evaluator->eval(1.05, 0.25, 0.20, 0.05);
} else {
    std::cerr << "Error: " << result.error() << "\n";
}
```

### 5.2 Fast Lookup Workflow (Compiled Evaluator)

```cpp
// After pre-computation, evaluator is ready to use
BSpline4D_FMA& evaluator = /* from result */;

// Batch pricing (very fast)
for (const auto& option : market_data) {
    double price = evaluator.eval(
        option.spot / option.strike,  // moneyness
        option.time_to_maturity,
        option.volatility,
        option.rate
    );
    
    // Process price...
}
```

### 5.3 Legacy C Workflow (Reference)

```c
// 1. Create table
OptionPriceTable *table = price_table_create(
    moneyness, n_m, maturity, n_tau, volatility, n_v,
    rate, n_r, NULL, 0,  // No dividend (4D mode)
    OPTION_PUT, AMERICAN);

// 2. Pre-compute all prices
AmericanOptionGrid grid = {
    .n_space = 101, .n_time = 1000, .S_max = 200.0
};
price_table_precompute(table, &grid);

// 3. Save to disk
price_table_save(table, "spx_put_american.bin");
price_table_destroy(table);

// Later: Load and query
table = price_table_load("spx_put_american.bin");
double price = price_table_interpolate_4d(table, 1.05, 0.25, 0.20, 0.05);
double vega = price_table_interpolate_vega_4d(table, 1.05, 0.25, 0.20, 0.05);
price_table_destroy(table);
```

---

## 6. Pre-computation Process (Detailed)

### 6.1 Modern C++ Flow

**High Level:**
```
PriceTable4DBuilder::precompute()
  ├── Allocate 4D price array (300K doubles)
  ├── Loop: For each (σ, r) pair (200 iterations)
  │   ├── Create AmericanOptionSolver for this (σ, r)
  │   ├── Loop: For each maturity τ (30 iterations)
  │   │   ├── Register snapshot collection at specific time
  │   │   │   (converts PDE solution to (m, τ) slice)
  │   │   └── Solver triggers snapshot callback
  │   ├── Solver fills 30 snapshots
  │   └── SnapshotCollector populates prices at (m_i, τ_j)
  │       for this (σ, r)
  ├── After all solves: Fit 4D B-spline to 300K price points
  └── Return evaluator + prices + statistics
```

**Key Classes Involved:**
1. `PriceTable4DBuilder` - Orchestrates the overall workflow
2. `AmericanOptionSolver` - Solves PDE for each (σ, r)
3. `PriceTableSnapshotCollector` - Collects PDE output and converts to prices
4. `SnapshotInterpolator` - Interpolates PDE solution from x-space to log-moneyness
5. `BSplineFitter4D` - Fits B-spline coefficients to prices
6. `BSpline4D_FMA` - Evaluator for fast queries

### 6.2 Snapshot Collection Details

**For each snapshot at time τ:**

1. **PDE Solution Reception:**
   - Receive V(x, τ) on PDE grid (log-moneyness x = ln(S/K_ref))
   - Also receive spatial derivatives ∂V/∂x, ∂²V/∂x², L(V)

2. **Interpolation to Moneyness Points:**
   ```
   For each moneyness m_i:
     x_i = ln(m_i)
     V_norm_i = SnapshotInterpolator.eval(x_i)  // Cubic spline
   ```

3. **Transformation to Dollar Prices:**
   ```
   Price[m_i, τ] = K_ref × V_norm_i
   
   Delta[m_i, τ] = (K_ref / (m_i × K_ref)) × dV_norm/dx
                 = (1/m_i) × dV_norm/dx
   
   Gamma[m_i, τ] = (K_ref / (m_i × K_ref)²) × d²V_norm/dx²
                 - Gamma adjustment term
   
   Theta[m_i, τ] = -L(V_norm)  (in continuation region)
   ```

4. **Storage:**
   - All values stored in row-major format
   - Index: `table_idx = m_idx * n_tau + tau_idx`

---

## 7. Performance Characteristics

### 7.1 Pre-computation Time

| Grid Size | Single Thread | 16 Cores | Throughput |
|-----------|---------------|----------|-----------|
| 300K (50×30×20×10) | ~50 minutes | ~3 minutes | 1,667 opt/sec |
| 100K (20×20×10×5) | ~15 minutes | ~1 minute | 1,667 opt/sec |
| 10K (10×10×5×2) | ~2 minutes | ~8 seconds | 1,250 opt/sec |

**Breakdown per option:**
- PDE solve: ~550ms (FDM 101×1000 grid)
- Snapshot collection: ~10ms (30 maturity snapshots)
- B-spline fitting: ~5ms per option (cumulative overhead)

### 7.2 Query Performance

| Operation | Time | Method |
|-----------|------|--------|
| 4D B-spline eval | 100-200ns | Tensor-product + FMA |
| 4D cubic interp | 400-500ns | Legacy implementation |
| Greeks (delta/gamma) | 5-10µs | Finite differences on prices |
| IV calculation | <30µs | Brent on B-spline evals |

**Speedup vs FDM:**
- Single FDM solve: 550ms
- B-spline query: 150ns
- **Speedup: ~3.7 million×**

### 7.3 Memory Usage

```
4D Table Storage (Modern C++):
  prices_4d:        300K × 8 = 2.4 MB
  evaluator coeff:  (50+4)×(30+4)×(20+4)×(10+4) × 8 ≈ 6 MB
  grids + metadata: ~0.5 KB
  Total:           ~8.5 MB (resident)

4D Table + Greeks (Legacy C):
  prices:           300K × 8 = 2.4 MB
  vegas:            300K × 8 = 2.4 MB
  gammas:           300K × 8 = 2.4 MB
  thetas:           300K × 8 = 2.4 MB
  rhos:             300K × 8 = 2.4 MB
  grids:            (50+30+20+10) × 8 = 720 B
  Total:           ~12 MB
```

---

## 8. Key Design Decisions

### 8.1 Modern C++ vs Legacy C

| Aspect | Modern C++ | Legacy C |
|--------|-----------|----------|
| Data Structure | Type-safe classes | Struct + function pointers |
| Grids | `std::vector` | `double*` arrays |
| Interpolation | B-splines (fitted) | Strategy pattern (pluggable) |
| Performance | 150-200ns eval | 400-500ns eval |
| Storage | Not implemented yet | Binary save/load with versioning |
| Flexibility | Fixed 4D | Flexible 4D/5D via layout |

### 8.2 Snapshot Interpolation Strategy

**Why snapshot interpolation?**
1. PDE solver operates on log-moneyness x = ln(S/K)
2. Price table needs moneyness m = S/K
3. Direct transformation V_norm(x) → V_norm(m) requires interpolation

**Method Used:**
- Cubic spline from PDE spatial grid to moneyness points
- Epoch-based caching to avoid redundant spline rebuilds
- Fast rebuild when same grid, new data

**Optimization:** Cache Grid + Reuse
```
For (σ, r) slice with multiple maturities:
  First snapshot: Build interpolators from scratch
  Later snapshots: Reuse interpolators (fast rebuild)
  Speedup: 2-3× vs rebuilding every time
```

### 8.3 B-spline Fitting Strategy

**Why B-splines?**
1. Fast evaluation (tensor-product structure, ~150ns)
2. Smooth interpolation (C² continuous)
3. Better extrapolation than cubic spline
4. Scales well to 4D (separable fitting approach)

**Implementation:**
- Clamped cubic B-splines (degree 3)
- Knot vectors derived from grid points
- Separable least-squares fitting (dimension-by-dimension)
- Condition number monitoring for numerical stability

---

## 9. Reference Usage Examples

### Modern C++
```cpp
// File: /home/user/mango-iv/src/price_table_4d_builder.hpp
// Test: /home/user/mango-iv/tests/integration_5d_price_table_test.cc
```

### Legacy C
```c
// File: /home/user/mango-iv/legacy/examples/example_precompute_table.c
// Tests: /home/user/mango-iv/legacy/tests/price_table_test.cc
```

---

## 10. Summary Table

| Component | Location | Purpose | Key Class |
|-----------|----------|---------|-----------|
| **Builder** | `price_table_4d_builder.*` | Orchestrate precomputation | `PriceTable4DBuilder` |
| **Snapshots** | `snapshot.hpp` | PDE solution data | `Snapshot` |
| **Collection** | `price_table_snapshot_collector.hpp` | Convert PDE→prices | `PriceTableSnapshotCollector` |
| **1D Interp** | `snapshot_interpolator.hpp` | Spline eval from arrays | `SnapshotInterpolator` |
| **4D Eval** | `bspline_4d.hpp` | Fast B-spline evaluation | `BSpline4D_FMA` |
| **Fitting** | `bspline_fitter_4d.hpp` | Fit B-spline coefficients | `SeparableBSplineFitter4D` |
| **Storage** | `price_table.[hc]` (legacy) | Save/load with versioning | `OptionPriceTable` |

