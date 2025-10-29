# Interpolation-Based Option Pricing Engine: Design Document

## Executive Summary

This document proposes an **interpolation-based option pricing engine** to complement the existing FDM (Finite Difference Method) solver. The goal is to achieve **sub-microsecond** pricing and IV calculations during trading sessions by pre-computing results during downtime.

**Key Performance Targets:**
- **Pre-computation**: Minutes to hours (acceptable during downtime)
- **Real-time lookup**: <1µs per query (100-1000x faster than current 21.7ms)
- **Accuracy**: <0.5% relative error for typical market conditions
- **Memory**: <100MB for typical IV surface (manageable)

**Current Performance (FDM):**
- American option: 21.7ms per option
- IV calculation (Black-Scholes): <1µs (already fast)
- Batch processing: 60x+ speedup with OpenMP

**Problem Statement:**
During trading sessions, we need to:
1. Price thousands of options per second
2. Calculate implied volatilities from market prices
3. Compute Greeks for risk management
4. Handle changing spot prices, interest rates

FDM is too slow for this (21.7ms × 1000 options = 21.7 seconds). Interpolation can reduce this to <1ms total.

**Design Philosophy:**
- **Strategy Pattern**: Uses dependency injection for interpolation algorithms (see `notes/INTERPOLATION_STRATEGY_DESIGN.md`)
- **Runtime Selection**: Switch between linear, cubic, or custom algorithms without recompilation
- **Extensibility**: Users can implement custom interpolation strategies
- **Consistency**: Follows existing callback-based architecture from `PDESolver`

---

## Design Alternatives and Trade-offs

### Alternative 1: Pre-compute Option Prices

**Concept:** Build a multi-dimensional grid of option prices for various parameters.

**Dimensions:**
- Moneyness: `m = S/K` (spot/strike ratio) or `ln(S/K)`
- Time to maturity: `τ = T - t`
- Volatility: `σ`
- Interest rate: `r` (optional, usually stable)
- Dividend yield: `q` or discrete dividend schedule

**Grid Example (4D):**
- Moneyness: 50 points in [0.5, 1.5] (log-spaced)
- Maturity: 30 points in [1 day, 2 years]
- Volatility: 20 points in [0.05, 1.0]
- Rate: 10 points in [0.0, 0.1]

**Storage:** 50 × 30 × 20 × 10 = 300,000 doubles = 2.4MB per option type (call/put)

**Pros:**
- ✅ Direct price lookup (no IV inversion needed)
- ✅ Can pre-compute Greeks via finite differences
- ✅ Handles American options naturally
- ✅ Small memory footprint

**Cons:**
- ❌ Requires market IV to look up price (chicken-and-egg for IV calculation)
- ❌ 4D or 5D interpolation is slower than 2D/3D
- ❌ Separate table needed for each dividend schedule

**Best For:** Given market IV, need fast option prices

---

### Alternative 2: Pre-compute Implied Volatility Surfaces

**Concept:** Build a 2D/3D grid of IV values based on observed market patterns.

**Dimensions:**
- Moneyness: `m = S/K` or `ln(S/K)`
- Time to maturity: `τ`
- (Optional) Normalized price: `p / S` for American options

**Grid Example (2D):**
- Moneyness: 50 points in [0.5, 1.5]
- Maturity: 30 points in [1 day, 2 years]

**Storage:** 50 × 30 = 1,500 doubles = 12KB per surface (tiny!)

**Pros:**
- ✅ Extremely small memory footprint
- ✅ Fast 2D interpolation (<100ns)
- ✅ Natural representation of market IV surface
- ✅ Easy to update with market data

**Cons:**
- ❌ Doesn't directly give option prices (need BS/FDM after IV lookup)
- ❌ American options require additional price calculation
- ❌ Greeks need analytical formulas or additional computation

**Best For:** Market data fitting, volatility surface visualization, IV lookup

---

### Alternative 3: Hybrid Approach (RECOMMENDED)

**Concept:** Combine both approaches for maximum flexibility.

**Two-tier system:**

1. **Tier 1: IV Surface (2D/3D)**
   - Pre-compute representative IV surface from historical data
   - Fast lookup for market IV interpolation
   - Use for: IV curve visualization, initial guesses

2. **Tier 2: Option Price Tables (4D/5D)**
   - Pre-compute option prices for grid of (m, τ, σ, r, q)
   - Fast lookup for pricing given IV
   - Use for: Fast pricing, Greeks calculation

**Workflow:**

```
A) Given market price, find IV:
   market_price → [2D IV surface lookup for initial guess]
   → [Brent's method with FDM] → IV

B) Given IV, find option price:
   IV + (S, K, τ, r, q) → [4D price table lookup] → price

C) Build IV surface from market data:
   market_prices[] → [Brent's + FDM for each] → IV_surface[]
   → [fit 2D spline] → continuous IV surface
```

**Pros:**
- ✅ Best of both worlds
- ✅ Fast IV lookup AND fast pricing
- ✅ Modular design (can use either tier independently)
- ✅ Natural separation of concerns

**Cons:**
- ❌ More complex implementation
- ❌ Two separate data structures to manage

**Memory Estimate:**
- Tier 1 (IV surface): 12KB per underlying
- Tier 2 (price tables): 2.4MB per option type × 2 (call/put) = 4.8MB per underlying
- Total: ~5MB per underlying (very manageable)

---

## Recommended Architecture: Hybrid Design

### Component 1: IV Surface Manager

**Purpose:** Manage 2D/3D implied volatility surfaces

**Data Structure:**
```c
typedef struct {
    // Grid definition
    size_t n_moneyness;        // Number of moneyness points
    size_t n_maturity;          // Number of maturity points
    double *moneyness_grid;     // Moneyness values (m = S/K)
    double *maturity_grid;      // Maturity values (τ = T - t)

    // IV data (row-major: moneyness varies fastest)
    double *iv_surface;         // n_moneyness × n_maturity

    // Interpolation method
    enum {
        IV_INTERP_LINEAR,
        IV_INTERP_CUBIC,
        IV_INTERP_NATURAL_CUBIC
    } interp_method;

    // Metadata
    char underlying[32];        // "SPX", "NDX", etc.
    time_t last_update;         // Timestamp
} IVSurface;
```

**API:**
```c
// Create/destroy
IVSurface* iv_surface_create(const double *moneyness, size_t n_m,
                              const double *maturity, size_t n_tau);
void iv_surface_destroy(IVSurface *surface);

// Set/get IV values
void iv_surface_set(IVSurface *surface, const double *iv_data);
double iv_surface_get(const IVSurface *surface, size_t i_m, size_t i_tau);

// Interpolation (main query interface)
double iv_surface_interpolate(const IVSurface *surface,
                               double moneyness, double maturity);

// Build from market data
int iv_surface_fit_from_market(IVSurface *surface,
                                const MarketData *market_data,
                                PDESolver *pde_solver);  // For IV calculation

// I/O
int iv_surface_save(const IVSurface *surface, const char *filename);
IVSurface* iv_surface_load(const char *filename);
```

**Grid Recommendations:**
- Moneyness: Log-spaced in [0.5, 1.5] with 50-100 points
  - More points near ATM (m = 1.0) for better accuracy
- Maturity: Log-spaced in [1 day, 2 years] with 20-40 points
  - Denser grid for short maturities (higher gamma)

---

### Component 2: Option Price Table

**Purpose:** Multi-dimensional price lookup table

**Data Structure:**
```c
typedef struct {
    // Grid definition (4D or 5D)
    size_t n_moneyness;         // S/K dimension
    size_t n_maturity;          // τ dimension
    size_t n_volatility;        // σ dimension
    size_t n_rate;              // r dimension (optional)
    size_t n_dividend;          // q dimension (optional)

    double *moneyness_grid;
    double *maturity_grid;
    double *volatility_grid;
    double *rate_grid;
    double *dividend_grid;

    // Option prices (multi-dimensional array)
    double *prices;             // Flattened array

    // Metadata
    OptionType type;            // CALL or PUT
    ExerciseType exercise;      // EUROPEAN or AMERICAN
    char underlying[32];
    time_t generation_time;

    // Pre-computed for fast indexing
    size_t stride_m, stride_tau, stride_sigma, stride_r, stride_q;
} OptionPriceTable;
```

**API:**
```c
// Create/destroy
OptionPriceTable* price_table_create(
    const double *moneyness, size_t n_m,
    const double *maturity, size_t n_tau,
    const double *volatility, size_t n_sigma,
    const double *rate, size_t n_r,
    const double *dividend, size_t n_q,
    OptionType type, ExerciseType exercise);

void price_table_destroy(OptionPriceTable *table);

// Pre-computation (batch processing with OpenMP)
int price_table_precompute(OptionPriceTable *table,
                            PDESolver *pde_solver_template);

// Fast lookup (main query interface)
double price_table_interpolate(const OptionPriceTable *table,
                                double moneyness,
                                double maturity,
                                double volatility,
                                double rate,
                                double dividend);

// Greeks via finite differences (on pre-computed grid)
OptionGreeks price_table_greeks(const OptionPriceTable *table,
                                 double moneyness, double maturity,
                                 double volatility, double rate,
                                 double dividend);

// I/O
int price_table_save(const OptionPriceTable *table, const char *filename);
OptionPriceTable* price_table_load(const char *filename);
```

**Storage Format:**
```
File format (binary):
- Header: magic number, version, dimensions, metadata
- Grid arrays: moneyness[], maturity[], volatility[], rate[], dividend[]
- Price data: prices[] (flattened multi-dimensional array)
- Checksum: CRC32 or SHA256 for integrity
```

**Indexing Formula (row-major):**
```c
// 5D indexing: price[i_m, i_tau, i_sigma, i_r, i_q]
size_t idx = i_m * stride_m + i_tau * stride_tau + i_sigma * stride_sigma
           + i_r * stride_r + i_q * stride_q;
double price = table->prices[idx];
```

---

### Component 3: Multi-dimensional Interpolation

**Challenge:** Need efficient interpolation in 4D/5D space

**Architecture:** Uses **Strategy Pattern** with dependency injection to allow runtime algorithm selection. See `notes/INTERPOLATION_STRATEGY_DESIGN.md` for complete design.

**Available Strategies:**

#### 3.1. Separable Multi-linear Interpolation (RECOMMENDED)

**Method:** Perform 1D linear interpolation along each dimension sequentially

**Algorithm (4D example):**
```
1. Find bracketing indices for each dimension:
   - moneyness: i_m, i_m+1 where m[i_m] <= query_m < m[i_m+1]
   - maturity: i_tau, i_tau+1
   - volatility: i_sigma, i_sigma+1
   - rate: i_r, i_r+1

2. Perform 16 lookups at hypercube corners (2^4 = 16 for 4D)

3. Interpolate recursively:
   - 8 interpolations along moneyness dimension
   - 4 interpolations along maturity dimension
   - 2 interpolations along volatility dimension
   - 1 interpolation along rate dimension
   Total: 15 interpolations
```

**Complexity:**
- Time: O(2^d) lookups + O(2^d - 1) linear interpolations, where d = dimensions
- Memory: O(1) (no temporary arrays)
- For d=4: 16 lookups + 15 interpolations ≈ 31 operations

**Pros:**
- ✅ Simple to implement
- ✅ Fast (sub-microsecond)
- ✅ Guaranteed to stay within bounds
- ✅ Continuous (C0) everywhere

**Cons:**
- ❌ Not smooth (only C0 continuous)
- ❌ Can't extrapolate (must clamp to grid bounds)
- ❌ Second derivatives (gamma) are approximately zero within cells - cannot accurately capture curvature. For accurate gamma calculations, cubic spline interpolation would be needed.

#### 3.2. Tensor-Product Cubic Splines

**Method:** Use natural cubic splines along each dimension

**Algorithm:**
```
1. Build 1D cubic spline coefficients for each dimension (pre-computed)
2. Evaluate spline along each dimension sequentially
3. Requires O(2^d) cubic evaluations instead of linear
```

**Complexity:**
- Pre-computation: O(n_total) for all spline coefficients
- Query: O(2^d) cubic spline evaluations
- Memory: O(n_total) for coefficients

**Pros:**
- ✅ Smooth interpolation (C2 continuous)
- ✅ Better accuracy than linear
- ✅ Already have cubic spline implementation

**Cons:**
- ❌ 3-5x slower than linear (still <1µs though)
- ❌ More memory for coefficients
- ❌ Can overshoot (need clamping for option prices)

**Recommendation:** Start with multi-linear, add cubic as optional upgrade

---

### Component 4: Integration with Existing Code

**New Files:**
```
src/
├── interp_strategy.h     # Interpolation strategy interface (DI)
├── iv_surface.h          # IV surface API
├── iv_surface.c          # IV surface implementation
├── price_table.h         # Option price table API
├── price_table.c         # Price table implementation
├── interp_multilinear.c  # Multi-linear strategy implementation
└── interp_cubic.c        # Cubic spline strategy implementation

examples/
├── example_iv_surface.c       # Build IV surface from market data
├── example_price_table.c      # Pre-compute and query price table
└── example_fast_greeks.c      # Greeks calculation via interpolation

tests/
├── interp_strategy_test.cc    # Strategy interface tests
├── iv_surface_test.cc         # IV surface tests
├── price_table_test.cc        # Price table tests
├── interp_accuracy_test.cc    # Interpolation accuracy analysis
└── interp_benchmark.cc        # Strategy performance comparison
```

**Integration Points:**
1. Use existing `PDESolver` for pre-computation (batch mode)
2. Use existing `american_option_price()` for individual price calculations
3. Use existing `CubicSpline` for 1D spline interpolation (if using cubic)
4. Leverage existing OpenMP batch API for parallel pre-computation

---

## Implementation Phases

### Phase 1: Core Infrastructure (Week 1-2)

**Goals:**
- Implement `IVSurface` data structure and API
- Implement `OptionPriceTable` data structure and API
- Implement multi-linear interpolation (4D/5D)
- Add comprehensive unit tests

**Deliverables:**
- `src/iv_surface.{h,c}` - complete and tested
- `src/price_table.{h,c}` - complete and tested
- `src/multilinear_interp.{h,c}` - complete and tested
- `tests/iv_surface_test.cc` - >90% coverage
- `tests/price_table_test.cc` - >90% coverage
- `tests/interp_test.cc` - accuracy verification

**Success Criteria:**
- All tests pass
- Interpolation accuracy <0.1% for on-grid points
- Query time <100ns for IV surface, <500ns for price table

---

### Phase 2: Pre-computation Engine (Week 3-4)

**Goals:**
- Implement batch pre-computation using existing FDM solver
- Optimize for parallel execution (OpenMP)
- Add progress tracking (USDT probes)
- Implement save/load functionality

**Deliverables:**
- `price_table_precompute()` function with OpenMP parallelization
- `iv_surface_fit_from_market()` for building IV surfaces
- Binary file format for storing/loading tables
- `examples/example_precompute_table.c`

**Success Criteria:**
- Pre-compute 100,000 option prices in <5 minutes (50ms per option × parallelism)
- Saved tables can be loaded in <100ms
- Memory-mapped I/O for large tables (optional)

**Pre-computation Workflow:**
```bash
# Pre-compute price table for SPX American puts
./bazel-bin/examples/example_precompute_table \
    --underlying SPX \
    --type put \
    --exercise american \
    --moneyness 0.7:1.3:50 \
    --maturity 0.027:2.0:30 \     # 10 days to 2 years
    --volatility 0.1:0.8:20 \
    --rate 0.0:0.1:5 \
    --dividend 0.0:0.05:3 \
    --output spx_put_american.bin

# Takes ~5 minutes to compute 50×30×20×5×3 = 450,000 prices
# Parallelizes across 450,000 grid points using OpenMP
```

---

### Phase 3: High-Level APIs and Examples (Week 5)

**Goals:**
- User-friendly APIs for common use cases
- Example programs demonstrating workflows
- Documentation and benchmarks

**Deliverables:**
- `examples/example_iv_surface.c` - Build and query IV surface
- `examples/example_fast_pricer.c` - Fast pricing via interpolation
- `examples/example_fast_greeks.c` - Greeks via finite differences on table
- Benchmark comparing FDM vs interpolation speed/accuracy
- Documentation in `INTERPOLATION_ENGINE.md`

**Example Usage:**
```c
// Example 1: Fast option pricing
OptionPriceTable *table = price_table_load("spx_put_american.bin");

// Query at specific point
double price = price_table_interpolate(table,
    1.05,    // moneyness (spot/strike)
    0.25,    // maturity (3 months)
    0.20,    // volatility (20%)
    0.05,    // rate (5%)
    0.02);   // dividend yield (2%)

// Get Greeks
OptionGreeks greeks = price_table_greeks(table, 1.05, 0.25, 0.20, 0.05, 0.02);
printf("Delta: %.4f, Gamma: %.4f, Vega: %.4f\n",
       greeks.delta, greeks.gamma, greeks.vega);

price_table_destroy(table);
```

```c
// Example 2: IV surface lookup
IVSurface *surface = iv_surface_load("spx_iv_surface.bin");

// Query IV at specific point
double iv = iv_surface_interpolate(surface, 1.0, 0.5);  // ATM, 6 months

// Visualize surface
for (double m = 0.8; m <= 1.2; m += 0.05) {
    for (double tau = 0.1; tau <= 2.0; tau += 0.1) {
        double iv = iv_surface_interpolate(surface, m, tau);
        printf("%.2f\t%.2f\t%.4f\n", m, tau, iv);
    }
}

iv_surface_destroy(surface);
```

---

### Phase 4: Advanced Features (Week 6-8, Optional)

**Possible Extensions:**

1. **Adaptive Grids**
   - Dense grid near ATM and short maturities
   - Sparse grid for OTM and long maturities
   - Reduces memory while maintaining accuracy

2. **GPU Acceleration**
   - Upload price table to GPU memory
   - Batch interpolation on GPU (thousands of queries in parallel)
   - Useful for portfolio-level risk calculations

3. **Real-time Updates**
   - Incremental table updates as market data arrives
   - Background thread for continuous re-computation
   - Lock-free data structures for concurrent access

4. **Calibration Framework**
   - Fit IV surface parameters to match market prices
   - Arbitrage-free interpolation (ensures call-put parity, no calendar arbitrage)
   - Regularization to prevent overfitting

5. **Exotic Options**
   - Extend to barriers, digitals, etc.
   - May need higher-dimensional tables (add barrier levels)

---

## Performance Analysis

### Expected Query Performance

**Target: Sub-microsecond queries**

**IV Surface (2D interpolation):**
- Grid search: O(log n) binary search × 2 dimensions = ~10-20 comparisons
- 4-point linear interpolation: 4 lookups + 3 lerp operations
- **Estimated time: 50-100ns** (dominated by cache misses)

**Price Table (4D interpolation):**
- Grid search: O(log n) × 4 dimensions = ~20-40 comparisons
- 16-point multi-linear interpolation: 16 lookups + 15 lerp operations
- **Estimated time: 200-500ns** (more cache misses for 4D)

**Comparison to FDM:**
- FDM (American option): 21.7ms = 21,700,000ns
- Interpolation (4D): ~500ns
- **Speedup: ~43,000x** 🚀

**Throughput:**
- FDM: ~46 prices per second (single-threaded)
- Interpolation: ~2,000,000 prices per second (single-threaded)
- With parallelization: **>10M prices per second** (limited by memory bandwidth)

---

### Accuracy Analysis

**Expected Interpolation Error:**

**Linear Interpolation:**
- On-grid points: 0% error (exact)
- Mid-points: <0.1% error for smooth functions
- Near boundaries: up to 1% error if function curvature is high

**Factors Affecting Accuracy:**
1. Grid resolution (finer = more accurate but more memory)
2. Function smoothness (option prices are smooth except at boundaries)
3. Moneyness (higher curvature near ATM and near expiry)

**Recommended Grid Density:**
- ATM (0.9 < m < 1.1): 20-30 points for <0.5% error
- OTM/ITM: 10-15 points sufficient
- Short maturity (<1 month): 10-15 points
- Long maturity (>1 year): 5-10 points

**Validation Strategy:**
1. Compare interpolated prices to FDM prices at random test points
2. Measure mean absolute error, max error, RMS error
3. Check Greeks accuracy (finite difference from interpolated values)
4. Verify no arbitrage violations (call-put parity, monotonicity)

---

### Memory Requirements

**IV Surface (2D):**
- 50 × 30 points = 1,500 doubles = 12KB per surface
- For 100 underlyings: 1.2MB total (negligible)

**Price Table (4D American Options):**
- 50 × 30 × 20 × 10 = 300,000 doubles = 2.4MB per table
- Per underlying: call + put = 4.8MB
- For 100 underlyings: 480MB (manageable)

**Price Table (5D with dividends):**
- Add dividend dimension: 50 × 30 × 20 × 10 × 5 = 1.5M doubles = 12MB per table
- Per underlying: call + put = 24MB
- For 100 underlyings: 2.4GB (requires careful memory management)

**Memory Optimization:**
- Use float32 instead of float64 (2x smaller, still adequate precision)
- Compress tables using lossless compression (zstd, lz4)
- Memory-map files (mmap) for on-demand loading
- Hierarchical loading (load only needed maturity slices)

---

## Risk Assessment and Mitigation

### Technical Risks

**Risk 1: Interpolation Accuracy**
- **Impact**: Mispricing options, incorrect Greeks
- **Likelihood**: Medium (depends on grid resolution)
- **Mitigation**:
  - Extensive accuracy testing against FDM
  - Adaptive grids with denser points where needed
  - Validation against market data
  - User-configurable accuracy/memory trade-off

**Risk 2: Extrapolation Issues**
- **Impact**: Unbounded errors outside grid range
- **Likelihood**: High (markets move unpredictably)
- **Mitigation**:
  - Clamp queries to grid boundaries
  - Optionally fall back to FDM for out-of-range queries
  - Warning system for extrapolation
  - Regular table updates to cover current market regime

**Risk 3: Memory Overhead**
- **Impact**: System slowdown, out-of-memory errors
- **Likelihood**: Low (for 4D), Medium (for 5D+)
- **Mitigation**:
  - Use float32 for tables (2x reduction)
  - Implement lazy loading (mmap)
  - Provide memory budget configuration
  - Profile memory usage in production

**Risk 4: Stale Data**
- **Impact**: Prices don't reflect current market conditions
- **Likelihood**: High (during volatile markets)
- **Mitigation**:
  - Timestamp all tables
  - Automatic staleness detection
  - Background re-computation threads
  - Hybrid mode: interpolate + adjustment for spot/rate changes

---

### Operational Risks

**Risk 5: Pre-computation Time**
- **Impact**: Tables not ready when needed
- **Likelihood**: Medium (during market opens, after holidays)
- **Mitigation**:
  - Pre-compute overnight before trading
  - Incremental updates (only changed parameters)
  - Prioritize high-volume options
  - Fall back to FDM if table unavailable

**Risk 6: File Corruption**
- **Impact**: Load failures, incorrect prices
- **Likelihood**: Low
- **Mitigation**:
  - Checksums in file format (CRC32, SHA256)
  - Validation on load (sanity checks)
  - Backup tables with versioning
  - Atomic file writes (write to temp, then rename)

---

## Alternative Approaches Considered

### Approach A: Neural Network Approximation

**Concept:** Train a neural network to approximate option prices

**Pros:**
- Can learn complex patterns
- Handles high-dimensional spaces naturally
- Continuous everywhere (no grid artifacts)

**Cons:**
- Black box (hard to debug)
- Training complexity
- GPU needed for fast inference
- Risk of overfitting or extrapolation errors
- Not deterministic

**Verdict:** ❌ Rejected - Too complex, not interpretable, overkill for this problem

---

### Approach B: Analytical Approximations

**Concept:** Use analytical formulas (e.g., Barone-Adesi-Whaley for American options)

**Pros:**
- Extremely fast (<100ns)
- No pre-computation needed
- No memory overhead
- Well-studied, validated

**Cons:**
- Limited accuracy for American options (especially with dividends)
- Doesn't handle complex payoffs or exotic options
- Less flexible than FDM

**Verdict:** ⚠️ Complementary - Use for European options, interpolation for American

---

### Approach C: Polynomial Regression

**Concept:** Fit high-order polynomials to FDM prices

**Pros:**
- Smooth interpolation
- Analytical derivatives (for Greeks)
- Compact representation

**Cons:**
- Overfitting with high-order polynomials
- Runge's phenomenon (oscillations near boundaries)
- Negative prices or other unphysical results
- Hard to control local accuracy

**Verdict:** ❌ Rejected - Interpolation is safer and more predictable

---

## Comparison to Industry Practice

**How other systems handle this:**

1. **Bloomberg OVME**:
   - Uses fitted IV surfaces (SVI, SABR models)
   - Analytical approximations for pricing
   - FDM for complex cases

2. **Reuters/Refinitiv**:
   - Pre-computed tables for standard options
   - Real-time FDM for custom structures

3. **Exchanges (CBOE, CME)**:
   - Publish theoretical prices using proprietary models
   - Likely use pre-computed tables + adjustments

4. **Quant Trading Firms**:
   - Hybrid approach: fast approximations + occasional FDM recalculations
   - GPU-accelerated batch pricing
   - Aggressive caching and memoization

**Our Approach:**
Similar to industry standard (hybrid interpolation + FDM), but with:
- Open-source implementation
- Research-grade FDM solver
- Explicit accuracy/performance trade-offs

---

## Success Metrics

**Phase 1 (Core Infrastructure):**
- ✅ All unit tests pass (>90% coverage)
- ✅ IV surface query <100ns
- ✅ Price table query <500ns
- ✅ Interpolation accuracy <0.5% RMS error

**Phase 2 (Pre-computation):**
- ✅ Pre-compute 100K prices in <5 minutes
- ✅ File I/O <100ms for typical tables
- ✅ Batch speedup >50x with OpenMP

**Phase 3 (High-Level APIs):**
- ✅ Example programs run successfully
- ✅ Documentation complete and accurate
- ✅ Benchmark shows >10,000x speedup vs FDM

**Phase 4 (Production Ready):**
- ✅ Memory usage <100MB per underlying
- ✅ No crashes or memory leaks under stress testing
- ✅ Validated against market data (if available)

---

## Open Questions

1. **Grid Selection Strategy:**
   - Should grid be uniform or adaptive?
   - How to choose optimal grid density?
   - Should users specify grids manually or use automatic heuristics?

2. **Update Frequency:**
   - How often to re-compute tables?
   - Incremental updates or full rebuild?
   - Trigger based on time or market movement?

3. **Dividend Handling:**
   - Separate table per dividend schedule?
   - Interpolate dividend yield?
   - Handle discrete dividends explicitly?

4. **Greeks Calculation:**
   - Finite differences on interpolated prices? ✓ Implemented
   - **Note:** Gamma (second derivative) is approximately zero for multilinear interpolation within cells, as it is piecewise linear. For accurate gamma, cubic spline interpolation or storing pre-computed Greeks would be needed.
   - Store pre-computed Greeks in tables? (Future work)
   - Analytical Greeks from spline derivatives? (Requires cubic spline strategy)

5. **Error Bounds:**
   - Can we provide theoretical error bounds?
   - How to estimate error without computing FDM price?
   - Trade-off between error estimation overhead and safety?

---

## Conclusion and Recommendation

**Recommendation: Proceed with Hybrid Design (Alternative 3)**

**Rationale:**
1. ✅ Achieves target <1µs query performance (40,000x faster than FDM)
2. ✅ Manageable memory footprint (~5MB per underlying)
3. ✅ Expected accuracy <0.5% with reasonable grid resolution
4. ✅ Modular design allows incremental implementation
5. ✅ Leverages existing FDM solver for pre-computation
6. ✅ Industry-standard approach (proven in production)

**Implementation Priority:**
1. **Phase 1** (Weeks 1-2): Core data structures and interpolation
2. **Phase 2** (Weeks 3-4): Pre-computation engine with batch processing
3. **Phase 3** (Week 5): Examples and documentation
4. **Phase 4** (Optional): Advanced features as needed

**Expected Impact:**
- **Performance**: 10,000-40,000x speedup for real-time pricing
- **Throughput**: >1M option prices per second (single-threaded)
- **Memory**: <100MB for typical use cases (100 underlyings)
- **Accuracy**: <0.5% RMS error with recommended grid densities

**Next Steps:**
1. Create GitHub issue for tracking
2. Create feature branch for implementation
3. Start with Phase 1 (core infrastructure)
4. Validate design with prototype
5. Iterate based on performance/accuracy measurements
