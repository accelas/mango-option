# American Option Implied Volatility Implementation Design

**Date:** 2025-10-31
**Status:** Design Approved
**Related Issues:** #40 (Coordinate Transformations)

---

## Executive Summary

Implement implied volatility calculation for **American options** as the primary scope of the mango-iv project. This design removes the European option module and replaces it with:

1. **FDM-based American IV** - Reference implementation using nested Brent + PDE solver (~200-300ms)
2. **Interpolation-based American IV** - Fast queries via 3D precomputed price tables (~7.5µs)
3. **Let's Be Rational** - Fast European IV estimation for establishing intelligent upper bounds

**Key Performance Targets:**
- FDM IV: ~250ms per calculation (ground truth)
- Interpolation IV: ~7.5µs per query (40,000x speedup)
- Accuracy: < 1 basis point difference between methods

---

## Background & Motivation

### Current Problem

The project's stated goal is to calculate implied volatility for **American options**, but the current implementation uses European option pricing (Black-Scholes):

```c
// Current: WRONG for American options
IVResult implied_volatility_calculate(...) {
    // Uses black_scholes_price() - only correct for European!
}
```

**Why this is incorrect:**
- American options have early exercise premium
- Black-Scholes undervalues American options
- IV from European formula is systematically biased

### Architectural Mismatch

The codebase has all the pieces needed but they're not connected:
- ✅ `american_option.c` - Full PDE solver for American pricing
- ✅ `price_table.c` - Fast interpolation infrastructure
- ✅ `brent.h` - Root finding for IV calculation
- ❌ No American IV calculation combining these components

### Solution Approach

Implement two complementary methods:

1. **FDM-based IV** - Expensive but accurate, used for validation
2. **Interpolation-based IV** - Fast queries, validated against FDM

This mirrors the existing pattern: `american_option.c` (FDM) validates `price_table.c` (interpolation).

---

## Design Overview

### Module Structure

**New Module: `lets_be_rational.{h,c}`**
- Purpose: Fast European IV estimation for bound calculation
- Replaces: `european_option.{h,c}` (to be deleted)
- Performance: ~100ns per estimate
- Used only for establishing Brent upper bounds

**Updated Module: `implied_volatility.{h,c}`**
- Purpose: American option IV calculation
- Methods: FDM-based and interpolation-based
- Dependencies: `american_option.h`, `lets_be_rational.h`, `price_table.h`

**Removed: `european_option.{h,c}`**
- No longer needed for primary use case
- Simplified codebase focused on American options

### Data Flow

```
Market Price (American Option)
         ↓
[Let's Be Rational] → European IV estimate → Upper bound for Brent
         ↓
[Brent's Method] ← [American Option PDE] × 10-15 iterations
         ↓
American Implied Volatility (FDM-based, ~250ms)
         ↓
[Validation against pre-computed table]
         ↓
American Implied Volatility (Interpolation-based, ~7.5µs)
```

---

## Component 1: Let's Be Rational Implementation

### Purpose

Fast European IV estimation using Peter Jäckel's "Let's Be Rational" algorithm. Used **only** for establishing intelligent upper bounds in Brent's method.

### API Design

```c
// src/lets_be_rational.h

typedef struct {
    double implied_vol;      // Estimated European IV
    bool converged;          // Always true for valid inputs
    const char *error;       // Error message if failed
} LBRResult;

// Fast European IV estimation (~100ns)
LBRResult lbr_implied_volatility(double spot, double strike,
                                  double time_to_maturity,
                                  double risk_free_rate,
                                  double market_price,
                                  bool is_call);
```

### Algorithm

"Let's Be Rational" uses rational function approximations for direct computation (no iteration):

1. Normalize inputs to reduced form
2. Apply rational polynomial approximations
3. Transform back to implied volatility
4. Return result

**Key properties:**
- **No iteration** - direct computation
- **High accuracy** - relative error < 1e-15
- **Fast** - ~100 nanoseconds
- **Robust** - handles edge cases (deep ITM/OTM, near expiry)

### Usage in IV Calculation

```c
// In implied_volatility.c
LBRResult lbr = lbr_implied_volatility(params->spot_price, params->strike,
                                       params->time_to_maturity,
                                       params->risk_free_rate,
                                       params->market_price,
                                       params->is_call);

// Establish Brent bounds
double lower_bound = 1e-6;
double upper_bound = lbr.converged ? lbr.implied_vol * 1.5 : 3.0;  // fallback
```

**Multiplier rationale (1.5x):**
- American IV ≤ European IV (approximately)
- Early exercise premium is usually small
- 1.5x provides comfortable margin without being too wide
- Reduces Brent iterations compared to arbitrary bounds like [0.01, 10.0]

### Implementation Source

Reference implementation available under MIT/BSD license:
- Peter Jäckel's original C++ code
- Port to C23 with minimal modifications
- Maintain algorithmic accuracy guarantees

---

## Component 2: FDM-Based American IV

### Purpose

Reference implementation for American option implied volatility using finite-difference method. Provides ground truth for validating interpolation-based IV.

### Algorithm

**Nested iteration approach:**
1. Objective function uses `american_option_price()` (~21ms per call)
2. Brent's method finds σ where American_price(σ) = market_price
3. Converges in 10-15 iterations

**Total cost: 10-15 × 21ms = 210-315ms per IV calculation**

### API Design

```c
// src/implied_volatility.h

// Main FDM-based calculation
IVResult calculate_iv(const IVParams *params,
                     const AmericanOptionGrid *grid_params,
                     double tolerance, int max_iter);

// Convenience with defaults
IVResult calculate_iv_simple(const IVParams *params);
```

**Key change:** Now requires `AmericanOptionGrid` for PDE solver configuration.

### Implementation

```c
// Objective function for Brent's method
typedef struct {
    double spot;
    double strike;
    double time_to_maturity;
    double risk_free_rate;
    double market_price;
    bool is_call;
    const AmericanOptionGrid *grid;
} AmericanObjectiveData;

static double american_objective(double volatility, void *user_data) {
    AmericanObjectiveData *data = (AmericanObjectiveData *)user_data;

    // Setup American option with guessed volatility
    OptionData option = {
        .strike = data->strike,
        .volatility = volatility,  // This is what we're solving for
        .risk_free_rate = data->risk_free_rate,
        .time_to_maturity = data->time_to_maturity,
        .option_type = data->is_call ? OPTION_CALL : OPTION_PUT,
        .n_dividends = 0,
        .dividend_times = NULL,
        .dividend_amounts = NULL
    };

    // Solve American option PDE (~21ms)
    AmericanOptionResult result = american_option_price(&option, data->grid);
    double theoretical_price = american_option_get_value_at_spot(
        result.solver, data->spot, data->strike);
    american_option_free_result(&result);

    return theoretical_price - data->market_price;
}
```

### Calculation Flow

```c
IVResult calculate_iv(const IVParams *params,
                     const AmericanOptionGrid *grid_params,
                     double tolerance, int max_iter) {
    // 1. Get European IV for upper bound
    LBRResult lbr = lbr_implied_volatility(...);
    double lower_bound = 1e-6;
    double upper_bound = lbr.converged ? lbr.implied_vol * 1.5 : 3.0;

    // 2. Setup objective function
    AmericanObjectiveData obj_data = { ... };

    // 3. Run Brent's method
    BrentResult brent = brent_find_root(american_objective,
                                        lower_bound, upper_bound,
                                        tolerance, max_iter, &obj_data);

    // 4. Return result
    IVResult result = {
        .implied_vol = brent.root,
        .vega = 0.0,  // Could compute via finite differences
        .iterations = brent.iterations,
        .converged = brent.converged,
        .error = brent.converged ? NULL : "Failed to converge"
    };

    return result;
}
```

### Performance Characteristics

| Scenario | Iterations | Time | Notes |
|----------|-----------|------|-------|
| Typical | 10-12 | ~230ms | Good initial bounds |
| Wide bounds | 15-18 | ~350ms | Arbitrary bounds [0.01, 3.0] |
| Near convergence | 8-10 | ~180ms | Tight initial guess |

**Acceptable for:**
- Reference implementation
- Validation of interpolation
- Small batches (< 100 calculations)
- Non-latency-critical applications

---

## Component 3: Interpolation-Based American IV

### Purpose

Fast American option IV queries via precomputed 3D price tables. Achieves ~40,000x speedup over FDM by replacing expensive PDE solves with interpolation + 1D root finding.

### Design Approach: 3D Price Grid + 1D Inversion

**Recommended by expert guide** (see `docs/IV_SURFACE_PRECOMPUTATION_GUIDE.md`):

1. **Precompute prices** on (x, T, σ) grid using FDM
2. **Store in 3D price table** with cubic spline coefficients
3. **Runtime**: Interpolate along σ axis to find IV

### Grid Design

Following expert recommendations for 1bp accuracy:

**Coordinate transformations:**
- x = log(K/S) - log-moneyness
- T = time to maturity (possibly √T transform)
- σ = volatility

**Grid sizes:**

| Accuracy Target | Grid (x × T × σ) | Memory | Query Time |
|-----------------|------------------|---------|------------|
| Coarse (~few bp) | 50×30×20 | ~240 KB | ~3µs |
| Medium (~1 bp) | 100×80×40 | ~2.7 MB | ~7.5µs |
| High (<0.5 bp) | 200×160×80 | ~21 MB | ~15µs |

**Recommended default: 100×80×40** for 1bp accuracy.

### API Design

```c
// Fast IV query via interpolation
double calculate_iv_interpolated(const OptionPriceTable *table,
                                double spot, double strike,
                                double time_to_maturity, double rate,
                                double market_price);
```

### Implementation

```c
double calculate_iv_interpolated(const OptionPriceTable *table,
                                 double spot, double strike,
                                 double time_to_maturity, double rate,
                                 double market_price) {
    // Transform to grid coordinates
    double moneyness = spot / strike;
    double x = log(moneyness);  // Using COORD_LOG_SQRT from issue #40

    // 1D objective: find σ where interpolated_price(x, T, σ) = market_price
    double objective(double sigma) {
        return price_table_interpolate_3d(table, x, time_to_maturity, sigma)
               - market_price;
    }

    // Brent along σ dimension only
    // Each iteration: 500ns interpolation
    // Total: 10-15 iterations × 500ns = ~7.5µs
    BrentResult result = brent_find_root(objective, 0.01, 3.0,
                                        1e-6, 100, NULL);

    return result.converged ? result.root : NAN;
}
```

### Precomputation Workflow

```c
// Create 3D price table (x, T, σ)
OptionPriceTable *table = price_table_create_3d(
    x_grid, 100,      // log-moneyness: [-0.7, 0.7]
    T_grid, 80,       // maturity: [0.027, 2.0]
    sigma_grid, 40    // volatility: [0.1, 0.8]
);

// Populate via FDM (one-time expensive operation)
// Uses american_option_price_batch() with OpenMP
#pragma omp parallel for collapse(3)
for (size_t i = 0; i < 100; i++) {
    for (size_t j = 0; j < 80; j++) {
        for (size_t k = 0; k < 40; k++) {
            double x_i = x_grid[i];
            double T_j = T_grid[j];
            double sigma_k = sigma_grid[k];

            // Solve American option PDE
            OptionData option = { .volatility = sigma_k, ... };
            AmericanOptionResult result = american_option_price(&option, &grid);
            double price = american_option_get_value_at_spot(result.solver, ...);

            price_table_set_3d(table, i, j, k, price);
            american_option_free_result(&result);
        }
    }
}

// Build cubic spline coefficients
price_table_build_interpolation(table);

// Save for fast loading
price_table_save(table, "american_iv_table_100x80x40.bin");
```

**Precomputation cost:** 100×80×40 = 320,000 PDE solves
- Sequential: 320,000 × 21ms = ~1.9 hours
- 16 cores: ~7 minutes
- **One-time cost** - table can be reused for millions of queries

### Performance Analysis

**Query performance:**
- Interpolation: ~500ns per price lookup
- Brent iterations: 10-15 (along σ axis only)
- **Total: ~7.5µs per IV calculation**

**Speedup vs FDM:**
- FDM: ~250ms
- Interpolation: ~7.5µs
- **Speedup: 33,000x**

**Accuracy (per expert guide):**
- Target: < 1 basis point (0.0001 in IV units)
- Achievable with 100×80×40 grid
- Validated against FDM reference

---

## Testing Strategy

### Layer 1: Unit Tests

**tests/lets_be_rational_test.cc**
- Accuracy vs known European IV values
- Edge cases (deep ITM/OTM, near expiry)
- Boundary conditions (zero vol, very long maturity)
- Performance benchmarks (~100ns target)

**tests/implied_volatility_test.cc (updated)**
- FDM-based American IV convergence
- Brent iteration counts (10-15 expected)
- Handles arbitrage bounds correctly
- Grid resolution sensitivity
- Dividend support (if implemented)

### Layer 2: Cross-Validation

**tests/iv_accuracy_test.cc (new)**
- Compare FDM IV vs Interpolation IV
- Target: < 1bp difference
- Test scenarios:
  - ATM options (m=1.0, various T)
  - OTM/ITM options (m=0.7 to 1.3)
  - Short maturity (T=0.027 to 0.1)
  - Long maturity (T=1.0 to 2.0)
  - Low/high volatility (σ=0.1 to 0.8)

**Validation grid:**
- Generate 1,000 random test cases
- Calculate IV via both methods
- Measure absolute and relative errors
- Report statistics (mean, max, 95th percentile)

### Layer 3: Integration Tests

**tests/american_iv_integration_test.cc (new)**
- End-to-end workflow validation
- Real market scenarios
- Performance verification
- Memory leak checks (valgrind)

### Layer 4: Performance Benchmarks

**benchmarks/iv_benchmark.cc (new)**
```
Benchmark                        Time         Iterations
-----------------------------------------------------------
FDM_IV_ATM_1Year               230 ms              10
FDM_IV_OTM_ShortMaturity       195 ms              12
Interpolated_IV_ATM            7.2 µs         100000
Interpolated_IV_OTM            7.8 µs         100000
Batch_FDM_100_Sequential       23.5 s               1
Batch_Interpolated_100         750 µs          10000
```

**Success criteria:**
- FDM IV: 180-350ms (acceptable range)
- Interpolated IV: < 10µs
- Speedup: > 30,000x
- Accuracy: < 1bp difference in 95% of cases

---

## Migration Path

### Phase 1: Foundation (Weeks 1-2)

**Deliverables:**
1. Implement `lets_be_rational.{h,c}`
2. Add unit tests for LBR
3. Update `implied_volatility.c` with American objective function
4. Implement `calculate_iv()` using FDM
5. Comprehensive unit tests for FDM-based IV

**Success criteria:**
- LBR produces accurate European IV (<1e-10 error vs reference)
- FDM-based American IV converges in 10-15 iterations
- All tests pass

### Phase 2: Module Removal (Week 2)

**Deliverables:**
1. Delete `european_option.{h,c}`
2. Delete `tests/european_option_test.cc`
3. Update all BUILD.bazel files
4. Fix examples and benchmarks
5. Update ARCHITECTURE.md references

**Success criteria:**
- No compilation errors
- No broken tests
- No references to european_option remain

### Phase 3: Interpolation (Weeks 3-4)

**Deliverables:**
1. Extend `price_table` to support 3D grids
2. Implement precomputation workflow
3. Implement `calculate_iv_interpolated()`
4. Add accuracy validation tests
5. Performance benchmarks

**Success criteria:**
- < 1bp accuracy on validation set
- < 10µs query time
- > 30,000x speedup vs FDM

### Phase 4: Documentation (Week 4)

**Deliverables:**
1. Update README.md (American option focus)
2. Update ARCHITECTURE.md (new IV design)
3. Update PROJECT_OVERVIEW.md
4. Update QUICK_REFERENCE.md
5. Add IV calculation user guide

**Success criteria:**
- Documentation clearly states American option scope
- All design decisions documented
- Usage examples provided

---

## Risk Analysis

### Technical Risks

**Risk 1: Let's Be Rational implementation complexity**
- **Mitigation:** Use existing reference implementation (MIT/BSD)
- **Impact:** Low - well-established algorithm

**Risk 2: FDM IV convergence issues**
- **Mitigation:** Extensive testing with edge cases
- **Contingency:** Adaptive bounds, better initial guesses
- **Impact:** Medium - core functionality

**Risk 3: Interpolation accuracy insufficient**
- **Mitigation:** Follow expert guide recommendations (100×80×40 grid)
- **Contingency:** Increase grid resolution
- **Impact:** Medium - affects production use case

**Risk 4: Precomputation time too long**
- **Mitigation:** OpenMP parallelization, reasonable grid size
- **Contingency:** Coarser grid, adaptive refinement
- **Impact:** Low - one-time cost

### Performance Risks

**Risk 5: FDM IV too slow for practical use**
- **Current:** ~250ms per calculation
- **Mitigation:** This is acceptable for reference implementation
- **Contingency:** Use interpolation for production queries
- **Impact:** Low - interpolation provides fast path

**Risk 6: Memory usage too high for large grids**
- **Current:** 100×80×40 = 2.7 MB
- **Mitigation:** Reasonable for modern systems
- **Contingency:** Compression, adaptive grids
- **Impact:** Low - memory is cheap

---

## Success Metrics

### Functional Requirements
- ✅ FDM-based American IV converges reliably
- ✅ Interpolation-based IV < 1bp error vs FDM
- ✅ No European option dependencies remain
- ✅ All tests pass (unit, integration, validation)

### Performance Requirements
- ✅ FDM IV: 180-350ms per calculation
- ✅ Interpolated IV: < 10µs per query
- ✅ Speedup: > 30,000x
- ✅ Batch processing with OpenMP scaling

### Documentation Requirements
- ✅ README states American option focus
- ✅ Architecture documented
- ✅ API reference complete
- ✅ Usage examples provided

---

## References

1. **Peter Jäckel** (2015), *Let's Be Rational*, Wilmott Magazine
2. **Expert Guide**: `docs/IV_SURFACE_PRECOMPUTATION_GUIDE.md`
3. **Issue #40**: Coordinate Transformations
4. **Existing Code**:
   - `src/american_option.{h,c}` - FDM solver
   - `src/price_table.{h,c}` - Interpolation infrastructure
   - `src/brent.h` - Root finding

---

## Appendix A: API Summary

### lets_be_rational.h
```c
LBRResult lbr_implied_volatility(double spot, double strike,
                                  double time_to_maturity,
                                  double risk_free_rate,
                                  double market_price,
                                  bool is_call);
```

### implied_volatility.h
```c
// FDM-based
IVResult calculate_iv(const IVParams *params,
                     const AmericanOptionGrid *grid_params,
                     double tolerance, int max_iter);

IVResult calculate_iv_simple(const IVParams *params);

// Interpolation-based
double calculate_iv_interpolated(const OptionPriceTable *table,
                                double spot, double strike,
                                double time_to_maturity, double rate,
                                double market_price);
```

---

## Appendix B: File Changes Summary

**New Files:**
- `src/lets_be_rational.h`
- `src/lets_be_rational.c`
- `tests/lets_be_rational_test.cc`
- `tests/iv_accuracy_test.cc`
- `tests/american_iv_integration_test.cc`
- `benchmarks/iv_benchmark.cc`
- `docs/IV_SURFACE_PRECOMPUTATION_GUIDE.md` ✅ (already created)
- `docs/plans/2025-10-31-american-iv-implementation-design.md` ✅ (this file)

**Modified Files:**
- `src/implied_volatility.h` - Update API signatures
- `src/implied_volatility.c` - Replace European with American
- `tests/implied_volatility_test.cc` - Update tests for American
- `src/BUILD.bazel` - Update dependencies
- `tests/BUILD.bazel` - Add new tests
- `benchmarks/BUILD.bazel` - Add IV benchmark

**Deleted Files:**
- `src/european_option.h`
- `src/european_option.c`
- `tests/european_option_test.cc`

**Documentation Updates (after implementation):**
- `README.md` - State American option focus
- `docs/ARCHITECTURE.md` - Document new IV design
- `docs/PROJECT_OVERVIEW.md` - Update scope statement
- `docs/QUICK_REFERENCE.md` - Update API reference

---

**End of Design Document**
