<!-- SPDX-License-Identifier: MIT -->
# Interpolation Strategy Design: Dependency Injection

## Overview

This document describes a **strategy pattern** using dependency injection to allow runtime selection of interpolation algorithms. This provides:

- **Flexibility**: Switch between linear, cubic, or custom interpolation at runtime
- **Performance tuning**: Choose speed vs accuracy based on use case
- **Extensibility**: Add new algorithms without modifying existing code
- **Testability**: Easy to benchmark and compare algorithms

**Design Philosophy:** Follows the existing callback-based architecture used by `PDESolver`.

---

## Core Design: Interpolation Strategy Interface

### Data Structures

```c
// Forward declarations
typedef struct InterpolationStrategy InterpolationStrategy;
typedef struct OptionPriceTable OptionPriceTable;
typedef struct IVSurface IVSurface;

/**
 * Interpolation context: scratch space for algorithm-specific data
 * Each strategy can allocate and manage its own context
 */
typedef void* InterpContext;

/**
 * Strategy interface: function pointers for interpolation operations
 *
 * All strategies must implement these functions.
 * Follows "vtable" pattern common in C for polymorphism.
 */
typedef struct InterpolationStrategy {
    // Name of the strategy (for logging/debugging)
    const char *name;

    // Short description
    const char *description;

    // ---------- 2D Interpolation (IV Surfaces) ----------

    /**
     * Interpolate on 2D grid (moneyness, maturity)
     *
     * @param surface: IV surface data
     * @param moneyness: query point (m = S/K)
     * @param maturity: query point (tau = T-t)
     * @param context: algorithm-specific scratch space
     * @return interpolated IV value
     */
    double (*interpolate_2d)(const IVSurface *surface,
                             double moneyness,
                             double maturity,
                             InterpContext context);

    // ---------- 4D Interpolation (Price Tables) ----------

    /**
     * Interpolate on 4D grid (moneyness, maturity, volatility, rate)
     *
     * @param table: option price table
     * @param moneyness: query point
     * @param maturity: query point
     * @param volatility: query point
     * @param rate: query point
     * @param context: algorithm-specific scratch space
     * @return interpolated option price
     */
    double (*interpolate_4d)(const OptionPriceTable *table,
                             double moneyness,
                             double maturity,
                             double volatility,
                             double rate,
                             InterpContext context);

    /**
     * Interpolate on 5D grid (adds dividend dimension)
     * Optional: can be NULL if not supported
     */
    double (*interpolate_5d)(const OptionPriceTable *table,
                             double moneyness,
                             double maturity,
                             double volatility,
                             double rate,
                             double dividend,
                             InterpContext context);

    // ---------- Context Management ----------

    /**
     * Create algorithm-specific context (scratch space)
     * Called once when strategy is initialized
     *
     * @param dimensions: number of dimensions (2, 4, or 5)
     * @param grid_sizes: array of grid sizes for each dimension
     * @return opaque context pointer (owned by caller)
     */
    InterpContext (*create_context)(size_t dimensions,
                                     const size_t *grid_sizes);

    /**
     * Destroy context and free resources
     *
     * @param context: context created by create_context()
     */
    void (*destroy_context)(InterpContext context);

    // ---------- Optional: Pre-computation ----------

    /**
     * Optional: Pre-compute coefficients or data structures
     * For cubic splines: compute spline coefficients
     * For linear: no-op (return 0)
     *
     * @param grid_data: raw grid data
     * @param context: context to store pre-computed data
     * @return 0 on success, -1 on error
     */
    int (*precompute)(const void *grid_data, InterpContext context);

} InterpolationStrategy;
```

---

## Updated Data Structures

### IVSurface with Strategy

```c
typedef struct {
    // Grid definition
    size_t n_moneyness;
    size_t n_maturity;
    double *moneyness_grid;
    double *maturity_grid;
    double *iv_surface;

    // Metadata
    char underlying[32];
    time_t last_update;

    // --- NEW: Interpolation strategy (dependency injection) ---
    const InterpolationStrategy *strategy;  // Strategy vtable (not owned)
    InterpContext interp_context;           // Algorithm-specific context (owned)

} IVSurface;
```

### OptionPriceTable with Strategy

```c
typedef struct {
    // Grid definition (4D or 5D)
    size_t n_moneyness;
    size_t n_maturity;
    size_t n_volatility;
    size_t n_rate;
    size_t n_dividend;

    double *moneyness_grid;
    double *maturity_grid;
    double *volatility_grid;
    double *rate_grid;
    double *dividend_grid;

    double *prices;  // Flattened array

    // Metadata
    OptionType type;
    ExerciseType exercise;
    char underlying[32];
    time_t generation_time;

    // Indexing strides
    size_t stride_m, stride_tau, stride_sigma, stride_r, stride_q;

    // --- NEW: Interpolation strategy (dependency injection) ---
    const InterpolationStrategy *strategy;  // Strategy vtable (not owned)
    InterpContext interp_context;           // Algorithm-specific context (owned)

} OptionPriceTable;
```

---

## Built-in Strategies

### 1. Multi-Linear Interpolation

```c
// Global strategy instance (stateless, can be shared)
extern const InterpolationStrategy INTERP_MULTILINEAR;

// Implementation
double multilinear_interpolate_4d(const OptionPriceTable *table,
                                   double m, double tau, double sigma, double r,
                                   InterpContext ctx) {
    // 1. Find bracketing indices (binary search)
    size_t i_m = find_bracket(table->moneyness_grid, table->n_moneyness, m);
    size_t i_tau = find_bracket(table->maturity_grid, table->n_maturity, tau);
    size_t i_sigma = find_bracket(table->volatility_grid, table->n_volatility, sigma);
    size_t i_r = find_bracket(table->rate_grid, table->n_rate, r);

    // 2. Get 16 hypercube corner values (2^4 = 16)
    double values[16];
    for (int im = 0; im < 2; im++) {
        for (int it = 0; it < 2; it++) {
            for (int is = 0; is < 2; is++) {
                for (int ir = 0; ir < 2; ir++) {
                    size_t idx = (i_m + im) * table->stride_m
                               + (i_tau + it) * table->stride_tau
                               + (i_sigma + is) * table->stride_sigma
                               + (i_r + ir) * table->stride_r;
                    values[im*8 + it*4 + is*2 + ir] = table->prices[idx];
                }
            }
        }
    }

    // 3. Interpolate recursively (15 lerps for 4D)
    // ... (standard multilinear algorithm)

    return interpolated_value;
}

// Context management (no context needed for linear)
InterpContext multilinear_create_context(size_t dims, const size_t *sizes) {
    return NULL;  // Stateless
}

void multilinear_destroy_context(InterpContext ctx) {
    // No-op
}

// Strategy definition
const InterpolationStrategy INTERP_MULTILINEAR = {
    .name = "multilinear",
    .description = "Separable multi-linear interpolation (fast, C0)",
    .interpolate_2d = multilinear_interpolate_2d,
    .interpolate_4d = multilinear_interpolate_4d,
    .interpolate_5d = multilinear_interpolate_5d,
    .create_context = multilinear_create_context,
    .destroy_context = multilinear_destroy_context,
    .precompute = NULL  // No pre-computation needed
};
```

### 2. Tensor-Product Cubic Spline

```c
// Global strategy instance
extern const InterpolationStrategy INTERP_CUBIC_SPLINE;

// Context: stores pre-computed spline coefficients
typedef struct {
    CubicSpline *spline_moneyness;
    CubicSpline *spline_maturity;
    CubicSpline *spline_volatility;
    CubicSpline *spline_rate;
} CubicSplineContext;

// Implementation
double cubic_interpolate_4d(const OptionPriceTable *table,
                             double m, double tau, double sigma, double r,
                             InterpContext ctx) {
    CubicSplineContext *spline_ctx = (CubicSplineContext*)ctx;

    // Similar to multilinear, but uses cubic spline evaluations
    // ... (tensor-product cubic spline algorithm)

    return interpolated_value;
}

InterpContext cubic_create_context(size_t dims, const size_t *sizes) {
    CubicSplineContext *ctx = malloc(sizeof(CubicSplineContext));
    // Allocate spline coefficient storage
    // ...
    return ctx;
}

int cubic_precompute(const void *grid_data, InterpContext ctx) {
    // Build cubic spline coefficients for each dimension
    // ... (uses existing pde_spline_create())
    return 0;
}

void cubic_destroy_context(InterpContext ctx) {
    CubicSplineContext *spline_ctx = (CubicSplineContext*)ctx;
    // Free spline coefficient arrays
    free(spline_ctx);
}

const InterpolationStrategy INTERP_CUBIC_SPLINE = {
    .name = "cubic_spline",
    .description = "Tensor-product cubic splines (smooth, C2)",
    .interpolate_2d = cubic_interpolate_2d,
    .interpolate_4d = cubic_interpolate_4d,
    .interpolate_5d = cubic_interpolate_5d,
    .create_context = cubic_create_context,
    .destroy_context = cubic_destroy_context,
    .precompute = cubic_precompute
};
```

### 3. Custom Strategy (User-Defined)

Users can define their own strategies:

```c
// Example: Monotone cubic Hermite interpolation (prevents overshoot)
double monotone_cubic_interpolate_4d(...) {
    // Custom implementation
}

const InterpolationStrategy INTERP_MONOTONE_CUBIC = {
    .name = "monotone_cubic",
    .description = "Monotone-preserving cubic interpolation",
    .interpolate_4d = monotone_cubic_interpolate_4d,
    // ... other functions
};
```

---

## Updated APIs

### IVSurface with Strategy Injection

```c
/**
 * Create IV surface with specified interpolation strategy
 *
 * @param strategy: interpolation strategy (e.g., &INTERP_MULTILINEAR)
 *                  If NULL, defaults to multilinear
 */
IVSurface* iv_surface_create_with_strategy(
    const double *moneyness, size_t n_m,
    const double *maturity, size_t n_tau,
    const InterpolationStrategy *strategy);

// Convenience: create with default (multilinear)
IVSurface* iv_surface_create(const double *moneyness, size_t n_m,
                              const double *maturity, size_t n_tau) {
    return iv_surface_create_with_strategy(moneyness, n_m, maturity, n_tau,
                                            &INTERP_MULTILINEAR);
}

/**
 * Change interpolation strategy at runtime
 * Destroys old context, creates new one
 */
int iv_surface_set_strategy(IVSurface *surface,
                             const InterpolationStrategy *strategy);

/**
 * Interpolate using injected strategy
 */
double iv_surface_interpolate(const IVSurface *surface,
                               double moneyness, double maturity) {
    // Dispatch to strategy's interpolate_2d function
    return surface->strategy->interpolate_2d(surface, moneyness, maturity,
                                              surface->interp_context);
}

void iv_surface_destroy(IVSurface *surface) {
    if (surface->strategy && surface->strategy->destroy_context) {
        surface->strategy->destroy_context(surface->interp_context);
    }
    // ... free other resources
    free(surface);
}
```

### OptionPriceTable with Strategy Injection

```c
/**
 * Create price table with specified interpolation strategy
 */
OptionPriceTable* price_table_create_with_strategy(
    const double *moneyness, size_t n_m,
    const double *maturity, size_t n_tau,
    const double *volatility, size_t n_sigma,
    const double *rate, size_t n_r,
    const double *dividend, size_t n_q,
    OptionType type, ExerciseType exercise,
    const InterpolationStrategy *strategy);  // NEW parameter

// Convenience: defaults to multilinear
OptionPriceTable* price_table_create(...) {
    return price_table_create_with_strategy(..., &INTERP_MULTILINEAR);
}

/**
 * Change strategy at runtime
 */
int price_table_set_strategy(OptionPriceTable *table,
                              const InterpolationStrategy *strategy);

/**
 * Interpolate using injected strategy
 */
double price_table_interpolate(const OptionPriceTable *table,
                                double moneyness, double maturity,
                                double volatility, double rate) {
    // Dispatch to strategy's interpolate_4d function
    return table->strategy->interpolate_4d(table, moneyness, maturity,
                                            volatility, rate,
                                            table->interp_context);
}
```

---

## Usage Examples

### Example 1: Default (Multi-Linear)

```c
// Create with default strategy (multilinear)
OptionPriceTable *table = price_table_create(
    moneyness, n_m,
    maturity, n_tau,
    volatility, n_sigma,
    rate, n_r,
    nullptr, 0,  // no dividend dimension
    OPTION_PUT, AMERICAN);

// Interpolate (uses multilinear automatically)
double price = price_table_interpolate(table, 1.05, 0.25, 0.20, 0.05);

price_table_destroy(table);
```

### Example 2: Explicit Strategy Selection

```c
// Create with cubic spline strategy
OptionPriceTable *table = price_table_create_with_strategy(
    moneyness, n_m,
    maturity, n_tau,
    volatility, n_sigma,
    rate, n_r,
    nullptr, 0,
    OPTION_PUT, AMERICAN,
    &INTERP_CUBIC_SPLINE);  // Inject cubic spline

// Pre-compute spline coefficients (optional, for cubic)
if (table->strategy->precompute) {
    table->strategy->precompute(table, table->interp_context);
}

// Interpolate (uses cubic spline)
double price = price_table_interpolate(table, 1.05, 0.25, 0.20, 0.05);

price_table_destroy(table);  // Cleans up spline context automatically
```

### Example 3: Runtime Strategy Switching

```c
OptionPriceTable *table = price_table_create(...);  // defaults to multilinear

// During development: benchmark different strategies
printf("Testing multilinear...\n");
double price_linear = price_table_interpolate(table, 1.05, 0.25, 0.20, 0.05);

// Switch to cubic for higher accuracy
price_table_set_strategy(table, &INTERP_CUBIC_SPLINE);
printf("Testing cubic spline...\n");
double price_cubic = price_table_interpolate(table, 1.05, 0.25, 0.20, 0.05);

printf("Linear: %.6f, Cubic: %.6f, Diff: %.6f\n",
       price_linear, price_cubic, fabs(price_linear - price_cubic));

price_table_destroy(table);
```

### Example 4: User-Defined Strategy

```c
// User implements custom monotone cubic interpolation
const InterpolationStrategy MY_CUSTOM_STRATEGY = {
    .name = "my_algorithm",
    .description = "Custom monotone-preserving interpolation",
    .interpolate_4d = my_custom_interpolate_4d,
    // ... other required functions
};

// Use custom strategy
OptionPriceTable *table = price_table_create_with_strategy(
    ...,
    &MY_CUSTOM_STRATEGY);

// Works transparently
double price = price_table_interpolate(table, 1.05, 0.25, 0.20, 0.05);
```

### Example 5: Benchmark Suite

```c
// Compare all available strategies
const InterpolationStrategy *strategies[] = {
    &INTERP_MULTILINEAR,
    &INTERP_CUBIC_SPLINE,
    &INTERP_MONOTONE_CUBIC,
    NULL
};

OptionPriceTable *table = price_table_create(...);

for (int i = 0; strategies[i] != NULL; i++) {
    price_table_set_strategy(table, strategies[i]);

    // Benchmark
    clock_t start = clock();
    for (int j = 0; j < 100000; j++) {
        double price = price_table_interpolate(table, 1.0, 0.5, 0.2, 0.05);
    }
    clock_t end = clock();

    double time_per_query = (double)(end - start) / CLOCKS_PER_SEC / 100000.0;
    printf("%s: %.2f ns per query\n",
           strategies[i]->name, time_per_query * 1e9);
}

price_table_destroy(table);
```

---

## Benefits of This Design

### 1. **Flexibility**
- Switch algorithms at runtime without recompilation
- Easy A/B testing during development
- Production can choose based on performance profiling

### 2. **Extensibility**
- Add new algorithms without modifying existing code
- Third-party plugins possible
- Research can experiment with novel methods

### 3. **Performance Tuning**
- Default (multilinear): Fast, ~500ns per query
- High-accuracy (cubic): Smooth, ~1-2µs per query
- User can profile and choose optimal strategy

### 4. **Testability**
- Easy to write tests comparing algorithms
- Benchmark suite validates all strategies
- Accuracy tests ensure correctness

### 5. **Consistency with Existing Design**
- Follows same callback pattern as `PDESolver`
- Familiar API style for existing users
- Minimal changes to client code

### 6. **Safety**
- Type-safe dispatch through vtable
- Clear ownership semantics (table owns context, not strategy)
- No virtual function overhead (C function pointers)

---

## Implementation Phases

### Phase 1: Core Infrastructure (Week 1-2)
- ✅ Define `InterpolationStrategy` interface
- ✅ Implement `INTERP_MULTILINEAR` strategy
- ✅ Update `IVSurface` and `OptionPriceTable` to use strategy
- ✅ Add tests for strategy injection

### Phase 2: Cubic Spline Strategy (Week 3-4)
- ✅ Implement `INTERP_CUBIC_SPLINE` strategy
- ✅ Add pre-computation for spline coefficients
- ✅ Benchmark linear vs cubic

### Phase 3: Advanced Strategies (Optional)
- ⭐ Monotone cubic Hermite interpolation
- ⭐ Akima splines (local interpolation)
- ⭐ RBF (Radial Basis Functions) for scattered data

### Phase 4: Optimization
- ⭐ SIMD-optimized multilinear interpolation
- ⭐ Cache-friendly memory layouts per strategy
- ⭐ GPU-accelerated batch interpolation

---

## File Organization

```
src/
├── interp_strategy.h         # Strategy interface definition
├── interp_multilinear.c      # Multi-linear implementation
├── interp_cubic.c            # Cubic spline implementation
├── iv_surface.h/.c           # Updated to use strategy
└── price_table.h/.c          # Updated to use strategy

tests/
├── interp_strategy_test.cc   # Strategy interface tests
├── interp_benchmark.cc       # Performance comparison
└── interp_accuracy_test.cc   # Accuracy validation
```

---

## Comparison to Alternatives

### Alternative 1: Compile-Time Selection (#ifdef)

```c
#ifdef USE_CUBIC_SPLINE
double price_table_interpolate(...) {
    return cubic_interpolate_4d(...);
}
#else
double price_table_interpolate(...) {
    return multilinear_interpolate_4d(...);
}
#endif
```

**Problems:**
- ❌ Requires recompilation
- ❌ Can't compare algorithms in same binary
- ❌ Not extensible

### Alternative 2: Enum-Based Dispatch

```c
typedef enum {
    INTERP_LINEAR,
    INTERP_CUBIC
} InterpAlgorithm;

double price_table_interpolate(..., InterpAlgorithm algo) {
    switch (algo) {
        case INTERP_LINEAR: return multilinear_interpolate_4d(...);
        case INTERP_CUBIC: return cubic_interpolate_4d(...);
    }
}
```

**Problems:**
- ❌ Not extensible (can't add user strategies)
- ❌ Switch overhead on every call
- ❌ Tightly couples API to implementations

### Our Approach: Strategy Pattern (Function Pointers)

**Advantages:**
- ✅ Runtime selection without switch overhead
- ✅ Extensible (user-defined strategies)
- ✅ Loose coupling
- ✅ Standard C pattern

---

## Performance Impact

**Overhead:** Single indirect function call (~1-2ns)
- Multilinear: 500ns → 502ns (negligible)
- Cubic: 1500ns → 1502ns (negligible)

**Benefit:** Flexibility without measurable performance loss

---

## Conclusion

This dependency injection design provides:
- **Flexibility** without sacrificing performance
- **Extensibility** for research and experimentation
- **Consistency** with existing codebase patterns
- **Simplicity** through clean interfaces

**Recommended:** Implement in Phase 1 alongside multilinear interpolation. The overhead is negligible, and the benefits are substantial for long-term maintainability.
