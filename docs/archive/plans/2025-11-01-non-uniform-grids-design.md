<!-- SPDX-License-Identifier: MIT -->
# Non-Uniform Grid Spacing for Option Price Tables (P5)

**Date:** 2025-11-01
**Issue:** #39 (P5)
**Status:** Design Phase
**Branch:** `feature/non-uniform-grids`

## Background

Current price table implementation uses uniform or log-uniform spacing across all dimensions (moneyness, maturity, volatility, rate, dividend). While this provides consistent accuracy, it's inefficient because option price surfaces have varying curvature:

- **Near ATM** (m ≈ 1.0): Price changes rapidly with spot (high gamma)
- **Short maturities** (T < 3 months): Price surface is steep (high theta)
- **Far OTM/ITM**: Price surface is flat (low curvature)

Expert recommendation: Concentrate grid points where the price surface is steep to achieve better accuracy with fewer total points.

### Current Grid Sizes

Typical uniform grid for acceptable accuracy (~5% error):
- Moneyness: 30 points (log-spaced from 0.7 to 1.3)
- Maturity: 25 points (0.027 to 2.0 years)
- Volatility: 15 points (0.10 to 0.80)
- Rate: 10 points (0.0 to 0.10)
- **Total**: 30 × 25 × 15 × 10 = **112,500 points**

### Target with Non-Uniform Grids

- Reduce to **10,000-20,000 points** (5-10× reduction)
- Maintain or improve accuracy (< 1% average error)
- Focus density near critical regions

## Problem Analysis

### Price Surface Characteristics

1. **Moneyness Dimension (m = S/K)**
   - High curvature near ATM (0.95 < m < 1.05)
   - Low curvature far OTM (m > 1.3) and deep ITM (m < 0.7)
   - Second derivative (gamma) peaks at ATM
   - **Strategy**: Concentrate points near m = 1.0

2. **Maturity Dimension (τ = T - t)**
   - High curvature for short maturities (τ < 0.25)
   - Smoother for long maturities (τ > 1.0)
   - Theta magnitude increases as maturity decreases
   - **Strategy**: Concentrate points near τ = 0, spread out for τ > 0.5

3. **Volatility Dimension (σ)**
   - Moderate curvature across typical ranges
   - Vega relatively smooth
   - **Strategy**: Moderate concentration at typical trading vols (0.15-0.30)

4. **Rate Dimension (r)**
   - Low curvature (rho is relatively flat)
   - **Strategy**: Uniform spacing is acceptable

5. **Dividend Dimension (q)**
   - Similar to rate dimension
   - **Strategy**: Uniform spacing is acceptable

### Optimal Grid Spacing Strategies

#### 1. Chebyshev Nodes

Chebyshev nodes minimize Runge's phenomenon in polynomial interpolation:

```
x_i = cos((2i - 1)π / (2n)) for i = 1, 2, ..., n
```

Mapped to interval [a, b]:
```
x_i = (a + b)/2 + (b - a)/2 · cos((2i - 1)π / (2n))
```

**Advantages:**
- Optimal for polynomial interpolation (minimizes max error)
- Natural concentration at boundaries
- Well-studied theoretical properties

**Disadvantages:**
- Concentrates at BOTH boundaries (not ideal for moneyness)
- May not align with actual price surface characteristics

#### 2. Adaptive Concentration (Tanh-based)

Use hyperbolic tangent to create smooth concentration around a target point:

```
For concentration at center c with strength α:
t_i = i / (n-1)  (uniform on [0,1])
x_i = c + (x_max - c) · tanh(α(t_i - 0.5)) / tanh(α/2)
```

**Advantages:**
- Flexible concentration at any point (e.g., ATM for moneyness)
- Tunable concentration strength
- Symmetric or asymmetric concentration

**Disadvantages:**
- Requires parameter tuning (α)
- Not optimal for polynomial interpolation

#### 3. Exponential/Sinh Spacing

For one-sided concentration (e.g., short maturities):

```
Concentrate near t = 0:
τ_i = τ_max · (exp(αi/n) - 1) / (exp(α) - 1)

Or using sinh:
τ_i = τ_max · sinh(αi/n) / sinh(α)
```

**Advantages:**
- Natural for time-like dimensions
- One-sided concentration
- Smooth spacing

#### 4. Piecewise Density

Define density function and integrate:

```
For moneyness with high density at ATM:
ρ(m) = 1 + β·exp(-γ(m-1)²)  (Gaussian bump at m=1)

Grid points: Cumulative distribution matching
```

**Advantages:**
- Maximum control over point distribution
- Can match known surface characteristics
- Optimal for specific problem

**Disadvantages:**
- Complex to implement
- Requires numerical integration

### Recommendation

Use a **hybrid approach**:
1. **Moneyness**: Tanh-based concentration at m = 1.0 (ATM)
2. **Maturity**: Exponential spacing concentrating near τ = 0
3. **Volatility**: Tanh-based concentration at σ = 0.20 (typical vol)
4. **Rate/Dividend**: Uniform spacing (low curvature)

## Implementation Plan

### Task 1: Grid Generation Utilities

**New file**: `src/grid_generation.h` / `src/grid_generation.c`

#### API Design

```c
// Grid spacing strategies
typedef enum {
    GRID_UNIFORM,           // Uniform spacing
    GRID_LOG,               // Logarithmic spacing (existing)
    GRID_CHEBYSHEV,         // Chebyshev nodes
    GRID_TANH_CENTER,       // Concentration at center point
    GRID_SINH_ONESIDED,     // Concentration at one end
    GRID_CUSTOM             // User-provided spacing function
} GridSpacingType;

// Grid generation parameters
typedef struct {
    GridSpacingType type;
    double min;              // Minimum value
    double max;              // Maximum value
    size_t n_points;         // Number of points

    // Type-specific parameters
    union {
        struct {
            double center;      // Concentration center (TANH_CENTER)
            double strength;    // Concentration strength (0-10, default 3)
        } tanh_params;

        struct {
            double strength;    // Concentration strength (SINH_ONESIDED)
        } sinh_params;
    };
} GridSpec;

// Generate grid points according to specification
// Returns newly allocated array (caller must free)
double* grid_generate(const GridSpec *spec);

// Convenience functions
double* grid_uniform(double min, double max, size_t n);
double* grid_log(double min, double max, size_t n);
double* grid_chebyshev(double min, double max, size_t n);
double* grid_tanh_center(double min, double max, size_t n,
                         double center, double strength);
double* grid_sinh_onesided(double min, double max, size_t n,
                           double strength);

// Validate grid (sorted, no duplicates, in bounds)
bool grid_validate(const double *grid, size_t n, double min, double max);

// Grid quality metrics
typedef struct {
    double min_spacing;      // Minimum spacing between consecutive points
    double max_spacing;      // Maximum spacing
    double avg_spacing;      // Average spacing
    double spacing_ratio;    // max_spacing / min_spacing (uniformity)
} GridMetrics;

GridMetrics grid_compute_metrics(const double *grid, size_t n);
```

#### Implementation Details

**Chebyshev Nodes:**
```c
double* grid_chebyshev(double min, double max, size_t n) {
    double *grid = malloc(n * sizeof(double));
    if (!grid) return NULL;

    const double center = (min + max) / 2.0;
    const double radius = (max - min) / 2.0;

    for (size_t i = 0; i < n; i++) {
        double theta = (2.0 * i + 1.0) * M_PI / (2.0 * n);
        grid[i] = center + radius * cos(theta);
    }

    // Sort in ascending order (Chebyshev nodes are descending)
    qsort(grid, n, sizeof(double), compare_doubles);

    return grid;
}
```

**Tanh Concentration:**
```c
double* grid_tanh_center(double min, double max, size_t n,
                         double center, double strength) {
    double *grid = malloc(n * sizeof(double));
    if (!grid) return NULL;

    const double alpha = strength;  // Concentration strength
    const double tanh_alpha_half = tanh(alpha / 2.0);

    for (size_t i = 0; i < n; i++) {
        double t = (double)i / (double)(n - 1);  // [0, 1]
        double s = tanh(alpha * (t - 0.5)) / tanh_alpha_half;  // [-1, 1]

        // Map to [min, max] centered at 'center'
        if (s >= 0) {
            grid[i] = center + s * (max - center);
        } else {
            grid[i] = center + s * (center - min);
        }
    }

    return grid;
}
```

**Sinh One-Sided:**
```c
double* grid_sinh_onesided(double min, double max, size_t n,
                           double strength) {
    double *grid = malloc(n * sizeof(double));
    if (!grid) return NULL;

    const double alpha = strength;
    const double sinh_alpha = sinh(alpha);

    for (size_t i = 0; i < n; i++) {
        double t = (double)i / (double)(n - 1);
        double s = sinh(alpha * t) / sinh_alpha;
        grid[i] = min + s * (max - min);
    }

    return grid;
}
```

### Task 2: Preset Grid Configurations

**New file**: `src/grid_presets.h` / `src/grid_presets.c`

Provide ready-to-use grid configurations for common scenarios:

```c
// Preset grid configurations optimized for option pricing
typedef enum {
    GRID_PRESET_UNIFORM,        // Uniform (baseline)
    GRID_PRESET_LOG_STANDARD,   // Log-spaced (current default)
    GRID_PRESET_ADAPTIVE_FAST,  // Fast: ~5K points, moderate accuracy
    GRID_PRESET_ADAPTIVE_BALANCED, // Balanced: ~15K points, good accuracy
    GRID_PRESET_ADAPTIVE_ACCURATE, // Accurate: ~30K points, high accuracy
    GRID_PRESET_CUSTOM          // User-defined
} GridPreset;

// Grid configuration for all dimensions
typedef struct {
    GridSpec moneyness;
    GridSpec maturity;
    GridSpec volatility;
    GridSpec rate;
    GridSpec dividend;  // Only used if n_dividend > 0
} GridConfig;

// Get preset configuration
GridConfig grid_preset_get(GridPreset preset,
                          double m_min, double m_max,
                          double tau_min, double tau_max,
                          double sigma_min, double sigma_max,
                          double r_min, double r_max,
                          double q_min, double q_max);

// Generate all grids from configuration
typedef struct {
    double *moneyness;
    size_t n_moneyness;
    double *maturity;
    size_t n_maturity;
    double *volatility;
    size_t n_volatility;
    double *rate;
    size_t n_rate;
    double *dividend;
    size_t n_dividend;
    size_t total_points;
} GeneratedGrids;

GeneratedGrids grid_generate_all(const GridConfig *config);
void grid_free_all(GeneratedGrids *grids);
```

#### Preset Configurations

**GRID_PRESET_ADAPTIVE_FAST** (~5,000 points):
- Moneyness: 12 points, tanh at m=1.0, strength=3.0
- Maturity: 10 points, sinh concentration, strength=2.0
- Volatility: 8 points, tanh at σ=0.20, strength=1.5
- Rate: 5 points, uniform
- Total: 12 × 10 × 8 × 5 = **4,800 points**

**GRID_PRESET_ADAPTIVE_BALANCED** (~15,000 points):
- Moneyness: 20 points, tanh at m=1.0, strength=3.0
- Maturity: 15 points, sinh concentration, strength=2.5
- Volatility: 10 points, tanh at σ=0.20, strength=2.0
- Rate: 5 points, uniform
- Total: 20 × 15 × 10 × 5 = **15,000 points**

**GRID_PRESET_ADAPTIVE_ACCURATE** (~30,000 points):
- Moneyness: 25 points, tanh at m=1.0, strength=3.5
- Maturity: 20 points, sinh concentration, strength=3.0
- Volatility: 12 points, tanh at σ=0.20, strength=2.5
- Rate: 5 points, uniform
- Total: 25 × 20 × 12 × 5 = **30,000 points**

For comparison, current uniform grid: 30 × 25 × 15 × 10 = **112,500 points**

### Task 3: Integration with Price Table

**Modified**: `src/price_table.h` / `src/price_table.c`

Add new creation function that uses grid configurations:

```c
// Create price table from grid configuration
OptionPriceTable* price_table_create_from_config(
    const GridConfig *config,
    OptionType type,
    ExerciseType exercise,
    CoordinateSystem coord_system,
    MemoryLayout memory_layout);

// Create price table from preset
OptionPriceTable* price_table_create_from_preset(
    GridPreset preset,
    double m_min, double m_max,
    double tau_min, double tau_max,
    double sigma_min, double sigma_max,
    double r_min, double r_max,
    OptionType type,
    ExerciseType exercise);
```

**No changes needed** to interpolation code - it already handles arbitrary sorted grids.

### Task 4: Testing

**New file**: `tests/grid_generation_test.cc`

```cpp
TEST(GridGenerationTest, UniformBaseline) {
    double *grid = grid_uniform(0.0, 1.0, 11);
    ASSERT_NE(grid, nullptr);

    // Check endpoints
    EXPECT_DOUBLE_EQ(grid[0], 0.0);
    EXPECT_DOUBLE_EQ(grid[10], 1.0);

    // Check uniform spacing
    GridMetrics metrics = grid_compute_metrics(grid, 11);
    EXPECT_NEAR(metrics.spacing_ratio, 1.0, 0.01);

    free(grid);
}

TEST(GridGenerationTest, ChebyshevNodes) {
    double *grid = grid_chebyshev(-1.0, 1.0, 10);
    ASSERT_NE(grid, nullptr);

    // Check sorted
    for (size_t i = 0; i < 9; i++) {
        EXPECT_LT(grid[i], grid[i+1]);
    }

    // Chebyshev concentrates at boundaries
    GridMetrics metrics = grid_compute_metrics(grid, 10);
    EXPECT_GT(metrics.spacing_ratio, 2.0);  // Non-uniform

    free(grid);
}

TEST(GridGenerationTest, TanhConcentration) {
    // Concentrate at m = 1.0 (ATM)
    double *grid = grid_tanh_center(0.7, 1.3, 20, 1.0, 3.0);
    ASSERT_NE(grid, nullptr);

    // Check that points are denser near m = 1.0
    double spacing_near_atm = grid[11] - grid[9];
    double spacing_far = grid[19] - grid[17];
    EXPECT_LT(spacing_near_atm, spacing_far);

    free(grid);
}

TEST(GridGenerationTest, SinhOneSided) {
    // Concentrate at tau = 0 (short maturities)
    double *grid = grid_sinh_onesided(0.027, 2.0, 15, 2.5);
    ASSERT_NE(grid, nullptr);

    // Check that first spacing is smaller than last
    double first_spacing = grid[1] - grid[0];
    double last_spacing = grid[14] - grid[13];
    EXPECT_LT(first_spacing, last_spacing);

    free(grid);
}

TEST(GridGenerationTest, PresetGeneration) {
    GridConfig config = grid_preset_get(
        GRID_PRESET_ADAPTIVE_BALANCED,
        0.7, 1.3,  // moneyness
        0.027, 2.0,  // maturity
        0.10, 0.80,  // volatility
        0.0, 0.10,   // rate
        0.0, 0.0);   // no dividend

    GeneratedGrids grids = grid_generate_all(&config);

    EXPECT_EQ(grids.n_moneyness, 20);
    EXPECT_EQ(grids.n_maturity, 15);
    EXPECT_EQ(grids.n_volatility, 10);
    EXPECT_EQ(grids.n_rate, 5);
    EXPECT_EQ(grids.total_points, 15000);

    grid_free_all(&grids);
}
```

### Task 5: Accuracy Benchmark

**New file**: `benchmarks/grid_spacing_accuracy.cc`

Compare accuracy vs grid size for different spacing strategies:

```cpp
// Test configurations
struct GridTestCase {
    std::string name;
    GridPreset preset;
    size_t expected_points;
};

std::vector<GridTestCase> test_cases = {
    {"Uniform (baseline)", GRID_PRESET_UNIFORM, 112500},
    {"Log-spaced (current)", GRID_PRESET_LOG_STANDARD, 112500},
    {"Adaptive Fast", GRID_PRESET_ADAPTIVE_FAST, 4800},
    {"Adaptive Balanced", GRID_PRESET_ADAPTIVE_BALANCED, 15000},
    {"Adaptive Accurate", GRID_PRESET_ADAPTIVE_ACCURATE, 30000},
};

// For each configuration:
// 1. Generate grid
// 2. Precompute price table
// 3. Generate validation set (1000 random points)
// 4. Compare interpolated vs FDM prices
// 5. Report: avg error, max error, grid size, memory usage
```

Expected results:
- Adaptive Fast (4.8K): ~10% error, 23× smaller than uniform
- Adaptive Balanced (15K): ~3% error, 7.5× smaller
- Adaptive Accurate (30K): ~1% error, 3.75× smaller

### Task 6: Documentation

Update `CLAUDE.md`:

```markdown
## Non-Uniform Grid Spacing

The price table supports non-uniform grid spacing to optimize the tradeoff between accuracy and memory usage.

### Why Non-Uniform Grids?

Option price surfaces have varying curvature:
- High curvature near ATM and short maturities
- Low curvature far OTM/ITM and long maturities

Non-uniform grids concentrate points where needed, achieving better accuracy with fewer total points.

### Grid Presets

Three preset configurations optimized for different use cases:

**Fast** (~5K points):
- Good for backtesting, rapid prototyping
- ~10% average error
- 23× smaller than uniform grid

**Balanced** (~15K points):
- Production-ready accuracy
- ~3% average error
- 7.5× smaller than uniform grid

**Accurate** (~30K points):
- High-accuracy applications
- ~1% average error
- 3.75× smaller than uniform grid

### Usage

```c
// Create table with adaptive grid (balanced preset)
OptionPriceTable *table = price_table_create_from_preset(
    GRID_PRESET_ADAPTIVE_BALANCED,
    0.7, 1.3,      // moneyness range
    0.027, 2.0,    // maturity range
    0.10, 0.80,    // volatility range
    0.0, 0.10,     // rate range
    OPTION_PUT, AMERICAN);

// Or create custom grid configuration
GridConfig config = {
    .moneyness = {GRID_TANH_CENTER, 0.7, 1.3, 25, .tanh_params = {1.0, 3.5}},
    .maturity = {GRID_SINH_ONESIDED, 0.027, 2.0, 20, .sinh_params = {3.0}},
    .volatility = {GRID_TANH_CENTER, 0.10, 0.80, 12, .tanh_params = {0.20, 2.5}},
    .rate = {GRID_UNIFORM, 0.0, 0.10, 5},
};

OptionPriceTable *custom_table = price_table_create_from_config(
    &config, OPTION_PUT, AMERICAN, COORD_LOG_SQRT, LAYOUT_M_INNER);
```
```

## Implementation Schedule

### Phase 1: Core Infrastructure (Week 1)
- ✅ Design document
- ⬜ Implement grid generation utilities (grid_generation.c)
- ⬜ Add grid validation and metrics
- ⬜ Unit tests for grid generation

### Phase 2: Presets and Integration (Week 2)
- ⬜ Implement grid presets (grid_presets.c)
- ⬜ Integrate with price_table creation
- ⬜ Integration tests

### Phase 3: Benchmarking and Tuning (Week 3)
- ⬜ Implement accuracy benchmark
- ⬜ Tune preset parameters for optimal accuracy/size tradeoff
- ⬜ Performance comparison

### Phase 4: Documentation and PR (Week 4)
- ⬜ Update CLAUDE.md
- ⬜ Update example code
- ⬜ Create PR with results

## Success Criteria

- [ ] Grid generation utilities support all strategies (uniform, log, Chebyshev, tanh, sinh)
- [ ] Three working presets (Fast, Balanced, Accurate)
- [ ] Adaptive Balanced achieves < 3% average error with < 20K points
- [ ] Accuracy benchmark shows clear tradeoff curve
- [ ] All tests pass
- [ ] Documentation complete with usage examples

## Open Questions

1. **Dividend dimension**: Should we also use non-uniform spacing for dividend? Probably not needed (low curvature).

2. **Coordinate transform interaction**: Does COORD_LOG_SQRT affect optimal spacing? May need to adjust concentration parameters.

3. **Cubic vs multilinear**: Does spacing strategy affect interpolation method choice? Cubic might benefit more from optimal spacing.

4. **Dynamic grid refinement**: Should we support runtime grid refinement based on error estimates? Out of scope for P5, consider for P6.

## References

- Expert guidance (issue #39): Concentrate points near ATM and short-T
- Chebyshev nodes: Optimal for polynomial interpolation
- Tanh concentration: Flexible, tunable concentration
- Current grid sizes: 112,500 points (30×25×15×10)
- Target: 10,000-20,000 points with equal or better accuracy
