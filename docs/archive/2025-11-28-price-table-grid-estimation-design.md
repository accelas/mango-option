# Price Table Grid Estimation Design

## Problem

The `PriceTableBuilder::from_chain()` requires users to manually specify grids for each dimension (moneyness, maturity, volatility, rate). There's no automatic way to determine grid density based on desired accuracy.

Current behavior:
- User passes `OptionChain` with explicit `maturities`, `implied_vols`, `rates` vectors
- No guidance on how many points are needed for a given accuracy target
- Benchmark shows coarse grids (5×6×4) give ~10 bps error; denser grids (13×18×8) give ~4 bps

## Solution

Add an automatic grid estimation mechanism similar to `estimate_grid_for_option()` in the PDE solver. The estimator takes:
- Domain bounds (min/max for each dimension)
- Target IV error (e.g., 1 bps = 0.0001)
- Returns: recommended grid vectors for each dimension

## Theoretical Basis

### B-Spline Interpolation Error

For cubic B-spline collocation, the interpolation error at points between grid nodes is:

```
E(x) = O(h⁴ · f⁽⁴⁾)
```

Where:
- h = grid spacing
- f⁽⁴⁾ = 4th derivative of the function

For 4D tensor product B-splines on price surface P(m, τ, σ, r):

```
E ≤ C₁·h_m⁴·∂⁴P/∂m⁴ + C₂·h_τ⁴·∂⁴P/∂τ⁴ + C₃·h_σ⁴·∂⁴P/∂σ⁴ + C₄·h_r⁴·∂⁴P/∂r⁴
```

### Curvature Analysis by Dimension

| Dimension | Curvature | Rationale |
|-----------|-----------|-----------|
| σ (vol) | HIGH | Vega non-linearity, especially at low vol |
| m (moneyness) | HIGH near ATM | Gamma peak at ATM, lower in wings |
| τ (maturity) | MEDIUM | √τ behavior, spike near τ→0 |
| r (rate) | LOW | Nearly linear discounting effect |

### Budget Allocation Weights

Based on curvature analysis, allocate grid points with these relative weights:

| Dimension | Weight | Reasoning |
|-----------|--------|-----------|
| σ (vol) | 1.5 | Highest curvature, most sensitive |
| m (moneyness) | 1.2 | High near ATM (sinh spacing helps) |
| τ (maturity) | 1.0 | Baseline, moderate curvature |
| r (rate) | 0.6 | Nearly linear, low curvature |

## API Design

### Configuration Struct

```cpp
/// Grid estimation accuracy parameters for price table
struct PriceTableGridAccuracyParams {
    /// Target IV error in absolute terms (default: 10 bps = 0.001)
    double target_iv_error = 0.001;

    /// Minimum points per dimension (B-spline requires ≥4)
    size_t min_points = 4;

    /// Maximum points per dimension (cost control)
    size_t max_points = 50;

    /// Curvature weights for budget allocation [m, τ, σ, r]
    std::array<double, 4> curvature_weights = {1.2, 1.0, 1.5, 0.6};
};
```

### Estimation Result

```cpp
/// Result of grid estimation
struct PriceTableGridEstimate {
    std::vector<double> moneyness_grid;
    std::vector<double> maturity_grid;
    std::vector<double> volatility_grid;
    std::vector<double> rate_grid;
    size_t estimated_pde_solves;  // n_vol × n_rate
};
```

### Main Function

```cpp
/// Estimate optimal grid for price table based on target accuracy
///
/// @param m_min, m_max  Moneyness range (e.g., 0.8 to 1.2)
/// @param tau_min, tau_max  Maturity range in years (e.g., 0.01 to 2.0)
/// @param sigma_min, sigma_max  Volatility range (e.g., 0.05 to 0.50)
/// @param r_min, r_max  Rate range (e.g., 0.01 to 0.06)
/// @param accuracy  Accuracy parameters
/// @return Grid estimate with recommended vectors
PriceTableGridEstimate estimate_grid_for_price_table(
    double m_min, double m_max,
    double tau_min, double tau_max,
    double sigma_min, double sigma_max,
    double r_min, double r_max,
    const PriceTableGridAccuracyParams& accuracy = {});
```

### Factory Method with Auto-Estimation

```cpp
/// Factory from option chain with automatic grid estimation
///
/// @param chain Option chain (uses spot, strikes for moneyness bounds)
/// @param grid_spec PDE spatial grid specification
/// @param n_time Number of time steps
/// @param type Option type (PUT or CALL)
/// @param accuracy Grid accuracy parameters
/// @return Pair of (builder, axes) or error
static std::expected<std::pair<PriceTableBuilder<4>, PriceTableAxes<4>>, PriceTableError>
from_chain_auto(
    const OptionChain& chain,
    GridSpec<double> grid_spec,
    size_t n_time,
    OptionType type = OptionType::PUT,
    const PriceTableGridAccuracyParams& accuracy = {});
```

## Algorithm

### Step 1: Calculate Base Points from Target Error

Using the h⁴ error relationship:
```cpp
// For target error ε and domain width W:
// ε ≈ C · (W/n)⁴ · f⁽⁴⁾
// Solving for n: n ≈ W · (C · f⁽⁴⁾ / ε)^(1/4)

double base_points = std::pow(scale_factor / target_iv_error, 0.25);
```

The `scale_factor` is calibrated empirically from benchmark data:
- 13×18×8 grid → 4.3 bps error
- Need ~4× points per dim for ~1 bps → ~52×72×32

### Step 2: Apply Curvature Weights

```cpp
size_t n_m = clamp(base_points * weights[0], min_points, max_points);
size_t n_tau = clamp(base_points * weights[1], min_points, max_points);
size_t n_sigma = clamp(base_points * weights[2], min_points, max_points);
size_t n_rate = clamp(base_points * weights[3], min_points, max_points);
```

### Step 3: Generate Grid Vectors

- **Moneyness:** Uniform in log-space (matches internal storage as log-moneyness)
  - User specifies m_min=0.8, m_max=1.2
  - Generate uniform grid in [ln(0.8), ln(1.2)]
  - Convert back to moneyness for API: exp(ln_grid)
  - This matches the internal `PriceTableSurface` which transforms m → ln(m)
- **Maturity:** Concentration near short maturities (sqrt spacing)
- **Volatility:** Uniform spacing (highest curvature dimension)
- **Rate:** Uniform spacing (lowest curvature)

## Validation

Add benchmark `BM_RealData_GridEstimator` that:
1. Calls `estimate_grid_for_price_table()` with target error
2. Builds price table with estimated grids
3. Measures actual IV error vs FDM
4. Verifies actual error ≤ 2× target (allow safety margin)

## Cost Analysis

For target 1 bps error with default weights:
- Estimated: ~20-25 vol points, ~15-20 maturity, ~12-15 rate
- PDE solves: ~20 × 15 = 300-500
- Build time: ~200-500ms (parallel)
- Query time: unchanged (~100μs per IV)

## Files Changed

1. `src/option/table/price_table_grid_estimator.hpp` - New file
2. `src/option/table/price_table_builder.hpp` - Add `from_chain_auto()`
3. `src/option/table/price_table_builder.cpp` - Implement `from_chain_auto()`
4. `src/option/table/BUILD.bazel` - Add new header
5. `benchmarks/real_data_benchmark.cc` - Add validation benchmark
6. `tests/price_table_grid_estimator_test.cc` - Unit tests
