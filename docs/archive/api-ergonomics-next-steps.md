# API Ergonomics: Next Steps

## Current Branch Status

Branch: `feature/market-iv-e2e-benchmark`

**Uncommitted changes**: Major API improvements to price table and IV solver interfaces

### Changes Summary

#### New Types Added

1. **`PriceTableGrid`** - Consolidates grid vectors and metadata
   ```cpp
   struct PriceTableGrid {
       std::vector<double> moneyness;
       std::vector<double> maturity;
       std::vector<double> volatility;
       std::vector<double> rate;
       double K_ref = 0.0;
   };
   ```

2. **`PriceTableConfig`** - Consolidates PDE solver configuration
   ```cpp
   struct PriceTableConfig {
       OptionType option_type = OptionType::PUT;
       size_t n_space = 101;
       size_t n_time = 1000;
       double dividend_yield = 0.0;
       std::optional<std::pair<double, double>> x_bounds;
   };
   ```

3. **`PriceTableSurface`** - User-friendly wrapper with automatic bounds
   ```cpp
   class PriceTableSurface {
       double eval(double m, double tau, double sigma, double rate) const;
       double K_ref() const;
       std::pair<double, double> moneyness_range() const;
       std::pair<double, double> maturity_range() const;
       std::pair<double, double> volatility_range() const;
       std::pair<double, double> rate_range() const;
       // ... metadata accessors
   };
   ```

#### API Improvements

**Before (verbose, many parameters):**
```cpp
// Old API - 5 parameters plus K_ref
auto builder = PriceTable4DBuilder::create(
    moneyness, maturities, volatilities, rates, K_ref);

// Old API - 4 parameters
auto result = builder.precompute(OptionType::PUT, 51, 500, dividend);

// Old API - 7 parameters!
IVSolverInterpolated iv_solver(
    *result.evaluator,  // Dereference unique_ptr
    K_ref,
    m_range,
    tau_range,
    vol_range,
    rate_range,
    config
);
```

**After (clean, structured):**
```cpp
// New API - single grid struct
PriceTableGrid grid{
    .moneyness = {0.8, ..., 1.2},
    .maturity = {0.1, ..., 2.0},
    .volatility = {0.15, ..., 0.40},
    .rate = {0.02, ..., 0.05},
    .K_ref = 100.0
};
auto builder = PriceTable4DBuilder::create(grid);

// New API - single config struct
PriceTableConfig config{
    .option_type = OptionType::PUT,
    .n_space = 51,
    .n_time = 500,
    .dividend_yield = 0.015
};
auto result = builder.precompute(config);

// New API - single surface object (auto-extracts bounds)
IVSolverInterpolated iv_solver(result.surface);
```

## Migration Tasks

### 1. Update Existing Benchmarks (High Priority)

**Files to migrate:**
- `benchmarks/component_performance.cc` (lines 329-334)
- `benchmarks/readme_benchmarks.cc` (lines 119-124)

**Current pattern:**
```cpp
IVSolverInterpolated solver(
    *surf.evaluator,  // OLD: manual dereference
    surf.K_ref,
    {surf.m_grid.front(), surf.m_grid.back()},
    {surf.tau_grid.front(), surf.tau_grid.back()},
    {surf.sigma_grid.front(), surf.sigma_grid.back()},
    {surf.rate_grid.front(), surf.rate_grid.back()}
);
```

**Migration path:**
```cpp
// Option A: Wrap existing fixture in PriceTableSurface
PriceTableGrid grid{
    .moneyness = surf.m_grid,
    .maturity = surf.tau_grid,
    .volatility = surf.sigma_grid,
    .rate = surf.rate_grid,
    .K_ref = surf.K_ref
};

auto spline_result = BSpline4D::create(*surf.evaluator);
if (!spline_result.has_value()) {
    // handle error
}
PriceTableSurface surface(
    std::make_shared<BSpline4D>(std::move(spline_result.value())),  // Share ownership
    std::move(grid),
    0.0  // dividend
);

IVSolverInterpolated solver(surface);  // Clean!

// Option B: Update fixtures to use PriceTable4DBuilder
// (More invasive but cleaner long-term)
```

### 2. Higher-Level Builders (Medium Priority)

Add convenience builders that work with raw market data:

#### 2a. Builder from Strike List

```cpp
class PriceTable4DBuilder {
public:
    /// Create builder from strike prices (auto-computes moneyness)
    ///
    /// @param spot Current underlying price
    /// @param strikes Strike prices (sorted)
    /// @param maturities Time to expiration (years)
    /// @param volatilities Volatility grid
    /// @param rates Rate grid
    /// @return Builder ready for precomputation
    static PriceTable4DBuilder from_strikes(
        double spot,
        std::vector<double> strikes,
        std::vector<double> maturities,
        std::vector<double> volatilities,
        std::vector<double> rates)
    {
        // Auto-compute moneyness: m = spot / strike
        std::vector<double> moneyness;
        moneyness.reserve(strikes.size());
        for (double K : strikes) {
            moneyness.push_back(spot / K);
        }

        // Use ATM strike as reference
        auto atm_it = std::lower_bound(strikes.begin(), strikes.end(), spot);
        double K_ref = (atm_it != strikes.end()) ? *atm_it : strikes[strikes.size()/2];

        PriceTableGrid grid{
            .moneyness = std::move(moneyness),
            .maturity = std::move(maturities),
            .volatility = std::move(volatilities),
            .rate = std::move(rates),
            .K_ref = K_ref
        };

        return create(std::move(grid));
    }
};
```

**Usage:**
```cpp
// Much simpler for users with raw market data!
auto builder = PriceTable4DBuilder::from_strikes(
    spot,
    {90, 95, 100, 105, 110},  // Strikes (user-friendly)
    {0.1, 0.25, 0.5, 1.0},
    {0.15, 0.20, 0.25, 0.30},
    {0.03, 0.04, 0.05}
);
```

#### 2b. Builder from Market Chain Data

```cpp
/// Market option chain data (from exchanges)
struct OptionChain {
    std::string ticker;
    double spot;
    std::vector<double> strikes;
    std::vector<double> maturities;
    std::vector<double> implied_vols;  // Market IVs (for grid)
    std::vector<double> rates;
    double dividend_yield;
};

class PriceTable4DBuilder {
public:
    /// Create builder from market option chain
    static PriceTable4DBuilder from_chain(const OptionChain& chain) {
        // Extract unique strikes, maturities, IVs for grids
        auto strikes = unique_sorted(chain.strikes);
        auto maturities = unique_sorted(chain.maturities);
        auto vols = unique_sorted(chain.implied_vols);
        auto rates = unique_sorted(chain.rates);

        return from_strikes(
            chain.spot, std::move(strikes), std::move(maturities),
            std::move(vols), std::move(rates));
    }
};
```

**Usage:**
```cpp
// Direct from market data!
OptionChain spy_chain = fetch_from_exchange("SPY");
auto builder = PriceTable4DBuilder::from_chain(spy_chain);
auto surface = builder.precompute(config).value().surface;
```

### 3. Test Coverage (High Priority)

Add tests for new API:
- `tests/price_table_surface_test.cc` - Test `PriceTableSurface` interface
- `tests/price_table_grid_test.cc` - Test grid validation
- Update `tests/price_table_4d_integration_test.cc` to use new API

### 4. Documentation Updates (Medium Priority)

- Update `CLAUDE.md` API examples to use new structs
- Add `docs/api-migration-guide.md` for users upgrading
- Update README examples

## Rationale for Changes

### Problem: API Verbosity

**Old code required:**
- 5 vectors passed separately to builder
- Manual bounds extraction: `{grid.front(), grid.back()}` × 4
- Pointer dereferencing: `*evaluator`
- 7+ parameters to create IV solver

**Pain points:**
1. Easy to get parameter order wrong
2. Tedious bounds extraction boilerplate
3. Confusing unique_ptr dereferencing
4. Hard to discover what parameters are needed

### Solution: Structured Types

**Benefits:**
1. **Designated initializers** make intent clear
2. **Single surface object** bundles data + metadata
3. **Automatic bounds extraction** eliminates boilerplate
4. **Compiler-checked** parameter names (not positional)
5. **Discoverable** via IDE autocomplete

**Example comparison:**

```cpp
// Old: What order? What are these numbers?
builder.precompute(OptionType::PUT, 51, 500, 0.015);

// New: Crystal clear!
PriceTableConfig{
    .option_type = OptionType::PUT,
    .n_space = 51,      // Spatial grid points
    .n_time = 500,      // Time steps
    .dividend_yield = 0.015
};
```

## Next Steps (Priority Order)

### Immediate (Before PR Merge)

1. ✅ Review API changes (done - documented here)
2. ⏳ Migrate `component_performance.cc` to new API
3. ⏳ Migrate `readme_benchmarks.cc` to new API
4. ⏳ Test all benchmarks still compile and run
5. ⏳ Update commit message to include API improvements

### After PR Merge

1. Add `from_strikes()` builder (1-2 hours)
2. Add `from_chain()` builder (2-3 hours)
3. Write migration guide for existing users
4. Add comprehensive test coverage for new API

## Open Questions

1. **Should `PriceTableSurface` be copyable?**
   - Currently uses `shared_ptr<BSpline4D>` for sharing
   - Copying is cheap (just shared_ptr copy)
   - Seems reasonable for value semantics

2. **Should we deprecate old API?**
   - Keep for backward compatibility?
   - Or force migration with breaking change?

3. **Naming: `PriceTableGrid` vs `PriceTableGrids`?**
   - Current: `PriceTableGrid` (singular)
   - Alternative: `PriceTableGrids` (plural, emphasizes multiple axes)

## Files Modified (Uncommitted)

- `src/option/price_table_4d_builder.hpp` (+93 lines)
- `src/option/price_table_4d_builder.cpp` (+31 lines)
- `src/option/iv_solver_interpolated.hpp` (+12 lines)
- `src/option/iv_solver_interpolated.cpp` (+29 lines)
- `benchmarks/market_iv_e2e_benchmark.cc` (refactored to new API)
- `benchmarks/README_MARKET_IV_E2E.md` (updated examples)
- `src/option/BUILD.bazel` (+1 dependency)
