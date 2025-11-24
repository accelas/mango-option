# Floating-Point Templating Design

**Date:** 2025-11-23
**Status:** Design Complete, Ready for Implementation
**Goal:** Template entire codebase on `std::floating_point` to support both `float` (fp32) and `double` (fp64)

## Overview

Convert all numeric types from hardcoded `double` to template parameter `T` constrained by `std::floating_point`. This enables:
- FP64 (double) for maximum accuracy (current default)
- FP32 (float) for performance when precision requirements allow
- Consistent type safety across the entire stack

## Design Decisions

### Scope
- **Everything templated** - All structs, solvers, utilities use template parameter `T`
- **Big bang migration** - Single comprehensive PR, all changes together
- **Bottom-up implementation** - Start with foundation types, work up to solvers

### Template Parameters
- **Naming:** Use `T` throughout (standard C++ convention)
- **No defaults:** Require explicit template arguments everywhere for clarity
- **No type aliases:** Use full template syntax `IVSolverFDM<double>` (no `IVSolverFDMd` shortcuts)

### Type System
- **All numeric members:** Every floating-point value becomes `T`
- **Counts stay integral:** `size_t iterations`, `bool converged` unchanged
- **Errors un-templated:** `IVError`, `ValidationError` remain concrete types (cast `T` to `double` for diagnostics)

### Testing
- **Double-only initially:** All tests use `<double>` explicitly
- **Float tests deferred:** Add spot-check float tests in future PR
- **Acceptance:** All 57 existing tests must pass with `<double>`

## Architecture

### Core Type System

```cpp
// Foundation structs (src/option/iv_result.hpp, option_spec.hpp)
template<std::floating_point T>
struct IVSuccess {
    T implied_vol;
    size_t iterations;
    T final_error;
    std::optional<T> vega;
};

template<std::floating_point T>
struct OptionSpec {
    T spot;
    T strike;
    T maturity;
    T rate;
    T dividend_yield;
    OptionType type;
};

template<std::floating_point T>
struct IVQuery : OptionSpec<T> {
    T market_price;
};

template<std::floating_point T>
struct PricingParams : OptionSpec<T> {
    T volatility;
};

// IVError remains un-templated (error codes don't need precision)
struct IVError {
    IVErrorCode code;
    size_t iterations = 0;
    double final_error = 0.0;  // Cast from T when creating
    std::optional<double> last_vol;
};
```

### Validation Layer

```cpp
// Templated validation functions (src/option/option_spec.hpp)
template<std::floating_point T>
std::expected<void, ValidationError> validate_option_spec(const OptionSpec<T>& spec);

template<std::floating_point T>
std::expected<void, ValidationError> validate_iv_query(const IVQuery<T>& query);

template<std::floating_point T>
std::expected<void, ValidationError> validate_pricing_params(const PricingParams<T>& params);

// Implementation: move to header (inline) or explicit instantiation in .cpp
template<std::floating_point T>
std::expected<void, ValidationError> validate_option_spec(const OptionSpec<T>& spec) {
    if (spec.spot <= T{0} || !std::isfinite(spec.spot)) {
        return std::unexpected(ValidationError(
            ValidationErrorCode::InvalidSpotPrice,
            static_cast<double>(spec.spot)));  // Cast to double for diagnostics
    }
    // ... rest of validation
}
```

**ValidationError design:**
- Keep `ValidationError::value` as `double` (avoids templating entire error system)
- Cast `T` to `double` when creating validation errors

**Implementation location:**
- Move implementations to header (inline/constexpr) OR
- Use explicit template instantiation for `float` and `double` in .cpp

### Math Utilities

**New file:** `src/support/math_utils.hpp`

```cpp
#pragma once
#include <cmath>
#include <concepts>

namespace mango {

template<std::floating_point T>
inline constexpr T abs(T x) noexcept { return std::abs(x); }

template<std::floating_point T>
inline T log(T x) noexcept { return std::log(x); }

template<std::floating_point T>
inline T exp(T x) noexcept { return std::exp(x); }

template<std::floating_point T>
inline T sqrt(T x) noexcept { return std::sqrt(x); }

template<std::floating_point T>
inline bool isfinite(T x) noexcept { return std::isfinite(x); }

template<std::floating_point T>
inline T max(T a, T b) noexcept { return std::max(a, b); }

template<std::floating_point T>
inline T min(T a, T b) noexcept { return std::min(a, b); }

} // namespace mango
```

**Usage:** Replace `std::log(spot)` with `mango::log(spot)` throughout codebase.

**Error Conversion:** `validation_error_to_iv_error()` remains un-templated (already converts to un-templated `IVError`).

### IV Solvers

```cpp
// src/option/iv_solver_fdm.hpp
template<std::floating_point T>
struct IVSolverFDMConfig {
    RootFindingConfig<T> root_config;
    size_t batch_parallel_threshold = 4;
    bool use_manual_grid = false;
    size_t grid_n_space = 101;
    size_t grid_n_time = 1000;
    T grid_x_min = T{-3.0};
    T grid_x_max = T{3.0};
    T grid_alpha = T{2.0};
};

template<std::floating_point T>
class IVSolverFDM {
public:
    explicit IVSolverFDM(const IVSolverFDMConfig<T>& config);

    std::expected<IVSuccess<T>, IVError> solve_impl(const IVQuery<T>& query) const;
    BatchIVResult<T> solve_batch_impl(const std::vector<IVQuery<T>>& queries) const;

private:
    IVSolverFDMConfig<T> config_;
    mutable std::optional<SolverError> last_solver_error_;

    T estimate_upper_bound(const IVQuery<T>& query) const;
    T estimate_lower_bound() const;
    T objective_function(const IVQuery<T>& query, T volatility) const;
    // ... other members
};

// src/option/iv_solver_interpolated.hpp
template<std::floating_point T>
struct IVSolverInterpolatedConfig {
    RootFindingConfig<T> newton_config;
    size_t batch_parallel_threshold = 4;
};

template<std::floating_point T>
class IVSolverInterpolated {
    // ... same templating pattern
};
```

### Dependent Components

**American Option Solver:**
```cpp
template<std::floating_point T>
class AmericanOptionSolver {
public:
    static std::expected<AmericanOptionSolver<T>, ValidationError>
        create(const PricingParams<T>& params,
               PDEWorkspace<T> workspace, ...);

    std::expected<AmericanOptionResult<T>, SolverError> solve();
private:
    PricingParams<T> params_;
    // ...
};

template<std::floating_point T>
struct AmericanOptionResult {
    T value_at(T spot) const;
    std::shared_ptr<const SpatialGrid<T>> grid() const;
    // ...
};
```

**Root Finding:**
```cpp
template<std::floating_point T>
struct RootFindingConfig {
    size_t max_iter = 100;
    T tolerance = T{1e-6};
    T min_step = T{1e-10};
};

template<std::floating_point T>
struct RootFindingSuccess {
    T root;
    size_t iterations;
    T final_error;
};

template<std::floating_point T, typename Func>
std::expected<RootFindingSuccess<T>, RootFindingError>
brent_find_root(Func&& f, T a, T b, const RootFindingConfig<T>& config);
```

**Grid/PDE infrastructure:** Already templated - verify they work with both float and double.

### Price Table System

```cpp
// price_table_grid.hpp
template<std::floating_point T>
struct PriceTableGrid {
    std::vector<T> moneyness;
    std::vector<T> maturity;
    std::vector<T> volatility;
    std::vector<T> rate;
};

// price_table_4d_builder.hpp
template<std::floating_point T>
class PriceTable4DBuilder {
public:
    static std::expected<PriceTable4DBuilder<T>, std::string> create(
        std::vector<T> moneyness,
        std::vector<T> maturity,
        std::vector<T> volatility,
        std::vector<T> rate,
        T K_ref);

    void precompute(OptionType type, size_t n_space, size_t n_time);
    PriceTableSurface<T> get_surface() const;
private:
    PriceTableGrid<T> grid_;
    T K_ref_;
    std::vector<T> prices_4d_;
};

// bspline_price_table.hpp
template<std::floating_point T>
class BSpline4D {
public:
    static std::expected<BSpline4D<T>, std::string> create(
        const PriceTableWorkspace<T>& workspace);

    T eval(T m, T tau, T sigma, T r) const;
private:
    BSplineND<T, 4> spline_;
};

template<std::floating_point T>
struct PriceTableSurface {
    std::shared_ptr<const BSpline4D<T>> spline;
    T K_ref;
    // ...
};
```

**Note:** All B-spline infrastructure (`BSplineND<T, N>`) already templated - just verify compatibility.

## Implementation Plan

### Phase 1: Foundation (5 files)
1. `src/support/math_utils.hpp` - Create math wrappers
2. `src/option/iv_result.hpp` - Template IVSuccess, BatchIVResult
3. `src/option/option_spec.hpp` - Template OptionSpec, IVQuery, PricingParams
4. `src/option/option_spec.cpp` - Template validation functions (move to header or explicit instantiation)
5. `src/support/error_types.hpp` - Review, likely no changes needed

### Phase 2: Math/Grid Layer (3 files)
6. `src/math/root_finding.hpp` - Template RootFindingConfig, brent_find_root
7. `src/math/root_finding.cpp` - Move to header or explicit instantiation
8. Verify `src/pde/core/grid.hpp` and related already work with float

### Phase 3: American Option (4 files)
9. `src/option/american_option.hpp` - Template AmericanOptionSolver
10. `src/option/american_option.cpp` - Template implementations
11. `src/option/american_option_result.hpp` - Template AmericanOptionResult
12. `src/option/american_pde_solver.hpp` - Template PDE solver wrappers

### Phase 4: Price Table System (6 files)
13. `src/option/price_table_grid.hpp` - Template PriceTableGrid
14. `src/option/price_table_4d_builder.hpp` - Template builder
15. `src/option/bspline_price_table.hpp` - Template BSpline4D, PriceTableSurface
16. `src/option/price_table_extraction.hpp` - Template extraction functions
17. Verify `src/math/bspline_nd.hpp` works with float
18. Update workspace/metadata files

### Phase 5: IV Solvers (4 files)
19. `src/option/iv_solver_fdm.hpp` - Template IVSolverFDM
20. `src/option/iv_solver_fdm.cpp` - Template implementations
21. `src/option/iv_solver_interpolated.hpp` - Template IVSolverInterpolated
22. `src/option/iv_solver_interpolated.cpp` - Template implementations

### Phase 6: Tests (all test files)
23. Update all tests to use explicit `<double>` template arguments
24. Fix compilation errors
25. Verify all tests pass

## Testing Strategy

### Test Updates

All existing tests updated to use explicit `<double>` template arguments:

```cpp
// Before
IVSolverFDM solver(config);
IVQuery query{.option = spec, .market_price = 10.45};

// After
IVSolverFDM<double> solver(config);
IVQuery<double> query{.option = spec, .market_price = 10.45};
```

### Compilation Verification

```bash
# Verify both types compile
bazel build //src/option:iv_solver_fdm
bazel build //src/option:iv_solver_interpolated

# Run full test suite with double
bazel test //...
```

### Future Float Testing (Phase 7, not in this PR)

Later, add spot-check tests for float:
- One basic IV solver test with `<float>`
- One price table test with `<float>`
- Verify numerical stability differences are acceptable

### Documentation Updates

Update `docs/API_GUIDE.md` to show template usage:

```cpp
// FP64 (double) - default for accuracy
IVSolverFDM<double> solver_fp64(config);

// FP32 (float) - for performance when precision allows
IVSolverFDM<float> solver_fp32(config);
```

### Acceptance Criteria

- ✅ All 57 tests pass with `<double>`
- ✅ Code compiles without warnings
- ✅ No performance regression for double version
- ✅ API documentation updated

## Migration Examples

### Before (current code)

```cpp
// Option specification
OptionSpec spec{
    .spot = 100.0,
    .strike = 100.0,
    .maturity = 1.0,
    .rate = 0.05,
    .dividend_yield = 0.02,
    .type = OptionType::PUT
};

// IV query
IVQuery query{.option = spec, .market_price = 10.45};

// Solver
IVSolverFDMConfig config{
    .root_config = {.max_iter = 100, .tolerance = 1e-6}
};
IVSolverFDM solver(config);

// Solve
auto result = solver.solve_impl(query);
if (result.has_value()) {
    std::cout << "IV: " << result->implied_vol << "\n";
}
```

### After (templated code)

```cpp
// Option specification - explicit template argument
OptionSpec<double> spec{
    .spot = 100.0,
    .strike = 100.0,
    .maturity = 1.0,
    .rate = 0.05,
    .dividend_yield = 0.02,
    .type = OptionType::PUT
};

// IV query
IVQuery<double> query{.option = spec, .market_price = 10.45};

// Solver
IVSolverFDMConfig<double> config{
    .root_config = {.max_iter = 100, .tolerance = 1e-6}
};
IVSolverFDM<double> solver(config);

// Solve
auto result = solver.solve_impl(query);
if (result.has_value()) {
    std::cout << "IV: " << result->implied_vol << "\n";
}
```

### Using float for performance

```cpp
// Same API, just change template argument to float
OptionSpec<float> spec{
    .spot = 100.0f,
    .strike = 100.0f,
    .maturity = 1.0f,
    .rate = 0.05f,
    .dividend_yield = 0.02f,
    .type = OptionType::PUT
};

IVQuery<float> query{.option = spec, .market_price = 10.45f};
IVSolverFDMConfig<float> config{
    .root_config = {.max_iter = 100, .tolerance = 1e-6f}
};
IVSolverFDM<float> solver(config);

auto result = solver.solve_impl(query);
```

## Rationale

### Why template everything?

**Consistency:** Avoids type conversions and mixed-precision bugs
**Performance:** Enables full fp32 pipeline for GPU/SIMD in future
**Flexibility:** Users choose precision vs performance trade-off
**Type safety:** Compiler prevents accidental precision loss

### Why no defaults?

**Clarity:** Explicit template arguments prevent accidental type mismatches
**Intent:** Forces developer to consider precision requirements
**Maintainability:** Easier to search/replace when types are explicit

### Why template wrappers for math?

**Overload resolution:** Ensures correct `std::log(float)` vs `std::log(double)`
**Future-proofing:** Can add custom implementations (e.g., approximations)
**Consistency:** Single namespace for all math operations

### Why un-templated errors?

**Simplicity:** Error codes and diagnostics don't need precision
**Interop:** Common error types work with both `float` and `double` solvers
**Size:** Avoids code bloat from templating error handling infrastructure

## Risks and Mitigations

### Risk: Header-only bloat
**Impact:** Templated code must be in headers, increases compilation time
**Mitigation:** Use explicit template instantiation where possible (e.g., validate functions)

### Risk: Numerical stability with float
**Impact:** Some algorithms may fail with fp32 precision
**Mitigation:** Test with double first, add float tests incrementally, document precision requirements

### Risk: Breaking existing code
**Impact:** All code must add explicit template arguments
**Mitigation:** Big bang migration in single PR, update all call sites together

### Risk: Increased binary size
**Impact:** Instantiating templates for both float and double increases binary size
**Mitigation:** Most code will use only double initially, float instantiation on-demand

## Future Work (not in this design)

- **GPU support:** fp32-templated code enables easier CUDA/HIP porting
- **Mixed precision:** Use fp32 for bulk computation, fp64 for critical paths
- **SIMD optimization:** Template parameter enables vectorization hints
- **Benchmarking:** Compare float vs double performance across workloads
- **Automatic precision selection:** Runtime/compile-time precision tuning

## References

- C++20 `std::floating_point` concept: https://en.cppreference.com/w/cpp/concepts/floating_point
- Explicit template instantiation: https://en.cppreference.com/w/cpp/language/template_specialization
- Mixed-precision computing: https://developer.nvidia.com/blog/mixed-precision-programming-cuda-8/
