# Floating-Point Templating Design

**Date:** 2025-11-23
**Status:** Design Under Revision - Addressing Code Review Feedback
**Goal:** Template entire codebase on `std::floating_point` concept to support both `float` (fp32) and `double` (fp64)

## Overview

Convert all numeric types from hardcoded `double` to template parameter `T` constrained by `std::floating_point`. This enables:
- FP64 (double) for maximum accuracy (current default)
- FP32 (float) for performance when precision requirements allow
- Consistent type safety across the entire stack

## Design Decisions (Revised after Code Review)

### Scope
- **Everything templated** - All structs, solvers, utilities use template parameter `T`
- **Phased migration** - Layer-by-layer implementation with tests after each phase
- **Bottom-up implementation** - Start with foundation types, work up to solvers

### Template Parameters
- **Naming:** Use `T` throughout (standard C++ convention)
- **Default to double:** Use `template<std::floating_point T = double>` to minimize breaking changes
- **Provide type aliases:** Add `using IVSolverFDMf = IVSolverFDM<float>; using IVSolverFDMd = IVSolverFDM<double>;` for convenience
- **Standard concept:** Use `std::floating_point` from C++20 `<concepts>` header

### Type System
- **All numeric members:** Every floating-point value becomes `T`
- **Counts stay integral:** `size_t iterations`, `bool converged` unchanged
- **Errors templated:** `IVError<T>` and `ValidationError<T>` preserve full diagnostic precision
- **Type-specific tolerances:** Scale epsilon values based on `std::numeric_limits<T>::epsilon()`

**Rationale for templated errors:** Error types store diagnostic values (e.g., `final_error`, `value`) from computations. Templating them on `T` prevents precision loss when `T` is `long double` or preserves efficiency when `T` is `float`.

### Testing
- **Explicit instantiation tests:** Compile-time verification for both `<float>` and `<double>`
- **Runtime tests for double:** All 57 existing tests pass with `<double>`
- **Runtime tests for float:** At least one test per major component verifies `<float>` works
- **Acceptance:** Both precisions compile and core tests pass

### Implementation Strategy
- **Header-based templates:** Move template implementations to headers (`.ipp` files included at end)
- **Explicit instantiation units:** Provide `.cpp` files with explicit instantiations for faster compilation
- **No ADL-breaking wrappers:** Use qualified calls or local `using` declarations for math functions

## Architecture

### Core Type System

```cpp
// Foundation structs (src/option/iv_result.hpp, option_spec.hpp)
template<std::floating_point T = double>
struct IVSuccess {
    T implied_vol;
    size_t iterations;
    T final_error;
    std::optional<T> vega;
};

// Type aliases for convenience
using IVSuccessf = IVSuccess<float>;
using IVSuccessd = IVSuccess<double>;

template<std::floating_point T = double>
struct OptionSpec {
    T spot;
    T strike;
    T maturity;
    T rate;
    T dividend_yield;
    OptionType type;
};

template<std::floating_point T = double>
struct IVQuery : OptionSpec<T> {
    T market_price;
};

template<std::floating_point T = double>
struct PricingParams : OptionSpec<T> {
    T volatility;
};

// IVError is templated to preserve full diagnostic precision
template<std::floating_point T = double>
struct IVError {
    IVErrorCode code;
    size_t iterations = 0;
    T final_error = T{0};
    std::optional<T> last_vol;
};

// Type aliases for convenience
using IVErrorf = IVError<float>;
using IVErrord = IVError<double>;
```

### Validation Layer

```cpp
// Templated validation functions (src/option/option_spec.hpp)
template<std::floating_point T = double>
std::expected<void, ValidationError<T>> validate_option_spec(const OptionSpec<T>& spec);

template<std::floating_point T = double>
std::expected<void, ValidationError<T>> validate_iv_query(const IVQuery<T>& query);

template<std::floating_point T = double>
std::expected<void, ValidationError<T>> validate_pricing_params(const PricingParams<T>& params);
```

**ValidationError is templated to preserve full diagnostic precision:**
```cpp
template<std::floating_point T = double>
struct ValidationError {
    ValidationErrorCode code;
    T value = T{0};  // Preserves exact precision for diagnostics
    size_t index = 0;

    ValidationError(ValidationErrorCode code, T value = T{0}, size_t index = 0)
        : code(code), value(value), index(index) {}
};

// Type aliases for convenience
using ValidationErrorf = ValidationError<float>;
using ValidationErrord = ValidationError<double>;
```

**Implementation location (DECIDED):**
- **Header-based:** Move implementations to `option_spec.ipp` included at end of `option_spec.hpp`
- **Explicit instantiation:** Provide `option_spec_instantiations.cpp` with:
  ```cpp
  template std::expected<void, ValidationError<float>> validate_option_spec(const OptionSpec<float>&);
  template std::expected<void, ValidationError<double>> validate_option_spec(const OptionSpec<double>&);
  ```
- This gives both fast recompilation (if using precompiled `.cpp`) and correctness (inline available for other types)

### Math Utilities

**Decision: No custom wrappers - use standard library directly**

After code review, custom math wrappers would:
- Break ADL (prevent vendor-specific optimizations)
- Add unnecessary `constexpr` (std::abs not constexpr until C++23)
- Require extra includes (`<algorithm>` for min/max)
- Provide no actual value

**Instead:** Use qualified std:: calls directly throughout code:
```cpp
// Direct usage - no wrappers needed
if (!std::isfinite(spec.spot)) { ... }
T result = std::log(spot);
T maximum = std::max(a, b);  // Requires <algorithm>
```

The compiler automatically selects the correct overload (`std::log(float)` vs `std::log(double)`) based on argument type.

**Error Conversion:** `validation_error_to_iv_error()` is templated:

```cpp
// src/option/iv_result.hpp
template<std::floating_point T = double>
inline IVError<T> validation_error_to_iv_error(const ValidationError<T>& ve) {
    IVErrorCode code;
    switch (ve.code) {
        case ValidationErrorCode::InvalidSpotPrice:
            code = IVErrorCode::NegativeSpot;
            break;
        case ValidationErrorCode::InvalidStrike:
            code = IVErrorCode::NegativeStrike;
            break;
        // ... other cases
    }

    return IVError<T>{
        .code = code,
        .iterations = 0,
        .final_error = ve.value,  // Preserves exact T precision
        .last_vol = std::nullopt
    };
}
```

### Type-Specific Tolerances and Numeric Constants

**Critical Issue:** Hardcoded tolerances like `1e-10` are below float machine epsilon (~1e-7), causing them to collapse to zero in fp32 and disabling regularization.

**Solution:** Provide epsilon-aware helpers and type-specific defaults:

```cpp
// src/support/numeric_helpers.hpp
#pragma once
#include <concepts>
#include <limits>
#include <cmath>

namespace mango {

/// Compute default tolerance scaled by machine epsilon
template<std::floating_point T>
constexpr T default_tolerance() {
    constexpr T eps = std::numeric_limits<T>::epsilon();
    // sqrt(eps) is commonly used for finite difference tolerances
    // For float: sqrt(1e-7) ≈ 3e-4
    // For double: sqrt(1e-16) ≈ 1e-8
    return std::sqrt(eps) * T{100};  // Conservative scaling
}

/// Compute minimum step size for finite differences
template<std::floating_point T>
constexpr T default_min_step() {
    constexpr T eps = std::numeric_limits<T>::epsilon();
    // For float: 1e-7 * 100 = 1e-5
    // For double: 1e-16 * 100 = 1e-14
    return eps * T{100};
}

/// Check if value is effectively zero (within machine precision)
template<std::floating_point T>
constexpr bool is_effectively_zero(T value, T tolerance = default_tolerance<T>()) {
    return std::abs(value) < tolerance;
}

}  // namespace mango
```

**Usage in config structs:**

```cpp
// src/math/root_finding.hpp
template<std::floating_point T = double>
struct RootFindingConfig {
    int max_iter = 100;
    T tolerance = default_tolerance<T>();     // Type-aware default
    T min_step = default_min_step<T>();       // Type-aware default
};
```

**Updated config defaults for solvers:**

```cpp
// src/option/iv_solver_fdm.hpp
template<std::floating_point T = double>
struct IVSolverFDMConfig {
    RootFindingConfig<T> root_config{
        .max_iter = 100,
        .tolerance = default_tolerance<T>(),    // ~3e-4 for float, ~1e-8 for double
        .min_step = default_min_step<T>()       // ~1e-5 for float, ~1e-14 for double
    };
    size_t batch_parallel_threshold = 4;
    bool use_manual_grid = false;
    size_t grid_n_space = 101;
    size_t grid_n_time = 1000;
    T grid_x_min = T{-3.0};
    T grid_x_max = T{3.0};
    T grid_alpha = T{2.0};
};

// src/option/iv_solver_interpolated.hpp
template<std::floating_point T = double>
struct IVSolverInterpolatedConfig {
    int max_iterations = 50;
    T tolerance = default_tolerance<T>();      // Type-aware default
    T sigma_min = T{0.01};                     // 1% volatility minimum
    T sigma_max = T{3.0};                      // 300% volatility maximum
};
```

**Grid parameter adjustments for float:**

While spatial grid bounds (`x_min`, `x_max`) are dimensionless and work across precisions, time step selection may need adjustment:

```cpp
// In estimate_grid_for_option() or similar
template<std::floating_point T>
auto estimate_grid_for_option(const PricingParams<T>& params) {
    // For float, we may need fewer time steps since we can't resolve
    // differences smaller than ~1e-7
    const size_t base_n_time = std::is_same_v<T, float> ? 500 : 1000;

    // Scale based on maturity (standard CFL-like heuristic)
    const size_t n_time = static_cast<size_t>(
        base_n_time * std::sqrt(params.maturity)
    );

    // Grid bounds remain unchanged (dimensionless)
    return std::pair{grid_spec, n_time};
}
```

**Numeric stability validation:**

Add compile-time checks to ensure tolerances are above machine epsilon:

```cpp
// In RootFindingConfig constructor or validation
template<std::floating_point T>
struct RootFindingConfig {
    RootFindingConfig() {
        constexpr T eps = std::numeric_limits<T>::epsilon();
        static_assert(default_tolerance<T>() > eps * T{10},
                      "Default tolerance must be well above machine epsilon");
    }

    // ... members
};
```

**Risk Mitigation:**

1. **Validation on construction:** Config structs validate that user-provided tolerances are above `10 * epsilon`
2. **Documentation:** Add warnings in API docs about tolerance requirements for float
3. **Testing:** Explicit float tests verify convergence with scaled tolerances
4. **Diagnostics:** USDT probes report tolerance values for debugging precision issues

### IV Solvers

```cpp
// src/option/iv_solver_fdm.hpp
template<std::floating_point T = double>
class IVSolverFDM {
public:
    explicit IVSolverFDM(const IVSolverFDMConfig<T>& config);

    std::expected<IVSuccess<T>, IVError<T>> solve_impl(const IVQuery<T>& query) const;
    BatchIVResult<T> solve_batch_impl(const std::vector<IVQuery<T>>& queries) const;

private:
    IVSolverFDMConfig<T> config_;
    mutable std::optional<SolverError> last_solver_error_;

    T estimate_upper_bound(const IVQuery<T>& query) const;
    T estimate_lower_bound() const;
    T objective_function(const IVQuery<T>& query, T volatility) const;
    // ... other members
};

// Type aliases for convenience
using IVSolverFDMf = IVSolverFDM<float>;
using IVSolverFDMd = IVSolverFDM<double>;

// src/option/iv_solver_interpolated.hpp
template<std::floating_point T = double>
class IVSolverInterpolated {
public:
    static std::expected<IVSolverInterpolated<T>, ValidationError<T>> create(
        std::shared_ptr<const BSpline4D<T>> spline,
        T K_ref,
        std::pair<T, T> m_range,
        std::pair<T, T> tau_range,
        std::pair<T, T> sigma_range,
        std::pair<T, T> r_range,
        const IVSolverInterpolatedConfig<T>& config = {});

    std::expected<IVSuccess<T>, IVError<T>> solve_impl(const IVQuery<T>& query) const noexcept;
    BatchIVResult<T> solve_batch_impl(const std::vector<IVQuery<T>>& queries) const noexcept;

    // ... private members
};

// Type aliases for convenience
using IVSolverInterpolatedf = IVSolverInterpolated<float>;
using IVSolverInterpolatedd = IVSolverInterpolated<double>;
```

### Dependent Components

**American Option Solver:**
```cpp
template<std::floating_point T = double>
class AmericanOptionSolver {
public:
    static std::expected<AmericanOptionSolver<T>, ValidationError<T>>
        create(const PricingParams<T>& params,
               PDEWorkspace<T> workspace, ...);

    std::expected<AmericanOptionResult<T>, SolverError> solve();
private:
    PricingParams<T> params_;
    // ...
};

template<std::floating_point T = double>
struct AmericanOptionResult {
    T value_at(T spot) const;
    std::shared_ptr<const SpatialGrid<T>> grid() const;
    // ...
};
```

**Root Finding:**
```cpp
template<std::floating_point T = double>
struct RootFindingConfig {
    size_t max_iter = 100;
    T tolerance = default_tolerance<T>();  // Type-aware default
    T min_step = default_min_step<T>();    // Type-aware default
};

template<std::floating_point T = double>
struct RootFindingSuccess {
    T root;
    size_t iterations;
    T final_error;
};

template<std::floating_point T = double, typename Func>
std::expected<RootFindingSuccess<T>, RootFindingError>
brent_find_root(Func&& f, T a, T b, const RootFindingConfig<T>& config);
```

**Grid/PDE infrastructure:** Already templated - verify they work with both float and double.

### Price Table System

```cpp
// price_table_grid.hpp
template<std::floating_point T = double>
struct PriceTableGrid {
    std::vector<T> moneyness;
    std::vector<T> maturity;
    std::vector<T> volatility;
    std::vector<T> rate;
};

// price_table_4d_builder.hpp
template<std::floating_point T = double>
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

// Type aliases
using PriceTable4DBuilderf = PriceTable4DBuilder<float>;
using PriceTable4DBuilderd = PriceTable4DBuilder<double>;

// bspline_price_table.hpp
template<std::floating_point T = double>
class BSpline4D {
public:
    static std::expected<BSpline4D<T>, std::string> create(
        const PriceTableWorkspace<T>& workspace);

    T eval(T m, T tau, T sigma, T r) const;

    // Analytic vega computation
    void eval_price_and_vega_analytic(T m, T tau, T sigma, T r,
                                      T& price_out, T& vega_out) const;
private:
    BSplineND<T, 4> spline_;
};

template<std::floating_point T = double>
struct PriceTableSurface {
    std::shared_ptr<const BSpline4D<T>> spline;
    T K_ref;
    // ...
};
```

**Note:** All B-spline infrastructure (`BSplineND<T, N>`) already templated - just verify compatibility.

## Implementation Plan (Phased Migration)

**Critical:** Each phase MUST compile and pass all tests before proceeding to next phase.

### Phase 1: Foundation Layer

**Files:**
1. `src/support/numeric_helpers.hpp` (NEW) - Type-aware tolerance helpers
2. `src/support/error_types.hpp` - Template `ValidationError<T>` to preserve precision
3. `src/option/iv_result.hpp` - Template `IVSuccess<T>`, `IVError<T>`, `BatchIVResult<T>`
4. `src/option/option_spec.hpp` - Template `OptionSpec<T>`, `IVQuery<T>`, `PricingParams<T>`
5. `src/option/option_spec.ipp` (NEW) - Template implementations for validation
6. `src/option/option_spec_instantiations.cpp` (NEW) - Explicit instantiations for float/double

**Changes:**
- Add `= double` defaults to all templates
- Create type aliases (`OptionSpecf`, `OptionSpecd`, etc.)
- Move validation logic to `.ipp` for header-based templates
- Add explicit instantiations for faster compilation

**Testing:**
```bash
bazel test //tests:option_spec_test
bazel test //tests:iv_result_test
```

**Exit Criteria:**
- ✅ All foundation tests pass
- ✅ Both `<float>` and `<double>` instantiations compile
- ✅ No change in test behavior (still using double by default)

---

### Phase 2: Math and Root Finding

**Files:**
1. `src/math/root_finding.hpp` - Template `RootFindingConfig<T>`, `RootFindingSuccess<T>`
2. `src/math/root_finding.ipp` (NEW) - Template implementation of `brent_find_root<T>`
3. `src/math/root_finding_instantiations.cpp` (NEW) - Explicit instantiations
4. Verify `src/pde/core/grid.hpp` already works with float/double

**Changes:**
- Use `default_tolerance<T>()` and `default_min_step<T>()` in config defaults
- Add type aliases (`RootFindingConfigf`, `RootFindingConfigd`)

**Testing:**
```bash
bazel test //tests:root_finding_test
bazel test //tests:pde_solver_test  # Verify PDE layer still works
```

**Exit Criteria:**
- ✅ Root finding tests pass
- ✅ Float instantiation compiles and runs
- ✅ Tolerances scale correctly (verify `1e-4` for float, `1e-8` for double)

---

### Phase 3: American Option Pricing

**Files:**
1. `src/option/american_option.hpp` - Template `AmericanOptionSolver<T>`
2. `src/option/american_option.ipp` (NEW) - Template implementations
3. `src/option/american_option_result.hpp` - Template `AmericanOptionResult<T>`
4. `src/option/american_pde_solver.hpp` - Update PDE wrappers

**Changes:**
- Default `= double` on all classes
- Type aliases (`AmericanOptionSolverf`, `AmericanOptionSolverd`)
- Update grid estimation to use type-aware time step counts (500 for float, 1000 for double)

**Testing:**
```bash
bazel test //tests:american_option_test
bazel test //tests:american_option_integration_test
```

**Exit Criteria:**
- ✅ All American option tests pass
- ✅ Float version compiles
- ✅ Price accuracy within expected tolerances for float

---

### Phase 4: Price Table System

**Files:**
1. `src/option/price_table_grid.hpp` - Template `PriceTableGrid<T>`
2. `src/option/price_table_4d_builder.hpp` - Template `PriceTable4DBuilder<T>`
3. `src/option/bspline_price_table.hpp` - Template `BSpline4D<T>`, `PriceTableSurface<T>`
4. `src/option/price_table_extraction.hpp` - Template extraction functions
5. Verify `src/math/bspline_nd.hpp` already works with float/double

**Changes:**
- Default `= double` on all templates
- Type aliases (`PriceTable4DBuilderf`, `PriceTable4DBuilderd`)
- Ensure B-spline knot spacing respects float precision

**Testing:**
```bash
bazel test //tests:price_table_test
bazel test //tests:bspline_price_table_test
bazel test //tests:price_table_precompute_test
```

**Exit Criteria:**
- ✅ Price table tests pass
- ✅ Float instantiation compiles
- ✅ Interpolation accuracy acceptable for float

---

### Phase 5: IV Solvers

**Files:**
1. `src/option/iv_solver_fdm.hpp` - Template `IVSolverFDM<T>`, `IVSolverFDMConfig<T>`
2. `src/option/iv_solver_fdm.cpp` - Update to use templated types
3. `src/option/iv_solver_interpolated.hpp` - Template `IVSolverInterpolated<T>`, config
4. `src/option/iv_solver_interpolated.cpp` - Update to use templated types

**Changes:**
- Default `= double` on all templates
- Type aliases (`IVSolverFDMf`, `IVSolverFDMd`, `IVSolverInterpolatedf`, `IVSolverInterpolatedd`)
- Use `default_tolerance<T>()` in solver configs
- Update return types to `IVError<T>`, `ValidationError<T>`

**Testing:**
```bash
bazel test //tests:iv_solver_test
bazel test //tests:iv_solver_fdm_test
bazel test //tests:iv_solver_interpolated_test
```

**Exit Criteria:**
- ✅ All IV solver tests pass
- ✅ Float instantiation compiles
- ✅ Convergence behavior acceptable for float

---

### Phase 6: Examples and Documentation

**Files:**
1. Update all examples in `examples/` to show template usage
2. Update `docs/API_GUIDE.md` with template examples
3. Update `README.md` with float/double mention

**Changes:**
- Show both `<float>` and `<double>` usage
- Document precision/performance trade-offs
- Add float example to demonstrate lower memory usage

**Testing:**
```bash
bazel build //examples/...
bazel run //examples:example_iv_calculation
```

**Exit Criteria:**
- ✅ All examples compile and run
- ✅ Documentation updated
- ✅ No user-facing API breaks (defaults work)

## Testing Strategy

### Per-Phase Testing

**Critical:** Each phase includes its own tests. Tests must pass before moving to next phase.

See "Exit Criteria" in each phase of Implementation Plan for specific test commands.

### Explicit Float Instantiation Tests

**Requirement:** At each phase, add at least one explicit `<float>` instantiation test:

```cpp
// tests/root_finding_float_test.cc (Phase 2)
TEST(RootFindingFloatTest, BrentConvergesWithFloatTolerance) {
    RootFindingConfig<float> config;  // Uses default_tolerance<float>()

    auto f = [](float x) { return x * x - 2.0f; };
    auto result = brent_find_root(f, 0.0f, 2.0f, config);

    ASSERT_TRUE(result.has_value());
    EXPECT_NEAR(result->root, std::sqrt(2.0f), 1e-4f);  // Float tolerance
}

// tests/iv_solver_fdm_float_test.cc (Phase 5)
TEST(IVSolverFDMFloatTest, ATMPutConvergesInFloat) {
    OptionSpec<float> spec{
        .spot = 100.0f,
        .strike = 100.0f,
        .maturity = 1.0f,
        .rate = 0.05f,
        .dividend_yield = 0.02f,
        .type = OptionType::PUT
    };

    IVQuery<float> query{.option = spec, .market_price = 10.45f};
    IVSolverFDM<float> solver(IVSolverFDMConfig<float>{});

    auto result = solver.solve_impl(query);
    ASSERT_TRUE(result.has_value());
    EXPECT_NEAR(result->implied_vol, 0.20f, 1e-3f);  // Float tolerance
}
```

**Why:** This ensures templates actually instantiate for float and catches precision issues early.

### Runtime Float Tests (Required)

**Addressing Code Review Issue #1:** Add runtime float tests, not just compilation checks.

Each phase must include:
1. At least one test that runs with `<float>` instantiation
2. Verification that convergence occurs with float-appropriate tolerances
3. Comparison of float vs double results (document expected differences)

### Regression Testing

After each phase:
```bash
# Full test suite must still pass
bazel test //...

# Examples must still compile
bazel build //examples/...

# Benchmarks must still compile
bazel build //benchmarks/...
```

### Acceptance Criteria (Updated)

Per the code review feedback, acceptance criteria now include:

- ✅ All 57 existing tests pass with double (default)
- ✅ **NEW:** At least 6 runtime float tests (one per phase)
- ✅ **NEW:** Float instantiation compiles in all templated classes
- ✅ **NEW:** Tolerances scale correctly (verified via float tests)
- ✅ Code compiles without warnings
- ✅ No performance regression for double version
- ✅ API documentation updated with template examples
- ✅ **NEW:** Numeric stability differences documented (float vs double)

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
**Impact:** Code using unqualified types must add template arguments or rely on defaults
**Mitigation:**
- Default template arguments (`= double`) preserve existing behavior
- Phased migration allows testing after each layer
- Type aliases (`IVSolverFDMd`) provide explicit opt-in

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

- C++20 `std::floating_point` concept (for context): https://en.cppreference.com/w/cpp/concepts/floating_point
- Custom concepts in C++20: https://en.cppreference.com/w/cpp/language/constraints
- Explicit template instantiation: https://en.cppreference.com/w/cpp/language/template_specialization
- Mixed-precision computing: https://developer.nvidia.com/blog/mixed-precision-programming-cuda-8/
- Default template arguments: https://en.cppreference.com/w/cpp/language/template_parameters#Default_template_arguments
