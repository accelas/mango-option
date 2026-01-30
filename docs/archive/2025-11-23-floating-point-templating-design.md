<!-- SPDX-License-Identifier: MIT -->
# Floating-Point Templating Design

**Date:** 2025-11-23
**Status:** Design Under Revision - Addressing Second Code Review Feedback
**Goal:** Template entire codebase to support `float` (fp32) and `double` (fp64), with explicit instantiations only

## Overview

Convert all numeric types from hardcoded `double` to template parameter `T` constrained by `std::floating_point`. This enables:
- FP64 (double) for maximum accuracy (current default)
- FP32 (float) for performance when precision requirements allow
- Consistent type safety across the entire stack

**Explicit Support Policy:** Only `float` and `double` are explicitly instantiated and supported. `long double` is NOT supported (will cause linker errors).

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

### Type Consistency Enforcement

**Critical:** Prevent accidental type mixing (e.g., `IVSolverFDM<float>` with `OptionSpec<double>`).

```cpp
// In each templated class, add static assertions to verify type consistency

template<std::floating_point T = double>
class IVSolverFDM {
public:
    std::expected<IVSuccess<T>, IVError<T>> solve_impl(const IVQuery<T>& query) const {
        // Static assert to catch type mismatches at compile time
        static_assert(std::same_as<T, decltype(query.spot)>,
                      "IVQuery type T must match solver type T");
        // ... implementation
    }
};

// For inheritance hierarchies, add concept checks
template<std::floating_point T>
struct IVQuery : OptionSpec<T> {
    T market_price;

    // Ensure base and derived use same T
    static_assert(std::same_as<T, decltype(OptionSpec<T>::spot)>,
                  "IVQuery must use same T as OptionSpec");
};
```

**Validation function type safety:**
```cpp
// Validation functions must preserve T through the call chain
template<std::floating_point T = double>
std::expected<void, ValidationError<T>> validate_iv_query(const IVQuery<T>& query) {
    // Automatically inherits T from argument, preventing mix-ups
    if (query.spot <= T{0}) {
        return std::unexpected(ValidationError<T>{
            ValidationErrorCode::InvalidSpotPrice,
            query.spot,  // Preserves exact type T
            0
        });
    }
    return {};
}
```

### Non-Templated Types

**Critical:** Some types remain non-templated to avoid code bloat and ABI issues.

```cpp
// src/support/error_types.hpp - NOT templated
enum class SolverErrorCode {
    GridAllocationFailed,
    MatrixSingular,
    TimeStepFailed,
    // ...
};

struct SolverError {
    SolverErrorCode code;
    std::string message;
    // No floating-point values stored - remains non-templated
};

// Enums are never templated
enum class IVErrorCode {
    NegativeSpot,
    NegativeStrike,
    // ...
};

enum class ValidationErrorCode {
    InvalidSpotPrice,
    InvalidStrike,
    // ...
};

enum class OptionType {
    CALL,
    PUT
};
```

**Rationale:** Error codes, enums, and string-based errors don't depend on floating-point precision. Keeping them non-templated:
- Reduces binary size (no duplication for float/double)
- Simplifies ABI (stable across template instantiations)
- Avoids cascading template requirements in error handling code

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
///
/// Uses sqrt(epsilon) * scaling_factor as tolerance.
/// Rationale:
/// - Float: sqrt(1e-7) ≈ 3.16e-4. For IV solving, this allows ~0.03% relative error
///   in volatility, which translates to <0.1% option price error (acceptable for float).
/// - Double: sqrt(1e-16) ≈ 1e-8. Standard high-precision tolerance.
///
/// NOT constexpr because std::sqrt is not constexpr until C++26.
template<std::floating_point T>
inline T default_tolerance() {
    constexpr T eps = std::numeric_limits<T>::epsilon();
    // sqrt(eps) is commonly used for finite difference tolerances
    // For float: sqrt(1e-7) ≈ 3.16e-4
    // For double: sqrt(1e-16) ≈ 1e-8
    return std::sqrt(eps);  // No additional scaling - sqrt(eps) is the standard choice
}

/// Compute minimum step size for finite differences
///
/// Uses 10*sqrt(epsilon) for Brent's method min_step to prevent stagnation.
/// Rationale:
/// - Float: 10*sqrt(1e-7) ≈ 3.16e-3 (prevents steps smaller than representable)
/// - Double: 10*sqrt(1e-16) ≈ 1e-7 (standard for double precision root finding)
///
/// NOT constexpr because std::sqrt is not constexpr until C++26.
template<std::floating_point T>
inline T default_min_step() {
    constexpr T eps = std::numeric_limits<T>::epsilon();
    // 10*sqrt(eps) prevents underflow while avoiding premature termination
    // For float: 10*sqrt(1e-7) ≈ 3.16e-3
    // For double: 10*sqrt(1e-16) ≈ 1e-7
    return T{10} * std::sqrt(eps);
}

/// Check if value is effectively zero (within machine precision)
template<std::floating_point T>
inline bool is_effectively_zero(T value, T tolerance = default_tolerance<T>()) {
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

### Acceptance Criteria (Updated After Second Review)

**Compilation:**
- ✅ All code compiles with `-Wconversion -Wnarrowing` (no literal suffix warnings)
- ✅ Both float and double explicit instantiations compile without errors
- ✅ Attempting `long double` instantiation causes **linker error** (verified)

**Testing - Double Precision:**
- ✅ All 57 existing tests pass with double (default, no changes to results)
- ✅ No performance regression for double version (within 5%)

**Testing - Float Precision (Comprehensive):**
- ✅ **Minimum 6 runtime float tests** (one per phase, as originally planned)
- ✅ **NEW:** At least 15 float scenario tests covering:
  - ATM, ITM (10%, 20%, 30%), OTM (10%, 20%, 30%) strikes
  - Short (1 week), medium (3 months), long (1 year) maturities
  - Low vol (10%), medium vol (30%), high vol (100%)
- ✅ **NEW:** Float vs double determinism tests:
  - Same inputs → consistent ordering (float IV < double IV or vice versa)
  - Document expected relative error bounds (e.g., <0.5% for IV)
- ✅ **NEW:** Float convergence tests:
  - Verify convergence with `default_tolerance<float>()` (~3.16e-4)
  - Test that iteration counts don't exceed 2x double iteration count
  - Verify no stagnation (min_step prevents premature termination)

**Type Safety:**
- ✅ **NEW:** Static assert tests verify type mismatches caught at compile time:
  - `IVSolverFDM<float>` with `IVQuery<double>` → compile error
  - Validation returns `ValidationError<T>` matching input `T`
- ✅ **NEW:** Cross-type tests explicitly verify conversions fail gracefully

**Binary Size:**
- ✅ **NEW:** Binary size measured and documented (baseline vs float+double)
- ✅ **NEW:** Binary size increase <100% (verified via `size` and `bloaty`)

**Documentation:**
- ✅ API documentation updated with template examples (both float and double)
- ✅ **NEW:** Numeric stability differences documented with relative error bounds
- ✅ **NEW:** Literal suffix guidelines added to developer docs
- ✅ **NEW:** ABI stability policy documented

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

## Literal Suffix Guidelines

**Critical:** Prevent implicit conversions and narrowing warnings.

### Rules

1. **Always use type-appropriate literal suffixes:**
   ```cpp
   // Good - float
   float x = 2.0f;
   float y = std::sqrt(2.0f);

   // Good - double
   double x = 2.0;
   double y = std::sqrt(2.0);

   // BAD - creates double temporary, then narrows to float
   float x = 2.0;  // narrowing warning
   ```

2. **In template code, use `T{}` initialization:**
   ```cpp
   template<std::floating_point T>
   T compute(T x) {
       T zero = T{0};        // Correct for any T
       T two = T{2};         // Correct for any T
       T pi = T{3.141592653589793};  // Correct for any T

       // DON'T: return 2.0 * x;  // Always creates double
       return two * x;  // Uses T
   }
   ```

3. **For constants, use typed constexpr variables:**
   ```cpp
   template<std::floating_point T>
   struct Constants {
       static constexpr T pi = T{3.141592653589793};
       static constexpr T e = T{2.718281828459045};
       static constexpr T sqrt2 = T{1.414213562373095};
   };

   // Usage
   T area = Constants<T>::pi * r * r;
   ```

4. **Audit checklist for each phase:**
   - [ ] All numeric literals have appropriate suffixes (`f` for float contexts)
   - [ ] Template code uses `T{}` initialization
   - [ ] No bare `2.0`, `0.5`, etc. in template functions
   - [ ] Compiler runs with `-Wconversion -Wnarrowing` enabled

### Validation

During each phase, compile with:
```bash
bazel build --cxxopt=-Wconversion --cxxopt=-Wnarrowing //...
```

Any warnings indicate missing literal suffixes or implicit conversions.

## ABI and Binary Compatibility

**Critical:** Template instantiations in public headers affect ABI.

### Shared Library Considerations

**Current status:** This library is currently header-only / statically linked. No shared library ABI concerns exist today.

**If shared library support is added later:**

1. **Explicit instantiation exports required:**
   ```cpp
   // In libmango.so implementation
   template class __attribute__((visibility("default"))) IVSolverFDM<float>;
   template class __attribute__((visibility("default"))) IVSolverFDM<double>;
   // long double NOT instantiated - will cause linker error if attempted
   ```

2. **ABI stability guarantees:**
   - Only `float` and `double` instantiations have ABI stability
   - Adding member variables to templated classes breaks ABI
   - Changing default template arguments breaks ABI (old clients compile against old default)
   - Type alias changes (`using IVSolverFDMf = ...`) do NOT break ABI

3. **Version script for symbol visibility:**
   ```
   MANGO_1.0 {
     global:
       extern "C++" {
         mango::IVSolverFDM<float>*;
         mango::IVSolverFDM<double>*;
       };
     local:
       *;
   };
   ```

### Binary Size Analysis

**Expected impact of template instantiation:**

| Component | Baseline (double only) | With float+double | Increase |
|-----------|------------------------|-------------------|----------|
| IV Solvers | ~50 KB | ~85 KB | +70% |
| Price Tables | ~30 KB | ~50 KB | +67% |
| Root Finding | ~10 KB | ~17 KB | +70% |
| **Total library** | ~200 KB | ~320 KB | **+60%** |

**Mitigation strategies:**
1. Explicit instantiation prevents redundant instantiations across TUs
2. Only float/double supported (long double excluded → linker error)
3. Small inline functions marked `inline` to allow deduplication
4. Link-time optimization (LTO) can deduplicate identical instantiations

**Measurement during implementation:**
```bash
# After each phase, measure binary size
size bazel-bin/src/option/libiv_solver.a
bloaty bazel-bin/src/option/libiv_solver.a  # Detailed breakdown
```

### Cross-TU Instantiation Safety

**Problem:** If one translation unit instantiates `IVSolverFDM<long double>` and another provides only float/double, linker will fail with undefined symbols.

**Solution:** Provide explicit instantiation declarations in headers to prevent implicit instantiation:

```cpp
// src/option/iv_solver_fdm.hpp
template<std::floating_point T = double>
class IVSolverFDM {
    // ... definition
};

// Prevent implicit instantiation - only explicit instantiations in .cpp are allowed
extern template class IVSolverFDM<float>;
extern template class IVSolverFDM<double>;
// Note: NO extern template for long double - will cause linker error if used
```

```cpp
// src/option/iv_solver_fdm.cpp
// Explicit instantiation definitions
template class IVSolverFDM<float>;
template class IVSolverFDM<double>;
```

**Benefit:** Attempting `IVSolverFDM<long double>` anywhere in the codebase will result in a **linker error**, making the "float/double only" policy enforceable at link time.

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
