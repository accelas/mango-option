# API Safety Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Eliminate four classes of silent wrong-result bugs in the public API.

**Architecture:** Add `option_type()` and `dividend_yield()` to the `PriceSurface` concept and validate in the IV solver. Replace `pair<double,double>` dividends with a named `Dividend` struct. Remove positional constructors from `PricingParams` and `IVQuery`.

**Tech Stack:** C++23, Bazel, GoogleTest

---

### Task 1: Define `Dividend` struct in option_spec.hpp

**Files:**
- Modify: `src/option/option_spec.hpp`

**Step 1: Add the struct**

Add after the `OptionType` enum (after line 113), before `OptionSpec`:

```cpp
/// Discrete dividend event
///
/// Represents a known future dividend payment at a specific calendar time.
/// Calendar time is measured in years from the valuation date.
struct Dividend {
    double calendar_time = 0.0;  ///< Years from valuation date
    double amount = 0.0;         ///< Dollar amount
};
```

**Step 2: Update PricingParams to use Dividend**

Change line 216 from:
```cpp
std::vector<std::pair<double, double>> discrete_dividends;
```
to:
```cpp
std::vector<Dividend> discrete_dividends;
```

Update all constructor parameters similarly — every `std::vector<std::pair<double, double>> discrete_dividends_` becomes `std::vector<Dividend> discrete_dividends_`. Remove the `initializer_list<pair<double,double>>` overload (lines 247-257) since `Dividend` supports designated initializers and brace init directly.

**Step 3: Update IVQuery positional constructor**

No change needed — IVQuery doesn't take dividends.

**Step 4: Build to verify header compiles**

Run: `bazel build //src/option:option_spec`
Expected: Compilation errors in downstream files (expected — we fix those next).

**Step 5: Commit**

```
Add Dividend struct, replace pair<double,double> in option_spec
```

---

### Task 2: Propagate Dividend struct through config headers

**Files:**
- Modify: `src/option/iv_solver_factory.hpp:30`
- Modify: `src/option/table/price_table_config.hpp:19`
- Modify: `src/option/table/price_table_metadata.hpp:30`
- Modify: `src/option/table/segmented_price_surface.hpp:22,47-50,56`
- Modify: `src/option/table/segmented_multi_kref_builder.hpp:25`
- Modify: `src/option/table/segmented_price_table_builder.hpp:24`

**Step 1: Update each config struct**

In each file, change `std::vector<std::pair<double, double>>` to `std::vector<Dividend>`.

For `segmented_price_surface.hpp`:
- Change `Config::dividends` type to `std::vector<Dividend>`
- Delete the private `DividendEntry` struct (lines 47-50)
- Change `std::vector<DividendEntry> dividends_;` to `std::vector<Dividend> dividends_;`

**Step 2: Build headers**

Run: `bazel build //src/option/...`
Expected: Errors in .cpp files that use structured bindings or `.first`/`.second`.

**Step 3: Commit**

```
Propagate Dividend struct through all config headers
```

---

### Task 3: Update source files for Dividend struct

**Files:**
- Modify: `src/option/american_option.hpp` (lines 70, 87, 147)
- Modify: `src/option/american_option.cpp` (lines 129, 150, 168)
- Modify: `src/option/american_option_batch.cpp` (lines 61, 167, 170)
- Modify: `src/option/option_spec.cpp` (lines 115-116)
- Modify: `src/option/iv_solver_factory.cpp` (line 88)
- Modify: `src/option/table/price_table_builder.cpp` (lines 140, 199)
- Modify: `src/option/table/american_price_surface.cpp` (line 40)
- Modify: `src/option/table/segmented_price_surface.cpp` (lines 41-42, 47)
- Modify: `src/option/table/segmented_multi_kref_builder.cpp` (line 50)
- Modify: `src/option/table/segmented_price_table_builder.cpp` (line 457)
- Modify: `src/option/table/adaptive_grid_builder.cpp` (line 323)

**Step 1: Update structured bindings**

Change all `auto& [t_cal, amount]` patterns to use `.calendar_time` and `.amount`:

```cpp
// Before:
for (const auto& [t_cal, amount] : params.discrete_dividends) {
    mandatory_times.push_back(t_cal);
}
// After:
for (const auto& div : params.discrete_dividends) {
    mandatory_times.push_back(div.calendar_time);
}
```

**Step 2: Update segmented_price_surface.cpp**

The `DividendEntry{t, amount}` construction on line 42 becomes just pushing `Dividend` directly (types now match). Update the sort lambda to use `.calendar_time`.

**Step 3: Update validation in option_spec.cpp**

Line 116 changes from `const auto& [time, amount] = params.discrete_dividends[i]` to member access on `Dividend`.

**Step 4: Build source files**

Run: `bazel build //src/...`
Expected: Errors only in python bindings and test files.

**Step 5: Commit**

```
Update all source files for Dividend struct
```

---

### Task 4: Update Python bindings for Dividend struct

**Files:**
- Modify: `src/python/mango_bindings.cpp` (lines 280, 686)

**Step 1: Register Dividend with pybind11**

Add near other struct registrations:

```cpp
py::class_<mango::Dividend>(m, "Dividend")
    .def(py::init<>())
    .def(py::init<double, double>(), py::arg("calendar_time"), py::arg("amount"))
    .def_readwrite("calendar_time", &mango::Dividend::calendar_time)
    .def_readwrite("amount", &mango::Dividend::amount)
    .def("__repr__", [](const mango::Dividend& d) {
        return "Dividend(t=" + std::to_string(d.calendar_time) +
               ", amt=" + std::to_string(d.amount) + ")";
    });
```

The `def_readwrite("discrete_dividends", ...)` lines stay the same — pybind11 handles `vector<Dividend>` once `Dividend` is registered.

**Step 2: Build Python bindings**

Run: `bazel build //src/python:mango_option`
Expected: SUCCESS

**Step 3: Commit**

```
Register Dividend struct in Python bindings
```

---

### Task 5: Update all test files for Dividend struct

**Files:**
- Modify: All test files listed in the search results (~12 test files, ~50 sites)

**Step 1: Replace brace-init pairs with Dividend**

Change all `{{0.5, 3.0}}` patterns in dividend contexts:

```cpp
// Before:
.discrete_dividends = {{0.5, 3.0}}
// After:
.discrete_dividends = {{.calendar_time = 0.5, .amount = 3.0}}
```

For `std::vector` explicit constructions:
```cpp
// Before:
std::vector<std::pair<double, double>>{{0.4, 3.0}}
// After:
std::vector<mango::Dividend>{{.calendar_time = 0.4, .amount = 3.0}}
```

For `.first`/`.second` access in assertions:
```cpp
// Before:
EXPECT_DOUBLE_EQ(meta.discrete_dividends[0].first, 0.25);
// After:
EXPECT_DOUBLE_EQ(meta.discrete_dividends[0].calendar_time, 0.25);
```

**Step 2: Run all tests**

Run: `bazel test //tests/...`
Expected: ALL PASS

**Step 3: Commit**

```
Migrate all tests to Dividend struct
```

---

### Task 6: Add `option_type()` and `dividend_yield()` to PriceSurface concept

**Files:**
- Modify: `src/option/table/price_surface_concept.hpp`
- Modify: `src/option/table/american_price_surface.hpp`
- Modify: `src/option/table/american_price_surface.cpp`
- Modify: `src/option/table/segmented_price_surface.hpp`
- Modify: `src/option/table/segmented_price_surface.cpp`

**Step 1: Write regression tests**

Create test cases in `tests/iv_solver_interpolated_test.cc`:

```cpp
// Regression: IVSolverInterpolated must reject queries with wrong option type
// Bug: solve() accepted any IVQuery regardless of type, returning wrong IV
TEST(IVSolverInterpolatedRegressionTest, RejectsOptionTypeMismatch) {
    // Build a PUT surface (use existing test fixture setup)
    // Then query with a CALL — must return error
    // ... (use existing surface-building pattern from this test file)
}

// Regression: IVSolverInterpolated must reject queries with wrong dividend_yield
// Bug: AmericanPriceSurface bakes in dividend_yield at construction; callers
// with a different yield got wrong prices silently
TEST(IVSolverInterpolatedRegressionTest, RejectsDividendYieldMismatch) {
    // Build surface with dividend_yield=0.02
    // Query with dividend_yield=0.05 — must return error
}
```

Run: `bazel test //tests:iv_solver_interpolated_test`
Expected: FAIL (tests reference accessors that don't exist yet)

**Step 2: Extend PriceSurface concept**

In `price_surface_concept.hpp`, add to the concept:

```cpp
template <typename S>
concept PriceSurface = requires(const S& s, double spot, double strike,
                                double tau, double sigma, double rate) {
    { s.price(spot, strike, tau, sigma, rate) } -> std::same_as<double>;
    { s.vega(spot, strike, tau, sigma, rate) } -> std::same_as<double>;
    { s.option_type() } -> std::same_as<OptionType>;
    { s.dividend_yield() } -> std::convertible_to<double>;
    { s.m_min() } -> std::convertible_to<double>;
    // ... rest unchanged ...
};
```

This requires adding `#include "mango/option/option_spec.hpp"` to the concept header (for `OptionType`).

**Step 3: Add accessors to AmericanPriceSurface**

In `american_price_surface.hpp`, add in the public section:

```cpp
[[nodiscard]] OptionType option_type() const noexcept;
[[nodiscard]] double dividend_yield() const noexcept;
```

In `american_price_surface.cpp`:

```cpp
OptionType AmericanPriceSurface::option_type() const noexcept {
    return type_;
}

double AmericanPriceSurface::dividend_yield() const noexcept {
    return dividend_yield_;
}
```

**Step 4: Add accessors to SegmentedPriceSurface**

In `segmented_price_surface.hpp`, add:

```cpp
[[nodiscard]] OptionType option_type() const noexcept;
[[nodiscard]] double dividend_yield() const noexcept;
```

Store `option_type_` and `dividend_yield_` as private members. Set them in `create()` from the first segment:

In `segmented_price_surface.cpp`, in `create()` after validating segments:

```cpp
result.option_type_ = config.segments.front().surface.option_type();
result.dividend_yield_ = config.segments.front().surface.dividend_yield();
```

And the accessors:

```cpp
OptionType SegmentedPriceSurface::option_type() const noexcept {
    return option_type_;
}

double SegmentedPriceSurface::dividend_yield() const noexcept {
    return dividend_yield_;
}
```

**Step 5: Build**

Run: `bazel build //src/...`
Expected: SUCCESS (concept satisfied by both implementations)

**Step 6: Commit**

```
Add option_type() and dividend_yield() to PriceSurface concept
```

---

### Task 7: Add validation in IVSolverInterpolated

**Files:**
- Modify: `src/option/iv_solver_interpolated.hpp`

**Step 1: Add option_type to solver state**

In the private section, add:

```cpp
OptionType option_type_;
```

In `create()`, after constructing the solver, store the option type:

Actually, pass it through the constructor. Update the private constructor to accept and store `option_type`:

```cpp
IVSolverInterpolated(
    Surface surface,
    std::pair<double, double> m_range,
    std::pair<double, double> tau_range,
    std::pair<double, double> sigma_range,
    std::pair<double, double> r_range,
    OptionType option_type,
    double dividend_yield,
    const IVSolverInterpolatedConfig& config)
    : surface_(std::move(surface))
    , m_range_(m_range), tau_range_(tau_range)
    , sigma_range_(sigma_range), r_range_(r_range)
    , option_type_(option_type)
    , dividend_yield_(dividend_yield)
    , config_(config)
{}
```

In `create()`, extract from the surface:

```cpp
auto option_type = surface.option_type();
auto dividend_yield = surface.dividend_yield();

return IVSolverInterpolated(
    std::move(surface), m_range, tau_range, sigma_range, r_range,
    option_type, dividend_yield, config);
```

**Step 2: Add validation in validate_query()**

In `validate_query()`, add before the existing validation:

```cpp
if (query.type != option_type_) {
    return ValidationError{ValidationErrorCode::InvalidBounds, 0.0, 0};
}

if (std::abs(query.dividend_yield - dividend_yield_) > 1e-10) {
    return ValidationError{ValidationErrorCode::InvalidBounds, 0.0, 0};
}
```

**Step 3: Run regression tests**

Run: `bazel test //tests:iv_solver_interpolated_test`
Expected: ALL PASS (including the new regression tests from Task 6 Step 1)

**Step 4: Run full test suite**

Run: `bazel test //tests/...`
Expected: ALL PASS — existing tests already use matching option types and dividend yields.

**Step 5: Commit**

```
Validate option_type and dividend_yield in IVSolverInterpolated
```

---

### Task 8: Remove positional constructors from PricingParams

**Files:**
- Modify: `src/option/option_spec.hpp` (lines 228-276)

**Step 1: Remove positional constructors**

Delete the three positional constructors (lines 228-276):
- `PricingParams(double spot_, double strike_, double maturity_, double rate_, double dividend_yield_, OptionType type_, double volatility_, vector...)`
- `PricingParams(double spot_, ..., initializer_list...)`
- `PricingParams(double spot_, ..., RateSpec rate_, ...)`

Keep:
- `PricingParams() = default;`
- `PricingParams(const OptionSpec& spec, double volatility_, vector<Dividend> discrete_dividends_ = {})`

**Step 2: Build to see breakage**

Run: `bazel build //...`
Expected: Compilation errors at all positional call sites.

**Step 3: Commit**

```
Remove positional constructors from PricingParams
```

---

### Task 9: Migrate PricingParams call sites in source files

**Files:**
- Modify: `src/option/table/american_price_surface.cpp` (lines 64, 75, 86, 98, 111)
- Modify: `src/option/table/price_table_builder.cpp` (line 473-475)

**Step 1: Migrate american_price_surface.cpp**

All five sites follow the same pattern. Change:

```cpp
PricingParams(spot, strike, tau, rate, dividend_yield_, type_, sigma)
```

to:

```cpp
PricingParams(OptionSpec{.spot = spot, .strike = strike, .maturity = tau,
    .rate = rate, .dividend_yield = dividend_yield_, .type = type_}, sigma)
```

**Step 2: Migrate price_table_builder.cpp**

Similar pattern.

**Step 3: Build source**

Run: `bazel build //src/...`
Expected: SUCCESS

**Step 4: Commit**

```
Migrate source files to OptionSpec+vol constructor
```

---

### Task 10: Migrate PricingParams call sites in tests and benchmarks

**Files:**
- Modify: ~12 test files, ~3 benchmark files (~130 call sites)

**Step 1: Migrate test files**

For each positional call, convert to field-by-field assignment or the kept constructor. The most readable pattern for tests:

```cpp
// Before:
PricingParams(100.0, 100.0, 1.0, 0.05, 0.02, OptionType::PUT, 0.20)

// After:
PricingParams(OptionSpec{.spot = 100.0, .strike = 100.0, .maturity = 1.0,
    .rate = 0.05, .dividend_yield = 0.02, .type = OptionType::PUT}, 0.20)
```

For calls with discrete dividends:

```cpp
// Before:
PricingParams(100.0, 100.0, 1.0, 0.05, 0.0, OptionType::PUT, 0.20, {{0.5, 2.0}})

// After:
PricingParams(OptionSpec{.spot = 100.0, .strike = 100.0, .maturity = 1.0,
    .rate = 0.05, .type = OptionType::PUT}, 0.20,
    {{.calendar_time = 0.5, .amount = 2.0}})
```

**Step 2: Run all tests**

Run: `bazel test //tests/...`
Expected: ALL PASS

**Step 3: Build benchmarks**

Run: `bazel build //benchmarks/...`
Expected: SUCCESS

**Step 4: Commit**

```
Migrate tests and benchmarks to named PricingParams construction
```

---

### Task 11: Remove positional constructor from IVQuery

**Files:**
- Modify: `src/option/option_spec.hpp` (lines 164-179)

**Step 1: Remove the positional constructor**

Delete lines 164-179. Keep `IVQuery() = default;`.

**Step 2: Migrate call sites**

Convert all `IVQuery(spot, strike, ...)` calls in tests and benchmarks:

```cpp
// Before:
IVQuery(100.0, 100.0, 1.0, 0.05, 0.0, OptionType::PUT, 10.0)

// After:
IVQuery{.spot = 100.0, .strike = 100.0, .maturity = 1.0,
    .rate = 0.05, .type = OptionType::PUT, .market_price = 10.0}
```

Note: `IVQuery` inherits from `OptionSpec`. C++20 aggregate initialization through inheritance requires listing base-class fields first, then derived fields. Since `IVQuery` has a user-declared default constructor, it is NOT an aggregate. We need a different approach.

Add a constructor that takes `OptionSpec`:

```cpp
IVQuery(const OptionSpec& spec, double market_price_)
    : OptionSpec(spec), market_price(market_price_) {}
```

Then call sites become:

```cpp
IVQuery(OptionSpec{.spot = 100.0, .strike = 100.0, .maturity = 1.0,
    .rate = 0.05, .type = OptionType::PUT}, 10.0)
```

**Step 3: Run all tests**

Run: `bazel test //...`
Expected: ALL PASS

**Step 4: Build benchmarks**

Run: `bazel build //benchmarks/...`
Expected: SUCCESS

**Step 5: Commit**

```
Remove positional constructor from IVQuery
```

---

### Task 12: Final verification

**Step 1: Full build**

Run: `bazel build //...`
Expected: SUCCESS with no warnings

**Step 2: Full test suite**

Run: `bazel test //...`
Expected: ALL PASS

**Step 3: Python bindings**

Run: `bazel build //src/python:mango_option`
Expected: SUCCESS

**Step 4: Commit design doc**

```
git add docs/plans/2026-02-01-api-safety-design.md docs/plans/2026-02-01-api-safety-impl.md
git commit -m "Add API safety design and implementation plan"
```
