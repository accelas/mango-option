# IV Solver std::expected Refactoring Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Refactor IV solver API to use `std::expected<IVSuccess, IVError>` for type-safe error handling, mirroring the American option solver API pattern.

**Architecture:** Replace `IVResult` struct (boolean success flag + mixed success/failure data) with `std::expected` that separates success and failure paths. Reuse existing `SolverError` infrastructure and create IV-specific error types. Maintain backward compatibility during migration.

**Tech Stack:** C++23 std::expected, existing error_types.hpp infrastructure, GoogleTest for testing

---

## Phase 1: Define New Error Types

### Task 1.1: Create IVError type

**Files:**
- Modify: `src/support/error_types.hpp:19` (after SolverErrorCode enum)
- Test: `tests/iv_error_types_test.cc` (new file)

**Step 1: Write the failing test**

Create test file `tests/iv_error_types_test.cc`:

```cpp
#include <gtest/gtest.h>
#include "src/support/error_types.hpp"

using namespace mango;

TEST(IVErrorTypes, ErrorCodeEnum) {
    // Verify all error codes exist
    IVErrorCode code1 = IVErrorCode::NegativeSpot;
    IVErrorCode code2 = IVErrorCode::ArbitrageViolation;
    IVErrorCode code3 = IVErrorCode::MaxIterationsExceeded;
    IVErrorCode code4 = IVErrorCode::PDESolveFailed;

    // Verify they're different
    EXPECT_NE(static_cast<int>(code1), static_cast<int>(code2));
}

TEST(IVErrorTypes, ErrorStructConstruction) {
    IVError error{
        .code = IVErrorCode::MaxIterationsExceeded,
        .message = "Failed to converge after 100 iterations",
        .iterations = 100,
        .final_error = 0.05,
        .last_vol = 0.25
    };

    EXPECT_EQ(error.code, IVErrorCode::MaxIterationsExceeded);
    EXPECT_EQ(error.iterations, 100);
    EXPECT_EQ(error.final_error, 0.05);
    EXPECT_TRUE(error.last_vol.has_value());
    EXPECT_EQ(error.last_vol.value(), 0.25);
}
```

**Step 2: Run test to verify it fails**

Run: `bazel test //tests:iv_error_types_test --test_output=all`
Expected: BUILD FAIL - "IVErrorCode was not declared in this scope"

**Step 3: Add error types to error_types.hpp**

In `src/support/error_types.hpp`, after line 19 (after SolverErrorCode enum), add:

```cpp
/// IV solver error categories
enum class IVErrorCode {
    // Validation errors
    NegativeSpot,
    NegativeStrike,
    NegativeMaturity,
    NegativeMarketPrice,
    ArbitrageViolation,

    // Convergence errors
    MaxIterationsExceeded,
    BracketingFailed,
    NumericalInstability,

    // Solver errors
    PDESolveFailed
};

/// Detailed IV solver error with diagnostics
struct IVError {
    IVErrorCode code;
    std::string message;
    size_t iterations = 0;           ///< Iterations before failure
    double final_error = 0.0;        ///< Residual at failure
    std::optional<double> last_vol;  ///< Last volatility candidate tried
};
```

**Step 4: Add test to BUILD.bazel**

In `tests/BUILD.bazel`, add:

```python
cc_test(
    name = "iv_error_types_test",
    size = "small",
    srcs = ["iv_error_types_test.cc"],
    deps = [
        "//src/support:error_types",
        "@googletest//:gtest_main",
    ],
    copts = ["-std=c++23"],
)
```

**Step 5: Run test to verify it passes**

Run: `bazel test //tests:iv_error_types_test --test_output=all`
Expected: PASS (2 tests)

**Step 6: Commit**

```bash
git add src/support/error_types.hpp tests/iv_error_types_test.cc tests/BUILD.bazel
git commit -m "Add IVError types for std::expected refactoring

Define IVErrorCode enum with validation, convergence, and solver errors.
Add IVError struct with diagnostics (iterations, final_error, last_vol).
Add comprehensive tests for error type construction."
```

---

### Task 1.2: Create IVSuccess type

**Files:**
- Create: `src/option/iv_result.hpp` (new file)
- Test: `tests/iv_result_test.cc` (new file)

**Step 1: Write the failing test**

Create test file `tests/iv_result_test.cc`:

```cpp
#include <gtest/gtest.h>
#include "src/option/iv_result.hpp"

using namespace mango;

TEST(IVResult, SuccessConstruction) {
    IVSuccess success{
        .implied_vol = 0.25,
        .iterations = 12,
        .final_error = 1e-8,
        .vega = 15.3
    };

    EXPECT_EQ(success.implied_vol, 0.25);
    EXPECT_EQ(success.iterations, 12);
    EXPECT_EQ(success.final_error, 1e-8);
    EXPECT_TRUE(success.vega.has_value());
    EXPECT_EQ(success.vega.value(), 15.3);
}

TEST(IVResult, SuccessWithoutVega) {
    IVSuccess success{
        .implied_vol = 0.20,
        .iterations = 8,
        .final_error = 5e-9,
        .vega = std::nullopt
    };

    EXPECT_EQ(success.implied_vol, 0.20);
    EXPECT_FALSE(success.vega.has_value());
}
```

**Step 2: Run test to verify it fails**

Run: `bazel test //tests:iv_result_test --test_output=all`
Expected: BUILD FAIL - "iv_result.hpp: No such file or directory"

**Step 3: Create IVSuccess type**

Create file `src/option/iv_result.hpp`:

```cpp
/**
 * @file iv_result.hpp
 * @brief Success and error types for IV solver std::expected API
 */

#pragma once

#include <cstddef>
#include <optional>

namespace mango {

/// Success result from IV solver
struct IVSuccess {
    double implied_vol;              ///< Solved implied volatility
    size_t iterations;               ///< Number of iterations taken
    double final_error;              ///< |Price(σ) - Market_Price|
    std::optional<double> vega;      ///< Vega at solution (optional)
};

}  // namespace mango
```

**Step 4: Add to BUILD.bazel**

In `src/option/BUILD.bazel`, find the `cc_library` with `name = "iv_types"` and add:

```python
hdrs = [
    "iv_types.hpp",
    "iv_result.hpp",  # Add this
],
```

In `tests/BUILD.bazel`, add:

```python
cc_test(
    name = "iv_result_test",
    size = "small",
    srcs = ["iv_result_test.cc"],
    deps = [
        "//src/option:iv_types",
        "@googletest//:gtest_main",
    ],
    copts = ["-std=c++23"],
)
```

**Step 5: Run test to verify it passes**

Run: `bazel test //tests:iv_result_test --test_output=all`
Expected: PASS (2 tests)

**Step 6: Commit**

```bash
git add src/option/iv_result.hpp src/option/BUILD.bazel tests/iv_result_test.cc tests/BUILD.bazel
git commit -m "Add IVSuccess type for std::expected success path

Define IVSuccess struct with implied_vol, iterations, final_error, vega.
Add tests verifying construction with and without optional vega.
Prepare for std::expected<IVSuccess, IVError> API."
```

---

## Phase 2: Update IVSolverFDM API

### Task 2.1: Update solve() signature

**Files:**
- Modify: `src/option/iv_solver_fdm.hpp:121` (solve_impl declaration)
- Modify: `src/option/iv_solver_fdm.cpp` (solve_impl implementation)

**Step 1: Write the failing test**

In `tests/iv_solver_test.cc`, add at the top:

```cpp
TEST(IVSolverFDM, ExpectedAPISuccess) {
    OptionSpec spec{
        .spot = 100.0,
        .strike = 100.0,
        .maturity = 1.0,
        .rate = 0.05,
        .dividend_yield = 0.02,
        .type = OptionType::PUT
    };

    IVQuery query{.option = spec, .market_price = 10.45};

    IVSolverFDMConfig config;
    IVSolverFDM solver(config);

    auto result = solver.solve(query);

    // Should return std::expected
    ASSERT_TRUE(result.has_value());
    EXPECT_GT(result->implied_vol, 0.0);
    EXPECT_GT(result->iterations, 0);
    EXPECT_LT(result->final_error, 1e-4);
}

TEST(IVSolverFDM, ExpectedAPIValidationError) {
    OptionSpec spec{
        .spot = -100.0,  // Invalid: negative spot
        .strike = 100.0,
        .maturity = 1.0,
        .rate = 0.05,
        .dividend_yield = 0.02,
        .type = OptionType::PUT
    };

    IVQuery query{.option = spec, .market_price = 10.45};

    IVSolverFDMConfig config;
    IVSolverFDM solver(config);

    auto result = solver.solve(query);

    // Should return error
    ASSERT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code, IVErrorCode::NegativeSpot);
    EXPECT_FALSE(result.error().message.empty());
}
```

**Step 2: Run test to verify it fails**

Run: `bazel test //tests:iv_solver_test --test_output=all`
Expected: BUILD FAIL - "result.has_value() not valid for IVResult"

**Step 3: Update header signature**

In `src/option/iv_solver_fdm.hpp`, change line 121:

```cpp
// OLD:
IVResult solve_impl(const IVQuery& query);

// NEW:
std::expected<IVSuccess, IVError> solve_impl(const IVQuery& query);
```

Add includes at top:

```cpp
#include "src/option/iv_result.hpp"
#include "src/support/error_types.hpp"
#include <expected>
```

**Step 4: Update implementation skeleton**

In `src/option/iv_solver_fdm.cpp`, update `solve_impl`:

```cpp
std::expected<IVSuccess, IVError> IVSolverFDM::solve_impl(const IVQuery& query) {
    // Validate query
    auto validation = validate_query(query);
    if (!validation.has_value()) {
        // Convert validation error to IVError
        IVErrorCode code = IVErrorCode::NegativeSpot;  // Map appropriately
        return std::unexpected(IVError{
            .code = code,
            .message = validation.error(),
            .iterations = 0,
            .final_error = 0.0,
            .last_vol = std::nullopt
        });
    }

    // TODO: Implement Brent solving (next task)
    // For now, return dummy success to make tests compile
    return IVSuccess{
        .implied_vol = 0.20,
        .iterations = 1,
        .final_error = 0.0,
        .vega = std::nullopt
    };
}
```

**Step 5: Run test to verify it compiles**

Run: `bazel test //tests:iv_solver_test --test_output=all`
Expected: COMPILE SUCCESS (tests may fail - that's OK, we're just checking compilation)

**Step 6: Commit**

```bash
git add src/option/iv_solver_fdm.hpp src/option/iv_solver_fdm.cpp tests/iv_solver_test.cc
git commit -m "Update IVSolverFDM::solve() to return std::expected

Change return type from IVResult to std::expected<IVSuccess, IVError>.
Add skeleton validation error handling.
Add tests for expected API (success and validation error paths)."
```

---

### Task 2.2: Implement validation error mapping

**Files:**
- Modify: `src/option/iv_solver_fdm.cpp:validate_query()` implementation

**Step 1: Current validation returns std::expected<void, std::string>**

No test needed - we'll map existing validation to IVErrorCode.

**Step 2: Create validation error mapper**

In `src/option/iv_solver_fdm.cpp`, add helper function before `solve_impl`:

```cpp
namespace {

/// Map validation error message to IVErrorCode
IVErrorCode classify_validation_error(const std::string& message) {
    if (message.find("spot") != std::string::npos &&
        message.find("negative") != std::string::npos) {
        return IVErrorCode::NegativeSpot;
    }
    if (message.find("strike") != std::string::npos) {
        return IVErrorCode::NegativeStrike;
    }
    if (message.find("maturity") != std::string::npos) {
        return IVErrorCode::NegativeMaturity;
    }
    if (message.find("market price") != std::string::npos) {
        return IVErrorCode::NegativeMarketPrice;
    }
    if (message.find("arbitrage") != std::string::npos) {
        return IVErrorCode::ArbitrageViolation;
    }
    // Default: arbitrage violation (most common)
    return IVErrorCode::ArbitrageViolation;
}

}  // anonymous namespace
```

**Step 3: Use mapper in solve_impl**

Update `solve_impl` validation handling:

```cpp
std::expected<IVSuccess, IVError> IVSolverFDM::solve_impl(const IVQuery& query) {
    // Validate query
    auto validation = validate_query(query);
    if (!validation.has_value()) {
        IVErrorCode code = classify_validation_error(validation.error());
        return std::unexpected(IVError{
            .code = code,
            .message = validation.error(),
            .iterations = 0,
            .final_error = 0.0,
            .last_vol = std::nullopt
        });
    }

    // ... rest of implementation
}
```

**Step 4: Run test to verify validation errors work**

Run: `bazel test //tests:iv_solver_test --test_filter=*ValidationError --test_output=all`
Expected: PASS - validation error test should now pass with correct error code

**Step 5: Commit**

```bash
git add src/option/iv_solver_fdm.cpp
git commit -m "Add validation error classification for IVError

Map validation error messages to specific IVErrorCode values.
Enables precise error reporting for negative spot, strike, maturity, etc.
Validation error test now passes with correct error codes."
```

---

### Task 2.3: Implement Brent solver with expected return

**Files:**
- Modify: `src/option/iv_solver_fdm.cpp:solve_impl()` (Brent loop)

**Step 1: Tests already exist from Task 2.1**

The test `ExpectedAPISuccess` expects convergence, iterations > 0, and small final_error.

**Step 2: Implement Brent solving logic**

In `src/option/iv_solver_fdm.cpp`, replace the TODO in `solve_impl`:

```cpp
std::expected<IVSuccess, IVError> IVSolverFDM::solve_impl(const IVQuery& query) {
    // Validation (already done in Task 2.2)
    auto validation = validate_query(query);
    if (!validation.has_value()) {
        IVErrorCode code = classify_validation_error(validation.error());
        return std::unexpected(IVError{
            .code = code,
            .message = validation.error(),
            .iterations = 0,
            .final_error = 0.0,
            .last_vol = std::nullopt
        });
    }

    // Estimate bounds
    double vol_lower = estimate_lower_bound();
    double vol_upper = estimate_upper_bound(query);

    // Brent's method objective
    auto objective = [this, &query](double vol) -> double {
        return this->objective_function(query, vol);
    };

    // Run Brent solver
    auto brent_result = brent_root_find(
        objective, vol_lower, vol_upper,
        config_.root_config.tolerance,
        config_.root_config.max_iter
    );

    // Check convergence
    if (!brent_result.converged) {
        return std::unexpected(IVError{
            .code = IVErrorCode::MaxIterationsExceeded,
            .message = "Brent's method failed to converge after " +
                       std::to_string(brent_result.iterations) + " iterations",
            .iterations = brent_result.iterations,
            .final_error = std::abs(brent_result.residual),
            .last_vol = brent_result.root
        });
    }

    // Success
    return IVSuccess{
        .implied_vol = brent_result.root,
        .iterations = brent_result.iterations,
        .final_error = std::abs(brent_result.residual),
        .vega = std::nullopt  // TODO: Add vega computation later
    };
}
```

**Step 3: Run tests to verify convergence**

Run: `bazel test //tests:iv_solver_test --test_filter=*Success --test_output=all`
Expected: PASS - should converge with reasonable iterations and error

**Step 4: Commit**

```bash
git add src/option/iv_solver_fdm.cpp
git commit -m "Implement Brent solver with std::expected return

Replace dummy implementation with actual Brent root-finding.
Return IVSuccess on convergence with iterations and final_error.
Return IVError::MaxIterationsExceeded on failure with diagnostics."
```

---

## Phase 3: Update Batch API

### Task 3.1: Create BatchIVResult type

**Files:**
- Modify: `src/option/iv_result.hpp` (add BatchIVResult)
- Test: `tests/iv_result_test.cc`

**Step 1: Write the failing test**

Add to `tests/iv_result_test.cc`:

```cpp
TEST(IVResult, BatchResultConstruction) {
    std::vector<std::expected<IVSuccess, IVError>> results;
    results.push_back(IVSuccess{0.20, 10, 1e-8, std::nullopt});
    results.push_back(std::unexpected(IVError{
        IVErrorCode::MaxIterationsExceeded, "Failed", 100, 0.05, 0.25
    }));
    results.push_back(IVSuccess{0.25, 12, 5e-9, std::nullopt});

    BatchIVResult batch{std::move(results)};

    EXPECT_EQ(batch.results.size(), 3);
    EXPECT_EQ(batch.failed_count, 1);
    EXPECT_FALSE(batch.all_succeeded());
}

TEST(IVResult, BatchResultAllSuccess) {
    std::vector<std::expected<IVSuccess, IVError>> results;
    results.push_back(IVSuccess{0.20, 10, 1e-8, std::nullopt});
    results.push_back(IVSuccess{0.22, 11, 2e-8, std::nullopt});

    BatchIVResult batch{std::move(results)};

    EXPECT_EQ(batch.results.size(), 2);
    EXPECT_EQ(batch.failed_count, 0);
    EXPECT_TRUE(batch.all_succeeded());
}
```

**Step 2: Run test to verify it fails**

Run: `bazel test //tests:iv_result_test --test_output=all`
Expected: BUILD FAIL - "BatchIVResult was not declared"

**Step 3: Add BatchIVResult to header**

In `src/option/iv_result.hpp`, add after IVSuccess:

```cpp
#include <vector>
#include <expected>
#include "src/support/error_types.hpp"

/// Batch IV solver result
struct BatchIVResult {
    std::vector<std::expected<IVSuccess, IVError>> results;
    size_t failed_count;

    /// Construct from results vector
    explicit BatchIVResult(std::vector<std::expected<IVSuccess, IVError>> r)
        : results(std::move(r))
        , failed_count(0)
    {
        for (const auto& result : results) {
            if (!result.has_value()) {
                ++failed_count;
            }
        }
    }

    /// Check if all solves succeeded
    bool all_succeeded() const { return failed_count == 0; }
};
```

**Step 4: Run test to verify it passes**

Run: `bazel test //tests:iv_result_test --test_output=all`
Expected: PASS (4 tests total now)

**Step 5: Commit**

```bash
git add src/option/iv_result.hpp tests/iv_result_test.cc
git commit -m "Add BatchIVResult for batch IV solver API

Define BatchIVResult with vector of expected results and failure count.
Auto-compute failed_count in constructor.
Add all_succeeded() convenience method."
```

---

### Task 3.2: Update solve_batch API

**Files:**
- Modify: `src/option/iv_solver_fdm.hpp:131` (solve_batch_impl signature)
- Modify: `src/option/iv_solver_fdm.cpp` (solve_batch_impl implementation)

**Step 1: Write the failing test**

Add to `tests/iv_solver_test.cc`:

```cpp
TEST(IVSolverFDM, BatchExpectedAPI) {
    std::vector<IVQuery> queries;

    // Valid queries
    for (double strike : {95.0, 100.0, 105.0}) {
        OptionSpec spec{100.0, strike, 1.0, 0.05, 0.02, OptionType::PUT};
        queries.push_back(IVQuery{spec, 10.0 + (strike - 100.0) * 0.2});
    }

    // Invalid query (negative spot)
    queries.push_back(IVQuery{
        OptionSpec{-100.0, 100.0, 1.0, 0.05, 0.02, OptionType::PUT},
        10.0
    });

    IVSolverFDMConfig config;
    IVSolverFDM solver(config);

    auto batch = solver.solve_batch(queries);

    EXPECT_EQ(batch.results.size(), 4);
    EXPECT_EQ(batch.failed_count, 1);
    EXPECT_FALSE(batch.all_succeeded());

    // First 3 should succeed
    EXPECT_TRUE(batch.results[0].has_value());
    EXPECT_TRUE(batch.results[1].has_value());
    EXPECT_TRUE(batch.results[2].has_value());

    // Last should fail with validation error
    EXPECT_FALSE(batch.results[3].has_value());
    EXPECT_EQ(batch.results[3].error().code, IVErrorCode::NegativeSpot);
}
```

**Step 2: Run test to verify it fails**

Run: `bazel test //tests:iv_solver_test --test_filter=*BatchExpected* --test_output=all`
Expected: BUILD FAIL - type mismatch

**Step 3: Update header signature**

In `src/option/iv_solver_fdm.hpp`, change line 131:

```cpp
// OLD:
void solve_batch_impl(std::span<const IVQuery> queries,
                     std::span<IVResult> results);

// NEW:
BatchIVResult solve_batch_impl(std::span<const IVQuery> queries);
```

**Step 4: Update base class if needed**

In `src/option/iv_solver_base.hpp`, update the public `solve_batch` wrapper:

```cpp
BatchIVResult solve_batch(std::span<const IVQuery> queries) {
    return solve_batch_impl(queries);
}
```

**Step 5: Update implementation**

In `src/option/iv_solver_fdm.cpp`, rewrite `solve_batch_impl`:

```cpp
BatchIVResult IVSolverFDM::solve_batch_impl(std::span<const IVQuery> queries) {
    std::vector<std::expected<IVSuccess, IVError>> results;
    results.reserve(queries.size());

    #pragma omp parallel
    {
        // Thread-local solver instance
        IVSolverFDM local_solver(config_);

        #pragma omp for
        for (size_t i = 0; i < queries.size(); ++i) {
            results[i] = local_solver.solve_impl(queries[i]);
        }
    }

    return BatchIVResult(std::move(results));
}
```

**Step 6: Run test to verify it passes**

Run: `bazel test //tests:iv_solver_test --test_filter=*BatchExpected* --test_output=all`
Expected: PASS

**Step 7: Commit**

```bash
git add src/option/iv_solver_fdm.hpp src/option/iv_solver_fdm.cpp src/option/iv_solver_base.hpp tests/iv_solver_test.cc
git commit -m "Update solve_batch to return BatchIVResult

Change API from void fill to BatchIVResult return.
Use OpenMP with thread-local solvers for parallelism.
Add test verifying batch with mixed success/failure."
```

---

## Phase 4: Backward Compatibility & Migration

### Task 4.1: Add deprecated legacy API

**Files:**
- Modify: `src/option/iv_solver_fdm.hpp` (add deprecated wrapper)
- Modify: `src/option/iv_solver_fdm.cpp` (implement wrapper)

**Step 1: No test needed**

Backward compatibility - existing tests should still work.

**Step 2: Add deprecated method to header**

In `src/option/iv_solver_fdm.hpp`, add public method:

```cpp
/// Solve for implied volatility (legacy API - deprecated)
///
/// @deprecated Use solve() returning std::expected instead
[[deprecated("Use solve() returning std::expected<IVSuccess, IVError> instead")]]
IVResult solve_legacy(const IVQuery& query);
```

**Step 3: Implement conversion wrapper**

In `src/option/iv_solver_fdm.cpp`:

```cpp
IVResult IVSolverFDM::solve_legacy(const IVQuery& query) {
    auto result = solve_impl(query);

    if (result.has_value()) {
        return IVResult{
            .converged = true,
            .iterations = result->iterations,
            .implied_vol = result->implied_vol,
            .final_error = result->final_error,
            .failure_reason = std::nullopt,
            .vega = result->vega
        };
    } else {
        return IVResult{
            .converged = false,
            .iterations = result.error().iterations,
            .implied_vol = 0.0,
            .final_error = result.error().final_error,
            .failure_reason = result.error().message,
            .vega = std::nullopt
        };
    }
}
```

**Step 4: Build to verify no breakage**

Run: `bazel build //src/option:iv_solver_fdm`
Expected: BUILD SUCCESS (with deprecation warnings)

**Step 5: Commit**

```bash
git add src/option/iv_solver_fdm.hpp src/option/iv_solver_fdm.cpp
git commit -m "Add deprecated legacy API for backward compatibility

Provide solve_legacy() that converts std::expected to IVResult.
Marked with [[deprecated]] attribute.
Enables gradual migration of existing call sites."
```

---

### Task 4.2: Update existing tests to new API

**Files:**
- Modify: `tests/iv_solver_test.cc` (update all old-style tests)

**Step 1: Identify old-style tests**

Run: `rg "\.converged" tests/iv_solver_test.cc`

This finds all tests using the old `IVResult.converged` pattern.

**Step 2: Update one test as example**

Find a test like:

```cpp
TEST(IVSolverFDM, ATMPut) {
    // ... setup ...
    auto result = solver.solve(query);
    EXPECT_TRUE(result.converged);
    EXPECT_GT(result.implied_vol, 0.0);
}
```

Change to:

```cpp
TEST(IVSolverFDM, ATMPut) {
    // ... setup ...
    auto result = solver.solve(query);
    ASSERT_TRUE(result.has_value());
    EXPECT_GT(result->implied_vol, 0.0);
}
```

**Step 3: Apply pattern to all remaining tests**

Use search/replace or manual update for each test:
- `EXPECT_TRUE(result.converged)` → `ASSERT_TRUE(result.has_value())`
- `EXPECT_FALSE(result.converged)` → `ASSERT_FALSE(result.has_value())`
- `result.implied_vol` → `result->implied_vol`
- `result.iterations` → `result->iterations`
- `result.failure_reason` → `result.error().message`

**Step 4: Run all tests**

Run: `bazel test //tests:iv_solver_test --test_output=all`
Expected: ALL PASS

**Step 5: Commit**

```bash
git add tests/iv_solver_test.cc
git commit -m "Update IV solver tests to use std::expected API

Replace IVResult.converged checks with result.has_value().
Use operator-> for success data access.
Use result.error() for failure data access.
All tests passing with new API."
```

---

## Phase 5: Documentation & Cleanup

### Task 5.1: Update API documentation

**Files:**
- Modify: `src/option/iv_solver_fdm.hpp` (class-level docs)
- Modify: `CLAUDE.md` (update IV solver section)

**Step 1: Update class documentation**

In `src/option/iv_solver_fdm.hpp`, update the class comment (around line 48):

```cpp
/// **Usage:**
/// ```cpp
/// OptionSpec spec{
///     .spot = 100.0,
///     .strike = 100.0,
///     .maturity = 1.0,
///     .rate = 0.05,
///     .dividend_yield = 0.02,
///     .type = OptionType::PUT
/// };
///
/// IVQuery query{.option = spec, .market_price = 10.45};
///
/// IVSolverFDMConfig config;
/// IVSolverFDM solver(config);
/// auto result = solver.solve(query);
///
/// if (result.has_value()) {
///     std::cout << "IV: " << result->implied_vol << "\n";
///     std::cout << "Converged in " << result->iterations << " iterations\n";
/// } else {
///     std::cerr << "Error: " << result.error().message << "\n";
///     std::cerr << "Error code: " << static_cast<int>(result.error().code) << "\n";
/// }
/// ```
///
/// **Batch Usage:**
/// ```cpp
/// std::vector<IVQuery> queries = { ... };
///
/// auto batch = solver.solve_batch(queries);
///
/// for (size_t i = 0; i < batch.results.size(); ++i) {
///     if (batch.results[i].has_value()) {
///         std::cout << "IV[" << i << "]: " << batch.results[i]->implied_vol << "\n";
///     } else {
///         std::cerr << "Failed[" << i << "]: " << batch.results[i].error().message << "\n";
///     }
/// }
/// ```
```

**Step 2: Update CLAUDE.md**

In `CLAUDE.md`, find the "Implied Volatility Solver" section and update:

```markdown
### Basic Usage

```cpp
#include "src/option/iv_solver_fdm.hpp"

// Setup option parameters
OptionSpec spec{
    .spot = 100.0,
    .strike = 100.0,
    .maturity = 1.0,
    .rate = 0.05,
    .dividend_yield = 0.02,
    .type = OptionType::PUT
};

IVQuery query{.option = spec, .market_price = 10.45};

IVSolverFDMConfig config;
IVSolverFDM solver(config);
auto result = solver.solve(query);

if (result.has_value()) {
    std::cout << "Implied Volatility: " << result->implied_vol << "\n";
    std::cout << "Iterations: " << result->iterations << "\n";
    std::cout << "Final Error: " << result->final_error << "\n";
} else {
    const auto& error = result.error();
    std::cerr << "Failed: " << error.message << "\n";
    std::cerr << "Error code: " << static_cast<int>(error.code) << "\n";
}
```

### Error Handling

```cpp
auto result = solver.solve(query);

if (!result.has_value()) {
    const auto& error = result.error();

    switch (error.code) {
        case IVErrorCode::NegativeSpot:
        case IVErrorCode::NegativeStrike:
        case IVErrorCode::ArbitrageViolation:
            // Validation error - fix input
            break;

        case IVErrorCode::MaxIterationsExceeded:
            // Convergence issue - try different bounds or tolerance
            std::cerr << "Iterations: " << error.iterations << "\n";
            std::cerr << "Final error: " << error.final_error << "\n";
            break;

        case IVErrorCode::PDESolveFailed:
            // Underlying PDE solver issue
            break;
    }
}
```
```

**Step 3: Commit documentation**

```bash
git add src/option/iv_solver_fdm.hpp CLAUDE.md
git commit -m "Update IV solver API documentation

Update class-level usage examples to use std::expected.
Add error handling examples with switch on error codes.
Update CLAUDE.md with new API patterns."
```

---

### Task 5.2: Remove old IVResult type

**Files:**
- Modify: `src/option/iv_types.hpp` (mark IVResult deprecated)

**Step 1: Mark IVResult as deprecated**

In `src/option/iv_types.hpp`:

```cpp
/// Unified result type for implied volatility solvers (DEPRECATED)
/// @deprecated Use std::expected<IVSuccess, IVError> instead
[[deprecated("Use std::expected<IVSuccess, IVError> from iv_result.hpp instead")]]
struct IVResult {
    bool converged = false;
    std::size_t iterations = 0;
    double implied_vol = 0.0;
    double final_error = 0.0;
    std::optional<std::string> failure_reason;
    std::optional<double> vega;
};
```

**Step 2: Build to verify deprecation warnings**

Run: `bazel build //... 2>&1 | grep -i "deprecated"`
Expected: See deprecation warnings for any remaining IVResult usage

**Step 3: Commit deprecation**

```bash
git add src/option/iv_types.hpp
git commit -m "Deprecate IVResult in favor of std::expected API

Mark IVResult struct with [[deprecated]] attribute.
Directs users to std::expected<IVSuccess, IVError>.
Prepares for eventual removal in next major version."
```

---

## Phase 6: Integration Testing

### Task 6.1: Add end-to-end integration test

**Files:**
- Create: `tests/iv_solver_integration_test.cc`

**Step 1: Write comprehensive integration test**

Create `tests/iv_solver_integration_test.cc`:

```cpp
#include <gtest/gtest.h>
#include "src/option/iv_solver_fdm.hpp"
#include "src/option/american_option.hpp"

using namespace mango;

TEST(IVSolverIntegration, RoundTrip_PriceToIV) {
    // 1. Price an option
    PricingParams params{
        .spot = 100.0,
        .strike = 100.0,
        .maturity = 1.0,
        .rate = 0.05,
        .dividend_yield = 0.02,
        .type = OptionType::PUT,
        .volatility = 0.25
    };

    auto price_result = solve_american_option_auto(params);
    ASSERT_TRUE(price_result.has_value());
    double market_price = price_result->value();

    // 2. Recover IV from price
    OptionSpec spec{
        .spot = params.spot,
        .strike = params.strike,
        .maturity = params.maturity,
        .rate = params.rate,
        .dividend_yield = params.dividend_yield,
        .type = params.type
    };

    IVQuery query{.option = spec, .market_price = market_price};

    IVSolverFDMConfig config;
    IVSolverFDM solver(config);

    auto iv_result = solver.solve(query);

    // 3. Verify we recovered original volatility
    ASSERT_TRUE(iv_result.has_value());
    EXPECT_NEAR(iv_result->implied_vol, 0.25, 0.01);  // Within 1% (grid discretization)
    EXPECT_LT(iv_result->final_error, 1e-4);
}

TEST(IVSolverIntegration, ErrorPropagation) {
    // Verify that PDE solve failures propagate correctly
    OptionSpec spec{
        .spot = 100.0,
        .strike = 100.0,
        .maturity = 1e-10,  // Extremely small maturity - may cause issues
        .rate = 0.05,
        .dividend_yield = 0.02,
        .type = OptionType::PUT
    };

    IVQuery query{.option = spec, .market_price = 5.0};

    IVSolverFDMConfig config;
    IVSolverFDM solver(config);

    auto result = solver.solve(query);

    // Should either succeed or fail gracefully with proper error
    if (!result.has_value()) {
        EXPECT_FALSE(result.error().message.empty());
        EXPECT_GE(result.error().iterations, 0);
    }
}

TEST(IVSolverIntegration, BatchConsistencyWithSerial) {
    // Verify batch results match serial solving
    std::vector<IVQuery> queries;
    std::vector<std::expected<IVSuccess, IVError>> serial_results;

    for (double strike : {90.0, 95.0, 100.0, 105.0, 110.0}) {
        OptionSpec spec{100.0, strike, 1.0, 0.05, 0.02, OptionType::PUT};
        IVQuery q{spec, 8.0 + (100.0 - strike) * 0.3};
        queries.push_back(q);
    }

    IVSolverFDMConfig config;
    IVSolverFDM solver(config);

    // Solve serially
    for (const auto& q : queries) {
        serial_results.push_back(solver.solve(q));
    }

    // Solve in batch
    auto batch = solver.solve_batch(queries);

    // Compare
    ASSERT_EQ(batch.results.size(), serial_results.size());

    for (size_t i = 0; i < queries.size(); ++i) {
        EXPECT_EQ(batch.results[i].has_value(), serial_results[i].has_value());

        if (batch.results[i].has_value() && serial_results[i].has_value()) {
            EXPECT_NEAR(batch.results[i]->implied_vol,
                       serial_results[i]->implied_vol,
                       0.001);
        }
    }
}
```

**Step 2: Add to BUILD.bazel**

In `tests/BUILD.bazel`:

```python
cc_test(
    name = "iv_solver_integration_test",
    size = "medium",
    srcs = ["iv_solver_integration_test.cc"],
    deps = [
        "//src/option:iv_solver_fdm",
        "//src/option:american_option",
        "@googletest//:gtest_main",
    ],
    copts = ["-std=c++23"],
)
```

**Step 3: Run integration tests**

Run: `bazel test //tests:iv_solver_integration_test --test_output=all`
Expected: PASS (all 3 tests)

**Step 4: Commit integration tests**

```bash
git add tests/iv_solver_integration_test.cc tests/BUILD.bazel
git commit -m "Add IV solver integration tests

Test round-trip: price → IV → price recovery.
Test error propagation from PDE solver failures.
Test batch/serial consistency.
Verifies end-to-end correctness of std::expected API."
```

---

## Summary & Verification

### Final Verification Checklist

Run all these commands to verify complete implementation:

```bash
# 1. All tests pass
bazel test //tests:iv_error_types_test --test_output=all
bazel test //tests:iv_result_test --test_output=all
bazel test //tests:iv_solver_test --test_output=all
bazel test //tests:iv_solver_integration_test --test_output=all

# 2. Full build succeeds
bazel build //...

# 3. No unexpected deprecation warnings (only IVResult should be deprecated)
bazel build //... 2>&1 | grep -i deprecated

# 4. Example usage compiles
cat > /tmp/test_iv_api.cc <<'EOF'
#include "src/option/iv_solver_fdm.hpp"
int main() {
    mango::OptionSpec spec{100, 100, 1, 0.05, 0.02, mango::OptionType::PUT};
    mango::IVQuery q{spec, 10.0};
    mango::IVSolverFDM solver({});
    auto result = solver.solve(q);
    return result.has_value() ? 0 : 1;
}
EOF
g++ -std=c++23 -I. /tmp/test_iv_api.cc -c
```

Expected:
- ✅ All tests pass
- ✅ Build succeeds
- ✅ Only IVResult shows deprecation warnings
- ✅ Example compiles

### What We Built

- **Type-safe error handling**: `std::expected<IVSuccess, IVError>`
- **Rich error information**: Error codes, diagnostics, iteration counts
- **Consistent API**: Mirrors AmericanOptionSolver pattern
- **Backward compatibility**: Deprecated wrapper for migration
- **Batch support**: Parallel solving with proper error handling
- **Comprehensive tests**: Unit, integration, and round-trip tests
- **Updated documentation**: Usage examples and error handling patterns

### Migration Path for Users

1. Update to use `std::expected` API (recommended)
2. Use deprecated `solve_legacy()` temporarily
3. Migrate gradually with compiler warnings
4. Remove IVResult usage before next major version
