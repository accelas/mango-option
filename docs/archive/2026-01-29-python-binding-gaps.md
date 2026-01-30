<!-- SPDX-License-Identifier: MIT -->
# Python Binding Gaps Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Expose batch option pricing, auto-grid estimation, discrete dividends, and yield curve support in the Python bindings.

**Architecture:** Modify `python/mango_bindings.cpp` to add `BatchAmericanOptionSolver` class binding and upgrade `american_option_price` to use `solve_american_option_auto()`. Add `//src/option:american_option_batch` to BUILD deps. No C++ library changes needed.

**Tech Stack:** pybind11, C++23, Bazel

---

### Task 1: Add batch solver dep to BUILD.bazel

**Files:**
- Modify: `python/BUILD.bazel`

**Step 1: Add the dependency**

Add `"//src/option:american_option_batch"` to the `deps` list in `python/BUILD.bazel`:

```python
deps = [
    "//src/option:iv_solver_fdm",
    "//src/option:iv_solver_interpolated",
    "//src/option:american_option",
    "//src/option:american_option_batch",  # NEW
    "//src/option:option_chain",
    "//src/option/table:price_table_builder",
    "//src/option/table:price_table_workspace",
    "//src/option/table:price_table_surface",
    "//src/math:root_finding",
    "//src/support:error_types",
],
```

**Step 2: Verify it compiles**

Run: `bazel build //python:mango_option`
Expected: BUILD SUCCESS

**Step 3: Commit**

```bash
git add python/BUILD.bazel
git commit -m "Add batch solver dep to Python bindings BUILD"
```

---

### Task 2: Expose GridAccuracyParams and BatchAmericanOptionSolver

**Files:**
- Modify: `python/mango_bindings.cpp` (add include + class bindings)

**Step 1: Add include**

At the top of `mango_bindings.cpp`, after the existing includes, add:

```cpp
#include "src/option/american_option_batch.hpp"
```

**Step 2: Bind GridAccuracyParams struct**

After the `GridAccuracyProfile` enum binding (line ~79), add:

```cpp
// GridAccuracyParams structure
py::class_<mango::GridAccuracyParams>(m, "GridAccuracyParams")
    .def(py::init<>())
    .def_readwrite("n_sigma", &mango::GridAccuracyParams::n_sigma)
    .def_readwrite("alpha", &mango::GridAccuracyParams::alpha)
    .def_readwrite("tol", &mango::GridAccuracyParams::tol)
    .def_readwrite("c_t", &mango::GridAccuracyParams::c_t)
    .def_readwrite("min_spatial_points", &mango::GridAccuracyParams::min_spatial_points)
    .def_readwrite("max_spatial_points", &mango::GridAccuracyParams::max_spatial_points)
    .def_readwrite("max_time_steps", &mango::GridAccuracyParams::max_time_steps);
```

**Step 3: Bind BatchAmericanOptionSolver class**

After the `american_option_price` function binding (after line ~333), add:

```cpp
// =========================================================================
// Batch American Option Solver (with normalized chain optimization)
// =========================================================================

py::class_<mango::BatchAmericanOptionSolver>(m, "BatchAmericanOptionSolver")
    .def(py::init<>())
    .def("set_grid_accuracy",
        [](mango::BatchAmericanOptionSolver& self, mango::GridAccuracyProfile profile) {
            self.set_grid_accuracy(mango::grid_accuracy_profile(profile));
            return &self;
        },
        py::arg("profile"),
        py::return_value_policy::reference_internal,
        R"pbdoc(
            Set grid accuracy using a profile.

            Args:
                profile: GridAccuracyProfile (LOW/MEDIUM/HIGH/ULTRA)

            Returns:
                Self for method chaining
        )pbdoc")
    .def("set_grid_accuracy_params",
        [](mango::BatchAmericanOptionSolver& self, const mango::GridAccuracyParams& params) {
            self.set_grid_accuracy(params);
            return &self;
        },
        py::arg("params"),
        py::return_value_policy::reference_internal,
        R"pbdoc(
            Set grid accuracy using explicit parameters.

            Args:
                params: GridAccuracyParams with fine-grained control

            Returns:
                Self for method chaining
        )pbdoc")
    .def("set_use_normalized",
        [](mango::BatchAmericanOptionSolver& self, bool enable) {
            self.set_use_normalized(enable);
            return &self;
        },
        py::arg("enable") = true,
        py::return_value_policy::reference_internal,
        "Enable/disable normalized chain optimization")
    .def("solve_batch",
        [](mango::BatchAmericanOptionSolver& self,
           const std::vector<mango::AmericanOptionParams>& params,
           bool use_shared_grid) {
            auto batch_result = self.solve_batch(params, use_shared_grid);

            py::list results;
            for (auto& r : batch_result.results) {
                if (r.has_value()) {
                    results.append(py::make_tuple(true, std::move(r.value()),
                        mango::SolverError{}));
                } else {
                    results.append(py::make_tuple(false, py::none(), r.error()));
                }
            }
            return py::make_tuple(results, batch_result.failed_count);
        },
        py::arg("params"),
        py::arg("use_shared_grid") = false,
        R"pbdoc(
            Solve a batch of American options in parallel.

            Automatically routes to the normalized chain solver when eligible
            (same maturity, same type, no discrete dividends, use_shared_grid=True).
            The normalized path solves one PDE and reuses it for all strikes.

            Args:
                params: List of AmericanOptionParams
                use_shared_grid: If True, all options share one global grid
                                 (required for normalized chain optimization)

            Returns:
                Tuple of (results, failed_count) where results is a list of
                (success: bool, result: AmericanOptionResult|None, error: SolverError) tuples
        )pbdoc");
```

**Step 4: Bind SolverError and SolverErrorCode for batch error reporting**

Before the `BatchAmericanOptionSolver` binding, add:

```cpp
// SolverErrorCode enum
py::enum_<mango::SolverErrorCode>(m, "SolverErrorCode")
    .value("ConvergenceFailure", mango::SolverErrorCode::ConvergenceFailure)
    .value("InvalidConfiguration", mango::SolverErrorCode::InvalidConfiguration)
    .value("NumericalInstability", mango::SolverErrorCode::NumericalInstability)
    .export_values();

// SolverError structure
py::class_<mango::SolverError>(m, "SolverError")
    .def(py::init<>())
    .def_readwrite("code", &mango::SolverError::code)
    .def_readwrite("iterations", &mango::SolverError::iterations)
    .def("__repr__", [](const mango::SolverError& e) {
        return "<SolverError code=" + std::to_string(static_cast<int>(e.code)) +
               " iterations=" + std::to_string(e.iterations) + ">";
    });
```

**Step 5: Verify it compiles**

Run: `bazel build //python:mango_option`
Expected: BUILD SUCCESS

**Step 6: Commit**

```bash
git add python/mango_bindings.cpp
git commit -m "Expose BatchAmericanOptionSolver in Python bindings"
```

---

### Task 3: Upgrade american_option_price to use auto-grid

**Files:**
- Modify: `python/mango_bindings.cpp` (replace `american_option_price` implementation)

**Step 1: Replace the american_option_price function**

Replace the existing `american_option_price` lambda (lines ~283-333) with:

```cpp
m.def(
    "american_option_price",
    [](const mango::AmericanOptionParams& params,
       std::optional<mango::GridAccuracyProfile> accuracy_profile) {
        mango::GridAccuracyParams accuracy;
        if (accuracy_profile.has_value()) {
            accuracy = mango::grid_accuracy_profile(accuracy_profile.value());
        }

        // Estimate grid automatically
        auto [grid_spec, time_domain] = mango::estimate_grid_for_option(params, accuracy);

        // Allocate workspace buffer
        size_t n = grid_spec.n_points();
        std::vector<double> buffer(mango::PDEWorkspace::required_size(n));

        auto workspace_result = mango::PDEWorkspace::from_buffer(buffer, n);
        if (!workspace_result) {
            throw py::value_error(
                "Failed to create workspace: " + workspace_result.error());
        }

        mango::AmericanOptionSolver solver(
            params, workspace_result.value(), std::nullopt,
            std::make_pair(grid_spec, time_domain));
        auto solve_result = solver.solve();
        if (!solve_result) {
            auto error = solve_result.error();
            throw py::value_error(
                "American option solve failed (error code " +
                std::to_string(static_cast<int>(error.code)) + ")");
        }

        return std::move(solve_result.value());
    },
    py::arg("params"),
    py::arg("accuracy") = py::none(),
    R"pbdoc(
        Price an American option using the PDE solver with automatic grid estimation.

        Uses sinh-spaced grids with clustering near the strike for optimal accuracy.
        Supports yield curves (via params.rate) and discrete dividends
        (via params.discrete_dividends).

        Args:
            params: AmericanOptionParams with contract and market parameters.
            accuracy: Optional GridAccuracyProfile (LOW/MEDIUM/HIGH/ULTRA).
                      If not specified, uses default parameters.

        Returns:
            AmericanOptionResult with value and Greeks.
    )pbdoc");
```

**Step 2: Verify it compiles**

Run: `bazel build //python:mango_option`
Expected: BUILD SUCCESS

**Step 3: Commit**

```bash
git add python/mango_bindings.cpp
git commit -m "Upgrade american_option_price to auto-grid estimation"
```

---

### Task 4: Add Python tests for new bindings

**Files:**
- Modify: `python/test_bindings.py`

**Step 1: Add test for auto-grid pricing**

Add after `test_american_option_price`:

```python
def test_american_option_price_with_accuracy():
    """Test american_option_price with accuracy profile"""
    print("Testing american_option_price with accuracy profile...")

    params = mango_option.AmericanOptionParams()
    params.strike = 100.0
    params.spot = 100.0
    params.maturity = 1.0
    params.volatility = 0.20
    params.rate = 0.05
    params.dividend_yield = 0.02
    params.type = mango_option.OptionType.PUT

    result = mango_option.american_option_price(
        params, accuracy=mango_option.GridAccuracyProfile.HIGH)
    price = result.value_at(100.0)
    assert price > 0, f"Expected positive price, got {price}"
    print(f"  Price (HIGH accuracy): {price:.6f}")
    print("  Delta:", result.delta())
    print("  Gamma:", result.gamma())
    print("  Theta:", result.theta())
    print("✓ Accuracy profile works")
```

**Step 2: Add test for discrete dividends**

```python
def test_american_option_discrete_dividends():
    """Test american_option_price with discrete dividends"""
    print("Testing american_option_price with discrete dividends...")

    params = mango_option.AmericanOptionParams()
    params.strike = 100.0
    params.spot = 100.0
    params.maturity = 1.0
    params.volatility = 0.20
    params.rate = 0.05
    params.dividend_yield = 0.0
    params.type = mango_option.OptionType.PUT
    params.discrete_dividends = [(0.25, 2.0), (0.75, 2.0)]

    result = mango_option.american_option_price(params)
    price_div = result.value_at(100.0)
    assert price_div > 0, f"Expected positive price, got {price_div}"

    # Compare with no dividends
    params.discrete_dividends = []
    result_no_div = mango_option.american_option_price(params)
    price_no_div = result_no_div.value_at(100.0)

    # Put with dividends should differ from no-dividend case
    print(f"  Price with dividends: {price_div:.6f}")
    print(f"  Price without dividends: {price_no_div:.6f}")
    print("✓ Discrete dividends work")
```

**Step 3: Add test for yield curve in pricing**

```python
def test_american_option_yield_curve():
    """Test american_option_price with yield curve rate"""
    print("Testing american_option_price with yield curve...")

    params = mango_option.AmericanOptionParams()
    params.strike = 100.0
    params.spot = 100.0
    params.maturity = 1.0
    params.volatility = 0.20
    params.rate = mango_option.YieldCurve.flat(0.05)
    params.dividend_yield = 0.02
    params.type = mango_option.OptionType.PUT

    result = mango_option.american_option_price(params)
    price = result.value_at(100.0)
    assert price > 0, f"Expected positive price, got {price}"
    print(f"  Price with YieldCurve: {price:.6f}")
    print("✓ Yield curve pricing works")
```

**Step 4: Add test for batch solver**

```python
def test_batch_solver():
    """Test BatchAmericanOptionSolver"""
    print("Testing BatchAmericanOptionSolver...")

    # Create batch of puts with same maturity, different strikes
    batch = []
    for K in [90.0, 95.0, 100.0, 105.0, 110.0]:
        p = mango_option.AmericanOptionParams()
        p.spot = 100.0
        p.strike = K
        p.maturity = 1.0
        p.volatility = 0.20
        p.rate = 0.05
        p.dividend_yield = 0.02
        p.type = mango_option.OptionType.PUT
        batch.append(p)

    solver = mango_option.BatchAmericanOptionSolver()
    solver.set_grid_accuracy(mango_option.GridAccuracyProfile.LOW)

    results, failed_count = solver.solve_batch(batch, use_shared_grid=True)
    assert failed_count == 0, f"Expected 0 failures, got {failed_count}"
    assert len(results) == 5, f"Expected 5 results, got {len(results)}"

    for i, (success, result, error) in enumerate(results):
        assert success, f"Option {i} failed: {error}"
        price = result.value_at(100.0)
        assert price > 0, f"Option {i}: expected positive price, got {price}"
        print(f"  K={batch[i].strike}: price={price:.4f}, delta={result.delta():.4f}")

    print(f"✓ Batch solver works ({len(results)} options, {failed_count} failed)")
```

**Step 5: Add test for batch solver with per-option grids**

```python
def test_batch_solver_per_option_grids():
    """Test BatchAmericanOptionSolver with per-option grid estimation"""
    print("Testing BatchAmericanOptionSolver with per-option grids...")

    batch = []
    for T in [0.25, 0.5, 1.0]:
        p = mango_option.AmericanOptionParams()
        p.spot = 100.0
        p.strike = 100.0
        p.maturity = T
        p.volatility = 0.20
        p.rate = 0.05
        p.dividend_yield = 0.02
        p.type = mango_option.OptionType.PUT
        batch.append(p)

    solver = mango_option.BatchAmericanOptionSolver()
    results, failed_count = solver.solve_batch(batch, use_shared_grid=False)
    assert failed_count == 0
    assert len(results) == 3

    for i, (success, result, error) in enumerate(results):
        assert success
        price = result.value_at(100.0)
        print(f"  T={batch[i].maturity}: price={price:.4f}")

    print("✓ Per-option grid batch works")
```

**Step 6: Register new tests in the test list**

Add the new test functions to the `tests` list at bottom of the file:

```python
tests = [
    test_option_types,
    test_yield_curve,
    test_iv_query,
    test_iv_solver_fdm,
    test_american_option_price,
    test_american_option_price_with_accuracy,   # NEW
    test_american_option_discrete_dividends,     # NEW
    test_american_option_yield_curve,            # NEW
    test_batch_solver,                           # NEW
    test_batch_solver_per_option_grids,          # NEW
    test_price_table_workspace,
    test_price_table_surface,
    test_iv_solver_interpolated,
    test_load_error_enum,
    test_error_handling,
    test_surface_to_solver_integration,
]
```

**Step 7: Build and run**

Run: `bazel build //python:mango_option`
Expected: BUILD SUCCESS

Run the Python tests manually:
```bash
PYTHONPATH=bazel-bin/python python3 python/test_bindings.py
```
Expected: All tests pass

**Step 8: Commit**

```bash
git add python/test_bindings.py
git commit -m "Add Python tests for batch solver and auto-grid pricing"
```

---

### Task 5: Verify full CI build

**Step 1: Run all C++ tests**

Run: `bazel test //...`
Expected: All tests pass (no regressions)

**Step 2: Build all targets from pre-PR checklist**

Run: `bazel build //examples/... && bazel build //benchmarks/...`
Expected: BUILD SUCCESS

**Step 3: Commit (if any fixes needed)**

Only if issues found in previous steps.
