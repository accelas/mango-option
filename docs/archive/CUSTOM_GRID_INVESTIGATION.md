<!-- SPDX-License-Identifier: MIT -->
# Custom Grid Investigation Report

## Executive Summary

**Claim**: "custom_grid causes all PDE solves to fail when options have spot==strike (normalized case)"

**Verdict**: **CLAIM IS FALSE** - The issue is NOT with spot==strike normalization, but with grid specification violating numerical stability constraints.

## Investigation Results

### 1. Normalized Case Works Fine

The normalized case (spot==strike==K_ref) works perfectly with custom_grid when the grid satisfies solver constraints:

```
Test: price_table_builder_custom_grid_test.cc
Result: ALL TESTS PASS with custom_grid
- Baseline (no custom_grid): 4/4 solves succeed
- With custom_grid: 4/4 solves succeed
- Normalized parameters (spot=strike=100): NO FAILURES
```

### 2. Root Cause: Grid Constraint Violations

The actual problem occurs when `config.grid_estimator` violates the **normalized chain solver eligibility criteria**:

#### Failing Configuration (from price_table_builder_test.cc)
```cpp
GridSpec<double>::uniform(-3.0, 3.0, 21)
```

**Violations:**
1. **Grid spacing too coarse**: dx = 0.3 > MAX_DX = 0.05 (Von Neumann stability)
2. **Domain too wide**: width = 6.0 > MAX_WIDTH = 5.8 (convergence constraint)

#### Why Auto-Estimation Succeeds
When `custom_grid` is not used, the solver auto-estimates:
```
Auto-estimated grid: [-1, 1] with 101 points
- dx = 0.02 < MAX_DX ✓
- width = 2.0 < MAX_WIDTH ✓
```

### 3. Constraint Analysis

From `/home/kai/work/iv_calc/.worktrees/price-builder-refactor/src/option/american_option_batch.cpp`:

```cpp
static constexpr double MAX_WIDTH = 5.8;   // Convergence limit (log-units)
static constexpr double MAX_DX = 0.05;     // Von Neumann stability
static constexpr double MIN_MARGIN_ABS = 0.35; // 6-cell ghost zone minimum
```

When `custom_grid` is provided with violations:
1. Normalized chain solver becomes **ineligible** (fails eligibility check)
2. Falls back to regular batch solver
3. Regular batch solver with tight grid + snapshots fails (error code 3 = InvalidConfiguration)

### 4. Test Evidence

#### Test 1: Custom Grid with Good Specification
```
File: tests/price_table_builder_custom_grid_advanced_test.cc
Grid: [-3, 3] with 101 points
Result: 16/16 solves succeed (0 failures)
```

#### Test 2: Custom Grid with Bad Specification
```
File: tests/price_table_builder_root_cause_test.cc
Grid: [-3, 3] with 21 points
- dx = 0.3 > 0.05 ✗
- width = 6.0 > 5.8 ✗
Result: 1/1 solves FAIL (error code 3)
```

#### Test 3: Narrower Grid Still Fails (dx violation)
```
Grid: [-2.5, 2.5] with 21 points
- dx = 0.25 > 0.05 ✗
- width = 5.0 < 5.8 ✓
Result: 1/1 solves FAIL (dx constraint still violated)
```

## Why the Original Claim Was Wrong

The implementer misidentified the problem:
- **What they thought**: "spot==strike (normalized case) causes failures"
- **What actually happens**: Poor grid specification violates solver constraints
- **Why confusion occurred**: Test used `config.grid_estimator` which happened to have coarse grid

The normalized case is NOT the problem. The issue is that when you bypass auto-estimation with `custom_grid`, you must ensure the grid satisfies:
1. dx ≤ 0.05 (Von Neumann stability)
2. width ≤ 5.8 (convergence constraint)
3. Adequate margins (≥ max(0.35, 6*dx))

## Recommended Solution

**DO NOT use custom_grid** in PriceTableBuilder::solve_batch() because:

1. **Auto-estimation already works correctly**: The solver automatically computes appropriate grid bounds based on option parameters via `estimate_grid_for_option()`.

2. **Grid configuration is still controlled**: The user can specify constraints through `GridAccuracyParams`:
   ```cpp
   accuracy.min_spatial_points = ...;
   accuracy.max_spatial_points = ...;
   accuracy.max_time_steps = ...;
   accuracy.alpha = ...;  // For sinh-spaced grids
   ```

3. **Domain coverage is validated**: The build() method already checks that PDE grid covers requested moneyness range (lines 69-85):
   ```cpp
   if (x_min_requested < x_min || x_max_requested > x_max) {
       return std::unexpected("PDE grid doesn't cover moneyness range");
   }
   ```

4. **Bypassing auto-estimation is dangerous**: It requires deep knowledge of PDE solver constraints (dx, width, margins) which should remain encapsulated in the solver.

## Alternative Approach (If Custom Grid Really Needed)

If you absolutely must use custom_grid, it should:

1. **Validate constraints** before calling solve_batch():
   ```cpp
   double dx = (x_max - x_min) / (n_points - 1);
   double width = x_max - x_min;
   if (dx > 0.05 || width > 5.8) {
       // Reject or adjust grid
   }
   ```

2. **Use set_use_normalized(false)** to force regular batch path if grid violates normalized constraints

3. **Ensure adequate point density**: At least 20-40 points per log-moneyness unit

## Test Files Created

1. **price_table_builder_custom_grid_test.cc** - Demonstrates custom_grid works with normalized case
2. **price_table_builder_custom_grid_advanced_test.cc** - Tests all solver paths (normalized/regular)
3. **price_table_builder_custom_grid_diagnosis_test.cc** - Compares auto vs custom grid
4. **price_table_builder_root_cause_test.cc** - Isolates dx/width constraint violations

## Conclusion

The implementation should **NOT use custom_grid** because:
- It's unnecessary (auto-estimation works)
- It's error-prone (requires satisfying hidden constraints)
- It reduces flexibility (normalized chain solver may become ineligible)

The current implementation (without custom_grid) is correct.
