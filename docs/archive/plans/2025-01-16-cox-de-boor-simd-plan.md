<!-- SPDX-License-Identifier: MIT -->
# Cox-de Boor SIMD Vectorization Implementation Plan

**Date**: 2025-01-16
**Phase**: 2 (after Banded Solver + PMR Workspace)
**Branch**: feature/cox-de-boor-simd
**Target**: 1.14× incremental speedup (2.45ms → 2.15ms)

## Executive Summary

Vectorize Cox-de Boor basis function evaluation using `std::experimental::simd` to achieve 2.5× speedup in basis evaluation hot path, translating to **1.14× end-to-end speedup** after Phases 0 and 1.

**Current bottleneck** (after Phase 0+1):
- Cox-de Boor evaluation: ~0.5ms (~20% of 2.45ms total runtime)
- Scalar recursion processes 4 basis functions sequentially
- High instruction-level parallelism opportunity (independent operations)

**Optimization strategy**:
- Vectorize 4 cubic basis functions simultaneously using SIMD
- Use `std::experimental::simd` for portable vectorization
- `[[gnu::target_clones]]` for automatic AVX2/AVX512 dispatch
- Expected 2.5× speedup in basis evaluation → 1.14× end-to-end

## Performance Targets

| Metric | Before (Phase 0+1) | After Phase 2 | Improvement |
|--------|-------------------|---------------|-------------|
| Cox-de Boor time | 0.5ms | 0.2ms | **2.5×** |
| Total fitting time (24K) | 86.7ms | ~76ms | **1.14×** |
| Combined speedup | 1.38× (Phase 1) | 1.57× | **10% incremental** |

**Combined Phases 0+1+2**: 5ms / 2.15ms = **2.33× total speedup**

## Background: Cox-de Boor Recursion

The Cox-de Boor recursion computes cubic B-spline basis functions:

```cpp
// Degree 0 (piecewise constant)
N_{i,0}(x) = 1 if t_i ≤ x < t_{i+1}, else 0

// Degree k (recursive)
N_{i,k}(x) = (x - t_i)/(t_{i+k} - t_i) * N_{i,k-1}(x)
           + (t_{i+k+1} - x)/(t_{i+k+1} - t_{i+1}) * N_{i+1,k-1}(x)
```

**For cubic B-splines** (degree 3), we compute 4 basis functions `N[0..3]` at each evaluation point.

**Current implementation** (scalar):
```cpp
// Evaluate 4 cubic basis functions sequentially
for (int j = 0; j < 4; ++j) {
    // Degree 0
    N[j] = (t[i-j] <= x && x < t[i-j+1]) ? 1.0 : 0.0;

    // Degrees 1-3 (recursive)
    for (int p = 1; p <= 3; ++p) {
        double left = /* ... */;
        double right = /* ... */;
        N[j] = left * N[j] + right * N[j+1];
    }
}
```

**Vectorization opportunity**: All 4 basis functions are independent → process simultaneously with SIMD.

## Implementation Tasks

### Task 1: Add SIMD basis function infrastructure

**Goal**: Create vectorized Cox-de Boor implementation using `std::experimental::simd`

**Files to modify**:
- `src/interpolation/bspline_fitter_4d.hpp`

**Steps**:

1. Add SIMD type aliases:
```cpp
#include <experimental/simd>

namespace stdx = std::experimental;

// SIMD types for 4-wide vectors (4 basis functions)
using simd4d = stdx::fixed_size_simd<double, 4>;
using simd4_mask = stdx::fixed_size_simd_mask<double, 4>;
```

2. Implement vectorized degree-0 initialization:
```cpp
[[gnu::target_clones("default","avx2","avx512f")]]
inline simd4d cubic_basis_degree0_simd(
    const std::vector<double>& t,
    int i,
    double x)
{
    // Gather knot values for 4 basis functions
    std::array<double, 4> t_left, t_right;
    for (int lane = 0; lane < 4; ++lane) {
        int idx = i - lane;
        t_left[lane] = t[idx];
        t_right[lane] = t[idx + 1];
    }

    // Load into SIMD vectors
    simd4d t_left_vec, t_right_vec;
    t_left_vec.copy_from(t_left.data(), stdx::element_aligned);
    t_right_vec.copy_from(t_right.data(), stdx::element_aligned);

    // Vectorized interval check: t_left <= x < t_right
    simd4d x_vec(x);  // Broadcast x to all lanes
    auto in_interval = (t_left_vec <= x_vec) && (x_vec < t_right_vec);

    // Return 1.0 if in interval, 0.0 otherwise
    return stdx::where(in_interval, simd4d(1.0), simd4d(0.0));
}
```

3. Implement vectorized recursive degrees 1-3:
```cpp
[[gnu::target_clones("default","avx2","avx512f")]]
inline void cubic_basis_nonuniform_simd(
    const std::vector<double>& t,
    int i,
    double x,
    double N[4])
{
    // Degree 0
    simd4d N_curr = cubic_basis_degree0_simd(t, i, x);
    simd4d N_next(0.0);  // N_{i+1,k-1} (shifted basis)

    // Degrees 1-3 (recursive)
    for (int p = 1; p <= 3; ++p) {
        // Gather denominator knot differences
        std::array<double, 4> denom_left, denom_right;
        for (int lane = 0; lane < 4; ++lane) {
            int idx = i - lane;
            denom_left[lane] = t[idx + p] - t[idx];
            denom_right[lane] = t[idx + p + 1] - t[idx + 1];
        }

        simd4d denom_left_vec, denom_right_vec;
        denom_left_vec.copy_from(denom_left.data(), stdx::element_aligned);
        denom_right_vec.copy_from(denom_right.data(), stdx::element_aligned);

        // Gather numerator knot values
        std::array<double, 4> t_base, t_end;
        for (int lane = 0; lane < 4; ++lane) {
            int idx = i - lane;
            t_base[lane] = t[idx];
            t_end[lane] = t[idx + p + 1];
        }

        simd4d t_base_vec, t_end_vec;
        t_base_vec.copy_from(t_base.data(), stdx::element_aligned);
        t_end_vec.copy_from(t_end.data(), stdx::element_aligned);

        // Compute left and right terms
        simd4d x_vec(x);
        simd4d left_num = x_vec - t_base_vec;
        simd4d right_num = t_end_vec - x_vec;

        // Handle division by zero (uniform knots)
        auto left_valid = denom_left_vec != simd4d(0.0);
        auto right_valid = denom_right_vec != simd4d(0.0);

        simd4d left_term = stdx::where(left_valid,
            (left_num / denom_left_vec) * N_curr,
            simd4d(0.0));

        simd4d right_term = stdx::where(right_valid,
            (right_num / denom_right_vec) * N_next,
            simd4d(0.0));

        // Update for next iteration (shift N_next)
        N_next = N_curr;
        N_curr = left_term + right_term;
    }

    // Store result
    N_curr.copy_to(N, stdx::element_aligned);
}
```

**Verification**:
- Add unit test comparing SIMD vs scalar results (< 1e-14 difference)
- Verify correctness on uniform and non-uniform knot sequences
- Test edge cases (x at knot boundaries, repeated knots)

---

### Task 2: Integrate SIMD basis functions into BSplineCollocation1D

**Goal**: Replace scalar Cox-de Boor calls with SIMD version

**Files to modify**:
- `src/interpolation/bspline_fitter_4d.hpp`

**Current scalar code** (in `build_collocation_matrix()`):
```cpp
for (size_t i = 0; i < n_; ++i) {
    double N[4];
    cubic_basis_nonuniform(knots_, i + k, data_points_[i], N);  // ← Scalar

    // Fill banded matrix row
    for (size_t j = 0; j < 4; ++j) {
        band_values_[i * 4 + j] = N[j];
    }
}
```

**Updated SIMD code**:
```cpp
for (size_t i = 0; i < n_; ++i) {
    alignas(32) double N[4];  // Align for SIMD loads/stores
    cubic_basis_nonuniform_simd(knots_, i + k, data_points_[i], N);  // ← SIMD

    // Fill banded matrix row (unchanged)
    for (size_t j = 0; j < 4; ++j) {
        band_values_[i * 4 + j] = N[j];
    }
}
```

**Key changes**:
- Replace `cubic_basis_nonuniform()` with `cubic_basis_nonuniform_simd()`
- Align output buffer for efficient SIMD stores
- No other changes needed (rest of solver unchanged)

**Verification**:
- All existing tests should pass (numerical results identical)
- Performance tests should show ~1.14× speedup

---

### Task 3: Add SIMD correctness tests

**Goal**: Comprehensive testing for SIMD basis functions

**Files to create**:
- `tests/bspline_simd_test.cc`

**Test cases**:

1. **Scalar vs SIMD equivalence**:
```cpp
TEST(BSplineSIMDTest, ScalarSIMDEquivalence) {
    // Test on various knot sequences
    std::vector<std::vector<double>> test_knots = {
        {0, 0, 0, 0, 1, 2, 3, 4, 4, 4, 4},  // Clamped cubic
        {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10}, // Uniform
        {0, 0, 0.5, 1, 1.5, 2.5, 3, 3, 3}   // Non-uniform
    };

    for (const auto& knots : test_knots) {
        for (double x = knots.front(); x <= knots.back(); x += 0.1) {
            double N_scalar[4], N_simd[4];

            cubic_basis_nonuniform(knots, 3, x, N_scalar);
            cubic_basis_nonuniform_simd(knots, 3, x, N_simd);

            for (int i = 0; i < 4; ++i) {
                EXPECT_NEAR(N_scalar[i], N_simd[i], 1e-14)
                    << "Mismatch at x=" << x << ", basis " << i;
            }
        }
    }
}
```

2. **Partition of unity** (basis functions sum to 1):
```cpp
TEST(BSplineSIMDTest, PartitionOfUnity) {
    std::vector<double> knots = {0, 0, 0, 0, 1, 2, 3, 4, 5, 6, 6, 6, 6};

    for (double x = 0.0; x < 6.0; x += 0.01) {
        double sum = 0.0;
        for (int i = 3; i < 9; ++i) {  // All contributing basis functions
            double N[4];
            cubic_basis_nonuniform_simd(knots, i, x, N);
            sum += N[0];  // Only first basis (others are adjacent)
        }

        EXPECT_NEAR(sum, 1.0, 1e-12) << "Partition of unity failed at x=" << x;
    }
}
```

3. **Edge cases**:
```cpp
TEST(BSplineSIMDTest, EdgeCases) {
    std::vector<double> knots = {0, 0, 0, 0, 1, 2, 3, 4, 4, 4, 4};

    // Test at knot boundaries
    for (double knot : {0.0, 1.0, 2.0, 3.0, 4.0}) {
        double N[4];
        cubic_basis_nonuniform_simd(knots, 3, knot, N);

        // Results should be well-defined (no NaN/Inf)
        for (int i = 0; i < 4; ++i) {
            EXPECT_TRUE(std::isfinite(N[i]));
        }
    }

    // Test with repeated knots (multiplicity)
    std::vector<double> repeated = {0, 0, 0, 0, 1, 1, 2, 3, 3, 3, 3};
    double N[4];
    cubic_basis_nonuniform_simd(repeated, 3, 1.5, N);

    for (int i = 0; i < 4; ++i) {
        EXPECT_TRUE(std::isfinite(N[i]));
    }
}
```

**Verification**:
- All 3 tests passing
- No numerical regressions vs scalar implementation

---

### Task 4: Add performance benchmark

**Goal**: Measure SIMD speedup in isolation and end-to-end

**Files to modify**:
- `tests/bspline_4d_end_to_end_performance_test.cc` (add new test)

**Benchmark structure**:

```cpp
TEST_F(BSpline4DEndToEndPerformanceTest, SIMDSpeedupRegression) {
    // Medium grid: 20×15×10×8 = 24K points
    auto moneyness = create_moneyness_grid(20);
    auto maturity = create_maturity_grid(15);
    auto volatility = create_volatility_grid(10);
    auto rate = create_rate_grid(8);

    auto values = generate_test_values(moneyness, maturity, volatility, rate);

    auto fitter_result = BSplineFitter4D::create(moneyness, maturity, volatility, rate);
    ASSERT_TRUE(fitter_result.has_value());

    // Run 5 times for stable measurement
    std::vector<double> times_us;
    for (int run = 0; run < 5; ++run) {
        auto start = std::chrono::high_resolution_clock::now();
        auto fit_result = fitter_result.value().fit(values, 1e-6);
        auto end = std::chrono::high_resolution_clock::now();

        ASSERT_TRUE(fit_result.success);
        times_us.push_back(
            std::chrono::duration_cast<std::chrono::microseconds>(end - start).count());
    }

    double mean = std::accumulate(times_us.begin(), times_us.end(), 0.0) / times_us.size();

    std::cout << "\nSIMD Performance (24K grid, 5 runs):\n";
    std::cout << "  Mean: " << mean << " µs (" << (mean / 1000.0) << " ms)\n";
    std::cout << "  Min: " << *std::min_element(times_us.begin(), times_us.end()) << " µs\n";
    std::cout << "  Max: " << *std::max_element(times_us.begin(), times_us.end()) << " µs\n";

    // Performance regression check
    // Baseline (Phase 0+1): 86.7ms
    // Target (Phase 0+1+2): ~76ms (1.14× speedup)
    // Allow 3× margin for CI variability
    EXPECT_LT(mean, 230000.0)  // <230ms (3× target)
        << "SIMD optimization performance regression";
}
```

**Expected results**:
- Mean time: ~76ms (vs 86.7ms baseline = 1.14× speedup)
- Confirms Cox-de Boor vectorization delivers expected improvement

---

### Task 5: Update documentation

**Goal**: Document SIMD optimization

**Files to create/modify**:
- `docs/plans/COX_DE_BOOR_SIMD_SUMMARY.md` (create)
- `CLAUDE.md` (update)

**Summary document structure** (following PMR workspace pattern):

1. **Executive Summary**: 1.14× speedup, SIMD vectorization
2. **Problem Statement**: Scalar Cox-de Boor bottleneck
3. **Solution Approach**: std::experimental::simd, target_clones
4. **Performance Results**: Measured speedup on realistic grids
5. **Implementation Details**: Files modified, key algorithms
6. **Testing Methodology**: SIMD correctness tests
7. **Key Technical Decisions**: Why std::simd vs intrinsics
8. **Future Work**: Phase 3 (re-entrancy) and Phase 4 (OpenMP)

**CLAUDE.md updates**:
- Add "Cox-de Boor SIMD Vectorization (Phase 2)" section after workspace optimization
- Performance table with measured results
- Code examples showing SIMD usage
- Reference to complete summary document

---

## Success Criteria

### Performance
- ✅ Medium grid (24K): ≤ 76ms (target: 1.14× speedup from 86.7ms)
- ✅ Combined speedup (Phases 0+1+2): ~2.33× vs original baseline

### Correctness
- ✅ All existing tests pass (48/48)
- ✅ SIMD tests pass: scalar-SIMD equivalence, partition of unity, edge cases
- ✅ Numerical accuracy: SIMD results match scalar to < 1e-14

### Code Quality
- ✅ Clean SIMD implementation using std::experimental::simd
- ✅ Portable across architectures (target_clones for AVX2/AVX512)
- ✅ Well-documented with inline comments
- ✅ No code duplication (SIMD and scalar paths separate)

## Risk Mitigation

### Risk: SIMD numerical accuracy issues
- **Mitigation**: Comprehensive tests comparing SIMD vs scalar
- **Test**: Partition of unity, edge cases, boundary conditions

### Risk: Compiler support for std::experimental::simd
- **Mitigation**: Falls back to scalar if not available
- **Test**: CI builds on different compilers/architectures

### Risk: Performance not meeting 1.14× target
- **Mitigation**: Profile to identify bottlenecks, adjust SIMD strategy
- **Fallback**: Keep scalar path as default, SIMD as opt-in

## Timeline

| Task | Estimated Time | Dependencies |
|------|---------------|--------------|
| 1. SIMD basis infrastructure | 4-6 hours | None |
| 2. Integration into solver | 2-3 hours | Task 1 |
| 3. SIMD correctness tests | 3-4 hours | Task 1 |
| 4. Performance benchmark | 1-2 hours | Tasks 1-2 |
| 5. Documentation | 2-3 hours | Tasks 1-4 |

**Total**: 12-18 hours (~2 days)

## References

- Design doc: `docs/plans/2025-01-14-bspline-fitter-optimization-design-v2.md`
- Phase 0 summary: `docs/plans/BSPLINE_BANDED_SOLVER_SUMMARY.md`
- Phase 1 summary: `docs/plans/PMR_WORKSPACE_SUMMARY.md`
- std::experimental::simd: [P0214R9](https://wg21.link/p0214r9)
- GCC target_clones: [GCC Documentation](https://gcc.gnu.org/onlinedocs/gcc/Common-Function-Attributes.html#index-target_005fclones-function-attribute)
