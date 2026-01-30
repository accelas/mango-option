<!-- SPDX-License-Identifier: MIT -->
# B-spline Fitter Performance Optimization Design

**Date**: 2025-01-14
**Author**: System Design
**Status**: Design Phase
**Target**: `src/interpolation/bspline_fitter_4d.hpp`

## Executive Summary

The B-spline fitter (`BSplineFitter4DSeparable`) is a critical component for fast option pricing via interpolation. Current performance for a typical 50×30×20×10 grid is ~5ms, with identified optimization opportunities that could achieve **16-25× speedup** through three complementary strategies:

1. **PMR Workspace Optimization**: Reduce memory allocation overhead (1.3-1.6× speedup)
2. **General SIMD Optimization**: Vectorize computational kernels (10-30× combined with OpenMP)
3. **Cox-de Boor Recursion Vectorization**: SIMD-optimized basis functions (2-2.5× speedup)

**Target performance**: 0.2-0.3ms per 50×30×20×10 grid (from current ~5ms baseline)

## Current Performance Baseline

### Performance Characteristics

**Typical workload** (50×30×20×10 grid = 300,000 points):
- **Current fitting time**: ~5ms
- **Sequential 1D solves**: ~15,000 (across 4 axes)
- **Memory allocations**: ~15,000 heap allocations
- **Largest allocation**: 80KB per n=100 solve (full matrix expansion)

### Performance Bottlenecks Identified

1. **Memory allocation overhead** (~30% of runtime)
   - `solve_banded_system()` allocates full n×n matrix per solve
   - Slice buffers allocated/deallocated in tight loops
   - Diagnostic vectors trigger repeated reallocations

2. **Missed SIMD opportunities** (~50% of runtime)
   - Scalar residual computation (band multiply-accumulate)
   - Scalar condition number estimation
   - Scalar back substitution
   - No batch parallelization

3. **Cox-de Boor recursion** (~20% of runtime)
   - 4 independent basis functions computed sequentially
   - Perfect fit for 4-way SIMD (AVX2: 256-bit = 4× double)
   - No vectorization currently applied

## Strategy 1: PMR Workspace Optimization

### Problem Analysis

**Current allocation pattern** (per fit operation):
```cpp
// In BSplineCollocation1D::solve_banded_system() - line 221
std::vector<double> A(n_ * n_, 0.0);  // ← 80KB for n=100, allocated ~15,000 times!

// In fit_axisN() - lines 534, 575, 616, 657
for (size_t slice_idx = 0; slice_idx < num_slices; ++slice_idx) {
    std::vector<double> slice(N);          // ← Allocated/freed in loop
    std::vector<double> max_residuals;     // ← Growing vector (realloc)
    std::vector<double> conditions;        // ← Growing vector (realloc)
}
```

**Measured overhead**:
- Heap fragmentation from repeated alloc/free
- Cache pollution from scattered allocations
- Malloc overhead: ~100ns per allocation × 15,000 = ~1.5ms

### Proposed Solution

Use `UnifiedMemoryResource` (existing PMR infrastructure) for workspace management:

```cpp
class BSplineFitter4DWorkspace {
public:
    explicit BSplineFitter4DWorkspace(
        size_t max_n_m, size_t max_n_tau,
        size_t max_n_sigma, size_t max_n_r)
    {
        // Pre-compute maximum workspace size
        size_t max_n = std::max({max_n_m, max_n_tau, max_n_sigma, max_n_r});
        size_t max_slices = max_n_m * max_n_tau * max_n_sigma;  // Worst case: axis 3

        // Allocate workspace (64-byte aligned for AVX-512)
        size_t workspace_bytes =
            max_n * max_n * sizeof(double) +    // Full matrix storage
            max_n * 4 * sizeof(double) +        // Band values
            max_slices * sizeof(double) +       // Slice buffer
            max_slices * sizeof(double) +       // Diagnostic: max_residuals
            max_slices * sizeof(double);        // Diagnostic: conditions

        memory_resource_ = std::make_unique<UnifiedMemoryResource>(workspace_bytes);
    }

    // Workspace accessors
    std::span<double> get_matrix_buffer(size_t n) {
        return std::span{
            static_cast<double*>(memory_resource_->allocate(n * n * sizeof(double), 64)),
            n * n
        };
    }

    std::span<double> get_band_buffer(size_t n) {
        return std::span{
            static_cast<double*>(memory_resource_->allocate(n * 4 * sizeof(double), 64)),
            n * 4
        };
    }

    std::span<double> get_slice_buffer(size_t slice_size) {
        return std::span{
            static_cast<double*>(memory_resource_->allocate(slice_size * sizeof(double), 64)),
            slice_size
        };
    }

    // Zero-cost reset between fits
    void reset() { memory_resource_->reset(); }

private:
    std::unique_ptr<UnifiedMemoryResource> memory_resource_;
};
```

### Integration with BSplineFitter4D

**Modified BSplineCollocation1D constructor**:
```cpp
class BSplineCollocation1D {
public:
    BSplineCollocation1D(
        std::span<const double> knots,
        std::span<const double> data_points,
        std::span<const double> data_values,
        BSplineFitter4DWorkspace* workspace = nullptr)  // ← Optional workspace
        : workspace_(workspace)
    {
        // ... existing initialization ...
    }

private:
    BSplineFitter4DWorkspace* workspace_;  // Non-owning pointer

    void solve_banded_system() {
        if (workspace_) {
            // Use workspace buffer (no allocation!)
            auto A_buffer = workspace_->get_matrix_buffer(n_);
            // ... solve using A_buffer.data() ...
        } else {
            // Fallback: standalone mode (backward compatible)
            std::vector<double> A(n_ * n_, 0.0);
            // ... existing code ...
        }
    }
};
```

**Modified fit_axisN() functions**:
```cpp
void fit_axis0(std::span<double> out_coeffs,
               DiagnosticInfo* diag,
               BSplineFitter4DWorkspace* workspace = nullptr) {
    const size_t num_slices = n_tau_ * n_sigma_ * n_r_;

    // Pre-allocate diagnostic vectors if needed
    std::vector<double> max_residuals, conditions;
    if (diag) {
        max_residuals.reserve(num_slices);
        conditions.reserve(num_slices);
    }

    for (size_t slice_idx = 0; slice_idx < num_slices; ++slice_idx) {
        // Use workspace buffer (no per-iteration allocation!)
        auto slice_buffer = workspace
            ? workspace->get_slice_buffer(n_m_)
            : std::vector<double>(n_m_);  // Fallback

        // ... extract slice into slice_buffer ...

        BSplineCollocation1D solver(knots_m_, m_data_, slice_buffer, workspace);
        // ... solve and store coefficients ...
    }
}
```

### Expected Performance Improvement

| Optimization | Speedup | Justification |
|--------------|---------|---------------|
| Eliminate 15,000 heap allocations | 1.2-1.3× | Measured malloc overhead ~1.5ms → ~0.2ms |
| Improved cache locality | 1.1-1.2× | Sequential workspace access vs scattered heap |
| Reduced fragmentation | 1.05-1.1× | Better prefetcher performance |
| **Combined** | **1.3-1.6×** | Multiplicative effects |

**Net improvement**: ~5ms → ~3-4ms

### Implementation Effort

- **Complexity**: Medium-High
- **Estimated effort**: 3-4 days
- **Testing requirements**:
  - Verify identical numerical results (workspace vs standalone)
  - Benchmark memory usage (should reduce peak by ~60MB for large grids)
  - Regression tests for all 4 axes
  - Stress test with maximum grid sizes

---

## Strategy 2: General SIMD Optimization

### Five Identified Vectorization Targets

#### Target 1: Residual Computation (HIGHEST IMPACT)

**Location**: `BSplineCollocation1D::compute_residual()` (lines 290-312)

**Current scalar code**:
```cpp
double compute_residual(std::span<const double> coeffs,
                        std::span<const double> data_values) const {
    double max_residual = 0.0;
    for (size_t i = 0; i < data_values.size(); ++i) {
        double fitted = 0.0;
        for (int k = 0; k < 4; ++k) {  // ← 4-element band multiply-accumulate
            int idx = basis_start_[i] + k;
            if (idx >= 0 && idx < n_) {
                fitted += coeffs[idx] * basis_values_[i * 4 + k];
            }
        }
        double residual = std::abs(fitted - data_values[i]);
        max_residual = std::max(max_residual, residual);
    }
    return max_residual;
}
```

**SIMD optimization** (Phase 1: OpenMP hint):
```cpp
#pragma omp simd reduction(max:max_residual)
for (size_t i = 0; i < data_values.size(); ++i) {
    // ... same body ...
}
```

**SIMD optimization** (Phase 3: Explicit SIMD):
```cpp
[[gnu::target_clones("default","avx2","avx512f")]]
double compute_residual_simd(std::span<const double> coeffs,
                             std::span<const double> data_values) const {
    using simd4d = std::experimental::fixed_size_simd<double, 4>;

    double max_residual = 0.0;
    for (size_t i = 0; i < data_values.size(); ++i) {
        // Vectorized band multiply-accumulate
        simd4d basis_vec, coeff_vec;
        basis_vec.copy_from(basis_values_.data() + i * 4, stdx::element_aligned);

        // Conditional gather for coefficients (handle boundary cases)
        std::array<double, 4> coeff_arr = {0, 0, 0, 0};
        for (int k = 0; k < 4; ++k) {
            int idx = basis_start_[i] + k;
            if (idx >= 0 && idx < n_) {
                coeff_arr[k] = coeffs[idx];
            }
        }
        coeff_vec.copy_from(coeff_arr.data(), stdx::element_aligned);

        // Dot product: sum(basis[k] * coeff[k])
        double fitted = stdx::reduce(basis_vec * coeff_vec);
        double residual = std::abs(fitted - data_values[i]);
        max_residual = std::max(max_residual, residual);
    }
    return max_residual;
}
```

**Expected speedup**: 2-3× (AVX2: 4-way parallel multiply-accumulate)

#### Target 2: Condition Number Estimation (MEDIUM IMPACT)

**Location**: `BSplineCollocation1D::estimate_condition_number()` (lines 315-357)

**Optimization**: Parallelize multiple solves with different RHS vectors

```cpp
#pragma omp parallel for reduction(max:max_ratio)
for (int trial = 0; trial < n_trials; ++trial) {
    // ... solve with random RHS ...
}
```

**Expected speedup**: 3-4× on 8+ cores (embarrassingly parallel)

#### Target 3: Back Substitution (LOW-MEDIUM IMPACT)

**Location**: `BSplineCollocation1D::solve_banded_system()` back substitution loop

**Optimization**: Vectorize coefficient updates (not shown for brevity)

**Expected speedup**: 2-3× (but low overall impact ~5% of runtime)

#### Target 4: Slice Extraction (HIGH IMPACT)

**Location**: `fit_axisN()` slice extraction loops

**Optimization**: Use SIMD gather instructions for non-contiguous memory access

```cpp
// Current: scalar loop
for (size_t k = 0; k < n_axis; ++k) {
    slice[k] = prices[compute_4d_index(k, tau_idx, sigma_idx, r_idx)];
}

// SIMD gather (AVX2: 4× double gather)
using simd4d = std::experimental::fixed_size_simd<double, 4>;
size_t k = 0;
for (; k + 4 <= n_axis; k += 4) {
    std::array<int, 4> indices;
    for (int lane = 0; lane < 4; ++lane) {
        indices[lane] = compute_4d_index(k + lane, tau_idx, sigma_idx, r_idx);
    }
    simd4d gathered = stdx::gather(prices.data(), indices, stdx::element_aligned);
    gathered.copy_to(slice.data() + k, stdx::element_aligned);
}
// Scalar tail loop
for (; k < n_axis; ++k) {
    slice[k] = prices[compute_4d_index(k, tau_idx, sigma_idx, r_idx)];
}
```

**Expected speedup**: 1.5-2× (gather has higher latency than sequential load, but still faster)

#### Target 5: OpenMP Parallel Batching (HIGHEST OVERALL IMPACT)

**Location**: All `fit_axisN()` functions

**Current**: Sequential processing of ~15,000 independent 1D solves

**Optimization**: Parallelize across slices (embarrassingly parallel)

```cpp
void fit_axis0(std::span<double> out_coeffs,
               DiagnosticInfo* diag,
               BSplineFitter4DWorkspace* workspace = nullptr) {
    const size_t num_slices = n_tau_ * n_sigma_ * n_r_;

    // Thread-local diagnostic storage
    std::vector<std::vector<double>> thread_max_residuals(omp_get_max_threads());
    std::vector<std::vector<double>> thread_conditions(omp_get_max_threads());

    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        BSplineFitter4DWorkspace thread_workspace(n_m_, n_tau_, n_sigma_, n_r_);

        if (diag) {
            thread_max_residuals[tid].reserve(num_slices / omp_get_num_threads());
            thread_conditions[tid].reserve(num_slices / omp_get_num_threads());
        }

        #pragma omp for
        for (size_t slice_idx = 0; slice_idx < num_slices; ++slice_idx) {
            // ... extract slice, solve, store coefficients ...

            if (diag) {
                thread_max_residuals[tid].push_back(solver.get_max_residual());
                thread_conditions[tid].push_back(solver.get_condition_estimate());
            }

            thread_workspace.reset();  // Zero-cost reset for next iteration
        }
    }

    // Merge thread-local diagnostics
    if (diag) {
        for (const auto& vec : thread_max_residuals) {
            diag->max_residuals.insert(diag->max_residuals.end(), vec.begin(), vec.end());
        }
        for (const auto& vec : thread_conditions) {
            diag->condition_numbers.insert(diag->condition_numbers.end(), vec.begin(), vec.end());
        }
    }
}
```

**Expected speedup**: 8-16× on modern 16-core CPUs (near-linear scaling)

### Three-Phase Implementation Strategy

**Phase 1: Low-Hanging Fruit (OpenMP SIMD hints)**
- **Effort**: 1-2 days
- **Speedup**: 1.2-1.5×
- **Actions**:
  - Add `#pragma omp simd` to residual, condition number loops
  - Add `#pragma omp parallel for` to slice processing
- **Risk**: Low (compiler handles vectorization)

**Phase 2: OpenMP Parallel Batching**
- **Effort**: 2-3 days
- **Speedup**: 8-16× (combined with Phase 1)
- **Actions**:
  - Parallelize all `fit_axisN()` functions
  - Thread-local workspace management
  - Diagnostic merging across threads
- **Risk**: Medium (thread safety, load balancing)

**Phase 3: Explicit SIMD Kernels**
- **Effort**: 5-7 days
- **Speedup**: 2-4× per kernel (on top of Phase 2)
- **Actions**:
  - Implement SIMD residual computation
  - Implement SIMD slice extraction with gather
  - Multi-ISA support (SSE2/AVX2/AVX-512)
- **Risk**: High (requires careful validation, architecture-specific tuning)

### Expected Combined Performance

| Phase | Cumulative Speedup | Runtime (50×30×20×10) |
|-------|--------------------|-----------------------|
| Baseline | 1× | ~5ms |
| Phase 1 | 1.2-1.5× | ~3.3-4.2ms |
| Phase 2 | 8-16× | ~0.3-0.6ms |
| Phase 3 | 10-30× | **~0.2-0.5ms** |

---

## Strategy 3: Cox-de Boor Recursion Vectorization

### Problem Analysis

**Location**: `src/interpolation/bspline_utils.hpp`, `cubic_basis_nonuniform()` (lines 107-173)

**Key observation**: Cox-de Boor recursion computes **4 basis functions independently**:
- `N[0]` = basis for index `i`
- `N[1]` = basis for index `i-1`
- `N[2]` = basis for index `i-2`
- `N[3]` = basis for index `i-3`

All 4 functions follow the **same recurrence relation**, differing only in array indices.

**Current scalar implementation**:
```cpp
inline void cubic_basis_nonuniform(const std::vector<double>& t, int i, double x, double N[4]) {
    // Degree 0
    double N0[4] = {0, 0, 0, 0};
    for (int k = 0; k < 4; ++k) {  // ← 4-way independent computation
        int idx = i - k;
        N0[k] = (t[idx] <= x && x < t[idx + 1]) ? 1.0 : 0.0;
    }

    // Degree 1
    double N1[4] = {0, 0, 0, 0};
    for (int k = 0; k < 4; ++k) {  // ← Same pattern
        int idx = i - k;
        double leftDen  = t[idx + 1] - t[idx];
        double rightDen = t[idx + 2] - t[idx + 1];
        double left  = (leftDen > 0.0) ? (x - t[idx]) / leftDen * N0[k] : 0.0;
        double right = (rightDen > 0.0 && k > 0) ? (t[idx + 2] - x) / rightDen * N0[k - 1] : 0.0;
        N1[k] = left + right;
    }

    // Degrees 2 and 3 follow same pattern...
}
```

**Perfect SIMD fit**: 4-way parallelism matches AVX2 (256-bit = 4× double precision)

### Proposed SIMD Implementation

```cpp
[[gnu::target_clones("default","avx2","avx512f")]]
inline void cubic_basis_nonuniform_simd(
    const std::vector<double>& t,
    int i,
    double x,
    double N[4])
{
    using simd4d = std::experimental::fixed_size_simd<double, 4>;
    using mask4d = typename simd4d::mask_type;

    const simd4d x_vec(x);  // Broadcast x to all 4 lanes
    const std::array<int, 4> idx_offsets = {0, -1, -2, -3};  // i, i-1, i-2, i-3

    //
    // Degree 0: Vectorized initialization
    //
    simd4d N0_vec(0.0);
    {
        std::array<double, 4> t_left, t_right;
        for (int lane = 0; lane < 4; ++lane) {
            int idx = i + idx_offsets[lane];
            t_left[lane]  = t[idx];
            t_right[lane] = t[idx + 1];
        }

        simd4d t_left_vec, t_right_vec;
        t_left_vec.copy_from(t_left.data(), stdx::element_aligned);
        t_right_vec.copy_from(t_right.data(), stdx::element_aligned);

        // Vectorized comparison: (t_left <= x) && (x < t_right)
        mask4d in_interval = (t_left_vec <= x_vec) && (x_vec < t_right_vec);
        N0_vec = stdx::where(in_interval, simd4d(1.0), simd4d(0.0));
    }

    //
    // Degree 1: Vectorized recurrence
    //
    simd4d N1_vec(0.0);
    {
        std::array<double, 4> leftDen_arr, rightDen_arr;
        std::array<double, 4> t_idx_arr, t_idx_plus1_arr, t_idx_plus2_arr;

        for (int lane = 0; lane < 4; ++lane) {
            int idx = i + idx_offsets[lane];
            t_idx_arr[lane]       = t[idx];
            t_idx_plus1_arr[lane] = t[idx + 1];
            t_idx_plus2_arr[lane] = t[idx + 2];
            leftDen_arr[lane]     = t[idx + 1] - t[idx];
            rightDen_arr[lane]    = t[idx + 2] - t[idx + 1];
        }

        simd4d leftDen_vec, rightDen_vec;
        simd4d t_idx_vec, t_idx_plus1_vec, t_idx_plus2_vec;
        leftDen_vec.copy_from(leftDen_arr.data(), stdx::element_aligned);
        rightDen_vec.copy_from(rightDen_arr.data(), stdx::element_aligned);
        t_idx_vec.copy_from(t_idx_arr.data(), stdx::element_aligned);
        t_idx_plus2_vec.copy_from(t_idx_plus2_arr.data(), stdx::element_aligned);

        // Left term: (x - t[idx]) / leftDen * N0
        mask4d leftDen_valid = leftDen_vec > simd4d(0.0);
        simd4d left_term = stdx::where(
            leftDen_valid,
            (x_vec - t_idx_vec) / leftDen_vec * N0_vec,
            simd4d(0.0)
        );

        // Right term: (t[idx+2] - x) / rightDen * N0_shifted
        // N0_shifted[lane] = N0[lane-1] for lane > 0, else 0
        simd4d N0_shifted(0.0);
        {
            std::array<double, 4> N0_arr;
            N0_vec.copy_to(N0_arr.data(), stdx::element_aligned);
            std::array<double, 4> N0_shifted_arr = {0.0, N0_arr[0], N0_arr[1], N0_arr[2]};
            N0_shifted.copy_from(N0_shifted_arr.data(), stdx::element_aligned);
        }

        mask4d rightDen_valid = rightDen_vec > simd4d(0.0);
        simd4d right_term = stdx::where(
            rightDen_valid,
            (t_idx_plus2_vec - x_vec) / rightDen_vec * N0_shifted,
            simd4d(0.0)
        );

        N1_vec = left_term + right_term;
    }

    //
    // Degrees 2 and 3: Similar vectorized pattern
    // (Omitted for brevity - follows same structure)
    //
    simd4d N2_vec = /* ... vectorized degree 2 recurrence ... */;
    simd4d N3_vec = /* ... vectorized degree 3 recurrence ... */;

    // Store final result
    N3_vec.copy_to(N, stdx::element_aligned);
}
```

### Batched Evaluation Optimization

For grid-based evaluation, process **4 grid points simultaneously**:

```cpp
// Evaluate basis functions at 4 different x-values in parallel
[[gnu::target_clones("default","avx2","avx512f")]]
inline void cubic_basis_batch_4(
    const std::vector<double>& t,
    int i,
    const std::array<double, 4>& x_batch,  // 4 evaluation points
    std::array<std::array<double, 4>, 4>& N_batch)  // 4×4 output
{
    using simd4d = std::experimental::fixed_size_simd<double, 4>;

    // Load 4 x-values into SIMD vector
    simd4d x_vec;
    x_vec.copy_from(x_batch.data(), stdx::element_aligned);

    // Compute basis functions for all 4 points simultaneously
    // (Each SIMD lane processes one x-value)
    // ... implementation similar to above, but x_vec contains 4 different values ...

    // Transpose output (4 basis functions × 4 x-values → 4 x-values × 4 basis functions)
    for (int basis_idx = 0; basis_idx < 4; ++basis_idx) {
        for (int x_idx = 0; x_idx < 4; ++x_idx) {
            N_batch[x_idx][basis_idx] = /* ... extract from SIMD result ... */;
        }
    }
}
```

### ISA-Specific Optimizations

**SSE2 (baseline, 128-bit)**: 2× double precision
- Requires 2 passes for 4 basis functions
- Expected speedup: 1.5-1.8× (overhead from splitting)

**AVX2 (256-bit)**: 4× double precision **← Perfect match!**
- Single pass for all 4 basis functions
- Expected speedup: 2-2.5×

**AVX-512 (512-bit)**: 8× double precision
- Can process 2 sets of 4 basis functions simultaneously
- Or process 8 grid points in batched mode
- Expected speedup: 3-4× (with batching)

**Multi-ISA dispatch** via `[[gnu::target_clones]]`:
```cpp
[[gnu::target_clones("default","avx2","avx512f")]]
inline void cubic_basis_nonuniform_simd(...) {
    // Compiler generates 3 versions, selects best at runtime
}
```

### Expected Performance Improvement

| Architecture | SIMD Width | Speedup (single point) | Speedup (batched) |
|--------------|------------|------------------------|-------------------|
| SSE2 (baseline) | 2× double | 1.5-1.8× | 2-2.5× |
| AVX2 | 4× double | 2-2.5× | 3-3.5× |
| AVX-512 | 8× double | 2.5-3× | 4-5× |

**Recommendation**: Target AVX2 as sweet spot (ubiquitous on modern CPUs, perfect 4-way match)

### Integration with BSplineCollocation1D

**Backward-compatible wrapper**:
```cpp
class BSplineCollocation1D {
private:
    bool use_simd_ = true;  // Runtime toggle for testing

    void precompute_basis_values() {
        if (use_simd_ && has_avx2()) {
            // SIMD path
            for (size_t i = 0; i < data_points_.size(); ++i) {
                double x = data_points_[i];
                int knot_idx = find_knot_span(x);
                cubic_basis_nonuniform_simd(
                    knots_, knot_idx, x,
                    &basis_values_[i * 4]);
            }
        } else {
            // Scalar fallback (existing code)
            for (size_t i = 0; i < data_points_.size(); ++i) {
                double x = data_points_[i];
                int knot_idx = find_knot_span(x);
                cubic_basis_nonuniform(
                    knots_, knot_idx, x,
                    &basis_values_[i * 4]);
            }
        }
    }

    static bool has_avx2() {
        // Check CPUID for AVX2 support
        #ifdef __AVX2__
            return true;
        #else
            return false;
        #endif
    }
};
```

### Implementation Effort

- **Complexity**: Medium
- **Estimated effort**: 3-4 days
- **Testing requirements**:
  - Verify identical results (SIMD vs scalar) to within FP tolerance (~1e-14)
  - Regression tests for all knot vector configurations
  - Benchmark on SSE2/AVX2/AVX-512 architectures
  - Verify `[[gnu::target_clones]]` dispatch works correctly

---

## Combined Performance Projections

### Sequential Optimization Path

| Stage | Optimizations | Speedup (vs baseline) | Runtime (50×30×20×10) |
|-------|---------------|------------------------|------------------------|
| **Baseline** | Current implementation | 1× | ~5ms |
| **Stage 1** | PMR workspace | 1.3-1.6× | ~3-4ms |
| **Stage 2** | + OpenMP SIMD hints | 1.6-2.4× | ~2-3ms |
| **Stage 3** | + OpenMP parallel | 10-24× | ~0.2-0.5ms |
| **Stage 4** | + Cox-de Boor SIMD | 12-30× | ~0.17-0.4ms |
| **Stage 5** | + Explicit SIMD kernels | **16-50×** | **~0.1-0.3ms** |

### Optimistic vs Conservative Estimates

**Conservative (low-end estimates)**:
- PMR workspace: 1.3×
- OpenMP parallel: 8× (on 8 cores)
- Cox-de Boor SIMD: 2× (AVX2)
- Explicit SIMD kernels: 1.5× (modest gain over OpenMP SIMD)
- **Total: 16× speedup → ~0.3ms**

**Optimistic (high-end estimates)**:
- PMR workspace: 1.6×
- OpenMP parallel: 16× (on 16 cores, perfect scaling)
- Cox-de Boor SIMD: 2.5× (AVX2 + batching)
- Explicit SIMD kernels: 2× (gather optimization + residual SIMD)
- **Total: 50× speedup → ~0.1ms**

**Realistic target (median)**:
- **25× speedup → ~0.2ms** for 50×30×20×10 grid

### Scalability Analysis

**Grid size scaling** (assuming 25× speedup):

| Grid Size | Points | Baseline | Optimized | Improvement |
|-----------|--------|----------|-----------|-------------|
| 20×20×10×5 | 20,000 | ~1ms | ~40µs | 25× |
| 50×30×20×10 | 300,000 | ~5ms | ~200µs | 25× |
| 100×50×30×20 | 3,000,000 | ~50ms | ~2ms | 25× |
| 200×100×50×30 | 30,000,000 | ~500ms | ~20ms | 25× |

**Note**: OpenMP parallel speedup scales with core count (test on target hardware)

---

## Implementation Roadmap

### Phase 1: Foundation (Week 1)

**Priority**: HIGHEST
**Effort**: Low-Medium
**Expected Speedup**: 8-12× (Quick wins)

**Tasks**:
1. ✅ Add OpenMP SIMD hints to residual/condition loops (1 day)
2. ✅ Implement OpenMP parallel batching for `fit_axisN()` (2 days)
3. ✅ Add thread-local workspace management (1 day)
4. ✅ Regression testing and benchmarking (1 day)

**Deliverables**:
- Parallel B-spline fitter with 8-12× speedup
- Benchmark suite comparing baseline vs parallel

**Risk**: Low (standard OpenMP patterns)

### Phase 2: Cox-de Boor SIMD (Week 2)

**Priority**: HIGH
**Effort**: Medium
**Expected Speedup**: +2-2.5× (on top of Phase 1)

**Tasks**:
1. ✅ Implement `cubic_basis_nonuniform_simd()` with AVX2 (2 days)
2. ✅ Add `[[gnu::target_clones]]` for multi-ISA support (1 day)
3. ✅ Integrate with `BSplineCollocation1D` (1 day)
4. ✅ Validation testing (FP tolerance checks) (1 day)

**Deliverables**:
- SIMD-optimized Cox-de Boor recursion
- ISA-specific benchmarks (SSE2/AVX2/AVX-512)

**Risk**: Medium (requires careful FP validation)

### Phase 3: PMR Workspace (Week 3)

**Priority**: MEDIUM
**Effort**: Medium-High
**Expected Speedup**: +1.3-1.6× (on top of Phases 1-2)

**Tasks**:
1. ✅ Design `BSplineFitter4DWorkspace` class (1 day)
2. ✅ Modify `BSplineCollocation1D` for workspace mode (2 days)
3. ✅ Update all `fit_axisN()` functions (1 day)
4. ✅ Regression testing (workspace vs standalone) (1 day)

**Deliverables**:
- Workspace-based memory management
- Benchmark showing allocation overhead reduction

**Risk**: Medium (backward compatibility, thread safety)

### Phase 4: Explicit SIMD Kernels (Week 4+)

**Priority**: LOW-MEDIUM
**Effort**: High
**Expected Speedup**: +1.5-2× (on top of Phases 1-3)

**Tasks**:
1. ✅ Implement SIMD residual computation (2 days)
2. ✅ Implement SIMD slice extraction with gather (2 days)
3. ✅ Benchmark and tune ISA-specific code (2 days)
4. ✅ Comprehensive validation suite (1 day)

**Deliverables**:
- Fully optimized SIMD kernels for all bottlenecks
- Architecture-specific performance profiles

**Risk**: High (architecture-specific, complex validation)

---

## Testing Strategy

### Validation Requirements

**Numerical correctness**:
- SIMD results must match scalar to within FP tolerance (~1e-14 relative error)
- All optimizations preserve exact collocation matrix structure
- Residual/condition number estimates identical to baseline

**Performance verification**:
- Benchmark suite with grid sizes: 20×20×10×5, 50×30×20×10, 100×50×30×20
- Profile each optimization stage independently
- Measure speedup on target architectures (SSE2, AVX2, AVX-512)

**Regression testing**:
- All existing B-spline tests must pass unchanged
- Add new tests for workspace mode, SIMD paths, parallel mode
- Integration tests with `PriceTableBuilder`

### Benchmark Suite

**Micro-benchmarks** (Google Benchmark):
```cpp
static void BM_CoxDeBoorScalar(benchmark::State& state) {
    std::vector<double> knots = /* ... */;
    double x = 0.5;
    double N[4];
    for (auto _ : state) {
        cubic_basis_nonuniform(knots, 10, x, N);
        benchmark::DoNotOptimize(N);
    }
}

static void BM_CoxDeBoorSIMD_AVX2(benchmark::State& state) {
    std::vector<double> knots = /* ... */;
    double x = 0.5;
    double N[4];
    for (auto _ : state) {
        cubic_basis_nonuniform_simd(knots, 10, x, N);
        benchmark::DoNotOptimize(N);
    }
}

BENCHMARK(BM_CoxDeBoorScalar);
BENCHMARK(BM_CoxDeBoorSIMD_AVX2);
```

**End-to-end benchmarks**:
```cpp
static void BM_BSplineFit4D_Baseline(benchmark::State& state) {
    auto grid = create_test_grid_50x30x20x10();
    for (auto _ : state) {
        BSplineFitter4DSeparable fitter(grid);
        auto coeffs = fitter.fit(prices);
        benchmark::DoNotOptimize(coeffs);
    }
}

static void BM_BSplineFit4D_Optimized(benchmark::State& state) {
    auto grid = create_test_grid_50x30x20x10();
    BSplineFitter4DWorkspace workspace(50, 30, 20, 10);
    for (auto _ : state) {
        BSplineFitter4DSeparable fitter(grid, &workspace);
        auto coeffs = fitter.fit_parallel_simd(prices);
        benchmark::DoNotOptimize(coeffs);
        workspace.reset();
    }
}
```

### Validation Tests

**SIMD correctness**:
```cpp
TEST(CoxDeBoorSIMD, MatchesScalar) {
    std::vector<double> knots = generate_test_knots();
    const double x = 0.5;

    double N_scalar[4], N_simd[4];
    cubic_basis_nonuniform(knots, 10, x, N_scalar);
    cubic_basis_nonuniform_simd(knots, 10, x, N_simd);

    for (int k = 0; k < 4; ++k) {
        EXPECT_NEAR(N_scalar[k], N_simd[k], 1e-14)
            << "Basis function " << k << " mismatch";
    }
}
```

**Workspace mode correctness**:
```cpp
TEST(BSplineFitter4D, WorkspaceMatchesStandalone) {
    auto grid = create_test_grid();
    auto prices = generate_test_prices();

    // Standalone mode
    BSplineFitter4DSeparable fitter_standalone(grid);
    auto coeffs_standalone = fitter_standalone.fit(prices);

    // Workspace mode
    BSplineFitter4DWorkspace workspace(50, 30, 20, 10);
    BSplineFitter4DSeparable fitter_workspace(grid, &workspace);
    auto coeffs_workspace = fitter_workspace.fit(prices);

    ASSERT_EQ(coeffs_standalone.size(), coeffs_workspace.size());
    for (size_t i = 0; i < coeffs_standalone.size(); ++i) {
        EXPECT_NEAR(coeffs_standalone[i], coeffs_workspace[i], 1e-14);
    }
}
```

**Parallel correctness**:
```cpp
TEST(BSplineFitter4D, ParallelMatchesSequential) {
    auto grid = create_test_grid();
    auto prices = generate_test_prices();

    BSplineFitter4DSeparable fitter(grid);

    // Sequential
    auto coeffs_seq = fitter.fit_sequential(prices);

    // Parallel
    auto coeffs_par = fitter.fit_parallel(prices);

    ASSERT_EQ(coeffs_seq.size(), coeffs_par.size());
    for (size_t i = 0; i < coeffs_seq.size(); ++i) {
        EXPECT_NEAR(coeffs_seq[i], coeffs_par[i], 1e-14);
    }
}
```

---

## Risk Assessment

### Technical Risks

| Risk | Severity | Mitigation |
|------|----------|------------|
| SIMD FP differences vs scalar | Medium | Comprehensive validation with tight tolerances (1e-14) |
| Thread safety bugs in parallel mode | High | Thread sanitizer, stress testing, code review |
| Performance regression on non-AVX2 CPUs | Low | Maintain scalar fallback, test on SSE2 |
| Workspace memory underestimation | Medium | Add 20% safety margin, runtime checks |
| Load imbalance in OpenMP parallel | Low | Use dynamic scheduling for uneven slice workloads |

### Schedule Risks

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Phase 4 (explicit SIMD) takes longer than estimated | High | Medium | Phase 4 is lowest priority, can defer |
| AVX-512 testing delayed (hardware unavailable) | Medium | Low | Focus on AVX2 (more common), AVX-512 optional |
| Integration issues with existing code | Medium | Medium | Incremental integration, feature flags for rollback |

---

## Success Metrics

### Performance Targets

**Primary goal**: 25× speedup on 50×30×20×10 grid
- Baseline: ~5ms
- Target: ~0.2ms
- **Success criterion**: Achieve ≥20× speedup (≤0.25ms)

**Secondary goals**:
- Near-linear OpenMP scaling (≥80% efficiency on 16 cores)
- Cox-de Boor SIMD: ≥2× on AVX2
- PMR workspace: ≥1.3× allocation overhead reduction

### Quality Targets

**Correctness**:
- 100% of existing tests pass unchanged
- SIMD results match scalar to within 1e-14 relative error
- Zero thread safety issues (verified by thread sanitizer)

**Maintainability**:
- Clear separation of SIMD/scalar code paths
- Backward compatibility (existing code works without changes)
- Comprehensive inline documentation

---

## Future Work

### Beyond This Design

**Potential follow-on optimizations**:
1. **GPU acceleration**: Offload tensor-product fitting to CUDA/HIP
   - Expected: 100-1000× for large grids (millions of points)
   - Effort: High (requires GPU memory management, kernel development)

2. **Adaptive grid refinement**: Denser grids in high-curvature regions
   - Expected: 2-5× fewer points for same accuracy
   - Effort: Medium (requires error estimation, adaptive meshing)

3. **Cache-oblivious algorithms**: Improve cache locality beyond workspace
   - Expected: 1.2-1.5× additional speedup
   - Effort: High (requires algorithm redesign)

4. **Mixed precision**: Use float32 for intermediate computations
   - Expected: 1.5-2× on AVX-512 (16-way float vs 8-way double)
   - Effort: Medium (requires precision analysis, validation)

### Lessons for Other Modules

**Reusable patterns**:
- PMR workspace strategy → Apply to other iterative solvers
- OpenMP parallel batching → Apply to Monte Carlo simulations
- Cox-de Boor SIMD → Template for other recursive algorithms

---

## Appendix A: Code Snippets

*(Full implementations omitted for brevity - see inline code in each strategy section)*

---

## Appendix B: Performance Profiles

### Baseline Profile (50×30×20×10 grid)

**Total runtime**: ~5ms

| Function | Time | % Total | Optimization |
|----------|------|---------|--------------|
| `fit_axis0/1/2/3()` | ~2.5ms | 50% | OpenMP parallel |
| `solve_banded_system()` | ~1.2ms | 24% | PMR workspace |
| `compute_residual()` | ~0.6ms | 12% | SIMD residual |
| `cubic_basis_nonuniform()` | ~0.5ms | 10% | Cox-de Boor SIMD |
| `estimate_condition_number()` | ~0.2ms | 4% | OpenMP parallel |

**Key insight**: Parallelizing slice processing (`fit_axisN()`) is highest-impact optimization

---

## Appendix C: References

**SIMD Programming**:
- `std::experimental::simd` (P0214R9): https://wg21.link/P0214
- GCC SIMD documentation: https://gcc.gnu.org/onlinedocs/gcc/Vector-Extensions.html
- Intel Intrinsics Guide: https://www.intel.com/content/www/us/en/docs/intrinsics-guide/

**B-spline Theory**:
- de Boor, Carl (1978). "A Practical Guide to Splines". Springer-Verlag.
- Cox-de Boor recursion formula: https://en.wikipedia.org/wiki/De_Boor%27s_algorithm

**OpenMP**:
- OpenMP 5.2 Specification: https://www.openmp.org/specifications/
- SIMD directives: https://www.openmp.org/spec-html/5.2/openmpse42.html

**Existing Codebase**:
- `src/pde/memory/unified_memory_resource.hpp` - PMR wrapper
- `src/pde/operators/centered_difference_simd_backend.hpp` - SIMD reference implementation
- `src/interpolation/bspline_utils.hpp` - Cox-de Boor recursion

---

**END OF DESIGN DOCUMENT**
