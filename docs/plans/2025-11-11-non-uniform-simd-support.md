# Non-Uniform Grid Support for CenteredDifferenceSIMD

**Date**: 2025-11-11
**Status**: Design Complete, Ready for Implementation
**Context**: Extends CenteredDifferenceSIMD to support tanh-clustered grids for adaptive mesh refinement

---

## Design Decisions

### 1. Use Case
**Primary use case**: Tanh-clustered grids for adaptive mesh refinement around strikes/barriers in option pricing.

**Not in scope**: Arbitrary external meshes, dynamic grid adaptation during solve.

### 2. Vectorization Strategy
**Approach**: Precomputed weight arrays in SoA (Structure-of-Arrays) layout.

**Rationale**:
- Grids are fixed during solve (tanh-clustered, not dynamic)
- One-time precomputation cost is negligible (~1-2 µs for n=100)
- Clean SIMD loads with no per-lane divisions
- Maintains ISA dispatch benefits (AVX2/AVX-512)
- Extra memory (5×(n-1)×8 bytes ≈ 4KB for n=100) is tiny

**Alternatives rejected**:
- Hybrid (scalar fallback): Loses 4-8x speedup on non-uniform grids
- Per-lane computation: Expensive SIMD divisions (~10-15 cycles vs 4 for mul)
- Masked operations: Complex, no benefit for fully non-uniform grids

### 3. GridSpacing Extension
**Approach**: Eager construction with single contiguous buffer.

**Implementation**:
```cpp
class GridSpacing {
    GridView<T> grid_;
    std::vector<T> precomputed_;  // Single buffer, SoA layout

    explicit GridSpacing(GridView<T> grid) : grid_(grid) {
        if (!is_uniform()) {
            precompute_non_uniform_data();  // Done in constructor
        }
    }
};
```

**Rationale**:
- Simple, no hidden mutable state
- Immutable after construction
- No "did you remember to call precompute?" footguns
- Single allocation for better cache locality

### 4. Memory Layout
**Single contiguous buffer** with 5 arrays:

```
Buffer size: 5 * (n-1) * sizeof(T)

Array layout (all for interior points i ∈ [1, n-1)):
[0        .. n-2     ] : dx_left_inv[i]   = 1 / (x[i] - x[i-1])
[n-1      .. 2n-3    ] : dx_right_inv[i]  = 1 / (x[i+1] - x[i])
[2n-2     .. 3n-4    ] : dx_center_inv[i] = 2 / (dx_left + dx_right)
[3n-3     .. 4n-5    ] : w_left[i]        = dx_right / (dx_left + dx_right)
[4n-4     .. 5n-6    ] : w_right[i]       = dx_left / (dx_left + dx_right)
```

**Memory overhead**: ~4KB for n=100 (5×99×8 bytes)

**Zero-copy accessors**:
```cpp
std::span<const T> dx_left_inv() const {
    assert(!is_uniform() && "dx_left_inv only available for non-uniform grids");
    return {precomputed_.data(), n_ - 1};
}

std::span<const T> dx_right_inv() const {
    assert(!is_uniform());
    return {precomputed_.data() + (n_ - 1), n_ - 1};
}

std::span<const T> dx_center_inv() const {
    assert(!is_uniform());
    return {precomputed_.data() + 2 * (n_ - 1), n_ - 1};
}

std::span<const T> w_left() const {
    assert(!is_uniform());
    return {precomputed_.data() + 3 * (n_ - 1), n_ - 1};
}

std::span<const T> w_right() const {
    assert(!is_uniform());
    return {precomputed_.data() + 4 * (n_ - 1), n_ - 1};
}
```

**Assertions ensure fail-fast** if dispatch is forgotten.

### 5. CenteredDifferenceSIMD API
**Approach**: Separate methods + convenience wrapper (all with target_clones).

```cpp
class CenteredDifferenceSIMD {
    // Performance methods (called by SpatialOperator)
    [[gnu::target_clones("default","avx2","avx512f")]]
    void compute_second_derivative_uniform(...) const;

    [[gnu::target_clones("default","avx2","avx512f")]]
    void compute_second_derivative_non_uniform(...) const;

    [[gnu::target_clones("default","avx2","avx512f")]]
    void compute_first_derivative_uniform(...) const;

    [[gnu::target_clones("default","avx2","avx512f")]]
    void compute_first_derivative_non_uniform(...) const;

    // Convenience wrappers (tests, examples, direct use)
    [[gnu::target_clones("default","avx2","avx512f")]]
    void compute_second_derivative(...) const {
        if (spacing_.is_uniform())
            compute_second_derivative_uniform(...);
        else
            compute_second_derivative_non_uniform(...);
    }

    [[gnu::target_clones("default","avx2","avx512f")]]
    void compute_first_derivative(...) const {
        if (spacing_.is_uniform())
            compute_first_derivative_uniform(...);
        else
            compute_first_derivative_non_uniform(...);
    }
};
```

**Rationale**:
- Prevents `if (is_uniform())` boilerplate at every call site
- Tests/examples get clean API without caring about grid type
- Performance-critical paths (SpatialOperator) can call explicit methods
- All methods get ISA-specific variants via target_clones

---

## Implementation Details

### Non-Uniform Second Derivative (SIMD Kernel)

**Formula**: d²u/dx² = 2 × [(u[i+1] - u[i])/dx_right - (u[i] - u[i-1])/dx_left] / (dx_left + dx_right)

**Key insight**: Precompute `dx_center_inv = 2 / (dx_left + dx_right)` to avoid divisions in hot loop.

```cpp
[[gnu::target_clones("default","avx2","avx512f")]]
void compute_second_derivative_non_uniform(
    std::span<const T> u, std::span<T> d2u_dx2,
    size_t start, size_t end) const
{
    assert(start >= 1 && "start must allow u[i-1] access");
    assert(end <= u.size() - 1 && "end must allow u[i+1] access");
    assert(!spacing_.is_uniform() && "Use compute_second_derivative_uniform for uniform grids");

    // Get precomputed arrays (zero-copy spans)
    auto dx_left_inv = spacing_.dx_left_inv();
    auto dx_right_inv = spacing_.dx_right_inv();
    auto dx_center_inv = spacing_.dx_center_inv();

    // Vectorized main loop
    size_t i = start;
    for (; i + simd_width <= end; i += simd_width) {
        // Load u values
        simd_t u_left, u_center, u_right;
        u_left.copy_from(u.data() + i - 1, stdx::element_aligned);
        u_center.copy_from(u.data() + i, stdx::element_aligned);
        u_right.copy_from(u.data() + i + 1, stdx::element_aligned);

        // Load precomputed spacing inverses
        simd_t dxl_inv, dxr_inv, dxc_inv;
        dxl_inv.copy_from(dx_left_inv.data() + i - 1, stdx::element_aligned);
        dxr_inv.copy_from(dx_right_inv.data() + i - 1, stdx::element_aligned);
        dxc_inv.copy_from(dx_center_inv.data() + i - 1, stdx::element_aligned);

        // Compute second derivative (only multiplications, no divisions!)
        const simd_t forward_diff = (u_right - u_center) * dxr_inv;
        const simd_t backward_diff = (u_center - u_left) * dxl_inv;
        const simd_t result = (forward_diff - backward_diff) * dxc_inv;

        result.copy_to(d2u_dx2.data() + i, stdx::element_aligned);
    }

    // Scalar tail: use precomputed values to match SIMD path exactly
    for (; i < end; ++i) {
        const T dxl_inv = dx_left_inv[i - 1];
        const T dxr_inv = dx_right_inv[i - 1];
        const T dxc_inv = dx_center_inv[i - 1];

        const T forward_diff = (u[i+1] - u[i]) * dxr_inv;
        const T backward_diff = (u[i] - u[i-1]) * dxl_inv;
        d2u_dx2[i] = (forward_diff - backward_diff) * dxc_inv;
    }
}
```

**Critical**: Scalar tail uses precomputed arrays, not recomputed spacing. This ensures:
- SIMD and scalar produce **identical** results
- No divisions in scalar tail either
- Consistency for numerical validation

### Non-Uniform First Derivative (SIMD Kernel)

**Formula**: du/dx = w_left × (u[i] - u[i-1])/dx_left + w_right × (u[i+1] - u[i])/dx_right

```cpp
[[gnu::target_clones("default","avx2","avx512f")]]
void compute_first_derivative_non_uniform(
    std::span<const T> u, std::span<T> du_dx,
    size_t start, size_t end) const
{
    assert(start >= 1 && "start must allow u[i-1] access");
    assert(end <= u.size() - 1 && "end must allow u[i+1] access");
    assert(!spacing_.is_uniform() && "Use compute_first_derivative_uniform for uniform grids");

    // Get precomputed arrays
    auto w_left = spacing_.w_left();
    auto w_right = spacing_.w_right();
    auto dx_left_inv = spacing_.dx_left_inv();
    auto dx_right_inv = spacing_.dx_right_inv();

    // Vectorized main loop
    size_t i = start;
    for (; i + simd_width <= end; i += simd_width) {
        // Load u values
        simd_t u_left, u_center, u_right;
        u_left.copy_from(u.data() + i - 1, stdx::element_aligned);
        u_center.copy_from(u.data() + i, stdx::element_aligned);
        u_right.copy_from(u.data() + i + 1, stdx::element_aligned);

        // Load precomputed weights and inverses
        simd_t wl, wr, dxl_inv, dxr_inv;
        wl.copy_from(w_left.data() + i - 1, stdx::element_aligned);
        wr.copy_from(w_right.data() + i - 1, stdx::element_aligned);
        dxl_inv.copy_from(dx_left_inv.data() + i - 1, stdx::element_aligned);
        dxr_inv.copy_from(dx_right_inv.data() + i - 1, stdx::element_aligned);

        // Compute first derivative
        const simd_t term1 = wl * (u_center - u_left) * dxl_inv;
        const simd_t term2 = wr * (u_right - u_center) * dxr_inv;
        const simd_t result = term1 + term2;

        result.copy_to(du_dx.data() + i, stdx::element_aligned);
    }

    // Scalar tail: use precomputed values
    for (; i < end; ++i) {
        const T wl = w_left[i - 1];
        const T wr = w_right[i - 1];
        const T dxl_inv = dx_left_inv[i - 1];
        const T dxr_inv = dx_right_inv[i - 1];

        const T term1 = wl * (u[i] - u[i-1]) * dxl_inv;
        const T term2 = wr * (u[i+1] - u[i]) * dxr_inv;
        du_dx[i] = term1 + term2;
    }
}
```

### Tiled Variants

Add tiled versions for cache-friendly execution (same pattern as uniform):

```cpp
[[gnu::target_clones("default","avx2","avx512f")]]
void compute_second_derivative_tiled(
    std::span<const T> u, std::span<T> d2u_dx2,
    size_t start, size_t end) const
{
    for (size_t tile_start = start; tile_start < end; tile_start += l1_tile_size_) {
        const size_t tile_end = std::min(tile_start + l1_tile_size_, end);

        if (spacing_.is_uniform()) {
            compute_second_derivative_uniform(u, d2u_dx2, tile_start, tile_end);
        } else {
            compute_second_derivative_non_uniform(u, d2u_dx2, tile_start, tile_end);
        }
    }
}
```

---

## Performance Expectations

### Precomputation Cost
- **Time**: O(n) loop, ~1-2 microseconds for n=100
- **When**: Once during GridSpacing construction
- **Amortized**: Negligible compared to PDE solve (milliseconds)

### SIMD Speedup (Non-Uniform)
- **Expected**: 3-6x over scalar non-uniform
- **Slightly slower than uniform** due to extra loads (5 arrays vs 1 scalar)
- **Still much faster** than scalar non-uniform path
- **No divisions in hot loop** (only multiplications)

### Memory Overhead
- **Size**: 5×(n-1)×8 bytes
- **For n=100**: ~4KB (negligible)
- **For n=1000**: ~40KB (still small)
- **Cache impact**: Minimal - arrays stream sequentially

### ISA Dispatch
- **AVX-512**: 8-wide SIMD (8 doubles per vector)
- **AVX2**: 4-wide SIMD (4 doubles per vector)
- **Default**: 2-wide SIMD (SSE2 baseline)

---

## Testing Strategy

### 1. GridSpacing Precomputation Tests
```cpp
TEST(GridSpacingTest, NonUniformPrecomputationCorrectness) {
    // Tanh-clustered grid
    std::vector<double> x = generate_tanh_grid(101, -3.0, 3.0, 0.0);
    auto grid = GridView<double>(x);
    auto spacing = GridSpacing<double>(grid);

    ASSERT_FALSE(spacing.is_uniform());

    // Verify precomputed values match manual computation
    for (size_t i = 1; i < 100; ++i) {
        const double dx_left = x[i] - x[i-1];
        const double dx_right = x[i+1] - x[i];

        EXPECT_DOUBLE_EQ(spacing.dx_left_inv()[i-1], 1.0 / dx_left);
        EXPECT_DOUBLE_EQ(spacing.dx_right_inv()[i-1], 1.0 / dx_right);
        EXPECT_DOUBLE_EQ(spacing.dx_center_inv()[i-1], 2.0 / (dx_left + dx_right));
        EXPECT_DOUBLE_EQ(spacing.w_left()[i-1], dx_right / (dx_left + dx_right));
        EXPECT_DOUBLE_EQ(spacing.w_right()[i-1], dx_left / (dx_left + dx_right));
    }
}

TEST(GridSpacingTest, UniformGridNoPrecomputation) {
    // Uniform grid
    std::vector<double> x(101);
    for (size_t i = 0; i < 101; ++i) x[i] = i * 0.1;

    auto grid = GridView<double>(x);
    auto spacing = GridSpacing<double>(grid);

    ASSERT_TRUE(spacing.is_uniform());

    // Should not allocate precomputed buffer
    EXPECT_DEATH(spacing.dx_left_inv(), "Assertion.*failed");
}
```

### 2. SIMD Numerical Correctness Tests
```cpp
TEST(CenteredDifferenceSIMDTest, NonUniformSecondDerivative) {
    // Tanh-clustered grid
    std::vector<double> x = generate_tanh_grid(101, -3.0, 3.0, 0.0);
    auto grid = GridView<double>(x);
    auto spacing = GridSpacing<double>(grid);
    auto stencil = CenteredDifferenceSIMD<double>(spacing);

    // Test function: f(x) = x^2, f''(x) = 2
    std::vector<double> u(101);
    for (size_t i = 0; i < 101; ++i) u[i] = x[i] * x[i];

    std::vector<double> d2u_dx2(101, 0.0);
    stencil.compute_second_derivative_non_uniform(u, d2u_dx2, 1, 100);

    // Should be close to 2.0 (with truncation error O(dx^2))
    for (size_t i = 1; i < 100; ++i) {
        EXPECT_NEAR(d2u_dx2[i], 2.0, 0.01) << "at index " << i;
    }
}
```

### 3. SIMD vs Scalar Baseline (Regression)
```cpp
TEST(CenteredDifferenceSIMDTest, NonUniformMatchesScalarBaseline) {
    // Tanh-clustered grid
    std::vector<double> x = generate_tanh_grid(101, -3.0, 3.0, 0.0);
    auto grid = GridView<double>(x);
    auto spacing = GridSpacing<double>(grid);

    // Scalar baseline (old CenteredDifference)
    auto scalar_stencil = CenteredDifference<double>(spacing);

    // SIMD version (new CenteredDifferenceSIMD)
    auto simd_stencil = CenteredDifferenceSIMD<double>(spacing);

    // Test function: f(x) = sin(x)
    std::vector<double> u(101);
    for (size_t i = 0; i < 101; ++i) u[i] = std::sin(x[i]);

    // Compute with scalar
    std::vector<double> d2u_dx2_scalar(101, 0.0);
    scalar_stencil.compute_all_second(u, d2u_dx2_scalar, 1, 100);

    // Compute with SIMD
    std::vector<double> d2u_dx2_simd(101, 0.0);
    simd_stencil.compute_second_derivative_non_uniform(u, d2u_dx2_simd, 1, 100);

    // Should match EXACTLY (no tolerance)
    for (size_t i = 1; i < 100; ++i) {
        EXPECT_DOUBLE_EQ(d2u_dx2_simd[i], d2u_dx2_scalar[i])
            << "Mismatch at index " << i;
    }
}

TEST(CenteredDifferenceSIMDTest, UniformSIMDMatchesScalarBaseline) {
    // Uniform grid
    std::vector<double> x(101);
    for (size_t i = 0; i < 101; ++i) x[i] = i * 0.1;

    auto grid = GridView<double>(x);
    auto spacing = GridSpacing<double>(grid);

    auto scalar_stencil = CenteredDifference<double>(spacing);
    auto simd_stencil = CenteredDifferenceSIMD<double>(spacing);

    std::vector<double> u(101);
    for (size_t i = 0; i < 101; ++i) u[i] = std::sin(x[i]);

    std::vector<double> d2u_dx2_scalar(101, 0.0);
    scalar_stencil.compute_all_second(u, d2u_dx2_scalar, 1, 100);

    std::vector<double> d2u_dx2_simd(101, 0.0);
    simd_stencil.compute_second_derivative_uniform(u, d2u_dx2_simd, 1, 100);

    for (size_t i = 1; i < 100; ++i) {
        EXPECT_DOUBLE_EQ(d2u_dx2_simd[i], d2u_dx2_scalar[i]);
    }
}
```

### 4. Convenience Wrapper Tests
```cpp
TEST(CenteredDifferenceSIMDTest, ConvenienceWrapperDispatchesCorrectly) {
    // Test uniform grid
    {
        std::vector<double> x(101);
        for (size_t i = 0; i < 101; ++i) x[i] = i * 0.1;
        auto grid = GridView<double>(x);
        auto spacing = GridSpacing<double>(grid);
        auto stencil = CenteredDifferenceSIMD<double>(spacing);

        std::vector<double> u(101);
        for (size_t i = 0; i < 101; ++i) u[i] = x[i] * x[i];

        std::vector<double> d2u_explicit(101, 0.0);
        stencil.compute_second_derivative_uniform(u, d2u_explicit, 1, 100);

        std::vector<double> d2u_wrapper(101, 0.0);
        stencil.compute_second_derivative(u, d2u_wrapper, 1, 100);

        // Wrapper should dispatch to uniform method
        for (size_t i = 1; i < 100; ++i) {
            EXPECT_DOUBLE_EQ(d2u_wrapper[i], d2u_explicit[i]);
        }
    }

    // Test non-uniform grid
    {
        std::vector<double> x = generate_tanh_grid(101, -3.0, 3.0, 0.0);
        auto grid = GridView<double>(x);
        auto spacing = GridSpacing<double>(grid);
        auto stencil = CenteredDifferenceSIMD<double>(spacing);

        std::vector<double> u(101);
        for (size_t i = 0; i < 101; ++i) u[i] = x[i] * x[i];

        std::vector<double> d2u_explicit(101, 0.0);
        stencil.compute_second_derivative_non_uniform(u, d2u_explicit, 1, 100);

        std::vector<double> d2u_wrapper(101, 0.0);
        stencil.compute_second_derivative(u, d2u_wrapper, 1, 100);

        // Wrapper should dispatch to non-uniform method
        for (size_t i = 1; i < 100; ++i) {
            EXPECT_DOUBLE_EQ(d2u_wrapper[i], d2u_explicit[i]);
        }
    }
}
```

### 5. Integration Tests
```cpp
TEST(SpatialOperatorTest, NonUniformSIMDIntegration) {
    // End-to-end test with SpatialOperator
    std::vector<double> x = generate_tanh_grid(101, -3.0, 3.0, 0.0);
    auto grid = GridView<double>(x);
    auto spacing = std::make_shared<GridSpacing<double>>(grid);

    // Create PDE with spatial operator
    auto pde = BlackScholesPDE(/*sigma=*/0.2, /*r=*/0.05, /*q=*/0.0);
    SpatialOperator spatial_op(pde, spacing);

    // Test that non-uniform SIMD path is used
    std::vector<double> u(101);
    std::vector<double> Lu(101, 0.0);

    for (size_t i = 0; i < 101; ++i) u[i] = std::exp(x[i]);

    spatial_op.apply(/*t=*/0.0, u, Lu);

    // Verify results are reasonable (analytical validation)
    // ...
}
```

---

## Implementation Tasks

### Task 1: Extend GridSpacing with Precomputed Arrays
**File**: `src/operators/grid_spacing.hpp`

**Changes**:
1. Add `std::vector<T> precomputed_` member
2. Implement `precompute_non_uniform_data()` in constructor
3. Add zero-copy span accessors with assertions
4. Update existing tests

**TDD Steps**:
1. Write test for precomputation correctness (tanh grid)
2. Write test for uniform grid (no precomputation)
3. Write test for accessor assertions
4. Implement precomputation logic
5. Verify all tests pass

### Task 2: Add Non-Uniform Methods to CenteredDifferenceSIMD
**File**: `src/operators/centered_difference_simd.hpp`

**Changes**:
1. Add `compute_second_derivative_non_uniform()`
2. Add `compute_first_derivative_non_uniform()`
3. Add `compute_second_derivative_tiled()` (dispatching version)
4. Update documentation

**TDD Steps**:
1. Write numerical correctness test (f(x) = x^2)
2. Write SIMD vs scalar baseline test
3. Implement SIMD kernels
4. Verify tests pass
5. Run on different CPUs (AVX2, AVX-512)

### Task 3: Add Convenience Wrappers
**File**: `src/operators/centered_difference_simd.hpp`

**Changes**:
1. Add `compute_second_derivative()` wrapper
2. Add `compute_first_derivative()` wrapper
3. All wrappers get `[[gnu::target_clones]]`

**TDD Steps**:
1. Write wrapper dispatch test (uniform grid)
2. Write wrapper dispatch test (non-uniform grid)
3. Implement wrappers
4. Verify tests pass

### Task 4: Update SpatialOperator (Optional)
**File**: `src/operators/spatial_operator.hpp`

**Changes**:
- SpatialOperator already dispatches based on `spacing_->is_uniform()`
- No changes needed if using explicit methods
- If using wrappers, can simplify by removing dispatch

### Task 5: Add Comprehensive Tests
**Files**: `tests/operators/centered_difference_simd_test.cc`, `tests/operators/grid_spacing_test.cc`

**Coverage**:
1. GridSpacing precomputation correctness
2. SIMD numerical accuracy (non-uniform)
3. SIMD vs scalar baseline (regression)
4. Convenience wrapper dispatch
5. Integration with SpatialOperator

---

## Migration Path

### For Existing Code
**No breaking changes** - uniform grid path is unchanged.

### For New Code Using Non-Uniform Grids
```cpp
// Before (scalar only)
auto spacing = GridSpacing<double>(grid);
auto stencil = CenteredDifference<double>(spacing);

// After (SIMD support)
auto spacing = GridSpacing<double>(grid);  // Precomputes if non-uniform
auto stencil = CenteredDifferenceSIMD<double>(spacing);

// Use convenience wrapper (automatic dispatch)
stencil.compute_second_derivative(u, d2u_dx2, 1, n-1);

// Or explicit method (performance-critical path)
if (spacing.is_uniform()) {
    stencil.compute_second_derivative_uniform(u, d2u_dx2, 1, n-1);
} else {
    stencil.compute_second_derivative_non_uniform(u, d2u_dx2, 1, n-1);
}
```

---

## Future Work

### Potential Optimizations
1. **Aligned allocator** for precomputed buffer (64-byte alignment)
2. **Prefetching** for better cache utilization
3. **FMA intrinsics** for term1 + term2 in first derivative

### Not Planned
- Dynamic grid adaptation (grids are fixed during solve)
- Support for arbitrary external meshes (tanh-clustered grids cover use case)
- Multithreading within operator (handled at PDE solver level)

---

## References

- **Old scalar implementation**: `src/operators/centered_difference.hpp`
- **Uniform SIMD implementation**: `src/operators/centered_difference_simd.hpp`
- **GridSpacing**: `src/operators/grid_spacing.hpp`
- **SpatialOperator**: `src/operators/spatial_operator.hpp`
- **Memory management refactor**: `docs/plans/2025-11-10-unified-memory-management-c++23-refactor.md`
