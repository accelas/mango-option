# OpenMP SIMD as Alternative to std::experimental::simd

## Current Situation

We have **two backends** for vectorization:

### 1. ScalarBackend (uses OpenMP SIMD)
```cpp
MANGO_PRAGMA_SIMD  // Expands to: #pragma omp simd
for (size_t i = start; i < end; ++i) {
    const T forward_diff = (u[i+1] - u[i]) * dxr_inv;
    const T backward_diff = (u[i] - u[i-1]) * dxl_inv;
    d2u_dx2[i] = (forward_diff - backward_diff) * dxc_inv;
}
```
**Compiler:** Auto-vectorizes to SIMD instructions
**Works with:** ✅ GCC, ✅ Clang, ✅ All standard libraries

### 2. SimdBackend (uses std::experimental::simd)
```cpp
using simd_t = stdx::native_simd<T>;
simd_t u_left, u_center, u_right;
u_left.copy_from(u.data() + i - 1, stdx::element_aligned);
u_center.copy_from(u.data() + i, stdx::element_aligned);
u_right.copy_from(u.data() + i + 1, stdx::element_aligned);
const simd_t result = stdx::fma(sum, dx2_inv_vec, minus_two * u_center * dx2_inv_vec);
result.copy_to(d2u_dx2.data() + i, stdx::element_aligned);
```
**Compiler:** Explicit SIMD with manual lane control
**Works with:** ✅ GCC + libstdc++, ❌ Clang + libstdc++ (linking issues), ❌ Clang + libc++ (incomplete)

## The Proposal: Unify on OpenMP SIMD

### Option A: Keep Both (Current)
✅ Pros:
- Explicit SIMD for when we need it
- Can use advanced features (gather, scatter, masks)

❌ Cons:
- Doesn't work with Clang
- More complexity
- Maintenance burden

### Option B: Simplify to OpenMP SIMD Only
✅ Pros:
- **Works everywhere**: GCC, Clang, libstdc++, libc++
- Simpler codebase (one backend instead of two)
- Compiler chooses optimal SIMD width
- Can switch to Clang for 15-49% speedup
- Can use std::mdspan with Clang + libc++

❌ Cons:
- Less control over vectorization
- Can't use advanced SIMD features (if needed later)

## Performance Question

**Does explicit SIMD (SimdBackend) provide measurable benefit over OpenMP SIMD (ScalarBackend)?**

We should benchmark:
1. ScalarBackend (OpenMP SIMD) with GCC
2. SimdBackend (std::experimental::simd) with GCC
3. ScalarBackend (OpenMP SIMD) with Clang

If the performance difference is negligible, **Option B is clearly better**.

## Implementation Plan (if we choose Option B)

1. **Benchmark** current backends to quantify difference
2. **Remove** SimdBackend (centered_difference_simd_backend.hpp)
3. **Simplify** CenteredDifference facade to use ScalarBackend only
4. **Update** .bazelrc to use Clang + libc++
5. **Fix** CubicSplineND hot-path with std::mdspan (now available!)
6. **Benchmark** final results

Expected outcome:
- Simpler codebase
- **15-49% faster** with Clang
- **Zero-allocation** N-D splines with mdspan
- **Portable** across all compilers

## Recommendation

**Benchmark first**, then decide:
- If SimdBackend is <5% faster → Use OpenMP SIMD only ✅
- If SimdBackend is >10% faster → Keep both, stay with GCC ❌

My hypothesis: OpenMP SIMD is already quite good (compilers have gotten very good at auto-vectorization), so the explicit SIMD backend likely provides minimal benefit for our stencil operations.

## Related Files

- `src/support/parallel.hpp` - MANGO_PRAGMA_SIMD definition
- `src/pde/operators/centered_difference_scalar.hpp` - OpenMP SIMD backend
- `src/pde/operators/centered_difference_simd_backend.hpp` - std::experimental::simd backend
- `src/pde/operators/centered_difference_facade.hpp` - Facade that chooses backend
