# Compiler and Standard Library Trade-offs

## Investigation Summary

We investigated switching from GCC to Clang to compare performance and access newer C++23 features like `std::mdspan`.

## Key Findings

### Standard Library Feature Support

| Feature | GCC 14.2 + libstdc++ | Clang 19 + libstdc++ | Clang 19 + libc++ |
|---------|---------------------|---------------------|-------------------|
| `std::mdspan` (C++23) | ❌ Not available | ❌ Not available | ✅ **Available** |
| `std::experimental::simd` | ✅ **Working** | ⚠️ Linking issues | ❌ Stub only (no operators) |
| Performance | Baseline | **+15-49% faster** | **+15-49% faster** |

### Performance Results (Scalar backend only)

| Benchmark | GCC 14.2.0 | Clang 19 | Speedup |
|-----------|------------|----------|---------|
| American single (101x498) | 1.28 ms | 1.11 ms | **1.15x** |
| American sequential (64) | 81.17 ms | 69.77 ms | **1.16x** |
| American parallel batch (64) | 5.90 ms | 3.96 ms | **1.49x** 🔥 |
| American IV (FDM, 101x1k) | 17.99 ms | 15.70 ms | **1.15x** |

## The Three Options

### Option 1: **GCC + libstdc++** (Current)
✅ **Pros:**
- std::experimental::simd works (SIMD vectorization)
- Proven, stable setup
- No migration effort

❌ **Cons:**
- 15-49% slower than Clang
- No std::mdspan (would have to wait for GCC 15/16)
- Hot-path allocations in CubicSplineND remain unfixed

### Option 2: **Clang + libc++**
✅ **Pros:**
- **15-49% faster** than GCC
- std::mdspan available NOW
- Can fix CubicSplineND hot-path allocations properly
- Modern LLVM ecosystem

❌ **Cons:**
- Lose SIMD backend (libc++ simd is incomplete)
- Would need to stay on Scalar backend (still 15% faster than GCC though!)
- Migration effort required

### Option 3: **Clang + libstdc++** (Attempted, failed)
✅ **Pros:**
- 15-49% faster than GCC
- Keep SIMD backend

❌ **Cons:**
- **Linking failures** with std::experimental::simd + target_clones
- No std::mdspan
- Mixing Clang with GCC's stdlib is non-standard

## Recommendation **[UPDATED AFTER SIMD BENCHMARK]**

### New Finding: SimdBackend Provides No Benefit

Benchmark results show **ScalarBackend (OpenMP SIMD) is faster than SimdBackend in 9 out of 12 test cases**, with margins of 15-45% in many scenarios. The explicit SIMD backend provides no measurable performance advantage.

**This changes the decision matrix:**

### Updated Plan

**Phase 1 (Now):** Remove SimdBackend
- Simplify to OpenMP SIMD only (ScalarBackend)
- OpenMP SIMD works with both GCC and Clang
- Eliminates the SIMD portability constraint

**Phase 2 (1-2 weeks):** Validate simplification
- Run full test suite with ScalarBackend only
- Verify no performance regressions
- Test on multiple compilers

**Phase 3 (1 month):** Switch to Clang + libc++
- Get 15-49% performance boost immediately
- Access to std::mdspan for fixing CubicSplineND hot-path
- Modern toolchain with better diagnostics

**Phase 4 (2 months):** Refactor with mdspan
- Eliminate hot-path allocations in CubicSplineND
- Use std::mdspan for zero-allocation tensor indexing
- Further performance improvements

### Why This Works

By removing the SimdBackend dependency:
- ✅ OpenMP SIMD works everywhere (GCC, Clang, both stdlibs)
- ✅ Simpler codebase (one vectorization strategy)
- ✅ Enables immediate Clang migration
- ✅ Unlocks mdspan usage
- ✅ 15-49% faster overall

**Previous constraint:** "Stay with GCC because SIMD doesn't work with Clang"
**New reality:** "SIMD backend is slower anyway, use OpenMP SIMD with Clang"

## Technical Details

### Why Clang + libstdc++ SIMD Failed

The linking errors occur because:
1. Clang compiles std::experimental::simd code differently than GCC
2. `[[gnu::target_clones]]` creates multiple function versions
3. Clang's name mangling for these symbols doesn't match what libstdc++ expects
4. Results in undefined references at link time

### Why libc++ SIMD Is Incomplete

Checking `/usr/lib/llvm-19/include/c++/v1/experimental/__simd/`:
- Basic class structure exists
- Constructors and accessors work
- **Arithmetic operators missing** (operator+, operator*, etc.)
- Header comment says "TODO: implement simd class"

This is a work-in-progress implementation.

### Why mdspan Matters

`std::mdspan` would solve the hot-path allocation issue in `CubicSplineND::compute_flat_index()` (line 226-269) identified by code review:
- Currently allocates vectors on every element fetch
- mdspan provides zero-allocation multi-dimensional array views
- Precomputed strides eliminate repeated calculations
- Would fix both performance and correctness concerns

## Conclusion

The 15-49% performance improvement from Clang is significant, but losing SIMD vectorization would offset those gains in compute-intensive code. The manual optimization of stride calculation (without mdspan) is the pragmatic short-term solution while we wait for either:
- libc++ to complete SIMD support, OR
- GCC to ship mdspan

We should re-evaluate this decision in 6-12 months.
