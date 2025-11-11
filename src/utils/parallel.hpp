#pragma once

/**
 * @file parallel.hpp
 * @brief Parallelization macros for portability across OpenMP, SYCL, and sequential execution
 *
 * This header provides abstraction macros for parallel constructs to enable
 * easy porting between different parallel programming models:
 * - OpenMP (current default)
 * - SYCL (future GPU backend)
 * - Sequential (no parallelization)
 *
 * Usage:
 *   MANGO_PRAGMA_SIMD
 *   for (size_t i = 0; i < n; ++i) { ... }
 *
 *   MANGO_PRAGMA_PARALLEL_FOR
 *   for (size_t i = 0; i < n; ++i) { ... }
 */

// Future extension point: When porting to SYCL, define MANGO_USE_SYCL
// and implement SYCL-specific parallel constructs here
#if defined(MANGO_USE_SYCL)
    // SYCL parallel constructs (future implementation)
    #define MANGO_PRAGMA_SIMD              // SYCL will use different syntax
    #define MANGO_PRAGMA_PARALLEL_FOR      // SYCL will use parallel_for with nd_range
    #warning "SYCL backend not yet implemented"
#elif defined(_OPENMP)
    // OpenMP parallel constructs (current implementation)
    #define MANGO_PRAGMA_SIMD              _Pragma("omp simd")
    #define MANGO_PRAGMA_PARALLEL_FOR      _Pragma("omp parallel for")
#else
    // Sequential execution (no parallelization)
    #define MANGO_PRAGMA_SIMD
    #define MANGO_PRAGMA_PARALLEL_FOR
#endif

/**
 * Design notes:
 *
 * 1. OpenMP: Uses `#pragma omp simd` for vectorization and `#pragma omp parallel for`
 *    for multi-threading. The compiler auto-vectorizes SIMD loops when available.
 *
 * 2. SYCL: Will use `parallel_for` with nd_range for explicit kernel execution on GPUs.
 *    SIMD vectorization is implicit in SYCL (work-items map to SIMD lanes).
 *
 * 3. Sequential: No-op for debugging or environments without parallelization support.
 *
 * 4. Why _Pragma instead of #pragma: The _Pragma operator allows using pragmas in macro
 *    definitions, which is required for our abstraction layer.
 */
