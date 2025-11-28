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
    #define MANGO_PRAGMA_SIMD                           // SYCL will use different syntax
    #define MANGO_PRAGMA_PARALLEL_FOR                   // SYCL will use parallel_for with nd_range
    #define MANGO_PRAGMA_PARALLEL                       // SYCL parallel region
    #define MANGO_PRAGMA_FOR                            // SYCL single loop inside parallel region
    #define MANGO_PRAGMA_FOR_STATIC                     // SYCL static scheduling
    #define MANGO_PRAGMA_FOR_COLLAPSE2                  // SYCL nested parallel loops
    #define MANGO_PRAGMA_FOR_COLLAPSE2_DYNAMIC          // SYCL nested parallel loops with scheduling
    #define MANGO_PRAGMA_ATOMIC                         // SYCL atomic operations
    #define MANGO_PRAGMA_CRITICAL                       // SYCL critical section
    #warning "SYCL backend not yet implemented"
#elif defined(_OPENMP)
    // OpenMP parallel constructs (current implementation)
    #define MANGO_PRAGMA_SIMD                           _Pragma("omp simd")
    #define MANGO_PRAGMA_PARALLEL_FOR                   _Pragma("omp parallel for")
    #define MANGO_PRAGMA_PARALLEL                       _Pragma("omp parallel")
    #define MANGO_PRAGMA_FOR                            _Pragma("omp for")
    #define MANGO_PRAGMA_FOR_STATIC                     _Pragma("omp for schedule(static)")
    #define MANGO_PRAGMA_FOR_COLLAPSE2                  _Pragma("omp for collapse(2)")
    #define MANGO_PRAGMA_FOR_COLLAPSE2_DYNAMIC          _Pragma("omp for collapse(2) schedule(dynamic, 1)")
    #define MANGO_PRAGMA_ATOMIC                         _Pragma("omp atomic")
    #define MANGO_PRAGMA_CRITICAL                       _Pragma("omp critical")
#else
    // Sequential execution (no parallelization)
    #define MANGO_PRAGMA_SIMD
    #define MANGO_PRAGMA_PARALLEL_FOR
    #define MANGO_PRAGMA_PARALLEL
    #define MANGO_PRAGMA_FOR
    #define MANGO_PRAGMA_FOR_STATIC
    #define MANGO_PRAGMA_FOR_COLLAPSE2
    #define MANGO_PRAGMA_FOR_COLLAPSE2_DYNAMIC
    #define MANGO_PRAGMA_ATOMIC
    #define MANGO_PRAGMA_CRITICAL
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
 *
 * Available macros:
 * - MANGO_PRAGMA_SIMD: Vectorize single loop
 * - MANGO_PRAGMA_PARALLEL_FOR: Parallelize single loop
 * - MANGO_PRAGMA_PARALLEL: Start parallel region (use with FOR or FOR_COLLAPSE2)
 * - MANGO_PRAGMA_FOR: Single loop inside parallel region
 * - MANGO_PRAGMA_FOR_STATIC: Single loop with static scheduling (avoid false sharing)
 * - MANGO_PRAGMA_FOR_COLLAPSE2: Collapse 2 nested loops (inside parallel region)
 * - MANGO_PRAGMA_FOR_COLLAPSE2_DYNAMIC: Collapse 2 loops with dynamic scheduling
 * - MANGO_PRAGMA_ATOMIC: Atomic operation (increment, etc.)
 * - MANGO_PRAGMA_CRITICAL: Critical section (only one thread executes at a time)
 */
