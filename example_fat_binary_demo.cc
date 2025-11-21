/**
 * Demonstration: Combining target_clones + OpenMP SIMD + AVX-512
 *
 * This shows how ONE function definition generates THREE optimized versions:
 * 1. SSE2 baseline (2-wide SIMD)
 * 2. AVX2 (4-wide SIMD)
 * 3. AVX-512 (8-wide SIMD)
 *
 * Runtime CPU detection automatically selects the best version.
 */

#include <vector>
#include <cstdio>
#include <cmath>
#include <chrono>

// ============================================================================
// The Magic: Three attributes working together
// ============================================================================

/**
 * target_clones: Tells compiler to generate multiple ISA versions
 * - "default": SSE2 baseline (works on any x86-64 CPU)
 * - "avx2": AVX2 version (Haswell+, 2013+)
 * - "avx512f": AVX-512 version (Skylake-X+, 2017+)
 *
 * #pragma omp simd: Tells compiler to vectorize this loop
 * - Compiler generates SIMD instructions automatically
 * - Width depends on ISA: 2-wide (SSE), 4-wide (AVX2), 8-wide (AVX-512)
 *
 * These work TOGETHER:
 * - target_clones creates 3 function versions
 * - OpenMP SIMD vectorizes loop in each version
 * - Each version uses its ISA's optimal SIMD width
 */
[[gnu::target_clones("default", "avx2", "avx512f")]]
void compute_stencil(const double* u, double* result, size_t n, double dx2_inv) {
    // OpenMP SIMD directive: compiler auto-vectorizes
    #pragma omp simd
    for (size_t i = 1; i < n - 1; ++i) {
        // Second derivative stencil: d²u/dx² = (u[i+1] - 2*u[i] + u[i-1]) / dx²
        result[i] = std::fma(u[i+1] + u[i-1], dx2_inv, -2.0 * u[i] * dx2_inv);
    }
}

// ============================================================================
// CPU Feature Detection (for demonstration)
// ============================================================================

#include <cpuid.h>

const char* detect_cpu_isa() {
    unsigned int eax, ebx, ecx, edx;

    // Check CPUID leaf 7 for AVX2 and AVX-512
    __cpuid_count(7, 0, eax, ebx, ecx, edx);

    if (ebx & bit_AVX512F) {
        return "AVX-512F detected → will use .avx512f version (8-wide SIMD)";
    } else if (ebx & bit_AVX2) {
        return "AVX2 detected → will use .avx2 version (4-wide SIMD)";
    } else {
        return "SSE2 baseline → will use .default version (2-wide SIMD)";
    }
}

// ============================================================================
// Demo Program
// ============================================================================

int main() {
    printf("=== Fat Binary Demo: target_clones + OpenMP SIMD ===\n\n");

    // Show CPU capabilities
    printf("CPU Detection:\n");
    printf("  %s\n\n", detect_cpu_isa());

    // Setup test data
    const size_t n = 10000;
    std::vector<double> u(n);
    std::vector<double> result(n);

    // Initialize with smooth function
    const double dx = 1.0 / (n - 1);
    const double dx2_inv = 1.0 / (dx * dx);

    for (size_t i = 0; i < n; ++i) {
        u[i] = std::sin(2.0 * M_PI * i * dx);
    }

    // Benchmark: First call includes resolver overhead
    auto start = std::chrono::high_resolution_clock::now();
    compute_stencil(u.data(), result.data(), n, dx2_inv);
    auto end = std::chrono::high_resolution_clock::now();

    auto duration_first = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    printf("First call (includes resolver): %ld ns\n", duration_first);

    // Benchmark: Subsequent calls are direct (zero overhead)
    start = std::chrono::high_resolution_clock::now();
    for (int iter = 0; iter < 1000; ++iter) {
        compute_stencil(u.data(), result.data(), n, dx2_inv);
    }
    end = std::chrono::high_resolution_clock::now();

    auto duration_avg = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count() / 1000;
    printf("Average call (direct): %ld ns\n\n", duration_avg);

    printf("How it works:\n");
    printf("1. Compiler generates 3 versions of compute_stencil():\n");
    printf("   - compute_stencil.default   (SSE2, 2×64-bit = 128-bit)\n");
    printf("   - compute_stencil.avx2      (AVX2, 4×64-bit = 256-bit)\n");
    printf("   - compute_stencil.avx512f   (AVX-512, 8×64-bit = 512-bit)\n\n");

    printf("2. Resolver function runs once at first call:\n");
    printf("   - Calls CPUID to detect CPU features\n");
    printf("   - Selects best version for this CPU\n");
    printf("   - Updates function pointer (GNU IFUNC)\n\n");

    printf("3. All subsequent calls go directly to selected version:\n");
    printf("   - Zero runtime overhead\n");
    printf("   - Optimal SIMD width for this CPU\n\n");

    printf("Verify multi-ISA code generation:\n");
    printf("  nm -C %s | grep compute_stencil\n", "example_fat_binary_demo");
    printf("  objdump -d %s | grep -A 20 'compute_stencil.*avx512'\n", "example_fat_binary_demo");

    return 0;
}
