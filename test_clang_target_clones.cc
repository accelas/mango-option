// Test if Clang supports target_clones with OpenMP SIMD
#include <cstdio>
#include <cmath>
#include <vector>

// Test 1: target_clones with OpenMP SIMD
[[gnu::target_clones("default","avx2","avx512f")]]
void test_openmp_simd(const double* u, double* result, size_t n) {
    #pragma omp simd
    for (size_t i = 1; i < n - 1; ++i) {
        result[i] = std::fma(u[i+1] + u[i-1], 0.5, -u[i]);
    }
}

// Test 2: target_clones without OpenMP SIMD (baseline)
[[gnu::target_clones("default","avx2","avx512f")]]
void test_no_simd(const double* u, double* result, size_t n) {
    for (size_t i = 1; i < n - 1; ++i) {
        result[i] = std::fma(u[i+1] + u[i-1], 0.5, -u[i]);
    }
}

int main() {
    std::vector<double> u(100, 1.0);
    std::vector<double> result(100, 0.0);

    test_openmp_simd(u.data(), result.data(), u.size());
    std::printf("OpenMP SIMD result: %.6f\n", result[50]);

    test_no_simd(u.data(), result.data(), u.size());
    std::printf("No SIMD result: %.6f\n", result[50]);

    return 0;
}
