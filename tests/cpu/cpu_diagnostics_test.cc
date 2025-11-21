#include "src/support/cpu/cpu_diagnostics.hpp"
#include <gtest/gtest.h>
#include <iostream>

TEST(CPUDiagnosticsTest, DetectFeatures) {
    auto features = mango::cpu::detect_cpu_features();

    // x86-64 baseline guarantees SSE2
    EXPECT_TRUE(features.has_sse2);

    // Print detected features for diagnostic
    std::cout << "CPU Features:\n";
    std::cout << "  SSE2: " << features.has_sse2 << "\n";
    std::cout << "  AVX2: " << features.has_avx2 << "\n";
    std::cout << "  AVX512F: " << features.has_avx512f << "\n";
    std::cout << "  FMA: " << features.has_fma << "\n";
}

TEST(CPUDiagnosticsTest, DescribeFeatures) {
    std::string description = mango::cpu::describe_cpu_features();

    std::cout << "CPU: " << description << "\n";

    // Should return a known description
    EXPECT_TRUE(description == "SSE2 (2-wide SIMD)" ||
                description == "AVX2+FMA (4-wide SIMD)" ||
                description == "AVX512F+FMA (8-wide SIMD)");
}

TEST(CPUDiagnosticsTest, OSAVXSupport) {
    auto features = mango::cpu::detect_cpu_features();

    if (features.has_avx2 || features.has_avx512f) {
        // If CPU reports AVX, OS support must be enabled
        EXPECT_TRUE(mango::cpu::check_os_avx_support());
    }

    if (features.has_avx512f) {
        // If CPU reports AVX-512, OS support must be enabled
        EXPECT_TRUE(mango::cpu::check_os_avx512_support());
    }
}
