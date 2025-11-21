#pragma once

#include <cpuid.h>
#include <string>
#include <immintrin.h>

namespace mango::cpu {

/// CPU feature flags detected at runtime (diagnostic only)
struct CPUFeatures {
    bool has_sse2 = false;
    bool has_avx2 = false;
    bool has_avx512f = false;
    bool has_fma = false;
};

/**
 * Check if OS has enabled xsave for AVX/AVX-512 state
 *
 * CRITICAL: AVX/AVX-512 require OS support for YMM/ZMM register state.
 * Without OSXSAVE check, executing AVX instructions will SIGILL even if
 * CPUID reports support.
 */
__attribute__((target("xsave")))
inline bool check_os_avx_support() {
    unsigned int eax, ebx, ecx, edx;

    // Check OSXSAVE bit (OS has enabled XSAVE)
    if (!__get_cpuid(1, &eax, &ebx, &ecx, &edx)) {
        return false;
    }

    if ((ecx & bit_OSXSAVE) == 0) {
        return false;
    }

    // Check XCR0 register via XGETBV
    unsigned long long xcr0 = _xgetbv(0);

    // AVX requires SSE (bit 1) and YMM (bit 2)
    constexpr unsigned long long AVX_MASK = (1ULL << 1) | (1ULL << 2);

    return (xcr0 & AVX_MASK) == AVX_MASK;
}

/// Check if OS has enabled AVX-512 state
__attribute__((target("xsave")))
inline bool check_os_avx512_support() {
    if (!check_os_avx_support()) {
        return false;
    }

    unsigned long long xcr0 = _xgetbv(0);
    // AVX-512 requires SSE, YMM, and ZMM state (bits 5, 6, 7)
    constexpr unsigned long long AVX512_MASK = (1ULL << 1) | (1ULL << 2) |
                                               (1ULL << 5) | (1ULL << 6) | (1ULL << 7);
    return (xcr0 & AVX512_MASK) == AVX512_MASK;
}

/**
 * Detect CPU features for diagnostic purposes
 *
 * NOTE: Do NOT use this for dispatch. Use [[gnu::target_clones]]
 * which provides zero-overhead IFUNC resolution.
 */
inline CPUFeatures detect_cpu_features() {
    CPUFeatures features;

    unsigned int eax, ebx, ecx, edx;

    // Check for SSE2 (standard in x86-64)
    if (__get_cpuid(1, &eax, &ebx, &ecx, &edx)) {
        features.has_sse2 = (edx & bit_SSE2) != 0;
        features.has_fma = (ecx & bit_FMA) != 0;
    }

    // Check OS support for AVX/AVX-512 state
    bool os_avx_support = check_os_avx_support();
    bool os_avx512_support = os_avx_support && check_os_avx512_support();

    // Check for AVX2 (requires OS support)
    if (os_avx_support && __get_cpuid_count(7, 0, &eax, &ebx, &ecx, &edx)) {
        features.has_avx2 = (ebx & bit_AVX2) != 0;
    }

    // Check for AVX-512 (requires OS support)
    if (os_avx512_support && __get_cpuid_count(7, 0, &eax, &ebx, &ecx, &edx)) {
        features.has_avx512f = (ebx & bit_AVX512F) != 0;
    }

    return features;
}

/**
 * Get human-readable description of CPU features (for logging)
 */
inline std::string describe_cpu_features() {
    static const CPUFeatures features = detect_cpu_features();

    if (features.has_avx512f) {
        return "AVX512F+FMA (8-wide SIMD)";
    } else if (features.has_avx2 && features.has_fma) {
        return "AVX2+FMA (4-wide SIMD)";
    } else if (features.has_sse2) {
        return "SSE2 (2-wide SIMD)";
    } else {
        return "UNKNOWN";
    }
}

} // namespace mango::cpu
