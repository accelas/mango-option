#pragma once

#include <cpuid.h>
#include <string>
#include <iostream>
#include <immintrin.h>

namespace mango::cpu {

/// CPU feature flags detected at runtime
struct CPUFeatures {
    bool has_sse2 = false;
    bool has_avx2 = false;
    bool has_avx512f = false;
    bool has_fma = false;
};

/**
 * ISA target enum for dispatch
 *
 * NOTE: This is DIAGNOSTIC ONLY. Actual dispatch happens via
 * [[gnu::target_clones]] IFUNC resolution at link time.
 */
enum class ISATarget {
    DEFAULT,   // SSE2 baseline
    AVX2,      // Haswell+ (2013+)
    AVX512F    // Skylake-X+ (2017+)
};

/**
 * Check if OS has enabled xsave for AVX/AVX-512 state
 *
 * AVX/AVX-512 require OS support for YMM/ZMM register state.
 * Without this check, CPUID may report AVX support but executing
 * AVX instructions will SIGILL.
 *
 * SAFETY: __attribute__((target("xsave"))) ensures compiler generates
 * code compatible with xsave-enabled CPUs. The OSXSAVE check before
 * _xgetbv() ensures the CPU actually supports the instruction.
 */
__attribute__((target("xsave")))
inline bool check_os_avx_support() {
    unsigned int eax, ebx, ecx, edx;

    // Check OSXSAVE bit (OS has enabled XSAVE)
    if (!__get_cpuid(1, &eax, &ebx, &ecx, &edx)) {
        return false;
    }

    // CRITICAL: Must check OSXSAVE before calling _xgetbv()
    // Without OSXSAVE, _xgetbv() will cause SIGILL
    if ((ecx & bit_OSXSAVE) == 0) {
        return false;  // OS hasn't enabled xsave
    }

    // SAFE: OSXSAVE confirmed, _xgetbv() is now safe to call
    // Check XCR0 register via XGETBV
    // XCR0[1] = SSE state, XCR0[2] = YMM state
    unsigned long long xcr0 = _xgetbv(0);

    // AVX requires SSE (bit 1) and YMM (bit 2)
    constexpr unsigned long long AVX_MASK = (1ULL << 1) | (1ULL << 2);

    return (xcr0 & AVX_MASK) == AVX_MASK;
}

/// Check if OS has enabled AVX-512 state
__attribute__((target("xsave")))
inline bool check_os_avx512_support() {
    unsigned long long xcr0 = _xgetbv(0);
    // AVX-512 requires SSE, YMM, and ZMM state (bits 5, 6, 7)
    constexpr unsigned long long AVX512_MASK = (1ULL << 1) | (1ULL << 2) |
                                               (1ULL << 5) | (1ULL << 6) | (1ULL << 7);
    return (xcr0 & AVX512_MASK) == AVX512_MASK;
}

/**
 * Detect CPU features once at program startup
 *
 * Uses CPUID instruction to query supported ISA extensions,
 * with OS support validation via XGETBV.
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

        // Emit diagnostic if FMA is missing (AVX2 CPUs typically have it)
        if (features.has_avx2 && !features.has_fma) {
            #ifndef NDEBUG
            std::cerr << "Warning: AVX2 detected but FMA not available\n";
            #endif
        }
    }

    // Check for AVX-512 (requires OS support)
    if (os_avx512_support && __get_cpuid_count(7, 0, &eax, &ebx, &ecx, &edx)) {
        features.has_avx512f = (ebx & bit_AVX512F) != 0;
    }

    return features;
}

/**
 * Select best ISA target for current CPU
 *
 * Called once at solver construction, result cached in solver.
 *
 * NOTE: This is DIAGNOSTIC ONLY for logging/stats. The actual kernel
 * dispatch happens automatically via [[gnu::target_clones]] IFUNC
 * resolution at runtime. This function merely reports what the IFUNC
 * resolver will choose.
 */
inline ISATarget select_isa_target() {
    static const CPUFeatures features = detect_cpu_features();

    if (features.has_avx512f) {
        return ISATarget::AVX512F;
    } else if (features.has_avx2 && features.has_fma) {
        return ISATarget::AVX2;
    } else {
        return ISATarget::DEFAULT;
    }
}

/// Get human-readable ISA target name (for logging/diagnostics)
inline std::string isa_target_name(ISATarget target) {
    switch (target) {
        case ISATarget::DEFAULT: return "SSE2";
        case ISATarget::AVX2: return "AVX2+FMA";
        case ISATarget::AVX512F: return "AVX512F";
        default: return "UNKNOWN";
    }
}

} // namespace mango::cpu
