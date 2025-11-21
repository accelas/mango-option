# CPU Detection Refactoring

**Date:** 2025-11-21
**Context:** After removing SimdBackend and adopting target_clones everywhere

## Question

Do we still need manual CPU detection in `src/support/cpu/feature_detection.hpp` now that `[[gnu::target_clones]]` handles dispatch automatically?

## Answer: Yes, but refactor for clarity

### What to Keep

1. **`detect_cpu_features()`** - Diagnostic function to query CPU capabilities
2. **`check_os_avx_support()`** - Validates OS has enabled XSAVE (safety check)
3. **`check_os_avx512_support()`** - Validates AVX-512 state is enabled
4. **`isa_target_name()`** - Human-readable ISA names for logging

### What to Remove

1. **`ISATarget` enum** - No longer used for dispatch
2. **`select_isa_target()`** - Duplicates what target_clones resolver does

### What to Change

Rename the header to clarify its purpose:
- **Old:** `feature_detection.hpp` (implies it's used for dispatch)
- **New:** `cpu_diagnostics.hpp` (clarifies it's for logging/info only)

## Refactored API

```cpp
namespace mango::cpu {

/// CPU feature flags detected at runtime (diagnostic only)
struct CPUFeatures {
    bool has_sse2 = false;
    bool has_avx2 = false;
    bool has_avx512f = false;
    bool has_fma = false;
};

/**
 * Detect CPU features for diagnostic purposes
 *
 * NOTE: Do NOT use this for dispatch. Use [[gnu::target_clones]]
 * which provides zero-overhead IFUNC resolution.
 */
CPUFeatures detect_cpu_features();

/**
 * Check if OS has enabled XSAVE for AVX state
 *
 * IMPORTANT: AVX/AVX-512 require OS support. Without OSXSAVE,
 * executing AVX instructions will SIGILL even if CPUID reports support.
 */
bool check_os_avx_support();

/**
 * Check if OS has enabled AVX-512 state (ZMM registers)
 */
bool check_os_avx512_support();

/**
 * Get human-readable description of CPU features (for logging)
 *
 * Example: "AVX512F+FMA (8-wide SIMD)" or "AVX2+FMA (4-wide SIMD)"
 */
std::string describe_cpu_features();

/**
 * Print CPU features to stdout (for debugging)
 */
void print_cpu_info();

} // namespace mango::cpu
```

## Usage Examples

### Example 1: Diagnostic Logging

```cpp
#include "src/support/cpu/cpu_diagnostics.hpp"

void log_solver_config() {
    auto features = mango::cpu::detect_cpu_features();

    if (features.has_avx512f) {
        std::cout << "Using AVX-512 (8-wide SIMD)\n";
    } else if (features.has_avx2) {
        std::cout << "Using AVX2 (4-wide SIMD)\n";
    } else {
        std::cout << "Using SSE2 (2-wide SIMD)\n";
    }
}
```

### Example 2: Python Bindings

```python
import mango_iv

# Query CPU info
cpu_info = mango_iv.get_cpu_info()
print(f"CPU supports: {cpu_info['features']}")
print(f"SIMD width: {cpu_info['simd_width']}")
```

### Example 3: Testing

```cpp
TEST(CPUDiagnosticsTest, OSAVXSupport) {
    auto features = mango::cpu::detect_cpu_features();

    if (features.has_avx2 || features.has_avx512f) {
        // If CPU reports AVX, OS support must be enabled
        EXPECT_TRUE(mango::cpu::check_os_avx_support());
    }
}
```

## What NOT to Do

### ❌ Do NOT use for dispatch

```cpp
// WRONG: Manual dispatch based on CPU features
void my_function() {
    auto features = mango::cpu::detect_cpu_features();

    if (features.has_avx512f) {
        my_function_avx512();
    } else if (features.has_avx2) {
        my_function_avx2();
    } else {
        my_function_sse2();
    }
}
```

### ✅ Do use target_clones for dispatch

```cpp
// CORRECT: Let compiler handle dispatch
[[gnu::target_clones("default","avx2","avx512f")]]
void my_function() {
    #pragma omp simd
    for (...) { /* vectorized loop */ }
}
```

## Migration Plan

### Phase 1: Refactor API (Now)
- [ ] Rename `feature_detection.hpp` → `cpu_diagnostics.hpp`
- [ ] Remove `ISATarget` enum
- [ ] Remove `select_isa_target()` function
- [ ] Add `describe_cpu_features()` helper
- [ ] Add `print_cpu_info()` convenience function
- [ ] Update all includes

### Phase 2: Update CenteredDifference (Now)
- [ ] Remove `Mode` enum (no longer needed)
- [ ] Remove CPU detection from constructor
- [ ] Always use ScalarBackend (with target_clones)
- [ ] Remove virtual dispatch overhead (direct calls)

### Phase 3: Add Python Bindings (Later)
- [ ] Expose `get_cpu_info()` to Python
- [ ] Return dict with CPU features
- [ ] Show in solver diagnostic output

### Phase 4: Documentation (Later)
- [ ] Add section to README: "How does ISA selection work?"
- [ ] Explain target_clones automatic dispatch
- [ ] Document diagnostic API for advanced users

## Simplified Architecture After Refactoring

```
┌─────────────────────────────────────────────────────┐
│  User Code                                          │
│                                                     │
│  CenteredDifference stencil(spacing);              │
│  stencil.compute_second_derivative(...);           │
│                                                     │
└──────────────────┬──────────────────────────────────┘
                   │
                   ↓ (no Mode enum, no virtual dispatch)
         ┌─────────────────────┐
         │  ScalarBackend       │
         │  (OpenMP SIMD)       │
         └──────────┬───────────┘
                    │
                    ↓ (target_clones)
      ┌─────────────┴─────────────┐
      │   Compiler generates 3     │
      │   ISA-specific versions    │
      └─────────────┬──────────────┘
                    ↓
    ┌───────────────┼───────────────┐
    ↓               ↓               ↓
┌─────────┐    ┌─────────┐    ┌──────────┐
│.default │    │ .avx2   │    │ .avx512f │
│(SSE2)   │    │(4-wide) │    │(8-wide)  │
└─────────┘    └─────────┘    └──────────┘
    ↑               ↑               ↑
    └───────────────┴───────────────┘
                    │
              ┌─────┴──────┐
              │ .resolver   │
              │             │
              │ CPUID check │
              │ at runtime  │
              └─────────────┘
```

**Diagnostic API (separate, not in critical path):**
```
mango::cpu::detect_cpu_features() → for logging/stats only
```

## Benefits of This Approach

1. **Zero dispatch overhead** - Direct function calls, no virtual dispatch
2. **Simpler code** - Remove Mode enum, Backend interface, unique_ptr indirection
3. **Clear purpose** - CPU diagnostics separate from performance-critical code
4. **Better performance** - ~5-10ns virtual dispatch overhead eliminated
5. **Maintainable** - One vectorization strategy, not two

## Summary

**Keep CPU detection code, but:**
- ✅ Rename to emphasize it's diagnostic only
- ✅ Remove dispatch-related functions (ISATarget, select_isa_target)
- ✅ Add convenience functions for logging
- ✅ Document clearly: "Do NOT use for dispatch"

**Use target_clones for dispatch:**
- ✅ Automatic ISA selection (zero overhead)
- ✅ Compiler handles CPUID + resolver generation
- ✅ Simpler code (no Mode enum, no virtual dispatch)
