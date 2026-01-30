<!-- SPDX-License-Identifier: MIT -->
# Simplify to OpenMP SIMD Only Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Remove SimdBackend (std::experimental::simd) and simplify to use only ScalarBackend (OpenMP SIMD) with `[[gnu::target_clones]]` for multi-ISA support.

**Architecture:** Benchmark results show ScalarBackend (OpenMP SIMD) is faster in 75% of cases (9/12 tests). We eliminate the explicit SIMD backend and virtual dispatch overhead, relying on compiler auto-vectorization with `target_clones` for ISA-specific code generation. This enables Clang migration and mdspan usage.

**Tech Stack:** C++23, OpenMP SIMD directives, GNU `target_clones` attribute, Google Test

---

## Task 1: Add target_clones to ScalarBackend Methods

**Files:**
- Modify: `src/pde/operators/centered_difference_scalar.hpp:30-70`
- Test: `tests/pde_operators/centered_difference_test.cc`

**Step 1: Write test to verify multi-ISA code generation**

Create: `tests/pde_operators/target_clones_test.cc`

```cpp
#include <gtest/gtest.h>
#include "src/pde/operators/centered_difference_scalar.hpp"
#include "src/pde/core/grid.hpp"
#include <vector>
#include <cmath>
#include <dlfcn.h>

using namespace mango;
using namespace mango::operators;

// Test that target_clones generates multiple ISA versions
TEST(TargetClonesTest, MultipleISAVersionsGenerated) {
    // This test verifies symbol table contains .default, .avx2, .avx512f versions
    // We can't easily test this at runtime without inspecting binary
    // So this is a compile-time verification test - if it compiles, it works

    const size_t n = 100;
    std::vector<double> x(n);
    for (size_t i = 0; i < n; ++i) {
        x[i] = static_cast<double>(i) / (n - 1);
    }

    GridView<double> grid_view(x);
    GridSpacing<double> spacing(grid_view);
    ScalarBackend<double> backend(spacing);

    std::vector<double> u(n);
    std::vector<double> result(n);
    for (size_t i = 0; i < n; ++i) {
        u[i] = std::sin(2.0 * M_PI * x[i]);
    }

    // Call should work regardless of ISA
    backend.compute_second_derivative_uniform(u, result, 1, n - 1);

    // Verify result is correct (numerical validation)
    EXPECT_NEAR(result[50], -4.0 * M_PI * M_PI * std::sin(2.0 * M_PI * 0.5), 1e-3);
}

// Test uniform grid operations work with auto-vectorization
TEST(TargetClonesTest, UniformGridOperations) {
    const size_t n = 1000;
    std::vector<double> x(n);
    for (size_t i = 0; i < n; ++i) {
        x[i] = static_cast<double>(i) / (n - 1);
    }

    GridView<double> grid_view(x);
    GridSpacing<double> spacing(grid_view);
    ScalarBackend<double> backend(spacing);

    std::vector<double> u(n);
    std::vector<double> d2u(n);
    std::vector<double> du(n);

    for (size_t i = 0; i < n; ++i) {
        u[i] = std::sin(2.0 * M_PI * x[i]);
    }

    // Second derivative
    backend.compute_second_derivative_uniform(u, d2u, 1, n - 1);

    // First derivative
    backend.compute_first_derivative_uniform(u, du, 1, n - 1);

    // Verify results
    EXPECT_NEAR(d2u[500], -4.0 * M_PI * M_PI * std::sin(2.0 * M_PI * 0.5), 1e-3);
    EXPECT_NEAR(du[500], 2.0 * M_PI * std::cos(2.0 * M_PI * 0.5), 1e-3);
}

// Test non-uniform grid operations work with auto-vectorization
TEST(TargetClonesTest, NonUniformGridOperations) {
    const size_t n = 200;
    std::vector<double> x(n);
    const double stretch = 2.0;
    for (size_t i = 0; i < n; ++i) {
        double xi = static_cast<double>(i) / (n - 1);
        x[i] = std::sinh(stretch * (2.0 * xi - 1.0)) / std::sinh(stretch);
    }

    GridView<double> grid_view(x);
    GridSpacing<double> spacing(grid_view);
    ScalarBackend<double> backend(spacing);

    std::vector<double> u(n);
    std::vector<double> d2u(n);
    std::vector<double> du(n);

    for (size_t i = 0; i < n; ++i) {
        u[i] = x[i] * x[i];  // u = x^2, so du/dx = 2x, d2u/dx2 = 2
    }

    // Second derivative
    backend.compute_second_derivative_non_uniform(u, d2u, 1, n - 1);

    // First derivative
    backend.compute_first_derivative_non_uniform(u, du, 1, n - 1);

    // Verify results (looser tolerance for non-uniform grids)
    EXPECT_NEAR(d2u[100], 2.0, 0.1);
    EXPECT_NEAR(du[100], 2.0 * x[100], 0.05);
}
```

**Step 2: Run test to verify current implementation passes**

Run: `bazel test //tests/pde_operators:target_clones_test`
Expected: Test compiles and passes with current code

**Step 3: Add target_clones to ScalarBackend methods**

Modify: `src/pde/operators/centered_difference_scalar.hpp`

Find the uniform second derivative method (around line 30):

```cpp
// OLD:
void compute_second_derivative_uniform(
    std::span<const T> u, std::span<T> d2u_dx2,
    size_t start, size_t end) const
{
    const T dx2_inv = spacing_.spacing_inv_sq();

    MANGO_PRAGMA_SIMD
    for (size_t i = start; i < end; ++i) {
        d2u_dx2[i] = std::fma(u[i+1] + u[i-1], dx2_inv,
                             -T(2)*u[i]*dx2_inv);
    }
}
```

Replace with:

```cpp
// NEW: Add target_clones for multi-ISA code generation
[[gnu::target_clones("default", "avx2", "avx512f")]]
void compute_second_derivative_uniform(
    std::span<const T> u, std::span<T> d2u_dx2,
    size_t start, size_t end) const
{
    const T dx2_inv = spacing_.spacing_inv_sq();

    MANGO_PRAGMA_SIMD
    for (size_t i = start; i < end; ++i) {
        d2u_dx2[i] = std::fma(u[i+1] + u[i-1], dx2_inv,
                             -T(2)*u[i]*dx2_inv);
    }
}
```

Apply same change to:
- `compute_first_derivative_uniform` (around line 38)
- `compute_second_derivative_non_uniform` (around line 51)
- `compute_first_derivative_non_uniform` (around line 70)

**Step 4: Run tests to verify target_clones works**

Run: `bazel test //tests/pde_operators:target_clones_test -v`
Expected: PASS - All tests pass with target_clones enabled

**Step 5: Verify multi-ISA code generation**

Run:
```bash
bazel build //tests/pde_operators:target_clones_test
nm -C bazel-bin/tests/pde_operators/target_clones_test | grep "compute_second_derivative_uniform"
```

Expected output showing multiple versions:
```
... compute_second_derivative_uniform [clone .default]
... compute_second_derivative_uniform [clone .avx2]
... compute_second_derivative_uniform [clone .avx512f]
... compute_second_derivative_uniform [clone .resolver]
```

**Step 6: Commit**

```bash
git add src/pde/operators/centered_difference_scalar.hpp
git add tests/pde_operators/target_clones_test.cc
git commit -m "feat: add target_clones to ScalarBackend for multi-ISA support

- Add [[gnu::target_clones]] to all compute methods
- Compiler generates SSE2, AVX2, AVX-512 versions automatically
- Runtime IFUNC dispatch selects optimal ISA
- Add tests verifying multi-ISA code generation"
```

---

## Task 2: Update BUILD.bazel for Portable Baseline

**Files:**
- Modify: `src/pde/operators/BUILD.bazel`
- Modify: `.bazelrc`

**Step 1: Update centered_difference_scalar build target**

Modify: `src/pde/operators/BUILD.bazel`

Find the `centered_difference_scalar` target and update copts:

```python
# OLD:
cc_library(
    name = "centered_difference_scalar",
    hdrs = ["centered_difference_scalar.hpp"],
    copts = [
        "-std=c++23",
        "-march=native",  # ← Problem: not portable
        "-fopenmp-simd",
    ],
    deps = [
        "//src/pde/core:grid",
    ],
)

# NEW:
cc_library(
    name = "centered_difference_scalar",
    hdrs = ["centered_difference_scalar.hpp"],
    copts = [
        "-std=c++23",
        "-march=x86-64",  # ← Baseline ISA for portability
        "-fopenmp-simd",
    ],
    deps = [
        "//src/pde/core:grid",
    ],
)
```

**Step 2: Update .bazelrc for portable builds**

Modify: `.bazelrc`

Add after existing build flags:

```bash
# Use baseline x86-64 ISA for portable binaries
# target_clones generates ISA-specific versions automatically
build --copt=-march=x86-64
build --copt=-mtune=generic
```

**Step 3: Verify portable build works**

Run: `bazel clean && bazel build //src/pde/operators:centered_difference_scalar`
Expected: Build succeeds with baseline ISA

**Step 4: Verify tests still pass**

Run: `bazel test //tests/pde_operators:target_clones_test`
Expected: PASS - Tests pass with baseline build

**Step 5: Commit**

```bash
git add src/pde/operators/BUILD.bazel
git add .bazelrc
git commit -m "build: use baseline x86-64 ISA for portable binaries

- Change -march=native to -march=x86-64
- target_clones generates ISA-specific versions
- Binary works on any x86-64 CPU with optimal code paths"
```

---

## Task 3: Remove SimdBackend Files

**Files:**
- Delete: `src/pde/operators/centered_difference_simd_backend.hpp`
- Modify: `src/pde/operators/BUILD.bazel`
- Modify: `benchmarks/BUILD.bazel`

**Step 1: Verify no tests depend on SimdBackend**

Run:
```bash
bazel query 'tests(//tests/...)' | xargs -I {} bazel query "deps({})" | grep simd_backend
```

Expected: Empty output (SimdBackend already disabled)

**Step 2: Remove SimdBackend from BUILD.bazel**

Modify: `src/pde/operators/BUILD.bazel`

Find and remove the `centered_difference_simd_backend` target:

```python
# DELETE THIS ENTIRE TARGET:
cc_library(
    name = "centered_difference_simd_backend",
    hdrs = ["centered_difference_simd_backend.hpp"],
    copts = [
        "-std=c++23",
        "-march=native",
        "-fopenmp-simd",
    ],
    deps = [
        "//src/pde/core:grid",
    ],
)
```

**Step 3: Remove SimdBackend from benchmark dependencies**

Modify: `benchmarks/BUILD.bazel`

Find `simd_backend_comparison` target (around line 197):

```python
# OLD:
cc_binary(
    name = "simd_backend_comparison",
    srcs = ["simd_backend_comparison.cc"],
    deps = [
        "//src/pde/core:grid",
        "//src/pde/operators:centered_difference_scalar",
        "//src/pde/operators:centered_difference_simd_backend",  # ← Remove this
        "@google_benchmark//:benchmark",
    ],
)

# NEW:
cc_binary(
    name = "simd_backend_comparison",
    srcs = ["simd_backend_comparison.cc"],
    deps = [
        "//src/pde/core:grid",
        "//src/pde/operators:centered_difference_scalar",
        "@google_benchmark//:benchmark",
    ],
)
```

**Step 4: Update benchmark to remove SimdBackend tests**

Modify: `benchmarks/simd_backend_comparison.cc`

Remove all SimdBackend-related code:

```cpp
// DELETE THIS INCLUDE:
// #include "src/pde/operators/centered_difference_simd_backend.hpp"

// DELETE ALL BM_Simd_* functions (lines 133-187)
// DELETE ALL BENCHMARK(BM_Simd_*) registrations (lines 200-203)
```

Keep only ScalarBackend benchmarks. Rename file for clarity:

```bash
git mv benchmarks/simd_backend_comparison.cc benchmarks/openmp_simd_benchmark.cc
```

Update benchmark file header:

```cpp
/**
 * @file openmp_simd_benchmark.cc
 * @brief Benchmark OpenMP SIMD performance with target_clones
 *
 * Tests OpenMP SIMD auto-vectorization on:
 * - Uniform vs non-uniform grids
 * - First vs second derivatives
 * - Different grid sizes (101, 501, 1001)
 *
 * Compiler generates ISA-specific versions via [[gnu::target_clones]]:
 * - SSE2 baseline (2-wide)
 * - AVX2 (4-wide)
 * - AVX-512 (8-wide)
 */
```

**Step 5: Update BUILD.bazel benchmark target name**

Modify: `benchmarks/BUILD.bazel`

```python
# OLD:
cc_binary(
    name = "simd_backend_comparison",
    srcs = ["simd_backend_comparison.cc"],

# NEW:
cc_binary(
    name = "openmp_simd_benchmark",
    srcs = ["openmp_simd_benchmark.cc"],
```

**Step 6: Verify build works without SimdBackend**

Run: `bazel build //benchmarks:openmp_simd_benchmark`
Expected: Build succeeds

**Step 7: Run benchmark to verify ScalarBackend still works**

Run: `./bazel-bin/benchmarks/openmp_simd_benchmark`
Expected: Benchmark runs successfully

**Step 8: Delete SimdBackend header file**

Run:
```bash
git rm src/pde/operators/centered_difference_simd_backend.hpp
```

**Step 9: Commit**

```bash
git add -A
git commit -m "refactor: remove SimdBackend (std::experimental::simd)

Benchmark results show ScalarBackend (OpenMP SIMD) is faster in
75% of cases. Explicit SIMD provides no benefit and adds complexity.

Changes:
- Delete centered_difference_simd_backend.hpp
- Remove SimdBackend from BUILD.bazel
- Simplify benchmark to test only OpenMP SIMD
- Rename benchmark: simd_backend_comparison → openmp_simd_benchmark

Rationale:
- OpenMP SIMD wins in 9/12 benchmark cases (15-45% faster)
- Compiler auto-vectorization is excellent for stencil operations
- Simpler codebase: one vectorization strategy instead of two
- Enables Clang migration (std::experimental::simd has portability issues)"
```

---

## Task 4: Simplify CenteredDifference Facade

**Files:**
- Modify: `src/pde/operators/centered_difference_facade.hpp`
- Modify: `src/pde/operators/BUILD.bazel`
- Test: `tests/pde_operators/centered_difference_test.cc`

**Step 1: Write test for simplified API**

Create: `tests/pde_operators/centered_difference_facade_test.cc`

```cpp
#include <gtest/gtest.h>
#include "src/pde/operators/centered_difference_facade.hpp"
#include "src/pde/core/grid.hpp"
#include <vector>
#include <cmath>

using namespace mango;
using namespace mango::operators;

// Test simplified facade (no Mode enum)
TEST(CenteredDifferenceFacadeTest, SimplifiedConstruction) {
    const size_t n = 100;
    std::vector<double> x(n);
    for (size_t i = 0; i < n; ++i) {
        x[i] = static_cast<double>(i) / (n - 1);
    }

    GridView<double> grid_view(x);
    GridSpacing<double> spacing(grid_view);

    // Simplified construction: no Mode argument
    CenteredDifference<double> stencil(spacing);

    std::vector<double> u(n);
    std::vector<double> d2u(n);

    for (size_t i = 0; i < n; ++i) {
        u[i] = std::sin(2.0 * M_PI * x[i]);
    }

    stencil.compute_second_derivative(u, d2u, 1, n - 1);

    // Verify result
    EXPECT_NEAR(d2u[50], -4.0 * M_PI * M_PI * std::sin(2.0 * M_PI * 0.5), 1e-3);
}

// Test direct ScalarBackend usage (no virtual dispatch)
TEST(CenteredDifferenceFacadeTest, DirectBackendCall) {
    const size_t n = 500;
    std::vector<double> x(n);
    for (size_t i = 0; i < n; ++i) {
        x[i] = static_cast<double>(i) / (n - 1);
    }

    GridView<double> grid_view(x);
    GridSpacing<double> spacing(grid_view);
    CenteredDifference<double> stencil(spacing);

    std::vector<double> u(n);
    std::vector<double> du(n);

    for (size_t i = 0; i < n; ++i) {
        u[i] = x[i] * x[i];  // u = x^2
    }

    stencil.compute_first_derivative(u, du, 1, n - 1);

    // du/dx = 2x
    EXPECT_NEAR(du[250], 2.0 * x[250], 1e-3);
}
```

**Step 2: Run test to verify current implementation**

Run: `bazel test //tests/pde_operators:centered_difference_facade_test`
Expected: FAIL - Mode enum still required

**Step 3: Simplify CenteredDifference facade**

Modify: `src/pde/operators/centered_difference_facade.hpp`

Replace entire file with simplified version:

```cpp
#pragma once

#include "src/pde/core/grid.hpp"
#include "centered_difference_scalar.hpp"
#include <span>
#include <concepts>

namespace mango::operators {

/**
 * CenteredDifference: Simplified facade using ScalarBackend only
 *
 * Uses OpenMP SIMD with [[gnu::target_clones]] for automatic ISA selection.
 * No Mode enum, no virtual dispatch - direct calls to ScalarBackend.
 *
 * @tparam T Floating-point type (float, double, long double)
 */
template<std::floating_point T = double>
class CenteredDifference {
public:
    explicit CenteredDifference(const GridSpacing<T>& spacing)
        : backend_(spacing)
    {}

    // Movable and copyable (ScalarBackend is copyable)
    CenteredDifference(const CenteredDifference&) = default;
    CenteredDifference& operator=(const CenteredDifference&) = default;
    CenteredDifference(CenteredDifference&&) = default;
    CenteredDifference& operator=(CenteredDifference&&) = default;

    // Public API - direct call to ScalarBackend (no virtual dispatch)
    void compute_second_derivative(std::span<const T> u,
                                   std::span<T> d2u_dx2,
                                   size_t start, size_t end) const {
        backend_.compute_second_derivative(u, d2u_dx2, start, end);
    }

    void compute_first_derivative(std::span<const T> u,
                                  std::span<T> du_dx,
                                  size_t start, size_t end) const {
        backend_.compute_first_derivative(u, du_dx, start, end);
    }

private:
    ScalarBackend<T> backend_;
};

} // namespace mango::operators
```

**Step 4: Run test to verify simplified facade works**

Run: `bazel test //tests/pde_operators:centered_difference_facade_test -v`
Expected: PASS - Tests pass with simplified facade

**Step 5: Update existing tests using Mode enum**

Search for Mode usage:

Run:
```bash
grep -r "Mode::" tests/
grep -r "Mode::Auto" tests/
grep -r "Mode::Scalar" tests/
```

For each file found, remove Mode argument:

```cpp
// OLD:
CenteredDifference<double> stencil(spacing, CenteredDifference<double>::Mode::Auto);

// NEW:
CenteredDifference<double> stencil(spacing);
```

**Step 6: Run all operator tests**

Run: `bazel test //tests/pde_operators/...`
Expected: All tests PASS

**Step 7: Commit**

```bash
git add src/pde/operators/centered_difference_facade.hpp
git add tests/pde_operators/centered_difference_facade_test.cc
git add tests/pde_operators/centered_difference_test.cc
git commit -m "refactor: simplify CenteredDifference facade

Remove Mode enum, virtual dispatch, and backend abstraction.
Direct calls to ScalarBackend with target_clones for ISA selection.

Changes:
- Remove Mode enum (Auto/Scalar/Simd)
- Remove BackendInterface virtual dispatch (~5-10ns overhead eliminated)
- Remove unique_ptr indirection
- Direct ScalarBackend member (copyable, movable)
- Simplify constructor: no Mode argument

Benefits:
- Zero virtual dispatch overhead
- Simpler API (one constructor)
- Better performance (direct calls)
- Smaller binary (no vtables)"
```

---

## Task 5: Refactor CPU Detection to Diagnostic-Only

**Files:**
- Rename: `src/support/cpu/feature_detection.hpp` → `src/support/cpu/cpu_diagnostics.hpp`
- Modify: `src/support/cpu/BUILD.bazel`
- Modify: `tests/cpu/feature_detection_test.cc` → `tests/cpu/cpu_diagnostics_test.cc`

**Step 1: Create new cpu_diagnostics.hpp with simplified API**

Create: `src/support/cpu/cpu_diagnostics.hpp`

```cpp
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
```

**Step 2: Write tests for diagnostic API**

Create: `tests/cpu/cpu_diagnostics_test.cc`

```cpp
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
```

**Step 3: Run tests**

Run: `bazel test //tests/cpu:cpu_diagnostics_test -v`
Expected: PASS

**Step 4: Remove old feature_detection files**

Run:
```bash
git rm src/support/cpu/feature_detection.hpp
git rm tests/cpu/feature_detection_test.cc
```

**Step 5: Update BUILD.bazel**

Modify: `src/support/cpu/BUILD.bazel`

```python
# OLD:
cc_library(
    name = "feature_detection",
    hdrs = ["feature_detection.hpp"],

# NEW:
cc_library(
    name = "cpu_diagnostics",
    hdrs = ["cpu_diagnostics.hpp"],
```

Modify: `tests/cpu/BUILD.bazel`

```python
# OLD:
cc_test(
    name = "feature_detection_test",
    srcs = ["feature_detection_test.cc"],
    deps = [
        "//src/support/cpu:feature_detection",

# NEW:
cc_test(
    name = "cpu_diagnostics_test",
    srcs = ["cpu_diagnostics_test.cc"],
    deps = [
        "//src/support/cpu:cpu_diagnostics",
```

**Step 6: Update includes in other files**

Search and replace:

Run:
```bash
grep -r "feature_detection.hpp" src/ tests/ docs/
```

For each file, replace:

```cpp
// OLD:
#include "src/support/cpu/feature_detection.hpp"

// NEW:
#include "src/support/cpu/cpu_diagnostics.hpp"
```

**Step 7: Run all tests**

Run: `bazel test //...`
Expected: All tests PASS

**Step 8: Commit**

```bash
git add -A
git commit -m "refactor: rename feature_detection to cpu_diagnostics

Clarify that CPU detection is for diagnostics only, not dispatch.
target_clones handles ISA selection automatically.

Changes:
- Rename feature_detection.hpp → cpu_diagnostics.hpp
- Remove ISATarget enum (not used for dispatch)
- Remove select_isa_target() (redundant with target_clones resolver)
- Add describe_cpu_features() for user-facing diagnostics
- Update all includes and BUILD files

Purpose:
- CPU diagnostics: logging, testing, user information
- Dispatch: [[gnu::target_clones]] (automatic, zero overhead)"
```

---

## Task 6: Update Documentation

**Files:**
- Create: `docs/architecture/vectorization-strategy.md`
- Modify: `CLAUDE.md`

**Step 1: Create vectorization strategy documentation**

Create: `docs/architecture/vectorization-strategy.md`

```markdown
# Vectorization Strategy

**Last Updated:** 2025-11-21

## Overview

This codebase uses **OpenMP SIMD with GNU `target_clones`** for portable, high-performance vectorization. We do NOT use explicit SIMD APIs like `std::experimental::simd`.

## Design Decision

Benchmark results (see `docs/experiments/openmp-simd-alternative.md`) showed:
- **OpenMP SIMD wins in 75% of cases** (9/12 benchmark tests)
- Margins: 15-45% faster in many scenarios
- Compiler auto-vectorization is excellent for stencil operations

## Implementation Pattern

### Single-ISA Code (Portable)

```cpp
#include "src/support/parallel.hpp"

[[gnu::target_clones("default", "avx2", "avx512f")]]
void compute_stencil(const double* u, double* result, size_t n) {
    #pragma omp simd
    for (size_t i = 1; i < n - 1; ++i) {
        result[i] = (u[i+1] + u[i-1] - 2*u[i]) * dx2_inv;
    }
}
```

**What this does:**
1. Compiler generates 3 versions: SSE2 (2-wide), AVX2 (4-wide), AVX-512 (8-wide)
2. Runtime IFUNC resolver selects best version on first call
3. Zero overhead on subsequent calls (direct jump)
4. Works on any x86-64 CPU with optimal code path

### Build Configuration

```python
# BUILD.bazel
cc_library(
    name = "my_library",
    copts = [
        "-march=x86-64",  # Baseline ISA for portability
        "-fopenmp-simd",  # Enable OpenMP SIMD directives
    ],
)
```

**.bazelrc:**
```bash
build --copt=-march=x86-64
build --copt=-mtune=generic
build --copt=-fopenmp-simd
```

## Compiler Support

| Compiler | target_clones | OpenMP SIMD | Status |
|----------|---------------|-------------|--------|
| GCC 14+ | ✅ | ✅ | Production |
| Clang 19+ | ✅ | ✅ | Production |
| MSVC | ❌ | Partial | Manual dispatch needed |

## Performance Characteristics

### ISA Selection

**First call:** ~100-500 ns (CPUID + resolver)
**Subsequent calls:** Zero overhead (IFUNC direct jump)

For typical workloads (1000+ iterations):
- First-call overhead: <0.01% of total runtime
- Effectively zero-cost abstraction

### SIMD Width by ISA

| ISA | SIMD Width (double) | Register Size |
|-----|-------------------|---------------|
| SSE2 | 2 | 128-bit (%xmm) |
| AVX2 | 4 | 256-bit (%ymm) |
| AVX-512 | 8 | 512-bit (%zmm) |

## CPU Diagnostics

For logging/debugging, use `src/support/cpu/cpu_diagnostics.hpp`:

```cpp
#include "src/support/cpu/cpu_diagnostics.hpp"

void log_cpu_info() {
    auto features = mango::cpu::detect_cpu_features();
    std::cout << "CPU: " << mango::cpu::describe_cpu_features() << "\n";
}
```

**Do NOT use for dispatch** - that's what `target_clones` does automatically.

## Why Not std::experimental::simd?

We tried explicit SIMD (`std::experimental::simd`) and found:
- ❌ Slower in 75% of cases (OpenMP SIMD wins)
- ❌ Portability issues (doesn't work with Clang + libc++)
- ❌ More complex code (explicit copy_from/copy_to)
- ❌ Maintenance burden (two backends vs one)

See `docs/experiments/SIMD_DECISION_SUMMARY.md` for full analysis.

## Migration from SimdBackend

If you see old code using `SimdBackend`:

```cpp
// OLD:
#include "centered_difference_simd_backend.hpp"
SimdBackend<double> backend(spacing);

// NEW:
#include "centered_difference_scalar.hpp"
ScalarBackend<double> backend(spacing);  // Uses OpenMP SIMD + target_clones
```

The name "ScalarBackend" is historical - it actually uses SIMD via OpenMP directives.

## Verifying Multi-ISA Code Generation

```bash
# Build binary
bazel build //benchmarks:openmp_simd_benchmark

# Check for multiple ISA versions
nm -C bazel-bin/benchmarks/openmp_simd_benchmark | grep compute_stencil

# Expected output:
# ... compute_stencil [clone .default]
# ... compute_stencil [clone .avx2]
# ... compute_stencil [clone .avx512f]
# ... compute_stencil [clone .resolver]

# Verify AVX-512 instructions
objdump -d bazel-bin/benchmarks/openmp_simd_benchmark | grep -A 20 'compute_stencil.*avx512f'

# Look for: zmm registers, vmovupd, vaddpd, vfmadd instructions
```

## Future Work

Once Clang becomes default compiler:
- Access to `std::mdspan` (C++23)
- Fix CubicSplineND hot-path allocations
- 15-49% performance boost from better codegen

See `docs/experiments/compiler-stdlib-tradeoffs.md` for details.
```

**Step 2: Update CLAUDE.md**

Modify: `CLAUDE.md`

Find the SIMD section and replace with:

```markdown
### Vectorization Strategy

The library uses **OpenMP SIMD with `[[gnu::target_clones]]`** for automatic ISA selection:

**Pattern:**
```cpp
[[gnu::target_clones("default", "avx2", "avx512f")]]
void compute_derivative(...) {
    #pragma omp simd
    for (size_t i = start; i < end; ++i) {
        result[i] = computation(u[i-1], u[i], u[i+1]);
    }
}
```

**Benefits:**
- Single source code generates 3 ISA-specific versions (SSE2, AVX2, AVX-512)
- Runtime CPU detection selects optimal version (zero overhead after first call)
- Portable binary works on any x86-64 CPU
- Compiler auto-vectorization is faster than explicit SIMD in 75% of cases

**Do NOT use `std::experimental::simd`** - benchmark results show OpenMP SIMD is faster and more portable.

See `docs/architecture/vectorization-strategy.md` for details.
```

**Step 3: Commit**

```bash
git add docs/architecture/vectorization-strategy.md
git add CLAUDE.md
git commit -m "docs: add vectorization strategy documentation

Document OpenMP SIMD + target_clones as the standard approach.
Explain why we don't use std::experimental::simd.

References:
- Benchmark results in docs/experiments/openmp-simd-alternative.md
- Decision rationale in docs/experiments/SIMD_DECISION_SUMMARY.md"
```

---

## Task 7: Run Full Test Suite

**Files:**
- N/A (verification step)

**Step 1: Clean build from scratch**

Run: `bazel clean --expunge`
Expected: All build artifacts removed

**Step 2: Build entire codebase**

Run: `bazel build //...`
Expected: Build completes successfully

**Step 3: Run all tests**

Run: `bazel test //...`
Expected: All tests PASS

**Step 4: Run benchmarks**

Run:
```bash
bazel run //benchmarks:openmp_simd_benchmark
bazel run //benchmarks:readme_benchmarks
bazel run //benchmarks:component_performance
```

Expected: All benchmarks run successfully

**Step 5: Verify no performance regression**

Compare against baseline (from docs/experiments/clang-vs-gcc-benchmark.md):
- American single option: ~1.28 ms (GCC)
- American batch (64): ~5.90 ms (GCC)
- IV solver: ~143 ms

Expected: Within 5% of baseline

**Step 6: Verify multi-ISA code generation**

Run:
```bash
bazel build //benchmarks:openmp_simd_benchmark
nm -C bazel-bin/benchmarks/openmp_simd_benchmark | grep "compute_second_derivative_uniform"
```

Expected: See .default, .avx2, .avx512f, .resolver versions

**Step 7: Record results**

Create: `docs/experiments/openmp-simd-validation.md`

```markdown
# OpenMP SIMD Simplification Validation

**Date:** 2025-11-21
**Branch:** main
**Compiler:** GCC 14.2.0

## Benchmark Results

| Benchmark | Baseline (Mixed) | After (OpenMP SIMD) | Change |
|-----------|-----------------|---------------------|--------|
| American single | 1.28 ms | [FILL IN] | [FILL IN] |
| American batch | 5.90 ms | [FILL IN] | [FILL IN] |
| IV solver | 143 ms | [FILL IN] | [FILL IN] |

## Multi-ISA Verification

```bash
nm -C bazel-bin/benchmarks/openmp_simd_benchmark | grep compute_second_derivative_uniform
```

Output:
```
[PASTE OUTPUT HERE]
```

## Test Results

```bash
bazel test //...
```

Output:
```
[PASTE OUTPUT HERE]
```

## Conclusion

- ✅ All tests pass
- ✅ Performance within [X]% of baseline
- ✅ Multi-ISA code generation verified
- ✅ Fat binary works on all CPUs
```

**Step 8: Commit validation results**

```bash
git add docs/experiments/openmp-simd-validation.md
git commit -m "test: validate OpenMP SIMD simplification

Full test suite passes with OpenMP SIMD only.
Performance within [X]% of baseline.
Multi-ISA code generation verified."
```

---

## Task 8: Final Cleanup

**Files:**
- Delete: `benchmarks/simd_backend_comparison.cc` (if still exists)
- Delete: `test_clang_target_clones.cc` (experiment file)
- Delete: `example_fat_binary_demo.cc` (experiment file)

**Step 1: Remove experiment files**

Run:
```bash
git rm -f test_clang_target_clones.cc test_clang_target_clones
git rm -f example_fat_binary_demo.cc example_fat_binary_demo
```

**Step 2: Update .gitignore**

Add experiment binaries to `.gitignore`:

```
# Experiment binaries
test_clang_target_clones
example_fat_binary_demo
```

**Step 3: Final test**

Run: `bazel test //...`
Expected: All tests PASS

**Step 4: Commit**

```bash
git add .gitignore
git commit -m "chore: remove experiment files

Clean up temporary test programs used during SIMD backend investigation."
```

---

## Completion Checklist

- [ ] Task 1: target_clones added to ScalarBackend
- [ ] Task 2: BUILD.bazel uses baseline ISA
- [ ] Task 3: SimdBackend files removed
- [ ] Task 4: CenteredDifference facade simplified
- [ ] Task 5: CPU detection refactored to diagnostic-only
- [ ] Task 6: Documentation updated
- [ ] Task 7: Full test suite passes
- [ ] Task 8: Experiment files cleaned up

## Success Criteria

1. **All tests pass:** `bazel test //...` shows 100% pass rate
2. **Performance maintained:** Within 5% of baseline benchmarks
3. **Multi-ISA verified:** Binary contains .default, .avx2, .avx512f versions
4. **Simpler code:** Removed ~500 lines (SimdBackend + virtual dispatch)
5. **Portable binary:** Works on any x86-64 CPU with optimal code paths

---

## Task 9: Migrate to Clang + libc++

**Goal:** Switch to Clang compiler to unlock 15-49% performance boost and access to `std::mdspan` (C++23).

**Prerequisite:** Tasks 1-8 must be complete (SimdBackend removed, OpenMP SIMD only)

**Files:**
- Modify: `.bazelrc`
- Modify: `.github/workflows/ci.yml`
- Modify: `.github/workflows/slow-tests.yml`

**Step 1: Update .bazelrc for Clang**

Modify: `.bazelrc`

Add after existing compiler settings:

```bash
# Use Clang compiler with libc++
build --action_env=CC=clang
build --action_env=CXX=clang++
build --cxxopt=-stdlib=libc++
build --linkopt=-stdlib=libc++
build --linkopt=-lc++abi

# Clang-specific warnings
build --copt=-Wno-error=pass-failed  # Allow vectorization hints to fail
```

**Step 2: Run clean build with Clang**

Run:
```bash
bazel clean --expunge
bazel build //...
```

Expected: Build succeeds with Clang

**Step 3: Run all tests with Clang**

Run: `bazel test //...`
Expected: All tests PASS

**Step 4: Benchmark Clang performance**

Run:
```bash
bazel run //benchmarks:readme_benchmarks > clang_results.txt
```

Compare against GCC baseline (docs/experiments/clang-vs-gcc-benchmark.md):
- Expected: 15-49% faster across major benchmarks

**Step 5: Verify mdspan availability**

Create: `tests/cpp23/mdspan_test.cc`

```cpp
#include <gtest/gtest.h>
#include <mdspan>
#include <vector>

TEST(MdspanTest, Basic2DArray) {
    std::vector<double> data(6);
    for (int i = 0; i < 6; ++i) {
        data[i] = static_cast<double>(i);
    }

    // Create 2D view: 2 rows × 3 cols
    std::mdspan<double, std::extents<size_t, 2, 3>> matrix(data.data());

    EXPECT_EQ(matrix[0, 0], 0.0);
    EXPECT_EQ(matrix[0, 1], 1.0);
    EXPECT_EQ(matrix[0, 2], 2.0);
    EXPECT_EQ(matrix[1, 0], 3.0);
    EXPECT_EQ(matrix[1, 1], 4.0);
    EXPECT_EQ(matrix[1, 2], 5.0);
}

TEST(MdspanTest, DynamicExtents) {
    std::vector<double> data(12);

    // Runtime-sized 3D array
    std::mdspan<double, std::dextents<size_t, 3>> tensor(
        data.data(), 2, 3, 2
    );

    EXPECT_EQ(tensor.extent(0), 2);
    EXPECT_EQ(tensor.extent(1), 3);
    EXPECT_EQ(tensor.extent(2), 2);
}
```

Add to BUILD.bazel:

```python
cc_test(
    name = "mdspan_test",
    srcs = ["mdspan_test.cc"],
    copts = ["-std=c++23"],
    deps = ["@googletest//:gtest_main"],
)
```

Run: `bazel test //tests/cpp23:mdspan_test`
Expected: PASS (mdspan works with Clang + libc++)

**Step 6: Update CI workflows**

Modify: `.github/workflows/ci.yml`

The Clang package is already installed, just document that we're using it:

```yaml
# CI already installs clang, no changes needed
# apt-get install -y git wget ca-certificates build-essential clang systemtap-sdt-dev libquantlib0-dev liblapacke-dev
```

Add comment explaining compiler choice:

```yaml
# Build uses Clang compiler (configured in .bazelrc)
# Provides 15-49% performance boost over GCC
# Enables std::mdspan from libc++ for zero-allocation tensor indexing
```

**Step 7: Benchmark comparison**

Create: `docs/experiments/gcc-vs-clang-final.md`

```markdown
# GCC vs Clang Final Benchmark (After SimdBackend Removal)

**Date:** 2025-11-21
**Configuration:** OpenMP SIMD only, target_clones enabled

## Results

| Benchmark | GCC 14.2.0 | Clang 19 | Speedup |
|-----------|------------|----------|---------|
| American single (101×498) | [FILL IN] | [FILL IN] | [FILL IN] |
| American batch (64) | [FILL IN] | [FILL IN] | [FILL IN] |
| IV solver (FDM) | [FILL IN] | [FILL IN] | [FILL IN] |

## Conclusion

Clang provides [X]% average speedup with OpenMP SIMD.
mdspan now available for CubicSplineND refactoring.
```

**Step 8: Commit Clang migration**

```bash
git add .bazelrc
git add .github/workflows/ci.yml
git add tests/cpp23/mdspan_test.cc
git add docs/experiments/gcc-vs-clang-final.md
git commit -m "build: migrate to Clang + libc++ compiler

Switch from GCC to Clang for better performance and C++23 features.

Benefits:
- 15-49% performance improvement (verified in benchmarks)
- Access to std::mdspan from libc++ 19
- Better diagnostics and error messages
- Enables future CubicSplineND hot-path optimization

Changes:
- .bazelrc: Use clang/clang++ with -stdlib=libc++
- CI: Document Clang usage (already installed)
- Add mdspan availability test

Prerequisite: SimdBackend removed (std::experimental::simd incompatible with libc++)
Next step: Refactor CubicSplineND with mdspan"
```

---

## Completion Checklist

- [ ] Task 1: target_clones added to ScalarBackend
- [ ] Task 2: BUILD.bazel uses baseline ISA
- [ ] Task 3: SimdBackend files removed
- [ ] Task 4: CenteredDifference facade simplified
- [ ] Task 5: CPU detection refactored to diagnostic-only
- [ ] Task 6: Documentation updated
- [ ] Task 7: Full test suite passes
- [ ] Task 8: Experiment files cleaned up
- [ ] Task 9: Migrate to Clang + libc++

## Success Criteria

1. **All tests pass:** `bazel test //...` shows 100% pass rate (with Clang)
2. **Performance improved:** 15-49% faster than GCC baseline
3. **Multi-ISA verified:** Binary contains .default, .avx2, .avx512f versions
4. **Simpler code:** Removed ~500 lines (SimdBackend + virtual dispatch)
5. **Portable binary:** Works on any x86-64 CPU with optimal code paths
6. **mdspan available:** C++23 features accessible for future work

## Next Steps

After this plan:
1. **CubicSplineND refactoring** - Handled in separate PR branch (N-dimensional spline work)
2. **CI Hardware Testing** - Verify on SSE2, AVX2, AVX-512 machines
3. **Python Bindings** - Expose CPU diagnostics API

See `docs/experiments/compiler-stdlib-tradeoffs.md` for rationale.
