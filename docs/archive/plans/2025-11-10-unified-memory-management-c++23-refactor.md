# Unified Memory Management Module: C++23 Refactor

**Date:** 2025-11-10
**Status:** Design
**Target:** All workspace classes (WorkspaceStorage, NewtonWorkspace, SliceSolverWorkspace)

## Executive Summary

Refactor the existing workspace memory management to leverage C++23 features for improved safety, performance, and maintainability. The design introduces a **layered architecture** with:

1. **Core memory management** using `std::pmr::monotonic_buffer_resource`
2. **Unified workspace abstraction** with full SoA layout and SIMD padding
3. **ISA-aware operators** using `std::experimental::simd` and `[[gnu::target_clones]]`
4. **Top-level CPU feature detection** for diagnostic reporting

Key improvements:
- **Safety:** RAII replaces manual `aligned_alloc`/`free`
- **Performance:** SoA layout enables clean SIMD vectorization without gather/scatter
- **Maintainability:** Separation of concerns (memory, computation, ISA dispatch)
- **Scalability:** Operator-level tiling for cache-friendly execution across CPU tiers

---

## Architecture Overview

### Layer 1: Core Memory Management

**`UnifiedMemoryResource`** - RAII wrapper around `std::pmr::monotonic_buffer_resource`:

```cpp
namespace mango::memory {

class UnifiedMemoryResource {
public:
    explicit UnifiedMemoryResource(size_t initial_buffer_size = 1024 * 1024)
        : upstream_(std::pmr::get_default_resource())
        , monotonic_(initial_buffer_size, upstream_)
        , bytes_allocated_(0)
    {}

    void* allocate(size_t bytes, size_t alignment = 64) {
        void* ptr = monotonic_.allocate(bytes, alignment);
        bytes_allocated_ += bytes;
        return ptr;
    }

    void reset() {
        monotonic_.release();
        bytes_allocated_ = 0;
    }

    size_t bytes_allocated() const { return bytes_allocated_; }

private:
    std::pmr::memory_resource* upstream_;
    std::pmr::monotonic_buffer_resource monotonic_;
    size_t bytes_allocated_;  // Manual tracking
};

} // namespace mango::memory
```

**Design principles:**
- Each workspace owns one instance (no shared state → naturally thread-safe)
- 64-byte default alignment for AVX-512
- `reset()` allows zero-cost reuse between solves
- Manual `bytes_allocated_` tracking (PMR doesn't expose this)

---

### Layer 2: Unified Workspace Abstraction

**Base workspace** - provides allocator and tiling infrastructure:

```cpp
namespace mango {

struct TileMetadata {
    size_t tile_start;
    size_t tile_size;
    size_t padded_size;
    size_t alignment;
};

class WorkspaceBase {
public:
    explicit WorkspaceBase(size_t initial_buffer_size = 1024 * 1024)
        : resource_(initial_buffer_size)
    {}

    static TileMetadata tile_info(size_t n, size_t tile_idx, size_t num_tiles) {
        assert(num_tiles > 0 && "num_tiles must be positive");
        assert(tile_idx < num_tiles && "tile_idx out of bounds");

        const size_t base_tile_size = n / num_tiles;
        const size_t remainder = n % num_tiles;
        const size_t tile_size = base_tile_size + (tile_idx < remainder ? 1 : 0);
        const size_t tile_start = tile_idx * base_tile_size + std::min(tile_idx, remainder);
        const size_t padded_size = pad_to_simd(tile_size);

        return {tile_start, tile_size, padded_size, 64};
    }

    static constexpr size_t SIMD_WIDTH = 8;
    static constexpr size_t pad_to_simd(size_t n) {
        return ((n + SIMD_WIDTH - 1) / SIMD_WIDTH) * SIMD_WIDTH;
    }

    size_t bytes_allocated() const { return resource_.bytes_allocated(); }

protected:
    memory::UnifiedMemoryResource resource_;
};

} // namespace mango
```

**PDE-specific workspace** - solver use case with SoA layout:

```cpp
namespace mango {

/**
 * PDEWorkspace: workspace for PDE solver with SoA layout
 *
 * LIFETIME REQUIREMENTS:
 * - The `grid` span passed to constructor must remain valid for the lifetime
 *   of this workspace (stored for reset() reinit).
 *
 * INVALIDATION WARNING:
 * - reset() invalidates all previously returned std::span objects.
 * - After reset(), caller MUST re-acquire spans via accessors.
 */
class PDEWorkspace : public WorkspaceBase {
public:
    explicit PDEWorkspace(size_t n, std::span<const double> grid,
                         size_t initial_buffer_size = 1024 * 1024)
        : WorkspaceBase(initial_buffer_size)
        , n_(n)
        , padded_n_(pad_to_simd(n))
        , grid_(grid)
    {
        assert(!grid.empty() && "grid must not be empty");
        assert(grid.size() == n && "grid size must match n");
        allocate_and_initialize();
    }

    // SoA array accessors (logical size)
    std::span<double> u_current() { return {u_current_, n_}; }
    std::span<const double> u_current() const { return {u_current_, n_}; }

    std::span<double> u_next() { return {u_next_, n_}; }
    std::span<double> u_stage() { return {u_stage_, n_}; }
    std::span<double> rhs() { return {rhs_, n_}; }
    std::span<double> lu() { return {lu_, n_}; }
    std::span<double> psi_buffer() { return {psi_, n_}; }

    // Padded accessors for kernels that need tail processing
    std::span<double> u_current_padded() { return {u_current_, padded_n_}; }
    std::span<const double> u_current_padded() const { return {u_current_, padded_n_}; }

    std::span<double> u_next_padded() { return {u_next_, padded_n_}; }
    std::span<double> lu_padded() { return {lu_, padded_n_}; }

    // Grid spacing (SIMD-padded, zero-filled tail)
    std::span<const double> dx() const { return {dx_, n_ - 1}; }
    std::span<const double> dx_padded() const { return {dx_, pad_to_simd(n_ - 1)}; }

    TileMetadata tile_info(size_t tile_idx, size_t num_tiles) const {
        return WorkspaceBase::tile_info(n_, tile_idx, num_tiles);
    }

    void reset() {
        resource_.reset();
        allocate_and_initialize();
    }

    size_t logical_size() const { return n_; }
    size_t padded_size() const { return padded_n_; }

private:
    void allocate_and_initialize() {
        allocate_arrays();
        precompute_grid_spacing();
    }

    void allocate_arrays() {
        const size_t array_bytes = padded_n_ * sizeof(double);
        u_current_ = static_cast<double*>(resource_.allocate(array_bytes));
        u_next_    = static_cast<double*>(resource_.allocate(array_bytes));
        u_stage_   = static_cast<double*>(resource_.allocate(array_bytes));
        rhs_       = static_cast<double*>(resource_.allocate(array_bytes));
        lu_        = static_cast<double*>(resource_.allocate(array_bytes));
        psi_       = static_cast<double*>(resource_.allocate(array_bytes));

        // Zero-initialize padding for safe SIMD tail processing
        std::fill(u_current_ + n_, u_current_ + padded_n_, 0.0);
        std::fill(u_next_ + n_, u_next_ + padded_n_, 0.0);
        std::fill(u_stage_ + n_, u_stage_ + padded_n_, 0.0);
        std::fill(rhs_ + n_, rhs_ + padded_n_, 0.0);
        std::fill(lu_ + n_, lu_ + padded_n_, 0.0);
        std::fill(psi_ + n_, psi_ + padded_n_, 0.0);
    }

    void precompute_grid_spacing() {
        const size_t dx_padded = pad_to_simd(n_ - 1);
        const size_t dx_bytes = dx_padded * sizeof(double);
        dx_ = static_cast<double*>(resource_.allocate(dx_bytes));

        for (size_t i = 0; i < n_ - 1; ++i) {
            dx_[i] = grid_[i + 1] - grid_[i];
        }
        // Zero padding for safe SIMD tail
        std::fill(dx_ + (n_ - 1), dx_ + dx_padded, 0.0);
    }

    size_t n_;
    size_t padded_n_;
    std::span<const double> grid_;  // Caller must keep alive!

    // SoA arrays (separate, SIMD-aligned)
    double* u_current_;
    double* u_next_;
    double* u_stage_;
    double* rhs_;
    double* lu_;
    double* psi_;
    double* dx_;
};

} // namespace mango
```

**Key design points:**
- **Full SoA layout:** Each state array separate and SIMD-padded
- **Dual accessors:** Logical size for safety, padded size for SIMD kernels
- **Grid lifetime dependency:** Documented and asserted
- **Reset safety:** Recreates all pointers after PMR release
- **Zero-padded tails:** Safe for vectorized loops without masking

**Future extension:** `InterpolationWorkspace` can inherit from `WorkspaceBase`, use same PMR pattern, but different buffer layout (scattered points, coefficients, etc.).

---

### Layer 3: Operator Kernels with ISA Dispatch

**SIMD-aware stencil operator** using `std::experimental::simd` and `[[gnu::target_clones]]`:

```cpp
#include <experimental/simd>
#include <span>
#include <concepts>
#include <cassert>

namespace mango::operators {

namespace stdx = std::experimental;

/**
 * CenteredDifferenceSIMD: Vectorized stencil operator
 *
 * REQUIREMENTS:
 * - Input spans must be PADDED (use workspace.u_current_padded(), etc.)
 * - start must be ≥ 1 (no boundary point)
 * - end must be ≤ u.size() - 1 (no boundary point)
 * - Boundary conditions handled separately by caller
 */
template<std::floating_point T = double>
class CenteredDifferenceSIMD {
public:
    using simd_t = stdx::native_simd<T>;
    static constexpr size_t simd_width = simd_t::size();

    explicit CenteredDifferenceSIMD(const GridSpacing<T>& spacing,
                                   size_t l1_tile_size = 1024)
        : spacing_(spacing)
        , l1_tile_size_(l1_tile_size)
    {}

    /**
     * Vectorized second derivative kernel (uniform grid)
     *
     * Marked with target_clones for ISA-specific code generation:
     * - default: SSE2 baseline (simd_width = 2 for double)
     * - avx2: Haswell+ (simd_width = 4 for double)
     * - avx512f: Skylake-X+ (simd_width = 8 for double)
     *
     * Verify with: objdump -d <binary> | grep -A20 compute_second_derivative_uniform
     */
    [[gnu::target_clones("default","avx2","avx512f")]]
    void compute_second_derivative_uniform(
        std::span<const T> u,
        std::span<T> d2u_dx2,
        size_t start,
        size_t end) const
    {
        assert(start >= 1 && "start must allow u[i-1] access");
        assert(end <= u.size() - 1 && "end must allow u[i+1] access");

        const T dx2_inv = spacing_.spacing_inv_sq();
        const simd_t dx2_inv_vec(dx2_inv);
        const simd_t minus_two(T(-2));

        // Vectorized main loop
        size_t i = start;
        for (; i + simd_width <= end; i += simd_width) {
            // SoA layout ensures contiguous loads (no gather needed)
            simd_t u_left, u_center, u_right;
            u_left.copy_from(u.data() + i - 1, stdx::element_aligned);
            u_center.copy_from(u.data() + i, stdx::element_aligned);
            u_right.copy_from(u.data() + i + 1, stdx::element_aligned);

            // d2u/dx2 = (u[i+1] + u[i-1] - 2*u[i]) * dx2_inv
            const simd_t sum = u_left + u_right;
            const simd_t result = stdx::fma(sum, dx2_inv_vec,
                                           minus_two * u_center * dx2_inv_vec);

            result.copy_to(d2u_dx2.data() + i, stdx::element_aligned);
        }

        // Scalar tail (zero-padded arrays allow safe i+1 access)
        for (; i < end; ++i) {
            d2u_dx2[i] = std::fma(u[i+1] + u[i-1], dx2_inv, T(-2) * u[i] * dx2_inv);
        }
    }

    /**
     * Tiled second derivative (cache-friendly)
     *
     * Operator decides tile size based on stencil width and cache target.
     * For centered difference (3-point stencil), aim for L1 cache (~32 KB).
     */
    [[gnu::target_clones("default","avx2","avx512f")]]
    void compute_second_derivative_tiled(
        std::span<const T> u,
        std::span<T> d2u_dx2,
        size_t start,
        size_t end) const
    {
        assert(start >= 1 && "start must allow u[i-1] access");
        assert(end <= u.size() - 1 && "end must allow u[i+1] access");

        // Tile size: target L1 cache (configurable per stencil type)
        for (size_t tile_start = start; tile_start < end; tile_start += l1_tile_size_) {
            const size_t tile_end = std::min(tile_start + l1_tile_size_, end);
            compute_second_derivative_uniform(u, d2u_dx2, tile_start, tile_end);
        }
    }

    [[gnu::target_clones("default","avx2","avx512f")]]
    void compute_first_derivative_uniform(
        std::span<const T> u,
        std::span<T> du_dx,
        size_t start,
        size_t end) const
    {
        assert(start >= 1 && "start must allow u[i-1] access");
        assert(end <= u.size() - 1 && "end must allow u[i+1] access");

        const T half_dx_inv = spacing_.spacing_inv() * T(0.5);
        const simd_t half_dx_inv_vec(half_dx_inv);

        size_t i = start;
        for (; i + simd_width <= end; i += simd_width) {
            simd_t u_left, u_right;
            u_left.copy_from(u.data() + i - 1, stdx::element_aligned);
            u_right.copy_from(u.data() + i + 1, stdx::element_aligned);

            const simd_t result = (u_right - u_left) * half_dx_inv_vec;
            result.copy_to(du_dx.data() + i, stdx::element_aligned);
        }

        for (; i < end; ++i) {
            du_dx[i] = (u[i+1] - u[i-1]) * half_dx_inv;
        }
    }

    size_t tile_size() const { return l1_tile_size_; }

private:
    const GridSpacing<T>& spacing_;
    size_t l1_tile_size_;  // Configurable per stencil type
};

} // namespace mango::operators
```

**Key design points:**
- **`[[gnu::target_clones]]`:** Generates 3 ISA-specific versions (default/avx2/avx512f)
- **`std::experimental::simd`:** Portable SIMD, compiler adapts to ISA
- **SoA-friendly:** Contiguous `copy_from()`/`copy_to()`, no gather/scatter
- **Operator-level tiling:** Configurable tile size, targets L1 cache
- **Clean tail handling:** Zero-padded arrays allow safe overflow access

---

### Layer 4: Top-Level ISA Selection

**CPU feature detection** with OS support validation:

```cpp
#include <cpuid.h>
#include <string>
#include <iostream>
#include <immintrin.h>

namespace mango::cpu {

struct CPUFeatures {
    bool has_sse2 = false;
    bool has_avx2 = false;
    bool has_avx512f = false;
    bool has_fma = false;
};

enum class ISATarget {
    DEFAULT,   // SSE2 baseline
    AVX2,      // Haswell+ (2013+)
    AVX512F    // Skylake-X+ (2017+)
};

inline bool check_os_avx_support() {
    unsigned int eax, ebx, ecx, edx;
    if (!__get_cpuid(1, &eax, &ebx, &ecx, &edx)) return false;
    if ((ecx & bit_OSXSAVE) == 0) return false;

    unsigned long long xcr0 = _xgetbv(0);
    constexpr unsigned long long AVX_MASK = (1ULL << 1) | (1ULL << 2);
    return (xcr0 & AVX_MASK) == AVX_MASK;
}

inline bool check_os_avx512_support() {
    unsigned long long xcr0 = _xgetbv(0);
    constexpr unsigned long long AVX512_MASK = (1ULL << 1) | (1ULL << 2) |
                                               (1ULL << 5) | (1ULL << 6) | (1ULL << 7);
    return (xcr0 & AVX512_MASK) == AVX512_MASK;
}

inline CPUFeatures detect_cpu_features() {
    CPUFeatures features;
    unsigned int eax, ebx, ecx, edx;

    if (__get_cpuid(1, &eax, &ebx, &ecx, &edx)) {
        features.has_sse2 = (edx & bit_SSE2) != 0;
        features.has_fma = (ecx & bit_FMA) != 0;
    }

    bool os_avx_support = check_os_avx_support();
    bool os_avx512_support = os_avx_support && check_os_avx512_support();

    if (os_avx_support && __get_cpuid_count(7, 0, &eax, &ebx, &ecx, &edx)) {
        features.has_avx2 = (ebx & bit_AVX2) != 0;

        if (features.has_avx2 && !features.has_fma) {
            #ifndef NDEBUG
            std::cerr << "Warning: AVX2 detected but FMA not available\n";
            #endif
        }
    }

    if (os_avx512_support && __get_cpuid_count(7, 0, &eax, &ebx, &ecx, &edx)) {
        features.has_avx512f = (ebx & bit_AVX512F) != 0;
    }

    return features;
}

/**
 * Select best ISA target for current CPU
 *
 * NOTE: This is DIAGNOSTIC ONLY. The actual kernel dispatch happens
 * automatically via [[gnu::target_clones]] IFUNC resolution.
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

inline std::string isa_target_name(ISATarget target) {
    switch (target) {
        case ISATarget::DEFAULT: return "SSE2";
        case ISATarget::AVX2: return "AVX2+FMA";
        case ISATarget::AVX512F: return "AVX512F";
        default: return "UNKNOWN";
    }
}

} // namespace mango::cpu
```

**Integration with PDESolver:**

```cpp
template<std::floating_point T = double>
class PDESolver {
public:
    PDESolver(/* ... */)
        : workspace_(n, grid)
        , isa_target_(cpu::select_isa_target())  // DIAGNOSTIC ONLY
        , stencil_(spacing, /*l1_tile_size=*/1024)
    {
        #ifndef NDEBUG
        std::cout << "PDESolver ISA target: "
                  << cpu::isa_target_name(isa_target_) << "\n";
        #endif
    }

    void solve() {
        for (size_t step = 0; step < n_steps_; ++step) {
            // CRITICAL: Use padded spans for SIMD kernels
            auto u_padded = workspace_.u_current_padded();
            auto lu_padded = workspace_.lu_padded();

            // IFUNC resolver picks best ISA variant automatically
            stencil_.compute_second_derivative_tiled(
                u_padded, lu_padded,
                /*start=*/1, /*end=*/workspace_.logical_size() - 1
            );

            apply_boundary_conditions();  // Separate, no SIMD
            // ... rest of time-stepping ...
        }
    }

private:
    PDEWorkspace workspace_;
    cpu::ISATarget isa_target_;  // Diagnostic only
    operators::CenteredDifferenceSIMD<T> stencil_;
};
```

**Key design points:**
- **OS support guard:** XGETBV validates YMM/ZMM state enabled (prevents SIGILL)
- **Static cache:** Features detected once, cached in static variable
- **IFUNC dispatch:** Compiler/linker resolve best ISA at runtime, not our code
- **Diagnostic only:** `select_isa_target()` reports what IFUNC will choose

---

## Implementation Plan

### Phase 1: Core Memory Management (Week 1)
1. Implement `UnifiedMemoryResource` with PMR
2. Write unit tests for allocation/reset/stats
3. Verify alignment with `std::align` checks

### Phase 2: Workspace Refactor (Week 2)
1. Implement `WorkspaceBase` with tiling helpers
2. Implement `PDEWorkspace` with SoA layout
3. Replace `WorkspaceStorage` in existing code
4. Update tests to use padded accessors

### Phase 3: SIMD Operators (Week 3)
1. Implement `CenteredDifferenceSIMD` with `target_clones`
2. Verify ISA clones with `objdump`
3. Benchmark vs existing scalar operators
4. Update `PDESolver` to use SIMD operators

### Phase 4: ISA Detection (Week 4)
1. Implement CPU feature detection with XGETBV
2. Add diagnostic logging
3. Integration testing on different CPUs
4. Performance benchmarks (AVX2 vs AVX-512)

### Phase 5: Full Integration (Week 5)
1. Replace `NewtonWorkspace` with `PDEWorkspace` + borrowing
2. Replace `SliceSolverWorkspace` wrapper
3. Update all examples and tests
4. Final benchmarks and validation

---

## Verification Strategy

### Correctness Verification
- Unit tests for each layer (memory, workspace, operators)
- Compare SIMD results vs scalar (tolerance: 1e-14 for double)
- Verify boundary condition handling unchanged
- Run entire test suite (must pass 100%)

### Performance Verification
```bash
# Verify IFUNC clones generated
objdump -d bazel-bin/src/libpde_solver.so | grep -A5 "compute_second_derivative"

# Verify SIMD register usage
objdump -d bazel-bin/src/libpde_solver.so | grep -E "(xmm|ymm|zmm)"

# Runtime ISA selection
./bazel-bin/examples/example_heat_equation
# Expected output: "PDESolver ISA target: AVX2+FMA"

# Performance benchmarks
bazel run //tests:pde_solver_benchmark
# Target: 20-30% speedup on AVX2, 40-50% on AVX-512
```

### Safety Verification
- Run with AddressSanitizer (`-fsanitize=address`)
- Run with UndefinedBehaviorSanitizer (`-fsanitize=undefined`)
- Valgrind memcheck for leaks
- Static analysis with clang-tidy

---

## Migration Path for Existing Code

### Minimal Changes Required

**Before:**
```cpp
WorkspaceStorage workspace(n, grid);
auto u = workspace.u_current();
```

**After:**
```cpp
PDEWorkspace workspace(n, grid);
auto u = workspace.u_current();  // Same API!
```

**For SIMD kernels:**
```cpp
// Old: scalar FMA
for (size_t i = 1; i < n-1; ++i) {
    Lu[i] = std::fma(u[i+1] + u[i-1], dx2_inv, -2*u[i]*dx2_inv);
}

// New: SIMD operator (same semantics)
auto u_padded = workspace.u_current_padded();
auto lu_padded = workspace.lu_padded();
stencil.compute_second_derivative_tiled(u_padded, lu_padded, 1, n-1);
```

---

## Expected Performance Improvements

| Metric | Baseline (current) | After refactor | Improvement |
|--------|-------------------|----------------|-------------|
| Memory allocation overhead | ~5-10% per solve | ~0.1% (PMR reset) | 50-100x faster |
| SIMD throughput (AVX2) | 1x (scalar) | 3-4x | 3-4x |
| SIMD throughput (AVX-512) | 1x (scalar) | 6-8x | 6-8x |
| Cache efficiency | Variable | Consistent (tiling) | 10-20% faster |
| Binary size | 1x | 1.3x (3 ISA clones) | Acceptable |

---

## Risks and Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| `std::experimental::simd` not widely supported | High | Provide scalar fallback, test on GCC 11+/Clang 14+ |
| OS doesn't support AVX-512 | Medium | XGETBV guard prevents SIGILL, falls back to AVX2 |
| Cache tiling heuristic suboptimal | Low | Configurable tile size, future auto-tuning |
| Binary bloat from `target_clones` | Low | Only applied to hot kernels (~5-10 functions) |

---

## Future Extensions

1. **Interpolation workspace:** Separate workspace for B-spline/cubic spline (different layout)
2. **Auto-tuning:** Runtime cache size detection, adaptive tile selection
3. **GPU backend:** SYCL-based workspace (same API, different allocator)
4. **Multi-level tiling:** L1/L2/L3 blocking for larger grids
5. **Non-uniform grid SIMD:** Vectorize non-uniform stencils (harder, lower priority)

---

## References

- [C++23 Standard](https://en.cppreference.com/w/cpp/23)
- [std::pmr Documentation](https://en.cppreference.com/w/cpp/memory/memory_resource)
- [std::experimental::simd](https://en.cppreference.com/w/cpp/experimental/simd/simd)
- [GCC target_clones](https://gcc.gnu.org/onlinedocs/gcc/Common-Function-Attributes.html#index-target_005fclones-function-attribute)
- [Intel Intrinsics Guide](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html)
