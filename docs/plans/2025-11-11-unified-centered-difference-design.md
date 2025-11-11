# Unified CenteredDifference Design

**Date:** 2025-11-11
**Status:** Approved
**Author:** Claude Code (with user guidance)

## Overview

Unify `CenteredDifference` (scalar) and `CenteredDifferenceSIMD` (explicit SIMD) behind a single public API with automatic runtime ISA selection. This eliminates the need for callers to choose between implementations while preserving testability and existing optimizations.

## Motivation

### Current State

Two separate implementations:
- **CenteredDifference**: Scalar with `#pragma omp simd` compiler hints
- **CenteredDifferenceSIMD**: Explicit `std::experimental::simd` with precomputed arrays and `[[gnu::target_clones]]`

**Problems:**
- Callers must explicitly choose which to use
- No automatic runtime ISA selection
- API duplication (need to know which class to instantiate)
- Tests require switching between two different types

### Goals

1. **Single Public API**: One `CenteredDifference` class for all callers
2. **Automatic ISA Selection**: Runtime detection picks optimal backend (Mode::Auto)
3. **Testability**: Explicit Mode::Scalar / Mode::Simd for regression tests
4. **Zero API Changes**: Existing code continues to work
5. **Preserve Optimizations**: Keep existing tuned scalar and SIMD loops

## Design

### Architecture: Façade + Backend Pattern

```
┌─────────────────────────────────────┐
│   CenteredDifference (Façade)       │
│   - Mode enum (Auto/Scalar/Simd)    │
│   - CPU feature detection            │
│   - Virtual dispatch to backend      │
└──────────────┬──────────────────────┘
               │
       ┌───────┴────────┐
       ▼                ▼
┌─────────────┐  ┌──────────────┐
│ScalarBackend│  │ SimdBackend  │
│(#pragma omp)│  │([[target_    │
│             │  │  clones]])   │
└─────────────┘  └──────────────┘
```

### File Structure

```
src/pde/operators/
├── centered_difference.hpp              # Public façade
├── centered_difference_scalar.hpp       # ScalarBackend (internal)
├── centered_difference_simd.hpp         # SimdBackend (internal)
└── grid_spacing.hpp                     # Unchanged
```

### Core Implementation

#### Public Façade (centered_difference.hpp)

```cpp
namespace mango::operators {

class CenteredDifference {
public:
    enum class Mode { Auto, Scalar, Simd };

    explicit CenteredDifference(const GridSpacing<double>& spacing,
                                Mode mode = Mode::Auto);

    // Public API - keeps [[gnu::target_clones]] for consistent symbols
    [[gnu::target_clones("default","avx2","avx512f")]]
    void compute_second_derivative(std::span<const double> u,
                                   std::span<double> d2u_dx2,
                                   size_t start, size_t end) const {
        assert(start >= 1 && "start must allow u[i-1] access");
        assert(end <= u.size() - 1 && "end must allow u[i+1] access");
        impl_->compute_second_derivative(u, d2u_dx2, start, end);
    }

    [[gnu::target_clones("default","avx2","avx512f")]]
    void compute_first_derivative(std::span<const double> u,
                                  std::span<double> du_dx,
                                  size_t start, size_t end) const;

    [[gnu::target_clones("default","avx2","avx512f")]]
    void compute_second_derivative_tiled(std::span<const double> u,
                                         std::span<double> d2u_dx2,
                                         size_t start, size_t end) const;

private:
    struct BackendInterface {
        virtual ~BackendInterface() = default;
        virtual void compute_second_derivative(
            std::span<const double> u, std::span<double> d2u_dx2,
            size_t start, size_t end) const = 0;
        virtual void compute_first_derivative(
            std::span<const double> u, std::span<double> du_dx,
            size_t start, size_t end) const = 0;
        virtual void compute_second_derivative_tiled(
            std::span<const double> u, std::span<double> d2u_dx2,
            size_t start, size_t end) const = 0;
    };

    template<typename Backend>
    struct BackendImpl final : BackendInterface {
        Backend backend_;  // Stored by value - keeps GridSpacing reference valid

        explicit BackendImpl(const GridSpacing<double>& spacing)
            : backend_(spacing) {}

        void compute_second_derivative(
            std::span<const double> u, std::span<double> d2u_dx2,
            size_t start, size_t end) const override {
            backend_.compute_second_derivative(u, d2u_dx2, start, end);
        }

        void compute_first_derivative(
            std::span<const double> u, std::span<double> du_dx,
            size_t start, size_t end) const override {
            backend_.compute_first_derivative(u, du_dx, start, end);
        }

        void compute_second_derivative_tiled(
            std::span<const double> u, std::span<double> d2u_dx2,
            size_t start, size_t end) const override {
            backend_.compute_second_derivative_tiled(u, d2u_dx2, start, end);
        }
    };

    std::unique_ptr<BackendInterface> impl_;
};

} // namespace mango::operators
```

#### Mode::Auto Selection Logic

```cpp
CenteredDifference::CenteredDifference(const GridSpacing<double>& spacing,
                                       Mode mode)
{
    if (mode == Mode::Auto) {
        // Check CPU features AND OS support
        auto features = cpu::detect_features();
        bool os_supports_avx = cpu::check_os_avx_support();

        // Use SIMD if both CPU and OS support it
        if ((features.has_avx2 || features.has_avx512f) && os_supports_avx) {
            mode = Mode::Simd;
        } else {
            mode = Mode::Scalar;
        }
    }

    switch (mode) {
        case Mode::Scalar:
            impl_ = std::make_unique<BackendImpl<ScalarBackend>>(spacing);
            break;
        case Mode::Simd:
            impl_ = std::make_unique<BackendImpl<SimdBackend>>(spacing);
            break;
        case Mode::Auto:
            // Already resolved above
            break;
    }
}
```

**Key guarantees:**
- OS XSAVE check prevents SIGILL on AVX-capable but XSAVE-disabled systems
- Backend stored by value → GridSpacing reference stays valid
- [[gnu::target_clones]] on wrapper methods → consistent symbol names
- Virtual dispatch overhead: ~5-10ns per call (negligible vs computation cost)

### Backend Implementations

#### ScalarBackend (centered_difference_scalar.hpp)

Extracted from current `CenteredDifference`, with key changes:

```cpp
template<std::floating_point T = double>
class ScalarBackend {
public:
    explicit ScalarBackend(const GridSpacing<T>& spacing)
        : spacing_(spacing) {}

    // Uniform grid methods (existing logic)
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

    // Non-uniform grid methods - NOW USES PRECOMPUTED ARRAYS
    void compute_second_derivative_non_uniform(
        std::span<const T> u, std::span<T> d2u_dx2,
        size_t start, size_t end) const
    {
        // Use precomputed arrays (matches SIMD strategy)
        auto dx_left_inv = spacing_.dx_left_inv();
        auto dx_right_inv = spacing_.dx_right_inv();
        auto dx_center_inv = spacing_.dx_center_inv();

        MANGO_PRAGMA_SIMD
        for (size_t i = start; i < end; ++i) {
            const T dxl_inv = dx_left_inv[i - 1];
            const T dxr_inv = dx_right_inv[i - 1];
            const T dxc_inv = dx_center_inv[i - 1];

            const T forward_diff = (u[i+1] - u[i]) * dxr_inv;
            const T backward_diff = (u[i] - u[i-1]) * dxl_inv;
            d2u_dx2[i] = (forward_diff - backward_diff) * dxc_inv;
        }
    }

    // Auto-dispatch wrapper
    void compute_second_derivative(
        std::span<const T> u, std::span<T> d2u_dx2,
        size_t start, size_t end) const
    {
        if (spacing_.is_uniform()) {
            compute_second_derivative_uniform(u, d2u_dx2, start, end);
        } else {
            compute_second_derivative_non_uniform(u, d2u_dx2, start, end);
        }
    }

    // ... similar for first derivative, tiled variants

private:
    const GridSpacing<T>& spacing_;
};
```

**Key change:** Non-uniform methods now use `spacing_.dx_left_inv()` etc. precomputed arrays instead of computing divisions on-the-fly. This:
- Matches SIMD strategy (both load from same coefficients)
- Improves scalar performance (O(n multiplications) vs O(n divisions))
- Ensures exact numerical match between scalar and SIMD tails

#### SimdBackend (centered_difference_simd.hpp)

Simply rename current `CenteredDifferenceSIMD` → `SimdBackend`:

```cpp
template<std::floating_point T = double>
class SimdBackend {
public:
    using simd_t = std::experimental::native_simd<T>;
    static constexpr size_t simd_width = simd_t::size();

    explicit SimdBackend(const GridSpacing<T>& spacing,
                        size_t l1_tile_size = 1024)
        : spacing_(spacing)
        , l1_tile_size_(l1_tile_size) {}

    // Keep ALL existing methods from CenteredDifferenceSIMD unchanged:
    // - compute_second_derivative_uniform (with [[gnu::target_clones]])
    // - compute_first_derivative_uniform
    // - compute_second_derivative_non_uniform
    // - compute_first_derivative_non_uniform
    // - compute_second_derivative (auto-dispatch wrapper)
    // - compute_first_derivative (auto-dispatch wrapper)
    // - compute_second_derivative_tiled

    // Just rename the class, all implementation identical

private:
    const GridSpacing<T>& spacing_;
    size_t l1_tile_size_;
};
```

**No changes** to implementation - pure rename.

## Migration Strategy

### Step 1: Create Backend Files (Non-Breaking)

- Extract current `CenteredDifference` → `centered_difference_scalar.hpp` as `ScalarBackend`
- Rename current `CenteredDifferenceSIMD` → `centered_difference_simd.hpp` as `SimdBackend`
- Update ScalarBackend non-uniform methods to use precomputed arrays
- Keep old files temporarily for compatibility

### Step 2: Create Façade (Non-Breaking)

- Implement new `CenteredDifference` façade in `centered_difference.hpp`
- Add Mode enum and constructor with Auto/Scalar/Simd selection
- Forward all methods through BackendInterface virtual calls
- All existing code continues using old headers

### Step 3: Update Call Sites

- Update `#include` statements to use new façade
- Remove explicit `CenteredDifference` vs `CenteredDifferenceSIMD` distinctions
- Everything uses `CenteredDifference(spacing)` with Mode::Auto

### Step 4: Cleanup (Breaking)

- Delete old `centered_difference.hpp` (scalar-only version)
- Delete old `centered_difference_simd.hpp`
- Backends are now internal implementation details

## Testing Strategy

### Regression Tests (Explicit Backend Selection)

```cpp
TEST(CenteredDifferenceTest, ScalarVsSimdMatch) {
    auto spacing = /* tanh-clustered grid */;

    // Force scalar backend
    auto scalar_stencil = CenteredDifference(spacing,
                                             CenteredDifference::Mode::Scalar);

    // Force SIMD backend
    auto simd_stencil = CenteredDifference(spacing,
                                          CenteredDifference::Mode::Simd);

    // Compare results with EXPECT_NEAR(1e-14) for FP rounding
    std::vector<double> u = /* test data */;
    std::vector<double> d2u_scalar(u.size()), d2u_simd(u.size());

    scalar_stencil.compute_second_derivative(u, d2u_scalar, 1, u.size()-1);
    simd_stencil.compute_second_derivative(u, d2u_simd, 1, u.size()-1);

    for (size_t i = 1; i < u.size()-1; ++i) {
        EXPECT_NEAR(d2u_scalar[i], d2u_simd[i], 1e-14);
    }
}
```

### Production Tests (Auto Mode)

```cpp
TEST(CenteredDifferenceTest, AutoModeSelectsCorrectly) {
    auto spacing = /* ... */;
    auto stencil = CenteredDifference(spacing);  // Mode::Auto

    // Just verify it works, don't care which backend
    std::vector<double> u = /* test data */;
    std::vector<double> d2u(u.size());

    stencil.compute_second_derivative(u, d2u, 1, u.size()-1);

    // Verify correctness against analytical solution
    for (size_t i = 1; i < u.size()-1; ++i) {
        EXPECT_NEAR(d2u[i], analytical_d2u[i], 1e-12);
    }
}
```

### Performance Verification

- Benchmark scalar vs SIMD backends separately
- Verify Mode::Auto overhead is negligible (<1% on hot paths)
- Confirm [[gnu::target_clones]] still generates multi-ISA code

## Performance Analysis

### Virtual Dispatch Overhead

**Per-call cost:** ~5-10ns (vtable lookup + indirect jump)

**Hot loop cost analysis (100-point grid, second derivative):**
- Virtual dispatch: ~10ns
- Computation: ~5,000ns (vectorized) to ~20,000ns (scalar)
- Overhead: 0.05% - 0.2% (negligible)

**Why acceptable:**
- Solver calls `compute_second_derivative()` once per time step
- Inside the call, vectorized loop processes 100 points in a tight loop
- Virtual dispatch amortized over hundreds of operations
- Alternative (compile-time dispatch) would require templating `SpatialOperator` and `PDESolver`, spreading complexity throughout codebase

### Backend Performance (Unchanged)

- **Uniform grids:** Both backends use constant spacing (no precomputation overhead)
- **Non-uniform grids:**
  - Scalar: O(n multiplications) with precomputed arrays (improved from O(n divisions))
  - SIMD: 3-6x speedup via vectorization + precomputed arrays

## Compatibility

### Existing Code

No changes needed in:
- `SpatialOperator` (still holds `CenteredDifference` by value)
- `PDESolver` (still templated on `SpatialOperator`)
- Any production code constructing stencils

### Tests

- Existing regression tests continue to work (explicitly construct backends)
- New tests use Mode::Auto for production-like testing
- All tests maintain current coverage

## Documentation Updates

### CLAUDE.md

Add section:

```markdown
### CenteredDifference: Automatic ISA Selection

The `CenteredDifference` stencil operator automatically selects the optimal backend:
- **Mode::Auto** (default): Runtime CPU detection chooses Scalar or SIMD
- **Mode::Scalar**: Force scalar backend (tests only)
- **Mode::Simd**: Force SIMD backend (tests only)

Production code should always use the default Mode::Auto:
```cpp
auto stencil = CenteredDifference(spacing);  // Auto-selects optimal backend
```

Tests can force specific backends for regression testing:
```cpp
auto scalar = CenteredDifference(spacing, CenteredDifference::Mode::Scalar);
auto simd = CenteredDifference(spacing, CenteredDifference::Mode::Simd);
```

**Performance:** Virtual dispatch overhead is ~5-10ns per call, negligible compared to computation cost.
```

## Implementation Checklist

- [ ] Create `centered_difference_scalar.hpp` (extract from current `CenteredDifference`)
- [ ] Update ScalarBackend to use precomputed arrays for non-uniform grids
- [ ] Rename `centered_difference_simd.hpp` to `SimdBackend`
- [ ] Implement `CenteredDifference` façade with Mode enum
- [ ] Add Mode::Auto constructor with CPU detection + XSAVE check
- [ ] Update tests to use Mode::Scalar / Mode::Simd explicitly
- [ ] Verify all 40 tests pass
- [ ] Benchmark virtual dispatch overhead
- [ ] Update call sites to use new façade
- [ ] Delete old header files
- [ ] Update CLAUDE.md documentation

## Risks and Mitigations

### Risk: Virtual Dispatch Overhead

**Mitigation:** Analysis shows 0.05%-0.2% overhead, negligible in practice.

### Risk: SIGILL on AVX-Capable but XSAVE-Disabled Systems

**Mitigation:** Mode::Auto checks both CPU features AND OS XSAVE support before selecting SIMD.

### Risk: Breaking Existing Code

**Mitigation:** Migration happens in stages, old headers kept until all call sites updated.

### Risk: Loss of Optimization

**Mitigation:** Backends are pure renames/extracts, all [[gnu::target_clones]] and SIMD code preserved unchanged.

## Future Work

- Consider non-virtual dispatch via compile-time policy (if overhead becomes measurable)
- Extend to other operators (upwind, higher-order stencils)
- Add AVX-512 specific optimizations (16-wide doubles)

## References

- Current implementations: `src/pde/operators/centered_difference.hpp`, `src/pde/operators/centered_difference_simd.hpp`
- CPU detection: `src/support/cpu/feature_detection.hpp`
- Usage: `src/pde/operators/spatial_operator.hpp`
