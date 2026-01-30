<!-- SPDX-License-Identifier: MIT -->
# Kokkos Migration Design

**Date:** 2025-11-25
**Status:** Draft
**Goal:** Full Kokkos port for GPU acceleration (SYCL first), portable parallelism, and Kokkos Views replacing mdspan/PMR.

---

## Overview

Rewrite mango-option core to use Kokkos for:
- **GPU acceleration** — SYCL primary, CUDA/HIP supported
- **Portable parallelism** — Replace OpenMP with Kokkos::parallel_for
- **Unified memory model** — Kokkos::View replaces mdspan + PMR arenas

**Approach:** Bottom-up greenfield rewrite in `kokkos/` folder. Existing code untouched until switchover.

---

## 1. Memory Model

### Execution Space Aliases

```cpp
// kokkos/src/support/execution_space.hpp

#if defined(KOKKOS_ENABLE_SYCL)
using DefaultExecSpace = Kokkos::Experimental::SYCL;
using DefaultMemSpace = Kokkos::Experimental::SYCLSharedUSMSpace;
#elif defined(KOKKOS_ENABLE_CUDA)
using DefaultExecSpace = Kokkos::Cuda;
using DefaultMemSpace = Kokkos::CudaSpace;
#elif defined(KOKKOS_ENABLE_HIP)
using DefaultExecSpace = Kokkos::HIP;
using DefaultMemSpace = Kokkos::HIPSpace;
#else
using DefaultExecSpace = Kokkos::OpenMP;  // or Serial
using DefaultMemSpace = Kokkos::HostSpace;
#endif
```

### View Replacements

| Current | Kokkos Replacement |
|---------|-------------------|
| `std::experimental::mdspan<double, dextents<N>>` | `Kokkos::View<double*..., MemSpace>` |
| `AlignedArena` | Deleted — Kokkos handles alignment |
| `std::pmr::vector<double>` | `Kokkos::View<double*, MemSpace>` |
| `PriceTensor<N>` | `Kokkos::View<double****, MemSpace>` |

### Pipeline-Level Execution Target

User selects CPU or GPU once at pipeline construction:

```cpp
// Internal: templated on MemSpace
template <typename MemSpace>
class PricingPipeline {
    PDEWorkspace<MemSpace> workspace_;
    PriceTableBuilder<MemSpace> builder_;
    IVSolverInterpolated<MemSpace> iv_solver_;

public:
    auto build_price_table(const PriceTableAxes<4>& axes) -> PriceTableResult<4>;
    auto solve_iv_batch(std::span<const IVQuery> queries) -> std::vector<IVResult>;
};

// Public: type-erased handle
class PricingPipelineHandle {
    struct Concept { virtual ~Concept() = default; /* interface */ };

    template <typename MemSpace>
    struct Model : Concept { PricingPipeline<MemSpace> impl; };

    std::unique_ptr<Concept> impl_;

public:
    static PricingPipelineHandle create(ExecutionTarget target);
};
```

Usage:
```cpp
auto pipeline = PricingPipelineHandle::create(ExecutionTarget::SYCL);
auto table = pipeline.build_price_table(axes);  // Runs on GPU
```

---

## 2. Parallel Loop Replacement

### Delete `parallel.hpp`

Replace `MANGO_PRAGMA_*` macros with Kokkos:

| OpenMP Macro | Kokkos Replacement |
|--------------|-------------------|
| `MANGO_PRAGMA_PARALLEL_FOR` | `Kokkos::parallel_for` |
| `MANGO_PRAGMA_SIMD` | `Kokkos::parallel_for` (small ranges) |
| `MANGO_PRAGMA_FOR_COLLAPSE2` | `Kokkos::MDRangePolicy<Rank<2>>` |
| `MANGO_PRAGMA_ATOMIC` | `Kokkos::atomic_add` |

### Example Migration

```cpp
// Before
MANGO_PRAGMA_PARALLEL_FOR
for (size_t i = 0; i < n; ++i) {
    results[i] = solve_option(params[i]);
}

// After
Kokkos::parallel_for("solve_options", n, KOKKOS_LAMBDA(size_t i) {
    results(i) = solve_option(params(i));
});
```

### Nested Parallelism

```cpp
// Before
MANGO_PRAGMA_PARALLEL
MANGO_PRAGMA_FOR_COLLAPSE2
for (size_t i = 0; i < ni; ++i)
    for (size_t j = 0; j < nj; ++j) { ... }

// After
Kokkos::parallel_for("build_table",
    Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {ni, nj}),
    KOKKOS_LAMBDA(size_t i, size_t j) { ... });
```

---

## 3. Tridiagonal Solver: Kokkidio Integration

### Why Kokkidio

[Kokkidio](https://github.com/RL-S/Kokkidio) bridges Eigen and Kokkos:
- `ViewMap` — Kokkos View with Eigen syntax
- `ParallelRange` — Auto-chunking (vectorized blocks on CPU, elements on GPU)
- SYCL support via Kokkos backend
- Header-only

### Batched Thomas Solver

```cpp
#include <kokkidio/kokkidio.hpp>

template <typename MemSpace>
class BatchedTridiagonalSolver {
    Kokkos::View<double**, MemSpace> lower_;
    Kokkos::View<double**, MemSpace> diag_;
    Kokkos::View<double**, MemSpace> upper_;
    Kokkos::View<double**, MemSpace> rhs_;

public:
    void solve_batch() {
        using namespace kokkidio;

        ParallelRange range(rhs_.extent(0));

        parallel_for("thomas_batch", range, KOKKOS_LAMBDA(auto batch_idx) {
            auto d = ViewMap(Kokkos::subview(diag_, batch_idx, Kokkos::ALL));
            auto r = ViewMap(Kokkos::subview(rhs_, batch_idx, Kokkos::ALL));

            // Thomas algorithm with Eigen syntax
            // Forward elimination and back substitution
        });
    }
};
```

---

## 4. B-Spline Evaluation

### GPU-Batched Design

```cpp
template <size_t N, typename MemSpace>
class BSplineSurface {
    Kokkos::View<double*, MemSpace> coeffs_;
    Kokkos::View<double*, MemSpace> knots_[N];
    std::array<int, N> degrees_;
    std::array<size_t, N> dims_;

public:
    // Single query
    KOKKOS_INLINE_FUNCTION
    double operator()(const Kokkos::Array<double, N>& point) const;

    // Batch query
    void evaluate_batch(
        Kokkos::View<const double*[N], MemSpace> points,
        Kokkos::View<double*, MemSpace> results
    ) const {
        auto self = *this;
        Kokkos::parallel_for("bspline_batch", points.extent(0),
            KOKKOS_LAMBDA(size_t i) {
                Kokkos::Array<double, N> pt;
                for (size_t d = 0; d < N; ++d) pt[d] = points(i, d);
                results(i) = self(pt);
            });
    }
};
```

### Delete `target_clones`

CPU SIMD handled by Kokkos/compiler. Remove `[[gnu::target_clones]]` annotations.

---

## 5. Build System

### New Dependencies

| Dependency | Type | Purpose |
|------------|------|---------|
| Kokkos | Library | Core framework |
| Kokkidio | Header-only | Eigen+Kokkos bridge |
| Eigen | Header-only | Linear algebra |

### MODULE.bazel

```python
bazel_dep(name = "kokkos", version = "4.3.0")

http_archive(
    name = "eigen",
    urls = ["https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.tar.gz"],
    strip_prefix = "eigen-3.4.0",
    build_file_content = """
cc_library(
    name = "eigen",
    hdrs = glob(["Eigen/**", "unsupported/**"]),
    includes = ["."],
    visibility = ["//visibility:public"],
)
""",
)

http_archive(
    name = "kokkidio",
    urls = ["https://github.com/RL-S/Kokkidio/archive/refs/heads/main.tar.gz"],
    strip_prefix = "Kokkidio-main",
    build_file_content = """
cc_library(
    name = "kokkidio",
    hdrs = glob(["include/**/*.hpp"]),
    includes = ["include"],
    deps = ["@eigen//:eigen", "@kokkos//:kokkos"],
    visibility = ["//visibility:public"],
)
""",
)
```

### .bazelrc Configurations

```bash
# CPU only (development)
build:openmp --@kokkos//:enable_openmp

# SYCL GPU (Intel/generic)
build:sycl --@kokkos//:enable_sycl

# CUDA GPU (NVIDIA)
build:cuda --@kokkos//:enable_cuda

# HIP GPU (AMD)
build:hip --@kokkos//:enable_hip
```

### Build Commands

```bash
bazel build //kokkos/... --config=openmp   # CPU
bazel build //kokkos/... --config=sycl     # SYCL GPU
bazel test //kokkos/... --config=openmp    # Tests
```

---

## 6. Directory Structure

### During Development

```
mango-option/
├── src/                    # Current code (untouched)
├── tests/                  # Current tests (untouched)
├── benchmarks/             # Current benchmarks (untouched)
├── examples/               # Current examples (untouched)
├── kokkos/                 # New implementation
│   ├── src/
│   │   ├── pde/
│   │   │   ├── core/
│   │   │   └── operators/
│   │   ├── option/
│   │   │   └── table/
│   │   ├── math/
│   │   ├── pipeline/
│   │   └── support/
│   ├── tests/
│   ├── benchmarks/
│   └── examples/
└── docs/
```

### After Switchover

```
mango-option/
├── src/                    # Was kokkos/src/
├── tests/                  # Was kokkos/tests/
├── benchmarks/             # Was kokkos/benchmarks/
├── examples/               # Was kokkos/examples/
├── legacy/                 # Everything old
│   ├── src/
│   ├── tests/
│   ├── benchmarks/
│   └── examples/
└── docs/
```

### Switchover Commands

```bash
mkdir legacy
mv src/ legacy/
mv tests/ legacy/
mv benchmarks/ legacy/
mv examples/ legacy/

mv kokkos/src/ src/
mv kokkos/tests/ tests/
mv kokkos/benchmarks/ benchmarks/
mv kokkos/examples/ examples/
rmdir kokkos/
```

---

## 7. Migration Phases

### Phase 1: Foundation

- Create `kokkos/` directory structure
- Set up Bazel with Kokkos/Kokkidio/Eigen
- Implement `execution_space.hpp`
- Port `Grid`, `PDEWorkspace` with Kokkos::View

### Phase 2: PDE Solver

- Port `PDESolver` templated on MemSpace
- Port spatial operators with `KOKKOS_INLINE_FUNCTION`
- Implement batched Thomas solver with Kokkidio
- Port `AmericanOptionSolver`

### Phase 3: Price Table Pipeline

- Port `PriceTableBuilder` with batched PDE solves
- Port `BSplineSurface` with batch evaluation
- Port `PriceTableSurface`

### Phase 4: IV Solvers

- Port `IVSolverInterpolated` with batched Newton
- Port `IVSolverFDM` with batched Brent
- Port root-finding utilities

### Phase 5: Public API and Switchover

- Implement `PricingPipelineHandle` (type-erased)
- Port all tests, examples, benchmarks
- Validate against legacy implementation
- Execute switchover

---

## 8. Testing Strategy

### Unit Tests

```cpp
TEST(ThomasSolverKokkos, SolvesTridiagonalSystem) {
    auto solver = BatchedTridiagonalSolver<HostSpace>(...);
    solver.solve_batch();
    // Verify results
}
```

### Integration Tests (Compare to Legacy)

```cpp
TEST(PricingPipeline, MatchesLegacyOutput) {
    auto legacy_price = legacy::AmericanOptionSolver(...).solve();
    auto kokkos_price = PricingPipeline<HostSpace>(...).solve();
    EXPECT_NEAR(legacy_price, kokkos_price, 1e-10);
}
```

### Backend-Parameterized Tests

```cpp
template <typename MemSpace>
class PricingPipelineTest : public ::testing::Test {};

using MemSpaces = ::testing::Types<HostSpace, SYCLSharedUSMSpace>;
TYPED_TEST_SUITE(PricingPipelineTest, MemSpaces);

TYPED_TEST(PricingPipelineTest, BuildsPriceTable) {
    PricingPipeline<TypeParam> pipeline(...);
    // Same test, different backend
}
```

---

## 9. Acceptance Criteria

- [ ] All legacy tests pass against new implementation (CPU backend)
- [ ] SYCL backend produces identical results (within FP tolerance)
- [ ] CPU performance ≥ legacy (no regression)
- [ ] GPU performance > 5× speedup on price table build
- [ ] All examples and benchmarks work
- [ ] CI runs OpenMP, SYCL, CUDA (if available)

---

## 10. Risks and Mitigations

| Risk | Mitigation |
|------|------------|
| Numerical drift (CPU vs GPU) | Golden file tests with tolerance; compare to legacy |
| SYCL backend immaturity | Test on Intel DevCloud early; OpenMP fallback |
| Kokkidio edge cases | Start simple; escalate to upstream if issues |
| CPU performance regression | Benchmark against legacy before switchover |
| Build complexity | CI matrix across backends |
