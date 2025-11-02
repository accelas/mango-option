# Phase 2: TR-BDF2 Kernel (Black-Scholes)

**Date**: 2025-11-01
**Status**: Design
**Parent**: [Rust+GPU Rewrite](2025-11-01-rust-gpu-rewrite.md)
**Timeline**: Week 2-5 (4 weeks)

## Summary

Port TR-BDF2 time-stepping solver from C to Rust with Black-Scholes operator hardcoded. Implementation is `#[no_std]` compatible and generic over f32/f64, preparing for GPU compilation in Phase 3. Focus is CPU validation: prove Rust numerics match C (<1e-6 relative error) before tackling SPIR-V compilation.

## Goals

1. **Numerical accuracy**: Match C implementation (<1e-6 relative error on 100 random configs)
2. **GPU-ready code**: `#[no_std]`, zero heap allocation, generic float type
3. **Black-Scholes only**: Hardcoded operator for simplicity (extensibility deferred)
4. **De-risk rust-gpu**: Validate Rust implementation on CPU before GPU toolchain complexity

## Non-Goals (Deferred)

- SPIR-V compilation → Phase 3
- GPU execution → Phase 3
- Multiple PDE support → Future (if needed)
- Performance optimization → Phase 7

## Architecture

### Crate Structure

```
crates/kernel/
├── Cargo.toml          # #[no_std] compatible, edition 2024
├── build.rs            # C FFI bindings for validation
├── src/
│   ├── lib.rs          # Public API, #[no_std] config
│   ├── float.rs        # GpuFloat trait (f32/f64 abstraction)
│   ├── black_scholes.rs  # Hardcoded BS operator
│   ├── tridiagonal.rs  # Thomas algorithm
│   └── trbdf2.rs       # Two-stage time stepping
└── tests/
    └── cpu_tests.rs    # Validation against C
```

### Key Design Decisions

**Decision 1: Monolithic approach**
- Hardcode Black-Scholes operator directly into kernel
- No trait-based PDE abstraction (traits don't work on GPU)
- Rationale: Simplicity, best GPU performance, matches "Black-Scholes only" requirement

**Decision 2: Generic float type**
- Single implementation for f32 and f64 using `GpuFloat` trait
- Enables performance/precision trade-off benchmarking
- Zero code duplication

**Decision 3: CPU validation first**
- Phase 2 = CPU-only Rust implementation
- Validate numerics before GPU toolchain complexity
- FFI bridge to C for direct comparison
- SPIR-V compilation deferred to Phase 3

## Components

### 1. GpuFloat Trait

```rust
// crates/kernel/src/float.rs
pub trait GpuFloat: Copy + Sized {
    const ZERO: Self;
    const ONE: Self;
    const HALF: Self;
    const TWO: Self;

    fn add(self, rhs: Self) -> Self;
    fn sub(self, rhs: Self) -> Self;
    fn mul(self, rhs: Self) -> Self;
    fn div(self, rhs: Self) -> Self;
    fn fma(self, a: Self, b: Self) -> Self;  // self * a + b
    fn sqrt(self) -> Self;
    fn exp(self) -> Self;
    fn max(self, other: Self) -> Self;

    fn from_f64(v: f64) -> Self;
    fn to_f64(self) -> f64;
}
```

**Rationale**: Abstracts float operations for CPU (`f32::exp()`) and future GPU (SPIR-V intrinsics). No standard library dependency.

### 2. KernelParams Type

```rust
#[repr(C)]
#[derive(Copy, Clone)]
pub struct KernelParams<F: GpuFloat> {
    // Option parameters
    pub strike: F,
    pub volatility: F,
    pub risk_free_rate: F,
    pub time_to_maturity: F,
    pub option_type: OptionType,  // From types crate

    // Grid parameters (log-price space)
    pub x_min: F,
    pub x_max: F,
    pub dx: F,
    pub n_points: usize,

    // Time parameters
    pub dt: F,
    pub gamma: F,  // TR-BDF2 parameter ≈ 0.5858
}
```

**Separation from `types::OptionParams`**:
- `types::OptionParams`: Host-side configuration (spot_price, dividend_yield)
- `KernelParams`: Kernel-ready (pre-computed x_min/x_max, no callbacks)

### 3. Black-Scholes Operator

**Spatial operator**: L(V) = (σ²/2)∂²V/∂x² + (r - σ²/2)∂V/∂x - rV

```rust
pub fn spatial_operator<F: GpuFloat>(
    x: &[F],           // Log-price grid
    V: &[F],           // Option values
    LV: &mut [F],      // Output
    params: &KernelParams<F>,
)
```

**Implementation details**:
- Non-uniform grid finite differences (second-order accurate)
- Matches C implementation exactly (lines 91-142 in `src/american_option.c`)
- Hardcoded coefficients: coeff_2nd = σ²/2, coeff_1st = r - σ²/2, coeff_0th = -r

**Boundary conditions**:
- **Left** (S→0): Call = 0, Put = K·exp(-r·τ)
- **Right** (S→∞): Call = S_max - K·exp(-r·τ), Put = 0

**Obstacle condition**: V ≥ max(S - K, 0) for call, V ≥ max(K - S, 0) for put

### 4. Tridiagonal Solver (Thomas Algorithm)

```rust
pub fn solve_tridiagonal<F: GpuFloat>(
    n: usize,
    lower: &[F],      // n-1 elements
    diag: &[F],       // n elements
    upper: &[F],      // n-1 elements
    rhs: &[F],        // n elements
    solution: &mut [F],
    workspace: &mut [F],  // 2n elements: [c_prime, d_prime]
)
```

**Algorithm**:
1. Forward sweep: Compute c_prime and d_prime
2. Back substitution: Solve for solution

**Workspace layout**: `[c_prime (n), d_prime (n)]` = 2n total

### 5. TR-BDF2 Time Stepping

```rust
pub fn trbdf2_step<F: GpuFloat>(
    u_current: &[F],
    u_stage: &mut [F],
    u_next: &mut [F],
    workspace: &mut [F],  // 6n: Lu, rhs, tridiag(2n), matrix(3n)
    x_grid: &[F],
    tau: F,
    params: &KernelParams<F>,
) -> Result<(), ConvergenceError>
```

**Two-stage scheme**:

**Stage 1 (Trapezoidal rule, γ·dt)**:
- Compute L(u^n)
- RHS = u^n + (γ·dt/2)·L(u^n)
- Solve (I - γ·dt/2·L)·u* = RHS (Newton iteration)
- Apply boundaries + obstacle

**Stage 2 (BDF2, full dt)**:
- Coefficients: coeff = (1-γ)·dt / (2-γ)
- RHS = [u* / (γ(2-γ))] - [(1-γ)² / (γ(2-γ))]·u^n
- Solve (I - coeff·L)·u^{n+1} = RHS
- Apply boundaries + obstacle

**Newton iteration**:
- Max 20 iterations
- Tolerance: 1e-8 relative error
- Build Jacobian using finite differences
- Solve linearized system with tridiagonal solver

**Workspace management** (critical for GPU):
```
Total: 9n elements
- Lu_temp: n (spatial operator output)
- rhs: n (right-hand side)
- tridiag_ws: 2n (Thomas algorithm workspace)
- matrix: 3n (diag, upper, lower)
```

### 6. Memory Management

**Zero heap allocation**:
- All arrays passed as slices
- Workspace pre-allocated by caller
- No Vec, Box, or dynamic allocation
- Validated by miri in tests

**Workspace slicing**:
```rust
let (Lu_temp, rest) = workspace.split_at_mut(n);
let (rhs, rest) = rest.split_at_mut(n);
let (tridiag_ws, matrix_ws) = rest.split_at_mut(2 * n);
// ... continue slicing
```

This pattern avoids allocation while maintaining clean code structure.

## Testing Strategy

### Tier 1: Unit Tests

```rust
#[test]
fn test_tridiagonal_solver() {
    // Known system with analytical solution
    // Verify A*x = b within floating-point precision
}

#[test]
fn test_black_scholes_operator() {
    // Test finite difference formulas
    // Compare against analytical derivatives (uniform grid)
}

#[test]
fn test_boundary_conditions() {
    // Verify call/put boundaries match analytical formulas
}
```

### Tier 2: Integration Tests

```rust
#[test]
fn test_trbdf2_single_step() {
    // Full time step with all components
    // Verify convergence, solution bounds, monotonicity
}

#[test]
fn test_american_option_convergence() {
    // Run to maturity
    // Verify payoff at t=0 matches terminal condition
}
```

### Tier 3: Validation Against C

```rust
#[test]
fn test_matches_c_american_put() {
    // FFI call to C implementation
    let c_price = unsafe { american_option_price(&c_params) };

    // Rust implementation
    let rust_price = price_american_option_rust(&rust_params);

    // Target: <1e-6 relative error
    assert_relative_eq!(rust_price, c_price, epsilon = 1e-6);
}

#[test]
fn test_rust_vs_c_random_options() {
    // 100 random configurations
    // Strike: 80-120, Vol: 0.1-0.5, Rate: 0.0-0.1, Maturity: 0.1-2.0
    // Verify: max_error < 1e-6, mean_error < 1e-7
}
```

### Test Dependencies

```toml
[dev-dependencies]
rand = "0.8"          # Random test generation
approx = "0.5"        # assert_relative_eq! macro

[build-dependencies]
bindgen = "0.69"      # C FFI bindings
```

**Build script** (`build.rs`):
- Generate FFI bindings to `src/american_option.h`
- Link against Bazel-built C libraries for validation
- Only used in test builds (not production)

## Build System

### Cargo Configuration

```toml
[package]
name = "mango-kernel"
edition = "2024"

[lib]
name = "mango_kernel"
path = "src/lib.rs"

[dependencies]
mango-types = { path = "../types" }

[features]
default = ["std"]
std = []  # Enable for CPU tests, disable for GPU
```

**`#[no_std]` compatibility**:
```rust
#![cfg_attr(not(feature = "std"), no_std)]

#[cfg(not(feature = "std"))]
use core::*;

#[cfg(feature = "std")]
use std::*;
```

### Bazel Integration

```python
# crates/kernel/BUILD.bazel
rust_library(
    name = "kernel",
    srcs = glob(["src/**/*.rs"]),
    edition = "2024",
    deps = ["//crates/types"],
    visibility = ["//visibility:public"],
)

rust_test(
    name = "kernel_test",
    crate = ":kernel",
    edition = "2024",
    deps = [
        "//src:pde_solver",      # C library
        "//src:american_option",
        "@crates//:rand",
    ],
)
```

### Workspace Update

```toml
# Root Cargo.toml
[workspace]
members = [
    "crates/types",
    "crates/kernel",  # New
]
resolver = "2"

[workspace.package]
edition = "2024"  # Updated from 2021
```

## Validation Criteria (Phase 2 Success)

1. ✅ **Numerical accuracy**: <1e-6 relative error vs C (100 random configs)
2. ✅ **Compilation**: Both f32 and f64 compile and run
3. ✅ **No heap allocation**: Validated by miri
4. ✅ **All tests pass**: Unit, integration, validation
5. ✅ **`#[no_std]` compatible**: Ready for Phase 3 GPU compilation

## Risk Mitigation

| Risk | Impact | Mitigation |
|------|--------|------------|
| Numerical accuracy drift | High | Extensive validation (100+ test cases) |
| `#[no_std]` incompatibility | High | Early testing, avoid std library |
| GpuFloat trait insufficient | Medium | Add operations as needed during implementation |
| C FFI binding failures | Low | Use well-tested bindgen, fallback to manual FFI |

## Phase 2 Deliverables

1. **`crates/kernel/` crate** (~600-700 LOC)
   - float.rs: GpuFloat trait
   - black_scholes.rs: Operator + boundaries
   - tridiagonal.rs: Thomas algorithm
   - trbdf2.rs: Two-stage time stepping
   - lib.rs: Public API

2. **Test suite** (tests/cpu_tests.rs)
   - Unit tests (10+)
   - Integration tests (5+)
   - Validation suite (100+ random configs)

3. **Build system**
   - Cargo.toml with features
   - BUILD.bazel
   - build.rs for C FFI

4. **Documentation**
   - Inline doc comments
   - This design document

## Follow-Up Work

**Phase 3 (Build Vulkan Runtime)** will:
- Add spirv-std dependency
- Compile kernel to SPIR-V
- Create VulkanCompute runtime
- Dispatch kernels to GPU

**Phase 4 (Port Interpolation)** will:
- Move to CPU-only interpolation (not on critical path)

## References

- C implementation: `src/pde_solver.c`, `src/american_option.c`
- TR-BDF2 paper: Ascher, Ruuth, Wetton (1995)
- rust-gpu: https://github.com/EmbarkStudios/rust-gpu
- Thomas algorithm: https://en.wikipedia.org/wiki/Tridiagonal_matrix_algorithm
