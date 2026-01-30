<!-- SPDX-License-Identifier: MIT -->
# Rust+GPU Rewrite Design

**Date**: 2025-11-01
**Status**: Design
**Goal**: Rewrite mango-option in Rust with unified CPU/GPU kernels for trading applications

## Problem

Current C implementation prices American options at 21ms per option on CPU. A trading session receives updates every 100ms, each containing 100-400 option prices across 10-20 equities. The batch requires 2-8 seconds to price on CPU (400 options × 21ms ÷ 16 cores = 525ms optimistically), exceeding the 100ms budget. Precomputing price tables takes 15-20 minutes for 300K grid points, limiting market response time.

## Goals

1. **Real-time pricing**: Price 100-400 options within 100ms update window (<0.25ms per option)
2. **Fast precomputation**: Reduce 15-20 minute table generation to seconds
3. **Code reuse**: Write computational kernels once, compile for CPU and GPU
4. **Incremental migration**: Ship value during transition from C to Rust

## Architecture

### rust-gpu Unified Kernel Approach

Computational kernels compile to both native code (CPU) and SPIR-V (GPU) from single Rust source. This eliminates duplication between backend implementations.

**Compilation paths**:
- **GPU**: `rustc` → `rustc_codegen_spirv` → SPIR-V → Vulkan compute shader
- **CPU**: `rustc` → native x86-64 binary

**Key constraint**: Kernels must be `#[no_std]` (no heap, no dynamic dispatch, no standard library). Current array-based PDE solver fits this constraint naturally.

### Project Structure

```
mango-option-rs/
├── crates/
│   ├── types/            # Shared types (OptionParams, Grid, etc.)
│   ├── kernel/           # GPU-compatible kernels (#[no_std])
│   │   ├── trbdf2.rs     # TR-BDF2 time-stepping
│   │   ├── tridiagonal.rs # Thomas algorithm (batched)
│   │   └── spatial.rs     # Black-Scholes operator
│   ├── runtime/          # Host runtime (Vulkan/CPU dispatch)
│   │   ├── vulkan.rs     # GPU backend via ash
│   │   └── cpu.rs        # CPU backend (direct kernel calls)
│   ├── interpolation/    # Pure Rust (CPU-only, already fast)
│   └── ffi-bridge/       # Temporary C interop during migration
└── build.rs              # Dual compilation (SPIR-V + native)
```

**Dependency rules**:
- `kernel` depends only on `types` (enforces `#[no_std]`)
- `runtime` depends on `kernel`, `types`, `ash`
- `interpolation` stays on CPU (500ns queries, GPU transfer overhead not worthwhile)
- No circular dependencies

### Kernel Design

**TR-BDF2 time-stepping kernel**:
```rust
#[spirv(compute(threads(256)))]
pub fn trbdf2_stage1(
    #[spirv(global_invocation_id)] gid: Vec3,
    #[spirv(storage_buffer)] u_current: &[f32],
    #[spirv(storage_buffer)] u_stage: &mut [f32],
    #[spirv(push_constant)] params: &TimeStepParams,
) {
    let i = gid.x as usize;
    if i >= params.n_points { return; }

    // Stage 1: Trapezoidal rule (γ·Δt step)
    let Lu = spatial_operator(i, u_current, params);
    u_stage[i] = u_current[i] + params.gamma * params.dt * 0.5 * Lu;
}
```

**Storage buffers**: All arrays in GPU memory
**Push constants**: Small parameters (<128 bytes) for efficiency
**Workgroup size**: 256 threads (tunable: benchmark 64, 128, 256, 512, 1024)

### Tridiagonal Solver Strategy

TR-BDF2 requires tridiagonal solves (Thomas algorithm), which is sequential. GPU solution: batch 256 independent systems in parallel.

**Batched solving**:
```rust
#[spirv(compute(threads(256)))]
pub fn solve_tridiagonal_batch(
    #[spirv(global_invocation_id)] gid: Vec3,
    #[spirv(storage_buffer)] diag: &[f32],
    #[spirv(storage_buffer)] rhs: &[f32],
    #[spirv(storage_buffer)] solution: &mut [f32],
    #[spirv(push_constant)] params: &BatchParams,
) {
    let system_id = gid.x as usize;
    let offset = system_id * params.system_size;

    // Each thread solves one option's tridiagonal system
    thomas_solve(&diag[offset..], &rhs[offset..], &mut solution[offset..]);
}
```

**Perfect fit**: 100-400 options per update = 100-400 independent PDEs. Each GPU thread prices one option.

### Memory Management

**Host↔Device data flow**:
```
[Host]                   [Device]
OptionParams[400]  →     GPU Buffers
  (upload ~15μs)         u, Lu, rhs

                         Compute Pipeline
                         - Stage 1
                         - Stage 2
                         - 1000 timesteps

Results[400]       ←     Solution
  (download ~5μs)
```

**Buffer strategy**:
- **Staging buffers** (CPU-visible): Small (params/results)
- **Device-local buffers** (GPU-only): Large (intermediate arrays)
- **Persistent allocation**: Reuse buffers across frames (avoid alloc per update)

**Transfer budget** (100ms update window):
- Upload: ~15μs (400 options × 48 bytes = 19KB)
- Compute: ~2ms (target 20× speedup over 21ms CPU)
- Download: ~5μs (400 × 4 bytes = 1.6KB)
- **Remaining**: 98ms for application logic

### Build System

**Cargo workspace**:
```toml
[workspace]
members = ["crates/types", "crates/kernel", "crates/runtime",
           "crates/interpolation", "crates/ffi-bridge"]
```

**Build script** (`crates/runtime/build.rs`):
```rust
fn main() {
    // Compile kernel to SPIR-V using spirv-builder
    let result = spirv_builder::SpirvBuilder::new(
        "../kernel",
        "spirv-unknown-vulkan1.2"
    ).build().expect("Failed to build SPIR-V");

    // Embed SPIR-V path for runtime loading
    println!("cargo:rustc-env=KERNEL_SPIRV_PATH={}",
             result.module.unwrap_single().display());
}
```

**Runtime loads pre-compiled SPIR-V**:
```rust
impl VulkanCompute {
    pub fn new() -> Result<Self> {
        let spirv_bytes = include_bytes!(env!("KERNEL_SPIRV_PATH"));
        let shader_module = device.create_shader_module(spirv_bytes)?;
        // Create compute pipeline...
    }
}
```

**Integration with Bazel**: Call Cargo for rust-gpu compilation (Bazel lacks native support).

## Migration Strategy

### Phase Timeline (20 weeks)

**Phase 1: Port Type Definitions** (Week 1)
```rust
#[repr(C)]
pub struct OptionParams {
    pub spot_price: f64,
    pub strike: f64,
    pub volatility: f64,
    // ...
}
```

**Phase 2: Port Interpolation** (Week 2-3)
- Replace `interp_cubic.c` (1364 LOC)
- Benefit: Eliminate bounds-check bugs, memory safety
- Target: Pure Rust, CPU-only (already 500ns)
- High bug surface area (pointer arithmetic)

**Phase 3: Port Kernels to rust-gpu** (Week 4-8)
- Replace `pde_solver.c` core loops (638 LOC → ~400 LOC)
- Write `#[no_std]` Rust kernels
- This is where rust-gpu enters

**Phase 4: Build Vulkan Runtime** (Week 9-12)
- New `VulkanCompute` using `ash`
- Upload params → dispatch compute → download results
- Device buffer allocation, pipeline creation

**Phase 5: Build CPU Fallback** (Week 13-14)
- `CpuCompute` calls rust-gpu kernels directly (no Vulkan)
- Fallback for systems without GPU

**Phase 6: FFI Bridge** (Week 15-16)
```rust
#[no_mangle]
pub extern "C" fn rust_price_batch(params: *const COptionParams, ...) {
    let engine = VulkanCompute::new().unwrap();
    engine.price_batch(params)
}
```
C code forwards to Rust during transition.

**Phase 7: Validation & Benchmarking** (Week 17-20)
- Run C test suite against Rust implementation
- Compare numerical accuracy (target: <1e-4 relative error)
- Benchmark GPU vs CPU

**Phase 8: Remove FFI Bridge** (Week 21+)
- Delete C code once all modules ported
- Pure Rust system

### C Code Reorganization (Phase 0)

Move FFI boundaries to dedicated headers before migration:
```
src/
├── core/                   # Core algorithms (stays C longest)
│   ├── pde_solver.c
│   ├── american_option.c
│   └── tridiagonal.c
└── ffi/                    # FFI exports for Rust
    ├── ffi_types.h         # C/Rust shared types
    ├── ffi_pde.h           # PDE solver exports
    └── ffi_option.h        # Option pricing exports
```

This makes `bindgen` (Rust FFI generator) clean.

## Error Handling

**Kernel errors** (no panic in `#[no_std]`):
```rust
#[spirv(compute(threads(256)))]
pub fn price_option(
    #[spirv(storage_buffer)] params: &[OptionParams],
    #[spirv(storage_buffer)] results: &mut [PricingResult],
) {
    let i = get_global_id();

    if params[i].volatility <= 0.0 {
        results[i].price = f32::NAN;  // Signal error
        results[i].error_code = 1;    // Invalid volatility
        return;
    }

    results[i].price = computed_price;
    results[i].error_code = 0;  // Success
}
```

**Host errors** (normal Rust):
```rust
pub enum ComputeError {
    VulkanError(ash::vk::Result),
    InvalidParams { index: usize, reason: String },
    DeviceOutOfMemory,
}

impl VulkanCompute {
    pub fn price_batch(&self, params: &[OptionParams])
        -> Result<Vec<f64>, ComputeError>
    {
        self.dispatch_compute()?;

        // Check kernel errors
        for (i, result) in results.iter().enumerate() {
            if result.error_code != 0 {
                return Err(ComputeError::InvalidParams {
                    index: i,
                    reason: decode_error(result.error_code)
                });
            }
        }
        Ok(results.iter().map(|r| r.price).collect())
    }
}
```

## Testing Strategy

**Three tiers**:

1. **Kernel tests** (CPU-only, fast):
```rust
#[test]
fn test_trbdf2_convergence() {
    let mut u_stage = vec![0.0; 101];
    kernel::trbdf2_stage1(...);  // Call directly
    assert!((u_stage[50] - expected).abs() < 1e-6);
}
```

2. **Numerical accuracy** (Rust vs C):
```rust
#[test]
fn test_rust_matches_c_pricing() {
    let c_price = unsafe { ffi::american_option_price(&params) };
    let rust_price = VulkanCompute::new()?.price_single(&params)?;
    assert_relative_eq!(rust_price, c_price, epsilon = 1e-4);
}
```

3. **End-to-end GPU** (integration):
```rust
#[test]
fn test_batch_400_options() {
    let params = vec![make_random_params(); 400];
    let prices = VulkanCompute::new()?.price_batch(&params)?;
    assert_eq!(prices.len(), 400);
    assert!(prices.iter().all(|&p| p > 0.0 && p.is_finite()));
}
```

## Performance Optimization

**GPU-specific strategies**:

1. **Coalesced memory access**: Current `LAYOUT_M_INNER` already optimized (moneyness varies fastest)
2. **Workgroup size tuning**: Benchmark 64, 128, 256, 512, 1024 threads
3. **Push constants**: Use for grid params (dt, dx, n_points), storage buffers for data
4. **Double buffering**: Prepare next batch while GPU processes current
5. **Batching strategy**: Round up to nearest workgroup multiple (256 threads)

**Expected performance**:
- Current CPU (OpenMP): ~500ms for 400 options (16-core parallelism)
- Target GPU (Vulkan): <2ms for 400 options
- **Speedup**: 250× over CPU

**Optimization order**:
1. Start: Get correct results (any speed)
2. Profile: Identify bottlenecks (transfer vs compute)
3. Optimize: Tune based on measurements

## Risk Mitigation

| Risk | Impact | Mitigation |
|------|--------|------------|
| rust-gpu instability | High | Keep C fallback, pin dependencies |
| GPU driver bugs | Medium | Test multiple GPUs, CPU fallback |
| Numerical accuracy drift | High | Comprehensive Rust vs C validation |
| Memory transfer overhead | Medium | Persistent buffers, double buffering, profile |
| Learning curve | Medium | Start with interpolation (no GPU) |
| Vendor lock-in | Low | Vulkan supports all GPUs (NVIDIA, AMD, Intel) |

**Fallback mechanism**:
```rust
pub enum Backend {
    Vulkan(VulkanCompute),    // Try first
    Cpu(CpuCompute),          // Fallback if no GPU
    CFallback,                // Last resort: original C code
}

impl Backend {
    pub fn auto_select() -> Result<Self> {
        VulkanCompute::new()
            .map(Backend::Vulkan)
            .or_else(|_| CpuCompute::new().map(Backend::Cpu))
            .unwrap_or(Backend::CFallback)
    }
}
```

**Validation**: Run 1000 random options through C and Rust, assert relative error <1e-4.

## Success Criteria

1. **Correctness**: <1e-4 relative error vs C implementation (1000 random samples)
2. **Latency**: <2ms for 400 options on GPU (<0.25ms per option average)
3. **Precomputation**: <60 seconds for 300K grid points (vs 15-20 minutes CPU)
4. **Reliability**: CPU fallback works when GPU unavailable
5. **Migration**: Each phase ships incremental value

## Non-Goals

- Multi-GPU support (single GPU sufficient for 400 options)
- Exotic options (barriers, Asians) remain future work
- Python bindings (focus on Rust first)

## References

- **rust-gpu**: https://github.com/EmbarkStudios/rust-gpu
- **ash**: https://crates.io/crates/ash (Vulkan bindings)
- **gpu-allocator**: https://crates.io/crates/gpu-allocator
- **Elements of Style**: Strunk & White (1918) - writing guidelines
