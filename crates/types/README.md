# mango-types

Core type definitions for mango-iv Rust+GPU option pricing.

## Overview

This crate provides Rust types for GPU compute kernels. Types use `#[repr(C)]` for stable memory layout in GPU shaders, but are **not intended for C FFI in Phase 1**. FFI bridge will be added in Phase 6 (Week 15-16).

## Types

- `OptionType`: Call or Put enum
- `ExerciseType`: European or American enum
- `OptionParams`: Option parameters for GPU pricing kernels
- `GridParams`: Spatial grid parameters for PDE solver
- `TimeParams`: Time domain parameters

## Usage

```rust
use mango_types::{OptionParams, OptionType, GridParams, TimeParams};

// Create option parameters for GPU kernel
let params = OptionParams::american_put(100.0, 100.0, 1.0, 0.05, 0.20);
assert_eq!(params.option_type, OptionType::Put);

// Create uniform grid
let grid = GridParams::uniform(-1.0, 1.0, 101);
assert_eq!(grid.n_points, 101);

// Create time domain
let time = TimeParams::uniform(0.0, 1.0, 1000);
assert_eq!(time.n_steps, 1000);
```

## Phase 1 Scope

This crate is part of Phase 1 (Type Definitions) of the Rust+GPU rewrite. These types are designed for:
- GPU compute shader parameters
- Rust kernel function signatures
- rust-gpu `#[spirv(storage_buffer)]` compatibility

C FFI will be addressed in Phase 6.
