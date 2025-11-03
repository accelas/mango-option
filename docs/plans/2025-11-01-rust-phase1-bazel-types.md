# Rust Phase 1: Bazel + rules_rust Setup and Type Definitions

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Set up Rust toolchain in Bazel and port core C type definitions to Rust with FFI compatibility.

**Architecture:** Add rules_rust to MODULE.bazel, create Cargo workspace for Rust crates, port OptionParams and related types with `#[repr(C)]` for C interop.

**Tech Stack:** Rust (1.75+), Bazel (7.x), rules_rust (0.48+), Cargo

---

## Task 1: Add rules_rust to Bazel

**Files:**
- Modify: `MODULE.bazel`

**Step 1: Add rules_rust dependency to MODULE.bazel**

Add after existing dependencies:

```python
# Rust support
bazel_dep(name = "rules_rust", version = "0.48.0")

# Rust toolchain
rust = use_extension("@rules_rust//rust:extensions.bzl", "rust")
rust.toolchain(
    edition = "2021",
    versions = ["1.75.0"],
)
use_repo(rust, "rust_toolchains")
register_toolchains("@rust_toolchains//:all")

# Crates repository for third-party Rust dependencies
crate = use_extension("@rules_rust//crate_universe:extension.bzl", "crate")
crate.from_cargo(
    name = "crates",
    cargo_lockfile = "//:Cargo.lock",
    manifests = [
        "//:Cargo.toml",
        "//crates/types:Cargo.toml",
    ],
)
use_repo(crate, "crates")
```

**Step 2: Verify Bazel can load rules_rust**

Run: `bazel query @rules_rust//...`
Expected: List of rules_rust targets (no errors)

**Step 3: Commit**

```bash
git add MODULE.bazel
git commit -m "Add rules_rust to Bazel configuration

- Add rules_rust 0.48.0 dependency
- Configure Rust 1.75.0 toolchain with edition 2021
- Setup crate_universe for third-party dependencies"
```

---

## Task 2: Create Cargo workspace structure

**Files:**
- Create: `Cargo.toml` (workspace root)
- Create: `Cargo.lock` (empty, will be generated)

**Step 1: Create workspace Cargo.toml**

```toml
[workspace]
members = [
    "crates/types",
]
resolver = "2"

[workspace.package]
version = "0.1.0"
edition = "2021"
authors = ["mango-iv contributors"]
license = "MIT OR Apache-2.0"

[workspace.dependencies]
# Shared dependencies across crates (none yet)
```

**Step 2: Initialize Cargo.lock**

Run: `touch Cargo.lock`
Expected: Empty Cargo.lock file created

**Step 3: Commit**

```bash
git add Cargo.toml Cargo.lock
git commit -m "Create Cargo workspace structure

- Add workspace Cargo.toml with crates/types member
- Initialize empty Cargo.lock (will be generated)"
```

---

## Task 3: Create crates/types with OptionParams

**Files:**
- Create: `crates/types/Cargo.toml`
- Create: `crates/types/src/lib.rs`
- Create: `crates/types/BUILD.bazel`

**Step 1: Create crates/types/Cargo.toml**

```toml
[package]
name = "mango-types"
version.workspace = true
edition.workspace = true
authors.workspace = true
license.workspace = true

[lib]
name = "mango_types"
path = "src/lib.rs"

[dependencies]
# No dependencies yet - pure types
```

**Step 2: Create crates/types/src/lib.rs with OptionType**

```rust
//! Core types for mango-iv option pricing
//!
//! These types are `#[repr(C)]` compatible with the existing C codebase.

/// Option type: call or put
#[repr(C)]
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum OptionType {
    Call = 0,
    Put = 1,
}

/// Exercise style: European or American
#[repr(C)]
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum ExerciseType {
    European = 0,
    American = 1,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_option_type_repr() {
        // Verify C-compatible memory layout
        assert_eq!(std::mem::size_of::<OptionType>(), 4);
        assert_eq!(OptionType::Call as i32, 0);
        assert_eq!(OptionType::Put as i32, 1);
    }

    #[test]
    fn test_exercise_type_repr() {
        assert_eq!(std::mem::size_of::<ExerciseType>(), 4);
        assert_eq!(ExerciseType::European as i32, 0);
        assert_eq!(ExerciseType::American as i32, 1);
    }
}
```

**Step 3: Create crates/types/BUILD.bazel**

```python
load("@rules_rust//rust:defs.bzl", "rust_library", "rust_test")

rust_library(
    name = "types",
    srcs = ["src/lib.rs"],
    edition = "2021",
    visibility = ["//visibility:public"],
)

rust_test(
    name = "types_test",
    crate = ":types",
    edition = "2021",
)
```

**Step 4: Generate Cargo.lock**

Run: `cargo generate-lockfile`
Expected: Cargo.lock updated with mango-types

**Step 5: Run Cargo tests**

Run: `cargo test -p mango-types`
Expected:
```
running 2 tests
test tests::test_option_type_repr ... ok
test tests::test_exercise_type_repr ... ok

test result: ok. 2 passed; 0 failed
```

**Step 6: Run Bazel tests**

Run: `bazel test //crates/types:types_test`
Expected: PASSED

**Step 7: Commit**

```bash
git add crates/types/Cargo.toml crates/types/src/lib.rs crates/types/BUILD.bazel Cargo.lock
git commit -m "Add crates/types with OptionType and ExerciseType

- Create mango-types crate with C-compatible enums
- Add #[repr(C)] for FFI compatibility
- Add tests verifying memory layout
- Wire up Bazel build with rust_library and rust_test"
```

---

## Task 4: Add OptionParams struct

**Files:**
- Modify: `crates/types/src/lib.rs`

**Step 1: Add OptionParams struct to lib.rs**

Add after ExerciseType definition:

```rust
/// Option parameters for pricing calculations
///
/// C-compatible struct matching `OptionData` from american_option.h
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct OptionParams {
    pub spot_price: f64,
    pub strike: f64,
    pub time_to_maturity: f64,
    pub risk_free_rate: f64,
    pub dividend_yield: f64,
    pub volatility: f64,
    pub option_type: OptionType,
    pub exercise_type: ExerciseType,
}

impl OptionParams {
    /// Create American put option parameters
    pub fn american_put(
        spot: f64,
        strike: f64,
        maturity: f64,
        rate: f64,
        volatility: f64,
    ) -> Self {
        Self {
            spot_price: spot,
            strike,
            time_to_maturity: maturity,
            risk_free_rate: rate,
            dividend_yield: 0.0,
            volatility,
            option_type: OptionType::Put,
            exercise_type: ExerciseType::American,
        }
    }

    /// Create American call option parameters
    pub fn american_call(
        spot: f64,
        strike: f64,
        maturity: f64,
        rate: f64,
        volatility: f64,
    ) -> Self {
        Self {
            spot_price: spot,
            strike,
            time_to_maturity: maturity,
            risk_free_rate: rate,
            dividend_yield: 0.0,
            volatility,
            option_type: OptionType::Call,
            exercise_type: ExerciseType::American,
        }
    }
}
```

**Step 2: Add tests for OptionParams**

Add to tests module in lib.rs:

```rust
#[test]
fn test_option_params_layout() {
    // Verify struct size and alignment
    assert_eq!(std::mem::size_of::<OptionParams>(), 56);
    assert_eq!(std::mem::align_of::<OptionParams>(), 8);
}

#[test]
fn test_american_put_constructor() {
    let params = OptionParams::american_put(100.0, 100.0, 1.0, 0.05, 0.20);
    assert_eq!(params.spot_price, 100.0);
    assert_eq!(params.strike, 100.0);
    assert_eq!(params.time_to_maturity, 1.0);
    assert_eq!(params.risk_free_rate, 0.05);
    assert_eq!(params.volatility, 0.20);
    assert_eq!(params.dividend_yield, 0.0);
    assert_eq!(params.option_type, OptionType::Put);
    assert_eq!(params.exercise_type, ExerciseType::American);
}

#[test]
fn test_american_call_constructor() {
    let params = OptionParams::american_call(100.0, 105.0, 0.5, 0.03, 0.25);
    assert_eq!(params.option_type, OptionType::Call);
    assert_eq!(params.strike, 105.0);
}
```

**Step 3: Run tests**

Run: `cargo test -p mango-types`
Expected: 5 tests pass

Run: `bazel test //crates/types:types_test`
Expected: PASSED

**Step 4: Commit**

```bash
git add crates/types/src/lib.rs
git commit -m "Add OptionParams struct with constructors

- Add C-compatible OptionParams struct
- Add american_put() and american_call() constructors
- Add tests for struct layout and constructors
- Matches OptionData from C codebase"
```

---

## Task 5: Add Grid types

**Files:**
- Modify: `crates/types/src/lib.rs`

**Step 1: Add GridParams struct**

Add after OptionParams:

```rust
/// Grid parameters for PDE solver
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct GridParams {
    pub n_points: usize,
    pub x_min: f64,
    pub x_max: f64,
    pub dx: f64,
}

impl GridParams {
    /// Create uniform grid
    pub fn uniform(x_min: f64, x_max: f64, n_points: usize) -> Self {
        let dx = (x_max - x_min) / (n_points as f64 - 1.0);
        Self {
            n_points,
            x_min,
            x_max,
            dx,
        }
    }
}

/// Time domain parameters
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct TimeParams {
    pub t_start: f64,
    pub t_end: f64,
    pub dt: f64,
    pub n_steps: usize,
}

impl TimeParams {
    /// Create time domain with uniform steps
    pub fn uniform(t_start: f64, t_end: f64, n_steps: usize) -> Self {
        let dt = (t_end - t_start) / n_steps as f64;
        Self {
            t_start,
            t_end,
            dt,
            n_steps,
        }
    }
}
```

**Step 2: Add tests**

```rust
#[test]
fn test_grid_params() {
    let grid = GridParams::uniform(-1.0, 1.0, 101);
    assert_eq!(grid.n_points, 101);
    assert_eq!(grid.x_min, -1.0);
    assert_eq!(grid.x_max, 1.0);
    assert!((grid.dx - 0.02).abs() < 1e-10);
}

#[test]
fn test_time_params() {
    let time = TimeParams::uniform(0.0, 1.0, 1000);
    assert_eq!(time.n_steps, 1000);
    assert_eq!(time.t_start, 0.0);
    assert_eq!(time.t_end, 1.0);
    assert!((time.dt - 0.001).abs() < 1e-10);
}
```

**Step 3: Run tests**

Run: `cargo test -p mango-types`
Expected: 7 tests pass

**Step 4: Commit**

```bash
git add crates/types/src/lib.rs
git commit -m "Add GridParams and TimeParams types

- Add GridParams with uniform() constructor
- Add TimeParams with uniform() constructor
- Add tests for grid and time domain parameters"
```

---

## Task 6: Add documentation and update workspace

**Files:**
- Create: `crates/types/README.md`
- Modify: `Cargo.toml`

**Step 1: Create crates/types/README.md**

```markdown
# mango-types

Core type definitions for mango-iv option pricing library.

## Overview

This crate provides C-compatible types (`#[repr(C)]`) for interoperability with the existing C codebase during the Rust migration.

## Types

- `OptionType`: Call or Put
- `ExerciseType`: European or American
- `OptionParams`: Option parameters for pricing
- `GridParams`: Spatial grid parameters for PDE solver
- `TimeParams`: Time domain parameters

## Usage

```rust
use mango_types::{OptionParams, OptionType};

let params = OptionParams::american_put(100.0, 100.0, 1.0, 0.05, 0.20);
assert_eq!(params.option_type, OptionType::Put);
```

## FFI Compatibility

All types use `#[repr(C)]` for direct C interoperability:

```c
// C side
extern OptionParams rust_create_params(double spot, double strike, ...);
```

```rust
// Rust side
#[no_mangle]
pub extern "C" fn rust_create_params(
    spot: f64,
    strike: f64,
    ...
) -> OptionParams {
    OptionParams::american_put(spot, strike, ...)
}
```
```

**Step 2: Add workspace metadata to root Cargo.toml**

Add after `[workspace.package]`:

```toml
[workspace.metadata]
description = "Rust+GPU option pricing library"
repository = "https://github.com/yourusername/mango-iv"
readme = "README.md"
keywords = ["options", "pricing", "finance", "gpu", "rust"]
categories = ["science", "mathematics"]
```

**Step 3: Build full workspace**

Run: `cargo build --workspace`
Expected: Compiling mango-types v0.1.0, Finished dev

Run: `bazel build //crates/types:types`
Expected: Build completed successfully

**Step 4: Commit**

```bash
git add crates/types/README.md Cargo.toml
git commit -m "Add types documentation and workspace metadata

- Add README for mango-types crate
- Document FFI compatibility patterns
- Add workspace metadata for crates.io
- Verify full workspace builds"
```

---

## Task 7: Update main project docs

**Files:**
- Modify: `README.md`
- Modify: `docs/plans/2025-11-01-rust-gpu-rewrite.md`

**Step 1: Add Rust section to README.md**

Add after "## Project Structure" section:

```markdown
### Rust Crates (Experimental)

**Status**: Phase 1 - Type Definitions

Rust rewrite for GPU acceleration is in progress. See `docs/plans/2025-11-01-rust-gpu-rewrite.md` for the full design.

```
mango-iv-rs/
├── Cargo.toml           # Workspace root
└── crates/
    └── types/           # Core type definitions (#[repr(C)] for FFI)
```

**Build Rust crates**:
```bash
cargo build --workspace   # Cargo build
bazel build //crates/...  # Bazel build
cargo test -p mango-types # Run tests
```

**Branch**: `experimental/rust-gpu-rewrite` (not merged to main)
```

**Step 2: Update status in rust-gpu-rewrite.md**

Find "### Phase Timeline (20 weeks)" section and update:

```markdown
**Phase 1: Port Type Definitions** (Week 1) ✅ COMPLETE
- Created mango-types crate with C-compatible types
- Added OptionParams, GridParams, TimeParams
- Wired up Bazel + rules_rust integration
- All tests passing
```

**Step 3: Verify documentation renders**

Run: `ls README.md crates/types/README.md docs/plans/2025-11-01-rust-gpu-rewrite.md`
Expected: All files exist

**Step 4: Commit**

```bash
git add README.md docs/plans/2025-11-01-rust-gpu-rewrite.md
git commit -m "Update docs with Rust phase 1 completion

- Add Rust crates section to README
- Mark Phase 1 complete in design doc
- Document experimental status and branch location"
```

---

## Verification Steps

After completing all tasks, verify the setup:

1. **Cargo workspace builds**:
   ```bash
   cargo clean
   cargo build --workspace
   cargo test --workspace
   ```
   Expected: All pass

2. **Bazel builds Rust**:
   ```bash
   bazel clean
   bazel build //crates/types:types
   bazel test //crates/types:types_test
   ```
   Expected: All pass

3. **C codebase still builds**:
   ```bash
   bazel test //tests:american_option_test //tests:pde_solver_test
   ```
   Expected: All pass (C code unaffected)

4. **Git status clean**:
   ```bash
   git status
   ```
   Expected: On branch experimental/rust-gpu-rewrite, nothing to commit

---

## Next Steps

After Phase 1 completion:

- **Phase 2**: Port interpolation module to Rust (Week 2-3)
- **Phase 3**: Port PDE kernels to rust-gpu (Week 4-8)
- **Phase 4**: Build Vulkan runtime (Week 9-12)

See `docs/plans/2025-11-01-rust-gpu-rewrite.md` for complete roadmap.

---

## Troubleshooting

**Problem**: `bazel query @rules_rust//...` fails
**Solution**: Run `bazel sync --only=rules_rust` to fetch dependency

**Problem**: Cargo.lock conflicts
**Solution**: Run `cargo generate-lockfile` to regenerate

**Problem**: Rust tests fail with "can't find crate"
**Solution**: Run `cargo build` first to ensure crate is compiled

**Problem**: Bazel can't find Rust toolchain
**Solution**: Verify `register_toolchains("@rust_toolchains//:all")` in MODULE.bazel
