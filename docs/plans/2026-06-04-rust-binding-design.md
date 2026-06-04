# Rust Binding Design (v1: Pricing + IV)

**Date**: 2026-06-04
**Status**: Design
**Goal**: Expose the existing mango-option C++23 pricing library to Rust callers
via a safe, idiomatic binding covering American option pricing and FDM implied
volatility.

## Problem

`mango-option` is a C++23 library with one language binding today (Python, via
pybind11). Rust consumers need to call the pricer (American value + Greeks) and
the FDM implied-vol solver. The public C++ API leans on features that do not
cross a C ABI directly — `std::expected<T, E>`, `std::span`, templates
(`PriceTableBuilder<4>`), PMR, concepts, and move-only result types with
`shared_ptr` internals — so a binding cannot be generated mechanically from the
headers.

Note: two archived 2025-11-01 plans (`rust-gpu-rewrite`, `rust-phase1-bazel-types`)
describe a *rewrite* of the kernels in Rust + GPU. That effort was never merged
and is **out of scope**. This design is a *binding* (let Rust call the existing
C++), not a port.

## Goals

1. Price American options from Rust: `value`, `value_at(spot)`, `delta`,
   `gamma`, `theta`.
2. Solve FDM implied volatility from Rust, with optional solver config.
3. Full input fidelity: constant rate **or** yield curve; continuous **and**
   discrete dividends.
4. Idiomatic Rust surface: callers write zero `unsafe`; errors are `Result`.
5. Build and test in-tree via Bazel `rules_rust`; leave existing C++ targets
   untouched.

## Non-Goals (v1)

- Batch solving (`solve_batch`), price tables, interpolated IV solver.
- Explicit custom PDE grids (`PDEGridSpec` variant), snapshots, custom initial
  conditions. Pricing uses the auto-grid convenience path
  (`solve_american_option`).
- Publishing to crates.io / standalone Cargo build. v1 ships in-tree via Bazel.
- Thread-safe concurrent use of a single result handle (documented, not
  enforced beyond Rust's `!Send`/`!Sync`).

## Approach

Chosen: **hand-written `extern "C"` shim + raw `-sys` crate + safe wrapper
crate** (over `cxx`, `autocxx`, header `bindgen`, which all choke on the C++23
surface). The `-sys` declarations are hand-written (small surface) and guarded
by a bidirectional ABI layout test rather than generated via `rust_bindgen`
(which would pull a libclang toolchain into Bazel).

## Architecture

Three build units:

```
src/ffi/
  mango_c_api.h          # stable C header — single source of truth for the ABI
  mango_c_api.cpp        # extern "C" implementation over the C++23 API
  BUILD.bazel            # cc_library :mango_c_api
crates/
  mango-option-sys/      # raw FFI: repr(C) structs + extern "C" decls (hand-written)
    src/lib.rs
    tests/layout.rs      # ABI size/align assertions
    Cargo.toml
    BUILD.bazel
  mango-option/          # safe wrapper: idiomatic structs, Result, RAII
    src/lib.rs
    src/types.rs         # OptionType, Rate, TenorPoint, Dividend, OptionSpec
    src/error.rs         # Error, ErrorKind, MangoError -> Error mapping
    src/pricing.rs       # PricingParams, price_american, PriceResult (RAII handle)
    src/iv.rs            # IvQuery, IvConfig, IvSuccess, solve_iv
    tests/               # integration tests against C++ reference values
    Cargo.toml
    BUILD.bazel
MODULE.bazel             # + rules_rust toolchain (no crate_universe; zero 3rd-party crates)
```

v1 has **zero third-party Rust dependencies** (std only), so no `crate_universe`
extension is needed — minimal Bazel wiring.

### Data flow

```
Rust safe API (PricingParams / IvQuery, owns Vec<TenorPoint>, Vec<Dividend>)
   │  builds repr(C) structs; keeps Vecs alive across the call
   ▼
mango-option-sys  (unsafe extern "C" call into the shim)
   ▼
src/ffi/mango_c_api.cpp  (extern "C")
   │  rebuilds OptionSpec / PricingParams / IVQuery / YieldCurve
   │  calls solve_american_option / IVSolver::solve
   │  std::expected -> MangoStatus + out-param; try/catch wraps any exception
   ▼
mango-option C++23 library
```

## Component: C ABI shim (`src/ffi/mango_c_api.h`)

### Status and error

```c
typedef enum {
  MANGO_OK = 0,
  MANGO_ERR_VALIDATION = 1,
  MANGO_ERR_ARBITRAGE = 2,
  MANGO_ERR_NO_CONVERGENCE = 3,
  MANGO_ERR_BRACKETING = 4,
  MANGO_ERR_SOLVER = 5
} MangoStatus;

typedef struct {
  int32_t code;        // a MangoStatus value, for category
  char    message[256];// null-terminated human-readable diagnostic
} MangoError;
```

Every fallible function takes a nullable `MangoError* out_err`, filled only on a
non-`MANGO_OK` return. The shim never lets a C++ exception cross the boundary:
the body is wrapped in `try { ... } catch (...) { -> MANGO_ERR_SOLVER }`.

### Input types (flat POD + pointer/length arrays)

```c
typedef enum { MANGO_CALL = 0, MANGO_PUT = 1 } MangoOptionType;

typedef struct { double calendar_time; double amount; } MangoDividend;
typedef struct { double tenor; double log_discount; } MangoTenorPoint;

// Rate: constant when n_tenor_points == 0 (use rate_const),
//       else a yield curve rebuilt from the tenor points.
typedef struct {
  double spot;
  double strike;
  double maturity;
  double dividend_yield;            // continuous yield
  double volatility;
  double rate_const;                // used iff n_tenor_points == 0
  const MangoTenorPoint* tenor_points;
  size_t n_tenor_points;
  const MangoDividend* dividends;   // discrete schedule (may be null/0)
  size_t n_dividends;
  MangoOptionType option_type;
} MangoPricingParams;

typedef struct {
  double spot;
  double strike;
  double maturity;
  double dividend_yield;
  double market_price;
  double rate_const;
  const MangoTenorPoint* tenor_points;
  size_t n_tenor_points;
  const MangoDividend* dividends;
  size_t n_dividends;
  MangoOptionType option_type;
} MangoIvQuery;
```

The exact field order/padding is the ABI contract; mirrored by `static_assert`
in the header and by `assert!(size_of/align_of ...)` in `-sys`.

### Pricing — opaque handle (preserves `value_at`/Greeks without re-solving)

```c
typedef struct MangoAmericanResult MangoAmericanResult; // opaque

MangoStatus mango_price_american(const MangoPricingParams* params,
                                 MangoAmericanResult** out_result,
                                 MangoError* out_err);

double mango_american_value(const MangoAmericanResult*);
double mango_american_value_at(const MangoAmericanResult*, double spot);
double mango_american_delta(const MangoAmericanResult*);
double mango_american_gamma(const MangoAmericanResult*);
double mango_american_theta(const MangoAmericanResult*);
void   mango_american_result_free(MangoAmericanResult*);
```

`mango_price_american` heap-allocates a struct owning the move-only
`AmericanOptionResult`; accessors delegate to its const methods;
`..._free` deletes it.

### IV — POD result

```c
typedef struct {
  double implied_vol;
  size_t iterations;
  double final_error;
  double vega;        // valid iff has_vega != 0
  int    has_vega;
} MangoIvSuccess;

// All-zero means "use library defaults".
typedef struct {
  double target_price_error; // 0 => default
  int    max_iter;           // 0 => default
  double tolerance;          // 0 => default
} MangoIvConfig;

MangoStatus mango_solve_iv(const MangoIvQuery* query,
                           const MangoIvConfig* config, // nullable => defaults
                           MangoIvSuccess* out_success,
                           MangoError* out_err);
```

### Shim implementation notes

- Rebuild `RateSpec`: `n_tenor_points == 0` → `double rate_const`; else
  construct a `YieldCurve` from the `(tenor, log_discount)` points.
- Rebuild discrete dividends into `std::vector<Dividend>`.
- Map `OptionType` ↔ `MangoOptionType`.
- Pricing calls `solve_american_option(params)` (auto grid).
- IV builds `IVSolverConfig` from `MangoIvConfig` (only overriding fields that
  are non-zero), then `IVSolver::solve(query)`.
- Error mapping: `ValidationError` → `MANGO_ERR_VALIDATION`; IV
  `ArbitrageViolation`/`BracketingFailed`/`MaxIterationsExceeded` →
  `MANGO_ERR_ARBITRAGE`/`MANGO_ERR_BRACKETING`/`MANGO_ERR_NO_CONVERGENCE`;
  `SolverError` and any caught exception → `MANGO_ERR_SOLVER`. The `message`
  carries the C++ diagnostic text (and, for IV failures, iterations / last_vol
  when available).

## Component: `-sys` crate (`mango-option-sys`)

- `#[repr(C)]` mirrors of every struct/enum above, plus `extern "C"` blocks
  declaring all functions. No safe API, no logic.
- `tests/layout.rs`: `const _: () = assert!(size_of::<MangoPricingParams>() == N)`
  and matching `align_of` for each struct, where `N` matches a `static_assert`
  in `mango_c_api.h`. Drift on either side breaks the build.

## Component: safe crate (`mango-option`)

```rust
pub enum OptionType { Call, Put }
pub enum Rate { Const(f64), Curve(Vec<TenorPoint>) }
pub struct TenorPoint { pub tenor: f64, pub log_discount: f64 }
pub struct Dividend { pub calendar_time: f64, pub amount: f64 }

pub struct OptionSpec {
    pub spot: f64,
    pub strike: f64,
    pub maturity: f64,
    pub dividend_yield: f64,
    pub rate: Rate,
    pub discrete_dividends: Vec<Dividend>,
    pub option_type: OptionType,
}

pub struct PricingParams { pub spec: OptionSpec, pub volatility: f64 }

pub struct PriceResult { /* owns *mut MangoAmericanResult; Drop frees */ }
impl PriceResult {
    pub fn value(&self) -> f64;
    pub fn value_at(&self, spot: f64) -> f64;
    pub fn delta(&self) -> f64;
    pub fn gamma(&self) -> f64;
    pub fn theta(&self) -> f64;
}
pub fn price_american(params: &PricingParams) -> Result<PriceResult, Error>;

pub struct IvQuery { pub spec: OptionSpec, pub market_price: f64 }
#[derive(Default)]
pub struct IvConfig {
    pub target_price_error: Option<f64>,
    pub max_iter: Option<u32>,
    pub tolerance: Option<f64>,
}
pub struct IvSuccess {
    pub implied_vol: f64,
    pub iterations: usize,
    pub final_error: f64,
    pub vega: Option<f64>,
}
pub fn solve_iv(query: &IvQuery, config: &IvConfig) -> Result<IvSuccess, Error>;

pub struct Error { pub kind: ErrorKind, pub message: String }
pub enum ErrorKind { Validation, Arbitrage, NoConvergence, Bracketing, Solver }
impl std::fmt::Display for Error { /* "{kind}: {message}" */ }
impl std::error::Error for Error {}
```

Responsibilities of the safe layer:

- Convert `OptionSpec` into the C structs: split `Rate` into
  `rate_const`/`tenor_points`; pass `discrete_dividends` as `ptr+len`. The
  source `Vec`s stay owned by a local binding for the duration of the FFI call
  (no dangling pointers).
- `price_american` calls the shim, checks `MangoStatus`, and on success wraps
  the opaque pointer in `PriceResult`. `PriceResult` holds a raw pointer
  (therefore `!Send + !Sync` automatically — matching the C++ result's
  not-concurrently-usable contract) and frees it in `Drop`.
- `solve_iv` builds `MangoIvConfig` from the `Option` fields (None => 0 =>
  library default) and copies `MangoIvSuccess` into `IvSuccess`.
- Map `MangoError` → `Error` (code → `ErrorKind`, message → `String`).

## Build (rules_rust)

- `MODULE.bazel`: add `bazel_dep(name = "rules_rust", version = "<pinned>")`,
  register a Rust toolchain (edition 2021, pinned `rustc` version). No
  `crate_universe` (zero third-party crates).
- `src/ffi/BUILD.bazel`: `cc_library(name = "mango_c_api", srcs/hdrs, deps =
  [//src/option:american_option, //src/option:iv_solver, ...])`, same copts as
  the existing library.
- `crates/mango-option-sys/BUILD.bazel`: `rust_library` with the `cc_library`
  `:mango_c_api` in `deps` (rules_rust links cc deps directly), plus a
  `rust_test` for `layout.rs`.
- `crates/mango-option/BUILD.bazel`: `rust_library` depending on
  `:mango_option_sys`, plus `rust_test` targets for the integration tests.
- All targets build/test under `bazel build //...` / `bazel test //...`;
  existing C++ targets are unaffected.

## Error Handling

| Source (C++)                              | MangoStatus                | Rust ErrorKind |
|-------------------------------------------|----------------------------|----------------|
| `ValidationError` (bad spot/strike/etc.)  | `MANGO_ERR_VALIDATION`     | `Validation`   |
| IV `ArbitrageViolation`                   | `MANGO_ERR_ARBITRAGE`      | `Arbitrage`    |
| IV `MaxIterationsExceeded`                | `MANGO_ERR_NO_CONVERGENCE` | `NoConvergence`|
| IV `BracketingFailed`                     | `MANGO_ERR_BRACKETING`     | `Bracketing`   |
| `SolverError` / caught C++ exception      | `MANGO_ERR_SOLVER`         | `Solver`       |

No C++ exception or `std::expected` ever crosses the boundary; the diagnostic
string is preserved in `Error::message`.

## Testing

- **ABI layout guard** (`-sys`, `tests/layout.rs`): `size_of`/`align_of`
  assertions for every `repr(C)` struct, matched by `static_assert` in
  `mango_c_api.h`. Bidirectional: drift on either side fails the build.
- **Integration** (`mango-option`, against existing C++ reference values):
  - ATM American put value matches the figure asserted in the existing C++
    `american_option_test` (cross-checked, not invented).
  - **Round-trip**: price at σ → `solve_iv` recovers σ within tolerance.
  - Yield-curve input path (`Rate::Curve`) prices/solves without error and
    differs from the matching flat-rate case as expected.
  - Discrete-dividend input path prices without error and lowers a call /
    raises a put vs. the no-dividend case (sanity direction).
  - `value_at(spot)` at off-spot points and Greeks (`delta`/`gamma`/`theta`)
    return finite values with expected signs (put delta < 0, gamma > 0).
  - Error cases: negative spot → `Validation`; arbitrage-violating market price
    → `Arbitrage`.
- **Regression tests**: per CLAUDE.md, every bug found during review/execution
  gets a regression test.

## Risks / Mitigations

- **ABI drift between header and `-sys`** → bidirectional static/const
  assertions fail the build.
- **Dangling pointers from temporary `Vec`s** → safe layer binds the `Vec`s to
  named locals spanning the FFI call; covered by the array-input tests under
  ASan-enabled `bazel test`.
- **C++ exception escaping** → shim-wide `try/catch` → `MANGO_ERR_SOLVER`.
- **`rules_rust` toolchain bootstrap in CI** → pin versions; verify
  `bazel test //crates/...` plus the untouched C++ suite both pass before PR.

## Future Extensions (not v1)

- Batch IV (`solve_batch`) with an array-in/array-out C function.
- Price-table precomputation + interpolated IV solver bindings.
- Standalone Cargo build (`build.rs` + `cc`) for `cargo add` consumers.
- Explicit grid specs / snapshots for advanced users.
