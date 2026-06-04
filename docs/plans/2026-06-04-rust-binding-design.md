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
3. Input fidelity matching each underlying C++ API:
   - **Pricing**: constant rate **or** yield curve; continuous **and** discrete
     dividends (`PricingParams` supports all of these).
   - **IV**: constant rate **or** yield curve; continuous dividend yield.
     Discrete dividends are **not** supported — `IVQuery : OptionSpec` has no
     discrete-dividend field and `IVSolver::objective_function` prices with none
     (see `src/option/iv_solver.cpp`). The Rust IV API therefore rejects queries
     carrying discrete dividends rather than silently ignoring them.
4. Idiomatic Rust surface: callers write zero `unsafe`; errors are `Result`.
5. Build and test in-tree via Bazel `rules_rust`; leave existing C++ targets
   untouched.

## Non-Goals (v1)

- Batch solving (`solve_batch`), price tables, interpolated IV solver.
- Discrete-dividend implied volatility (unsupported by the FDM IV API; needs the
  interpolated solver, which is itself out of scope).
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

All enums and counts use fixed-width types (`int32_t`, `uint64_t`) so the ABI is
stable regardless of platform enum/`size_t` width, and every field offset is
pinned by `static_assert(offsetof(...))` (see ABI guard below).

```c
typedef int32_t MangoOptionType;  // 0 = call, 1 = put
#define MANGO_CALL 0
#define MANGO_PUT  1

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
  uint64_t n_dividends;
  MangoOptionType option_type;
} MangoPricingParams;

// IV has no discrete-dividend field: the FDM IV API does not support them.
typedef struct {
  double spot;
  double strike;
  double maturity;
  double dividend_yield;            // continuous yield only
  double market_price;
  double rate_const;
  const MangoTenorPoint* tenor_points;
  uint64_t n_tenor_points;
  MangoOptionType option_type;
} MangoIvQuery;
```

(`n_tenor_points` is also `uint64_t` in `MangoPricingParams`.) The exact field
order/padding is the ABI contract; pinned by per-field `static_assert(offsetof
…))` in the header and mirrored by per-field offset asserts in `-sys` (see ABI
guard).

### Pricing — opaque handle (preserves `value_at`/Greeks without re-solving)

The C++ accessors are **not** `noexcept` — `value`/`delta`/`gamma`/`theta`/
`value_at` trigger lazy cubic-spline / `CenteredDifference` construction and
allocate (`gamma` allocates vectors; see `american_option_result.cpp`). Letting
any of them throw across the C ABI is undefined behaviour. Two consequences:

- `mango_price_american` **eagerly computes** `value`, `delta`, `gamma`, `theta`
  (the at-spot Greeks) inside its `try/catch` and stores them in the handle as
  plain doubles. The corresponding accessors are then pure, `noexcept` getters.
  If any eager computation throws, the whole price call fails with a mapped
  error — no half-built handle escapes.
- `value_at(spot)` (arbitrary spot) still drives the lazy spline, so it stays
  **fallible**: a status-returning function wrapped in `try/catch`.

```c
typedef struct MangoAmericanResult MangoAmericanResult; // opaque

MangoStatus mango_price_american(const MangoPricingParams* params,
                                 MangoAmericanResult** out_result,
                                 MangoError* out_err);

// noexcept getters of values precomputed at solve time:
double mango_american_value(const MangoAmericanResult*);
double mango_american_delta(const MangoAmericanResult*);
double mango_american_gamma(const MangoAmericanResult*);
double mango_american_theta(const MangoAmericanResult*);

// fallible (lazy spline): writes *out on MANGO_OK.
MangoStatus mango_american_value_at(const MangoAmericanResult*, double spot,
                                    double* out, MangoError* out_err);

void mango_american_result_free(MangoAmericanResult*);
```

`mango_price_american` heap-allocates a struct owning the move-only
`AmericanOptionResult` plus the four precomputed at-spot quantities; `..._free`
deletes it. Every shim body is wrapped in `try/catch (...) -> MANGO_ERR_SOLVER`.

### IV — POD result

```c
typedef struct {
  double implied_vol;
  size_t iterations;
  double final_error;
  double vega;        // valid iff has_vega != 0
  int    has_vega;
} MangoIvSuccess;

// Maps onto IVSolverConfig.root_config (RootFindingConfig). All-zero means
// "use library defaults". IVSolverConfig has NO target_price_error field;
// Brent uses root_config.max_iter and root_config.brent_tol_abs.
typedef struct {
  int32_t max_iter;     // 0 => default; negative rejected as validation error
  double  brent_tol_abs;// 0 => default
} MangoIvConfig;

MangoStatus mango_solve_iv(const MangoIvQuery* query,
                           const MangoIvConfig* config, // nullable => defaults
                           MangoIvSuccess* out_success,
                           MangoError* out_err);
```

### Shim implementation notes

- **Input validation at the boundary** (before touching C++): reject null array
  pointers with non-zero counts, non-finite doubles, and — for yield curves —
  enforce what `YieldCurve::from_points` requires (a `tenor=0, log_discount=0`
  point present, strictly increasing tenors). `validate_option_spec` does **not**
  re-validate a curve after construction, so the shim must, mapping any
  `from_points` failure to `MANGO_ERR_VALIDATION`. Reject negative
  `MangoIvConfig.max_iter` before converting to `size_t`.
- Rebuild `RateSpec`: `n_tenor_points == 0` → `double rate_const`; else
  construct a `YieldCurve` from the validated `(tenor, log_discount)` points.
- Rebuild discrete dividends into `std::vector<Dividend>` (pricing only).
- Map `OptionType` ↔ `MangoOptionType`.
- **Pricing path** (to preserve validation errors): call
  `AmericanOptionSolver::create(params)` directly — **not** the
  `solve_american_option` convenience wrapper, which collapses create-time
  `ValidationError` into `SolverError::InvalidConfiguration` (see
  `american_option.cpp`). Map the `create()` `ValidationError`, then call
  `solve()` and map its `SolverError`. Auto grid (default) is used.
- **IV path**: build `IVSolverConfig` by overriding only the non-zero
  `root_config` fields (`max_iter`, `brent_tol_abs`); leave `grid` at its
  default (`GridAccuracyParams{}`). Then `IVSolver::solve(query)`.
- Error mapping: `ValidationError` → `MANGO_ERR_VALIDATION`; IV
  `ArbitrageViolation`/`BracketingFailed`/`MaxIterationsExceeded` →
  `MANGO_ERR_ARBITRAGE`/`MANGO_ERR_BRACKETING`/`MANGO_ERR_NO_CONVERGENCE`;
  `SolverError` and any caught exception → `MANGO_ERR_SOLVER`. The C++ error
  types are mostly enum code + numeric fields, not strings, so the shim
  **synthesizes** a human-readable `message` (error code name plus available
  diagnostics such as IV iterations / `last_vol`); it is best-effort and
  truncated to fit the 256-byte buffer.

## Component: `-sys` crate (`mango-option-sys`)

- `#[repr(C)]` mirrors of every struct above (enums/counts as fixed-width
  `i32`/`u64`), plus `extern "C"` blocks declaring all functions. No safe API,
  no logic.
- `tests/layout.rs`: per-struct `size_of`/`align_of` **and per-field
  `offset_of!`** assertions (e.g. `assert_eq!(offset_of!(MangoPricingParams,
  rate_const), 40)`), with the same offsets pinned by `static_assert(offsetof(
  MangoPricingParams, rate_const) == 40)` in `mango_c_api.h`. Size/align alone
  miss same-size field reorderings; offsets catch them. Drift on either side
  breaks the build.

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
    pub fn value(&self) -> f64;                       // precomputed getter
    pub fn delta(&self) -> f64;                       // precomputed getter
    pub fn gamma(&self) -> f64;                       // precomputed getter
    pub fn theta(&self) -> f64;                       // precomputed getter
    pub fn value_at(&self, spot: f64) -> Result<f64, Error>; // lazy spline -> fallible
}
pub fn price_american(params: &PricingParams) -> Result<PriceResult, Error>;

pub struct IvQuery { pub spec: OptionSpec, pub market_price: f64 }
#[derive(Default)]
pub struct IvConfig {
    pub max_iter: Option<u32>,       // -> root_config.max_iter
    pub brent_tol_abs: Option<f64>,  // -> root_config.brent_tol_abs
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
  the opaque pointer in `PriceResult`. `PriceResult` holds a raw pointer, so it
  is `!Send + !Sync` automatically — which is also *semantically* correct: even
  though the precomputed getters are pure, `value_at` drives the
  `AmericanOptionResult`'s lazy, `mutable` spline/operator caches, so a single
  handle is not safe for concurrent use. `Drop` calls
  `mango_american_result_free`.
- `solve_iv` first rejects any query whose `spec.discrete_dividends` is non-empty
  with `ErrorKind::Validation` (FDM IV cannot honour them). It then builds
  `MangoIvConfig` from the `Option` fields (None => 0 => library default) and
  copies `MangoIvSuccess` into `IvSuccess`.
- Map `MangoError` → `Error` (code → `ErrorKind`, synthesized message →
  `String`).

## Build (rules_rust)

- `MODULE.bazel`: add `bazel_dep(name = "rules_rust", version = "<pinned>")`,
  register a Rust toolchain (edition 2021, pinned `rustc` version). No
  `crate_universe` (zero third-party crates).
- `src/ffi/BUILD.bazel`: `cc_library(name = "mango_c_api", srcs/hdrs, deps =
  [//src/option:american_option, //src/option:iv_solver, ...])`, same copts as
  the existing library.
- `crates/mango-option-sys/BUILD.bazel`: `rust_library` that links the
  `cc_library` `:mango_c_api`. In current `rules_rust`, native (`cc_library`)
  dependencies of a `rust_library` belong in **`link_deps`** (or `cc_deps`,
  depending on the pinned version) — *not* in `deps`, which is for Rust crates.
  The plan resolves the exact attribute against the pinned `rules_rust`
  version's docs. Plus a `rust_test` for `layout.rs`.
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

For pricing, the `Validation` vs `Solver` split only works because the shim
calls `AmericanOptionSolver::create()`/`solve()` directly rather than the
convenience wrapper (see shim notes). No C++ exception or `std::expected` ever
crosses the boundary; every accessor and solve is wrapped in `try/catch`. The
`Error::message` is a best-effort synthesized string (the C++ error types carry
codes/numbers, not messages) and may be truncated.

## Testing

- **ABI layout guard** (`-sys`, `tests/layout.rs`): per-struct
  `size_of`/`align_of` **and per-field `offset_of!`** assertions, mirrored by
  `static_assert(sizeof/alignof/offsetof …)` in `mango_c_api.h`. Bidirectional:
  drift (including same-size field reorderings) on either side fails the build.
- **Integration** (`mango-option`, against existing C++ reference values):
  - ATM American put value within the **same loose tolerance the existing C++
    test uses** (`tests/american_option_test.cc` asserts ≈ `6.35 ± 0.5`, not a
    tight golden value); do not invent a tighter number.
  - **Round-trip**: price at σ → `solve_iv` recovers σ within tolerance.
  - Yield-curve input path (`Rate::Curve`) prices/solves without error and
    differs from the matching flat-rate case as expected; an invalid curve
    (missing `t=0` point / non-increasing tenors) → `Validation`.
  - Discrete-dividend **pricing** path prices without error and moves the value
    in the expected direction vs. the no-dividend case. A `solve_iv` query with
    non-empty discrete dividends → `Validation` (not silently ignored).
  - `value_at(spot)` at off-spot points returns `Ok(finite)`; Greeks
    (`delta`/`gamma`/`theta`) are finite with expected signs (put delta < 0,
    gamma > 0).
  - Error cases: negative spot → `Validation`; arbitrage-violating market price
    → `Arbitrage`.
- **Regression tests**: per CLAUDE.md, every bug found during review/execution
  gets a regression test.

## Risks / Mitigations

- **ABI drift between header and `-sys`** → bidirectional `offsetof`/`offset_of!`
  + size/align assertions fail the build (catches field reorderings too).
- **Dangling pointers from temporary `Vec`s** → safe layer binds the `Vec`s to
  named locals spanning the FFI call; covered by the array-input tests under
  ASan-enabled `bazel test`.
- **C++ exception escaping** → every shim body (solve *and* every fallible
  accessor) wrapped in `try/catch` → mapped status; at-spot Greeks precomputed
  so their accessors are `noexcept` getters.
- **Throwing accessors crossing FFI** → at-spot value/Greeks computed eagerly
  inside `mango_price_american`; only `value_at` stays fallible.
- **`rules_rust` toolchain bootstrap in CI** → pin versions; verify
  `bazel test //crates/...` plus the untouched C++ suite both pass before PR.

## Future Extensions (not v1)

- Batch IV (`solve_batch`) with an array-in/array-out C function.
- Price-table precomputation + interpolated IV solver bindings (would also
  unlock discrete-dividend IV).
- Standalone Cargo build (`build.rs` + `cc`) for `cargo add` consumers.
- Explicit grid specs / snapshots for advanced users.

## Design Review Resolution (2026-06-04, Codex)

A pre-implementation Codex review (read-only, against the real C++ sources)
surfaced mismatches between the first draft and the actual API. All findings
were folded in:

- **IV discrete dividends** — `IVQuery` has no such field; v1 IV rejects
  discrete-dividend queries instead of silently ignoring them.
- **IV config** — `IVSolverConfig` has no `target_price_error`; exposed
  `root_config.max_iter` and `root_config.brent_tol_abs` instead.
- **Pricing errors** — shim calls `AmericanOptionSolver::create()`/`solve()`
  (not the convenience wrapper, which loses `ValidationError`).
- **Throwing accessors** — at-spot value/Greeks precomputed eagerly (noexcept
  getters); `value_at` made fallible; all bodies `try/catch`-wrapped.
- **rules_rust linkage** — native `cc_library` goes in `link_deps`, not `deps`.
- **ABI guard** — added per-field `offsetof`/`offset_of!` checks (size/align
  alone miss reorderings); fixed-width enum/count types.
- **Yield-curve validation** — shim validates curve arrays at the boundary
  (`t=0` point, increasing tenors) and maps failures to `Validation`.
- **Reference values** — integration tests reuse the existing loose ATM
  tolerance (`6.35 ± 0.5`) rather than an invented golden value.
- **Minor** — message strings are synthesized/truncatable; `!Send/!Sync`
  justified by the lazy mutable caches; negative `max_iter` rejected.
