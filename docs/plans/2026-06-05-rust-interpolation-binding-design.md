# Rust Binding Design (v2: Interpolation Path)

**Date**: 2026-06-05
**Status**: Design
**Goal**: Extend the existing Rust binding to cover the *interpolation path* — the
pre-computed B-spline price surface and the fast interpolated implied-vol solver.

## Problem

The merged v1 binding (PR #430) covers the *FDM path*: per-query American
pricing (value + Greeks) and FDM implied vol (~8 ms/query). The library also
exposes an *interpolation path* that pre-computes a B-spline price surface once
and then answers queries cheaply — interpolated IV at ~3.5 µs/query versus ~8 ms
for FDM, a ~2000× speedup for repeated solves over the same surface. Rust
consumers doing volume IV work (option chains, calibration) need this path.

The Python parity layer (PR #427) already exposes it. The C++ entry points are:

- `make_interpolated_iv_solver(IVSolverFactoryConfig) -> std::expected<AnyInterpIVSolver, ValidationError>`
  — builds a surface and wraps it in a solver with `.solve(IVQuery)` /
  `.solve_batch(...)`.
- `make_price_table(IVSolverFactoryConfig) -> std::expected<AnyPriceTable, ValidationError>`
  — builds the reusable surface, queried via `PricingParams` for
  `price`/`vega`/`delta`/`gamma`/`theta`/`rho`, plus `make_iv_solver(...)` to
  derive a solver from an already-built table without rebuilding the surface.

These lean on the same non-C-ABI features v1 dealt with (`std::expected`,
type-erased pimpl handles, move-only results) **plus** a large nested config
(`IVSolverFactoryConfig`) carrying vectors, two optional sub-structs, and a
`std::variant` backend. The binding cannot be generated mechanically.

## Goals

1. Build an interpolated IV solver from Rust and solve single + batch IV queries
   against it.
2. Build a reusable price table (surface) from Rust and query
   `price`/`vega`/`delta`/`gamma`/`theta`/`rho` via `PricingParams`, plus derive
   a solver from a built table.
3. Faithful config fidelity for the **B-spline** path: the `IVGrid`
   (moneyness/vol/rate), the B-spline `maturity_grid`, the Newton
   `solver_config`, optional **adaptive grid** refinement, and optional
   **discrete dividends** (schedule + multi-K_ref config).
4. Idiomatic Rust surface: callers write zero `unsafe`; errors are `Result`;
   handles are RAII (`Drop`) and `Send + Sync` (surfaces are immutable and the
   C++ guarantees query thread-safety).
5. Reuse v1 FFI types and the existing Bazel `rules_rust` link plumbing; the only
   new C++ surface is the additional `extern "C"` shim functions. No library
   changes.

## Non-Goals (v2)

- **Non-B-spline backends.** `ChebyshevBackend` and `DimensionlessBackend` (the
  other two `std::variant` arms) are out of scope. Modeling a tagged union across
  the C ABI is deferred; B-spline is the documented, headline backend. Adding the
  others later is an additive change (new ABI tag + struct), not a rewrite.
- **Lower-level surface construction** (`PriceTableBuilder<4>::from_vectors` /
  `make_bspline_surface` / raw `BSplinePriceTable` with the 5-arg
  `price(spot,strike,tau,sigma,rate)`). Rust uses the type-erased
  `AnyPriceTable` (which queries via `PricingParams`), matching the Python
  surface.
- **Bounds *accessors*** (`m_min`/`tau_min`/…). The type-erased `AnyPriceTable`
  does not expose them. `price()`/`vega()` mirror the raw C++ signature (return
  `f64`, no implicit validation — out-of-domain queries extrapolate); the
  extrapolation caveat is documented. **Explicit bounds *validation* IS
  exposed**, however: `AnyPriceTable::validate_pricing_params` exists
  (`price_table_factory.hpp:30`), so the binding surfaces it as an opt-in
  `PriceTable::validate(&params) -> Result<()>` rather than silently making
  `price`/`vega` fallible. Callers who want the Python layer's safety call
  `validate()` first.
- **Surface serialization** (Parquet load/save) and `surface_type()` string
  accessor.
- Publishing to crates.io. v2 ships in-tree via Bazel, like v1.

## Approach

Same as v1: **hand-written `extern "C"` shim + raw `-sys` crate + safe wrapper
crate**, guarded by bidirectional ABI layout tests (`static_assert(offsetof)` in
the C header mirrored by `offset_of!` in `-sys`). New types and functions are
added to the existing `src/ffi/mango_c_api.{h,cpp}`, `crates/mango-option-sys`,
and `crates/mango-option`.

Because B-spline is the only backend, the `std::variant` collapses to a single
`maturity_grid` array — **no tagged union crosses the C ABI**. The factory config
becomes a flat struct using the same pointer+length idiom already used for
`tenor_points`/`dividends` in v1, plus two nullable sub-struct pointers for the
optionals (`adaptive`, `discrete_dividends`).

## C ABI (added to `src/ffi/mango_c_api.h` / `.cpp`)

Reused unchanged from v1: `MangoStatus`/`MANGO_ERR_*`, `MangoOptionType`,
`MangoError`, `MangoDividend`, `MangoTenorPoint`, `MangoPricingParams`,
`MangoIvQuery`.

**One shared-struct change**: `MangoIvSuccess` gains a trailing
`int32_t used_rate_approximation` field. The interpolated solver sets
`IVSuccess::used_rate_approximation` when a yield-curve query is collapsed to a
flat rate (`iv_result.hpp:25`, `interpolated_iv_solver.hpp:536`) — dropping it
would lose fidelity exactly on the path v2 adds. It slots into the existing tail
padding (`has_vega` is `int32_t` at offset 32; the new field goes at offset 36),
so **`sizeof(MangoIvSuccess)` stays 40 and no other v1 offset moves**. This
touches v1's struct, its `static_assert`s, the `-sys` mirror, the v1 layout
test, the safe `IvSuccess` (gains `pub used_rate_approximation: bool`), and the
v1 `mango_solve_iv` mapping (sets the new field). It is an additive,
non-reordering change.

New POD config types (all sizes/offsets pinned by `static_assert`):

```c
// InterpolatedIVSolverConfig (Newton config). Fields applied VERBATIM (no
// zero-means-default magic): vega_threshold == 0 is a *legal* value meaning
// "disable the vega pre-check" (interpolated_iv_solver.hpp:55), so zero cannot
// be overloaded as "unset". Raw C callers must fill every field; the safe Rust
// layer always does, because InterpSolverConfig::default() carries the real
// C++ defaults below.
typedef struct {
  uint64_t max_iter;        // 50
  double tolerance;         // 1e-6
  double sigma_min;         // 0.01
  double sigma_max;         // 3.0
  double vega_threshold;    // 1e-4 (0 disables the check)
} MangoInterpSolverConfig;

// AdaptiveGridParams (all 9 fields; passed only when non-null)
typedef struct {
  double   target_iv_error;
  uint64_t max_iter;
  uint64_t max_points_per_dim;
  uint64_t min_moneyness_points;
  uint64_t validation_samples;
  double   refinement_factor;
  uint64_t lhs_seed;
  double   vega_floor;
  double   max_failure_rate;
} MangoAdaptiveGridParams;

// MultiKRefConfig
typedef struct {
  const double* K_refs;     // may be null when n_K_refs == 0 (auto mode)
  uint64_t      n_K_refs;
  int32_t       K_ref_count; // used iff K_refs empty; must be >= 1 (auto: 11)
  double        K_ref_span;
} MangoMultiKRef;

// DiscreteDividendConfig
typedef struct {
  double               maturity;
  const MangoDividend* dividends;   // may be null when n_dividends == 0
  uint64_t             n_dividends;
  MangoMultiKRef       kref_config;
} MangoDiscreteDividendConfig;

// IVSolverFactoryConfig (B-spline backend only)
typedef struct {
  MangoOptionType option_type;
  double          spot;
  double          dividend_yield;
  const double*   moneyness;   uint64_t n_moneyness;  // IVGrid
  const double*   vol;         uint64_t n_vol;
  const double*   rate;        uint64_t n_rate;
  const double*   maturity_grid; uint64_t n_maturity; // BSplineBackend
  MangoInterpSolverConfig solver_config;
  const MangoAdaptiveGridParams*     adaptive;            // null => fixed grid
  const MangoDiscreteDividendConfig* discrete_dividends;  // null => continuous
} MangoIvFactoryConfig;

// Per-query batch result slot (caller-allocated array of length n)
typedef struct {
  int32_t        status;   // MangoStatus: MANGO_OK or an error category
  MangoIvSuccess success;  // valid iff status == MANGO_OK
} MangoIvBatchSlot;

typedef struct MangoInterpIvSolver MangoInterpIvSolver;  // opaque
typedef struct MangoPriceTable     MangoPriceTable;      // opaque
```

Functions:

```c
// Interpolated IV solver
MangoStatus mango_make_interp_iv_solver(const MangoIvFactoryConfig* cfg,
                                        MangoInterpIvSolver** out, MangoError* err);
MangoStatus mango_interp_iv_solve(const MangoInterpIvSolver* s,
                                  const MangoIvQuery* q,
                                  MangoIvSuccess* out, MangoError* err);
MangoStatus mango_interp_iv_solve_batch(const MangoInterpIvSolver* s,
                                        const MangoIvQuery* queries, uint64_t n,
                                        MangoIvBatchSlot* out_slots /*[n]*/,
                                        uint64_t* out_failed_count, MangoError* err);
void mango_interp_iv_solver_free(MangoInterpIvSolver* s);

// Price table (type-erased AnyPriceTable)
MangoStatus mango_make_price_table(const MangoIvFactoryConfig* cfg,
                                   MangoPriceTable** out, MangoError* err);
MangoStatus mango_price_table_validate(const MangoPriceTable* t, const MangoPricingParams* p,
                                       MangoError* err);  // bounds check (opt-in)
double mango_price_table_price(const MangoPriceTable* t, const MangoPricingParams* p);
double mango_price_table_vega (const MangoPriceTable* t, const MangoPricingParams* p);
MangoStatus mango_price_table_delta(const MangoPriceTable* t, const MangoPricingParams* p,
                                    double* out, MangoError* err);
MangoStatus mango_price_table_gamma(const MangoPriceTable* t, const MangoPricingParams* p,
                                    double* out, MangoError* err);
MangoStatus mango_price_table_theta(const MangoPriceTable* t, const MangoPricingParams* p,
                                    double* out, MangoError* err);
MangoStatus mango_price_table_rho  (const MangoPriceTable* t, const MangoPricingParams* p,
                                    double* out, MangoError* err);
MangoOptionType mango_price_table_option_type(const MangoPriceTable* t);
double          mango_price_table_dividend_yield(const MangoPriceTable* t);
MangoStatus mango_price_table_make_iv_solver(const MangoPriceTable* t,
                                             const MangoInterpSolverConfig* cfg /*nullable*/,
                                             MangoInterpIvSolver** out, MangoError* err);
void mango_price_table_free(MangoPriceTable* t);
```

### Shim behavior

- **Config translation.** The shim reads the flat `MangoIvFactoryConfig` into a
  C++ `IVSolverFactoryConfig`: copies the three grid vectors and `maturity_grid`
  into `IVGrid` / `BSplineBackend{maturity_grid}`; sets `backend` to the
  B-spline arm; applies `solver_config` **verbatim** (no zero-means-default);
  sets `adaptive`/`discrete_dividends` only when the corresponding pointer is
  non-null. `K_refs == null` (n=0) maps to auto K_ref selection; in that case a
  `K_ref_count <= 0` is bumped to the C++ default (11) so a zero-initialized
  `MangoMultiKRef` doesn't trip the factory's `K_ref_count >= 1` check
  (`price_table_factory.cpp:149`).
- **`maturity_grid` is not unconditionally required.** The discrete-dividend
  B-spline path branches before consuming `BSplineBackend` and uses
  `DiscreteDividendConfig::maturity` instead (`price_table_factory.cpp:228,332`).
  The shim does **not** pre-reject an empty `maturity_grid`; it lets the C++
  factory return `ValidationError` when the grid is actually needed (continuous
  path) and absent.
- **Pre-validation in the shim** (before calling into C++, matching v1's IV
  pre-validation): reject non-finite `spot`/`dividend_yield`; reject
  `option_type` not in {0,1}; reject any non-finite dividend amount/time in a
  discrete schedule. These map to `MANGO_ERR_VALIDATION`.
- **Error mapping.** Factory `ValidationError` → `MANGO_ERR_VALIDATION` with its
  message. `IVError` → existing `map_iv_error` (Validation / Arbitrage /
  NoConvergence / Bracketing / Solver). `GreekError` → `MANGO_ERR_SOLVER` (or
  Validation for an out-of-domain param) with message. Reuse v1's
  `set_err`/`map_*` helpers.
- **Batch.** `solve_batch` runs the C++ batch (OpenMP). For each result, the
  slot gets `status = MANGO_OK` + the `IVSuccess`, or `status =` the mapped error
  category and a zeroed `IVSuccess`. `*out_failed_count` mirrors
  `BatchIVResult::failed_count`. **Known limitation:** the per-slot result keeps
  only an error *category*, not the full `IVError` message/iterations — callers
  needing detail re-run the single `solve()`. (This matches the Python batch
  shape, which returns coded results.) The function itself returns `MANGO_OK`
  unless the batch call throws/setup fails; per-query failures live in the slots.
- **Greeks / price / vega** call through to `AnyPriceTable`. `price`/`vega`
  return `double` directly (no out-of-band error; extrapolates out of domain).
  Greeks are `std::expected` → `MangoStatus` + out-param.
- **Null-guarding.** Every getter null-checks its handle and returns `nan("")`
  (price/vega) or a validation error (greeks) on null, as v1 does.

### ABI guards

`static_assert(sizeof(...))` and `static_assert(offsetof(...))` for every new
struct in the header (C11 `_Static_assert` / C++ `static_assert`), mirrored
field-for-field by `offset_of!` assertions in
`crates/mango-option-sys/tests/layout.rs`. Exact byte offsets are computed during
implementation (Task: header + layout test land together, the layout test is
written to fail first against wrong offsets).

## Safe Rust API (`crates/mango-option`)

Two new modules, `interp.rs` and `table.rs`, re-exported from `lib.rs`. Reuse v1
`OptionType`, `Rate`, `Dividend`, `PricingParams`, `IvQuery`, `IvSuccess`,
`Error`, `ErrorKind`.

```rust
// ---- interp.rs ----
/// IVGrid axes (S/K moneyness, not log). `Default` mirrors the C++ defaults.
pub struct IvGrid { pub moneyness: Vec<f64>, pub vol: Vec<f64>, pub rate: Vec<f64> }
impl Default for IvGrid { /* {0.7..1.3}, {0.05..0.5}, {0.01..0.10} */ }

/// Newton config for the interpolated solver. `Default` mirrors C++ defaults.
pub struct InterpSolverConfig {
    pub max_iter: usize,        // 50
    pub tolerance: f64,         // 1e-6
    pub sigma_min: f64,         // 0.01
    pub sigma_max: f64,         // 3.0
    pub vega_threshold: f64,    // 1e-4
}
impl Default for InterpSolverConfig { /* … */ }

/// Adaptive grid refinement params. `Default` mirrors C++ defaults.
pub struct AdaptiveGridParams {
    pub target_iv_error: f64,        // 2e-5
    pub max_iter: usize,             // 5
    pub max_points_per_dim: usize,   // 160
    pub min_moneyness_points: usize, // 60
    pub validation_samples: usize,   // 64
    pub refinement_factor: f64,      // 1.3
    pub lhs_seed: u64,               // 42
    pub vega_floor: f64,             // 1e-4
    pub max_failure_rate: f64,       // 0.5
}
impl Default for AdaptiveGridParams { /* … */ }

pub struct MultiKRef { pub k_refs: Vec<f64>, pub k_ref_count: i32, pub k_ref_span: f64 }
impl Default for MultiKRef { /* {empty}, 11, 0.3 */ }

pub struct DiscreteDividendConfig {
    pub maturity: f64,
    pub dividends: Vec<Dividend>,
    pub kref_config: MultiKRef,
}

pub struct FactoryConfig {
    pub option_type: OptionType,
    pub spot: f64,
    pub dividend_yield: f64,
    pub grid: IvGrid,
    pub maturity_grid: Vec<f64>,
    pub solver: InterpSolverConfig,
    pub adaptive: Option<AdaptiveGridParams>,
    pub discrete_dividends: Option<DiscreteDividendConfig>,
}
// No blanket Default: maturity_grid (continuous path) / discrete_dividends
// (segmented path) carry the surface maturity and callers must choose one.

pub struct BatchResult {
    pub results: Vec<Result<IvSuccess, Error>>,
    pub failed: usize,
}

pub struct InterpIvSolver { /* owns *mut sys::MangoInterpIvSolver; Drop frees */ }
impl InterpIvSolver {
    pub fn new(cfg: &FactoryConfig) -> Result<Self, Error>;
    pub fn solve(&self, query: &IvQuery) -> Result<IvSuccess, Error>;
    pub fn solve_batch(&self, queries: &[IvQuery]) -> BatchResult;
}
unsafe impl Send for InterpIvSolver {}
unsafe impl Sync for InterpIvSolver {}

// ---- table.rs ----
pub struct PriceTable { /* owns *mut sys::MangoPriceTable; Drop frees */ }
impl PriceTable {
    pub fn new(cfg: &FactoryConfig) -> Result<Self, Error>;
    /// Opt-in bounds check (out-of-domain => Err). price/vega do NOT call this.
    pub fn validate(&self, params: &PricingParams) -> Result<(), Error>;
    pub fn price(&self, params: &PricingParams) -> f64;   // extrapolates out of domain
    pub fn vega(&self, params: &PricingParams) -> f64;
    pub fn delta(&self, params: &PricingParams) -> Result<f64, Error>;
    pub fn gamma(&self, params: &PricingParams) -> Result<f64, Error>;
    pub fn theta(&self, params: &PricingParams) -> Result<f64, Error>;
    pub fn rho(&self,   params: &PricingParams) -> Result<f64, Error>;
    pub fn option_type(&self) -> OptionType;
    pub fn dividend_yield(&self) -> f64;
    /// Derive a solver from this already-built surface (no rebuild).
    pub fn iv_solver(&self, cfg: Option<&InterpSolverConfig>) -> Result<InterpIvSolver, Error>;
}
unsafe impl Send for PriceTable {}
unsafe impl Sync for PriceTable {}
```

The safe layer owns all the temporary slices it passes as pointers (the `Vec`s
in `FactoryConfig` outlive the FFI call; sub-struct C views are stack locals
populated just before the call). `solve_batch` allocates a `Vec<MangoIvBatchSlot>`
of `queries.len()`, calls the C batch, then maps each slot to
`Result<IvSuccess, Error>` (status `MANGO_OK` → `Ok`, else `Err` with the
per-kind message).

## Error fidelity & documented tradeoffs

1. **Batch loses per-query message detail** (category only). Single `solve()`
   keeps full fidelity. Documented in the rustdoc for `solve_batch`.
2. **`price()`/`vega()` do not bounds-validate** and extrapolate out of domain;
   documented on both methods. Callers wanting the Python layer's safety call
   `PriceTable::validate(&params)` first.
3. **No zero-means-default at the C boundary.** `MangoInterpSolverConfig` is
   applied verbatim (a zero `vega_threshold` legitimately *disables* the vega
   pre-check). The safe `InterpSolverConfig::default()` carries the real C++
   defaults, so Rust callers starting from `Default` get correct behavior; raw C
   callers must fill every field.
4. **Interpolated IV may approximate a yield-curve rate.** The interpolated
   solver collapses a `YieldCurve` query to a flat rate and flags it via
   `used_rate_approximation` (now carried through `MangoIvSuccess` →
   `IvSuccess::used_rate_approximation`). Surfaced, not hidden.

## Testing

**`-sys` layout tests** (`crates/mango-option-sys/tests/layout.rs`): `offset_of!`
for every field of every new struct, plus `size_of` equality with the C
`static_assert`s.

**Safe-crate integration tests** (`crates/mango-option/tests/`):

1. **Build + IV round-trip (continuous).** Build a small fixed-grid B-spline
   solver (PUT). Price a known option with the v1 FDM `price_american` at a
   target σ in-grid, feed that price as `market_price`, recover σ within a
   tolerance (e.g. 1e-3). Contrast with a different σ to ensure the recovered
   value tracks input.
2. **Price table queries.** Build a `PriceTable`, query `price`/`vega` and all
   four Greeks via `PricingParams`; assert finite, signs sane (PUT delta < 0),
   and `price` ≈ FDM price within interpolation tolerance. Also assert
   `validate(&in_domain_params)` is `Ok` and `validate(&out_of_domain_params)`
   (e.g. σ far outside the grid) is `Err(ErrorKind::Validation)`.
3. **`PriceTable::iv_solver`.** Derive a solver from a built table and solve;
   assert it matches a solver built directly from the same config.
4. **Batch solve with a deliberate failure.** A batch of queries where one has
   an arbitrage-violating market price; assert `failed == 1`, the bad slot is
   `Err` with `ErrorKind::Arbitrage` (or the mapped category), the rest `Ok`.
5. **Adaptive grid build.** Build with `adaptive = Some(AdaptiveGridParams{
   target_iv_error: 1e-3, max_iter: 2, .. })` (small, fast) and solve one query
   successfully.
6. **Discrete-dividend build.** Build with `discrete_dividends = Some(...)` (one
   dividend) and explicit `K_refs`; solve and recover a σ. Mirrors the v1 FDM
   discrete-dividend round-trip.
7. **Validation errors.** On the *continuous* path (no `discrete_dividends`),
   an empty `maturity_grid` returns `Err(ErrorKind::Validation)` from `new` (the
   C++ factory rejects it). A non-finite `spot` also returns
   `Err(ErrorKind::Validation)`. (Do **not** assert empty `maturity_grid` fails
   on the discrete path — it legitimately uses `DiscreteDividendConfig::maturity`.)
8. **`Send + Sync` smoke test.** Move a solver into two threads (`std::thread`)
   and solve concurrently; assert both succeed (exercises the immutability
   guarantee).

**Bazel**: new tests wired into `crates/mango-option/BUILD.bazel` and
`crates/mango-option-sys/BUILD.bazel`; reuse the v1 link plumbing (the
`mango_c_api` cc_library + cpuinfo/libstdc++ deps already propagate through the
`-sys` crate). The new shim functions live in the existing `mango_c_api` target,
so no new C++ targets are needed.

## File-by-file change list

- `src/ffi/mango_c_api.h` — new POD structs, opaque handles, 16 new fn decls,
  `static_assert` ABI guards; **+`int32_t used_rate_approximation` appended to
  `MangoIvSuccess`** (offset 36, sizeof stays 40; add an offset assert).
- `src/ffi/mango_c_api.cpp` — config translation, pre-validation, 16 fn impls,
  error mapping (reusing v1 helpers); set `used_rate_approximation` in the
  existing `mango_solve_iv` mapping too.
- `crates/mango-option-sys/src/lib.rs` — `repr(C)` mirrors + `extern "C"` decls;
  add the new `MangoIvSuccess` field.
- `crates/mango-option-sys/tests/layout.rs` — `offset_of!`/`size_of` asserts for
  new structs + the new `MangoIvSuccess` field.
- `crates/mango-option/src/iv.rs` — surface `IvSuccess::used_rate_approximation`.
- `crates/mango-option/src/interp.rs` — `FactoryConfig` family + `InterpIvSolver`.
- `crates/mango-option/src/table.rs` — `PriceTable`.
- `crates/mango-option/src/lib.rs` — module decls + re-exports.
- `crates/mango-option/tests/interpolation.rs` — integration tests above.
- `crates/mango-option/BUILD.bazel`, `crates/mango-option-sys/BUILD.bazel` —
  wire new test sources.
- `docs/RUST_GUIDE.md` — add an interpolation-path section.

## Risks

- **Offset drift**: mitigated by bidirectional ABI guards (build fails on
  mismatch; layout test fails on mismatch).
- **Adaptive-grid build time in tests**: keep `max_iter` small and grids tiny so
  CI stays fast; adaptive runs extra PDE solves per validation sample.
- **Batch error detail loss**: accepted and documented; single-solve path is the
  escape hatch.
