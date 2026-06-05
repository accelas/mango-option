# Rust Interpolation-Path Binding Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development
> to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Add a safe, idiomatic Rust binding for the mango-option interpolation
path — the interpolated IV solver (`make_interpolated_iv_solver`) and the
reusable B-spline price table (`make_price_table` / `AnyPriceTable`).

**Architecture:** Extends the v1 binding (PR #430). New `extern "C"` shim
functions in `src/ffi/mango_c_api.{h,cpp}` over a flat factory-config struct
(B-spline backend only — no `std::variant` across the ABI), raw mirrors in
`crates/mango-option-sys`, and a safe wrapper in `crates/mango-option`
(`interp.rs`, `table.rs`). Opaque RAII handles (`Drop`), `Send + Sync` (immutable
surfaces). Reuses v1 marshalling helpers and Bazel link plumbing.

**Tech Stack:** C++23, Rust edition 2021, Bazel `rules_rust`, GoogleTest not used
here (Rust `#[test]` + `-sys` layout tests).

**Spec:** `docs/plans/2026-06-05-rust-interpolation-binding-design.md`

**Reference patterns (read these first, do not re-derive):**
- `src/ffi/mango_c_api.h` / `.cpp` — existing structs, `set_err`,
  `validate_option_type`, `build_rate`, `build_dividends`, `map_iv_error`,
  `format_iv_error`, `format_validation_error`, opaque-handle + `new`/`free`.
- `crates/mango-option-sys/src/lib.rs` + `tests/layout.rs` — `repr(C)` mirrors,
  `offset_of!`/`size_of` asserts.
- `crates/mango-option/src/{pricing,iv,error,types}.rs` — marshalling helpers
  (`tenor_array`, `dividend_array`, `ptr_or_null`, `div_ptr_or_null`,
  `blank_error`), `Error::from_c`, RAII `Drop`, `OptionType::to_c`.
- C++ entry points: `src/option/interpolated_iv_solver.hpp`
  (`make_interpolated_iv_solver`, `AnyInterpIVSolver::{solve,solve_batch}`,
  `IVSolverFactoryConfig`, `IVGrid`, `BSplineBackend`, `InterpolatedIVSolverConfig`,
  `AdaptiveGridParams`, `DiscreteDividendConfig`), `src/option/price_table_factory.hpp`
  (`make_price_table`, `AnyPriceTable`), `src/option/grid_spec_types.hpp`
  (`MultiKRefConfig`), `src/option/table/greek_types.hpp` (`GreekError`).

**Computed ABI offsets** (x86-64 SysV, 8-byte alignment; pin these exactly):

| Struct | size | field offsets |
|---|---|---|
| `MangoIvSuccess` (modified) | 40 | …`has_vega`@32, `used_rate_approximation`@36 |
| `MangoInterpSolverConfig` | 40 | `max_iter`@0, `tolerance`@8, `sigma_min`@16, `sigma_max`@24, `vega_threshold`@32 |
| `MangoAdaptiveGridParams` | 72 | `target_iv_error`@0, `max_iter`@8, `max_points_per_dim`@16, `min_moneyness_points`@24, `validation_samples`@32, `refinement_factor`@40, `lhs_seed`@48, `vega_floor`@56, `max_failure_rate`@64 |
| `MangoMultiKRef` | 32 | `K_refs`@0, `n_K_refs`@8, `K_ref_count`@16, `K_ref_span`@24 |
| `MangoDiscreteDividendConfig` | 56 | `maturity`@0, `dividends`@8, `n_dividends`@16, `kref_config`@24 |
| `MangoIvFactoryConfig` | 144 | `option_type`@0, `spot`@8, `dividend_yield`@16, `moneyness`@24, `n_moneyness`@32, `vol`@40, `n_vol`@48, `rate`@56, `n_rate`@64, `maturity_grid`@72, `n_maturity`@80, `solver_config`@88, `adaptive`@128, `discrete_dividends`@136 |
| `MangoIvBatchSlot` | 48 | `status`@0, `success`@8 |

---

## Task 1: Extend `MangoIvSuccess` with `used_rate_approximation`

The interpolated solver sets `IVSuccess::used_rate_approximation`; the field
slots into existing tail padding so `sizeof` stays 40 and no v1 offset moves.

**Files:**
- Modify: `src/ffi/mango_c_api.h` (struct + static_asserts)
- Modify: `src/ffi/mango_c_api.cpp` (add `fill_iv_success` helper, use in `mango_solve_iv`)
- Modify: `crates/mango-option-sys/src/lib.rs`
- Modify: `crates/mango-option-sys/tests/layout.rs`
- Modify: `crates/mango-option/src/iv.rs`

- [ ] **Step 1: Update the `-sys` layout test first (must fail to compile/run)**

In `crates/mango-option-sys/tests/layout.rs`, extend `small_struct_layouts`:

```rust
    assert_eq!(size_of::<MangoIvSuccess>(), 40);
    assert_eq!(offset_of!(MangoIvSuccess, has_vega), 32);
    assert_eq!(offset_of!(MangoIvSuccess, used_rate_approximation), 36);
```

- [ ] **Step 2: Run the layout test — expect a COMPILE error**

Run: `bazel test //crates/mango-option-sys:layout_test`
Expected: FAIL — `no field used_rate_approximation on MangoIvSuccess`.

- [ ] **Step 3: Add the field to the `-sys` struct**

In `crates/mango-option-sys/src/lib.rs`, `MangoIvSuccess`:

```rust
#[repr(C)]
pub struct MangoIvSuccess {
    pub implied_vol: f64,
    pub iterations: u64,
    pub final_error: f64,
    pub vega: f64,
    pub has_vega: i32,
    pub used_rate_approximation: i32,
}
```

- [ ] **Step 4: Add the field + static_asserts to the C header**

In `src/ffi/mango_c_api.h`, `MangoIvSuccess`:

```c
typedef struct {
  double implied_vol;
  uint64_t iterations;
  double final_error;
  double vega;
  int32_t has_vega;
  int32_t used_rate_approximation;
} MangoIvSuccess;
```

Update BOTH the `__cplusplus` and `_Static_assert` guard blocks: keep
`sizeof(MangoIvSuccess) == 40` and `offsetof(..., has_vega) == 32`, and ADD
`offsetof(MangoIvSuccess, used_rate_approximation) == 36`.

- [ ] **Step 5: Add `fill_iv_success` helper and use it in `mango_solve_iv`**

In `src/ffi/mango_c_api.cpp`, add to the anonymous namespace:

```cpp
void fill_iv_success(MangoIvSuccess* out, const mango::IVSuccess& s) {
  out->implied_vol = s.implied_vol;
  out->iterations = static_cast<uint64_t>(s.iterations);
  out->final_error = s.final_error;
  out->vega = s.vega.value_or(0.0);
  out->has_vega = s.vega.has_value() ? 1 : 0;
  out->used_rate_approximation = s.used_rate_approximation ? 1 : 0;
}
```

Replace the five `out_success->… = s.…;` lines in `mango_solve_iv` with:
`fill_iv_success(out_success, s);`

- [ ] **Step 6: Surface the field on the safe `IvSuccess`**

In `crates/mango-option/src/iv.rs`, add to `IvSuccess`:

```rust
#[derive(Debug)]
pub struct IvSuccess {
    pub implied_vol: f64,
    pub iterations: usize,
    pub final_error: f64,
    pub vega: Option<f64>,
    pub used_rate_approximation: bool,
}
```

And in `solve_iv`'s success arm, add the field:

```rust
        Ok(IvSuccess {
            implied_vol: out.implied_vol,
            iterations: out.iterations as usize,
            final_error: out.final_error,
            vega: if out.has_vega != 0 { Some(out.vega) } else { None },
            used_rate_approximation: out.used_rate_approximation != 0,
        })
```

- [ ] **Step 7: Build the shim + run all v1 tests**

Run: `bazel test //crates/mango-option-sys:layout_test //crates/mango-option:integration_test`
Expected: PASS (FDM path still green; new offset asserted).

- [ ] **Step 8: Commit**

```bash
git add src/ffi/mango_c_api.h src/ffi/mango_c_api.cpp \
        crates/mango-option-sys/src/lib.rs crates/mango-option-sys/tests/layout.rs \
        crates/mango-option/src/iv.rs
git commit -m "Carry used_rate_approximation through MangoIvSuccess"
```

---

## Task 2: Add interpolation C structs, handles, and decls (header + `-sys` + layout test)

ABI types land together with their bidirectional guards, but NO shim
implementation yet (the `.cpp` library still compiles — undefined `extern "C"`
decls are fine until a consumer links them; the layout test references only types).

**Files:**
- Modify: `src/ffi/mango_c_api.h`
- Modify: `crates/mango-option-sys/src/lib.rs`
- Modify: `crates/mango-option-sys/tests/layout.rs`

- [ ] **Step 1: Write the layout asserts first**

Append to `crates/mango-option-sys/tests/layout.rs`:

```rust
#[test]
fn interp_solver_config_layout() {
    assert_eq!(size_of::<MangoInterpSolverConfig>(), 40);
    assert_eq!(offset_of!(MangoInterpSolverConfig, max_iter), 0);
    assert_eq!(offset_of!(MangoInterpSolverConfig, tolerance), 8);
    assert_eq!(offset_of!(MangoInterpSolverConfig, sigma_min), 16);
    assert_eq!(offset_of!(MangoInterpSolverConfig, sigma_max), 24);
    assert_eq!(offset_of!(MangoInterpSolverConfig, vega_threshold), 32);
}

#[test]
fn adaptive_grid_params_layout() {
    assert_eq!(size_of::<MangoAdaptiveGridParams>(), 72);
    assert_eq!(offset_of!(MangoAdaptiveGridParams, target_iv_error), 0);
    assert_eq!(offset_of!(MangoAdaptiveGridParams, max_iter), 8);
    assert_eq!(offset_of!(MangoAdaptiveGridParams, max_points_per_dim), 16);
    assert_eq!(offset_of!(MangoAdaptiveGridParams, min_moneyness_points), 24);
    assert_eq!(offset_of!(MangoAdaptiveGridParams, validation_samples), 32);
    assert_eq!(offset_of!(MangoAdaptiveGridParams, refinement_factor), 40);
    assert_eq!(offset_of!(MangoAdaptiveGridParams, lhs_seed), 48);
    assert_eq!(offset_of!(MangoAdaptiveGridParams, vega_floor), 56);
    assert_eq!(offset_of!(MangoAdaptiveGridParams, max_failure_rate), 64);
}

#[test]
fn multi_kref_layout() {
    assert_eq!(size_of::<MangoMultiKRef>(), 32);
    assert_eq!(offset_of!(MangoMultiKRef, K_refs), 0);
    assert_eq!(offset_of!(MangoMultiKRef, n_K_refs), 8);
    assert_eq!(offset_of!(MangoMultiKRef, K_ref_count), 16);
    assert_eq!(offset_of!(MangoMultiKRef, K_ref_span), 24);
}

#[test]
fn discrete_dividend_config_layout() {
    assert_eq!(size_of::<MangoDiscreteDividendConfig>(), 56);
    assert_eq!(offset_of!(MangoDiscreteDividendConfig, maturity), 0);
    assert_eq!(offset_of!(MangoDiscreteDividendConfig, dividends), 8);
    assert_eq!(offset_of!(MangoDiscreteDividendConfig, n_dividends), 16);
    assert_eq!(offset_of!(MangoDiscreteDividendConfig, kref_config), 24);
}

#[test]
fn iv_factory_config_layout() {
    assert_eq!(size_of::<MangoIvFactoryConfig>(), 144);
    assert_eq!(offset_of!(MangoIvFactoryConfig, option_type), 0);
    assert_eq!(offset_of!(MangoIvFactoryConfig, spot), 8);
    assert_eq!(offset_of!(MangoIvFactoryConfig, dividend_yield), 16);
    assert_eq!(offset_of!(MangoIvFactoryConfig, moneyness), 24);
    assert_eq!(offset_of!(MangoIvFactoryConfig, n_moneyness), 32);
    assert_eq!(offset_of!(MangoIvFactoryConfig, vol), 40);
    assert_eq!(offset_of!(MangoIvFactoryConfig, n_vol), 48);
    assert_eq!(offset_of!(MangoIvFactoryConfig, rate), 56);
    assert_eq!(offset_of!(MangoIvFactoryConfig, n_rate), 64);
    assert_eq!(offset_of!(MangoIvFactoryConfig, maturity_grid), 72);
    assert_eq!(offset_of!(MangoIvFactoryConfig, n_maturity), 80);
    assert_eq!(offset_of!(MangoIvFactoryConfig, solver_config), 88);
    assert_eq!(offset_of!(MangoIvFactoryConfig, adaptive), 128);
    assert_eq!(offset_of!(MangoIvFactoryConfig, discrete_dividends), 136);
}

#[test]
fn iv_batch_slot_layout() {
    assert_eq!(size_of::<MangoIvBatchSlot>(), 48);
    assert_eq!(offset_of!(MangoIvBatchSlot, status), 0);
    assert_eq!(offset_of!(MangoIvBatchSlot, success), 8);
}
```

- [ ] **Step 2: Run — expect compile failure (types undefined)**

Run: `bazel test //crates/mango-option-sys:layout_test`
Expected: FAIL — unknown types `MangoInterpSolverConfig`, etc.

- [ ] **Step 3: Add the C structs + handles + decls to the header**

In `src/ffi/mango_c_api.h`, before the ABI-guard block, add the structs from the
spec's "C ABI" section (`MangoInterpSolverConfig`, `MangoAdaptiveGridParams`,
`MangoMultiKRef`, `MangoDiscreteDividendConfig`, `MangoIvFactoryConfig`,
`MangoIvBatchSlot`, opaque `MangoInterpIvSolver`/`MangoPriceTable`) and the 16
function declarations. Use `uint64_t` for `size_t`-typed fields. Then add, to
BOTH guard blocks, `static_assert`/`_Static_assert` for every size + offset in
the "Computed ABI offsets" table.

- [ ] **Step 4: Mirror in `-sys` `lib.rs`**

In `crates/mango-option-sys/src/lib.rs`, add `#[repr(C)]` structs mirroring the
header (derive `Clone, Copy` on the POD config structs;
`MangoInterpIvSolver`/`MangoPriceTable` as opaque `_private: [u8; 0]`), and the
16 `extern "C"` function declarations. Field names/types EXACTLY match the
header. Example for the factory config:

```rust
#[repr(C)]
#[derive(Clone, Copy)]
pub struct MangoIvFactoryConfig {
    pub option_type: MangoOptionType,
    pub spot: f64,
    pub dividend_yield: f64,
    pub moneyness: *const f64,
    pub n_moneyness: u64,
    pub vol: *const f64,
    pub n_vol: u64,
    pub rate: *const f64,
    pub n_rate: u64,
    pub maturity_grid: *const f64,
    pub n_maturity: u64,
    pub solver_config: MangoInterpSolverConfig,
    pub adaptive: *const MangoAdaptiveGridParams,
    pub discrete_dividends: *const MangoDiscreteDividendConfig,
}
```

Declarations (full set):

```rust
extern "C" {
    pub fn mango_make_interp_iv_solver(
        cfg: *const MangoIvFactoryConfig,
        out: *mut *mut MangoInterpIvSolver,
        err: *mut MangoError,
    ) -> MangoStatus;
    pub fn mango_interp_iv_solve(
        s: *const MangoInterpIvSolver,
        q: *const MangoIvQuery,
        out: *mut MangoIvSuccess,
        err: *mut MangoError,
    ) -> MangoStatus;
    pub fn mango_interp_iv_solve_batch(
        s: *const MangoInterpIvSolver,
        queries: *const MangoIvQuery,
        n: u64,
        out_slots: *mut MangoIvBatchSlot,
        out_failed_count: *mut u64,
        err: *mut MangoError,
    ) -> MangoStatus;
    pub fn mango_interp_iv_solver_free(s: *mut MangoInterpIvSolver);

    pub fn mango_make_price_table(
        cfg: *const MangoIvFactoryConfig,
        out: *mut *mut MangoPriceTable,
        err: *mut MangoError,
    ) -> MangoStatus;
    pub fn mango_price_table_validate(
        t: *const MangoPriceTable,
        p: *const MangoPricingParams,
        err: *mut MangoError,
    ) -> MangoStatus;
    pub fn mango_price_table_price(t: *const MangoPriceTable, p: *const MangoPricingParams) -> f64;
    pub fn mango_price_table_vega(t: *const MangoPriceTable, p: *const MangoPricingParams) -> f64;
    pub fn mango_price_table_delta(t: *const MangoPriceTable, p: *const MangoPricingParams, out: *mut f64, err: *mut MangoError) -> MangoStatus;
    pub fn mango_price_table_gamma(t: *const MangoPriceTable, p: *const MangoPricingParams, out: *mut f64, err: *mut MangoError) -> MangoStatus;
    pub fn mango_price_table_theta(t: *const MangoPriceTable, p: *const MangoPricingParams, out: *mut f64, err: *mut MangoError) -> MangoStatus;
    pub fn mango_price_table_rho(t: *const MangoPriceTable, p: *const MangoPricingParams, out: *mut f64, err: *mut MangoError) -> MangoStatus;
    pub fn mango_price_table_option_type(t: *const MangoPriceTable) -> MangoOptionType;
    pub fn mango_price_table_dividend_yield(t: *const MangoPriceTable) -> f64;
    pub fn mango_price_table_make_iv_solver(
        t: *const MangoPriceTable,
        cfg: *const MangoInterpSolverConfig,
        out: *mut *mut MangoInterpIvSolver,
        err: *mut MangoError,
    ) -> MangoStatus;
    pub fn mango_price_table_free(t: *mut MangoPriceTable);
}
```

> Note: `K_refs`/`K_ref_count`/`n_K_refs` keep their C capitalization; add
> `#![allow(non_snake_case)]` is unnecessary because these are struct fields, not
> idents that trip `non_snake_case` for fns — but if the linter warns, allow it
> at the struct or crate level (`#![allow(non_snake_case)]` already paired with
> the crate's existing `#![allow(non_camel_case_types)]`).

- [ ] **Step 5: Build header static_asserts + run layout test**

Run: `bazel build //src/ffi:mango_c_api && bazel test //crates/mango-option-sys:layout_test`
Expected: PASS — header compiles (static_asserts hold), all layout tests green.
If a `static_assert` fires, the offset table is the source of truth; fix the
struct, not the assert.

- [ ] **Step 6: Commit**

```bash
git add src/ffi/mango_c_api.h crates/mango-option-sys/src/lib.rs crates/mango-option-sys/tests/layout.rs
git commit -m "Add interpolation-path C ABI structs and declarations"
```

---

## Task 3: Implement the interpolated IV solver shim

**Files:**
- Modify: `src/ffi/mango_c_api.cpp`
- Modify: `src/ffi/BUILD.bazel` (add interpolation dep)

- [ ] **Step 1: Add includes + dep**

In `mango_c_api.cpp` add:
```cpp
#include "mango/option/interpolated_iv_solver.hpp"
#include "mango/option/price_table_factory.hpp"
```
In `src/ffi/BUILD.bazel`, add to `mango_c_api`'s `deps`:
```python
        "//src/option:interpolated_iv_solver",
        "//src/option:price_table_factory",
```

- [ ] **Step 2: Add config/query translation helpers (anonymous namespace)**

```cpp
std::vector<double> to_vec(const double* p, uint64_t n) {
  if (n == 0 || p == nullptr) return {};
  return std::vector<double>(p, p + n);
}

MangoStatus map_greek_error(mango::GreekError e) {
  return e == mango::GreekError::OutOfDomain ? MANGO_ERR_VALIDATION
                                             : MANGO_ERR_SOLVER;
}
const char* greek_error_msg(mango::GreekError e) {
  return e == mango::GreekError::OutOfDomain
             ? "greek query point outside surface domain"
             : "greek numerical computation failed";
}

// Build a C++ IVQuery from the C struct (rate + dividends + option type).
bool build_iv_query(const MangoIvQuery* q, mango::IVQuery& out, MangoError* err) {
  mango::OptionSpec spec;
  spec.spot = q->spot; spec.strike = q->strike;
  spec.maturity = q->maturity; spec.dividend_yield = q->dividend_yield;
  if (!validate_option_type(q->option_type, err, spec.option_type)) return false;
  if (!build_rate(q->rate_const, q->tenor_points, q->n_tenor_points, spec.rate, err))
    return false;
  out = mango::IVQuery(spec, q->market_price);
  if (!build_dividends(q->dividends, q->n_dividends, out.discrete_dividends, err))
    return false;
  return true;
}

// Translate the flat factory config into the C++ IVSolverFactoryConfig.
bool build_factory_config(const MangoIvFactoryConfig* c,
                          mango::IVSolverFactoryConfig& out, MangoError* err) {
  if (!validate_option_type(c->option_type, err, out.option_type)) return false;
  if (!std::isfinite(c->spot)) {
    set_err(err, MANGO_ERR_VALIDATION, "spot must be finite"); return false;
  }
  if (!std::isfinite(c->dividend_yield)) {
    set_err(err, MANGO_ERR_VALIDATION, "dividend_yield must be finite"); return false;
  }
  out.spot = c->spot;
  out.dividend_yield = c->dividend_yield;
  out.grid.moneyness = to_vec(c->moneyness, c->n_moneyness);
  out.grid.vol = to_vec(c->vol, c->n_vol);
  out.grid.rate = to_vec(c->rate, c->n_rate);
  out.backend = mango::BSplineBackend{to_vec(c->maturity_grid, c->n_maturity)};
  out.solver_config.max_iter = static_cast<std::size_t>(c->solver_config.max_iter);
  out.solver_config.tolerance = c->solver_config.tolerance;
  out.solver_config.sigma_min = c->solver_config.sigma_min;
  out.solver_config.sigma_max = c->solver_config.sigma_max;
  out.solver_config.vega_threshold = c->solver_config.vega_threshold;
  if (c->adaptive) {
    mango::AdaptiveGridParams a;
    a.target_iv_error = c->adaptive->target_iv_error;
    a.max_iter = static_cast<std::size_t>(c->adaptive->max_iter);
    a.max_points_per_dim = static_cast<std::size_t>(c->adaptive->max_points_per_dim);
    a.min_moneyness_points = static_cast<std::size_t>(c->adaptive->min_moneyness_points);
    a.validation_samples = static_cast<std::size_t>(c->adaptive->validation_samples);
    a.refinement_factor = c->adaptive->refinement_factor;
    a.lhs_seed = c->adaptive->lhs_seed;
    a.vega_floor = c->adaptive->vega_floor;
    a.max_failure_rate = c->adaptive->max_failure_rate;
    out.adaptive = a;
  }
  if (c->discrete_dividends) {
    const auto* d = c->discrete_dividends;
    mango::DiscreteDividendConfig dd;
    dd.maturity = d->maturity;
    if (!build_dividends(d->dividends, d->n_dividends, dd.discrete_dividends, err))
      return false;
    dd.kref_config.K_refs = to_vec(d->kref_config.K_refs, d->kref_config.n_K_refs);
    dd.kref_config.K_ref_count =
        (d->kref_config.n_K_refs == 0 && d->kref_config.K_ref_count <= 0)
            ? 11 : d->kref_config.K_ref_count;
    dd.kref_config.K_ref_span = d->kref_config.K_ref_span;
    out.discrete_dividends = dd;
  }
  return true;
}
```

- [ ] **Step 3: Add the opaque wrapper structs (after `MangoAmericanResult`)**

```cpp
struct MangoInterpIvSolver { mango::AnyInterpIVSolver solver; };
struct MangoPriceTable { mango::AnyPriceTable table; };
```

- [ ] **Step 4: Implement solver functions inside `extern "C"`**

```cpp
MangoStatus mango_make_interp_iv_solver(const MangoIvFactoryConfig* cfg,
                                        MangoInterpIvSolver** out,
                                        MangoError* err) {
  if (!cfg || !out) {
    set_err(err, MANGO_ERR_VALIDATION, "null cfg or out");
    return MANGO_ERR_VALIDATION;
  }
  *out = nullptr;
  try {
    mango::IVSolverFactoryConfig fc;
    if (!build_factory_config(cfg, fc, err)) return MANGO_ERR_VALIDATION;
    auto result = mango::make_interpolated_iv_solver(fc);
    if (!result) {
      std::string msg = format_validation_error(result.error());
      set_err(err, MANGO_ERR_VALIDATION, msg.c_str());
      return MANGO_ERR_VALIDATION;
    }
    *out = new MangoInterpIvSolver{std::move(result.value())};
    return MANGO_OK;
  } catch (const std::exception& ex) {
    set_err(err, MANGO_ERR_SOLVER, ex.what());
    return MANGO_ERR_SOLVER;
  } catch (...) {
    set_err(err, MANGO_ERR_SOLVER, "unknown error");
    return MANGO_ERR_SOLVER;
  }
}

MangoStatus mango_interp_iv_solve(const MangoInterpIvSolver* s,
                                  const MangoIvQuery* q,
                                  MangoIvSuccess* out, MangoError* err) {
  if (!s || !q || !out) {
    set_err(err, MANGO_ERR_VALIDATION, "null solver, query, or out");
    return MANGO_ERR_VALIDATION;
  }
  try {
    mango::IVQuery query;
    if (!build_iv_query(q, query, err)) return MANGO_ERR_VALIDATION;
    auto result = s->solver.solve(query);
    if (!result) {
      auto code = map_iv_error(result.error());
      std::string msg = format_iv_error(result.error());
      set_err(err, code, msg.c_str());
      return code;
    }
    fill_iv_success(out, result.value());
    return MANGO_OK;
  } catch (const std::exception& ex) {
    set_err(err, MANGO_ERR_SOLVER, ex.what());
    return MANGO_ERR_SOLVER;
  } catch (...) {
    set_err(err, MANGO_ERR_SOLVER, "unknown error");
    return MANGO_ERR_SOLVER;
  }
}

MangoStatus mango_interp_iv_solve_batch(const MangoInterpIvSolver* s,
                                        const MangoIvQuery* queries, uint64_t n,
                                        MangoIvBatchSlot* out_slots,
                                        uint64_t* out_failed_count,
                                        MangoError* err) {
  if (!s || (!queries && n > 0) || (!out_slots && n > 0)) {
    set_err(err, MANGO_ERR_VALIDATION, "null solver, queries, or out_slots");
    return MANGO_ERR_VALIDATION;
  }
  try {
    std::vector<mango::IVQuery> qs;
    qs.reserve(n);
    for (uint64_t i = 0; i < n; ++i) {
      mango::IVQuery query;
      if (!build_iv_query(&queries[i], query, err)) return MANGO_ERR_VALIDATION;
      qs.push_back(std::move(query));
    }
    auto batch = s->solver.solve_batch(qs);  // noexcept
    for (uint64_t i = 0; i < n; ++i) {
      const auto& r = batch.results[i];
      if (r.has_value()) {
        out_slots[i].status = MANGO_OK;
        fill_iv_success(&out_slots[i].success, r.value());
      } else {
        out_slots[i].status = map_iv_error(r.error());
        out_slots[i].success = MangoIvSuccess{};
      }
    }
    if (out_failed_count) *out_failed_count = batch.failed_count;
    return MANGO_OK;
  } catch (const std::exception& ex) {
    set_err(err, MANGO_ERR_SOLVER, ex.what());
    return MANGO_ERR_SOLVER;
  } catch (...) {
    set_err(err, MANGO_ERR_SOLVER, "unknown error");
    return MANGO_ERR_SOLVER;
  }
}

void mango_interp_iv_solver_free(MangoInterpIvSolver* s) { delete s; }
```

- [ ] **Step 5: Build the shim**

Run: `bazel build //src/ffi:mango_c_api`
Expected: PASS. (Confirms `AnyInterpIVSolver`/`BatchIVResult`/`solve_batch` field
names — if `batch.results`/`batch.failed_count` mismatch, check
`interpolated_iv_solver.hpp` for the real `BatchIVResult` member names and fix.)

- [ ] **Step 6: Commit**

```bash
git add src/ffi/mango_c_api.cpp src/ffi/BUILD.bazel
git commit -m "Implement interpolated IV solver C shim"
```

---

## Task 4: Implement the price-table shim

**Files:**
- Modify: `src/ffi/mango_c_api.cpp`

- [ ] **Step 1: Add a `PricingParams` builder helper (anonymous namespace)**

```cpp
bool build_pricing_params(const MangoPricingParams* p, mango::PricingParams& out,
                          MangoError* err) {
  out.spot = p->spot; out.strike = p->strike;
  out.maturity = p->maturity; out.dividend_yield = p->dividend_yield;
  out.volatility = p->volatility;
  if (!validate_option_type(p->option_type, err, out.option_type)) return false;
  if (!build_rate(p->rate_const, p->tenor_points, p->n_tenor_points, out.rate, err))
    return false;
  if (!build_dividends(p->dividends, p->n_dividends, out.discrete_dividends, err))
    return false;
  return true;
}
```
(Requires `#include "mango/option/american_option.hpp"` for `mango::PricingParams`
— already pulled in transitively; add the include if compilation complains.)

- [ ] **Step 2: Implement the price-table functions inside `extern "C"`**

```cpp
MangoStatus mango_make_price_table(const MangoIvFactoryConfig* cfg,
                                   MangoPriceTable** out, MangoError* err) {
  if (!cfg || !out) {
    set_err(err, MANGO_ERR_VALIDATION, "null cfg or out");
    return MANGO_ERR_VALIDATION;
  }
  *out = nullptr;
  try {
    mango::IVSolverFactoryConfig fc;
    if (!build_factory_config(cfg, fc, err)) return MANGO_ERR_VALIDATION;
    auto result = mango::make_price_table(fc);
    if (!result) {
      std::string msg = format_validation_error(result.error());
      set_err(err, MANGO_ERR_VALIDATION, msg.c_str());
      return MANGO_ERR_VALIDATION;
    }
    *out = new MangoPriceTable{std::move(result.value())};
    return MANGO_OK;
  } catch (const std::exception& ex) {
    set_err(err, MANGO_ERR_SOLVER, ex.what());
    return MANGO_ERR_SOLVER;
  } catch (...) {
    set_err(err, MANGO_ERR_SOLVER, "unknown error");
    return MANGO_ERR_SOLVER;
  }
}

MangoStatus mango_price_table_validate(const MangoPriceTable* t,
                                       const MangoPricingParams* p,
                                       MangoError* err) {
  if (!t || !p) {
    set_err(err, MANGO_ERR_VALIDATION, "null table or params");
    return MANGO_ERR_VALIDATION;
  }
  try {
    mango::PricingParams pp;
    if (!build_pricing_params(p, pp, err)) return MANGO_ERR_VALIDATION;
    auto v = t->table.validate_pricing_params(pp);
    if (!v) {
      std::string msg = format_validation_error(v.error());
      set_err(err, MANGO_ERR_VALIDATION, msg.c_str());
      return MANGO_ERR_VALIDATION;
    }
    return MANGO_OK;
  } catch (const std::exception& ex) {
    set_err(err, MANGO_ERR_SOLVER, ex.what());
    return MANGO_ERR_SOLVER;
  } catch (...) {
    set_err(err, MANGO_ERR_SOLVER, "unknown error");
    return MANGO_ERR_SOLVER;
  }
}

// price/vega: infallible f64; nan on null/build failure (extrapolates in domain).
double mango_price_table_price(const MangoPriceTable* t, const MangoPricingParams* p) {
  if (!t || !p) return std::nan("");
  try {
    mango::PricingParams pp;
    if (!build_pricing_params(p, pp, nullptr)) return std::nan("");
    return t->table.price(pp);
  } catch (...) { return std::nan(""); }
}
double mango_price_table_vega(const MangoPriceTable* t, const MangoPricingParams* p) {
  if (!t || !p) return std::nan("");
  try {
    mango::PricingParams pp;
    if (!build_pricing_params(p, pp, nullptr)) return std::nan("");
    return t->table.vega(pp);
  } catch (...) { return std::nan(""); }
}
```

For the four Greeks, write a macro-free helper to avoid repetition:

```cpp
namespace {
template <typename Fn>
MangoStatus greek_call(const MangoPriceTable* t, const MangoPricingParams* p,
                       double* out, MangoError* err, Fn&& fn) {
  if (!t || !p || !out) {
    set_err(err, MANGO_ERR_VALIDATION, "null table, params, or out");
    return MANGO_ERR_VALIDATION;
  }
  try {
    mango::PricingParams pp;
    if (!build_pricing_params(p, pp, err)) return MANGO_ERR_VALIDATION;
    auto r = fn(t->table, pp);
    if (!r) {
      set_err(err, map_greek_error(r.error()), greek_error_msg(r.error()));
      return map_greek_error(r.error());
    }
    *out = r.value();
    return MANGO_OK;
  } catch (const std::exception& ex) {
    set_err(err, MANGO_ERR_SOLVER, ex.what());
    return MANGO_ERR_SOLVER;
  } catch (...) {
    set_err(err, MANGO_ERR_SOLVER, "unknown error");
    return MANGO_ERR_SOLVER;
  }
}
}  // namespace
```

> Note: this template helper must be declared in the anonymous namespace BEFORE
> the `extern "C"` block. Then:

```cpp
MangoStatus mango_price_table_delta(const MangoPriceTable* t, const MangoPricingParams* p,
                                    double* out, MangoError* err) {
  return greek_call(t, p, out, err,
                    [](const mango::AnyPriceTable& tb, const mango::PricingParams& pp) {
                      return tb.delta(pp);
                    });
}
// gamma, theta, rho: identical, calling tb.gamma / tb.theta / tb.rho.

MangoOptionType mango_price_table_option_type(const MangoPriceTable* t) {
  if (!t) return MANGO_PUT;  // arbitrary default on null; callers null-check handles
  return t->table.option_type() == mango::OptionType::CALL ? MANGO_CALL : MANGO_PUT;
}
double mango_price_table_dividend_yield(const MangoPriceTable* t) {
  return t ? t->table.dividend_yield() : std::nan("");
}

MangoStatus mango_price_table_make_iv_solver(const MangoPriceTable* t,
                                             const MangoInterpSolverConfig* cfg,
                                             MangoInterpIvSolver** out,
                                             MangoError* err) {
  if (!t || !out) {
    set_err(err, MANGO_ERR_VALIDATION, "null table or out");
    return MANGO_ERR_VALIDATION;
  }
  *out = nullptr;
  try {
    mango::InterpolatedIVSolverConfig sc;  // defaults
    if (cfg) {
      sc.max_iter = static_cast<std::size_t>(cfg->max_iter);
      sc.tolerance = cfg->tolerance;
      sc.sigma_min = cfg->sigma_min;
      sc.sigma_max = cfg->sigma_max;
      sc.vega_threshold = cfg->vega_threshold;
    }
    auto result = t->table.make_iv_solver(sc);
    if (!result) {
      std::string msg = format_validation_error(result.error());
      set_err(err, MANGO_ERR_VALIDATION, msg.c_str());
      return MANGO_ERR_VALIDATION;
    }
    *out = new MangoInterpIvSolver{std::move(result.value())};
    return MANGO_OK;
  } catch (const std::exception& ex) {
    set_err(err, MANGO_ERR_SOLVER, ex.what());
    return MANGO_ERR_SOLVER;
  } catch (...) {
    set_err(err, MANGO_ERR_SOLVER, "unknown error");
    return MANGO_ERR_SOLVER;
  }
}

void mango_price_table_free(MangoPriceTable* t) { delete t; }
```

- [ ] **Step 3: Build the shim**

Run: `bazel build //src/ffi:mango_c_api`
Expected: PASS. (Confirms `AnyPriceTable::{price,vega,delta,…,validate_pricing_params,make_iv_solver}`.)

- [ ] **Step 4: Commit**

```bash
git add src/ffi/mango_c_api.cpp
git commit -m "Implement price-table C shim"
```

---

## Task 5: Safe Rust `interp.rs` (factory config + interpolated IV solver)

**Files:**
- Create: `crates/mango-option/src/interp.rs`
- Modify: `crates/mango-option/src/lib.rs`
- Modify: `crates/mango-option/src/pricing.rs` (make `dividend_array` reusable — already `pub(crate)`)
- Create: `crates/mango-option/tests/interpolation.rs`
- Modify: `crates/mango-option/BUILD.bazel`

- [ ] **Step 1: Write `interp.rs`**

```rust
// SPDX-License-Identifier: MIT
use crate::error::{Error, ErrorKind};
use crate::iv::{IvQuery, IvSuccess};
use crate::pricing::{blank_error, div_ptr_or_null, dividend_array, ptr_or_null, tenor_array};
use crate::types::{Dividend, OptionType, Rate};
use mango_option_sys as sys;

/// IV grid axes (S/K moneyness, NOT log). Default mirrors the C++ IVGrid.
#[derive(Debug, Clone)]
pub struct IvGrid {
    pub moneyness: Vec<f64>,
    pub vol: Vec<f64>,
    pub rate: Vec<f64>,
}
impl Default for IvGrid {
    fn default() -> Self {
        IvGrid {
            moneyness: vec![0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3],
            vol: vec![0.05, 0.10, 0.20, 0.30, 0.50],
            rate: vec![0.01, 0.03, 0.05, 0.10],
        }
    }
}

/// Newton config for the interpolated solver. Default mirrors C++ defaults.
#[derive(Debug, Clone, Copy)]
pub struct InterpSolverConfig {
    pub max_iter: usize,
    pub tolerance: f64,
    pub sigma_min: f64,
    pub sigma_max: f64,
    pub vega_threshold: f64,
}
impl Default for InterpSolverConfig {
    fn default() -> Self {
        InterpSolverConfig {
            max_iter: 50,
            tolerance: 1e-6,
            sigma_min: 0.01,
            sigma_max: 3.0,
            vega_threshold: 1e-4,
        }
    }
}
impl InterpSolverConfig {
    fn to_c(self) -> sys::MangoInterpSolverConfig {
        sys::MangoInterpSolverConfig {
            max_iter: self.max_iter as u64,
            tolerance: self.tolerance,
            sigma_min: self.sigma_min,
            sigma_max: self.sigma_max,
            vega_threshold: self.vega_threshold,
        }
    }
}

/// Adaptive grid refinement params. Default mirrors C++ defaults.
#[derive(Debug, Clone, Copy)]
pub struct AdaptiveGridParams {
    pub target_iv_error: f64,
    pub max_iter: usize,
    pub max_points_per_dim: usize,
    pub min_moneyness_points: usize,
    pub validation_samples: usize,
    pub refinement_factor: f64,
    pub lhs_seed: u64,
    pub vega_floor: f64,
    pub max_failure_rate: f64,
}
impl Default for AdaptiveGridParams {
    fn default() -> Self {
        AdaptiveGridParams {
            target_iv_error: 2e-5,
            max_iter: 5,
            max_points_per_dim: 160,
            min_moneyness_points: 60,
            validation_samples: 64,
            refinement_factor: 1.3,
            lhs_seed: 42,
            vega_floor: 1e-4,
            max_failure_rate: 0.5,
        }
    }
}
impl AdaptiveGridParams {
    fn to_c(&self) -> sys::MangoAdaptiveGridParams {
        sys::MangoAdaptiveGridParams {
            target_iv_error: self.target_iv_error,
            max_iter: self.max_iter as u64,
            max_points_per_dim: self.max_points_per_dim as u64,
            min_moneyness_points: self.min_moneyness_points as u64,
            validation_samples: self.validation_samples as u64,
            refinement_factor: self.refinement_factor,
            lhs_seed: self.lhs_seed,
            vega_floor: self.vega_floor,
            max_failure_rate: self.max_failure_rate,
        }
    }
}

/// Multi reference-strike config for the discrete-dividend path.
#[derive(Debug, Clone)]
pub struct MultiKRef {
    pub k_refs: Vec<f64>,
    pub k_ref_count: i32,
    pub k_ref_span: f64,
}
impl Default for MultiKRef {
    fn default() -> Self {
        MultiKRef { k_refs: Vec::new(), k_ref_count: 11, k_ref_span: 0.3 }
    }
}

#[derive(Debug, Clone)]
pub struct DiscreteDividendConfig {
    pub maturity: f64,
    pub dividends: Vec<Dividend>,
    pub kref_config: MultiKRef,
}

#[derive(Debug, Clone)]
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

/// Result of a batch solve: per-query results plus the failure count.
#[derive(Debug)]
pub struct BatchResult {
    pub results: Vec<Result<IvSuccess, Error>>,
    pub failed: usize,
}

/// Interpolated IV solver over a pre-built B-spline surface. Thread-safe to
/// query concurrently (immutable surface).
pub struct InterpIvSolver {
    handle: *mut sys::MangoInterpIvSolver,
}
// SAFETY: the C++ AnyInterpIVSolver wraps an immutable surface; solve()/
// solve_batch() are const and documented thread-safe.
unsafe impl Send for InterpIvSolver {}
unsafe impl Sync for InterpIvSolver {}

impl core::fmt::Debug for InterpIvSolver {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "InterpIvSolver({:p})", self.handle)
    }
}

impl InterpIvSolver {
    pub fn new(cfg: &FactoryConfig) -> Result<Self, Error> {
        let mut handle: *mut sys::MangoInterpIvSolver = core::ptr::null_mut();
        let h = with_c_config(cfg, |c, err| unsafe {
            (sys::mango_make_interp_iv_solver(c, &mut handle, err), handle)
        })?;
        Ok(InterpIvSolver { handle: h })
    }

    pub fn solve(&self, query: &IvQuery) -> Result<IvSuccess, Error> {
        let (c, _keep) = iv_query_to_c(query)?;
        let mut out = blank_iv_success();
        let mut err = blank_error();
        let status =
            unsafe { sys::mango_interp_iv_solve(self.handle, &c, &mut out, &mut err) };
        if status == sys::MANGO_OK {
            Ok(iv_success_from_c(&out))
        } else {
            Err(Error::from_c(status, &err))
        }
    }

    pub fn solve_batch(&self, queries: &[IvQuery]) -> BatchResult {
        // Build C queries; keep their backing arrays alive until the call returns.
        let mut c_queries = Vec::with_capacity(queries.len());
        let mut keepalive = Vec::with_capacity(queries.len());
        for q in queries {
            match iv_query_to_c(q) {
                Ok((c, keep)) => {
                    c_queries.push(c);
                    keepalive.push(keep);
                }
                Err(e) => {
                    // A structurally-invalid query fails the whole batch up front,
                    // mirroring the shim's per-query build validation.
                    return BatchResult {
                        results: queries.iter().map(|_| Err(e.clone())).collect(),
                        failed: queries.len(),
                    };
                }
            }
        }
        let n = c_queries.len() as u64;
        let mut slots: Vec<sys::MangoIvBatchSlot> = (0..c_queries.len())
            .map(|_| sys::MangoIvBatchSlot { status: 0, success: blank_iv_success() })
            .collect();
        let mut failed: u64 = 0;
        let mut err = blank_error();
        let status = unsafe {
            sys::mango_interp_iv_solve_batch(
                self.handle,
                if c_queries.is_empty() { core::ptr::null() } else { c_queries.as_ptr() },
                n,
                if slots.is_empty() { core::ptr::null_mut() } else { slots.as_mut_ptr() },
                &mut failed,
                &mut err,
            )
        };
        if status != sys::MANGO_OK {
            let e = Error::from_c(status, &err);
            return BatchResult {
                results: queries.iter().map(|_| Err(e.clone())).collect(),
                failed: queries.len(),
            };
        }
        let results = slots
            .iter()
            .map(|s| {
                if s.status == sys::MANGO_OK {
                    Ok(iv_success_from_c(&s.success))
                } else {
                    Err(Error { kind: ErrorKind::from_status(s.status),
                                message: batch_error_message(s.status) })
                }
            })
            .collect();
        BatchResult { results, failed: failed as usize }
    }
}

impl Drop for InterpIvSolver {
    fn drop(&mut self) {
        unsafe { sys::mango_interp_iv_solver_free(self.handle) }
    }
}

// --- shared helpers used by interp.rs and table.rs ---

pub(crate) fn blank_iv_success() -> sys::MangoIvSuccess {
    sys::MangoIvSuccess {
        implied_vol: 0.0, iterations: 0, final_error: 0.0,
        vega: 0.0, has_vega: 0, used_rate_approximation: 0,
    }
}

pub(crate) fn iv_success_from_c(out: &sys::MangoIvSuccess) -> IvSuccess {
    IvSuccess {
        implied_vol: out.implied_vol,
        iterations: out.iterations as usize,
        final_error: out.final_error,
        vega: if out.has_vega != 0 { Some(out.vega) } else { None },
        used_rate_approximation: out.used_rate_approximation != 0,
    }
}

fn batch_error_message(status: i32) -> String {
    format!("interpolated IV solve failed ({})", ErrorKind::from_status(status))
}

// Backing arrays that must outlive an FFI call referencing their pointers.
pub(crate) struct IvQueryKeepalive {
    _tenors: Vec<sys::MangoTenorPoint>,
    _divs: Vec<sys::MangoDividend>,
}

pub(crate) fn iv_query_to_c(
    query: &IvQuery,
) -> Result<(sys::MangoIvQuery, IvQueryKeepalive), Error> {
    if matches!(&query.spec.rate, Rate::Curve(v) if v.is_empty()) {
        return Err(Error { kind: ErrorKind::Validation,
                           message: "yield curve has no tenor points".to_string() });
    }
    let tenors = tenor_array(&query.spec.rate);
    let divs = dividend_array(&query.spec.discrete_dividends);
    let rate_const = match query.spec.rate {
        Rate::Const(r) => r,
        Rate::Curve(_) => 0.0,
    };
    let c = sys::MangoIvQuery {
        spot: query.spec.spot,
        strike: query.spec.strike,
        maturity: query.spec.maturity,
        dividend_yield: query.spec.dividend_yield,
        market_price: query.market_price,
        rate_const,
        tenor_points: ptr_or_null(&tenors),
        n_tenor_points: tenors.len() as u64,
        dividends: div_ptr_or_null(&divs),
        n_dividends: divs.len() as u64,
        option_type: query.spec.option_type.to_c(),
    };
    Ok((c, IvQueryKeepalive { _tenors: tenors, _divs: divs }))
}

// Build the C factory config, keeping every Vec alive across the FFI call,
// then invoke `f` with a pointer to the config and the error out-param.
fn with_c_config<T>(
    cfg: &FactoryConfig,
    f: impl FnOnce(*const sys::MangoIvFactoryConfig, *mut sys::MangoError) -> (i32, T),
) -> Result<T, Error> {
    let moneyness = cfg.grid.moneyness.clone();
    let vol = cfg.grid.vol.clone();
    let rate = cfg.grid.rate.clone();
    let maturity = cfg.maturity_grid.clone();

    let adaptive_c = cfg.adaptive.as_ref().map(|a| a.to_c());
    // Discrete-dividend sub-config: keep its Vecs alive too.
    let dd = cfg.discrete_dividends.as_ref().map(|d| {
        let divs = dividend_array(&d.dividends);
        let krefs = d.kref_config.k_refs.clone();
        let c = sys::MangoDiscreteDividendConfig {
            maturity: d.maturity,
            dividends: div_ptr_or_null(&divs),
            n_dividends: divs.len() as u64,
            kref_config: sys::MangoMultiKRef {
                K_refs: if krefs.is_empty() { core::ptr::null() } else { krefs.as_ptr() },
                n_K_refs: krefs.len() as u64,
                K_ref_count: d.kref_config.k_ref_count,
                K_ref_span: d.kref_config.k_ref_span,
            },
        };
        (c, divs, krefs)
    });

    let f64_ptr = |v: &[f64]| if v.is_empty() { core::ptr::null() } else { v.as_ptr() };

    let c = sys::MangoIvFactoryConfig {
        option_type: cfg.option_type.to_c(),
        spot: cfg.spot,
        dividend_yield: cfg.dividend_yield,
        moneyness: f64_ptr(&moneyness),
        n_moneyness: moneyness.len() as u64,
        vol: f64_ptr(&vol),
        n_vol: vol.len() as u64,
        rate: f64_ptr(&rate),
        n_rate: rate.len() as u64,
        maturity_grid: f64_ptr(&maturity),
        n_maturity: maturity.len() as u64,
        solver_config: cfg.solver.to_c(),
        adaptive: adaptive_c.as_ref().map_or(core::ptr::null(), |a| a as *const _),
        discrete_dividends: dd.as_ref().map_or(core::ptr::null(), |(c, _, _)| c as *const _),
    };

    let mut err = blank_error();
    let (status, value) = f(&c, &mut err);
    // keepalive: moneyness/vol/rate/maturity/adaptive_c/dd live until here.
    let _ = (&moneyness, &vol, &rate, &maturity, &adaptive_c, &dd);
    if status == sys::MANGO_OK {
        Ok(value)
    } else {
        Err(Error::from_c(status, &err))
    }
}

pub(crate) fn make_factory_handle(
    cfg: &FactoryConfig,
) -> Result<*mut sys::MangoInterpIvSolver, Error> {
    let mut handle: *mut sys::MangoInterpIvSolver = core::ptr::null_mut();
    with_c_config(cfg, |c, err| unsafe {
        (sys::mango_make_interp_iv_solver(c, &mut handle, err), handle)
    })
}
```

> Implementer note: `InterpIvSolver::new` above calls `with_c_config` returning
> the handle. Ensure `with_c_config` is `pub(crate)` so `table.rs` can reuse it
> for `mango_make_price_table`. Adjust visibility accordingly (mark
> `with_c_config` `pub(crate)`).

- [ ] **Step 2: Wire `lib.rs` exports**

```rust
mod interp;
mod table;   // added in Task 6
pub use interp::{
    AdaptiveGridParams, BatchResult, DiscreteDividendConfig, FactoryConfig, InterpIvSolver,
    InterpSolverConfig, IvGrid, MultiKRef,
};
```
(Add `table` re-exports in Task 6.)

- [ ] **Step 3: Wire `BUILD.bazel`**

Add `"src/interp.rs"` (and `"src/table.rs"` in Task 6) to `rust_library`
`srcs`, and add a `rust_test` for the new integration file:

```python
rust_test(
    name = "interpolation_test",
    srcs = ["tests/interpolation.rs"],
    edition = "2021",
    deps = [":mango_option"],
)
```

- [ ] **Step 4: Write the solver integration tests (`tests/interpolation.rs`)**

```rust
// SPDX-License-Identifier: MIT
use mango_option::{
    price_american, AdaptiveGridParams, DiscreteDividendConfig, Dividend, FactoryConfig,
    InterpIvSolver, InterpSolverConfig, IvGrid, IvQuery, MultiKRef, OptionSpec, OptionType, Rate,
};

fn base_config() -> FactoryConfig {
    FactoryConfig {
        option_type: OptionType::Put,
        spot: 100.0,
        dividend_yield: 0.0,
        grid: IvGrid {
            moneyness: vec![0.8, 0.9, 1.0, 1.1, 1.2],
            vol: vec![0.10, 0.20, 0.30, 0.40],
            rate: vec![0.01, 0.03, 0.05],
        },
        maturity_grid: vec![0.25, 0.5, 1.0],
        solver: InterpSolverConfig::default(),
        adaptive: None,
        discrete_dividends: None,
    }
}

fn put_spec(sigma: f64) -> (OptionSpec, f64) {
    let spec = OptionSpec {
        spot: 100.0, strike: 100.0, maturity: 1.0, dividend_yield: 0.0,
        rate: Rate::Const(0.03), discrete_dividends: vec![], option_type: OptionType::Put,
    };
    (spec, sigma)
}

#[test]
fn iv_round_trip_continuous() {
    let solver = InterpIvSolver::new(&base_config()).expect("build solver");
    let (spec, sigma) = put_spec(0.25);
    // Reference price from the FDM path at the true sigma.
    let pp = mango_option::PricingParams { spec: spec.clone(), volatility: sigma };
    let price = price_american(&pp).unwrap().value();
    let q = IvQuery { spec, market_price: price };
    let r = solver.solve(&q).expect("solve iv");
    assert!((r.implied_vol - sigma).abs() < 1e-2, "iv={} sigma={}", r.implied_vol, sigma);
}

#[test]
fn batch_with_one_failure() {
    let solver = InterpIvSolver::new(&base_config()).unwrap();
    let (spec, sigma) = put_spec(0.25);
    let pp = mango_option::PricingParams { spec: spec.clone(), volatility: sigma };
    let good_price = price_american(&pp).unwrap().value();
    let good = IvQuery { spec: spec.clone(), market_price: good_price };
    // Arbitrage-violating price (above strike for a put) => solve failure.
    let bad = IvQuery { spec, market_price: 1_000.0 };
    let batch = solver.solve_batch(&[good, bad]);
    assert_eq!(batch.failed, 1);
    assert!(batch.results[0].is_ok());
    assert!(batch.results[1].is_err());
}

#[test]
fn adaptive_build_solves() {
    let mut cfg = base_config();
    cfg.adaptive = Some(AdaptiveGridParams {
        target_iv_error: 1e-3, max_iter: 2, validation_samples: 16,
        min_moneyness_points: 20, ..AdaptiveGridParams::default()
    });
    let solver = InterpIvSolver::new(&cfg).expect("adaptive build");
    let (spec, sigma) = put_spec(0.25);
    let pp = mango_option::PricingParams { spec: spec.clone(), volatility: sigma };
    let price = price_american(&pp).unwrap().value();
    let r = solver.solve(&IvQuery { spec, market_price: price }).expect("solve");
    assert!((r.implied_vol - sigma).abs() < 2e-2);
}

#[test]
fn discrete_dividend_build_solves() {
    let mut cfg = base_config();
    cfg.discrete_dividends = Some(DiscreteDividendConfig {
        maturity: 1.0,
        dividends: vec![Dividend { calendar_time: 0.5, amount: 2.0 }],
        kref_config: MultiKRef { k_refs: vec![90.0, 100.0, 110.0], ..MultiKRef::default() },
    });
    let solver = InterpIvSolver::new(&cfg).expect("discrete build");
    let spec = OptionSpec {
        spot: 100.0, strike: 100.0, maturity: 1.0, dividend_yield: 0.0,
        rate: Rate::Const(0.03),
        discrete_dividends: vec![Dividend { calendar_time: 0.5, amount: 2.0 }],
        option_type: OptionType::Put,
    };
    let pp = mango_option::PricingParams { spec: spec.clone(), volatility: 0.25 };
    let price = price_american(&pp).unwrap().value();
    let r = solver.solve(&IvQuery { spec, market_price: price }).expect("solve");
    assert!((r.implied_vol - 0.25).abs() < 3e-2, "iv={}", r.implied_vol);
}

#[test]
fn empty_maturity_grid_continuous_is_validation_error() {
    let mut cfg = base_config();
    cfg.maturity_grid = vec![];
    let err = InterpIvSolver::new(&cfg).unwrap_err();
    assert_eq!(err.kind, mango_option::ErrorKind::Validation);
}

#[test]
fn non_finite_spot_is_validation_error() {
    let mut cfg = base_config();
    cfg.spot = f64::NAN;
    let err = InterpIvSolver::new(&cfg).unwrap_err();
    assert_eq!(err.kind, mango_option::ErrorKind::Validation);
}

#[test]
fn solver_is_send_sync() {
    let solver = InterpIvSolver::new(&base_config()).unwrap();
    let (spec, sigma) = put_spec(0.25);
    let pp = mango_option::PricingParams { spec: spec.clone(), volatility: sigma };
    let price = price_american(&pp).unwrap().value();
    let s = std::sync::Arc::new(solver);
    let handles: Vec<_> = (0..2).map(|_| {
        let s = s.clone();
        let spec = spec.clone();
        std::thread::spawn(move || {
            s.solve(&IvQuery { spec, market_price: price }).map(|r| r.implied_vol)
        })
    }).collect();
    for h in handles { assert!(h.join().unwrap().is_ok()); }
}
```

> Implementer note: `lib.rs` must also re-export `PricingParams`, `Dividend`,
> `OptionSpec`, `Rate`, `OptionType`, `ErrorKind`, `price_american`, `IvQuery`
> for these tests — most already are. Add any missing re-export.

- [ ] **Step 5: Build + run**

Run: `bazel test //crates/mango-option-sys:layout_test //crates/mango-option:interpolation_test //crates/mango-option:integration_test`
Expected: PASS (all of the above; v1 tests still green).

- [ ] **Step 6: Commit**

```bash
git add crates/mango-option/src/interp.rs crates/mango-option/src/lib.rs \
        crates/mango-option/tests/interpolation.rs crates/mango-option/BUILD.bazel
git commit -m "Add safe Rust interpolated IV solver"
```

---

## Task 6: Safe Rust `table.rs` (price table)

**Files:**
- Create: `crates/mango-option/src/table.rs`
- Modify: `crates/mango-option/src/lib.rs` (re-exports)
- Modify: `crates/mango-option/tests/interpolation.rs` (table tests)
- Modify: `crates/mango-option/BUILD.bazel` (add `src/table.rs` to srcs)

- [ ] **Step 1: Write `table.rs`**

```rust
// SPDX-License-Identifier: MIT
use crate::error::Error;
use crate::interp::{make_price_table_handle, with_c_config, FactoryConfig, InterpIvSolver,
                    InterpSolverConfig};
use crate::pricing::{blank_error, pricing_params_to_c, PricingParams};
use crate::types::OptionType;
use mango_option_sys as sys;

/// A reusable, immutable B-spline price surface. Thread-safe to query.
pub struct PriceTable {
    handle: *mut sys::MangoPriceTable,
}
// SAFETY: the C++ AnyPriceTable wraps an immutable surface; all query methods
// are const and documented thread-safe.
unsafe impl Send for PriceTable {}
unsafe impl Sync for PriceTable {}

impl core::fmt::Debug for PriceTable {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "PriceTable({:p})", self.handle)
    }
}

impl PriceTable {
    pub fn new(cfg: &FactoryConfig) -> Result<Self, Error> {
        Ok(PriceTable { handle: make_price_table_handle(cfg)? })
    }

    /// Opt-in bounds check; out-of-domain params => Err. `price`/`vega` do NOT
    /// call this and will extrapolate.
    pub fn validate(&self, params: &PricingParams) -> Result<(), Error> {
        let (c, _keep) = pricing_params_to_c(params)?;
        let mut err = blank_error();
        let status = unsafe { sys::mango_price_table_validate(self.handle, &c, &mut err) };
        if status == sys::MANGO_OK { Ok(()) } else { Err(Error::from_c(status, &err)) }
    }

    /// Interpolated price. Extrapolates out of the surface domain; returns NaN
    /// only on an internal failure. Call `validate` first if you need bounds.
    pub fn price(&self, params: &PricingParams) -> f64 {
        match pricing_params_to_c(params) {
            Ok((c, _keep)) => unsafe { sys::mango_price_table_price(self.handle, &c) },
            Err(_) => f64::NAN,
        }
    }
    pub fn vega(&self, params: &PricingParams) -> f64 {
        match pricing_params_to_c(params) {
            Ok((c, _keep)) => unsafe { sys::mango_price_table_vega(self.handle, &c) },
            Err(_) => f64::NAN,
        }
    }

    pub fn delta(&self, params: &PricingParams) -> Result<f64, Error> {
        self.greek(params, sys::mango_price_table_delta)
    }
    pub fn gamma(&self, params: &PricingParams) -> Result<f64, Error> {
        self.greek(params, sys::mango_price_table_gamma)
    }
    pub fn theta(&self, params: &PricingParams) -> Result<f64, Error> {
        self.greek(params, sys::mango_price_table_theta)
    }
    pub fn rho(&self, params: &PricingParams) -> Result<f64, Error> {
        self.greek(params, sys::mango_price_table_rho)
    }

    fn greek(
        &self,
        params: &PricingParams,
        f: unsafe extern "C" fn(*const sys::MangoPriceTable, *const sys::MangoPricingParams,
                                *mut f64, *mut sys::MangoError) -> sys::MangoStatus,
    ) -> Result<f64, Error> {
        let (c, _keep) = pricing_params_to_c(params)?;
        let mut out = 0.0;
        let mut err = blank_error();
        let status = unsafe { f(self.handle, &c, &mut out, &mut err) };
        if status == sys::MANGO_OK { Ok(out) } else { Err(Error::from_c(status, &err)) }
    }

    pub fn option_type(&self) -> OptionType {
        let t = unsafe { sys::mango_price_table_option_type(self.handle) };
        if t == sys::MANGO_CALL { OptionType::Call } else { OptionType::Put }
    }
    pub fn dividend_yield(&self) -> f64 {
        unsafe { sys::mango_price_table_dividend_yield(self.handle) }
    }

    /// Derive an IV solver from this already-built surface (no rebuild).
    pub fn iv_solver(&self, cfg: Option<&InterpSolverConfig>) -> Result<InterpIvSolver, Error> {
        let c = cfg.map(|c| c.to_c());
        let mut handle: *mut sys::MangoInterpIvSolver = core::ptr::null_mut();
        let mut err = blank_error();
        let status = unsafe {
            sys::mango_price_table_make_iv_solver(
                self.handle,
                c.as_ref().map_or(core::ptr::null(), |c| c as *const _),
                &mut handle,
                &mut err,
            )
        };
        if status == sys::MANGO_OK {
            Ok(InterpIvSolver::from_raw(handle))
        } else {
            Err(Error::from_c(status, &err))
        }
    }
}

impl Drop for PriceTable {
    fn drop(&mut self) {
        unsafe { sys::mango_price_table_free(self.handle) }
    }
}
```

> Implementer notes (small refactors required to make the above compile):
> 1. In `interp.rs`, add `make_price_table_handle(cfg) -> Result<*mut sys::MangoPriceTable, Error>`
>    (analogous to `make_factory_handle`, calling `mango_make_price_table`), make
>    `with_c_config` and `InterpSolverConfig::to_c` `pub(crate)`, and add
>    `InterpIvSolver::from_raw(handle) -> Self` (a `pub(crate)` constructor).
> 2. In `pricing.rs`, factor the existing `MangoPricingParams` construction in
>    `price_american` into a `pub(crate) fn pricing_params_to_c(&PricingParams)
>    -> Result<(sys::MangoPricingParams, PricingKeepalive), Error>` (mirroring
>    `iv_query_to_c`), where `PricingKeepalive` owns the tenor/dividend Vecs.
>    Update `price_american` to use it. Keep behavior identical (the empty-curve
>    guard moves into the helper).

- [ ] **Step 2: Add table re-exports to `lib.rs`**

```rust
pub use table::PriceTable;
```

- [ ] **Step 3: Add `src/table.rs` to BUILD srcs** (already declared `mod table` in Task 5).

- [ ] **Step 4: Append table tests to `tests/interpolation.rs`**

```rust
use mango_option::{PriceTable, PricingParams};

#[test]
fn price_table_queries() {
    let table = PriceTable::new(&base_config()).expect("build table");
    let spec = OptionSpec {
        spot: 100.0, strike: 100.0, maturity: 1.0, dividend_yield: 0.0,
        rate: Rate::Const(0.03), discrete_dividends: vec![], option_type: OptionType::Put,
    };
    let pp = PricingParams { spec: spec.clone(), volatility: 0.25 };
    let price = table.price(&pp);
    assert!(price.is_finite() && price > 0.0);
    assert!(table.vega(&pp).is_finite());
    let delta = table.delta(&pp).expect("delta");
    assert!(delta < 0.0, "put delta should be negative: {}", delta);
    assert!(table.gamma(&pp).is_ok());
    assert!(table.theta(&pp).is_ok());
    assert!(table.rho(&pp).is_ok());
    assert_eq!(table.option_type(), OptionType::Put);

    // Reference FDM price for sanity (interpolation tolerance is loose).
    let fdm = price_american(&pp).unwrap().value();
    assert!((price - fdm).abs() < 0.5, "interp {} vs fdm {}", price, fdm);

    // Opt-in validation.
    assert!(table.validate(&pp).is_ok());
    let oob = PricingParams { spec, volatility: 5.0 }; // far outside vol grid
    assert!(table.validate(&oob).is_err());
}

#[test]
fn price_table_derives_iv_solver() {
    let table = PriceTable::new(&base_config()).unwrap();
    let solver = table.iv_solver(None).expect("derive solver");
    let (spec, sigma) = put_spec(0.25);
    let pp = PricingParams { spec: spec.clone(), volatility: sigma };
    let price = price_american(&pp).unwrap().value();
    let r = solver.solve(&IvQuery { spec, market_price: price }).expect("solve");
    assert!((r.implied_vol - sigma).abs() < 1e-2);
}
```

- [ ] **Step 5: Build + run all**

Run: `bazel test //crates/mango-option:interpolation_test //crates/mango-option:integration_test //crates/mango-option-sys:layout_test`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add crates/mango-option/src/table.rs crates/mango-option/src/lib.rs \
        crates/mango-option/src/pricing.rs crates/mango-option/src/interp.rs \
        crates/mango-option/tests/interpolation.rs crates/mango-option/BUILD.bazel
git commit -m "Add safe Rust price table"
```

---

## Task 7: Documentation

**Files:**
- Modify: `docs/RUST_GUIDE.md`

- [ ] **Step 1: Add an "Interpolation path" section**

Document, with a worked example each: building an `InterpIvSolver` from a
`FactoryConfig` (continuous), `solve`/`solve_batch`, building a `PriceTable` and
querying `price`/`vega`/greeks via `PricingParams`, `validate` (and the
extrapolation caveat on `price`/`vega`), deriving a solver with
`PriceTable::iv_solver`, and the adaptive + discrete-dividend config options.
Note the batch error-category limitation and `used_rate_approximation`. Mirror
the structure/tone of the existing FDM section.

- [ ] **Step 2: Verify the doc examples compile mentally against the public API**

(No build step; this is prose. Ensure type/method names match `lib.rs` exports.)

- [ ] **Step 3: Commit**

```bash
git add docs/RUST_GUIDE.md
git commit -m "Document the Rust interpolation-path binding"
```

---

## Final verification (before holistic review)

Run the full suite the CI gate checks:

```bash
bazel test //crates/... //tests:iv_solver_test
bazel build //src/python:mango_option   # ensure the shared shim still builds Python
```
Expected: all PASS. Then proceed to holistic review → push → PR → pre-merge review.
