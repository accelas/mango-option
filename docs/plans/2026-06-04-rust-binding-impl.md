# Rust Binding Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A safe Rust binding (`mango-option` over `mango-option-sys`) for American option pricing (value + Greeks) and FDM implied volatility, with full rate/dividend input fidelity, built in-tree via Bazel `rules_rust`.

**Architecture:** A hand-written `extern "C"` shim (`src/ffi/mango_c_api.{h,cpp}`) wraps the C++23 API (translating `std::expected` → status + out-param, flattening the `RateSpec` variant, marshalling dividend/tenor arrays). A raw `mango-option-sys` crate declares the FFI 1:1 (guarded by a bidirectional ABI offset test). A safe `mango-option` crate exposes idiomatic structs/`Result`/RAII so callers write zero `unsafe`.

**Tech Stack:** C++23, Bazel + Bzlmod, `rules_rust` 0.70.0, Rust edition 2021, GoogleTest (shim test), Rust built-in test harness.

**Design reference:** `docs/plans/2026-06-04-rust-binding-design.md`.

**Pre-req already landed:** discrete-dividend support in the FDM `IVSolver` (PR #429 / commit `8030bbe3`). This plan assumes `IVQuery::discrete_dividends` exists.

---

## File Structure

| Path | Responsibility |
|------|----------------|
| `MODULE.bazel` (modify) | Add `rules_rust` dep + Rust toolchain |
| `src/ffi/mango_c_api.h` (create) | Stable C ABI: structs, status enum, function decls, `static_assert` ABI guards |
| `src/ffi/mango_c_api.cpp` (create) | `extern "C"` impl over the C++23 API |
| `src/ffi/BUILD.bazel` (create) | `cc_library` `:mango_c_api` |
| `tests/ffi_c_api_test.cc` (create) | GoogleTest exercising the shim directly (C++ side) |
| `crates/mango-option-sys/src/lib.rs` (create) | `#[repr(C)]` structs + `extern "C"` decls |
| `crates/mango-option-sys/tests/layout.rs` (create) | Per-field `offset_of!` + size/align ABI asserts |
| `crates/mango-option-sys/{Cargo.toml,BUILD.bazel}` (create) | sys crate build |
| `crates/mango-option/src/{lib,types,error,pricing,iv}.rs` (create) | Safe API |
| `crates/mango-option/tests/integration.rs` (create) | Round-trip / fidelity / error tests |
| `crates/mango-option/{Cargo.toml,BUILD.bazel}` (create) | safe crate build |

**ABI offsets (both the C header `static_assert`s and the Rust `offset_of!` tests assert these exact numbers):**

`MangoPricingParams` (size 88, align 8): spot 0, strike 8, maturity 16, dividend_yield 24, volatility 32, rate_const 40, tenor_points 48, n_tenor_points 56, dividends 64, n_dividends 72, option_type 80.

`MangoIvQuery` (size 88, align 8): spot 0, strike 8, maturity 16, dividend_yield 24, market_price 32, rate_const 40, tenor_points 48, n_tenor_points 56, dividends 64, n_dividends 72, option_type 80.

`MangoDividend` (16): calendar_time 0, amount 8. `MangoTenorPoint` (16): tenor 0, log_discount 8.
`MangoError` (260, align 4): code 0, message 4. `MangoIvSuccess` (40, align 8): implied_vol 0, iterations 8, final_error 16, vega 24, has_vega 32. `MangoIvConfig` (16, align 8): brent_tol_abs 0, max_iter 8.

---

## Task 1: Bootstrap `rules_rust` in Bazel

**Files:**
- Modify: `MODULE.bazel`

- [ ] **Step 1: Add the dependency and toolchain to `MODULE.bazel`**

Add after the existing `bazel_dep(...)` lines (e.g. after `platforms`):

```python
# Rust support (for the mango-option Rust binding)
bazel_dep(name = "rules_rust", version = "0.70.0")

rust = use_extension("@rules_rust//rust:extensions.bzl", "rust")
rust.toolchain(edition = "2021")
use_repo(rust, "rust_toolchains")
register_toolchains("@rust_toolchains//:all")
```

(No `versions` pin → `rules_rust` 0.70.0 supplies its default stable `rustc`, which is ≥ 1.77 and therefore has stable `std::mem::offset_of!`. If a specific toolchain is later required, add `versions = ["1.85.0"]`.)

- [ ] **Step 2: Verify `rules_rust` loads**

Run: `bazel query '@rules_rust//rust/...' 2>&1 | head -3`
Expected: a list of targets, no load errors.

- [ ] **Step 3: Verify the C++ build still works (toolchain change is additive)**

Run: `bazel build //src/option:iv_solver 2>&1 | tail -2`
Expected: `Build completed successfully`.

- [ ] **Step 4: Commit**

```bash
git add MODULE.bazel MODULE.bazel.lock
git commit -m "Add rules_rust 0.70.0 toolchain to Bazel"
```

---

## Task 2: C ABI shim header (`mango_c_api.h`)

**Files:**
- Create: `src/ffi/mango_c_api.h`

- [ ] **Step 1: Write the header**

```c
// SPDX-License-Identifier: MIT
// Stable C ABI over the mango-option C++23 pricing library.
// This header is the single source of truth for the FFI boundary; the matching
// Rust declarations live in crates/mango-option-sys/src/lib.rs and are guarded
// by crates/mango-option-sys/tests/layout.rs against the offsets asserted here.
#ifndef MANGO_C_API_H
#define MANGO_C_API_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef int32_t MangoStatus;
#define MANGO_OK 0
#define MANGO_ERR_VALIDATION 1
#define MANGO_ERR_ARBITRAGE 2
#define MANGO_ERR_NO_CONVERGENCE 3
#define MANGO_ERR_BRACKETING 4
#define MANGO_ERR_SOLVER 5

typedef int32_t MangoOptionType;
#define MANGO_CALL 0
#define MANGO_PUT 1

typedef struct { int32_t code; char message[256]; } MangoError;
typedef struct { double calendar_time; double amount; } MangoDividend;
typedef struct { double tenor; double log_discount; } MangoTenorPoint;

typedef struct {
  double spot;
  double strike;
  double maturity;
  double dividend_yield;
  double volatility;
  double rate_const;                    // used iff n_tenor_points == 0
  const MangoTenorPoint* tenor_points;
  uint64_t n_tenor_points;
  const MangoDividend* dividends;        // may be null when n_dividends == 0
  uint64_t n_dividends;
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
  uint64_t n_tenor_points;
  const MangoDividend* dividends;
  uint64_t n_dividends;
  MangoOptionType option_type;
} MangoIvQuery;

typedef struct {
  double implied_vol;
  uint64_t iterations;
  double final_error;
  double vega;
  int32_t has_vega;
} MangoIvSuccess;

typedef struct { double brent_tol_abs; int32_t max_iter; } MangoIvConfig;  // 0 => default

typedef struct MangoAmericanResult MangoAmericanResult;  // opaque

MangoStatus mango_price_american(const MangoPricingParams* params,
                                 MangoAmericanResult** out_result,
                                 MangoError* out_err);
double mango_american_value(const MangoAmericanResult* r);
double mango_american_delta(const MangoAmericanResult* r);
double mango_american_gamma(const MangoAmericanResult* r);
double mango_american_theta(const MangoAmericanResult* r);
MangoStatus mango_american_value_at(const MangoAmericanResult* r, double spot,
                                    double* out, MangoError* out_err);
void mango_american_result_free(MangoAmericanResult* r);

MangoStatus mango_solve_iv(const MangoIvQuery* query,
                           const MangoIvConfig* config,   // nullable => defaults
                           MangoIvSuccess* out_success,
                           MangoError* out_err);

// --- ABI guards: any field reorder/resize breaks the build (mirrored in Rust) ---
_Static_assert(sizeof(MangoPricingParams) == 88, "MangoPricingParams size");
_Static_assert(offsetof(MangoPricingParams, rate_const) == 40, "rate_const off");
_Static_assert(offsetof(MangoPricingParams, tenor_points) == 48, "tenor_points off");
_Static_assert(offsetof(MangoPricingParams, n_dividends) == 72, "n_dividends off");
_Static_assert(offsetof(MangoPricingParams, option_type) == 80, "option_type off");
_Static_assert(sizeof(MangoIvQuery) == 88, "MangoIvQuery size");
_Static_assert(offsetof(MangoIvQuery, market_price) == 32, "market_price off");
_Static_assert(offsetof(MangoIvQuery, option_type) == 80, "iv option_type off");
_Static_assert(sizeof(MangoDividend) == 16, "MangoDividend size");
_Static_assert(sizeof(MangoTenorPoint) == 16, "MangoTenorPoint size");
_Static_assert(sizeof(MangoError) == 260, "MangoError size");
_Static_assert(offsetof(MangoError, message) == 4, "message off");
_Static_assert(sizeof(MangoIvSuccess) == 40, "MangoIvSuccess size");
_Static_assert(offsetof(MangoIvSuccess, has_vega) == 32, "has_vega off");
_Static_assert(sizeof(MangoIvConfig) == 16, "MangoIvConfig size");
_Static_assert(offsetof(MangoIvConfig, max_iter) == 8, "max_iter off");

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // MANGO_C_API_H
```

- [ ] **Step 2: Sanity-check it compiles as C++ (static_asserts hold)**

This is validated when the shim `.cpp` builds in Task 3; no separate command.

---

## Task 3: C ABI shim implementation + cc_library + C++ test

**Files:**
- Create: `src/ffi/mango_c_api.cpp`
- Create: `src/ffi/BUILD.bazel`
- Create: `tests/ffi_c_api_test.cc`
- Modify: `tests/BUILD.bazel` (add `ffi_c_api_test` target)

- [ ] **Step 1: Write the failing C++ test**

`tests/ffi_c_api_test.cc`:

```cpp
// SPDX-License-Identifier: MIT
#include <gtest/gtest.h>
#include "mango/ffi/mango_c_api.h"
#include <cmath>
#include <vector>

namespace {

MangoPricingParams make_put_params() {
  MangoPricingParams p{};
  p.spot = 100.0; p.strike = 100.0; p.maturity = 1.0;
  p.dividend_yield = 0.0; p.volatility = 0.25; p.rate_const = 0.05;
  p.tenor_points = nullptr; p.n_tenor_points = 0;
  p.dividends = nullptr; p.n_dividends = 0;
  p.option_type = MANGO_PUT;
  return p;
}

TEST(MangoCApi, PriceAmericanPutAndGreeks) {
  MangoPricingParams p = make_put_params();
  MangoAmericanResult* r = nullptr;
  MangoError err{};
  ASSERT_EQ(mango_price_american(&p, &r, &err), MANGO_OK) << err.message;
  ASSERT_NE(r, nullptr);

  double v = mango_american_value(r);
  EXPECT_GT(v, 0.0);
  EXPECT_TRUE(std::isfinite(mango_american_delta(r)));
  EXPECT_GT(mango_american_gamma(r), 0.0);
  EXPECT_TRUE(std::isfinite(mango_american_theta(r)));

  double off = 0.0;
  ASSERT_EQ(mango_american_value_at(r, 90.0, &off, &err), MANGO_OK) << err.message;
  EXPECT_GT(off, v);  // deeper ITM put worth more

  mango_american_result_free(r);
}

TEST(MangoCApi, SolveIvRoundTrip) {
  MangoPricingParams p = make_put_params();
  MangoAmericanResult* r = nullptr;
  MangoError err{};
  ASSERT_EQ(mango_price_american(&p, &r, &err), MANGO_OK);
  double market = mango_american_value(r);
  mango_american_result_free(r);

  MangoIvQuery q{};
  q.spot = 100.0; q.strike = 100.0; q.maturity = 1.0;
  q.dividend_yield = 0.0; q.market_price = market; q.rate_const = 0.05;
  q.tenor_points = nullptr; q.n_tenor_points = 0;
  q.dividends = nullptr; q.n_dividends = 0;
  q.option_type = MANGO_PUT;

  MangoIvSuccess out{};
  ASSERT_EQ(mango_solve_iv(&q, nullptr, &out, &err), MANGO_OK) << err.message;
  EXPECT_NEAR(out.implied_vol, 0.25, 0.01);
}

TEST(MangoCApi, NegativeSpotIsValidationError) {
  MangoPricingParams p = make_put_params();
  p.spot = -1.0;
  MangoAmericanResult* r = nullptr;
  MangoError err{};
  EXPECT_EQ(mango_price_american(&p, &r, &err), MANGO_ERR_VALIDATION);
  EXPECT_EQ(r, nullptr);
}

TEST(MangoCApi, DiscreteDividendRoundTripIv) {
  std::vector<MangoDividend> divs = {{0.5, 2.0}};
  MangoPricingParams p = make_put_params();
  p.dividends = divs.data(); p.n_dividends = divs.size();
  MangoAmericanResult* r = nullptr;
  MangoError err{};
  ASSERT_EQ(mango_price_american(&p, &r, &err), MANGO_OK);
  double market = mango_american_value(r);
  mango_american_result_free(r);

  MangoIvQuery q{};
  q.spot = 100.0; q.strike = 100.0; q.maturity = 1.0;
  q.dividend_yield = 0.0; q.market_price = market; q.rate_const = 0.05;
  q.dividends = divs.data(); q.n_dividends = divs.size();
  q.option_type = MANGO_PUT;

  MangoIvSuccess out{};
  ASSERT_EQ(mango_solve_iv(&q, nullptr, &out, &err), MANGO_OK) << err.message;
  EXPECT_NEAR(out.implied_vol, 0.25, 0.01);
}

}  // namespace
```

- [ ] **Step 2: Write the shim implementation**

`src/ffi/mango_c_api.cpp`:

```cpp
// SPDX-License-Identifier: MIT
#include "mango/ffi/mango_c_api.h"

#include "mango/option/american_option.hpp"
#include "mango/option/iv_solver.hpp"
#include "mango/option/option_spec.hpp"
#include "mango/option/yield_curve.hpp"

#include <cstring>
#include <exception>
#include <new>
#include <optional>
#include <string>
#include <vector>

namespace {

void set_err(MangoError* err, MangoStatus code, const std::string& msg) {
  if (!err) return;
  err->code = code;
  std::size_t n = msg.size() < 255 ? msg.size() : 255;
  std::memcpy(err->message, msg.data(), n);
  err->message[n] = '\0';
}

mango::OptionType to_cpp_type(MangoOptionType t) {
  return t == MANGO_PUT ? mango::OptionType::PUT : mango::OptionType::CALL;
}

// Build a RateSpec from (rate_const) or (tenor_points,n). Returns false +
// fills err on an invalid curve.
bool build_rate(double rate_const, const MangoTenorPoint* pts, uint64_t n,
                mango::RateSpec& out, MangoError* err) {
  if (n == 0) { out = rate_const; return true; }
  if (pts == nullptr) {
    set_err(err, MANGO_ERR_VALIDATION, "tenor_points is null but n_tenor_points > 0");
    return false;
  }
  std::vector<mango::TenorPoint> points;
  points.reserve(n);
  for (uint64_t i = 0; i < n; ++i) {
    points.push_back(mango::TenorPoint{pts[i].tenor, pts[i].log_discount});
  }
  auto curve = mango::YieldCurve::from_points(std::move(points));
  if (!curve) {
    set_err(err, MANGO_ERR_VALIDATION, "invalid yield curve: " + curve.error());
    return false;
  }
  out = curve.value();
  return true;
}

bool build_dividends(const MangoDividend* divs, uint64_t n,
                     std::vector<mango::Dividend>& out, MangoError* err) {
  if (n == 0) return true;
  if (divs == nullptr) {
    set_err(err, MANGO_ERR_VALIDATION, "dividends is null but n_dividends > 0");
    return false;
  }
  out.reserve(n);
  for (uint64_t i = 0; i < n; ++i) {
    out.push_back(mango::Dividend{divs[i].calendar_time, divs[i].amount});
  }
  return true;
}

MangoStatus map_solver_error(const mango::SolverError&) { return MANGO_ERR_SOLVER; }

MangoStatus map_iv_error(const mango::IVError& e) {
  switch (e.code) {
    case mango::IVErrorCode::ArbitrageViolation: return MANGO_ERR_ARBITRAGE;
    case mango::IVErrorCode::BracketingFailed: return MANGO_ERR_BRACKETING;
    case mango::IVErrorCode::MaxIterationsExceeded: return MANGO_ERR_NO_CONVERGENCE;
    case mango::IVErrorCode::NegativeSpot:
    case mango::IVErrorCode::NegativeStrike:
    case mango::IVErrorCode::NegativeMaturity:
    case mango::IVErrorCode::NegativeMarketPrice: return MANGO_ERR_VALIDATION;
    default: return MANGO_ERR_SOLVER;
  }
}

}  // namespace

struct MangoAmericanResult {
  mango::AmericanOptionResult result;
  double value;
  double delta;
  double gamma;
  double theta;
};

extern "C" {

MangoStatus mango_price_american(const MangoPricingParams* params,
                                 MangoAmericanResult** out_result,
                                 MangoError* out_err) {
  if (!params || !out_result) {
    set_err(out_err, MANGO_ERR_VALIDATION, "null params or out_result");
    return MANGO_ERR_VALIDATION;
  }
  *out_result = nullptr;
  try {
    mango::PricingParams pp;
    pp.spot = params->spot; pp.strike = params->strike;
    pp.maturity = params->maturity; pp.dividend_yield = params->dividend_yield;
    pp.volatility = params->volatility;
    pp.option_type = to_cpp_type(params->option_type);
    if (!build_rate(params->rate_const, params->tenor_points,
                    params->n_tenor_points, pp.rate, out_err)) {
      return MANGO_ERR_VALIDATION;
    }
    if (!build_dividends(params->dividends, params->n_dividends,
                         pp.discrete_dividends, out_err)) {
      return MANGO_ERR_VALIDATION;
    }

    auto solver = mango::AmericanOptionSolver::create(pp);
    if (!solver) {
      set_err(out_err, MANGO_ERR_VALIDATION, "validation failed");
      return MANGO_ERR_VALIDATION;
    }
    auto solved = solver->solve();
    if (!solved) {
      set_err(out_err, map_solver_error(solved.error()), "solve failed");
      return map_solver_error(solved.error());
    }
    auto& res = solved.value();
    // Eagerly compute the at-spot quantities so the getters are noexcept.
    double v = res.value();
    double d = res.delta();
    double g = res.gamma();
    double t = res.theta();
    *out_result = new MangoAmericanResult{std::move(res), v, d, g, t};
    return MANGO_OK;
  } catch (const std::exception& ex) {
    set_err(out_err, MANGO_ERR_SOLVER, ex.what());
    return MANGO_ERR_SOLVER;
  } catch (...) {
    set_err(out_err, MANGO_ERR_SOLVER, "unknown error");
    return MANGO_ERR_SOLVER;
  }
}

double mango_american_value(const MangoAmericanResult* r) { return r->value; }
double mango_american_delta(const MangoAmericanResult* r) { return r->delta; }
double mango_american_gamma(const MangoAmericanResult* r) { return r->gamma; }
double mango_american_theta(const MangoAmericanResult* r) { return r->theta; }

MangoStatus mango_american_value_at(const MangoAmericanResult* r, double spot,
                                    double* out, MangoError* out_err) {
  if (!r || !out) {
    set_err(out_err, MANGO_ERR_VALIDATION, "null result or out");
    return MANGO_ERR_VALIDATION;
  }
  try {
    *out = r->result.value_at(spot);
    return MANGO_OK;
  } catch (const std::exception& ex) {
    set_err(out_err, MANGO_ERR_SOLVER, ex.what());
    return MANGO_ERR_SOLVER;
  } catch (...) {
    set_err(out_err, MANGO_ERR_SOLVER, "unknown error");
    return MANGO_ERR_SOLVER;
  }
}

void mango_american_result_free(MangoAmericanResult* r) { delete r; }

MangoStatus mango_solve_iv(const MangoIvQuery* query,
                           const MangoIvConfig* config,
                           MangoIvSuccess* out_success,
                           MangoError* out_err) {
  if (!query || !out_success) {
    set_err(out_err, MANGO_ERR_VALIDATION, "null query or out_success");
    return MANGO_ERR_VALIDATION;
  }
  try {
    if (config && config->max_iter < 0) {
      set_err(out_err, MANGO_ERR_VALIDATION, "max_iter must be >= 0");
      return MANGO_ERR_VALIDATION;
    }
    mango::OptionSpec spec;
    spec.spot = query->spot; spec.strike = query->strike;
    spec.maturity = query->maturity; spec.dividend_yield = query->dividend_yield;
    spec.option_type = to_cpp_type(query->option_type);
    if (!build_rate(query->rate_const, query->tenor_points,
                    query->n_tenor_points, spec.rate, out_err)) {
      return MANGO_ERR_VALIDATION;
    }
    mango::IVQuery q(spec, query->market_price);
    if (!build_dividends(query->dividends, query->n_dividends,
                         q.discrete_dividends, out_err)) {
      return MANGO_ERR_VALIDATION;
    }

    mango::IVSolverConfig cfg;
    if (config) {
      if (config->max_iter > 0) {
        cfg.root_config.max_iter = static_cast<std::size_t>(config->max_iter);
      }
      if (config->brent_tol_abs > 0.0) {
        cfg.root_config.brent_tol_abs = config->brent_tol_abs;
      }
    }
    mango::IVSolver solver(cfg);
    auto result = solver.solve(q);
    if (!result) {
      set_err(out_err, map_iv_error(result.error()), "iv solve failed");
      return map_iv_error(result.error());
    }
    const auto& s = result.value();
    out_success->implied_vol = s.implied_vol;
    out_success->iterations = static_cast<uint64_t>(s.iterations);
    out_success->final_error = s.final_error;
    out_success->vega = s.vega.value_or(0.0);
    out_success->has_vega = s.vega.has_value() ? 1 : 0;
    return MANGO_OK;
  } catch (const std::exception& ex) {
    set_err(out_err, MANGO_ERR_SOLVER, ex.what());
    return MANGO_ERR_SOLVER;
  } catch (...) {
    set_err(out_err, MANGO_ERR_SOLVER, "unknown error");
    return MANGO_ERR_SOLVER;
  }
}

}  // extern "C"
```

Note: confirm `RootFindingConfig` field names are `max_iter` and `brent_tol_abs` (per `src/math/root_finding.hpp` and `tests/iv_solver_test.cc`). If `IVErrorCode` lacks any enum named above, adjust the `map_iv_error` switch to the actual enumerators in `src/option/iv_result.hpp` / error types header.

- [ ] **Step 3: Write `src/ffi/BUILD.bazel`**

```python
# SPDX-License-Identifier: MIT
cc_library(
    name = "mango_c_api",
    srcs = ["mango_c_api.cpp"],
    hdrs = ["mango_c_api.h"],
    includes = ["."],
    copts = ["-Wall", "-Wextra", "-O3", "-fopenmp", "-DHAVE_SYSTEMTAP_SDT"],
    linkopts = ["-fopenmp"],
    deps = [
        "//src/option:american_option",
        "//src/option:iv_solver",
    ],
    visibility = ["//visibility:public"],
)
```

Note: the header is included as `mango/ffi/mango_c_api.h`. Confirm how other `//src/...` headers get the `mango/` prefix (the repo maps `src/` → `mango/` via an `include_prefix`/`strip_include_prefix` or a top-level `includes`). Match the existing convention from a neighboring `BUILD.bazel` (e.g. `src/option/BUILD.bazel`): use the same `strip_include_prefix`/`include_prefix` attributes so `#include "mango/ffi/mango_c_api.h"` resolves.

- [ ] **Step 4: Add the test target to `tests/BUILD.bazel`**

```python
cc_test(
    name = "ffi_c_api_test",
    srcs = ["ffi_c_api_test.cc"],
    copts = ["-Wall", "-Wextra", "-fopenmp"],
    linkopts = ["-fopenmp"],
    deps = [
        "//src/ffi:mango_c_api",
        "@googletest//:gtest_main",
    ],
)
```

- [ ] **Step 5: Run the test (build first reveals include-prefix issues)**

Run: `bazel test //tests:ffi_c_api_test --test_output=errors`
Expected: PASS (4 tests). If the header doesn't resolve, fix the `include_prefix` in `src/ffi/BUILD.bazel` to match `src/option/BUILD.bazel` and re-run.

- [ ] **Step 6: Commit**

```bash
git add src/ffi/ tests/ffi_c_api_test.cc tests/BUILD.bazel
git commit -m "Add extern C shim (mango_c_api) over pricing + IV"
```

---

## Task 4: `mango-option-sys` crate (raw FFI + ABI guard)

**Files:**
- Create: `crates/mango-option-sys/src/lib.rs`
- Create: `crates/mango-option-sys/tests/layout.rs`
- Create: `crates/mango-option-sys/Cargo.toml`
- Create: `crates/mango-option-sys/BUILD.bazel`

- [ ] **Step 1: Write `src/lib.rs` (repr(C) mirrors + extern decls)**

```rust
// SPDX-License-Identifier: MIT
//! Raw FFI bindings to the mango-option C ABI (`src/ffi/mango_c_api.h`).
//! 1:1 with the C header; no safety. Use the `mango-option` crate instead.
#![allow(non_camel_case_types)]

pub type MangoStatus = i32;
pub const MANGO_OK: i32 = 0;
pub const MANGO_ERR_VALIDATION: i32 = 1;
pub const MANGO_ERR_ARBITRAGE: i32 = 2;
pub const MANGO_ERR_NO_CONVERGENCE: i32 = 3;
pub const MANGO_ERR_BRACKETING: i32 = 4;
pub const MANGO_ERR_SOLVER: i32 = 5;

pub type MangoOptionType = i32;
pub const MANGO_CALL: i32 = 0;
pub const MANGO_PUT: i32 = 1;

#[repr(C)]
pub struct MangoError {
    pub code: i32,
    pub message: [core::ffi::c_char; 256],
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct MangoDividend {
    pub calendar_time: f64,
    pub amount: f64,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct MangoTenorPoint {
    pub tenor: f64,
    pub log_discount: f64,
}

#[repr(C)]
pub struct MangoPricingParams {
    pub spot: f64,
    pub strike: f64,
    pub maturity: f64,
    pub dividend_yield: f64,
    pub volatility: f64,
    pub rate_const: f64,
    pub tenor_points: *const MangoTenorPoint,
    pub n_tenor_points: u64,
    pub dividends: *const MangoDividend,
    pub n_dividends: u64,
    pub option_type: MangoOptionType,
}

#[repr(C)]
pub struct MangoIvQuery {
    pub spot: f64,
    pub strike: f64,
    pub maturity: f64,
    pub dividend_yield: f64,
    pub market_price: f64,
    pub rate_const: f64,
    pub tenor_points: *const MangoTenorPoint,
    pub n_tenor_points: u64,
    pub dividends: *const MangoDividend,
    pub n_dividends: u64,
    pub option_type: MangoOptionType,
}

#[repr(C)]
pub struct MangoIvSuccess {
    pub implied_vol: f64,
    pub iterations: u64,
    pub final_error: f64,
    pub vega: f64,
    pub has_vega: i32,
}

#[repr(C)]
pub struct MangoIvConfig {
    pub brent_tol_abs: f64,
    pub max_iter: i32,
}

#[repr(C)]
pub struct MangoAmericanResult {
    _private: [u8; 0],
}

extern "C" {
    pub fn mango_price_american(
        params: *const MangoPricingParams,
        out_result: *mut *mut MangoAmericanResult,
        out_err: *mut MangoError,
    ) -> MangoStatus;
    pub fn mango_american_value(r: *const MangoAmericanResult) -> f64;
    pub fn mango_american_delta(r: *const MangoAmericanResult) -> f64;
    pub fn mango_american_gamma(r: *const MangoAmericanResult) -> f64;
    pub fn mango_american_theta(r: *const MangoAmericanResult) -> f64;
    pub fn mango_american_value_at(
        r: *const MangoAmericanResult,
        spot: f64,
        out: *mut f64,
        out_err: *mut MangoError,
    ) -> MangoStatus;
    pub fn mango_american_result_free(r: *mut MangoAmericanResult);
    pub fn mango_solve_iv(
        query: *const MangoIvQuery,
        config: *const MangoIvConfig,
        out_success: *mut MangoIvSuccess,
        out_err: *mut MangoError,
    ) -> MangoStatus;
}
```

- [ ] **Step 2: Write the ABI layout test `tests/layout.rs`**

```rust
// SPDX-License-Identifier: MIT
use core::mem::{align_of, offset_of, size_of};
use mango_option_sys::*;

#[test]
fn pricing_params_layout() {
    assert_eq!(size_of::<MangoPricingParams>(), 88);
    assert_eq!(align_of::<MangoPricingParams>(), 8);
    assert_eq!(offset_of!(MangoPricingParams, rate_const), 40);
    assert_eq!(offset_of!(MangoPricingParams, tenor_points), 48);
    assert_eq!(offset_of!(MangoPricingParams, n_dividends), 72);
    assert_eq!(offset_of!(MangoPricingParams, option_type), 80);
}

#[test]
fn iv_query_layout() {
    assert_eq!(size_of::<MangoIvQuery>(), 88);
    assert_eq!(offset_of!(MangoIvQuery, market_price), 32);
    assert_eq!(offset_of!(MangoIvQuery, option_type), 80);
}

#[test]
fn small_struct_layouts() {
    assert_eq!(size_of::<MangoDividend>(), 16);
    assert_eq!(size_of::<MangoTenorPoint>(), 16);
    assert_eq!(size_of::<MangoError>(), 260);
    assert_eq!(offset_of!(MangoError, message), 4);
    assert_eq!(size_of::<MangoIvSuccess>(), 40);
    assert_eq!(offset_of!(MangoIvSuccess, has_vega), 32);
    assert_eq!(size_of::<MangoIvConfig>(), 16);
    assert_eq!(offset_of!(MangoIvConfig, max_iter), 8);
}
```

- [ ] **Step 3: Write `Cargo.toml`**

```toml
# SPDX-License-Identifier: MIT
[package]
name = "mango-option-sys"
version = "0.1.0"
edition = "2021"
license = "MIT"
description = "Raw FFI bindings to the mango-option C ABI"

[lib]
name = "mango_option_sys"
path = "src/lib.rs"
```

- [ ] **Step 4: Write `BUILD.bazel`**

```python
# SPDX-License-Identifier: MIT
load("@rules_rust//rust:defs.bzl", "rust_library", "rust_test")

rust_library(
    name = "mango_option_sys",
    srcs = ["src/lib.rs"],
    edition = "2021",
    deps = ["//src/ffi:mango_c_api"],
    visibility = ["//visibility:public"],
)

rust_test(
    name = "layout_test",
    srcs = ["tests/layout.rs"],
    edition = "2021",
    deps = [":mango_option_sys"],
)
```

Note: linking a `cc_library` into a `rust_library` — `rules_rust` accepts the `cc_library` in `deps`. If `bazel build` reports the native lib is not linked (unresolved `mango_*` symbols at the integration-test link step in Task 6), move `//src/ffi:mango_c_api` from `deps` to the version-appropriate native attribute (`link_deps`) per the [rules_rust 0.70 docs](https://bazelbuild.github.io/rules_rust/rust.html). The layout test itself does not call any extern fn, so it links without the C lib.

- [ ] **Step 5: Run the layout test**

Run: `bazel test //crates/mango-option-sys:layout_test --test_output=errors`
Expected: PASS (3 tests). A mismatch here means the C header and Rust structs disagree — fix whichever is wrong.

- [ ] **Step 6: Commit**

```bash
git add crates/mango-option-sys/
git commit -m "Add mango-option-sys raw FFI crate with ABI layout guard"
```

---

## Task 5: Safe crate — types and errors

**Files:**
- Create: `crates/mango-option/src/types.rs`
- Create: `crates/mango-option/src/error.rs`
- Create: `crates/mango-option/src/lib.rs`
- Create: `crates/mango-option/Cargo.toml`
- Create: `crates/mango-option/BUILD.bazel`

- [ ] **Step 1: Write `src/error.rs`**

```rust
// SPDX-License-Identifier: MIT
use mango_option_sys as sys;

/// Error category mirroring the C ABI `MangoStatus`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ErrorKind {
    Validation,
    Arbitrage,
    NoConvergence,
    Bracketing,
    Solver,
}

/// A mango-option error: a category plus a (synthesized, possibly truncated)
/// diagnostic message from the C++ side.
#[derive(Debug, Clone)]
pub struct Error {
    pub kind: ErrorKind,
    pub message: String,
}

impl ErrorKind {
    pub(crate) fn from_status(status: i32) -> ErrorKind {
        match status {
            sys::MANGO_ERR_VALIDATION => ErrorKind::Validation,
            sys::MANGO_ERR_ARBITRAGE => ErrorKind::Arbitrage,
            sys::MANGO_ERR_NO_CONVERGENCE => ErrorKind::NoConvergence,
            sys::MANGO_ERR_BRACKETING => ErrorKind::Bracketing,
            _ => ErrorKind::Solver,
        }
    }
}

impl Error {
    /// Build an Error from a non-OK status and a populated MangoError.
    pub(crate) fn from_c(status: i32, err: &sys::MangoError) -> Error {
        // SAFETY: the C side always null-terminates message within 256 bytes.
        let msg = unsafe { core::ffi::CStr::from_ptr(err.message.as_ptr()) }
            .to_string_lossy()
            .into_owned();
        Error { kind: ErrorKind::from_status(status), message: msg }
    }
}

impl core::fmt::Display for Error {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "{:?}: {}", self.kind, self.message)
    }
}

impl std::error::Error for Error {}
```

- [ ] **Step 2: Write `src/types.rs`**

```rust
// SPDX-License-Identifier: MIT
use mango_option_sys as sys;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OptionType {
    Call,
    Put,
}

impl OptionType {
    pub(crate) fn to_c(self) -> sys::MangoOptionType {
        match self {
            OptionType::Call => sys::MANGO_CALL,
            OptionType::Put => sys::MANGO_PUT,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct TenorPoint {
    pub tenor: f64,
    pub log_discount: f64,
}

#[derive(Debug, Clone, Copy)]
pub struct Dividend {
    pub calendar_time: f64,
    pub amount: f64,
}

/// Risk-free rate: a constant, or a yield curve given as tenor points
/// (the first point must be `tenor = 0, log_discount = 0`).
#[derive(Debug, Clone)]
pub enum Rate {
    Const(f64),
    Curve(Vec<TenorPoint>),
}

#[derive(Debug, Clone)]
pub struct OptionSpec {
    pub spot: f64,
    pub strike: f64,
    pub maturity: f64,
    pub dividend_yield: f64,
    pub rate: Rate,
    pub discrete_dividends: Vec<Dividend>,
    pub option_type: OptionType,
}
```

- [ ] **Step 3: Write `src/lib.rs`**

```rust
// SPDX-License-Identifier: MIT
//! Safe Rust bindings for the mango-option American option pricer and FDM
//! implied-vol solver. Callers write no `unsafe`.
mod error;
mod iv;
mod pricing;
mod types;

pub use error::{Error, ErrorKind};
pub use iv::{solve_iv, IvConfig, IvQuery, IvSuccess};
pub use pricing::{price_american, PriceResult, PricingParams};
pub use types::{Dividend, OptionSpec, OptionType, Rate, TenorPoint};
```

(`pricing` and `iv` modules are added in Tasks 6–7; this `lib.rs` references them now so the crate only compiles once those exist. Implement `lib.rs` last within this task ordering, or stub the modules — but the plan adds them next, so build after Task 7.)

- [ ] **Step 4: Write `Cargo.toml`**

```toml
# SPDX-License-Identifier: MIT
[package]
name = "mango-option"
version = "0.1.0"
edition = "2021"
license = "MIT"
description = "Safe Rust bindings for the mango-option American option pricer"

[lib]
name = "mango_option"
path = "src/lib.rs"

[dependencies]
mango-option-sys = { path = "../mango-option-sys" }
```

- [ ] **Step 5: Write `BUILD.bazel`**

```python
# SPDX-License-Identifier: MIT
load("@rules_rust//rust:defs.bzl", "rust_library", "rust_test")

rust_library(
    name = "mango_option",
    srcs = [
        "src/error.rs",
        "src/iv.rs",
        "src/lib.rs",
        "src/pricing.rs",
        "src/types.rs",
    ],
    edition = "2021",
    deps = ["//crates/mango-option-sys:mango_option_sys"],
    visibility = ["//visibility:public"],
)

rust_test(
    name = "integration_test",
    srcs = ["tests/integration.rs"],
    edition = "2021",
    deps = [":mango_option"],
)
```

- [ ] **Step 6: Commit (compiles after Tasks 6–7 add the modules)**

```bash
git add crates/mango-option/
git commit -m "Add mango-option safe crate scaffolding (types, error)"
```

---

## Task 6: Safe pricing API (`price_american`, `PriceResult`)

**Files:**
- Create: `crates/mango-option/src/pricing.rs`

- [ ] **Step 1: Write `src/pricing.rs`**

```rust
// SPDX-License-Identifier: MIT
use crate::error::Error;
use crate::types::{Dividend, OptionSpec, Rate, TenorPoint};
use mango_option_sys as sys;

pub struct PricingParams {
    pub spec: OptionSpec,
    pub volatility: f64,
}

/// Owns a C++ `AmericanOptionResult`. `!Send + !Sync` (raw pointer): a single
/// result must not be used concurrently because `value_at` drives the C++
/// object's lazy, mutable spline/operator caches.
pub struct PriceResult {
    handle: *mut sys::MangoAmericanResult,
}

impl PriceResult {
    pub fn value(&self) -> f64 {
        unsafe { sys::mango_american_value(self.handle) }
    }
    pub fn delta(&self) -> f64 {
        unsafe { sys::mango_american_delta(self.handle) }
    }
    pub fn gamma(&self) -> f64 {
        unsafe { sys::mango_american_gamma(self.handle) }
    }
    pub fn theta(&self) -> f64 {
        unsafe { sys::mango_american_theta(self.handle) }
    }
    pub fn value_at(&self, spot: f64) -> Result<f64, Error> {
        let mut out = 0.0;
        let mut err = blank_error();
        let status = unsafe {
            sys::mango_american_value_at(self.handle, spot, &mut out, &mut err)
        };
        if status == sys::MANGO_OK { Ok(out) } else { Err(Error::from_c(status, &err)) }
    }
}

impl Drop for PriceResult {
    fn drop(&mut self) {
        unsafe { sys::mango_american_result_free(self.handle) }
    }
}

pub fn price_american(params: &PricingParams) -> Result<PriceResult, Error> {
    // Keep arrays alive for the duration of the call.
    let tenors = tenor_array(&params.spec.rate);
    let divs = dividend_array(&params.spec.discrete_dividends);
    let rate_const = match params.spec.rate {
        Rate::Const(r) => r,
        Rate::Curve(_) => 0.0,
    };
    let c = sys::MangoPricingParams {
        spot: params.spec.spot,
        strike: params.spec.strike,
        maturity: params.spec.maturity,
        dividend_yield: params.spec.dividend_yield,
        volatility: params.volatility,
        rate_const,
        tenor_points: ptr_or_null(&tenors),
        n_tenor_points: tenors.len() as u64,
        dividends: div_ptr_or_null(&divs),
        n_dividends: divs.len() as u64,
        option_type: params.spec.option_type.to_c(),
    };
    let mut handle: *mut sys::MangoAmericanResult = core::ptr::null_mut();
    let mut err = blank_error();
    let status = unsafe { sys::mango_price_american(&c, &mut handle, &mut err) };
    if status == sys::MANGO_OK {
        Ok(PriceResult { handle })
    } else {
        Err(Error::from_c(status, &err))
    }
}

// --- shared marshalling helpers (also used by iv.rs) ---

pub(crate) fn blank_error() -> sys::MangoError {
    sys::MangoError { code: 0, message: [0; 256] }
}

pub(crate) fn tenor_array(rate: &Rate) -> Vec<sys::MangoTenorPoint> {
    match rate {
        Rate::Const(_) => Vec::new(),
        Rate::Curve(points) => points
            .iter()
            .map(|p: &TenorPoint| sys::MangoTenorPoint {
                tenor: p.tenor,
                log_discount: p.log_discount,
            })
            .collect(),
    }
}

pub(crate) fn dividend_array(divs: &[Dividend]) -> Vec<sys::MangoDividend> {
    divs.iter()
        .map(|d| sys::MangoDividend { calendar_time: d.calendar_time, amount: d.amount })
        .collect()
}

pub(crate) fn ptr_or_null(v: &[sys::MangoTenorPoint]) -> *const sys::MangoTenorPoint {
    if v.is_empty() { core::ptr::null() } else { v.as_ptr() }
}

pub(crate) fn div_ptr_or_null(v: &[sys::MangoDividend]) -> *const sys::MangoDividend {
    if v.is_empty() { core::ptr::null() } else { v.as_ptr() }
}
```

- [ ] **Step 2: Build the crate (still needs `iv.rs` from Task 7)**

Defer the build/test to Task 7 Step 3, where both modules exist.

- [ ] **Step 3: Commit**

```bash
git add crates/mango-option/src/pricing.rs
git commit -m "Add safe pricing API (price_american, PriceResult)"
```

---

## Task 7: Safe IV API + integration tests

**Files:**
- Create: `crates/mango-option/src/iv.rs`
- Create: `crates/mango-option/tests/integration.rs`

- [ ] **Step 1: Write `src/iv.rs`**

```rust
// SPDX-License-Identifier: MIT
use crate::error::Error;
use crate::pricing::{blank_error, div_ptr_or_null, dividend_array, ptr_or_null, tenor_array};
use crate::types::{OptionSpec, Rate};
use mango_option_sys as sys;

pub struct IvQuery {
    pub spec: OptionSpec,
    pub market_price: f64,
}

#[derive(Default)]
pub struct IvConfig {
    pub max_iter: Option<u32>,
    pub brent_tol_abs: Option<f64>,
}

pub struct IvSuccess {
    pub implied_vol: f64,
    pub iterations: usize,
    pub final_error: f64,
    pub vega: Option<f64>,
}

pub fn solve_iv(query: &IvQuery, config: &IvConfig) -> Result<IvSuccess, Error> {
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
    let cfg = sys::MangoIvConfig {
        brent_tol_abs: config.brent_tol_abs.unwrap_or(0.0),
        max_iter: config.max_iter.map(|m| m as i32).unwrap_or(0),
    };
    let mut out = sys::MangoIvSuccess {
        implied_vol: 0.0,
        iterations: 0,
        final_error: 0.0,
        vega: 0.0,
        has_vega: 0,
    };
    let mut err = blank_error();
    let status = unsafe { sys::mango_solve_iv(&c, &cfg, &mut out, &mut err) };
    if status == sys::MANGO_OK {
        Ok(IvSuccess {
            implied_vol: out.implied_vol,
            iterations: out.iterations as usize,
            final_error: out.final_error,
            vega: if out.has_vega != 0 { Some(out.vega) } else { None },
        })
    } else {
        Err(Error::from_c(status, &err))
    }
}
```

- [ ] **Step 2: Write `tests/integration.rs`**

```rust
// SPDX-License-Identifier: MIT
use mango_option::{
    price_american, solve_iv, Dividend, ErrorKind, IvConfig, IvQuery, OptionSpec,
    OptionType, PricingParams, Rate, TenorPoint,
};

fn put_spec() -> OptionSpec {
    OptionSpec {
        spot: 100.0,
        strike: 100.0,
        maturity: 1.0,
        dividend_yield: 0.0,
        rate: Rate::Const(0.05),
        discrete_dividends: vec![],
        option_type: OptionType::Put,
    }
}

#[test]
fn atm_put_price_and_greeks() {
    let r = price_american(&PricingParams { spec: put_spec(), volatility: 0.20 }).unwrap();
    // Reuse the loose tolerance from tests/american_option_test.cc (~6.35 +/- 0.5).
    assert!((r.value() - 6.35).abs() < 0.5, "value = {}", r.value());
    assert!(r.delta() < 0.0);           // put delta negative
    assert!(r.gamma() > 0.0);
    assert!(r.theta().is_finite());
    let deep = r.value_at(90.0).unwrap();
    assert!(deep > r.value());          // deeper ITM worth more
}

#[test]
fn iv_round_trip() {
    let priced = price_american(&PricingParams { spec: put_spec(), volatility: 0.25 }).unwrap();
    let market = priced.value();
    let s = solve_iv(
        &IvQuery { spec: put_spec(), market_price: market },
        &IvConfig::default(),
    )
    .unwrap();
    assert!((s.implied_vol - 0.25).abs() < 0.01, "iv = {}", s.implied_vol);
}

#[test]
fn discrete_dividend_iv_round_trip() {
    let mut spec = put_spec();
    spec.discrete_dividends = vec![Dividend { calendar_time: 0.5, amount: 2.0 }];
    let market = price_american(&PricingParams { spec: spec.clone(), volatility: 0.25 })
        .unwrap()
        .value();
    let s = solve_iv(
        &IvQuery { spec, market_price: market },
        &IvConfig::default(),
    )
    .unwrap();
    assert!((s.implied_vol - 0.25).abs() < 0.01, "iv = {}", s.implied_vol);
}

#[test]
fn yield_curve_prices() {
    let mut spec = put_spec();
    spec.rate = Rate::Curve(vec![
        TenorPoint { tenor: 0.0, log_discount: 0.0 },
        TenorPoint { tenor: 1.0, log_discount: -0.05 },
    ]);
    let r = price_american(&PricingParams { spec, volatility: 0.20 }).unwrap();
    assert!(r.value() > 0.0 && r.value().is_finite());
}

#[test]
fn invalid_yield_curve_is_validation_error() {
    let mut spec = put_spec();
    // Missing the required tenor=0 anchor point.
    spec.rate = Rate::Curve(vec![TenorPoint { tenor: 1.0, log_discount: -0.05 }]);
    let e = price_american(&PricingParams { spec, volatility: 0.20 }).unwrap_err();
    assert_eq!(e.kind, ErrorKind::Validation);
}

#[test]
fn negative_spot_is_validation_error() {
    let mut spec = put_spec();
    spec.spot = -1.0;
    let e = price_american(&PricingParams { spec, volatility: 0.20 }).unwrap_err();
    assert_eq!(e.kind, ErrorKind::Validation);
}

#[test]
fn arbitrage_violating_price_is_arbitrage_error() {
    // Market price above the strike upper bound for a put.
    let e = solve_iv(
        &IvQuery { spec: put_spec(), market_price: 1000.0 },
        &IvConfig::default(),
    )
    .unwrap_err();
    assert_eq!(e.kind, ErrorKind::Arbitrage);
}
```

- [ ] **Step 3: Build the safe crate**

Run: `bazel build //crates/mango-option:mango_option --test_output=errors 2>&1 | tail -3`
Expected: `Build completed successfully`. If unresolved `mango_*` symbols appear, apply the `link_deps` fix from Task 4 Step 4 note.

- [ ] **Step 4: Run the integration tests**

Run: `bazel test //crates/mango-option:integration_test --test_output=errors`
Expected: PASS (7 tests). If `arbitrage_violating_price_is_arbitrage_error` returns a different `ErrorKind`, adjust the assertion to the kind the C++ `validate_iv_query` actually yields for an over-upper-bound price (it maps to `InvalidMarketPrice` → check whether the shim classifies it Validation or Arbitrage, and assert that).

- [ ] **Step 5: Run the whole suite (no C++ regressions)**

Run: `bazel test //... --test_output=errors 2>&1 | tail -5`
Expected: all tests pass (existing 130 + the new `ffi_c_api_test`, `layout_test`, `integration_test`).

- [ ] **Step 6: Commit**

```bash
git add crates/mango-option/src/iv.rs crates/mango-option/tests/integration.rs
git commit -m "Add safe IV API and Rust integration tests"
```

---

## Task 8: Docs

**Files:**
- Create: `docs/RUST_GUIDE.md`
- Modify: `CLAUDE.md` (Quick Reference table + build target list)

- [ ] **Step 1: Write `docs/RUST_GUIDE.md`**

A short usage guide: the two crates, a `price_american` example, a `solve_iv`
example (constant rate, yield curve, discrete dividends), the error model
(`ErrorKind`), and the `bazel build //crates/...` commands. Mirror the structure
of `docs/PYTHON_GUIDE.md` (read it first for tone/format).

- [ ] **Step 2: Add Rust rows to the `CLAUDE.md` Quick Reference table**

```markdown
| Build Rust binding | `bazel build //crates/mango-option:mango_option` |
| Test Rust binding | `bazel test //crates/mango-option:integration_test` |
```

- [ ] **Step 3: Verify the doc build targets named actually exist**

Run: `bazel query //crates/... 2>&1 | sort`
Expected: lists `mango_option`, `mango_option_sys`, `layout_test`, `integration_test`.

- [ ] **Step 4: Commit**

```bash
git add docs/RUST_GUIDE.md CLAUDE.md
git commit -m "Document the Rust binding"
```

---

## Verification (run before opening the binding PR)

1. `bazel test //... --test_output=errors` → all pass (C++ 130 + 3 new Rust/FFI targets).
2. `bazel build //benchmarks/...` → builds.
3. `bazel build //src/python:mango_option` → builds.
4. `bazel build //crates/...` → builds.
5. Confirm no `unsafe` leaks into the public `mango-option` API (only `-sys` and the safe crate's internals use `unsafe`).
