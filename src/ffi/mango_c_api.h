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

#ifdef __cplusplus
}  // extern "C"
#endif

// --- ABI guards: any field reorder/resize breaks the build (mirrored in Rust) ---
// Use _Static_assert (C11) in C, static_assert (C++11) in C++.
#ifdef __cplusplus
static_assert(sizeof(MangoPricingParams) == 88, "MangoPricingParams size");
static_assert(offsetof(MangoPricingParams, rate_const) == 40, "rate_const off");
static_assert(offsetof(MangoPricingParams, tenor_points) == 48, "tenor_points off");
static_assert(offsetof(MangoPricingParams, n_dividends) == 72, "n_dividends off");
static_assert(offsetof(MangoPricingParams, option_type) == 80, "option_type off");
static_assert(sizeof(MangoIvQuery) == 88, "MangoIvQuery size");
static_assert(offsetof(MangoIvQuery, market_price) == 32, "market_price off");
static_assert(offsetof(MangoIvQuery, tenor_points) == 48, "iv tenor_points off");
static_assert(offsetof(MangoIvQuery, n_tenor_points) == 56, "iv n_tenor_points off");
static_assert(offsetof(MangoIvQuery, dividends) == 64, "iv dividends off");
static_assert(offsetof(MangoIvQuery, n_dividends) == 72, "iv n_dividends off");
static_assert(offsetof(MangoIvQuery, option_type) == 80, "iv option_type off");
static_assert(sizeof(MangoDividend) == 16, "MangoDividend size");
static_assert(sizeof(MangoTenorPoint) == 16, "MangoTenorPoint size");
static_assert(sizeof(MangoError) == 260, "MangoError size");
static_assert(offsetof(MangoError, message) == 4, "message off");
static_assert(sizeof(MangoIvSuccess) == 40, "MangoIvSuccess size");
static_assert(offsetof(MangoIvSuccess, has_vega) == 32, "has_vega off");
static_assert(sizeof(MangoIvConfig) == 16, "MangoIvConfig size");
static_assert(offsetof(MangoIvConfig, max_iter) == 8, "max_iter off");
#else
_Static_assert(sizeof(MangoPricingParams) == 88, "MangoPricingParams size");
_Static_assert(offsetof(MangoPricingParams, rate_const) == 40, "rate_const off");
_Static_assert(offsetof(MangoPricingParams, tenor_points) == 48, "tenor_points off");
_Static_assert(offsetof(MangoPricingParams, n_dividends) == 72, "n_dividends off");
_Static_assert(offsetof(MangoPricingParams, option_type) == 80, "option_type off");
_Static_assert(sizeof(MangoIvQuery) == 88, "MangoIvQuery size");
_Static_assert(offsetof(MangoIvQuery, market_price) == 32, "market_price off");
_Static_assert(offsetof(MangoIvQuery, tenor_points) == 48, "iv tenor_points off");
_Static_assert(offsetof(MangoIvQuery, n_tenor_points) == 56, "iv n_tenor_points off");
_Static_assert(offsetof(MangoIvQuery, dividends) == 64, "iv dividends off");
_Static_assert(offsetof(MangoIvQuery, n_dividends) == 72, "iv n_dividends off");
_Static_assert(offsetof(MangoIvQuery, option_type) == 80, "iv option_type off");
_Static_assert(sizeof(MangoDividend) == 16, "MangoDividend size");
_Static_assert(sizeof(MangoTenorPoint) == 16, "MangoTenorPoint size");
_Static_assert(sizeof(MangoError) == 260, "MangoError size");
_Static_assert(offsetof(MangoError, message) == 4, "message off");
_Static_assert(sizeof(MangoIvSuccess) == 40, "MangoIvSuccess size");
_Static_assert(offsetof(MangoIvSuccess, has_vega) == 32, "has_vega off");
_Static_assert(sizeof(MangoIvConfig) == 16, "MangoIvConfig size");
_Static_assert(offsetof(MangoIvConfig, max_iter) == 8, "max_iter off");
#endif

#endif  // MANGO_C_API_H
