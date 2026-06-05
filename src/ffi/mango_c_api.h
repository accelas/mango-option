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
  int32_t used_rate_approximation;
} MangoIvSuccess;

typedef struct { double brent_tol_abs; int32_t max_iter; } MangoIvConfig;  // 0 => default

// --- Interpolation path (B-spline backend) ---

// InterpolatedIVSolverConfig (Newton config). Fields applied VERBATIM: a
// vega_threshold of 0 disables the vega pre-check, so zero cannot mean "unset".
typedef struct {
  uint64_t max_iter;
  double tolerance;
  double sigma_min;
  double sigma_max;
  double vega_threshold;
} MangoInterpSolverConfig;

// AdaptiveGridParams (all 9 fields; passed only when non-null).
typedef struct {
  double target_iv_error;
  uint64_t max_iter;
  uint64_t max_points_per_dim;
  uint64_t min_moneyness_points;
  uint64_t validation_samples;
  double refinement_factor;
  uint64_t lhs_seed;
  double vega_floor;
  double max_failure_rate;
} MangoAdaptiveGridParams;

// MultiKRefConfig.
typedef struct {
  const double* K_refs;       // may be null when n_K_refs == 0 (auto mode)
  uint64_t n_K_refs;
  int32_t K_ref_count;        // used iff K_refs empty; must be >= 1 (auto: 11)
  double K_ref_span;
} MangoMultiKRef;

// DiscreteDividendConfig.
typedef struct {
  double maturity;
  const MangoDividend* dividends;   // may be null when n_dividends == 0
  uint64_t n_dividends;
  MangoMultiKRef kref_config;
} MangoDiscreteDividendConfig;

// IVSolverFactoryConfig (B-spline backend only).
typedef struct {
  MangoOptionType option_type;
  double spot;
  double dividend_yield;
  const double* moneyness;
  uint64_t n_moneyness;
  const double* vol;
  uint64_t n_vol;
  const double* rate;
  uint64_t n_rate;
  const double* maturity_grid;
  uint64_t n_maturity;
  MangoInterpSolverConfig solver_config;
  const MangoAdaptiveGridParams* adaptive;              // null => fixed grid
  const MangoDiscreteDividendConfig* discrete_dividends; // null => continuous
} MangoIvFactoryConfig;

// Per-query batch result slot (caller-allocated array of length n).
typedef struct {
  int32_t status;          // MangoStatus: MANGO_OK or an error category
  MangoIvSuccess success;  // valid iff status == MANGO_OK
} MangoIvBatchSlot;

typedef struct MangoInterpIvSolver MangoInterpIvSolver;  // opaque
typedef struct MangoPriceTable MangoPriceTable;          // opaque

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

// Interpolated IV solver.
MangoStatus mango_make_interp_iv_solver(const MangoIvFactoryConfig* cfg,
                                        MangoInterpIvSolver** out, MangoError* err);
MangoStatus mango_interp_iv_solve(const MangoInterpIvSolver* s,
                                  const MangoIvQuery* q,
                                  MangoIvSuccess* out, MangoError* err);
MangoStatus mango_interp_iv_solve_batch(const MangoInterpIvSolver* s,
                                        const MangoIvQuery* queries, uint64_t n,
                                        MangoIvBatchSlot* out_slots,
                                        uint64_t* out_failed_count, MangoError* err);
void mango_interp_iv_solver_free(MangoInterpIvSolver* s);

// Price table (type-erased AnyPriceTable).
MangoStatus mango_make_price_table(const MangoIvFactoryConfig* cfg,
                                   MangoPriceTable** out, MangoError* err);
MangoStatus mango_price_table_validate(const MangoPriceTable* t,
                                       const MangoPricingParams* p, MangoError* err);
double mango_price_table_price(const MangoPriceTable* t, const MangoPricingParams* p);
double mango_price_table_vega(const MangoPriceTable* t, const MangoPricingParams* p);
MangoStatus mango_price_table_delta(const MangoPriceTable* t, const MangoPricingParams* p,
                                    double* out, MangoError* err);
MangoStatus mango_price_table_gamma(const MangoPriceTable* t, const MangoPricingParams* p,
                                    double* out, MangoError* err);
MangoStatus mango_price_table_theta(const MangoPriceTable* t, const MangoPricingParams* p,
                                    double* out, MangoError* err);
MangoStatus mango_price_table_rho(const MangoPriceTable* t, const MangoPricingParams* p,
                                  double* out, MangoError* err);
MangoOptionType mango_price_table_option_type(const MangoPriceTable* t);
double mango_price_table_dividend_yield(const MangoPriceTable* t);
MangoStatus mango_price_table_make_iv_solver(const MangoPriceTable* t,
                                             const MangoInterpSolverConfig* cfg,
                                             MangoInterpIvSolver** out, MangoError* err);
void mango_price_table_free(MangoPriceTable* t);

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
static_assert(offsetof(MangoIvSuccess, used_rate_approximation) == 36, "used_rate_approximation off");
static_assert(sizeof(MangoIvConfig) == 16, "MangoIvConfig size");
static_assert(offsetof(MangoIvConfig, max_iter) == 8, "max_iter off");
static_assert(sizeof(MangoInterpSolverConfig) == 40, "MangoInterpSolverConfig size");
static_assert(offsetof(MangoInterpSolverConfig, max_iter) == 0, "interp max_iter off");
static_assert(offsetof(MangoInterpSolverConfig, tolerance) == 8, "interp tolerance off");
static_assert(offsetof(MangoInterpSolverConfig, sigma_min) == 16, "interp sigma_min off");
static_assert(offsetof(MangoInterpSolverConfig, sigma_max) == 24, "interp sigma_max off");
static_assert(offsetof(MangoInterpSolverConfig, vega_threshold) == 32, "interp vega_threshold off");
static_assert(sizeof(MangoAdaptiveGridParams) == 72, "MangoAdaptiveGridParams size");
static_assert(offsetof(MangoAdaptiveGridParams, target_iv_error) == 0, "adaptive target_iv_error off");
static_assert(offsetof(MangoAdaptiveGridParams, max_iter) == 8, "adaptive max_iter off");
static_assert(offsetof(MangoAdaptiveGridParams, max_points_per_dim) == 16, "adaptive max_points_per_dim off");
static_assert(offsetof(MangoAdaptiveGridParams, min_moneyness_points) == 24, "adaptive min_moneyness_points off");
static_assert(offsetof(MangoAdaptiveGridParams, validation_samples) == 32, "adaptive validation_samples off");
static_assert(offsetof(MangoAdaptiveGridParams, refinement_factor) == 40, "adaptive refinement_factor off");
static_assert(offsetof(MangoAdaptiveGridParams, lhs_seed) == 48, "adaptive lhs_seed off");
static_assert(offsetof(MangoAdaptiveGridParams, vega_floor) == 56, "adaptive vega_floor off");
static_assert(offsetof(MangoAdaptiveGridParams, max_failure_rate) == 64, "adaptive max_failure_rate off");
static_assert(sizeof(MangoMultiKRef) == 32, "MangoMultiKRef size");
static_assert(offsetof(MangoMultiKRef, K_refs) == 0, "K_refs off");
static_assert(offsetof(MangoMultiKRef, n_K_refs) == 8, "n_K_refs off");
static_assert(offsetof(MangoMultiKRef, K_ref_count) == 16, "K_ref_count off");
static_assert(offsetof(MangoMultiKRef, K_ref_span) == 24, "K_ref_span off");
static_assert(sizeof(MangoDiscreteDividendConfig) == 56, "MangoDiscreteDividendConfig size");
static_assert(offsetof(MangoDiscreteDividendConfig, maturity) == 0, "dd maturity off");
static_assert(offsetof(MangoDiscreteDividendConfig, dividends) == 8, "dd dividends off");
static_assert(offsetof(MangoDiscreteDividendConfig, n_dividends) == 16, "dd n_dividends off");
static_assert(offsetof(MangoDiscreteDividendConfig, kref_config) == 24, "dd kref_config off");
static_assert(sizeof(MangoIvFactoryConfig) == 144, "MangoIvFactoryConfig size");
static_assert(offsetof(MangoIvFactoryConfig, option_type) == 0, "fc option_type off");
static_assert(offsetof(MangoIvFactoryConfig, spot) == 8, "fc spot off");
static_assert(offsetof(MangoIvFactoryConfig, dividend_yield) == 16, "fc dividend_yield off");
static_assert(offsetof(MangoIvFactoryConfig, moneyness) == 24, "fc moneyness off");
static_assert(offsetof(MangoIvFactoryConfig, n_moneyness) == 32, "fc n_moneyness off");
static_assert(offsetof(MangoIvFactoryConfig, vol) == 40, "fc vol off");
static_assert(offsetof(MangoIvFactoryConfig, n_vol) == 48, "fc n_vol off");
static_assert(offsetof(MangoIvFactoryConfig, rate) == 56, "fc rate off");
static_assert(offsetof(MangoIvFactoryConfig, n_rate) == 64, "fc n_rate off");
static_assert(offsetof(MangoIvFactoryConfig, maturity_grid) == 72, "fc maturity_grid off");
static_assert(offsetof(MangoIvFactoryConfig, n_maturity) == 80, "fc n_maturity off");
static_assert(offsetof(MangoIvFactoryConfig, solver_config) == 88, "fc solver_config off");
static_assert(offsetof(MangoIvFactoryConfig, adaptive) == 128, "fc adaptive off");
static_assert(offsetof(MangoIvFactoryConfig, discrete_dividends) == 136, "fc discrete_dividends off");
static_assert(sizeof(MangoIvBatchSlot) == 48, "MangoIvBatchSlot size");
static_assert(offsetof(MangoIvBatchSlot, status) == 0, "batch status off");
static_assert(offsetof(MangoIvBatchSlot, success) == 8, "batch success off");
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
_Static_assert(offsetof(MangoIvSuccess, used_rate_approximation) == 36, "used_rate_approximation off");
_Static_assert(sizeof(MangoIvConfig) == 16, "MangoIvConfig size");
_Static_assert(offsetof(MangoIvConfig, max_iter) == 8, "max_iter off");
_Static_assert(sizeof(MangoInterpSolverConfig) == 40, "MangoInterpSolverConfig size");
_Static_assert(offsetof(MangoInterpSolverConfig, max_iter) == 0, "interp max_iter off");
_Static_assert(offsetof(MangoInterpSolverConfig, tolerance) == 8, "interp tolerance off");
_Static_assert(offsetof(MangoInterpSolverConfig, sigma_min) == 16, "interp sigma_min off");
_Static_assert(offsetof(MangoInterpSolverConfig, sigma_max) == 24, "interp sigma_max off");
_Static_assert(offsetof(MangoInterpSolverConfig, vega_threshold) == 32, "interp vega_threshold off");
_Static_assert(sizeof(MangoAdaptiveGridParams) == 72, "MangoAdaptiveGridParams size");
_Static_assert(offsetof(MangoAdaptiveGridParams, target_iv_error) == 0, "adaptive target_iv_error off");
_Static_assert(offsetof(MangoAdaptiveGridParams, max_iter) == 8, "adaptive max_iter off");
_Static_assert(offsetof(MangoAdaptiveGridParams, max_points_per_dim) == 16, "adaptive max_points_per_dim off");
_Static_assert(offsetof(MangoAdaptiveGridParams, min_moneyness_points) == 24, "adaptive min_moneyness_points off");
_Static_assert(offsetof(MangoAdaptiveGridParams, validation_samples) == 32, "adaptive validation_samples off");
_Static_assert(offsetof(MangoAdaptiveGridParams, refinement_factor) == 40, "adaptive refinement_factor off");
_Static_assert(offsetof(MangoAdaptiveGridParams, lhs_seed) == 48, "adaptive lhs_seed off");
_Static_assert(offsetof(MangoAdaptiveGridParams, vega_floor) == 56, "adaptive vega_floor off");
_Static_assert(offsetof(MangoAdaptiveGridParams, max_failure_rate) == 64, "adaptive max_failure_rate off");
_Static_assert(sizeof(MangoMultiKRef) == 32, "MangoMultiKRef size");
_Static_assert(offsetof(MangoMultiKRef, K_refs) == 0, "K_refs off");
_Static_assert(offsetof(MangoMultiKRef, n_K_refs) == 8, "n_K_refs off");
_Static_assert(offsetof(MangoMultiKRef, K_ref_count) == 16, "K_ref_count off");
_Static_assert(offsetof(MangoMultiKRef, K_ref_span) == 24, "K_ref_span off");
_Static_assert(sizeof(MangoDiscreteDividendConfig) == 56, "MangoDiscreteDividendConfig size");
_Static_assert(offsetof(MangoDiscreteDividendConfig, maturity) == 0, "dd maturity off");
_Static_assert(offsetof(MangoDiscreteDividendConfig, dividends) == 8, "dd dividends off");
_Static_assert(offsetof(MangoDiscreteDividendConfig, n_dividends) == 16, "dd n_dividends off");
_Static_assert(offsetof(MangoDiscreteDividendConfig, kref_config) == 24, "dd kref_config off");
_Static_assert(sizeof(MangoIvFactoryConfig) == 144, "MangoIvFactoryConfig size");
_Static_assert(offsetof(MangoIvFactoryConfig, option_type) == 0, "fc option_type off");
_Static_assert(offsetof(MangoIvFactoryConfig, spot) == 8, "fc spot off");
_Static_assert(offsetof(MangoIvFactoryConfig, dividend_yield) == 16, "fc dividend_yield off");
_Static_assert(offsetof(MangoIvFactoryConfig, moneyness) == 24, "fc moneyness off");
_Static_assert(offsetof(MangoIvFactoryConfig, n_moneyness) == 32, "fc n_moneyness off");
_Static_assert(offsetof(MangoIvFactoryConfig, vol) == 40, "fc vol off");
_Static_assert(offsetof(MangoIvFactoryConfig, n_vol) == 48, "fc n_vol off");
_Static_assert(offsetof(MangoIvFactoryConfig, rate) == 56, "fc rate off");
_Static_assert(offsetof(MangoIvFactoryConfig, n_rate) == 64, "fc n_rate off");
_Static_assert(offsetof(MangoIvFactoryConfig, maturity_grid) == 72, "fc maturity_grid off");
_Static_assert(offsetof(MangoIvFactoryConfig, n_maturity) == 80, "fc n_maturity off");
_Static_assert(offsetof(MangoIvFactoryConfig, solver_config) == 88, "fc solver_config off");
_Static_assert(offsetof(MangoIvFactoryConfig, adaptive) == 128, "fc adaptive off");
_Static_assert(offsetof(MangoIvFactoryConfig, discrete_dividends) == 136, "fc discrete_dividends off");
_Static_assert(sizeof(MangoIvBatchSlot) == 48, "MangoIvBatchSlot size");
_Static_assert(offsetof(MangoIvBatchSlot, status) == 0, "batch status off");
_Static_assert(offsetof(MangoIvBatchSlot, success) == 8, "batch success off");
#endif

#endif  // MANGO_C_API_H
