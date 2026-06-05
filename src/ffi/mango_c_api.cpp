// SPDX-License-Identifier: MIT
#include "mango/ffi/mango_c_api.h"

#include "mango/option/american_option.hpp"
#include "mango/option/interpolated_iv_solver.hpp"
#include "mango/option/iv_solver.hpp"
#include "mango/option/option_spec.hpp"
#include "mango/option/price_table_factory.hpp"
#include "mango/option/yield_curve.hpp"
#include "mango/support/error_types.hpp"

#include <cmath>
#include <cstring>
#include <exception>
#include <new>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

namespace {

void set_err(MangoError* err, MangoStatus code, const char* msg) {
  if (!err) return;
  err->code = code;
  if (!msg) { err->message[0] = '\0'; return; }
  std::size_t n = std::strlen(msg);
  if (n > 255) n = 255;
  std::memcpy(err->message, msg, n);
  err->message[n] = '\0';
}

// Validate and convert MangoOptionType. Returns false and fills err when invalid.
bool validate_option_type(MangoOptionType t, MangoError* err,
                          mango::OptionType& out) {
  if (t == MANGO_CALL) { out = mango::OptionType::CALL; return true; }
  if (t == MANGO_PUT)  { out = mango::OptionType::PUT;  return true; }
  set_err(err, MANGO_ERR_VALIDATION, "invalid option_type");
  return false;
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
    std::string msg = "invalid yield curve: " + curve.error();
    set_err(err, MANGO_ERR_VALIDATION, msg.c_str());
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

// Build a C++ PricingParams from the C struct (rate + dividends + option type).
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

MangoStatus map_solver_error(const mango::SolverError& e) {
  if (e.code == mango::SolverErrorCode::ConvergenceFailure)
    return MANGO_ERR_NO_CONVERGENCE;
  return MANGO_ERR_SOLVER;
}

std::string format_solver_error(const mango::SolverError& e) {
  std::ostringstream m;
  m << e;  // SolverError has operator<<
  return m.str();
}

std::string format_validation_error(const mango::ValidationError& e) {
  std::ostringstream m;
  m << e;  // ValidationError has operator<<
  return m.str();
}

std::string format_iv_error(const mango::IVError& e) {
  std::ostringstream m;
  m << "IVError{code=" << static_cast<int>(e.code)
    << ", iterations=" << e.iterations
    << ", final_error=" << e.final_error;
  if (e.last_vol.has_value()) {
    m << ", last_vol=" << *e.last_vol;
  }
  m << "}";
  return m.str();
}

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

void fill_iv_success(MangoIvSuccess* out, const mango::IVSuccess& s) {
  out->implied_vol = s.implied_vol;
  out->iterations = static_cast<uint64_t>(s.iterations);
  out->final_error = s.final_error;
  out->vega = s.vega.value_or(0.0);
  out->has_vega = s.vega.has_value() ? 1 : 0;
  out->used_rate_approximation = s.used_rate_approximation ? 1 : 0;
}

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

}  // namespace

struct MangoAmericanResult {
  mango::AmericanOptionResult result;
  double value;
  double delta;
  double gamma;
  double theta;
};

struct MangoInterpIvSolver { mango::AnyInterpIVSolver solver; };
struct MangoPriceTable { mango::AnyPriceTable table; };

namespace {
// Shared body for the four fallible Greeks. `fn` returns a
// std::expected<double, GreekError> from the wrapped AnyPriceTable.
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
    if (!validate_option_type(params->option_type, out_err, pp.option_type))
      return MANGO_ERR_VALIDATION;
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
      std::string msg = format_validation_error(solver.error());
      set_err(out_err, MANGO_ERR_VALIDATION, msg.c_str());
      return MANGO_ERR_VALIDATION;
    }
    auto solved = solver->solve();
    if (!solved) {
      auto code = map_solver_error(solved.error());
      std::string msg = format_solver_error(solved.error());
      set_err(out_err, code, msg.c_str());
      return code;
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

double mango_american_value(const MangoAmericanResult* r) { return r ? r->value : std::nan(""); }
double mango_american_delta(const MangoAmericanResult* r) { return r ? r->delta : std::nan(""); }
double mango_american_gamma(const MangoAmericanResult* r) { return r ? r->gamma : std::nan(""); }
double mango_american_theta(const MangoAmericanResult* r) { return r ? r->theta : std::nan(""); }

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
    if (!validate_option_type(query->option_type, out_err, spec.option_type))
      return MANGO_ERR_VALIDATION;
    if (!build_rate(query->rate_const, query->tenor_points,
                    query->n_tenor_points, spec.rate, out_err)) {
      return MANGO_ERR_VALIDATION;
    }
    mango::IVQuery q(spec, query->market_price);
    if (!build_dividends(query->dividends, query->n_dividends,
                         q.discrete_dividends, out_err)) {
      return MANGO_ERR_VALIDATION;
    }

    // Pre-validate scalar rate/dividend so non-finite inputs surface as a
    // validation error. The C++ IV path maps InvalidRate/InvalidDividend to
    // ArbitrageViolation; without this the Rust API would report
    // ErrorKind::Arbitrage for invalid params, unlike the pricing path.
    if (!std::isfinite(query->dividend_yield)) {
      set_err(out_err, MANGO_ERR_VALIDATION, "dividend_yield must be finite");
      return MANGO_ERR_VALIDATION;
    }
    if (query->n_tenor_points == 0 && !std::isfinite(query->rate_const)) {
      set_err(out_err, MANGO_ERR_VALIDATION, "rate must be finite");
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
    for (uint64_t i = 0; i < query->n_dividends; ++i) {
      double ct = query->dividends[i].calendar_time;
      double amt = query->dividends[i].amount;
      if (!std::isfinite(ct) || ct < 0.0 || ct > query->maturity ||
          !std::isfinite(amt) || amt < 0.0) {
        set_err(out_err, MANGO_ERR_VALIDATION, "invalid discrete dividend");
        return MANGO_ERR_VALIDATION;
      }
    }
    mango::IVSolver solver(cfg);
    auto result = solver.solve(q);
    if (!result) {
      auto code = map_iv_error(result.error());
      std::string msg = format_iv_error(result.error());
      set_err(out_err, code, msg.c_str());
      return code;
    }
    const auto& s = result.value();
    fill_iv_success(out_success, s);
    return MANGO_OK;
  } catch (const std::exception& ex) {
    set_err(out_err, MANGO_ERR_SOLVER, ex.what());
    return MANGO_ERR_SOLVER;
  } catch (...) {
    set_err(out_err, MANGO_ERR_SOLVER, "unknown error");
    return MANGO_ERR_SOLVER;
  }
}

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

MangoStatus mango_price_table_delta(const MangoPriceTable* t, const MangoPricingParams* p,
                                    double* out, MangoError* err) {
  return greek_call(t, p, out, err,
                    [](const mango::AnyPriceTable& tb, const mango::PricingParams& pp) {
                      return tb.delta(pp);
                    });
}

MangoStatus mango_price_table_gamma(const MangoPriceTable* t, const MangoPricingParams* p,
                                    double* out, MangoError* err) {
  return greek_call(t, p, out, err,
                    [](const mango::AnyPriceTable& tb, const mango::PricingParams& pp) {
                      return tb.gamma(pp);
                    });
}

MangoStatus mango_price_table_theta(const MangoPriceTable* t, const MangoPricingParams* p,
                                    double* out, MangoError* err) {
  return greek_call(t, p, out, err,
                    [](const mango::AnyPriceTable& tb, const mango::PricingParams& pp) {
                      return tb.theta(pp);
                    });
}

MangoStatus mango_price_table_rho(const MangoPriceTable* t, const MangoPricingParams* p,
                                  double* out, MangoError* err) {
  return greek_call(t, p, out, err,
                    [](const mango::AnyPriceTable& tb, const mango::PricingParams& pp) {
                      return tb.rho(pp);
                    });
}

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

}  // extern "C"
