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
