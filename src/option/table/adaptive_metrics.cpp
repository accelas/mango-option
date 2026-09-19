// SPDX-License-Identifier: MIT
#include "mango/option/table/adaptive_metrics.hpp"
#include "mango/option/american_option.hpp"
#include "mango/option/dividend_utils.hpp"
#include "mango/option/surface_inversion.hpp"
#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>
#include <optional>
#include <utility>

namespace mango {

namespace {

// Rebuild the same generator family at a different point count.  Every
// GridSpec generator is a pure map of eta = i/(n-1) (grid.hpp generate()),
// so re-sampling at (n-1)/2^k + 1 points yields the every-2^k-th-node
// subsequence exactly (up to floating-point rounding of eta).
std::expected<GridSpec<double>, ValidationError>
resample(const GridSpec<double>& spec, size_t n) {
    switch (spec.type()) {
        case GridSpec<double>::Type::MultiSinhSpaced: {
            std::vector<MultiSinhCluster<double>> clusters(
                spec.clusters().begin(), spec.clusters().end());
            // auto_merge=false: the clusters were merged when the estimator
            // built the fine spec; merging again could move them.
            return GridSpec<double>::multi_sinh_spaced(
                spec.x_min(), spec.x_max(), n, std::move(clusters), /*auto_merge=*/false);
        }
        case GridSpec<double>::Type::SinhSpaced:
            return GridSpec<double>::sinh_spaced(spec.x_min(), spec.x_max(), n, spec.concentration());
        case GridSpec<double>::Type::Uniform:
            return GridSpec<double>::uniform(spec.x_min(), spec.x_max(), n);
        case GridSpec<double>::Type::LogSpaced:
            return GridSpec<double>::log_spaced(spec.x_min(), spec.x_max(), n);
    }
    return std::unexpected(ValidationError(ValidationErrorCode::InvalidGridSize, static_cast<double>(n)));
}

constexpr size_t kFamilyModulus = 16;  // odd through three coarsenings (spec D1)

// Shared create -> check -> solve -> check sequence for both ReferenceOracle
// solve variants (explicit grid and auto-estimated grid).
std::expected<double, SolverError> solve_pde_grid_spec(const PricingParams& p, PDEGridSpec spec) {
    auto solver = AmericanOptionSolver::create(p, std::move(spec));
    if (!solver) return std::unexpected(SolverError{.code = SolverErrorCode::InvalidConfiguration});
    auto r = solver->solve();
    if (!r) return std::unexpected(r.error());
    const double v = r->value();
    if (!std::isfinite(v)) return std::unexpected(SolverError{.code = SolverErrorCode::NonFiniteSolution});
    return v;
}

}  // namespace

std::expected<ReferenceGridFamily, ValidationError>
make_reference_grid_family(const PricingParams& params,
                           const GridAccuracyParams& accuracy,
                           size_t levels) {
    auto est = estimate_pde_grid(params, accuracy);
    if (!est) return std::unexpected(est.error());
    const auto& [spec0, time0] = *est;
    const size_t n0 = spec0.n_points();
    const size_t cap = accuracy.max_spatial_points;
    const size_t floor = std::max<size_t>(accuracy.min_spatial_points, 3);
    ReferenceGridFamily fam;
    // Smallest n >= n0 with n = 1 (mod 16), if it fits under the strict cap.
    size_t n = n0 + ((kFamilyModulus + 1 - (n0 % kFamilyModulus)) % kFamilyModulus);
    if (n > cap) {
        // Largest n <= cap with n = 1 (mod 16) that is still >= floor.
        n = cap - ((cap % kFamilyModulus) + kFamilyModulus - 1) % kFamilyModulus;
        if (n < floor || n < kFamilyModulus + 1) {
            return std::unexpected(ValidationError(
                ValidationErrorCode::InvalidGridSize, static_cast<double>(cap)));
        }
        fam.rounded_down = true;
    }
    const size_t n_time0 = time0.n_steps();
    for (size_t k = 0; k <= levels; ++k) {
        const size_t nk = ((n - 1) >> k) + 1;
        auto spec = resample(spec0, nk);
        if (!spec) return std::unexpected(spec.error());
        const size_t tk = (n_time0 + (size_t{1} << k) - 1) >> k;
        // mandatory_times stays empty: resolve_grid merges the dividend taus
        // into every explicit config (american_option.cpp:68), and copying
        // fine time nodes here would stop the coarse level from coarsening.
        fam.levels.push_back(PDEGridConfig{.grid_spec = std::move(*spec),
                                           .n_time = tk,
                                           .mandatory_times = {}});
        fam.point_counts.push_back(nk);
        fam.time_steps.push_back(tk);
    }
    return fam;
}

PricingParams ReferenceOracle::contract(double spot, double strike, double tau,
                                        double sigma, double rate) const {
    PricingParams p;
    p.spot = spot; p.strike = strike; p.maturity = tau; p.rate = rate;
    p.dividend_yield = dividend_yield; p.option_type = option_type;
    p.volatility = sigma;
    p.discrete_dividends = reference_maturity
        ? rolled_dividends(discrete_dividends, *reference_maturity, tau)
        : filter_and_merge_dividends(discrete_dividends, tau);
    return p;
}

std::expected<double, SolverError>
ReferenceOracle::solve(const PricingParams& p, const PDEGridConfig& grid) const {
    return solve_pde_grid_spec(p, PDEGridSpec{grid});
}

std::expected<double, SolverError>
ReferenceOracle::solve_estimated(const PricingParams& p) const {
    return solve_pde_grid_spec(p, PDEGridSpec{accuracy});
}

namespace {

// The inversion's own failure codes, restated as per-point outcomes of the
// shipped inversion (spec D3).
PointStatus status_of(const IVError& e) {
    switch (e.code) {
        case IVErrorCode::VegaTooSmall:          return PointStatus::SurfaceVegaTooSmall;
        case IVErrorCode::BracketingFailed:      return PointStatus::SurfaceNoRoot;
        case IVErrorCode::MultipleRoots:         return PointStatus::SurfaceAmbiguous;
        case IVErrorCode::MaxIterationsExceeded: return PointStatus::SurfaceNonConvergent;
        // NumericalInstability and any invariant violation: the surface
        // produced something the inversion could not use as a number.
        default:                                 return PointStatus::SurfaceNonFinite;
    }
}

// Reporting order across the three targets (spec D3): the worst outcome is
// the one the point is recorded under.  Higher wins.
int severity(PointStatus s) {
    switch (s) {
        case PointStatus::SurfaceNonFinite:     return 5;
        case PointStatus::SurfaceNonConvergent: return 4;
        case PointStatus::SurfaceAmbiguous:     return 3;
        case PointStatus::SurfaceNoRoot:        return 2;
        case PointStatus::SurfaceVegaTooSmall:  return 1;
        default:                                return 0;
    }
}

}  // namespace

// `ctx.option_type` is deliberately not read: `option_type` is the
// authoritative one, so a caller can score a surface for an option type the
// refinement context was not built around.
ScoreErrorFn make_round_trip_score_fn(const AdaptiveGridParams& params,
                                      const RefinementContext& ctx,
                                      OptionType option_type) {
    const double tau_iv = params.target_iv_error;
    const SurfaceBounds sample = ctx.sample_bounds;
    const SurfaceBounds fit = ctx.bounds;
    return [tau_iv, sample, fit, option_type](
        const SurfaceHandle& surface, const ErrorRefs& refs,
        double spot, double strike, double tau, double sigma, double rate) -> PointScore
    {
        PointScore out;
        if (surface.price) {
            const double surface_price = surface.price(spot, strike, tau, sigma, rate);
            if (std::isfinite(surface_price) && std::isfinite(refs.ref_price)) {
                out.price_residual = std::abs(surface_price - refs.ref_price) / strike;
            }
        }
        if (!refs.resolved) {
            out.status = PointStatus::ReferenceUnresolved;
            return out;
        }
        // A handle missing either callable cannot be round-tripped: calling an
        // empty std::function throws, and library code here does not throw.
        // The residual above needs only `price`, so it survives a missing vega.
        if (!surface.price || !surface.vega) {
            out.status = PointStatus::SurfaceNonFinite;
            return out;
        }

        // Bind every coordinate but sigma, exactly as the shipped solver does.
        const auto price_of = [&](double s) { return surface.price(spot, strike, tau, s, rate); };
        const auto vega_of = [&](double s) { return surface.vega(spot, strike, tau, s, rate); };

        const auto invert = [&](double target, double published_lo, double published_hi)
            -> std::expected<double, PointStatus>
        {
            SurfaceInversionPolicy policy;
            policy.published_sigma_min = published_lo;
            policy.published_sigma_max = published_hi;
            const auto bracket =
                effective_sigma_bracket(spot, strike, option_type, target, policy);
            const auto inverted =
                invert_price_on_surface(price_of, vega_of, target, bracket, spot, policy);
            if (!inverted) return std::unexpected(status_of(inverted.error()));
            // A non-finite root would otherwise leave the point Measured with
            // an understated error, since NaN loses every std::max against it.
            if (!std::isfinite(inverted->implied_vol)) {
                return std::unexpected(PointStatus::SurfaceNonFinite);
            }
            return inverted->implied_vol;
        };

        // `refs.resolved` implies all three targets are positive:
        // `stencil_resolved` requires y - delta > lo + delta_lo, and a price
        // is never negative, so `effective_sigma_bracket`'s `target_price > 0`
        // precondition holds for every one of them.
        const double targets[3] = {refs.ref_price - refs.delta, refs.ref_price,
                                   refs.ref_price + refs.delta};
        double worst = 0.0;
        PointStatus status = PointStatus::Measured;
        for (double target : targets) {
            const auto inverted = invert(target, sample.sigma_min, sample.sigma_max);
            if (!inverted) {
                if (severity(inverted.error()) > severity(status)) status = inverted.error();
                continue;
            }
            worst = std::max(worst, std::abs(*inverted - sigma));
        }

        out.status = status;
        if (status == PointStatus::Measured) {
            out.iv_error = worst;
            return out;
        }
        if (status == PointStatus::SurfaceNoRoot) {
            // Diagnostic only (spec D3): would a bracket widened by one
            // target_iv_error per side, clipped to the fit domain, have found
            // the root the published bracket missed?  The answer is recorded,
            // never substituted for the outcome above.
            const double lo = std::max(sample.sigma_min - tau_iv, fit.sigma_min);
            const double hi = std::min(sample.sigma_max + tau_iv, fit.sigma_max);
            bool all_inverted = true;
            for (double target : targets) {
                if (!invert(target, lo, hi)) { all_inverted = false; break; }
            }
            out.edge_band_rescue = all_inverted;
        }
        return out;
    };
}

bool stencil_resolved(const ErrorRefs& r) noexcept {
    const double v[] = {r.ref_price, r.bracket_lo_price, r.bracket_hi_price,
                        r.delta, r.delta_lo, r.delta_hi};
    for (double x : v) {
        if (!std::isfinite(x)) return false;
    }
    return (r.ref_price - r.delta > r.bracket_lo_price + r.delta_lo) &&
           (r.bracket_hi_price - r.delta_hi > r.ref_price + r.delta);
}

namespace {

// Two-grid Richardson error estimate (spec D1).  An estimate, never a
// certificate: a difference between two grids cannot see bias they share.
double richardson_estimate(double fine, double coarse) {
    return kRichardsonSafetyFactor * std::abs(fine - coarse)
         / (std::pow(2.0, kReferenceConvergenceOrder) - 1.0);
}

// Spec D2: each of the three targets must pass the product's own query
// validation -- finite, positive, at or above intrinsic, at or below the
// upper no-arbitrage limit.
bool target_is_valid_query(const PricingParams& p, double target) {
    IVQuery q;
    // PricingParams and IVQuery both derive from OptionSpec: copy spot,
    // strike, maturity, rate, dividend_yield and option_type in one move.
    static_cast<OptionSpec&>(q) = static_cast<const OptionSpec&>(p);
    q.market_price = target;
    q.discrete_dividends = p.discrete_dividends;
    return validate_iv_query(q).has_value();
}

}  // namespace

PrepareRefsFn make_stencil_refs_fn(const AdaptiveGridParams& params,
                                   ReferenceOracle oracle,
                                   std::shared_ptr<ReferenceSolveCounter> counter,
                                   StencilSolveFn solve) {
    if (!solve) {
        solve = [oracle](const PricingParams& p, const PDEGridConfig& g) {
            return oracle.solve(p, g);
        };
    }
    if (!counter) counter = std::make_shared<ReferenceSolveCounter>();
    const double tau_iv = params.target_iv_error;
    return [oracle, counter, solve, tau_iv](
        double spot, double strike, double tau, double sigma, double rate)
        -> std::expected<ErrorRefs, SolverError> {
        ErrorRefs out;                  // all-NaN, resolved = false
        out.sigma_lo = sigma - tau_iv;
        out.sigma_hi = sigma + tau_iv;

        // One family per preparation, chosen at the widest stencil member
        // (spec D1/L2): all six solves share this one fine/coarse pair.
        const PricingParams widest =
            oracle.contract(spot, strike, tau, out.sigma_hi, rate);
        auto fam = make_reference_grid_family(widest, oracle.accuracy, 1);
        if (!fam) {
            return std::unexpected(
                SolverError{.code = SolverErrorCode::InvalidConfiguration});
        }
        const PDEGridConfig& fine = fam->levels[0];
        const PDEGridConfig& coarse = fam->levels[1];
        out.fine_steps = static_cast<uint32_t>(fam->time_steps[0]);
        out.coarse_steps = static_cast<uint32_t>(fam->time_steps[1]);

        // Kept so the base solve's `unexpected` carries the solver's own
        // code rather than a bare default one.
        SolverError last_error{};
        auto run = [&](double s, const PDEGridConfig& g,
                       bool is_fine) -> std::optional<double> {
            (is_fine ? counter->fine_attempts
                     : counter->coarse_attempts).fetch_add(1);
            auto r = solve(oracle.contract(spot, strike, tau, s, rate), g);
            if (!r || !std::isfinite(*r)) {
                last_error = r ? SolverError{.code = SolverErrorCode::NonFiniteSolution}
                               : r.error();
                (is_fine ? counter->fine_failures
                         : counter->coarse_failures).fetch_add(1);
                return std::nullopt;
            }
            return *r;
        };

        auto y = run(sigma, fine, true);
        if (!y) return std::unexpected(last_error);  // invalid point
        out.ref_price = *y;
        if (!(out.sigma_lo > 0.0) || !std::isfinite(out.sigma_hi)) {
            return out;                                 // unresolved, base present
        }
        auto y2 = run(sigma, coarse, false);
        auto lo = run(out.sigma_lo, fine, true);
        auto lo2 = lo ? run(out.sigma_lo, coarse, false) : std::nullopt;
        auto hi = run(out.sigma_hi, fine, true);
        auto hi2 = hi ? run(out.sigma_hi, coarse, false) : std::nullopt;
        if (!y2 || !lo || !lo2 || !hi || !hi2) return out;
        out.bracket_lo_price = *lo;
        out.bracket_hi_price = *hi;
        out.delta = richardson_estimate(*y, *y2);
        out.delta_lo = richardson_estimate(*lo, *lo2);
        out.delta_hi = richardson_estimate(*hi, *hi2);
        if (!stencil_resolved(out)) return out;
        const PricingParams base = oracle.contract(spot, strike, tau, sigma, rate);
        for (double target : {out.ref_price - out.delta, out.ref_price,
                              out.ref_price + out.delta}) {
            if (!target_is_valid_query(base, target)) {
                return out;             // reference limitation, not a candidate's
            }
        }
        out.resolved = true;
        return out;
    };
}

ValidateFn make_validate_fn(double dividend_yield,
                            OptionType option_type,
                            const std::vector<Dividend>& discrete_dividends,
                            std::optional<double> reference_maturity) {
    // Segmented surfaces follow one fixed expiry across remaining life
    // (oracle.contract rolls dividends onto it); ordinary callers without an
    // anchor describe a contract from now (contract() filters to tau
    // directly). Single-price callers have no reference grid family in
    // hand, so this solves at the oracle's accuracy profile on an
    // auto-estimated grid rather than a fixed explicit one.
    ReferenceOracle oracle{dividend_yield, option_type, discrete_dividends,
                          reference_maturity, make_grid_accuracy(kReferenceAccuracy)};
    return [oracle](double spot, double strike, double tau, double sigma,
                    double rate) -> std::expected<double, SolverError> {
        return oracle.solve_estimated(oracle.contract(spot, strike, tau, sigma, rate));
    };
}

}  // namespace mango
