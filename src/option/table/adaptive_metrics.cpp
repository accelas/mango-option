// SPDX-License-Identifier: MIT
#include "mango/option/table/adaptive_metrics.hpp"
#include "mango/option/american_option.hpp"
#include "mango/option/dividend_utils.hpp"
#include <algorithm>
#include <cmath>
#include <optional>

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
        if (n < floor || n < 17) {
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
    auto solver = AmericanOptionSolver::create(p, PDEGridSpec{grid});
    if (!solver) return std::unexpected(SolverError{.code = SolverErrorCode::InvalidConfiguration});
    auto r = solver->solve();
    if (!r) return std::unexpected(r.error());
    const double v = r->value();
    if (!std::isfinite(v)) return std::unexpected(SolverError{});
    return v;
}

double compute_iv_error(double price_error, double vega,
                        double vega_floor, double target_iv_error) {
    double vega_clamped = std::max(std::abs(vega), vega_floor);
    double iv_error = price_error / vega_clamped;
    double price_tol = target_iv_error * vega_floor;
    if (price_error <= price_tol) {
        iv_error = std::min(iv_error, target_iv_error);
    }
    return iv_error;
}

PrepareRefsFn make_fd_vega_refs_fn(const AdaptiveGridParams& /*params*/,
                                    const ValidateFn& validate_fn) {
    // Copy validate_fn by value so the returned lambda is self-contained.
    return [validate_fn](
        double spot, double strike, double tau,
        double sigma, double rate) -> std::expected<ErrorRefs, SolverError>
    {
        auto fd_base = validate_fn(spot, strike, tau, sigma, rate);
        if (!fd_base.has_value()) {
            return std::unexpected(fd_base.error());
        }
        double ref_price = fd_base.value();
        if (!std::isfinite(ref_price)) {
            return std::unexpected(SolverError{});
        }

        // FD American vega via central difference
        double eps = std::max(1e-4, 0.01 * sigma);
        double sigma_dn = std::max(1e-4, sigma - eps);
        double sigma_up = sigma + eps;
        double effective_eps = (sigma_up - sigma_dn) / 2.0;

        auto fd_up = validate_fn(spot, strike, tau, sigma_up, rate);
        if (!fd_up.has_value()) {
            return std::unexpected(fd_up.error());
        }
        auto fd_dn = validate_fn(spot, strike, tau, sigma_dn, rate);
        if (!fd_dn.has_value()) {
            return std::unexpected(fd_dn.error());
        }

        double vega = 0.0;
        if (effective_eps > 1e-6) {
            vega = (fd_up.value() - fd_dn.value()) / (2.0 * effective_eps);
        }
        if (!std::isfinite(vega)) {
            return std::unexpected(SolverError{});
        }

        return ErrorRefs{.ref_price = ref_price, .vega = vega};
    };
}

ScoreErrorFn make_iv_score_fn(const AdaptiveGridParams& params,
                              OptionType option_type) {
    double vega_floor = params.vega_floor;
    double target = params.target_iv_error;
    return [vega_floor, target, option_type](
        double interp, const ErrorRefs& refs,
        double spot, double strike, double /*tau*/,
        double /*sigma*/, double /*rate*/) -> std::optional<double>
    {
        // TV/K filter: skip points where IV is undefined.  `nullopt`, not
        // 0.0: a skipped point is no measurement at all, and reporting it as
        // a perfect one let a surface nobody could measure look flawless.
        constexpr double kTVKThreshold = 1e-4;
        double intrinsic = intrinsic_value(spot, strike, option_type);
        if ((refs.ref_price - intrinsic) / strike < kTVKThreshold) {
            return std::nullopt;
        }

        // Vega floor: below it the price carries no volatility information,
        // so `price_error / vega_floor` is a price error in units of the
        // floor -- not an IV error.  Left unfiltered it reads as thousands
        // of IV points from a sub-cent price wobble (measured: a deep-ITM
        // put with vega = -3.5e-5 scoring 9,700 on a surface whose worst
        // *measurable* point scored 0.15), which the D5 viability gate then
        // condemns.  This is the documented meaning of `vega_floor` --
        // "when vega < floor, fall back to price-based tolerance" -- and
        // there is no IV tolerance to fall back to, so the point is skipped
        // like any other IV-undefined one.  Price accuracy where vega ~ 0
        // is not what the IV-error metric (or kViabilityBound) measures.
        if (std::abs(refs.vega) < vega_floor) {
            return std::nullopt;
        }

        double price_error = std::abs(interp - refs.ref_price);
        return compute_iv_error(price_error, refs.vega, vega_floor, target);
    };
}

ValidateFn make_validate_fn(double dividend_yield,
                            OptionType option_type,
                            const std::vector<Dividend>& discrete_dividends,
                            std::optional<double> reference_maturity) {
    return [dividend_yield, option_type, discrete_dividends, reference_maturity](
        double spot, double strike, double tau,
        double sigma, double rate) -> std::expected<double, SolverError>
    {
        PricingParams p;
        p.spot = spot;
        p.strike = strike;
        p.maturity = tau;
        p.rate = rate;
        p.dividend_yield = dividend_yield;
        p.option_type = option_type;
        p.volatility = sigma;
        // Segmented surfaces follow one fixed expiry across remaining life.
        // Ordinary callers without an anchor describe a contract from now.
        p.discrete_dividends = reference_maturity
            ? rolled_dividends(discrete_dividends, *reference_maturity, tau)
            : filter_and_merge_dividends(discrete_dividends, tau);
        auto solver = AmericanOptionSolver::create(
            p, PDEGridSpec{make_grid_accuracy(kReferenceAccuracy)});
        if (!solver) {
            return std::unexpected(SolverError{.code = SolverErrorCode::InvalidConfiguration});
        }
        auto fd = solver->solve();
        if (!fd.has_value()) return std::unexpected(fd.error());
        return fd->value();
    };
}

}  // namespace mango
