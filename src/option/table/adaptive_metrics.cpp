// SPDX-License-Identifier: MIT
#include "mango/option/table/adaptive_metrics.hpp"
#include "mango/option/american_option.hpp"
#include "mango/option/dividend_utils.hpp"
#include <algorithm>
#include <cmath>
#include <optional>

namespace mango {

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
        double sigma, double rate) -> ProviderResult<ErrorRefs, SolverError>
    {
        std::optional<PdeWork> work = PdeWork{};
        const auto add_work = [&work](const std::optional<PdeWork>& next) {
            if (work && next) *work += *next;
            else work.reset();
        };
        auto fd_base = validate_fn(spot, strike, tau, sigma, rate);
        add_work(fd_base.work);
        if (!fd_base.has_value()) {
            return {std::unexpected(fd_base.error()), work};
        }
        double ref_price = fd_base.value();
        if (!std::isfinite(ref_price)) {
            return {std::unexpected(SolverError{}), work};
        }

        // FD American vega via central difference
        double eps = std::max(1e-4, 0.01 * sigma);
        double sigma_dn = std::max(1e-4, sigma - eps);
        double sigma_up = sigma + eps;
        double effective_eps = (sigma_up - sigma_dn) / 2.0;

        auto fd_up = validate_fn(spot, strike, tau, sigma_up, rate);
        add_work(fd_up.work);
        if (!fd_up.has_value()) {
            return {std::unexpected(fd_up.error()), work};
        }
        auto fd_dn = validate_fn(spot, strike, tau, sigma_dn, rate);
        add_work(fd_dn.work);
        if (!fd_dn.has_value()) {
            return {std::unexpected(fd_dn.error()), work};
        }

        double vega = 0.0;
        if (effective_eps > 1e-6) {
            vega = (fd_up.value() - fd_dn.value()) / (2.0 * effective_eps);
        }
        if (!std::isfinite(vega)) {
            return {std::unexpected(SolverError{}), work};
        }

        return {ErrorRefs{.ref_price = ref_price, .vega = vega}, work};
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
        double sigma, double rate) -> ProviderResult<double, SolverError>
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
        auto solver = AmericanOptionSolver::create(p);
        if (!solver) return {std::unexpected(SolverError{
            .code = SolverErrorCode::InvalidConfiguration, .iterations = 0}), PdeWork{}};
        auto fd = solver->solve();
        const PdeWork work{.attempted = 1, .completed = fd.has_value() ? 1u : 0u,
                           .failed = fd.has_value() ? 0u : 1u};
        if (!fd) return {std::unexpected(fd.error()), work};
        return {fd->value(), work};
    };
}

}  // namespace mango
