// SPDX-License-Identifier: MIT
#include "mango/option/table/reference_strike_evaluator.hpp"
#include "mango/option/american_option.hpp"
#include "mango/option/dividend_utils.hpp"
#include "mango/option/table/splits/multi_kref.hpp"
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <limits>
#include <map>
#include <numeric>
#include <tuple>

namespace mango {
namespace {
struct Observation {
    double value = std::numeric_limits<double>::quiet_NaN();
    std::optional<double> allowance;
};

std::optional<double> convergence_allowance(
    const std::array<double, 3>& values, double floor) {
    if (!std::ranges::all_of(values, [](double v) { return std::isfinite(v); }))
        return std::nullopt;
    const double first = values[1] - values[0];
    const double last = values[2] - values[1];
    if (std::max(std::abs(first), std::abs(last)) <= floor) return floor;
    if (first * last <= 0.0 || std::abs(last) >= 0.75 * std::abs(first))
        return std::nullopt;
    const double ratio = std::abs(last / first);
    return std::max(floor, 2.0 * std::abs(last) * ratio / (1.0 - ratio));
}

struct PhysicalQuery {
    double spot, strike, tau, sigma, rate;
};

struct ErrorAccumulator {
    ReferenceErrorSummary summary;
    double l2_norm = 0.0;
    bool fails = false;
    bool ambiguous = false;
    bool headroom = true;

    void observe(double error, double uncertainty, double target) {
        ++summary.requested;
        ++summary.measured;
        summary.max_error = std::max(summary.max_error.value_or(0.0), error);
        summary.max_uncertainty = std::max(summary.max_uncertainty.value_or(0.0), uncertainty);
        l2_norm = std::hypot(l2_norm, error);
        summary.rms_error = l2_norm / std::sqrt(static_cast<double>(summary.measured));
        fails |= error - uncertainty > target;
        ambiguous |= error + uncertainty > target && error - uncertainty <= target;
        headroom &= error + uncertainty <= 0.5 * target;
    }
    void unresolved(bool refused = false) {
        ++summary.requested;
        if (refused) ++summary.refused;
        else ++summary.unresolved;
    }
    void filtered() { ++summary.requested; ++summary.filtered; }
    void exact() { ++summary.requested; ++summary.structurally_exact; }
    void untested(size_t count = 1) { summary.requested += count; summary.untested += count; }
    [[nodiscard]] bool qualified() const {
        return summary.unresolved == 0 && summary.refused == 0 && summary.untested == 0;
    }
    [[nodiscard]] bool passes() const { return qualified() && !fails && !ambiguous; }
};

struct AccuracyAccumulator {
    ErrorAccumulator price, iv;
    [[nodiscard]] ReferenceAccuracySummary result() const { return {price.summary, iv.summary}; }
    [[nodiscard]] bool qualified() const { return price.qualified() && iv.qualified(); }
    [[nodiscard]] bool passes() const { return price.passes() && iv.passes(); }
};
}  // namespace

struct ReferenceStrikeEvaluator::Impl {
    SegmentedAdaptiveConfig config;
    SurfaceBounds bounds;
    std::vector<PhysicalQuery> queries;
    double price_target, iv_target;
    double full_u;
    size_t solves = 0;
    static constexpr size_t kSolveBudget = 8192;
    using SolveKey = std::tuple<double, double, double, double, size_t, size_t, double>;
    std::map<SolveKey, std::shared_ptr<AmericanOptionResult>> solutions;
    using PriceKey = std::tuple<double, double, double, double, double, size_t>;
    std::map<PriceKey, Observation> prices;
    std::map<std::vector<double>, ReferenceAccuracySummary> reference_witnesses;

    std::shared_ptr<AmericanOptionResult> solve(
        double strike, double tau, double sigma, double rate,
        size_t nx, size_t nt, double u) {
        const SolveKey key{strike, tau, sigma, rate, nx, nt, u};
        if (const auto found = solutions.find(key); found != solutions.end()) return found->second;
        if (solves >= kSolveBudget) return {};
        ++solves;
        PricingParams params(OptionSpec{.spot = strike, .strike = strike, .maturity = tau,
            .rate = rate, .dividend_yield = config.dividend_yield,
            .option_type = config.option_type}, sigma);
        params.discrete_dividends = rolled_dividends(
            config.discrete_dividends, config.maturity, tau);
        const double radius = 0.05 * std::sinh(u);
        auto grid = GridSpec<double>::sinh_spaced(-radius, radius, nx, 2.0 * u);
        std::shared_ptr<AmericanOptionResult> result;
        if (grid) {
            auto solver = AmericanOptionSolver::create(params, PDEGridSpec{PDEGridConfig{*grid, nt, {}}});
            if (solver) {
                auto solved = solver->solve();
                if (solved) result = std::make_shared<AmericanOptionResult>(std::move(*solved));
            }
        }
        solutions.emplace(key, result);
        return result;
    }

    double sample(const PhysicalQuery& query, double strike, double sigma, size_t round,
                  size_t intervals, size_t time_steps, double u) {
        const double ratio = query.spot / query.strike;
        const bool homogeneous = rolled_dividends(
            config.discrete_dividends, config.maturity, query.tau).empty();
        const double local_k = homogeneous ? 1.0 : strike;
        const double scale = homogeneous ? strike : 1.0;
        const size_t factor = size_t{1} << round;
        auto result = solve(local_k, query.tau, sigma, query.rate,
                            intervals * factor + 1, time_steps * factor, u);
        const double x = std::log(ratio);
        if (!result || !(x > result->grid()->x().front() && x < result->grid()->x().back()))
            return std::numeric_limits<double>::quiet_NaN();
        return scale * result->value_at(local_k * ratio);
    }

    Observation price(const PhysicalQuery& query, double strike, double sigma, size_t round) {
        const double ratio = query.spot / query.strike;
        const bool homogeneous = rolled_dividends(
            config.discrete_dividends, config.maturity, query.tau).empty();
        const double scale = homogeneous ? strike : 1.0;
        const double local_k = homogeneous ? 1.0 : strike;
        const PriceKey key{ratio, local_k, query.tau, sigma, query.rate, round};
        if (const auto found = prices.find(key); found != prices.end()) {
            auto result = found->second;
            result.value *= scale;
            if (result.allowance) *result.allowance *= scale;
            return result;
        }
        auto at = [&](size_t intervals, size_t time_steps, double u) {
            return sample(query, local_k, sigma, round, intervals, time_steps, u);
        };
        const double finest = at(2048, 2048, full_u);
        const std::array space{at(512, 2048, full_u), at(1024, 2048, full_u), finest};
        const std::array time{at(2048, 512, full_u), at(2048, 1024, full_u), finest};
        // All domain levels retain the same interior u spacing and strike node.
        const std::array domain{at(1536, 2048, 0.75 * full_u),
                                at(1792, 2048, 0.875 * full_u), finest};
        const double floor = 256.0 * std::numeric_limits<double>::epsilon()
            * std::max({local_k, local_k * ratio, 1.0});
        Observation result{.value = finest};
        auto a = convergence_allowance(space, floor);
        auto b = convergence_allowance(time, floor);
        auto c = convergence_allowance(domain, floor);
        if (a && b && c) {
            const double allowance = *a + *b + *c;
            const double intrinsic = intrinsic_value(local_k * ratio, local_k, config.option_type);
            if (finest + allowance >= intrinsic) result.allowance = allowance;
        }
        prices.emplace(key, result);
        result.value *= scale;
        if (result.allowance) *result.allowance *= scale;
        return result;
    }

    Observation residual(const PhysicalQuery& query, const std::vector<double>& refs, size_t round) {
        const auto bracket = MultiKRefSplit(refs).bracket(
            query.spot, query.strike, query.tau, query.sigma, query.rate);
        auto difference = [&](size_t intervals, size_t steps, double u) {
            double blend = 0.0;
            for (size_t i = 0; i < bracket.count; ++i) {
                const auto entry = bracket.entries[i];
                const double k = refs[entry.index];
                blend += query.strike * entry.weight / k
                    * sample(query, k, query.sigma, round, intervals, steps, u);
            }
            return blend - sample(query, query.strike, query.sigma, round, intervals, steps, u);
        };
        const double finest = difference(2048, 2048, full_u);
        const double floor = 1024.0 * std::numeric_limits<double>::epsilon()
            * std::max({query.spot, query.strike, 1.0});
        const std::array space{difference(512, 2048, full_u),
                               difference(1024, 2048, full_u), finest};
        const std::array time{difference(2048, 512, full_u),
                              difference(2048, 1024, full_u), finest};
        const std::array domain{difference(1536, 2048, 0.75 * full_u),
                                difference(1792, 2048, 0.875 * full_u), finest};
        auto a = convergence_allowance(space, floor);
        auto b = convergence_allowance(time, floor);
        auto c = convergence_allowance(domain, floor);
        Observation result{.value = finest};
        if (a && b && c) result.allowance = *a + *b + *c;
        return result;
    }

    Observation vega(const PhysicalQuery& query, size_t round) {
        // Qualify the derivative itself across independent mesh directions.
        // Summing separately estimated price errors discards correlation and
        // can leave a converged derivative spuriously unqualified.
        std::array<double, 3> values{}, uncertainties{};
        for (size_t i = 0; i < 3; ++i) {
            const double h = std::ldexp(0.02 * query.sigma, -static_cast<int>(i));
            auto derivative = [&](size_t intervals, size_t steps, double u) {
                const double up = sample(query, query.strike, query.sigma + h,
                                         round, intervals, steps, u);
                const double down = sample(query, query.strike, query.sigma - h,
                                           round, intervals, steps, u);
                return (up - down) / (2.0 * h);
            };
            const double finest = derivative(2048, 2048, full_u);
            values[i] = finest;
            const double floor = 512.0 * std::numeric_limits<double>::epsilon()
                * std::max({query.spot, query.strike, 1.0}) / h;
            const std::array space{derivative(512, 2048, full_u),
                                   derivative(1024, 2048, full_u), finest};
            const std::array time{derivative(2048, 512, full_u),
                                  derivative(2048, 1024, full_u), finest};
            const std::array domain{derivative(1536, 2048, 0.75 * full_u),
                                    derivative(1792, 2048, 0.875 * full_u), finest};
            auto a = convergence_allowance(space, floor);
            auto b = convergence_allowance(time, floor);
            auto c = convergence_allowance(domain, floor);
            if (!a || !b || !c) return Observation{.value = finest};
            uncertainties[i] = *a + *b + *c;
        }
        Observation result{.value = values.back()};
        const double mesh = *std::ranges::max_element(uncertainties);
        auto bump = convergence_allowance(values, mesh);
        if (bump) result.allowance = mesh + *bump;
        return result;
    }

};

ReferenceStrikeEvaluator::ReferenceStrikeEvaluator(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}
ReferenceStrikeEvaluator::ReferenceStrikeEvaluator(ReferenceStrikeEvaluator&&) noexcept = default;
ReferenceStrikeEvaluator& ReferenceStrikeEvaluator::operator=(ReferenceStrikeEvaluator&&) noexcept = default;
ReferenceStrikeEvaluator::~ReferenceStrikeEvaluator() = default;

std::expected<ReferenceStrikeEvaluator, PriceTableError>
ReferenceStrikeEvaluator::create(const SegmentedAdaptiveConfig& config, const SurfaceBounds& requested,
    std::vector<std::pair<double, double>> admitted_times, double price_target, double iv_target) {
    const auto invalid = [] { return std::unexpected(PriceTableError{PriceTableErrorCode::InvalidConfig}); };
    const std::array endpoints{requested.m_min, requested.m_max, requested.tau_min, requested.tau_max,
        requested.sigma_min, requested.sigma_max, requested.rate_min, requested.rate_max};
    if (!std::ranges::all_of(endpoints, [](double v) { return std::isfinite(v); }) ||
        requested.m_min > requested.m_max || requested.tau_min < 0.0 ||
        !(requested.tau_max > requested.tau_min) || requested.sigma_min <= 0.0 ||
        requested.sigma_min > requested.sigma_max || requested.rate_min > requested.rate_max ||
        !std::isfinite(config.dividend_yield) ||
        !make_fixed_expiry_metadata(config.maturity, config.discrete_dividends).valid(requested.tau_max) ||
        !requested.strike_bounds || !requested.strike_bounds->valid() || admitted_times.empty() ||
        !std::isfinite(price_target) || price_target <= 0.0 || !std::isfinite(iv_target) || iv_target <= 0.0)
        return invalid();
    const MoneynessBounds ratios = requested.ratio_bounds.value_or(
        MoneynessBounds{std::exp(requested.m_min), std::exp(requested.m_max)});
    const MoneynessDomain domain(ratios);
    if (!domain.valid()) return invalid();
    auto impl = std::make_unique<Impl>();
    impl->config = config;
    impl->bounds = requested;
    impl->price_target = price_target;
    impl->iv_target = iv_target;
    const double reach = std::max(std::abs(requested.m_min), std::abs(requested.m_max))
        + 5.0 * 1.04 * requested.sigma_max * std::sqrt(config.maturity)
        + (std::max(std::abs(requested.rate_min), std::abs(requested.rate_max))
           + std::abs(config.dividend_yield)) * config.maturity;
    impl->full_u = std::max(4.0, (4.0 / 3.0) * std::asinh(reach / 0.05));
    if (!std::isfinite(impl->full_u) || !std::isfinite(0.05 * std::sinh(impl->full_u))) return invalid();
    std::vector<double> times;
    double previous = requested.tau_min;
    for (size_t i = 0; i < admitted_times.size(); ++i) {
        const auto [lo, hi] = admitted_times[i];
        if (!std::isfinite(lo) || !std::isfinite(hi) || lo < previous || !(hi > lo) || hi > requested.tau_max)
            return invalid();
        const double inset = std::min(1.0 / 365.0, 0.25 * (hi - lo));
        double early = lo + inset, late = hi - inset;
        if (i > 0) early = std::clamp(
            std::midpoint(admitted_times[i - 1].second, lo) + 1.0 / 365.0, lo, hi - inset);
        if (i + 1 < admitted_times.size()) late = std::clamp(
            std::midpoint(hi, admitted_times[i + 1].first) - 1.0 / 365.0, lo + inset, hi);
        times.push_back(early);
        times.push_back(late);
        previous = hi;
    }
    if (admitted_times.back().second == requested.tau_max) times.push_back(requested.tau_max);
    const auto strikes = *requested.strike_bounds;
    auto add = [&](double ratio, double k, double tau, double sigma, double rate) {
        const PhysicalQuery point{ratio * k, k, tau, sigma, rate};
        if (std::ranges::any_of(impl->queries, [&](const auto& old) {
            return old.spot == point.spot && old.strike == point.strike && old.tau == point.tau &&
                old.sigma == point.sigma && old.rate == point.rate;
        })) return;
        impl->queries.push_back(point);
    };
    std::vector<double> off_reference_strikes;
    for (double fraction : {0.355, 0.685, 0.065, 0.945})
        off_reference_strikes.push_back(std::lerp(strikes.min, strikes.max, fraction));
    if (!config.kref_config.K_refs.empty()) {
        auto refs = resolve_k_refs(config.kref_config, strikes);
        if (!refs) return std::unexpected(refs.error());
        // A fixed caller list cannot make every validation query coincide
        // with a reference: declare one interior point per served gap now,
        // before observing any candidate prices.
        for (size_t i = 1; i < refs->size(); ++i) {
            const double lo = std::max((*refs)[i - 1], strikes.min);
            const double hi = std::min((*refs)[i], strikes.max);
            const double midpoint = std::midpoint(lo, hi);
            if (lo < midpoint && midpoint < hi &&
                std::ranges::find(off_reference_strikes, midpoint) == off_reference_strikes.end())
                off_reference_strikes.push_back(midpoint);
        }
    }
    // Retain the original endpoint/ATM population, then supplement it with
    // off-reference tails and both sigma/rate strata. Exact-reference corners
    // alone cannot measure how cash dividends break strike homogeneity.
    for (double tau : times) {
        for (double fraction : {0.355, 0.685, 0.065, 0.945})
            add(std::clamp(1.0, ratios.min, ratios.max), std::lerp(strikes.min, strikes.max, fraction),
                tau, requested.sigma_min, requested.rate_max);
        size_t corner = 0;
        for (double k : {strikes.min, strikes.max}) for (double ratio : {ratios.min, ratios.max}) {
            add(ratio, k, tau, (corner % 2) ? requested.sigma_max : requested.sigma_min,
                (corner / 2) ? requested.rate_max : requested.rate_min);
            ++corner;
        }
        for (double ratio : {ratios.min, ratios.max})
            add(ratio, std::midpoint(strikes.min, strikes.max), tau, requested.sigma_max,
                std::midpoint(requested.rate_min, requested.rate_max));
    }
    for (double tau : times) for (double k : off_reference_strikes)
        for (double ratio : {ratios.min, std::clamp(1.0, ratios.min, ratios.max), ratios.max})
            for (double sigma : {requested.sigma_min, requested.sigma_max})
                for (double rate : {requested.rate_min, requested.rate_max})
                    add(ratio, k, tau, sigma, rate);
    for (size_t i = 0; i < impl->queries.size(); ++i) {
        const auto& q = impl->queries[i];
        if (!domain.contains_quote(q.spot, q.strike) || !(q.tau > 0.0) || !(q.sigma > 0.0))
            return std::unexpected(PriceTableError{PriceTableErrorCode::InvalidConfig, 0, i});
    }
    return ReferenceStrikeEvaluator(std::move(impl));
}


std::expected<ReferenceCandidateMetrics, PriceTableError>
ReferenceStrikeEvaluator::evaluate(std::span<const double> refs, const SurfaceHandle* fitted) {
    auto validated = validate_k_ref_values(refs, impl_->config.kref_config.max_references);
    if (!validated) return std::unexpected(validated.error());
    if (validated->front() > impl_->bounds.strike_bounds->min ||
        validated->back() < impl_->bounds.strike_bounds->max)
        return std::unexpected(PriceTableError{PriceTableErrorCode::InvalidConfig});
    const auto started = std::chrono::steady_clock::now();
    const size_t starting_solves = impl_->solves;
    ReferenceCandidateMetrics metrics;
    const MultiKRefSplit split(*validated);
    const auto prior_witness = impl_->reference_witnesses.find(*validated);
    const bool has_prior_witness = prior_witness != impl_->reference_witnesses.end();
    bool rejected_composed_rescue = false;
    AccuracyAccumulator ideal, fit, total;
    const double price_allowance = std::min(0.001, 0.1 * impl_->price_target);
    const double iv_allowance = std::min(2e-6, 0.1 * impl_->iv_target);
    for (size_t point = 0; point < impl_->queries.size(); ++point) {
        const auto& query = impl_->queries[point];
        const auto bracket = split.bracket(query.spot, query.strike, query.tau, query.sigma, query.rate);
        const bool invariant = rolled_dividends(impl_->config.discrete_dividends,
            impl_->config.maturity, query.tau).empty() ||
            (bracket.count == 1 && (*validated)[bracket.entries[0].index] == query.strike);
        if (invariant) {
            ideal.price.exact();
            ideal.iv.exact();
            if (!fitted) continue;
        }
        const double fitted_price = fitted ? fitted->price(query.spot, query.strike, query.tau,
            query.sigma, query.rate) : 0.0;
        Observation direct, residual, sensitivity;
        bool price_qualified = false, direct_qualified = false;
        bool price_refused = false, iv_qualified = false, iv_filtered = false;
        bool witnessed_price_failure = false;
        bool witnessed_composed_failure = false;
        for (size_t round = 0; round < 2; ++round) {
            iv_qualified = false;
            iv_filtered = false;
            residual = invariant ? Observation{0.0, 0.0} : impl_->residual(query, *validated, round);
            price_refused = !std::isfinite(residual.value);
            price_qualified = !price_refused && residual.allowance && *residual.allowance <= price_allowance;
            if (!price_qualified && !fitted) continue;
            const double error = std::abs(residual.value);
            const bool price_ambiguous = price_qualified && error + *residual.allowance > impl_->price_target &&
                error - *residual.allowance <= impl_->price_target;
            if (!fitted && price_qualified && error - *residual.allowance > impl_->price_target) {
                witnessed_price_failure = true;
                break;
            }
            direct = impl_->price(query, query.strike, query.sigma, round);
            direct_qualified = direct.allowance && *direct.allowance <= price_allowance;
            if (!direct_qualified) continue;
            if (fitted && has_prior_witness && std::isfinite(fitted_price) &&
                std::abs(fitted_price - direct.value) - *direct.allowance > impl_->price_target) {
                witnessed_composed_failure = true;
                break;
            }
            const double intrinsic = intrinsic_value(query.spot, query.strike, impl_->config.option_type);
            const double time_value = direct.value - intrinsic;
            if ((time_value + *direct.allowance) / query.strike < 1e-4) {
                iv_filtered = true;
                if (price_ambiguous && round == 0) continue;
                break;
            }
            if ((time_value - *direct.allowance) / query.strike < 1e-4) continue;
            sensitivity = impl_->vega(query, round);
            if (!sensitivity.allowance || !std::isfinite(sensitivity.value)) continue;
            const double low = sensitivity.value - *sensitivity.allowance;
            const double high = sensitivity.value + *sensitivity.allowance;
            if (std::max(std::abs(low), std::abs(high)) < 1e-4) {
                iv_filtered = true;
                if (price_ambiguous && round == 0) continue;
                break;
            }
            if (low > 1e-4) {
                iv_qualified = true;
                // Actual composed evidence has its own direct-oracle budget;
                // an unresolved ideal residual must not block a valid rescue.
                const double relevant_uncertainty = fitted ? *direct.allowance
                    : residual.allowance.value_or(std::numeric_limits<double>::infinity());
                if (relevant_uncertainty / low > iv_allowance) continue;
                const bool iv_ambiguous = price_qualified &&
                    (error + *residual.allowance) / low > impl_->iv_target &&
                    std::max(0.0, error - *residual.allowance) / high <= impl_->iv_target;
                if (!fitted && round == 0 && (price_ambiguous || iv_ambiguous)) continue;
                break;
            }
        }
        if (witnessed_price_failure) {
            ideal.price.observe(std::abs(residual.value), *residual.allowance, impl_->price_target);
            ideal.iv.untested();
            const size_t remaining = impl_->queries.size() - point - 1;
            ideal.price.untested(remaining);
            ideal.iv.untested(remaining);
            break;
        }
        if (witnessed_composed_failure) {
            if (price_qualified) {
                const double blend = direct.value + residual.value;
                fit.price.observe(std::abs(fitted_price - blend),
                    *direct.allowance + *residual.allowance, impl_->price_target);
            } else fit.price.unresolved();
            total.price.observe(std::abs(fitted_price - direct.value),
                                *direct.allowance, impl_->price_target);
            fit.iv.untested(); total.iv.untested();
            const size_t remaining = impl_->queries.size() - point - 1;
            fit.price.untested(remaining); fit.iv.untested(remaining);
            total.price.untested(remaining); total.iv.untested(remaining);
            rejected_composed_rescue = true;
            break;
        }
        const double ideal_error = std::abs(residual.value);
        const double ideal_uncertainty = residual.allowance.value_or(0.0);
        if (!invariant) {
            if (price_qualified) ideal.price.observe(ideal_error, ideal_uncertainty, impl_->price_target);
            else ideal.price.unresolved(price_refused);
        }
        const double blend = direct.value + residual.value;
        const double blend_uncertainty = direct_qualified ? *direct.allowance + ideal_uncertainty : 0.0;
        const double fit_error = std::abs(fitted_price - blend);
        const double total_error = std::abs(fitted_price - direct.value);
        const bool finite_fit = std::isfinite(fitted_price);
        if (fitted) {
            if (finite_fit && direct_qualified && price_qualified)
                fit.price.observe(fit_error, blend_uncertainty, impl_->price_target);
            else fit.price.unresolved(!finite_fit);
            if (finite_fit && direct_qualified)
                total.price.observe(total_error, *direct.allowance, impl_->price_target);
            else total.price.unresolved(!finite_fit);
        }
        auto record_iv = [&](ErrorAccumulator& accumulator, double error, double uncertainty,
                             bool valid, bool refused = false) {
            if (refused) { accumulator.unresolved(true); return; }
            if (!valid) { accumulator.unresolved(); return; }
            if (iv_filtered) { accumulator.filtered(); return; }
            if (!iv_qualified) { accumulator.unresolved(); return; }
            const double low = sensitivity.value - *sensitivity.allowance;
            const double high = sensitivity.value + *sensitivity.allowance;
            if (uncertainty / low > iv_allowance) { accumulator.unresolved(); return; }
            const double estimate = error / sensitivity.value;
            const double upper = (error + uncertainty) / low;
            const double lower = std::max(0.0, error - uncertainty) / high;
            accumulator.observe(estimate, std::max(upper - estimate, estimate - lower), impl_->iv_target);
        };
        if (!invariant) record_iv(ideal.iv, ideal_error, ideal_uncertainty,
                                  price_qualified && direct_qualified, price_refused);
        if (fitted) {
            record_iv(fit.iv, fit_error, blend_uncertainty, price_qualified && direct_qualified, !finite_fit);
            record_iv(total.iv, total_error, direct.allowance.value_or(0.0), direct_qualified, !finite_fit);
        }
        if (!fitted && ideal.iv.fails) {
            const size_t remaining = impl_->queries.size() - point - 1;
            ideal.price.untested(remaining);
            ideal.iv.untested(remaining);
            break;
        }
    }
    metrics.ideal_blend = rejected_composed_rescue ? prior_witness->second : ideal.result();
    if (fitted) {
        metrics.fit = fit.result();
        metrics.total = total.result();
        if (total.qualified() && total.iv.summary.measured > 0) metrics.total_target_met = total.passes();
        else if (total.price.fails || total.iv.fails) metrics.total_target_met = false;
    }
    if (rejected_composed_rescue) {
        metrics.decision = ReferenceCandidateDecision::RefineReferences;
    } else if (fitted && total.passes()) {
        metrics.decision = total.iv.summary.measured ? ReferenceCandidateDecision::Adequate
                                                   : ReferenceCandidateDecision::IvUnmeasured;
    } else if (ideal.price.fails || ideal.iv.fails) {
        metrics.decision = ReferenceCandidateDecision::RefineReferences;
    } else if (!ideal.qualified()) {
        metrics.decision = ReferenceCandidateDecision::ReferenceUnqualified;
    } else if (ideal.price.ambiguous || ideal.iv.ambiguous) {
        metrics.decision = ReferenceCandidateDecision::ThresholdAmbiguous;
    } else if (metrics.total_target_met == false) {
        metrics.decision = ReferenceCandidateDecision::FitLimited;
    } else if (ideal.price.summary.structurally_exact == impl_->queries.size()) {
        metrics.decision = ReferenceCandidateDecision::Adequate;
    } else if (ideal.iv.summary.measured == 0) {
        metrics.decision = ReferenceCandidateDecision::IvUnmeasured;
    } else {
        metrics.decision = ReferenceCandidateDecision::Adequate;
        metrics.prefer_refinement = !ideal.price.headroom || !ideal.iv.headroom;
    }
    if (!fitted && metrics.decision == ReferenceCandidateDecision::RefineReferences)
        impl_->reference_witnesses[*validated] = metrics.ideal_blend;
    metrics.pde_solves = impl_->solves - starting_solves;
    metrics.elapsed_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
    return metrics;
}

}  // namespace mango
