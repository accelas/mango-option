// SPDX-License-Identifier: MIT
#pragma once

#include "mango/option/table/accuracy/collector.hpp"
#include "mango/option/interpolated_iv_solver.hpp"
#include <algorithm>
#include <concepts>

namespace mango::detail::accuracy {

/// Closed financial payloads only. A callback with a Certified-looking getter
/// is not a candidate, and report metadata cannot authorize construction.
template <typename T>
concept FinalAccuracyTable =
    std::same_as<T,BSplinePriceTable> || std::same_as<T,BSplineMultiKRefSurface> ||
    std::same_as<T,ChebyshevSurface> || std::same_as<T,ChebyshevMultiKRefSurface> ||
    std::same_as<T,BSpline3DPriceTable> || std::same_as<T,Chebyshev3DPriceTable>;

/// Existing exploratory hard limit, not an actual-IV accuracy target.
inline constexpr double kFinalPriceVegaViabilityLimit = .20;

[[nodiscard]] Viability final_candidate_viability(
    std::span<const PhysicalReferenceRow> references, const CollectionResult& collected,
    const AccuracyRequest& request);

namespace final_candidate_detail {
template <FinalAccuracyTable Table>
bool admits_contract(const Table& table, const OptionSpec& query,
                     std::span<const Dividend> dividends) {
    if (!validate_option_spec(query) || query.option_type!=table.option_type() ||
        query.dividend_yield!=table.dividend_yield() ||
        !table.contains_moneyness(query.spot,query.strike) ||
        !table.contains_strike(query.strike) || !table.contains_maturity(query.maturity)) return false;
    const double rate=get_zero_rate(query.rate,query.maturity);
    if (!std::isfinite(rate) || rate<table.rate_min() || rate>table.rate_max()) return false;
    // Unlike a user query's omitted schedule, an independent reference must
    // contain its complete actual contract. Do not silently price another one.
    const auto expected=table.fixed_expiry()
        ? rolled_dividends(table.fixed_expiry()->discrete_dividends,
            table.fixed_expiry()->reference_maturity,query.maturity)
        : std::vector<Dividend>{};
    if (expected.size()!=dividends.size()) return false;
    for (size_t i=0; i<expected.size(); ++i) {
        if (expected[i].calendar_time!=dividends[i].calendar_time ||
            expected[i].amount!=dividends[i].amount) return false;
    }
    return true;
}
} // namespace final_candidate_detail

/// Measure the same immutable certified payload that will be published.
/// References have already been qualified for this complete frozen population.
/// No public acceptance, table construction, reference solve, or coefficient
/// copy occurs here; the existing IV solver retains a cheap Table handle.
template <FinalAccuracyTable Table>
[[nodiscard]] CollectionResult collect_final_candidate(const Table& table,
    std::span<const PhysicalReferenceRow> references, const AccuracyRequest& request,
    const InterpolatedIVSolverConfig& solver_config = {}) {
    const auto certificate=table.proof_status();
    std::optional<InterpolatedIVSolver<Table>> solver;
    if (request.valid() && certificate==PriceProofStatus::Certified &&
        std::ranges::any_of(references, [](const auto& row) {
            return row.iv_applicable && row.iv_qualification==IvQualification::Measurable;
        })) {
        auto candidate=InterpolatedIVSolver<Table>::create(table,solver_config);
        if (candidate) solver.emplace(std::move(*candidate));
    }
    auto collected=collect(references,
        [&](const PricingParams& query) -> std::optional<double> {
            if (!request.valid() || !validate_pricing_params(query) ||
                query.volatility<table.sigma_min() || query.volatility>table.sigma_max() ||
                !final_candidate_detail::admits_contract(table,query,query.discrete_dividends))
                return std::nullopt;
            return table.price(query.spot,query.strike,query.maturity,query.volatility,
                               get_zero_rate(query.rate,query.maturity));
        },
        [&](const IVQuery& query) -> std::optional<double> {
            if (!solver || !final_candidate_detail::admits_contract(table,query,query.discrete_dividends))
                return std::nullopt;
            auto result=solver->solve(query);
            if (!result) return std::nullopt;
            return result->implied_vol;
        }, {.viability=Viability::Unassessed,.certificate=certificate},
        {request.max_price_error,request.max_iv_error},request.policy);
    const auto viability=final_candidate_viability(references,collected,request);
    return {assess_request(collected.assessment.evidence(),
        {.viability=viability,.certificate=certificate},request,IvMetricKind::AbsoluteIvError),
        collected.rows};
}

} // namespace mango::detail::accuracy
