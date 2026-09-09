// SPDX-License-Identifier: MIT
#pragma once

#include "mango/option/table/reference_strike_evaluator.hpp"
#include <algorithm>
#include <type_traits>

namespace mango::detail {

template <class Built>
struct ReferenceSelectedBuild {
    Built build;
    std::shared_ptr<const ReferenceSelectionResult> selection;
};

/// Shared adapter for manual and adaptive numerical builders. Select reference
/// density before fitting; build a composed candidate only for final delivery
/// or to test whether actual composed accuracy rescues an explicit/final set.
template <class Build, class Price>
auto build_with_reference_selection(
    const SegmentedAdaptiveConfig& config, const SurfaceBounds& requested,
    std::vector<std::pair<double, double>> admitted_times,
    Build&& build, Price&& price, double price_target = 0.01, double iv_target = 2e-5,
    double vega_floor = 1e-4)
    -> std::expected<ReferenceSelectedBuild<
        typename std::invoke_result_t<Build, std::span<const double>>::value_type>, PriceTableError>
{
    using Built = typename std::invoke_result_t<Build, std::span<const double>>::value_type;
    auto evaluator = ReferenceStrikeEvaluator::create(
        config, requested, std::move(admitted_times), price_target, iv_target, vega_floor);
    if (!evaluator) return std::unexpected(evaluator.error());
    std::optional<Built> candidate;
    std::vector<double> candidate_refs;
    auto ensure_candidate = [&](std::span<const double> refs) -> std::expected<void, PriceTableError> {
        if (candidate && std::ranges::equal(candidate_refs, refs)) return {};
        auto result = build(refs);
        if (!result) return std::unexpected(result.error());
        candidate.emplace(std::move(*result));
        candidate_refs.assign(refs.begin(), refs.end());
        return {};
    };
    const SurfaceHandle handle{.price = [&](double s, double k, double t, double v, double r) {
        return price(*candidate, s, k, t, v, r);
    }};
    size_t evaluations = 0;
    auto selected = select_k_refs(config.kref_config, *requested.strike_bounds,
        [&](std::span<const double> refs) -> std::expected<ReferenceCandidateMetrics, PriceTableError> {
            ++evaluations;
            auto assessed = evaluator->evaluate(refs);
            if (!assessed) return std::unexpected(assessed.error());
            const bool inadequate = assessed->decision != ReferenceCandidateDecision::Adequate &&
                assessed->decision != ReferenceCandidateDecision::FitLimited &&
                assessed->decision != ReferenceCandidateDecision::IvUnmeasured;
            const bool final_set = !config.kref_config.K_refs.empty() ||
                refs.size() == config.kref_config.max_references ||
                evaluations == config.kref_config.max_selection_rounds;
            if (inadequate && final_set) {
                auto built = ensure_candidate(refs);
                if (!built) return std::unexpected(built.error());
                auto composed = evaluator->evaluate(refs, &handle);
                if (!composed) return std::unexpected(composed.error());
                composed->pde_solves += assessed->pde_solves;
                composed->elapsed_seconds += assessed->elapsed_seconds;
                return composed;
            }
            return assessed;
        });
    if (!selected) return std::unexpected(selected.error().error.value_or(
        PriceTableError{PriceTableErrorCode::NoViableSurface, 4,
                       static_cast<size_t>(selected.error().stop_reason)}));
    auto built = ensure_candidate(selected->refs);
    if (!built) return std::unexpected(built.error());
    // Ordinary selection establishes reference adequacy. Full fit/total
    // publication measurement remains with the acceptance gate; composed
    // rescue above is the exceptional path that measures a complete table.
    return ReferenceSelectedBuild<Built>{std::move(*candidate),
        std::make_shared<const ReferenceSelectionResult>(std::move(*selected))};
}

}  // namespace mango::detail
