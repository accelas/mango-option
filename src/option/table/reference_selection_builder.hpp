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
    RefinementWork work;
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
    RefinementWork work;
    const auto fail = [&work](PriceTableError error) {
        error.work = std::make_shared<const RefinementWork>(work);
        return std::unexpected(std::move(error));
    };
    auto evaluator = ReferenceStrikeEvaluator::create(
        config, requested, std::move(admitted_times), price_target, iv_target, vega_floor);
    if (!evaluator) return fail(evaluator.error());
    std::optional<Built> candidate;
    std::vector<double> candidate_refs;
    auto ensure_candidate = [&](std::span<const double> refs) -> std::expected<void, PriceTableError> {
        if (candidate && std::ranges::equal(candidate_refs, refs)) return {};
        auto result = build(refs);
        if (result) {
            if constexpr (requires { result->diagnostics.work; }) work += result->diagnostics.work;
            else if constexpr (requires { result.work; }) work.tables.record(true, result.work);
            else work.tables.record(true, std::nullopt);
        } else {
            if (result.error().work) work += *result.error().work;
            else work.tables.record(false, std::nullopt);
            return fail(result.error());
        }
        candidate.emplace(std::move(*result));
        candidate_refs.assign(refs.begin(), refs.end());
        return {};
    };
    const SurfaceHandle handle{.price = [&](double s, double k, double t, double v, double r) {
        return price(*candidate, s, k, t, v, r);
    }};
    size_t evaluations = 0;
    // Preserve available reference evidence if a later rescue build fails.
    std::vector<std::optional<ReferenceCandidateMetrics>> available_metrics;
    auto selected = select_k_refs(config.kref_config, *requested.strike_bounds,
        [&](std::span<const double> refs) -> std::expected<ReferenceCandidateMetrics, PriceTableError> {
            ++evaluations;
            available_metrics.emplace_back();
            auto assessed = evaluator->evaluate(refs);
            work.selection.record(assessed.has_value(), assessed ? assessed->provider_work
                : assessed.error().work ? assessed.error().work->total_pde() : std::nullopt);
            if (!assessed) return fail(assessed.error());
            available_metrics.back() = *assessed;
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
                work.selection.record(composed.has_value(), composed ? composed->provider_work
                    : composed.error().work ? composed.error().work->total_pde() : std::nullopt);
                if (!composed) return fail(composed.error());
                if (composed->provider_work && assessed->provider_work)
                    *composed->provider_work += *assessed->provider_work;
                else composed->provider_work.reset();
                composed->pde_solves += assessed->pde_solves;
                composed->elapsed_seconds += assessed->elapsed_seconds;
                available_metrics.back() = *composed;
                return composed;
            }
            return assessed;
        });
    if (!selected) {
        auto failure = std::move(selected.error());
        for (size_t i = 0; i < std::min(failure.candidates.size(), available_metrics.size()); ++i) {
            if (!failure.candidates[i].metrics) failure.candidates[i].metrics = available_metrics[i];
        }
        auto error = failure.error.value_or(PriceTableError{
            PriceTableErrorCode::NoViableSurface, 4, static_cast<size_t>(failure.stop_reason)});
        error.reference_selection = std::make_shared<const ReferenceSelectionHistory>(failure);
        return fail(std::move(error));
    }
    auto built = ensure_candidate(selected->refs);
    if (!built) {
        auto error = built.error();
        error.reference_selection = std::make_shared<const ReferenceSelectionHistory>(*selected);
        return fail(std::move(error));
    }
    // Ordinary selection establishes reference adequacy. Full fit/total
    // publication measurement remains with the acceptance gate; composed
    // rescue above is the exceptional path that measures a complete table.
    return ReferenceSelectedBuild<Built>{std::move(*candidate),
        std::make_shared<const ReferenceSelectionResult>(std::move(*selected)), std::move(work)};
}

}  // namespace mango::detail
