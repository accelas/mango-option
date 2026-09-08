// SPDX-License-Identifier: MIT
#include "mango/option/table/reference_selection.hpp"
#include "mango/option/table/adaptive_refinement.hpp"
#include <algorithm>
#include <cmath>
#include <numeric>

namespace mango {
namespace {

bool valid_summary(const std::optional<ReferenceErrorSummary>& optional) {
    if (!optional) return true;
    const auto& summary = *optional;
    // Subtract counts rather than summing them: malformed counters must not
    // wrap size_t and accidentally satisfy the accounting identity.
    size_t remaining = summary.requested;
    for (size_t count : {summary.measured, summary.filtered, summary.unresolved,
                         summary.refused, summary.structurally_exact, summary.untested}) {
        if (count > remaining) return false;
        remaining -= count;
    }
    if (remaining != 0) return false;
    if (summary.measured == 0) {
        return !summary.max_error && !summary.rms_error && !summary.max_uncertainty;
    }
    if (!summary.max_error) return false;
    for (const auto& error : {summary.max_error, summary.rms_error, summary.max_uncertainty}) {
        if (error && (!std::isfinite(*error) || *error < 0.0)) return false;
    }
    return true;
}

bool complete_evidence(const ReferenceAccuracySummary& summary) {
    // IV identifiability filters never remove a required price observation.
    if (summary.price && summary.price->filtered) return false;
    bool has_evidence = false;
    for (const auto* optional : {&summary.price, &summary.iv}) {
        if (!*optional) continue;
        const auto& channel = **optional;
        if (channel.unresolved || channel.refused || channel.untested) return false;
        has_evidence |= channel.measured > 0 || channel.structurally_exact > 0;
    }
    return has_evidence;
}

bool measured_evidence(const ReferenceAccuracySummary& summary) {
    return (summary.price && summary.price->measured > 0)
        || (summary.iv && summary.iv->measured > 0);
}

bool valid_metrics(const ReferenceCandidateMetrics& metrics) {
    if (!std::isfinite(metrics.elapsed_seconds) || metrics.elapsed_seconds < 0.0) return false;
    for (const auto* channel : {&metrics.ideal_blend, &metrics.fit, &metrics.total}) {
        if (!valid_summary(channel->price) || !valid_summary(channel->iv)) return false;
    }
    switch (metrics.decision) {
        case ReferenceCandidateDecision::Adequate:
            return complete_evidence(metrics.ideal_blend)
                || (metrics.total_target_met == true && complete_evidence(metrics.total));
        case ReferenceCandidateDecision::RefineReferences:
        case ReferenceCandidateDecision::ThresholdAmbiguous:
            return measured_evidence(metrics.ideal_blend);
        case ReferenceCandidateDecision::ReferenceUnqualified:
            return true;
        case ReferenceCandidateDecision::IvUnmeasured:
        case ReferenceCandidateDecision::FitLimited:
            return complete_evidence(metrics.ideal_blend);
    }
    return false;
}

std::vector<double> grow_references(const std::vector<double>& refs, size_t ceiling) {
    struct Gap { double width, left, middle; };
    std::vector<Gap> gaps;
    for (size_t i = 1; i < refs.size(); ++i) {
        const double middle = std::midpoint(refs[i-1], refs[i]);
        if (middle > refs[i-1] && middle < refs[i]) {
            gaps.push_back({refs[i]-refs[i-1], refs[i-1], middle});
        }
    }
    // Partial ceilings split the widest intervals first, leftmost on ties.
    // Every existing reference and both requested endpoints are retained.
    std::sort(gaps.begin(), gaps.end(), [](const Gap& a, const Gap& b) {
        return a.width != b.width ? a.width > b.width : a.left < b.left;
    });
    const size_t additions = std::min(ceiling-refs.size(), gaps.size());
    auto result = refs;
    result.reserve(refs.size()+additions);
    for (size_t i = 0; i < additions; ++i) result.push_back(gaps[i].middle);
    std::sort(result.begin(), result.end());
    return result;
}

}  // namespace

std::expected<ReferenceSelectionResult, ReferenceSelectionFailure>
select_k_refs(const MultiKRefConfig& config, const StrikeBounds& bounds,
              const ReferenceCandidateEvaluator& evaluate) {
    auto refs = resolve_k_refs(config, bounds);
    if (!refs) return std::unexpected(ReferenceSelectionFailure{
        ReferenceSelectionStopReason::InvalidConfig, refs.error(), {}});
    if (!evaluate) return std::unexpected(ReferenceSelectionFailure{
        ReferenceSelectionStopReason::InvalidConfig,
        PriceTableError{PriceTableErrorCode::InvalidConfig, 4}, {}});
    std::vector<ReferenceCandidateDiagnostics> history;
    const size_t max_references = config.max_references;
    const size_t max_rounds = config.max_selection_rounds;
    const bool explicit_refs = !config.K_refs.empty();
    std::optional<size_t> retained;
    const auto finish = [&](ReferenceSelectionStopReason reason,
                          std::optional<PriceTableError> error = std::nullopt)
        -> std::expected<ReferenceSelectionResult, ReferenceSelectionFailure> {
        if (retained) {
            auto selected_refs = history[*retained].refs;
            return ReferenceSelectionResult{std::move(selected_refs), *retained,
                                            std::move(history), reason};
        }
        return std::unexpected(ReferenceSelectionFailure{reason, error, std::move(history)});
    };
    while (true) {
        auto measured = evaluate(*refs);
        history.push_back({*refs, measured ? std::optional{*measured} : std::nullopt,
                           measured ? std::nullopt : std::optional{measured.error()}});
        if (!measured) return finish(ReferenceSelectionStopReason::EvaluatorFailed, measured.error());
        if (!valid_metrics(*measured)) return finish(ReferenceSelectionStopReason::InvalidMetrics);
        switch (measured->decision) {
            case ReferenceCandidateDecision::Adequate:
                retained = history.size()-1;
                if (!measured->prefer_refinement || explicit_refs) {
                    return finish(ReferenceSelectionStopReason::Adequate);
                }
                break;
            case ReferenceCandidateDecision::FitLimited:
                if (!retained) retained = history.size()-1;
                return finish(ReferenceSelectionStopReason::FitLimited);
            case ReferenceCandidateDecision::IvUnmeasured:
                if (!retained) retained = history.size()-1;
                return finish(ReferenceSelectionStopReason::IvUnmeasured);
            case ReferenceCandidateDecision::ReferenceUnqualified:
                return finish(ReferenceSelectionStopReason::ReferenceUnqualified);
            case ReferenceCandidateDecision::ThresholdAmbiguous:
                return finish(ReferenceSelectionStopReason::ThresholdAmbiguous);
            case ReferenceCandidateDecision::RefineReferences:
                break;
        }
        if (explicit_refs) return finish(ReferenceSelectionStopReason::ExplicitReferencesInadequate);
        if (refs->size() == max_references) return finish(ReferenceSelectionStopReason::ReferenceLimit);
        if (history.size() == max_rounds) return finish(ReferenceSelectionStopReason::RoundLimit);
        auto next = grow_references(*refs, max_references);
        if (next.size() == refs->size()) return finish(ReferenceSelectionStopReason::NoRepresentableRefinement);
        refs = std::move(next);
    }
}

}  // namespace mango
