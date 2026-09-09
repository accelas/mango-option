// SPDX-License-Identifier: MIT
#pragma once

#include "mango/option/table/adaptive_grid_types.hpp"
#include "mango/option/table/refinement_work.hpp"
#include "mango/support/error_types.hpp"
#include <cstddef>
#include <expected>
#include <functional>
#include <optional>
#include <span>
#include <vector>
#include <variant>

namespace mango {

/// Evaluator-owned classification of reference approximation, not Gate 7's
/// whole-table publication policy. FitLimited means references are adequate.
enum class ReferenceCandidateDecision {
    Adequate,
    RefineReferences,
    ThresholdAmbiguous,
    ReferenceUnqualified,
    IvUnmeasured,
    FitLimited,
};

/// Disjoint per-channel accounting. requested equals the sum of the six
/// outcome counts. Error statistics cover qualified measured rows only;
/// measured==0 means all three optional error fields are absent, never zero.
/// Filtering applies to IV; a filtered price row cannot support acceptance.
struct ReferenceErrorSummary {
    size_t requested = 0;
    size_t measured = 0;
    size_t filtered = 0;
    size_t unresolved = 0;
    size_t refused = 0;
    std::optional<double> max_error;
    std::optional<double> rms_error;
    std::optional<double> max_uncertainty;
    /// Exact zero quote-price perturbation by model identity, not a numeric
    /// price/IV observation. These rows do not contribute error statistics.
    size_t structurally_exact = 0;
    /// Remaining declared probes after a qualified early rejection witness.
    size_t untested = 0;
};

struct ReferenceAccuracySummary {
    std::optional<ReferenceErrorSummary> price;
    std::optional<ReferenceErrorSummary> iv;
};

struct ReferenceCandidateMetrics {
    ReferenceCandidateDecision decision = ReferenceCandidateDecision::ReferenceUnqualified;
    /// Optional pursuit of extra reference headroom after Adequate. A passing
    /// candidate is retained if later work fails or a resource ceiling is met.
    bool prefer_refinement = false;
    /// Unknown when no composed candidate was assessed. FitLimited reports
    /// false; reference adequacy alone does not establish whole-table accuracy.
    std::optional<bool> total_target_met;
    ReferenceAccuracySummary ideal_blend;
    ReferenceAccuracySummary fit;
    ReferenceAccuracySummary total;
    size_t pde_solves = 0;
    double elapsed_seconds = 0.0;
    /// Actual solve work when supplied by the numerical provider. The legacy
    /// pde_solves counter may count requests rejected before solver creation.
    std::optional<PdeWork> provider_work;
};

using ReferenceCandidateEvaluator = std::function<
    std::expected<ReferenceCandidateMetrics, PriceTableError>(std::span<const double>)>;

struct ReferenceCandidateDiagnostics {
    std::vector<double> refs;
    std::optional<ReferenceCandidateMetrics> metrics;
    std::optional<PriceTableError> evaluator_error;
};

/// Why selection stopped. A retained adequate candidate can be returned even
/// when optional refinement ends at a ceiling or fails; consult picked_candidate.
enum class ReferenceSelectionStopReason {
    Adequate,
    FitLimited,
    InvalidConfig,
    InvalidMetrics,
    ExplicitReferencesInadequate,
    ReferenceLimit,
    RoundLimit,
    NoRepresentableRefinement,
    ThresholdAmbiguous,
    ReferenceUnqualified,
    IvUnmeasured,
    EvaluatorFailed,
};

struct ReferenceSelectionResult {
    std::vector<double> refs;
    size_t picked_candidate = 0;
    std::vector<ReferenceCandidateDiagnostics> candidates;
    ReferenceSelectionStopReason stop_reason = ReferenceSelectionStopReason::Adequate;
};

struct ReferenceSelectionFailure {
    ReferenceSelectionStopReason stop_reason;
    std::optional<PriceTableError> error;
    std::vector<ReferenceCandidateDiagnostics> candidates;
};

/// Detached typed history for failed publication. Reference selection may
/// itself fail, or succeed before a subsequent numerical build fails.
class ReferenceSelectionHistory {
public:
    explicit ReferenceSelectionHistory(const ReferenceSelectionResult& result) : outcome_(result) {}
    explicit ReferenceSelectionHistory(const ReferenceSelectionFailure& failure) : outcome_(failure) {}
    [[nodiscard]] const auto& outcome() const noexcept { return outcome_; }
private:
    std::variant<ReferenceSelectionResult, ReferenceSelectionFailure> outcome_;
};

/// Validate using resolve_k_refs, then measure bounded covering candidates.
/// Automatic selection starts with resolve_k_refs' cheap covering seed and
/// inserts interval midpoints (normally 3->5->9->17->33->65). Explicit sets
/// are measured exactly once,
/// preserving all validated values. The seed is the first selection round.
///
/// The callback owns error budgets and evidence. Actual composed target
/// success can establish adequacy even when private component guidance misses.
/// It uses a fixed declared population and criteria. An Adequate assessment
/// remains valid for that candidate while later vectors are evaluated.
/// The reference span is valid only during the synchronous callback.
///
/// Accepting decisions require complete ideal evidence (at least one measured
/// or structurally exact row, no unresolved/refused/untested rows). Adequate
/// may instead use complete total evidence with total_target_met=true: every
/// composed price row must be measured, and composed IV evidence cannot use
/// ideal structural identities. The callback owns which targets were requested;
/// an absent or legitimately filtered IV channel need not prevent price-only
/// acceptance.
/// IvUnmeasured may instead use a complete measured composed-price assessment;
/// missing composed IV observations remain untested/unresolved, not structurally
/// exact or filtered by an ideal-blend identity. This makes no whole-IV claim.
/// RefineReferences/ThresholdAmbiguous require a measured ideal witness;
/// remaining probes may be explicitly untested. Malformed count/statistic
/// reports stop with InvalidMetrics and remain in the diagnostic history.
///
/// IvUnmeasured selects price-adequate refs while retaining absent/filtered IV
/// statistics. It makes no whole-IV-target claim; Gate 7 owns publication.
/// FitLimited selects adequate refs with unmet-total diagnostics; it neither
/// grows references nor introduces Gate 7's strict whole-table rejection.
[[nodiscard]] std::expected<ReferenceSelectionResult, ReferenceSelectionFailure>
select_k_refs(const MultiKRefConfig& config, const StrikeBounds& bounds,
              const ReferenceCandidateEvaluator& evaluate);

}  // namespace mango
