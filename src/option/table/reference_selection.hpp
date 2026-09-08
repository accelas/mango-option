// SPDX-License-Identifier: MIT
#pragma once

#include "mango/option/table/adaptive_grid_types.hpp"
#include "mango/support/error_types.hpp"
#include <cstddef>
#include <expected>
#include <functional>
#include <optional>
#include <span>
#include <vector>

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

/// Disjoint per-channel accounting. requested equals the sum of the four
/// outcome counts. Error statistics cover qualified measured rows only;
/// measured==0 means all three optional error fields are absent, never zero.
struct ReferenceErrorSummary {
    size_t requested = 0;
    size_t measured = 0;
    size_t filtered = 0;
    size_t unresolved = 0;
    size_t refused = 0;
    std::optional<double> max_error;
    std::optional<double> rms_error;
    std::optional<double> max_uncertainty;
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

/// Validate using resolve_k_refs, then measure bounded covering candidates.
/// Automatic selection starts at up to 17 refs and inserts interval midpoints
/// (17->33->65 with default ceilings). Explicit sets are measured exactly once,
/// preserving all validated values. The seed is the first selection round.
///
/// The callback owns error budgets and evidence. Actual composed target
/// success can establish adequacy even when private component guidance misses.
/// IvUnmeasured denotes missing requested IV evidence; a price-only evaluator
/// can instead return Adequate while retaining absent/filtered IV statistics.
/// FitLimited selects adequate refs with unmet-total diagnostics; it neither
/// grows references nor introduces Gate 7's strict whole-table rejection.
[[nodiscard]] std::expected<ReferenceSelectionResult, ReferenceSelectionFailure>
select_k_refs(const MultiKRefConfig& config, const StrikeBounds& bounds,
              const ReferenceCandidateEvaluator& evaluate);

}  // namespace mango
