// SPDX-License-Identifier: MIT
#pragma once

#include "mango/option/table/accuracy/request.hpp"
#include "mango/option/table/reference_selection.hpp"
#include "mango/option/table/certification/certificate_status.hpp"
#include <utility>

namespace mango {

enum class AccuracyIvMetricKind : uint8_t { Unknown, AbsoluteIvError, PriceVegaProxy };
enum class AccuracyViability : uint8_t { Unassessed, Passed, Failed };

/// Supplied by independent hard-viability checks and the final physical-price
/// proof. An unfinished proof is not a witnessed monotonicity violation.
struct AccuracyPrerequisites {
    AccuracyViability viability=AccuracyViability::Unassessed;
    PriceProofStatus certificate=PriceProofStatus::NotRun;
};

enum class AccuracyDecision : uint8_t {
    Accepted, AcceptedBestEffort, TargetsMissed,
    InvalidTargets, InvalidEvidence,
    ViabilityUnassessed, ViabilityFailed, CertificateNotRun, CertificateIndeterminate,
    CertificateViolated,
    PriceUnmeasured, IvUnmeasured, IncompletePriceEvidence, IncompleteIvEvidence,
    IvMetricNotActual,
};

class AccuracyReport;
namespace detail::accuracy {
struct Targets;
AccuracyReport assess(const ReferenceAccuracySummary&, AccuracyPrerequisites, Targets,
                      AccuracyPolicy, AccuracyIvMetricKind);
} // namespace detail::accuracy

/// Immutable measured-accuracy snapshot for a declared final population.
/// This metadata is not a proof token: copying certificate diagnostics cannot
/// certify a table, and loading must reprove the actual numeric payload.
/// Missing evidence remains absent. Reports own no table or solver payload.
class AccuracyReport {
public:
    [[nodiscard]] AccuracyDecision decision() const { return decision_; }
    [[nodiscard]] const ReferenceAccuracySummary& evidence() const { return evidence_; }
    [[nodiscard]] const AccuracyRequest& request() const { return request_; }
    [[nodiscard]] AccuracyPolicy policy() const { return request_.policy; }
    [[nodiscard]] AccuracyIvMetricKind iv_metric_kind() const { return iv_metric_kind_; }
    [[nodiscard]] AccuracyPrerequisites prerequisites() const { return prerequisites_; }
    [[nodiscard]] std::optional<bool> price_target_met() const { return price_target_met_; }
    [[nodiscard]] std::optional<bool> iv_target_met() const { return iv_target_met_; }

private:
    friend AccuracyReport detail::accuracy::assess(const ReferenceAccuracySummary&,
        AccuracyPrerequisites, detail::accuracy::Targets, AccuracyPolicy, AccuracyIvMetricKind);
    AccuracyReport(ReferenceAccuracySummary evidence, AccuracyPrerequisites prerequisites,
        AccuracyRequest request, AccuracyDecision decision, std::optional<bool> price_met,
        std::optional<bool> iv_met, AccuracyIvMetricKind iv_metric_kind)
        : evidence_(std::move(evidence)), request_(request), prerequisites_(prerequisites),
          decision_(decision), price_target_met_(price_met), iv_target_met_(iv_met),
          iv_metric_kind_(iv_metric_kind) {}

    const ReferenceAccuracySummary evidence_;
    const AccuracyRequest request_;
    const AccuracyPrerequisites prerequisites_;
    const AccuracyDecision decision_;
    const std::optional<bool> price_target_met_, iv_target_met_;
    const AccuracyIvMetricKind iv_metric_kind_;
};

} // namespace mango
