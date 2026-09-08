// SPDX-License-Identifier: MIT
#pragma once

#include "mango/option/table/reference_selection.hpp"
#include "mango/option/table/certification/certificate_status.hpp"
#include <cstdint>
#include <utility>

namespace mango::detail::accuracy {

/// Internal Gate 7 policy seam. No factory or persisted-format activation.
struct Targets {
    double price=0.01;                    // Absolute quote units.
    std::optional<double> iv=2e-5;         // Decimal volatility: 0.2 IV bp.
};

enum class Policy : uint8_t { Strict, BestEffort };
enum class IvMetricKind : uint8_t { Unknown, AbsoluteIvError, PriceVegaProxy };
enum class Viability : uint8_t { Unassessed, Passed, Failed };

/// Supplied by independent hard-viability checks and the final physical-price
/// proof. An unfinished proof is not a witnessed monotonicity violation.
struct Prerequisites {
    Viability viability=Viability::Unassessed;
    PriceProofStatus certificate=PriceProofStatus::NotRun;
};

enum class Decision : uint8_t {
    Accepted, AcceptedBestEffort, TargetsMissed,
    InvalidTargets, InvalidEvidence,
    ViabilityUnassessed, ViabilityFailed, CertificateNotRun, CertificateIndeterminate,
    CertificateViolated,
    PriceUnmeasured, IvUnmeasured, IncompletePriceEvidence, IncompleteIvEvidence,
    IvMetricNotActual,
};

class Assessment;
/// Input statistics describe the final composed surface against independently
/// qualified references on a fixed declared population. An IV claim requires
/// the explicit AbsoluteIvError tag; a price/vega proxy may be preserved as
/// diagnostics on a price-only assessment, but cannot satisfy an IV target.
/// A root residual or ideal-blend identity does not establish this evidence.
/// Oracle uncertainty/identifiability classification belongs to
/// the evaluator; unresolved rows remain counted. Maximum measured absolute
/// error gates acceptance; RMS and reference uncertainty are separate reports.
/// Price checks remain required where IV is filtered. BestEffort permits only
/// measured target misses, never incomplete requested evidence or failed proof.
[[nodiscard]] Assessment assess(const ReferenceAccuracySummary& evidence,
    Prerequisites prerequisites, Targets targets={}, Policy policy=Policy::Strict,
    IvMetricKind iv_metric_kind=IvMetricKind::Unknown);

/// Immutable owned snapshot: no table pointers, iteration vectors, or borrowed
/// statistics. This is measured accuracy on a declared population, not a
/// mathematical error certificate. Persistence must separately bind this
/// historical snapshot to its population/model/domain/coefficient payload;
/// copying its certificate status cannot replace certification on load.
class Assessment {
public:
    [[nodiscard]] Decision decision() const { return decision_; }
    [[nodiscard]] const ReferenceAccuracySummary& evidence() const { return evidence_; }
    [[nodiscard]] const Targets& targets() const { return targets_; }
    [[nodiscard]] Policy policy() const { return policy_; }
    [[nodiscard]] IvMetricKind iv_metric_kind() const { return iv_metric_kind_; }
    [[nodiscard]] Prerequisites prerequisites() const { return prerequisites_; }
    [[nodiscard]] std::optional<bool> price_target_met() const { return price_target_met_; }
    [[nodiscard]] std::optional<bool> iv_target_met() const { return iv_target_met_; }

private:
    friend Assessment assess(const ReferenceAccuracySummary&, Prerequisites, Targets, Policy, IvMetricKind);
    Assessment(ReferenceAccuracySummary evidence, Prerequisites prerequisites, Targets targets,
               Policy policy, Decision decision, std::optional<bool> price_met,
               std::optional<bool> iv_met, IvMetricKind iv_metric_kind)
        : evidence_(std::move(evidence)), targets_(targets), policy_(policy),
          prerequisites_(prerequisites), decision_(decision),
          price_target_met_(price_met), iv_target_met_(iv_met), iv_metric_kind_(iv_metric_kind) {}

    const ReferenceAccuracySummary evidence_;
    const Targets targets_;
    const Policy policy_;
    const Prerequisites prerequisites_;
    const Decision decision_;
    const std::optional<bool> price_target_met_, iv_target_met_;
    const IvMetricKind iv_metric_kind_;
};

} // namespace mango::detail::accuracy
