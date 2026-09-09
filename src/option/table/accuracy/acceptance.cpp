// SPDX-License-Identifier: MIT
#include "mango/option/table/accuracy/acceptance.hpp"
#include <cmath>

namespace mango::detail::accuracy {
namespace {

bool valid_channel(const std::optional<ReferenceErrorSummary>& channel, bool price) {
    if (!channel) return true;
    // Reuse #460's accounting vocabulary, but an ideal-blend identity is
    // never an observation of the final fitted price or IV error.
    if (channel->structurally_exact || (price && channel->filtered)) return false;
    size_t remaining=channel->requested;
    for (size_t count : {channel->measured, channel->filtered, channel->unresolved,
                         channel->refused, channel->untested}) {
        if (count > remaining) return false;
        remaining-=count;
    }
    if (remaining) return false;
    if (!channel->measured)
        return !channel->max_error && !channel->rms_error && !channel->max_uncertainty;
    if (!channel->max_error) return false;
    for (auto value : {channel->max_error, channel->rms_error, channel->max_uncertainty}) {
        if (value && (!std::isfinite(*value) || *value < 0.0)) return false;
    }
    return true;
}

bool incomplete(const std::optional<ReferenceErrorSummary>& channel) {
    return channel && (channel->unresolved || channel->refused || channel->untested);
}

std::optional<bool> target_met(const std::optional<ReferenceErrorSummary>& channel, double target) {
    if (!channel || !channel->max_error) return std::nullopt;
    if (*channel->max_error > target) return false;
    if (incomplete(channel)) return std::nullopt;
    return true;
}

} // namespace

Assessment assess(const ReferenceAccuracySummary& evidence, Prerequisites prerequisites,
                  Targets targets, Policy policy, IvMetricKind iv_metric_kind) {
    if (!AccuracyRequest{targets.price, targets.iv, policy}.valid()) {
        return {evidence, prerequisites, AccuracyRequest{targets.price, targets.iv, policy}, Decision::InvalidTargets,
                std::nullopt, std::nullopt, iv_metric_kind};
    }
    if (!valid_channel(evidence.price, true) || !valid_channel(evidence.iv, false)) {
        return {evidence, prerequisites, AccuracyRequest{targets.price, targets.iv, policy}, Decision::InvalidEvidence,
                std::nullopt, std::nullopt, iv_metric_kind};
    }
    const auto price_met=target_met(evidence.price, targets.price);
    std::optional<bool> iv_met=std::nullopt;
    if (targets.iv && iv_metric_kind==IvMetricKind::AbsoluteIvError)
        iv_met=target_met(evidence.iv, *targets.iv);
    auto decision=price_met==true && (!targets.iv || iv_met==true)
        ? Decision::Accepted : policy==Policy::BestEffort
            ? Decision::AcceptedBestEffort : Decision::TargetsMissed;
    if (prerequisites.viability!=Viability::Passed) {
        decision=prerequisites.viability==Viability::Failed
            ? Decision::ViabilityFailed : Decision::ViabilityUnassessed;
    } else if (prerequisites.certificate!=PriceProofStatus::Certified) {
        decision=prerequisites.certificate==PriceProofStatus::NegativeWitness
            ? Decision::CertificateViolated : prerequisites.certificate==PriceProofStatus::NotRun
                ? Decision::CertificateNotRun : Decision::CertificateIndeterminate;
    }
    else if (incomplete(evidence.price)) decision=Decision::IncompletePriceEvidence;
    else if (!evidence.price || evidence.price->measured==0) decision=Decision::PriceUnmeasured;
    else if (targets.iv && incomplete(evidence.iv)) decision=Decision::IncompleteIvEvidence;
    else if (targets.iv && (!evidence.iv || evidence.iv->measured==0)) decision=Decision::IvUnmeasured;
    else if (targets.iv && iv_metric_kind!=IvMetricKind::AbsoluteIvError) decision=Decision::IvMetricNotActual;
    return {evidence, prerequisites, AccuracyRequest{targets.price, targets.iv, policy}, decision, price_met, iv_met, iv_metric_kind};
}

Assessment assess_request(const ReferenceAccuracySummary& evidence,
    Prerequisites prerequisites, const AccuracyRequest& request, IvMetricKind iv_metric_kind) {
    return assess(evidence, prerequisites, Targets{request.max_price_error, request.max_iv_error},
                  request.policy, iv_metric_kind);
}

} // namespace mango::detail::accuracy
