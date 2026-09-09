// SPDX-License-Identifier: MIT
#pragma once

#include "mango/option/table/reference_selection.hpp"
#include "mango/option/table/accuracy/report.hpp"
#include "mango/option/table/certification/certificate_status.hpp"
#include <cstdint>
#include <utility>

namespace mango::detail::accuracy {

/// Internal Gate 7 policy seam. No factory or persisted-format activation.
struct Targets {
    double price=kDefaultMaxPriceError;                    // Absolute quote units.
    std::optional<double> iv=kDefaultMaxIvError;         // Decimal volatility: 0.2 IV bp.
};

using Policy = AccuracyPolicy;
using IvMetricKind = AccuracyIvMetricKind;
using Viability = AccuracyViability;
using Prerequisites = AccuracyPrerequisites;
using Decision = AccuracyDecision;
using Assessment = AccuracyReport;

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

/// Canonical request adapter. This evaluates policy only; it does not create
/// a certificate or grant publication rights to an untrusted surface.
[[nodiscard]] Assessment assess_request(const ReferenceAccuracySummary& evidence,
    Prerequisites prerequisites, const AccuracyRequest& request,
    IvMetricKind iv_metric_kind=IvMetricKind::Unknown);

} // namespace mango::detail::accuracy
