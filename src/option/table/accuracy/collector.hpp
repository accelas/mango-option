// SPDX-License-Identifier: MIT
#pragma once

#include "mango/option/table/accuracy/acceptance.hpp"
#include "mango/option/option_spec.hpp"
#include <array>
#include <functional>
#include <span>
#include <vector>

namespace mango::detail::accuracy {

enum class ReferenceSource : uint8_t { Unspecified, Analytic, ConvergedNumerical };
enum class PriceQualification : uint8_t { Unresolved, Refused, Qualified };
enum class IvQualification : uint8_t {
    Unresolved, Refused, Measurable, FilteredTimeValue, FilteredVega,
};

/// Identity of the independent qualification record, including its reference
/// method, convergence evidence, physical contract, and uncertainty budgets.
/// A nonzero digest is required for measured or filtered evidence. The private
/// adapter must verify that the record matches the row; this is not a proof
/// token and the collector does not relabel a coarse solve as converged.
struct ReferenceProvenance {
    ReferenceSource source=ReferenceSource::Unspecified;
    std::array<uint8_t, 32> qualification_digest{};
};

struct PhysicalReferenceRow {
    /// Complete valuation-relative contract. Fixed-expiry dividend schedules
    /// must already be rolled to this query's remaining maturity.
    PricingParams query;
    bool price_applicable=true;
    bool iv_applicable=true;
    double reference_price=0.0;
    PriceQualification price_qualification=PriceQualification::Unresolved;
    IvQualification iv_qualification=IvQualification::Unresolved;
    std::optional<double> price_uncertainty;
    std::optional<double> iv_uncertainty;
    ReferenceProvenance provenance;
    /// Independently qualified positive sensitivity lower bound. Used only
    /// for conditioning/legacy hard-viability checks, never actual IV error.
    std::optional<double> reference_vega_lower_bound;
};

enum class ObservationOutcome : uint8_t {
    NotApplicable, Measured, FilteredTimeValue, FilteredVega,
    ReferenceUnresolved, ReferenceRefused, InvalidReference,
    BackendRefused, BackendNonfinite, BackendInvalidValue,
};

struct RowEvidence {
    ReferenceProvenance provenance;
    ObservationOutcome price=ObservationOutcome::NotApplicable;
    ObservationOutcome iv=ObservationOutcome::NotApplicable;
    /// Only measured observations have errors, for later per-stratum reports.
    std::optional<double> price_error, iv_error;
};

/// Full row ledger remains outside the compact historical assessment. Every
/// supplied row has a ledger entry, including channel-not-applicable rows.
struct CollectionResult {
    const Assessment assessment;
    const std::vector<RowEvidence> rows;
};

using PriceObservation=std::function<std::optional<double>(const PricingParams&)>;
using IvObservation=std::function<std::optional<double>(const IVQuery&)>;

/// Collect independently qualified final-price and actual inverted-IV errors.
/// IV callbacks receive the reference quote, never the fitted price; this seam
/// delegates inversion to the existing solver supplied by the adapter.
[[nodiscard]] CollectionResult collect(std::span<const PhysicalReferenceRow> references,
    const PriceObservation& price, const IvObservation& iv, Prerequisites prerequisites,
    Targets targets={}, Policy policy=Policy::Strict);

} // namespace mango::detail::accuracy
