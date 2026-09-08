// SPDX-License-Identifier: MIT
#include "mango/option/table/accuracy/collector.hpp"
#include <algorithm>
#include <cmath>
#include <limits>

namespace mango::detail::accuracy {
namespace {

struct Accumulator {
    ReferenceErrorSummary summary;
    double scaled_sum_squares=0.0;
    void observe(double error, double uncertainty) {
        const double previous_max=summary.max_error.value_or(0.0);
        if (error>previous_max) {
            const double ratio=previous_max/error;
            scaled_sum_squares=scaled_sum_squares*ratio*ratio+1.0;
        } else if (error>0.0) {
            const double ratio=error/previous_max;
            scaled_sum_squares+=ratio*ratio;
        }
        ++summary.measured;
        summary.max_error=std::max(previous_max, error);
        summary.max_uncertainty=std::max(summary.max_uncertainty.value_or(0.0), uncertainty);
        summary.rms_error=*summary.max_error*std::sqrt(scaled_sum_squares/summary.measured);
        // If the true positive RMS is below representable range, preserve its
        // positivity instead of falsely claiming every observation was exact.
        if (*summary.max_error>0.0 && *summary.rms_error==0.0)
            summary.rms_error=std::numeric_limits<double>::denorm_min();
    }
    void skip(ObservationOutcome outcome) {
        if (outcome==ObservationOutcome::ReferenceRefused || outcome==ObservationOutcome::BackendRefused
            || outcome==ObservationOutcome::BackendNonfinite || outcome==ObservationOutcome::BackendInvalidValue)
            ++summary.refused;
        else if (outcome==ObservationOutcome::FilteredTimeValue
                 || outcome==ObservationOutcome::FilteredVega) ++summary.filtered;
        else ++summary.unresolved;
    }
};

ObservationOutcome collect_value(const std::optional<double>& value, double reference,
                                 double uncertainty, bool iv, Accumulator& errors,
                                 std::optional<double>& row_error) {
    ObservationOutcome outcome=ObservationOutcome::Measured;
    if (!value) outcome=ObservationOutcome::BackendRefused;
    else if (!std::isfinite(*value)) outcome=ObservationOutcome::BackendNonfinite;
    else if (*value<0.0 || (iv && *value==0.0)) outcome=ObservationOutcome::BackendInvalidValue;
    if (outcome==ObservationOutcome::Measured) {
        row_error=std::abs(*value-reference);
        errors.observe(*row_error, uncertainty);
    }
    else errors.skip(outcome);
    return outcome;
}

bool finite_uncertainty(const std::optional<double>& value) {
    return value && std::isfinite(*value) && *value>=0.0;
}

std::optional<ObservationOutcome> reference_failure(const PhysicalReferenceRow& row) {
    if (row.price_qualification==PriceQualification::Unresolved)
        return ObservationOutcome::ReferenceUnresolved;
    if (row.price_qualification==PriceQualification::Refused)
        return ObservationOutcome::ReferenceRefused;
    const bool identified=std::ranges::any_of(row.provenance.qualification_digest,
                                            [](uint8_t byte) { return byte!=0; });
    if (row.price_qualification!=PriceQualification::Qualified
        || (row.provenance.source!=ReferenceSource::Analytic
            && row.provenance.source!=ReferenceSource::ConvergedNumerical)
        || !identified || !finite_uncertainty(row.price_uncertainty)
        || !std::isfinite(row.reference_price) || row.reference_price<0.0
        || !validate_pricing_params(row.query)) return ObservationOutcome::InvalidReference;
    return std::nullopt;
}

std::optional<ObservationOutcome> iv_reference_outcome(const PhysicalReferenceRow& row) {
    switch (row.iv_qualification) {
        case IvQualification::Unresolved: return ObservationOutcome::ReferenceUnresolved;
        case IvQualification::Refused: return ObservationOutcome::ReferenceRefused;
        case IvQualification::FilteredTimeValue: return ObservationOutcome::FilteredTimeValue;
        case IvQualification::FilteredVega: return ObservationOutcome::FilteredVega;
        case IvQualification::Measurable:
            if (finite_uncertainty(row.iv_uncertainty)) return std::nullopt;
            break;
    }
    return ObservationOutcome::InvalidReference;
}

} // namespace

CollectionResult collect(std::span<const PhysicalReferenceRow> references,
    const PriceObservation& price, const IvObservation& iv, Prerequisites prerequisites,
    Targets targets, Policy policy) {
    Accumulator price_errors, iv_errors;
    std::vector<RowEvidence> rows;
    rows.reserve(references.size());
    for (const auto& row : references) {
        RowEvidence result{.provenance=row.provenance};
        const auto failure=reference_failure(row);
        if (row.price_applicable) {
            ++price_errors.summary.requested;
            if (failure) {
                price_errors.skip(*failure);
                result.price=*failure;
            } else {
                result.price=collect_value(price ? price(row.query) : std::nullopt,
                    row.reference_price, *row.price_uncertainty, false, price_errors, result.price_error);
            }
        }
        if (row.iv_applicable) {
            ++iv_errors.summary.requested;
            if (failure) {
                iv_errors.skip(*failure);
                result.iv=*failure;
            } else if (const auto iv_outcome=iv_reference_outcome(row)) {
                iv_errors.skip(*iv_outcome);
                result.iv=*iv_outcome;
            } else {
                IVQuery query(row.query, row.reference_price, row.query.discrete_dividends);
                result.iv=collect_value(iv ? iv(query) : std::nullopt,
                    row.query.volatility, *row.iv_uncertainty, true, iv_errors, result.iv_error);
            }
        }
        rows.push_back(result);
    }
    ReferenceAccuracySummary evidence{price_errors.summary, iv_errors.summary};
    return {assess(evidence, prerequisites, targets, policy, IvMetricKind::AbsoluteIvError),
            std::move(rows)};
}

} // namespace mango::detail::accuracy
