// SPDX-License-Identifier: MIT
#include "mango/option/table/accuracy/final_candidate.hpp"

namespace mango::detail::accuracy {
Viability final_candidate_viability(std::span<const PhysicalReferenceRow> references,
    const CollectionResult& collected, const AccuracyRequest& request) {
    if (!request.valid() || references.size()!=collected.rows.size()) return Viability::Unassessed;
    const auto& price=collected.assessment.evidence().price;
    if (!price || price->measured==0 || price->measured!=price->requested)
        return Viability::Unassessed;
    bool guarded=false;
    bool missing_required_guard=false;
    for (size_t i=0; i<references.size(); ++i) {
        const auto& reference=references[i];
        const auto& row=collected.rows[i];
        if (!reference.price_applicable) continue;
        if (!row.price_error) return Viability::Unassessed;
        const auto lower=reference.reference_vega_lower_bound;
        if (lower && std::isfinite(*lower) && *lower>0.) {
            guarded=true;
            // Division overflow is a witnessed failure of this conservative
            // guard, never an accepted nonfinite diagnostic.
            if (*row.price_error / *lower > kFinalPriceVegaViabilityLimit)
                return Viability::Failed;
        } else if (*row.price_error>request.max_price_error ||
            (request.max_iv_error && reference.iv_qualification==IvQualification::Measurable)) {
            missing_required_guard=true;
        }
    }
    if (!request.max_iv_error && price->max_error && *price->max_error<=request.max_price_error)
        return Viability::Passed;
    if (guarded && !missing_required_guard) return Viability::Passed;
    // No new absolute-price best-effort ceiling is invented for flat regions.
    return Viability::Unassessed;
}

// Compile every supported financial path here, including segmented composition
// and the dimensionless Greek/volatility mapping inherited by its IV solver.
#define MANGO_INSTANTIATE_FINAL_ACCURACY(Table) \
    template CollectionResult collect_final_candidate<Table>(const Table&, \
        std::span<const PhysicalReferenceRow>, const AccuracyRequest&, \
        const InterpolatedIVSolverConfig&);
MANGO_INSTANTIATE_FINAL_ACCURACY(BSplinePriceTable)
MANGO_INSTANTIATE_FINAL_ACCURACY(BSplineMultiKRefSurface)
MANGO_INSTANTIATE_FINAL_ACCURACY(ChebyshevSurface)
MANGO_INSTANTIATE_FINAL_ACCURACY(ChebyshevMultiKRefSurface)
MANGO_INSTANTIATE_FINAL_ACCURACY(BSpline3DPriceTable)
MANGO_INSTANTIATE_FINAL_ACCURACY(Chebyshev3DPriceTable)
#undef MANGO_INSTANTIATE_FINAL_ACCURACY

} // namespace mango::detail::accuracy
