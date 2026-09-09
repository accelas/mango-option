// SPDX-License-Identifier: MIT
#pragma once
#include "mango/math/bspline/bspline_nd.hpp"
#include "mango/math/chebyshev/chebyshev_modal_interpolant.hpp"
#include "mango/math/proof/bernstein.hpp"
#include "mango/option/table/bspline/bspline_types.hpp"
#include "mango/option/table/certification/certificate_status.hpp"
#include "mango/option/table/surface_bounds.hpp"
#include <array>
#include <optional>
namespace mango::detail::certification {
struct PhysicalCellProof {
    PriceProofStatus status = PriceProofStatus::Indeterminate;
    proof::StopReason reason = proof::StopReason::None;
    std::size_t nodes = 0;
    std::optional<PricingParams> witness;
    proof::Interval witness_vega_per_strike{0};
};
using ModalSegmentedLeaf = TransformLeaf<ChebyshevModalInterpolant<4>, StandardTransform4D>;
using ModalTauInner = SplitSurface<ModalSegmentedLeaf, TauSegmentSplit>;
using ModalMultiKRefInner = SplitSurface<ModalTauInner, MultiKRefSplit>;
PhysicalCellProof prove_segmented_chebyshev(const ModalMultiKRefInner &inner, OptionType type,
                                            double dividend_yield, const SurfaceBounds &requested,
                                            proof::ProofBudget budget = {});
PhysicalCellProof prove_continuous_chebyshev(const ChebyshevPolynomial &polynomial,
                                             double reference_strike, OptionType type,
                                             double dividend_yield, const SurfaceBounds &requested,
                                             proof::ProofBudget budget = {});
PhysicalCellProof prove_dimensionless_chebyshev(const ChebyshevPolynomial &polynomial,
                                                double reference_strike, OptionType type,
                                                const SurfaceBounds &requested,
                                                proof::ProofBudget budget = {});
/// Segmented normalized total-price proof. Positive reference weights are
/// proved at requested strike/bracket endpoints; no independent K axis.
/// This proves the current per-leaf zero floors BEFORE reference blending.
/// An outer intrinsic projection would require a different K-domain proof.
PhysicalCellProof prove_segmented_bspline(const BSplineMultiKRefInner &inner, OptionType type,
                                          double dividend_yield, const SurfaceBounds &requested,
                                          proof::ProofBudget budget = {});
/// Internal whole-domain continuous EEP proof, including grid-clamped
/// coordinate branches. Still diagnostic evidence; cannot publish a table.
PhysicalCellProof prove_continuous_bspline(const BSplineND<double, 4> &spline,
                                           double reference_strike, OptionType type,
                                           double dividend_yield, const SurfaceBounds &requested,
                                           proof::ProofBudget budget = {});
/// Dimensionless (x,sigma^2*tau/2,log(2*r/sigma^2)) EEP proof for q=0,r>0.
PhysicalCellProof prove_dimensionless_bspline(const BSplineND<double, 3> &spline,
                                              double reference_strike, OptionType type,
                                              const SurfaceBounds &requested,
                                              proof::ProofBudget budget = {});
/// Bounds one explicit stored-knot cell of the continuous EEP expression.
/// A certified cell is not a whole-table certificate. The caller must cover
/// every admitted cell, clamped coordinate branch and composition.
/// Negative witnesses must be representable physical quotes inside both this
/// cell and requested bounds. No sample or coefficient sign mints evidence.
PhysicalCellProof prove_continuous_bspline_cell(const BSplineND<double, 4> &spline,
                                                const std::array<std::size_t, 4> &spans,
                                                double reference_strike, OptionType type,
                                                double dividend_yield,
                                                const SurfaceBounds &requested,
                                                proof::ProofBudget budget = {});
} // namespace mango::detail::certification
