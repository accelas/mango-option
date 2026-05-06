// SPDX-License-Identifier: MIT
/**
 * @file interpolated_iv_solver.cpp
 * @brief Explicit template instantiations and type-erased solver wrapper.
 */

#include "mango/option/interpolated_iv_solver.hpp"

#include "mango/option/table/bspline/bspline_3d_surface.hpp"
#include "mango/option/table/bspline/bspline_surface.hpp"
#include "mango/option/table/chebyshev/chebyshev_3d_surface.hpp"
#include "mango/option/table/chebyshev/chebyshev_adaptive.hpp"
#include "mango/option/table/chebyshev/chebyshev_surface.hpp"

#include <memory>
#include <variant>

namespace mango {

// =====================================================================
// Explicit template instantiations
// =====================================================================

template class InterpolatedIVSolver<BSplinePriceTable>;
template class InterpolatedIVSolver<BSplineMultiKRefSurface>;
template class InterpolatedIVSolver<ChebyshevSurface>;
template class InterpolatedIVSolver<ChebyshevMultiKRefSurface>;
template class InterpolatedIVSolver<BSpline3DPriceTable>;
template class InterpolatedIVSolver<Chebyshev3DPriceTable>;
template class InterpolatedIVSolver<detail::SharedPriceTableSurface<BSplinePriceTable>>;
template class InterpolatedIVSolver<detail::SharedPriceTableSurface<BSplineMultiKRefSurface>>;
template class InterpolatedIVSolver<detail::SharedPriceTableSurface<ChebyshevSurface>>;
template class InterpolatedIVSolver<detail::SharedPriceTableSurface<ChebyshevMultiKRefSurface>>;
template class InterpolatedIVSolver<detail::SharedPriceTableSurface<BSpline3DPriceTable>>;
template class InterpolatedIVSolver<detail::SharedPriceTableSurface<Chebyshev3DPriceTable>>;

// =====================================================================
// AnyInterpIVSolver: pimpl implementation
// =====================================================================

struct AnyInterpIVSolver::Impl {
    using SolverVariant = std::variant<
        InterpolatedIVSolver<BSplinePriceTable>,
        InterpolatedIVSolver<BSplineMultiKRefSurface>,
        InterpolatedIVSolver<ChebyshevSurface>,
        InterpolatedIVSolver<ChebyshevMultiKRefSurface>,
        InterpolatedIVSolver<BSpline3DPriceTable>,
        InterpolatedIVSolver<Chebyshev3DPriceTable>,
        SharedPriceTableSolver<BSplinePriceTable>,
        SharedPriceTableSolver<BSplineMultiKRefSurface>,
        SharedPriceTableSolver<ChebyshevSurface>,
        SharedPriceTableSolver<ChebyshevMultiKRefSurface>,
        SharedPriceTableSolver<BSpline3DPriceTable>,
        SharedPriceTableSolver<Chebyshev3DPriceTable>>;

    SolverVariant solver;

    template <typename T>
    explicit Impl(T s) : solver(std::move(s)) {}
};

AnyInterpIVSolver::AnyInterpIVSolver(std::unique_ptr<Impl> impl)
    : impl_(std::move(impl)) {}

AnyInterpIVSolver::AnyInterpIVSolver(AnyInterpIVSolver&&) noexcept = default;
AnyInterpIVSolver& AnyInterpIVSolver::operator=(AnyInterpIVSolver&&) noexcept = default;
AnyInterpIVSolver::~AnyInterpIVSolver() = default;

std::expected<IVSuccess, IVError>
AnyInterpIVSolver::solve(const IVQuery& query) const {
    return std::visit([&](const auto& solver) {
        return solver.solve(query);
    }, impl_->solver);
}

BatchIVResult
AnyInterpIVSolver::solve_batch(const std::vector<IVQuery>& queries) const {
    return std::visit([&](const auto& solver) {
        return solver.solve_batch(queries);
    }, impl_->solver);
}

namespace {

template <typename Surface>
AnyInterpIVSolver make_any_solver(
    InterpolatedIVSolver<Surface> solver) {
    return AnyInterpIVSolver(
        std::make_unique<AnyInterpIVSolver::Impl>(std::move(solver)));
}

}  // namespace

#define MANGO_DEFINE_ANY_INTERP_OVERLOAD(Surface) \
    AnyInterpIVSolver make_any_interpolated_solver( \
        InterpolatedIVSolver<Surface> solver) { \
        return make_any_solver(std::move(solver)); \
    }

MANGO_DEFINE_ANY_INTERP_OVERLOAD(BSplinePriceTable)
MANGO_DEFINE_ANY_INTERP_OVERLOAD(BSplineMultiKRefSurface)
MANGO_DEFINE_ANY_INTERP_OVERLOAD(ChebyshevSurface)
MANGO_DEFINE_ANY_INTERP_OVERLOAD(ChebyshevMultiKRefSurface)
MANGO_DEFINE_ANY_INTERP_OVERLOAD(BSpline3DPriceTable)
MANGO_DEFINE_ANY_INTERP_OVERLOAD(Chebyshev3DPriceTable)
MANGO_DEFINE_ANY_INTERP_OVERLOAD(detail::SharedPriceTableSurface<BSplinePriceTable>)
MANGO_DEFINE_ANY_INTERP_OVERLOAD(detail::SharedPriceTableSurface<BSplineMultiKRefSurface>)
MANGO_DEFINE_ANY_INTERP_OVERLOAD(detail::SharedPriceTableSurface<ChebyshevSurface>)
MANGO_DEFINE_ANY_INTERP_OVERLOAD(detail::SharedPriceTableSurface<ChebyshevMultiKRefSurface>)
MANGO_DEFINE_ANY_INTERP_OVERLOAD(detail::SharedPriceTableSurface<BSpline3DPriceTable>)
MANGO_DEFINE_ANY_INTERP_OVERLOAD(detail::SharedPriceTableSurface<Chebyshev3DPriceTable>)

#undef MANGO_DEFINE_ANY_INTERP_OVERLOAD

}  // namespace mango
