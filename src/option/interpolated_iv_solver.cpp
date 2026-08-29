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
// Multiple-root bracket screen (spec D8.2)
// =====================================================================

detail::BracketScreen detail::screen_bracket(
    ObjectiveRef objective,
    double sigma_min, double sigma_max,
    double spot, double tolerance)
{
    BracketScreen out;
    out.lo = sigma_min;
    out.hi = sigma_max;

    constexpr size_t kScanPoints = 17;
    // Zero tolerance is a *price* tolerance in dollars, deliberately
    // distinct from `tolerance` (which the root finder uses for both its
    // interval and its objective convergence test).
    const double zero_tol = 1e-9 * spot;
    const double step = (sigma_max - sigma_min) / static_cast<double>(kScanPoints - 1);

    std::array<double, kScanPoints> scan_sigma{};
    std::array<double, kScanPoints> scan_f{};
    std::array<int, kScanPoints> scan_sign{};
    size_t zero_samples = 0;

    for (size_t i = 0; i < kScanPoints; ++i) {
        scan_sigma[i] = (i + 1 == kScanPoints)
            ? sigma_max
            : sigma_min + step * static_cast<double>(i);
        scan_f[i] = objective(scan_sigma[i]);
        if (!std::isfinite(scan_f[i])) {
            out.refusal = IVError{
                .code = IVErrorCode::NumericalInstability,
                .iterations = 0,
                .final_error = std::numeric_limits<double>::quiet_NaN(),
                .last_vol = scan_sigma[i]
            };
            return out;
        }
        scan_sign[i] = (std::abs(scan_f[i]) <= zero_tol)
            ? 0
            : (scan_f[i] > 0.0 ? 1 : -1);
        if (scan_sign[i] == 0) ++zero_samples;
    }

    // Every sample a zero: an unresolved continuum of roots.
    if (zero_samples == kScanPoints) {
        out.refusal = IVError{
            .code = IVErrorCode::MultipleRoots,
            .iterations = 0,
            .final_error = 0.0,
            .last_vol = sigma_min
        };
        return out;
    }

    // Walk the nonzero samples.  Consecutive zeros collapse into one
    // zero run; a run between opposite signs is one transition, a run
    // between equal signs is a tangency contact, and a run at a bracket
    // endpoint is a boundary root.
    size_t transitions = 0;
    size_t tangencies = 0;
    size_t boundary_roots = 0;
    bool leading_boundary = false;
    double lowest_feature_sigma = sigma_max;
    size_t first_lo = 0;
    size_t first_hi = 0;

    auto note_feature = [&](double sigma) {
        lowest_feature_sigma = std::min(lowest_feature_sigma, sigma);
    };

    int last_sign = 0;
    size_t last_idx = 0;
    bool have_last = false;
    for (size_t i = 0; i < kScanPoints; ++i) {
        if (scan_sign[i] == 0) continue;
        if (have_last) {
            if (scan_sign[i] != last_sign) {
                if (transitions == 0) {
                    first_lo = last_idx;
                    first_hi = i;
                }
                ++transitions;
                note_feature(scan_sigma[last_idx]);
            } else if (i > last_idx + 1) {
                ++tangencies;
                note_feature(scan_sigma[last_idx + 1]);
            }
        } else if (i > 0) {
            ++boundary_roots;
            leading_boundary = true;
            note_feature(sigma_min);
        }
        last_sign = scan_sign[i];
        last_idx = i;
        have_last = true;
    }
    if (last_idx + 1 < kScanPoints) {
        ++boundary_roots;
        note_feature(sigma_max);
    }

    // Root features found by the scan.  A tangency counts as two: an
    // even-multiplicity contact is at least a double root, which makes
    // it ambiguous on its own.  Anything beyond a single feature is
    // ambiguous by construction.
    const size_t features = transitions + 2 * tangencies + boundary_roots;
    if (features > 1) {
        out.refusal = IVError{
            .code = IVErrorCode::MultipleRoots,
            .iterations = 0,
            .final_error = static_cast<double>(features),
            .last_vol = lowest_feature_sigma
        };
        return out;
    }

    if (boundary_roots == 1) {
        // Boundary root: honor it only when it also satisfies the
        // solver's configured convergence tolerance.  zero_tol must
        // never silently loosen a user's tighter tolerance.
        const double endpoint = leading_boundary ? sigma_min : sigma_max;
        const double residual =
            std::abs(leading_boundary ? scan_f[0] : scan_f[kScanPoints - 1]);
        if (residual <= tolerance) {
            out.boundary_root = IVSuccess{
                .implied_vol = endpoint,
                .iterations = 0,
                .final_error = residual,
                .vega = std::nullopt,
                .used_rate_approximation = false
            };
            return out;
        }
        // The scan found no true bracket: report what the unscreened
        // path would have reported.
        out.refusal = IVError{
            .code = IVErrorCode::BracketingFailed,
            .iterations = 0,
            .final_error = residual,
            .last_vol = endpoint
        };
        return out;
    }

    if (transitions == 1) {
        out.lo = scan_sigma[first_lo];
        out.hi = scan_sigma[first_hi];
        out.f_lo = scan_f[first_lo];
        out.f_hi = scan_f[first_hi];
        out.check_slope = true;
    }
    // No transition: Brent runs on the full bracket, which reports
    // BracketingFailed exactly as the unscreened path does.
    return out;
}

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

    /// Diagnostics from adaptive grid refinement (spec D7), propagated from
    /// the `AnyPriceTable` this solver was built from.
    std::optional<BuildDiagnostics> diagnostics;

    template <typename T>
    explicit Impl(T s, std::optional<BuildDiagnostics> diag = std::nullopt)
        : solver(std::move(s)), diagnostics(std::move(diag)) {}
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

std::optional<BuildDiagnostics> AnyInterpIVSolver::build_diagnostics() const {
    return impl_->diagnostics;
}

namespace {

template <typename Surface>
AnyInterpIVSolver make_any_solver(
    InterpolatedIVSolver<Surface> solver,
    std::optional<BuildDiagnostics> diagnostics) {
    return AnyInterpIVSolver(std::make_unique<AnyInterpIVSolver::Impl>(
        std::move(solver), std::move(diagnostics)));
}

}  // namespace

#define MANGO_DEFINE_ANY_INTERP_OVERLOAD(Surface) \
    AnyInterpIVSolver make_any_interpolated_solver( \
        InterpolatedIVSolver<Surface> solver, \
        std::optional<BuildDiagnostics> diagnostics) { \
        return make_any_solver(std::move(solver), std::move(diagnostics)); \
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
