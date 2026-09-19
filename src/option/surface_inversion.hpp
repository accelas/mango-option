// SPDX-License-Identifier: MIT
/**
 * @file surface_inversion.hpp
 * @brief Price-to-volatility inversion on an interpolated price surface.
 *
 * The product inversion used by `InterpolatedIVSolver::solve`, factored out
 * of the solver template so that build-time validation can run the *same*
 * inversion the shipped solver runs.  Nothing here knows about price tables,
 * builders or queries: the caller binds every coordinate but sigma into the
 * two callbacks and hands over a policy.
 *
 * Deliberately free of table/builder dependencies — that is what lets the
 * adaptive refinement loop reuse it without a dependency cycle.
 */

#pragma once

#include "mango/option/iv_result.hpp"
#include "mango/option/option_spec.hpp"
#include "mango/support/error_types.hpp"

#include <cstddef>
#include <expected>
#include <optional>
#include <utility>

namespace mango {

/// Everything the inversion needs to know beyond the objective itself.
///
/// `config_*` are the caller's requested volatility limits; `published_*`
/// are the limits the surface actually covers.  `effective_sigma_bracket`
/// intersects the two (under a time-value-driven cap) and falls back to the
/// published range when the intersection is empty.
struct SurfaceInversionPolicy {
    double config_sigma_min = 0.01;    ///< Caller's minimum volatility
    double config_sigma_max = 3.0;     ///< Caller's maximum volatility
    double published_sigma_min = 0.0;  ///< Surface's minimum volatility
    double published_sigma_max = 0.0;  ///< Surface's maximum volatility

    /// Minimum signed surface vega to attempt the inversion; 0 disables the
    /// pre-check.  See `InterpolatedIVSolverConfig::vega_threshold`.
    double vega_threshold = 1e-4;

    /// Screen the bracket for multiple roots before inverting.
    /// See `InterpolatedIVSolverConfig::detect_multiple_roots`.
    bool detect_multiple_roots = true;

    double tolerance = 1e-6;  ///< Price convergence tolerance
    size_t max_iter = 50;     ///< Maximum root-finder iterations
};

namespace detail {

/// Non-owning view of the IV objective f(sigma).
///
/// `screen_bracket` runs on the `noexcept` solve path, where a
/// `std::function` conversion could heap-allocate (the solve objective
/// captures more than the small-object buffer holds) and so introduce a
/// `std::bad_alloc` that would terminate.  The view must not outlive the
/// callable it wraps; the screen only calls it during the scan.
class ObjectiveRef {
public:
    template <typename F>
    ObjectiveRef(const F& f) noexcept  // NOLINT(google-explicit-constructor)
        : ctx_(&f), call_([](const void* ctx, double sigma) {
              return (*static_cast<const F*>(ctx))(sigma);
          }) {}

    double operator()(double sigma) const { return call_(ctx_, sigma); }

private:
    const void* ctx_;
    double (*call_)(const void*, double);
};

/// Verdict of the multiple-root bracket screen (spec D8.2).
///
/// Exactly one of three outcomes is expressed:
///  - `refusal` engaged: the screen refuses the query (MultipleRoots,
///    NumericalInstability at a scan point, or BracketingFailed at an
///    endpoint whose residual misses the solver tolerance).
///  - `boundary_root` engaged: an endpoint satisfies the solver tolerance
///    and is the only root feature; the caller returns it directly (after
///    setting `used_rate_approximation`, which the screen cannot know).
///  - neither engaged: proceed to Brent on `[lo, hi]` — the full bracket,
///    or the single scan interval containing the one sign change, in which
///    case `check_slope` is set and `f_lo`/`f_hi` carry the scan samples
///    for the caller's post-hoc slope check.
struct BracketScreen {
    std::optional<IVError> refusal;
    std::optional<IVSuccess> boundary_root;
    double lo = 0.0;           ///< bracket to hand Brent
    double hi = 0.0;
    bool check_slope = false;  ///< post-hoc slope check applies to [lo, hi]
    double f_lo = 0.0;         ///< objective at lo (valid when check_slope)
    double f_hi = 0.0;         ///< objective at hi (valid when check_slope)
};

/// Screen the solve bracket for multiple roots before inverting (spec D8.2).
///
/// Samples `objective` at 17 equally spaced volatilities across
/// `[sigma_min, sigma_max]` and classifies the sign pattern: consecutive
/// zeros (|f| <= `zero_tol` = 1e-9 * spot) collapse into one run, a run
/// between opposite signs is a transition, between equal signs a tangency
/// (counted as two features — an even-multiplicity contact is at least a
/// double root), at an endpoint a boundary root.  More than one feature is
/// ambiguous by construction.  Pure function of its arguments; the
/// guarantees and blind spots are documented on
/// `InterpolatedIVSolverConfig::detect_multiple_roots`.
[[nodiscard]] BracketScreen screen_bracket(
    ObjectiveRef objective,
    double sigma_min, double sigma_max,
    double spot, double tolerance);

}  // namespace detail

/// Volatility bracket to invert over, given the policy and the quote.
///
/// Caps the upper limit by how much of the quote is time value (a quote that
/// is almost all time value needs room up to 300% vol; an almost-intrinsic
/// quote does not), intersects that with the caller's and the surface's
/// limits, and falls back to the published range when the intersection is
/// empty.
[[nodiscard]] std::pair<double, double> effective_sigma_bracket(
    double spot, double strike, OptionType type, double target_price,
    const SurfaceInversionPolicy& policy) noexcept;

/// Invert `price(sigma) = target_price` over `bracket`.
///
/// @param price   Surface price as a function of sigma alone.
/// @param vega    Surface vega as a function of sigma alone (pre-check only).
/// @param bracket Volatility bracket, normally from `effective_sigma_bracket`.
/// @param spot    Underlying spot; sets the screen's zero tolerance only.
///
/// Runs the signed quartile-vega pre-check, then the multiple-root screen,
/// then Brent, then the post-hoc slope check on a narrowed bracket.
/// `IVSuccess::used_rate_approximation` is left `false`: only the caller
/// knows whether a yield curve was collapsed to a zero rate.
[[nodiscard]] std::expected<IVSuccess, IVError> invert_price_on_surface(
    detail::ObjectiveRef price, detail::ObjectiveRef vega,
    double target_price, std::pair<double, double> bracket,
    double spot, const SurfaceInversionPolicy& policy) noexcept;

}  // namespace mango
