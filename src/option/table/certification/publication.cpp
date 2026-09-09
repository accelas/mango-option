// SPDX-License-Identifier: MIT
#include "mango/option/table/certification/publication.hpp"
#include "mango/option/table/certification/continuous_cell.hpp"
#include <algorithm>
#include <cmath>
#include <limits>
#include <type_traits>
namespace mango::detail::certification {
namespace {
bool metadata_valid(const SurfaceBounds &b, OptionType type, double q,
                    const std::optional<FixedExpiryMetadata> &model, bool segmented) {
    for (double value :
         {b.m_min, b.m_max, b.tau_min, b.tau_max, b.sigma_min, b.sigma_max, b.rate_min, b.rate_max})
        if (!std::isfinite(value))
            return false;
    const MoneynessBounds ratio =
        b.ratio_bounds.value_or(MoneynessBounds{std::exp(b.m_min), std::exp(b.m_max)});
    if (!ratio.valid() || b.m_max < b.m_min || b.tau_min < 0 || b.tau_max <= 0 ||
        b.tau_max < b.tau_min || b.sigma_min <= 0 || b.sigma_max < b.sigma_min ||
        b.rate_max < b.rate_min || !std::isfinite(q) || q < 0 ||
        (type != OptionType::PUT && type != OptionType::CALL) ||
        (b.strike_bounds && !b.strike_bounds->valid()) || (model && !model->valid(b.tau_max)))
        return false;
    if (segmented)
        return b.strike_bounds.has_value() && model.has_value();
    return !model || model->discrete_dividends.empty();
}
bool dimensionless_inputs_representable(const SurfaceBounds &b) {
    const double sigma_squared_min = b.sigma_min * b.sigma_min;
    const double sigma_squared_max = b.sigma_max * b.sigma_max;
    const double twice_rate_min = 2 * b.rate_min, twice_rate_max = 2 * b.rate_max;
    return sigma_squared_min > 0 && std::isfinite(sigma_squared_max) &&
           std::isfinite(sigma_squared_max * b.tau_max) && std::isfinite(twice_rate_max) &&
           twice_rate_min / sigma_squared_max > 0 &&
           std::isfinite(twice_rate_max / sigma_squared_min);
}

// Bound binary64 value arithmetic separately from the real-function proof.
// IEEE rounded operations satisfy |fl(z)-z| <= u|z|+eta with conservative
// u=epsilon and eta=smallest subnormal. Differences of binary64 knot/query
// values have the usual relative bound even when subnormal (then exact).
template <std::size_t N>
bool bspline_value_representable(const BSplineND<double, N> &spline, double output_scale = 1) {
    using proof::Interval;
    if (!std::isfinite(output_scale) || output_scale <= 0)
        return false;
    const Interval u(std::numeric_limits<double>::epsilon()),
        eta(std::numeric_limits<double>::denorm_min());
    const Interval one(1), limit(std::numeric_limits<double>::max());
    const auto rounded = square(one + u), ratio_error = rounded / (one - u);
    std::array<Interval, N> weight_sums;
    for (std::size_t d = 0; d < N; ++d) {
        const auto &knots = spline.knots(d);
        if (knots.size() < 8 || spline.grid(d).front() != knots.front() ||
            spline.grid(d).back() != knots.back())
            return false;
        const double width = knots.back() - knots.front();
        if (!std::isfinite(width) || width <= 0)
            return false;
        double minimum = std::numeric_limits<double>::infinity();
        for (std::size_t i = 1; i < knots.size(); ++i) {
            const double spacing = knots[i] - knots[i - 1];
            if (!std::isfinite(spacing) || spacing < 0)
                return false;
            if (spacing > 0)
                minimum = std::min(minimum, spacing);
        }
        const double native_ratio = width / minimum;
        if (!std::isfinite(native_ratio) || native_ratio < 1)
            return false;
        // Every computed ratio is finite; four times this computed bound
        // also encloses the corresponding exact-knot ratio.
        const Interval ratio = Interval(4) * Interval(native_ratio);
        Interval amplification(1), error(0);
        for (int degree = 0; degree < 3; ++degree) {
            error = rounded * (Interval(2) * ratio_error * ratio * error +
                               Interval(2) * eta * (amplification + error)) +
                    (Interval(3) + Interval(2) * u) * eta;
            amplification = rounded * ratio_error * amplification;
            if (!(limit - amplification - error).nonnegative())
                return false;
        }
        // Exact cubic bases are nonnegative and sum to one. Inactive bases
        // stay exact zero; each of the four active terms has absolute error E.
        weight_sums[d] = amplification + Interval(4) * error;
    }
    double maximum = 0;
    for (double c : spline.coefficients()) {
        if (!std::isfinite(c))
            return false;
        maximum = std::max(maximum, std::abs(c));
    }
    if (maximum == 0)
        return true;
    Interval contraction(1);
    for (int i = 0; i < 8; ++i)
        contraction = contraction * (one + u);
    Interval bound(maximum);
    for (std::size_t d = N; d > 0; --d) {
        bound = contraction * weight_sums[d - 1] * bound + Interval(16) * eta;
        if (!(limit - bound).nonnegative())
            return false;
    }
    // Allow the leaf's scale-identity rounding, then the actual dollar
    // normalization performed by a temporal reference member when present.
    bound = (bound * rounded * rounded + eta) * Interval(output_scale) * (one + u) + eta;
    return (limit - bound).nonnegative();
}
bool modal_value_representable(const ChebyshevPolynomial &polynomial, double output_scale = 1) {
    using proof::Interval;
    if (!std::isfinite(output_scale) || output_scale <= 0)
        return false;
    const Interval u(std::numeric_limits<double>::epsilon()),
        eta(std::numeric_limits<double>::denorm_min()), one(1);
    const auto local_error = Interval(16) * u + Interval(4) * eta;
    std::vector<std::vector<Interval>> basis(polynomial.shape().size());
    for (std::size_t d = 0; d < polynomial.shape().size(); ++d)
        for (std::size_t k = 0; k < polynomial.shape()[d]; ++k)
            basis[d].push_back(one + local_error *
                                         Interval(static_cast<double>(k * (k ? k - 1 : 0))) /
                                         Interval(2));
    Interval sum;
    for (std::size_t i = 0; i < polynomial.coefficients().size(); ++i) {
        const double c = polynomial.coefficients()[i];
        if (!std::isfinite(c))
            return false;
        if (c == 0)
            continue;
        Interval term(std::abs(c));
        auto index = i;
        for (std::size_t d = polynomial.shape().size(); d > 0; --d) {
            term = term * basis[d - 1][index % polynomial.shape()[d - 1]];
            index /= polynomial.shape()[d - 1];
        }
        sum = sum + term;
    }
    if (sum.exact_zero())
        return true;
    // For |x|<=1, degree<=256, local recurrence error<=16u+4eta is
    // propagated by |U_j(x)|<=j+1. It sums to <1e-9, closing the |T_hat|<2
    // induction. Auxiliary derivative recurrences stay finite (4^n,n*4^n).
    // A contraction uses <=8*coefficient_count operations; an absolute sum
    // bound handles signed cancellation without assuming positive coefficients.
    const Interval count(static_cast<double>(polynomial.coefficients().size()));
    const auto denominator = one - Interval(8) * count * u;
    if (!denominator.strictly_positive())
        return false;
    auto bound = (sum + Interval(512) * count * eta) / denominator;
    const auto rounded = square(one + u);
    bound = (bound * rounded * rounded + eta) * Interval(output_scale) * (one + u) + eta;
    return (Interval(std::numeric_limits<double>::max()) - bound).nonnegative();
}

bool reference_map_representable(const SurfaceBounds &b, double reference) {
    using proof::Interval;
    const auto requested =
        b.ratio_bounds.value_or(MoneynessBounds{std::exp(b.m_min), std::exp(b.m_max)});
    int exponent = 0;
    if (requested.min == requested.max && requested.min >= 1 &&
        std::frexp(requested.min, &exponent) == .5) {
        // For a fixed power-of-two ratio >=1 every admitted quote has that
        // exact ratio. The scale helper's quotient and reference product are
        // exact when finite; this includes subnormal unit-ratio reference K.
        const double mapped = requested.min * reference;
        return std::isfinite(mapped) && mapped > 0;
    }
    const auto ratio = MoneynessDomain(requested).enclosure();
    const Interval u(std::numeric_limits<double>::epsilon()),
        eta(std::numeric_limits<double>::denorm_min());
    const Interval one(1), scale(reference), limit(std::numeric_limits<double>::max());
    const auto upper = Interval(ratio.max) * scale * square(one + u) + eta;
    const auto lower = Interval(ratio.min) * scale * square(one - u) - eta;
    if (!(limit - upper).nonnegative() || !lower.strictly_positive())
        return false;
    return (limit - ((upper / scale) * (one + u) + eta)).nonnegative() &&
           ((lower / scale) * (one - u) - eta).strictly_positive();
}
template <class Inner>
std::expected<void, PriceTableError> segmented_numerics(const Inner &inner,
                                                        const SurfaceBounds &b) {
    const auto &refs = inner.split().k_refs();
    if (refs.empty() || refs.size() != inner.num_pieces() || !b.strike_bounds)
        return std::unexpected(PriceTableError{PriceTableErrorCode::InvalidConfig});
    for (std::size_t i = 0; i < refs.size(); ++i)
        if (!std::isfinite(refs[i]) || refs[i] <= 0 || (i && !(refs[i - 1] < refs[i])))
            return std::unexpected(PriceTableError{PriceTableErrorCode::InvalidConfig});
    if (b.strike_bounds->min < refs.front() || b.strike_bounds->max > refs.back())
        return std::unexpected(PriceTableError{PriceTableErrorCode::InvalidConfig});
    for (std::size_t i = 0; i < refs.size(); ++i) {
        // A reference endpoint does not evaluate the adjacent inactive member.
        const bool active = (i == 0 || b.strike_bounds->max > refs[i - 1]) &&
                            (i + 1 == refs.size() || b.strike_bounds->min < refs[i + 1]);
        if (!active)
            continue;
        const auto &member = inner.pieces()[i];
        if (member.split().K_ref() != refs[i])
            return std::unexpected(PriceTableError{PriceTableErrorCode::InvalidConfig});
        if (!reference_map_representable(b, refs[i]))
            return std::unexpected(
                PriceTableError{PriceTableErrorCode::CertificationIndeterminate});
        for (const auto &leaf : member.pieces()) {
            const auto &interp = leaf.interpolant();
            if constexpr (std::is_same_v<std::remove_cvref_t<decltype(interp)>,
                                         SharedBSplineInterp<4>>) {
                if (!interp.has_value())
                    return std::unexpected(PriceTableError{PriceTableErrorCode::InvalidConfig});
                if (!bspline_value_representable(interp.get(), refs[i]))
                    return std::unexpected(
                        PriceTableError{PriceTableErrorCode::CertificationIndeterminate});
            } else {
                static_assert(std::is_same_v<std::remove_cvref_t<decltype(interp)>,
                                             ChebyshevModalInterpolant<4>>);
                if (!modal_value_representable(interp.polynomial(), refs[i]))
                    return std::unexpected(
                        PriceTableError{PriceTableErrorCode::CertificationIndeterminate});
            }
        }
    }
    return {};
}
// Sufficient bounds for the actual double European parameter arithmetic.
// These are representability checks, not economic sigma/maturity caps.
bool european_inputs_representable(const SurfaceBounds &b, double q) {
    const MoneynessDomain moneyness(
        b.ratio_bounds.value_or(MoneynessBounds{std::exp(b.m_min), std::exp(b.m_max)}));
    const auto ratio = moneyness.enclosure();
    if (!ratio.valid() || !std::isfinite(q) || q < 0 || !std::isfinite(b.sigma_min) ||
        !std::isfinite(b.sigma_max) || b.sigma_min <= 0 || b.sigma_max < b.sigma_min ||
        !std::isfinite(b.tau_min) || !std::isfinite(b.tau_max) || b.tau_min < 0 || b.tau_max <= 0 ||
        b.tau_max < b.tau_min || !std::isfinite(b.rate_min) || !std::isfinite(b.rate_max) ||
        b.rate_max < b.rate_min)
        return false;
    const double variance_min = (.5 * b.sigma_min) * b.sigma_min;
    const double variance_max = (.5 * b.sigma_max) * b.sigma_max;
    const double minimum_tau =
        b.tau_min > 0 ? b.tau_min : std::numeric_limits<double>::denorm_min();
    const double denominator_min = b.sigma_min * std::sqrt(minimum_tau);
    const double denominator_max = b.sigma_max * std::sqrt(b.tau_max);
    if (!std::isfinite(variance_max) || !(denominator_min > 0) || !std::isfinite(denominator_max))
        return false;
    const double drift_min = (b.rate_min - q) + variance_min;
    const double drift_max = (b.rate_max - q) + variance_max;
    if (!std::isfinite(drift_min) || !std::isfinite(drift_max))
        return false;
    double product_min = std::numeric_limits<double>::infinity(), product_max = -product_min;
    for (double drift : {drift_min, drift_max})
        for (double tau : {b.tau_min, b.tau_max}) {
            const double product = drift * tau;
            if (!std::isfinite(product))
                return false;
            product_min = std::min(product_min, product);
            product_max = std::max(product_max, product);
        }
    return std::isfinite(std::log(ratio.min) + product_min) &&
           std::isfinite(std::log(ratio.max) + product_max) &&
           std::isfinite(std::exp(-std::min(0., b.rate_min) * b.tau_max));
}
std::expected<PublicationEvidence, PriceTableError> finish(const PhysicalCellProof &proof) {
    switch (proof.status) {
    case PriceProofStatus::Certified:
        return PublicationEvidence{proof.status, proof.nodes};
    case PriceProofStatus::NegativeWitness:
        return std::unexpected(
            PriceTableError{PriceTableErrorCode::NonMonotoneSurface, 0, proof.nodes});
    case PriceProofStatus::Indeterminate:
        return std::unexpected(
            PriceTableError{PriceTableErrorCode::CertificationIndeterminate, 0, proof.nodes});
    case PriceProofStatus::NotRun:
        return std::unexpected(PriceTableError{PriceTableErrorCode::UnsupportedRepresentation});
    }
    return std::unexpected(PriceTableError{PriceTableErrorCode::CertificationIndeterminate});
}
} // namespace

std::expected<PublicationEvidence, PriceTableError>
certify_payload(const BSplineLeaf &leaf, const SurfaceBounds &bounds, OptionType type, double q,
                const std::optional<FixedExpiryMetadata> &model, PriceTableProofBudget budget) {
    if (!metadata_valid(bounds, type, q, model, false) || !leaf.interpolant().has_value() ||
        leaf.eep().option_type() != type || leaf.eep().dividend_yield() != q)
        return std::unexpected(PriceTableError{PriceTableErrorCode::InvalidConfig});
    if (!european_inputs_representable(bounds, q) ||
        !bspline_value_representable(leaf.interpolant().get()))
        return std::unexpected(PriceTableError{PriceTableErrorCode::CertificationIndeterminate});
    return finish(prove_continuous_bspline(leaf.interpolant().get(), leaf.K_ref(), type, q, bounds,
                                           {budget.max_nodes, budget.max_depth}));
}
std::expected<PublicationEvidence, PriceTableError>
certify_payload(const BSpline3DLeaf &leaf, const SurfaceBounds &bounds, OptionType type, double q,
                const std::optional<FixedExpiryMetadata> &model, PriceTableProofBudget budget) {
    if (!metadata_valid(bounds, type, q, model, false) || q != 0 || bounds.rate_min <= 0 ||
        !leaf.interpolant().has_value() || leaf.eep().option_type() != type ||
        leaf.eep().dividend_yield() != q)
        return std::unexpected(PriceTableError{PriceTableErrorCode::InvalidConfig});
    if (!european_inputs_representable(bounds, q) || !dimensionless_inputs_representable(bounds) ||
        !bspline_value_representable(leaf.interpolant().get()))
        return std::unexpected(PriceTableError{PriceTableErrorCode::CertificationIndeterminate});
    return finish(prove_dimensionless_bspline(leaf.interpolant().get(), leaf.K_ref(), type, bounds,
                                              {budget.max_nodes, budget.max_depth}));
}
std::expected<PublicationEvidence, PriceTableError>
certify_payload(const BSplineMultiKRefInner &inner, const SurfaceBounds &bounds, OptionType type,
                double q, const std::optional<FixedExpiryMetadata> &model,
                PriceTableProofBudget budget) {
    if (!metadata_valid(bounds, type, q, model, true))
        return std::unexpected(PriceTableError{PriceTableErrorCode::InvalidConfig});
    auto numeric = segmented_numerics(inner, bounds);
    if (!numeric)
        return std::unexpected(numeric.error());
    return finish(
        prove_segmented_bspline(inner, type, q, bounds, {budget.max_nodes, budget.max_depth}));
}
std::expected<PublicationEvidence, PriceTableError>
certify_payload(const ChebyshevModalLeaf &leaf, const SurfaceBounds &bounds, OptionType type,
                double q, const std::optional<FixedExpiryMetadata> &model,
                PriceTableProofBudget budget) {
    if (!metadata_valid(bounds, type, q, model, false) || leaf.eep().option_type() != type ||
        leaf.eep().dividend_yield() != q)
        return std::unexpected(PriceTableError{PriceTableErrorCode::InvalidConfig});
    if (!european_inputs_representable(bounds, q) ||
        !modal_value_representable(leaf.interpolant().polynomial()))
        return std::unexpected(PriceTableError{PriceTableErrorCode::CertificationIndeterminate});
    return finish(prove_continuous_chebyshev(leaf.interpolant().polynomial(), leaf.K_ref(), type, q,
                                             bounds, {budget.max_nodes, budget.max_depth}));
}
std::expected<PublicationEvidence, PriceTableError>
certify_payload(const ChebyshevModal3DLeaf &leaf, const SurfaceBounds &bounds, OptionType type,
                double q, const std::optional<FixedExpiryMetadata> &model,
                PriceTableProofBudget budget) {
    if (!metadata_valid(bounds, type, q, model, false) || q != 0 || bounds.rate_min <= 0 ||
        leaf.eep().option_type() != type || leaf.eep().dividend_yield() != q)
        return std::unexpected(PriceTableError{PriceTableErrorCode::InvalidConfig});
    if (!european_inputs_representable(bounds, q) || !dimensionless_inputs_representable(bounds) ||
        !modal_value_representable(leaf.interpolant().polynomial()))
        return std::unexpected(PriceTableError{PriceTableErrorCode::CertificationIndeterminate});
    return finish(prove_dimensionless_chebyshev(leaf.interpolant().polynomial(), leaf.K_ref(), type,
                                                bounds, {budget.max_nodes, budget.max_depth}));
}
std::expected<PublicationEvidence, PriceTableError>
certify_payload(const ChebyshevModalMultiKRefInner &inner, const SurfaceBounds &bounds,
                OptionType type, double q, const std::optional<FixedExpiryMetadata> &model,
                PriceTableProofBudget budget) {
    if (!metadata_valid(bounds, type, q, model, true))
        return std::unexpected(PriceTableError{PriceTableErrorCode::InvalidConfig});
    auto numeric = segmented_numerics(inner, bounds);
    if (!numeric)
        return std::unexpected(numeric.error());
    return finish(
        prove_segmented_chebyshev(inner, type, q, bounds, {budget.max_nodes, budget.max_depth}));
}
} // namespace mango::detail::certification
