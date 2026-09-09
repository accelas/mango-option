// SPDX-License-Identifier: MIT
#pragma once
#include "mango/option/table/bspline/bspline_types.hpp"
#include "mango/option/table/certification/certificate_status.hpp"
#include "mango/option/table/chebyshev/modal_types.hpp"
#include "mango/option/table/fixed_expiry.hpp"
#include "mango/option/table/surface_bounds.hpp"
#include <expected>
namespace mango {
/// Finite proof-work allowance; zero cannot bypass publication certification.
struct PriceTableProofBudget {
    std::size_t max_nodes = 4096;
    std::size_t max_depth = 24;
};
namespace detail::certification {
/// Diagnostic evidence. Only PriceTable's own frozen-payload creation path
/// can attach this to publication; callers cannot supply an admission token.
struct PublicationEvidence {
    PriceProofStatus status = PriceProofStatus::NotRun;
    std::size_t work = 0;
};
// Deliberately closed overload set: a callback claiming a boolean or exposing
// a polynomial-shaped accessor cannot certify a different query function.
std::expected<PublicationEvidence, PriceTableError>
certify_payload(const BSplineLeaf &, const SurfaceBounds &, OptionType, double,
                const std::optional<FixedExpiryMetadata> &, PriceTableProofBudget);
std::expected<PublicationEvidence, PriceTableError>
certify_payload(const BSpline3DLeaf &, const SurfaceBounds &, OptionType, double,
                const std::optional<FixedExpiryMetadata> &, PriceTableProofBudget);
std::expected<PublicationEvidence, PriceTableError>
certify_payload(const BSplineMultiKRefInner &, const SurfaceBounds &, OptionType, double,
                const std::optional<FixedExpiryMetadata> &, PriceTableProofBudget);
std::expected<PublicationEvidence, PriceTableError>
certify_payload(const ChebyshevModalLeaf &, const SurfaceBounds &, OptionType, double,
                const std::optional<FixedExpiryMetadata> &, PriceTableProofBudget);
std::expected<PublicationEvidence, PriceTableError>
certify_payload(const ChebyshevModal3DLeaf &, const SurfaceBounds &, OptionType, double,
                const std::optional<FixedExpiryMetadata> &, PriceTableProofBudget);
std::expected<PublicationEvidence, PriceTableError>
certify_payload(const ChebyshevModalMultiKRefInner &, const SurfaceBounds &, OptionType, double,
                const std::optional<FixedExpiryMetadata> &, PriceTableProofBudget);
} // namespace detail::certification
} // namespace mango
