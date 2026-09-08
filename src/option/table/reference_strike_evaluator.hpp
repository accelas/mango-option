// SPDX-License-Identifier: MIT
#pragma once

#include "mango/option/table/adaptive_refinement.hpp"
#include "mango/option/table/reference_selection.hpp"
#include <memory>

namespace mango {

/// Cached direct-PDE evidence for reference approximation over one fixed
/// physical query population. IV statistics are price/vega error proxies,
/// never actual inverted-IV errors. This does not enforce whole-table publication.
class ReferenceStrikeEvaluator {
public:
    [[nodiscard]] static std::expected<ReferenceStrikeEvaluator, PriceTableError>
    create(const SegmentedAdaptiveConfig& config, const SurfaceBounds& requested,
           std::vector<std::pair<double, double>> admitted_times,
           double price_target = 0.01, double iv_target = 2e-5);

    ReferenceStrikeEvaluator(ReferenceStrikeEvaluator&&) noexcept;
    ReferenceStrikeEvaluator& operator=(ReferenceStrikeEvaluator&&) noexcept;
    ~ReferenceStrikeEvaluator();

    /// Omit fitted to measure ideal reference blending before expensive fits.
    /// Supplying it separates ideal, fitting, and actual composed errors.
    [[nodiscard]] std::expected<ReferenceCandidateMetrics, PriceTableError>
    evaluate(std::span<const double> refs, const SurfaceHandle* fitted = nullptr);

private:
    struct Impl;
    explicit ReferenceStrikeEvaluator(std::unique_ptr<Impl> impl);
    std::unique_ptr<Impl> impl_;
};

}  // namespace mango
