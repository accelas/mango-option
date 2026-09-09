// SPDX-License-Identifier: MIT
#pragma once

#include "mango/option/table/accuracy/population_types.hpp"
#include "mango/option/table/surface_bounds.hpp"
#include "mango/option/table/fixed_expiry.hpp"
#include "mango/option/table/refinement_work.hpp"
#include <expected>
#include <span>
#include <vector>

namespace mango::detail::accuracy {

/// RequiredPrice rows must be served; Admission rows may correctly refuse or
/// serve a qualified price. Refusal rows (including expiry) must refuse.
enum class PopulationExpectation { RequiredPrice, Admission, Refusal };

struct PopulationRow {
    PricingParams query;
    uint32_t strata=0;
    PopulationExpectation expectation=PopulationExpectation::RequiredPrice;
};

/// Resolved before fitting. The numerical constraint, not a fitted candidate's
/// eventual node count, determines the independent moneyness strip.
struct PopulationRequest {
    SurfaceBounds domain{};
    OptionType option_type=OptionType::PUT;
    double dividend_yield=0.;
    double spot=100.;
    std::optional<FixedExpiryMetadata> fixed_expiry;
    size_t max_moneyness_nodes=160;
    AccuracyEvaluationLimits limits;
    /// Complete bounded reference-plan node union, known before candidates.
    std::vector<double> possible_reference_strikes;
};

enum class PopulationFailureReason { InvalidInput, RowLimit, UnrepresentableDomain };
struct PopulationFailure {
    PopulationFailureReason reason;
    size_t required_rows=0;
    bool required_rows_exact=true;
    AccuracyEvaluationLimits limits;
    PdeWork work;  ///< Population planning performs no numerical solves.
};

class AccuracyPopulation {
public:
    [[nodiscard]] uint32_t profile_version() const noexcept { return counts_.profile_version; }
    [[nodiscard]] std::span<const PopulationRow> rows() const noexcept { return rows_; }
    [[nodiscard]] const AccuracyEvaluationLimits& limits() const noexcept { return limits_; }
    [[nodiscard]] const AccuracyPopulationCounts& counts() const noexcept { return counts_; }
private:
    friend std::expected<AccuracyPopulation,PopulationFailure>
        make_accuracy_population(const PopulationRequest&);
    AccuracyPopulation(std::vector<PopulationRow> rows, AccuracyEvaluationLimits limits,
                       AccuracyPopulationCounts counts)
        : rows_(std::move(rows)), limits_(limits), counts_(counts) {}
    const std::vector<PopulationRow> rows_;
    const AccuracyEvaluationLimits limits_;
    const AccuracyPopulationCounts counts_;
};

/// factory-accuracy-v1: global corners, per-regime interiors, event admission,
/// independent physical K/S movement and a declared-constraint m strip.
/// No fitted observations, training samples, or market-clock state are inputs.
[[nodiscard]] std::expected<AccuracyPopulation,PopulationFailure>
make_accuracy_population(const PopulationRequest& request);

} // namespace mango::detail::accuracy
