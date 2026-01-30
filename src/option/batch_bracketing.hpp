// SPDX-License-Identifier: MIT
/**
 * @file batch_bracketing.hpp
 * @brief Option bracketing/grouping for heterogeneous batch processing
 *
 * Groups heterogeneous options into homogeneous brackets for efficient
 * batch solving with shared grids. Each bracket contains similar options
 * that can share a common grid specification.
 */

#ifndef MANGO_BATCH_BRACKETING_HPP
#define MANGO_BATCH_BRACKETING_HPP

#include "src/option/option_spec.hpp"
#include "src/option/american_option.hpp"
#include "src/pde/core/grid.hpp"
#include "src/pde/core/time_domain.hpp"
#include <vector>
#include <span>
#include <expected>
#include <string>

namespace mango {

/**
 * Criteria for grouping options into brackets.
 *
 * Options are considered similar if their parameter differences
 * fall within specified tolerances.
 */
struct BracketingCriteria {
    /// Maximum maturity difference within bracket (years)
    double maturity_tolerance = 0.5;

    /// Maximum log-moneyness difference within bracket
    double moneyness_tolerance = 0.2;

    /// Maximum volatility difference within bracket
    double volatility_tolerance = 0.1;

    /// Maximum rate difference within bracket
    double rate_tolerance = 0.05;

    /// Maximum options per bracket (for memory/parallel efficiency)
    size_t max_bracket_size = 100;

    /// Minimum options to form a bracket (smaller groups use individual grids)
    size_t min_bracket_size = 3;
};

/**
 * A bracket of similar options sharing a common grid.
 *
 * Contains options grouped by similarity and the optimal grid
 * specification for solving them together.
 */
struct OptionBracket {
    /// Options in this bracket
    std::vector<PricingParams> options;

    /// Original indices mapping back to input order
    std::vector<size_t> original_indices;

    /// Optimal grid specification for this bracket
    GridSpec<double> grid_spec;

    /// Time domain for this bracket
    TimeDomain time_domain;

    /// Bracket statistics for diagnostics
    struct Stats {
        double min_maturity;
        double max_maturity;
        double min_moneyness;  // ln(S/K)
        double max_moneyness;
        double min_volatility;
        double max_volatility;
    } stats;
};

/**
 * Result of bracketing operation.
 */
struct BracketingResult {
    /// Brackets created from input options
    std::vector<OptionBracket> brackets;

    /// Total number of input options
    size_t total_options;

    /// Number of brackets created
    size_t num_brackets;

    /// Diagnostic: average bracket size
    double avg_bracket_size() const {
        return total_options / static_cast<double>(num_brackets);
    }
};

/**
 * Option bracketing/grouping utilities.
 *
 * Groups heterogeneous options into homogeneous brackets for
 * efficient batch processing with shared grids.
 */
class OptionBracketing {
public:
    /**
     * Compute distance between two options.
     *
     * Returns normalized Euclidean distance in parameter space
     * (maturity, moneyness, volatility, rate).
     *
     * @param a First option
     * @param b Second option
     * @param criteria Bracketing criteria (for normalization)
     * @return Normalized distance (0 = identical, larger = more different)
     */
    static double compute_distance(
        const PricingParams& a,
        const PricingParams& b,
        const BracketingCriteria& criteria);

    /**
     * Group options into brackets using greedy clustering.
     *
     * Algorithm:
     * 1. Sort options by maturity
     * 2. Greedily group nearby options until tolerance exceeded
     * 3. Start new bracket when distance threshold crossed
     *
     * @param options Input options (arbitrary order)
     * @param criteria Bracketing criteria
     * @return Bracketing result with option groups and grid specs
     */
    static std::expected<BracketingResult, std::string> group_options(
        std::span<const PricingParams> options,
        const BracketingCriteria& criteria = {});

    /**
     * Estimate optimal grid for a bracket of options.
     *
     * Computes grid bounds that cover all options in the bracket
     * with appropriate margins for boundary conditions.
     *
     * @param options Options in this bracket
     * @param accuracy Grid accuracy parameters
     * @return Grid specification and time steps
     */
    static std::expected<std::pair<GridSpec<double>, TimeDomain>, std::string>
    estimate_bracket_grid(
        std::span<const PricingParams> options,
        const GridAccuracyParams& accuracy = {});

private:
    /**
     * Compute bracket statistics for diagnostics.
     */
    static OptionBracket::Stats compute_bracket_stats(
        std::span<const PricingParams> options);

    /**
     * Validate that bracket options are compatible.
     *
     * Checks that all options have same option type (all calls or all puts).
     *
     * @return Error message if incompatible, nullopt if valid
     */
    static std::optional<std::string> validate_bracket_compatibility(
        std::span<const PricingParams> options);
};

} // namespace mango

#endif // MANGO_BATCH_BRACKETING_HPP
