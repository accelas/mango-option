/**
 * @file batch_bracketing.cpp
 * @brief Implementation of option bracketing algorithm
 */

#include "src/option/batch_bracketing.hpp"
#include <algorithm>
#include <cmath>
#include <limits>
#include <ranges>

namespace mango {

double OptionBracketing::compute_distance(
    const PricingParams& a,
    const PricingParams& b,
    const BracketingCriteria& criteria)
{
    // Validate tolerances (prevent division by zero/negative)
    if (criteria.maturity_tolerance <= 0.0 ||
        criteria.moneyness_tolerance <= 0.0 ||
        criteria.volatility_tolerance <= 0.0 ||
        criteria.rate_tolerance <= 0.0) {
        return std::numeric_limits<double>::infinity();  // Invalid criteria
    }

    // Validate option parameters (prevent log of non-positive)
    if (a.spot <= 0.0 || a.strike <= 0.0 || b.spot <= 0.0 || b.strike <= 0.0) {
        return std::numeric_limits<double>::infinity();  // Invalid options
    }

    // Normalized differences in each dimension
    double d_maturity = std::abs(a.maturity - b.maturity) / criteria.maturity_tolerance;

    double log_moneyness_a = std::log(a.spot / a.strike);
    double log_moneyness_b = std::log(b.spot / b.strike);
    double d_moneyness = std::abs(log_moneyness_a - log_moneyness_b) / criteria.moneyness_tolerance;

    double d_vol = std::abs(a.volatility - b.volatility) / criteria.volatility_tolerance;
    double d_rate = std::abs(a.rate - b.rate) / criteria.rate_tolerance;

    // Euclidean distance in normalized space
    return std::sqrt(
        d_maturity * d_maturity +
        d_moneyness * d_moneyness +
        d_vol * d_vol +
        d_rate * d_rate
    );
}

OptionBracket::Stats OptionBracketing::compute_bracket_stats(
    std::span<const PricingParams> options)
{
    if (options.empty()) {
        return {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    }

    // Validate before computing logs
    for (const auto& opt : options) {
        if (opt.spot <= 0.0 || opt.strike <= 0.0) {
            // Return invalid stats if any option has bad parameters
            return {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
        }
    }

    auto maturities = options | std::views::transform([](const auto& o) { return o.maturity; });
    auto moneynesses = options | std::views::transform([](const auto& o) {
        return std::log(o.spot / o.strike);
    });
    auto volatilities = options | std::views::transform([](const auto& o) { return o.volatility; });

    OptionBracket::Stats stats;
    stats.min_maturity = *std::ranges::min_element(maturities);
    stats.max_maturity = *std::ranges::max_element(maturities);
    stats.min_moneyness = *std::ranges::min_element(moneynesses);
    stats.max_moneyness = *std::ranges::max_element(moneynesses);
    stats.min_volatility = *std::ranges::min_element(volatilities);
    stats.max_volatility = *std::ranges::max_element(volatilities);

    return stats;
}

std::optional<std::string> OptionBracketing::validate_bracket_compatibility(
    std::span<const PricingParams> options)
{
    if (options.empty()) {
        return "Bracket cannot be empty";
    }

    // Check that all options have same type
    OptionType first_type = options[0].type;
    for (size_t i = 1; i < options.size(); ++i) {
        if (options[i].type != first_type) {
            return "Bracket contains mixed option types (calls and puts)";
        }
    }

    // Check for invalid parameters
    for (const auto& opt : options) {
        if (opt.spot <= 0.0) {
            return "Invalid spot price (<= 0)";
        }
        if (opt.strike <= 0.0) {
            return "Invalid strike price (<= 0)";
        }
        if (opt.maturity <= 0.0) {
            return "Invalid maturity (<= 0)";
        }
        if (opt.volatility <= 0.0) {
            return "Invalid volatility (<= 0)";
        }
    }

    return std::nullopt;
}

std::expected<BracketingResult, std::string> OptionBracketing::group_options(
    std::span<const PricingParams> options,
    const BracketingCriteria& criteria)
{
    if (options.empty()) {
        return std::unexpected("Cannot bracket empty option list");
    }

    // Copy options to vector for sorting
    std::vector<std::pair<PricingParams, size_t>> indexed_options;
    indexed_options.reserve(options.size());
    for (size_t i = 0; i < options.size(); ++i) {
        indexed_options.emplace_back(options[i], i);
    }

    // Sort by maturity (primary grouping dimension)
    std::ranges::sort(indexed_options, [](const auto& a, const auto& b) {
        return a.first.maturity < b.first.maturity;
    });

    BracketingResult result;
    result.total_options = options.size();

    // Greedy clustering algorithm
    std::vector<PricingParams> current_bracket;
    std::vector<size_t> current_indices;

    for (size_t i = 0; i < indexed_options.size(); ++i) {
        const auto& [opt, original_idx] = indexed_options[i];

        // Start new bracket if empty
        if (current_bracket.empty()) {
            current_bracket.push_back(opt);
            current_indices.push_back(original_idx);
            continue;
        }

        // Check if option fits in current bracket
        bool fits_in_bracket = true;

        // Check distance to all options in current bracket
        for (const auto& bracket_opt : current_bracket) {
            double dist = compute_distance(opt, bracket_opt, criteria);
            if (dist > 1.0) {  // Distance > 1.0 means exceeds tolerance
                fits_in_bracket = false;
                break;
            }
        }

        // Check bracket size limit
        if (current_bracket.size() >= criteria.max_bracket_size) {
            fits_in_bracket = false;
        }

        if (fits_in_bracket) {
            // Add to current bracket
            current_bracket.push_back(opt);
            current_indices.push_back(original_idx);
        } else {
            // Finalize current bracket and start new one
            if (current_bracket.size() >= criteria.min_bracket_size) {
                // Estimate grid for bracket
                auto grid_result = estimate_bracket_grid(current_bracket);
                if (!grid_result.has_value()) {
                    return std::unexpected("Failed to estimate grid: " + grid_result.error());
                }

                // Validate compatibility
                auto validation_error = validate_bracket_compatibility(current_bracket);
                if (validation_error.has_value()) {
                    return std::unexpected("Bracket validation failed: " + validation_error.value());
                }

                auto [grid_spec, n_time] = grid_result.value();
                auto stats = compute_bracket_stats(current_bracket);

                result.brackets.push_back(OptionBracket{
                    .options = std::move(current_bracket),
                    .original_indices = std::move(current_indices),
                    .grid_spec = grid_spec,
                    .time_domain = n_time,
                    .stats = stats
                });
            } else {
                // Bracket too small, treat as individual options (add to results as size-1 brackets)
                for (size_t j = 0; j < current_bracket.size(); ++j) {
                    auto single_grid = estimate_bracket_grid(std::span{&current_bracket[j], 1});
                    if (!single_grid.has_value()) {
                        return std::unexpected("Failed to estimate grid: " + single_grid.error());
                    }

                    auto [grid_spec, n_time] = single_grid.value();
                    auto stats = compute_bracket_stats(std::span{&current_bracket[j], 1});

                    result.brackets.push_back(OptionBracket{
                        .options = {current_bracket[j]},
                        .original_indices = {current_indices[j]},
                        .grid_spec = grid_spec,
                        .time_domain = n_time,
                        .stats = stats
                    });
                }
            }

            // Start new bracket with current option
            current_bracket.clear();
            current_indices.clear();
            current_bracket.push_back(opt);
            current_indices.push_back(original_idx);
        }
    }

    // Finalize last bracket
    if (!current_bracket.empty()) {
        if (current_bracket.size() >= criteria.min_bracket_size) {
            auto grid_result = estimate_bracket_grid(current_bracket);
            if (!grid_result.has_value()) {
                return std::unexpected("Failed to estimate grid: " + grid_result.error());
            }

            auto validation_error = validate_bracket_compatibility(current_bracket);
            if (validation_error.has_value()) {
                return std::unexpected("Bracket validation failed: " + validation_error.value());
            }

            auto [grid_spec, n_time] = grid_result.value();
            auto stats = compute_bracket_stats(current_bracket);

            result.brackets.push_back(OptionBracket{
                .options = std::move(current_bracket),
                .original_indices = std::move(current_indices),
                .grid_spec = grid_spec,
                .time_domain = n_time,
                .stats = stats
            });
        } else {
            // Last bracket too small, add as individual options
            for (size_t j = 0; j < current_bracket.size(); ++j) {
                auto single_grid = estimate_bracket_grid(std::span{&current_bracket[j], 1});
                if (!single_grid.has_value()) {
                    return std::unexpected("Failed to estimate grid: " + single_grid.error());
                }

                auto [grid_spec, n_time] = single_grid.value();
                auto stats = compute_bracket_stats(std::span{&current_bracket[j], 1});

                result.brackets.push_back(OptionBracket{
                    .options = {current_bracket[j]},
                    .original_indices = {current_indices[j]},
                    .grid_spec = grid_spec,
                    .time_domain = n_time,
                    .stats = stats
                });
            }
        }
    }

    result.num_brackets = result.brackets.size();
    return result;
}

std::expected<std::pair<GridSpec<double>, TimeDomain>, std::string>
OptionBracketing::estimate_bracket_grid(
    std::span<const PricingParams> options,
    const GridAccuracyParams& accuracy)
{
    if (options.empty()) {
        return std::unexpected("Cannot estimate grid for empty bracket");
    }

    // Validate all options before computing ranges
    for (const auto& opt : options) {
        if (opt.spot <= 0.0 || opt.strike <= 0.0 || opt.maturity <= 0.0 || opt.volatility <= 0.0) {
            return std::unexpected("Invalid option parameters (non-positive values)");
        }
    }

    // Find parameter ranges across all options
    double max_maturity = 0.0;
    double max_volatility = 0.0;
    double min_log_moneyness = std::numeric_limits<double>::max();
    double max_log_moneyness = std::numeric_limits<double>::lowest();

    for (const auto& opt : options) {
        max_maturity = std::max(max_maturity, opt.maturity);
        max_volatility = std::max(max_volatility, opt.volatility);

        double log_m = std::log(opt.spot / opt.strike);
        min_log_moneyness = std::min(min_log_moneyness, log_m);
        max_log_moneyness = std::max(max_log_moneyness, log_m);
    }

    // Add margins for boundary conditions (±3σ√T from extreme moneyness)
    double margin = 3.0 * max_volatility * std::sqrt(max_maturity);
    double x_min = min_log_moneyness - margin;
    double x_max = max_log_moneyness + margin;

    // Ensure minimum domain width
    double min_width = 3.0;  // At least 3 log-units
    if (x_max - x_min < min_width) {
        double center = (x_min + x_max) / 2.0;
        x_min = center - min_width / 2.0;
        x_max = center + min_width / 2.0;
    }

    // Estimate grid using bracket parameters
    // Create a representative option with bracket's max maturity and volatility
    PricingParams bracket_rep = options[0];
    bracket_rep.maturity = max_maturity;
    bracket_rep.volatility = max_volatility;
    bracket_rep.spot = std::exp((min_log_moneyness + max_log_moneyness) / 2.0);
    bracket_rep.strike = 1.0;  // ATM representative

    // Get grid estimation for bracket representative
    auto [grid_spec, time_domain] = estimate_grid_for_option(bracket_rep, accuracy);
    size_t n_time = time_domain.n_steps();

    // Recompute Nx to maintain dx for the widened domain
    double domain_width = x_max - x_min;
    size_t n_space = grid_spec.n_points();

    // Adjust n_space to maintain similar spacing as estimated
    double estimated_width = grid_spec.x_max() - grid_spec.x_min();
    if (domain_width > estimated_width) {
        // Need more points to maintain spacing
        double ratio = domain_width / estimated_width;
        n_space = static_cast<size_t>(std::ceil(n_space * ratio));

        // Ensure reasonable bounds
        n_space = std::max(n_space, size_t{51});    // Minimum 51 points
        n_space = std::min(n_space, size_t{1001});  // Maximum 1001 points
    }

    // Create grid with adjusted bounds and spacing
    grid_spec = GridSpec<double>::uniform(x_min, x_max, n_space).value();

    // Create TimeDomain from n_time
    TimeDomain result_time_domain = TimeDomain::from_n_steps(0.0, max_maturity, n_time);
    return std::make_pair(grid_spec, result_time_domain);
}

} // namespace mango
