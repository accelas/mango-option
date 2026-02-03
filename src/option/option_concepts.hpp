// SPDX-License-Identifier: MIT
/**
 * @file option_concepts.hpp
 * @brief C++20 concepts defining the common interface for option results and solvers
 *
 * These concepts enable generic programming over different option pricing models
 * (e.g., American, European) by specifying the required interface for results
 * and solvers.
 */

#pragma once

#include "mango/option/option_spec.hpp"
#include <concepts>
#include <expected>

namespace mango {

/**
 * @brief Concept for option pricing results
 *
 * Any type satisfying OptionResult must provide:
 * - value(): option value at current spot
 * - value_at(spot): option value at arbitrary spot price
 * - delta(), gamma(), theta(): Greeks
 * - spot(), strike(), maturity(), volatility(): pricing parameters
 * - option_type(): CALL or PUT
 */
template <typename R>
concept OptionResult = requires(const R& r, double spot_price) {
    { r.value() } -> std::convertible_to<double>;
    { r.value_at(spot_price) } -> std::convertible_to<double>;
    { r.delta() } -> std::convertible_to<double>;
    { r.gamma() } -> std::convertible_to<double>;
    { r.theta() } -> std::convertible_to<double>;
    { r.spot() } -> std::convertible_to<double>;
    { r.strike() } -> std::convertible_to<double>;
    { r.maturity() } -> std::convertible_to<double>;
    { r.volatility() } -> std::convertible_to<double>;
    { r.option_type() } -> std::same_as<OptionType>;
};

/**
 * @brief Refined concept for results that also provide vega
 *
 * European option results typically provide vega (sensitivity to volatility)
 * via closed-form formulas. American option results from FDM solvers do not.
 */
template <typename R>
concept OptionResultWithVega = OptionResult<R> && requires(const R& r) {
    { r.vega() } -> std::convertible_to<double>;
};

/**
 * @brief Concept for option solvers
 *
 * A solver must provide a solve() method returning std::expected<R, E>
 * where R satisfies OptionResult.
 */
template <typename S>
concept OptionSolver = requires(S& solver) {
    requires OptionResult<typename decltype(solver.solve())::value_type>;
};

}  // namespace mango
