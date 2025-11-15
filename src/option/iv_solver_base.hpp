/**
 * @file iv_solver_base.hpp
 * @brief Base class for IV solvers using C++23 deducing this
 */

#pragma once

#include "src/option/option_spec.hpp"
#include "src/option/iv_types.hpp"
#include <expected>
#include "src/support/error_types.hpp"
#include <span>
#include <format>
#include <utility>

namespace mango {

/**
 * @brief Base class for implied volatility solvers
 *
 * Uses C++23 deducing this for zero-overhead static polymorphism.
 * Derived classes must implement:
 *   - IVResult solve_impl(const IVQuery& query)
 *   - void solve_batch_impl(span<const IVQuery>, span<IVResult>) [optional]
 *
 * Thread safety:
 *   - solve(): thread-safety depends on derived class implementation
 *   - solve_batch(): derived class controls parallelization strategy
 */
class IVSolverBase {
public:
    /**
     * @brief Solve for implied volatility (single query)
     *
     * Forwards to derived class's solve_impl() via deducing this.
     * Propagates noexcept specification from derived implementation.
     */
    template <typename Self>
    IVResult solve(this Self&& self, const IVQuery& query)
        noexcept(noexcept(std::forward<Self>(self).solve_impl(query)))
    {
        return std::forward<Self>(self).solve_impl(query);
    }

    /**
     * @brief Solve for implied volatility (batch)
     *
     * If derived class provides solve_batch_impl(), uses it (potentially with
     * OpenMP parallelization). Otherwise, falls back to sequential solve_impl().
     *
     * @param queries Input queries (must match results.size())
     * @param results Output buffer (must match queries.size())
     * @return void on success, error message on size mismatch
     */
    template <typename Self>
    std::expected<void, std::string> solve_batch(this Self&& self,
                                             std::span<const IVQuery> queries,
                                             std::span<IVResult> results)
    {
        // Runtime validation
        if (queries.size() != results.size()) {
            return std::unexpected(std::format(
                "Size mismatch: {} queries but {} result slots",
                queries.size(), results.size()));
        }

        // Use derived class's batch implementation if available
        using ForwardedSelf = decltype(self);
        if constexpr (requires {
            { std::forward<ForwardedSelf>(self).solve_batch_impl(queries, results) }
                -> std::same_as<void>;
        }) {
            std::forward<ForwardedSelf>(self).solve_batch_impl(queries, results);
        } else {
            // Fallback: sequential processing
            for (size_t i = 0; i < queries.size(); ++i) {
                results[i] = std::forward<ForwardedSelf>(self).solve_impl(queries[i]);
            }
        }

        return {};
    }
};

} // namespace mango
