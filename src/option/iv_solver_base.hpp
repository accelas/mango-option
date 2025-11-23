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
 *
 * NOTE: This class uses the deprecated IVResult type for backward compatibility.
 * New solvers should implement solve_impl() returning std::expected<IVSuccess, IVError>
 * and provide a separate solve_legacy() wrapper for IVResult compatibility.
 *
 * Derived classes must implement:
 *   - IVResult solve_impl(const IVQuery& query)  [legacy interface]
 *   - void solve_batch_impl(span<const IVQuery>, span<IVResult>) [optional, legacy]
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
     *
     * NOTE: Returns deprecated IVResult for backward compatibility.
     * Use solver.solve_impl() directly for std::expected-based error handling.
     */
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
    template <typename Self>
    IVResult solve(this Self&& self, const IVQuery& query)
        noexcept(noexcept(std::forward<Self>(self).solve_impl(query)))
    {
        return std::forward<Self>(self).solve_impl(query);
    }
#pragma GCC diagnostic pop

    /**
     * @brief Solve for implied volatility (batch)
     *
     * If derived class provides solve_batch_impl(), uses it (potentially with
     * OpenMP parallelization). Otherwise, falls back to sequential solve_impl().
     *
     * NOTE: Uses deprecated IVResult for backward compatibility.
     * For new code, use solve_batch_impl() which returns BatchIVResult with
     * std::expected-based error handling.
     *
     * @param queries Input queries (must match results.size())
     * @param results Output buffer (must match queries.size())
     * @return void on success, error message on size mismatch
     */
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
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
#pragma GCC diagnostic pop
};

} // namespace mango
