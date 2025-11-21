/**
 * @file american_option.hpp
 * @brief American option pricing solver using finite difference method
 */

#ifndef MANGO_AMERICAN_OPTION_HPP
#define MANGO_AMERICAN_OPTION_HPP

#include "src/pde/core/pde_solver.hpp"
#include "src/pde/operators/black_scholes_pde.hpp"
#include "src/pde/operators/centered_difference_facade.hpp"
#include <expected>
#include "src/support/error_types.hpp"
#include "src/support/parallel.hpp"
#include "src/option/american_option_result.hpp"
#include "src/option/option_spec.hpp"  // For OptionType enum
#include "src/pde/core/pde_workspace.hpp"
#include "src/option/american_solver_workspace.hpp"  // For deprecated constructor
#include <vector>
#include <memory>
#include <stdexcept>
#include <cmath>
#include <functional>
#include <optional>

namespace mango {

/**
 * @brief Backward compatibility alias for PricingParams
 * @deprecated Use PricingParams from option_spec.hpp instead
 */
using AmericanOptionParams = PricingParams;

/**
 * American option pricing solver using finite difference method.
 *
 * Solves the Black-Scholes PDE with obstacle constraints in log-moneyness
 * coordinates using TR-BDF2 time stepping and projection method for
 * early exercise boundary.
 */
class AmericanOptionSolver {
public:
    /**
     * Direct PDEWorkspace constructor.
     *
     * This constructor takes PDEWorkspace directly, enabling flexible
     * memory management. The solver creates Grid internally and returns
     * the AmericanOptionResult wrapper.
     *
     * @param params Option pricing parameters
     * @param workspace PDEWorkspace with pre-allocated buffers
     * @param snapshot_times Optional times to record solution snapshots
     */
    AmericanOptionSolver(const PricingParams& params,
                        PDEWorkspace workspace,
                        std::optional<std::span<const double>> snapshot_times = std::nullopt);

    /**
     * DEPRECATED: Constructor with AmericanSolverWorkspace.
     *
     * This constructor is retained for backward compatibility with batch solver code.
     * New code should use the PDEWorkspace constructor instead.
     *
     * @param params Option pricing parameters
     * @param workspace Shared workspace with grid configuration
     * @deprecated Use PDEWorkspace constructor instead
     */
    [[deprecated("Use PDEWorkspace constructor instead")]]
    AmericanOptionSolver(const AmericanOptionParams& params,
                        std::shared_ptr<AmericanSolverWorkspace> workspace);

    /**
     * DEPRECATED: Factory method with expected-based validation.
     *
     * This method is retained for backward compatibility with batch solver code.
     * New code should use the PDEWorkspace constructor directly.
     *
     * @param params Option pricing parameters
     * @param workspace Shared workspace with grid configuration
     * @return Expected containing solver on success, error message on failure
     * @deprecated Use PDEWorkspace constructor instead
     */
    [[deprecated("Use PDEWorkspace constructor instead")]]
    static std::expected<AmericanOptionSolver, std::string> create(
        const AmericanOptionParams& params,
        std::shared_ptr<AmericanSolverWorkspace> workspace);

    /**
     * Solve for option value.
     *
     * Returns AmericanOptionResult wrapper containing Grid and PricingParams.
     * If snapshot_times were provided at construction, the Grid will contain
     * recorded solution snapshots.
     *
     * @return Result wrapper with value(), Greeks, and snapshot access
     */
    std::expected<AmericanOptionResult, SolverError> solve();

private:
    // Parameters
    PricingParams params_;

    // PDEWorkspace (owns spans to external buffer)
    std::optional<PDEWorkspace> workspace_;

    // Snapshot times for Grid creation
    std::vector<double> snapshot_times_;

    // DEPRECATED: Legacy workspace for backward compatibility
    std::shared_ptr<AmericanSolverWorkspace> legacy_workspace_;

    // Helper to determine which API was used
    bool using_new_api() const { return workspace_.has_value(); }
};

}  // namespace mango

#endif  // MANGO_AMERICAN_OPTION_HPP
