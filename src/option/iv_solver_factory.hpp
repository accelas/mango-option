// SPDX-License-Identifier: MIT
/**
 * @file iv_solver_factory.hpp
 * @brief Factory function that hides the two IV solver paths
 *
 * Provides make_interpolated_iv_solver() which builds the appropriate price surface
 * (AmericanPriceSurface for continuous dividends, SegmentedMultiKRefSurface
 * for discrete dividends) and wraps it in a type-erased AnyIVSolver.
 */

#pragma once

#include <vector>
#include <variant>
#include <expected>
#include "src/option/iv_solver_interpolated.hpp"
#include "src/option/table/american_price_surface.hpp"
#include "src/option/table/segmented_multi_kref_surface.hpp"
#include "src/option/table/segmented_multi_kref_builder.hpp"
#include "src/option/option_spec.hpp"
#include "src/support/error_types.hpp"

namespace mango {

/// Standard path: continuous dividends only, maturity grid for interpolation
struct StandardIVPath {
    std::vector<double> maturity_grid;
};

/// Segmented path: discrete dividends with multi-K_ref surface
struct SegmentedIVPath {
    double maturity = 1.0;
    std::vector<Dividend> discrete_dividends;
    MultiKRefConfig kref_config;  ///< defaults to auto
};

/// Configuration for the IV solver factory
struct IVSolverFactoryConfig {
    OptionType option_type = OptionType::PUT;
    double spot = 100.0;
    double dividend_yield = 0.0;
    std::vector<double> moneyness_grid;
    std::vector<double> vol_grid;
    std::vector<double> rate_grid;
    IVSolverInterpolatedConfig solver_config;  ///< Newton config
    std::variant<StandardIVPath, SegmentedIVPath> path;
};

/// Type-erased IV solver wrapping either path
class AnyIVSolver {
public:
    /// Solve for implied volatility (single query)
    std::expected<IVSuccess, IVError> solve(const IVQuery& query) const;

    /// Solve for implied volatility (batch with OpenMP)
    BatchIVResult solve_batch(const std::vector<IVQuery>& queries) const;

    /// Constructor from standard solver
    explicit AnyIVSolver(IVSolverInterpolated<AmericanPriceSurface> solver);

    /// Constructor from segmented solver
    explicit AnyIVSolver(IVSolverInterpolated<SegmentedMultiKRefSurface> solver);

private:
    using SolverVariant = std::variant<
        IVSolverInterpolated<AmericanPriceSurface>,
        IVSolverInterpolated<SegmentedMultiKRefSurface>
    >;
    SolverVariant solver_;
};

/// Factory function: build price surface and IV solver from config
///
/// If path holds StandardIVPath, uses the AmericanPriceSurface path.
/// If path holds SegmentedIVPath, uses the SegmentedMultiKRefSurface path.
///
/// @param config Solver configuration
/// @return Type-erased AnyIVSolver or ValidationError
std::expected<AnyIVSolver, ValidationError> make_interpolated_iv_solver(const IVSolverFactoryConfig& config);

}  // namespace mango
