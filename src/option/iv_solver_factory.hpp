// SPDX-License-Identifier: MIT
/**
 * @file iv_solver_factory.hpp
 * @brief Factory function that hides the two IV solver paths
 *
 * Provides make_iv_solver() which builds the appropriate price surface
 * (AmericanPriceSurface for continuous dividends, SegmentedMultiKRefSurface
 * for discrete dividends) and wraps it in a type-erased IVSolver.
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

/// Configuration for the IV solver factory
struct IVSolverConfig {
    OptionType option_type = OptionType::PUT;
    double spot = 100.0;
    double dividend_yield = 0.0;
    std::vector<std::pair<double, double>> discrete_dividends;  ///< (calendar_time, amount)
    std::vector<double> moneyness_grid;
    double maturity = 1.0;                     ///< Max maturity (T), used for segmented path
    std::vector<double> maturity_grid;         ///< For standard path (no dividends)
    std::vector<double> vol_grid;
    std::vector<double> rate_grid;
    MultiKRefConfig kref_config;               ///< For segmented path; defaults to auto
    IVSolverInterpolatedConfig solver_config;  ///< Newton config
};

/// Type-erased IV solver wrapping either path
class IVSolver {
public:
    /// Solve for implied volatility (single query)
    std::expected<IVSuccess, IVError> solve(const IVQuery& query) const;

    /// Solve for implied volatility (batch with OpenMP)
    BatchIVResult solve_batch(const std::vector<IVQuery>& queries) const;

    /// Constructor from standard solver
    explicit IVSolver(IVSolverInterpolated<AmericanPriceSurface> solver);

    /// Constructor from segmented solver
    explicit IVSolver(IVSolverInterpolated<SegmentedMultiKRefSurface> solver);

private:
    using SolverVariant = std::variant<
        IVSolverInterpolated<AmericanPriceSurface>,
        IVSolverInterpolated<SegmentedMultiKRefSurface>
    >;
    SolverVariant solver_;
};

/// Factory function: build price surface and IV solver from config
///
/// If discrete_dividends is empty, uses the standard AmericanPriceSurface path.
/// If discrete_dividends is non-empty, uses the SegmentedMultiKRefSurface path.
///
/// @param config Solver configuration
/// @return Type-erased IVSolver or ValidationError
std::expected<IVSolver, ValidationError> make_iv_solver(const IVSolverConfig& config);

}  // namespace mango
