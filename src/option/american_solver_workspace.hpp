/**
 * @file american_solver_workspace.hpp
 * @brief Reusable workspace for American option solving (grid config + pre-allocated storage)
 *
 * Combines grid configuration (spatial + temporal parameters) with pre-allocated
 * workspace (grid buffer, spacing, SIMD-aligned storage) to enable efficient
 * batch solving of American options with shared grid but different coefficients.
 */

#pragma once

#include "src/pde/memory/pde_workspace.hpp"
#include "src/pde/operators/grid_spacing.hpp"
#include "src/pde/core/grid.hpp"
#include <expected>
#include "src/support/error_types.hpp"
#include <memory>
#include <span>
#include <stdexcept>
#include <string>

namespace mango {

/**
 * Workspace for American option solving with grid configuration.
 *
 * Extends PDEWorkspace with grid configuration (spatial + temporal parameters).
 * This eliminates redundant allocations when solving multiple options with:
 * - Same spatial grid structure (x_min, x_max, n_space)
 * - Same temporal discretization (n_time)
 * - Different PDE coefficients (σ, r, q, K)
 *
 * Example usage:
 * ```cpp
 * auto workspace_result = AmericanSolverWorkspace::create(-3.0, 3.0, 101, 1000);
 * if (!workspace_result) {
 *     std::cerr << "Workspace creation failed: " << workspace_result.error() << "\n";
 *     return;
 * }
 * auto workspace = workspace_result.value();
 *
 * for (auto [sigma, rate] : parameter_grid) {
 *     AmericanOptionParams params{...};
 *     auto solver_result = AmericanOptionSolver::create(params, workspace);
 *     if (solver_result) {
 *         auto result = solver_result.value().solve();
 *     }
 * }
 * ```
 *
 * Inherits from PDEWorkspace, providing direct access to:
 * - u_current(), u_next(), u_stage() - state arrays
 * - rhs(), lu(), psi_buffer() - scratch arrays
 * - All arrays are SIMD-aligned and zero-padded
 *
 * Memory savings (vs creating grid+spacing+workspace per solver):
 * - Grid buffer: ~800 bytes per reuse
 * - GridSpacing: ~800 bytes per reuse
 * - PDEWorkspace: ~10n doubles per reuse (SIMD-aligned)
 *
 * Thread safety: **NOT thread-safe for concurrent solving**.
 *
 * Why workspace cannot be shared across threads:
 * 1. PDEWorkspace contains mutable scratch arrays (u_current, u_next, rhs, etc.)
 * 2. These arrays are modified during solve() operations
 * 3. While PMR allocations are thread-safe, concurrent modifications are NOT
 * 4. Sharing one workspace across threads causes data races in scratch arrays
 *
 * Correct usage patterns:
 * ```cpp
 * // ✓ CORRECT: Sequential solving (reuse workspace via factory)
 * auto workspace_result = AmericanSolverWorkspace::create(-3.0, 3.0, 101, 1000);
 * if (!workspace_result) {
 *     // Handle error: workspace_result.error()
 *     return;
 * }
 * auto workspace = workspace_result.value();
 * for (auto params : option_list) {
 *     auto solver_result = AmericanOptionSolver::create(params, workspace);
 *     if (solver_result) {
 *         solver_result.value().solve();  // Safe: no concurrent access
 *     }
 * }
 *
 * // ✓ CORRECT: Use BatchAmericanOptionSolver (recommended for parallel)
 * // This handles per-thread workspace creation and error handling automatically
 * auto results = BatchAmericanOptionSolver::solve_batch(
 *     option_list, -3.0, 3.0, 101, 1000);
 *
 * // ✗ WRONG: Shared workspace across threads (DATA RACE!)
 * auto workspace = AmericanSolverWorkspace::create(-3.0, 3.0, 101, 1000).value();
 * #pragma omp parallel for
 * for (size_t i = 0; i < option_list.size(); ++i) {
 *     auto solver = AmericanOptionSolver::create(option_list[i], workspace).value();
 *     results[i] = solver.solve();  // UNSAFE! Multiple threads mutate same scratch arrays
 * }
 *
 * // ✗ WRONG: Direct construction bypasses validation
 * auto workspace = std::make_shared<AmericanSolverWorkspace>(...);  // WILL NOT COMPILE
 * ```
 *
 * Performance note: Creating per-thread workspaces is cheap (~10 KB, <1ms).
 * The memory savings from workspace reuse (~1.6 KB per solver) only matter
 * for sequential solving where the same workspace is reused hundreds of times.
 */
// Helper to initialize grid before PDEWorkspace base class
class GridHolder {
protected:
    GridBuffer<double> grid_buffer_;
    GridView<double> grid_view_;

    GridHolder(double x_min, double x_max, size_t n_space)
        : grid_buffer_(GridSpec<>::uniform(x_min, x_max, n_space).value().generate())
        , grid_view_(grid_buffer_.span())
    {}
};

class AmericanSolverWorkspace : private GridHolder, public PDEWorkspace {
private:
    // Pass-key idiom to allow make_shared while keeping constructor private
    struct PrivateTag {};

public:
    // Public constructor that requires pass-key (only factories can provide it)
    // Note: GridHolder base is initialized first, then PDEWorkspace can use grid_view_
    // Note: Not noexcept - can throw during grid generation or allocation
    AmericanSolverWorkspace(PrivateTag, double x_min, double x_max, size_t n_space, size_t n_time)
        : GridHolder(x_min, x_max, n_space)
        , PDEWorkspace(n_space, grid_view_.span())
        , x_min_(x_min)
        , x_max_(x_max)
        , n_space_(n_space)
        , n_time_(n_time)
        , grid_spacing_(std::make_shared<operators::GridSpacing<double>>(grid_view_))
    {
    }

    /**
     * Validate workspace parameters without allocation.
     *
     * Enables fail-fast in batch operations before parallel region.
     *
     * @param x_min Minimum log-moneyness
     * @param x_max Maximum log-moneyness
     * @param n_space Number of spatial grid points
     * @param n_time Number of time steps
     * @return Success or error message
     */
    static std::expected<void, std::string> validate_params(
        double x_min,
        double x_max,
        size_t n_space,
        size_t n_time)
    {
        if (x_min >= x_max) {
            return std::unexpected("x_min must be < x_max");
        }
        if (n_space < 3) {
            return std::unexpected("n_space must be >= 3");
        }
        if (n_time < 1) {
            return std::unexpected("n_time must be >= 1");
        }

        double dx = (x_max - x_min) / (n_space - 1);
        if (dx >= 0.5) {
            return std::unexpected(
                "Grid too coarse: dx = " + std::to_string(dx) +
                " >= 0.5 (Von Neumann stability violated)");
        }

        return {};
    }

    /**
     * Factory method with expected-based validation.
     *
     * Creates a shared_ptr to the workspace, ensuring proper lifetime management
     * for use with AmericanOptionSolver.
     *
     * @param x_min Minimum log-moneyness
     * @param x_max Maximum log-moneyness
     * @param n_space Number of spatial grid points
     * @param n_time Number of time steps
     * @return Expected containing shared workspace on success, error message on failure
     */
    static std::expected<std::shared_ptr<AmericanSolverWorkspace>, std::string> create(
        double x_min,
        double x_max,
        size_t n_space,
        size_t n_time)
    {
        // Validate parameters
        if (n_space < 10) {
            return std::unexpected("n_space must be >= 10");
        }
        if (n_time < 10) {
            return std::unexpected("n_time must be >= 10");
        }
        if (x_min >= x_max) {
            return std::unexpected("x_min must be < x_max");
        }

        // Catch allocation failures and grid generation errors
        try {
            return std::make_shared<AmericanSolverWorkspace>(PrivateTag{}, x_min, x_max, n_space, n_time);
        } catch (const std::bad_alloc&) {
            return std::unexpected("Failed to allocate workspace (out of memory)");
        } catch (const std::exception& e) {
            return std::unexpected(std::string("Failed to create workspace: ") + e.what());
        }
    }

    /**
     * Convenience factory with standard log-moneyness bounds.
     *
     * Creates workspace with standard bounds [-3.0, 3.0] in log-moneyness,
     * which corresponds to moneyness range [e^-3, e^3] ≈ [0.05, 20.09].
     * This covers most practical option scenarios:
     * - Deep OTM: m < 0.2 (x < -1.61)
     * - ATM: m ≈ 1.0 (x ≈ 0)
     * - Deep ITM: m > 5.0 (x > 1.61)
     *
     * For options requiring wider moneyness ranges, use the full factory
     * method with custom x_min and x_max parameters.
     *
     * Example usage:
     * ```cpp
     * auto workspace_result = AmericanSolverWorkspace::create_standard(101, 1000);
     * if (!workspace_result) {
     *     std::cerr << "Failed: " << workspace_result.error() << "\n";
     *     return;
     * }
     * AmericanOptionSolver solver(params, workspace_result.value());
     * ```
     *
     * @param n_space Number of spatial grid points
     * @param n_time Number of time steps
     * @return Expected containing shared workspace on success, error message on failure
     */
    static std::expected<std::shared_ptr<AmericanSolverWorkspace>, std::string> create_standard(
        size_t n_space,
        size_t n_time)
    {
        constexpr double x_min = -3.0;  // Standard log-moneyness range
        constexpr double x_max = 3.0;
        return create(x_min, x_max, n_space, n_time);
    }

    // Grid configuration accessors
    double x_min() const { return x_min_; }
    double x_max() const { return x_max_; }
    size_t n_space() const { return n_space_; }
    size_t n_time() const { return n_time_; }

    // Grid and spacing accessors
    std::span<const double> grid_span() const { return grid_view_.span(); }
    std::shared_ptr<operators::GridSpacing<double>> grid_spacing() const { return grid_spacing_; }

    // Note: PDEWorkspace methods inherited directly:
    // - u_current(), u_next(), u_stage()
    // - rhs(), lu(), psi_buffer()
    // - No indirection through workspace() method

private:
    // Note: grid_buffer_ and grid_view_ inherited from GridHolder base class

    // Grid parameters
    double x_min_;
    double x_max_;
    size_t n_space_;
    size_t n_time_;

    // GridSpacing (shared for reuse)
    std::shared_ptr<operators::GridSpacing<double>> grid_spacing_;
};

}  // namespace mango
