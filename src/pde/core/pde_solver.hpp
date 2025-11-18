#pragma once

#include "src/pde/core/grid.hpp"
#include "src/pde/core/pde_workspace.hpp"
#include "src/support/cpu/feature_detection.hpp"
#include "src/pde/operators/centered_difference_facade.hpp"
#include "src/pde/core/boundary_conditions.hpp"
#include "src/pde/core/time_domain.hpp"
#include "src/pde/core/trbdf2_config.hpp"
#include "src/math/thomas_solver.hpp"
#include "src/pde/core/jacobian_view.hpp"
#include <expected>
#include "src/support/error_types.hpp"
#include <memory>
#include <span>
#include <vector>
#include <functional>
#include <optional>
#include <cmath>
#include <algorithm>
#include <limits>
#include <iostream>

namespace mango {

/// Concept to detect spatial operators with analytical Jacobian capability
template<typename SpatialOp>
concept HasAnalyticalJacobian = requires(const SpatialOp op, double coeff_dt, JacobianView jac) {
    { op.assemble_jacobian(coeff_dt, jac) } -> std::same_as<void>;
};

// Temporal event callback signature
using TemporalEventCallback = std::function<void(double t,
                                                  std::span<const double> x,
                                                  std::span<double> u)>;

// Obstacle callback signature
using ObstacleCallback = std::function<void(double t,
                                             std::span<const double> x,
                                             std::span<double> psi)>;

// Temporal event definition
struct TemporalEvent {
    double time;
    TemporalEventCallback callback;

    auto operator<=>(const TemporalEvent& other) const {
        return time <=> other.time;
    }
};

/// PDE Solver with TR-BDF2 time stepping using CRTP
///
/// Solves PDEs of the form: ∂u/∂t = L(u, x, t)
/// where L is a spatial operator (e.g., diffusion, advection, reaction)
///
/// Time stepping uses TR-BDF2 (Two-stage Runge-Kutta with BDF2):
/// - Stage 1: Trapezoidal rule to t_n + γ·dt
/// - Stage 2: BDF2 from t_n to t_n+1
/// where γ = 2 - √2 for L-stability
///
/// Uses CRTP to obtain boundary conditions and spatial operator from
/// derived class, eliminating redundant constructor parameters.
///
/// Derived classes must implement:
/// - left_boundary() - Returns left boundary condition object
/// - right_boundary() - Returns right boundary condition object
/// - spatial_operator() - Returns spatial operator object
///
/// @tparam Derived The derived solver class
template<typename Derived>
class PDESolver {
public:
    /// Constructor (CRTP version)
    ///
    /// @param grid Spatial grid (x coordinates)
    /// @param time Time domain configuration
    /// @param obstacle Optional obstacle condition ψ(x,t) for u ≥ ψ constraint
    /// @param external_workspace Optional external workspace for memory reuse
    /// @param output_buffer Optional buffer for collecting all time steps (size: (n_time+1)*n_space)
    ///                      Layout: [u_old_initial][step0][step1]...[step(n_time-1)]
    ///                      After step i, u_old points to step(i-1) (perfect cache locality!)
    ///                      If provided, solver writes directly to buffer (zero-copy)
    ///                      If not provided, solver uses internal workspace
    ///
    /// Note: Boundary conditions and spatial operator are obtained from derived class
    ///       via CRTP calls, not passed as constructor arguments
    /// Note: TR-BDF2 configuration uses defaults initially, can be changed via set_config()
    PDESolver(std::span<const double> grid,
              const TimeDomain& time,
              std::optional<ObstacleCallback> obstacle = std::nullopt,
              PDEWorkspace* external_workspace = nullptr,
              std::span<double> output_buffer = {})
        : grid_(grid)
        , time_(time)
        , config_{}  // Default-initialized
        , obstacle_(std::move(obstacle))
        , n_(grid.size())
        , workspace_owner_(nullptr)
        , workspace_(nullptr)
        , rhs_(n_)
        , jacobian_lower_(n_ - 1)
        , jacobian_diag_(n_)
        , jacobian_upper_(n_ - 1)
        , residual_(n_)
        , delta_u_(n_)
        , newton_u_old_(n_)
        , tridiag_workspace_(2 * n_)
        , isa_target_(cpu::select_isa_target())
    {
        // Acquire workspace (either external or create owned)
        acquire_workspace(grid, external_workspace);

        // Setup solution storage
        if (!output_buffer.empty()) {
            // External buffer provided - use it directly (zero-copy)
            // Buffer layout: [u_old_initial][step0][step1]...[step(n_time-1)]
            // Verify size
            size_t expected_size = (time.n_steps() + 1) * n_;
            if (output_buffer.size() < expected_size) {
                throw std::invalid_argument("Output buffer too small: need " +
                    std::to_string(expected_size) + " but got " + std::to_string(output_buffer.size()));
            }

            // u_old_ points to initial scratch, u_current_ points to step 0
            u_old_ = output_buffer.subspan(0, n_);
            u_current_ = output_buffer.subspan(n_, n_);
            output_buffer_ = output_buffer;
        } else {
            // No external buffer - use internal storage
            solution_storage_.resize(2 * n_);
            u_current_ = std::span{solution_storage_}.subspan(0, n_);
            u_old_ = std::span{solution_storage_}.subspan(n_, n_);
            output_buffer_ = {};
        }
    }

    /// Initialize with initial condition
    ///
    /// @param ic Initial condition function: ic(x, u)
    template<typename IC>
    void initialize(IC&& ic) {
        ic(grid_, std::span{u_current_});

        // Apply boundary conditions at t=0
        double t = time_.t_start();
        apply_obstacle(t, std::span{u_current_});
        apply_boundary_conditions(std::span{u_current_}, t);
    }

    /// Solve PDE from t_start to t_end
    ///
    /// @return expected success or solver error diagnostic
    std::expected<void, SolverError> solve() {
        double t = time_.t_start();
        const double dt = time_.dt();

        for (size_t step = 0; step < time_.n_steps(); ++step) {
            double t_old = t;

            // For internal storage only: copy u_current to u_old
            // For external buffer: u_old already points to previous slice (no copy!)
            if (output_buffer_.empty()) {
                std::copy(u_current_.begin(), u_current_.end(), u_old_.begin());
            }

            // Stage 1: Trapezoidal rule to t_n + γ·dt
            double t_stage1 = t + config_.gamma * dt;
            auto stage1_ok = solve_stage1(t, t_stage1, dt);
            if (!stage1_ok) {
                return std::unexpected(stage1_ok.error());
            }

            // Stage 2: BDF2 from t_n to t_n+1
            double t_next = t + dt;
            auto stage2_ok = solve_stage2(t_stage1, t_next, dt);
            if (!stage2_ok) {
                return std::unexpected(stage2_ok.error());
            }

            // Update time
            t = t_next;

            // Process temporal events AFTER completing the step
            process_temporal_events(t_old, t_next, step);

            // Advance pointers for next iteration (external buffer only)
            if (!output_buffer_.empty() && step + 1 < time_.n_steps()) {
                // Current slice becomes old for next iteration (perfect cache locality!)
                u_old_ = u_current_;
                // Advance to next slice: buffer layout is [initial][step0][step1]...
                // So step i is at offset (i+1)*n
                u_current_ = output_buffer_.subspan((step + 2) * n_, n_);
            }
        }

        return {};
    }

    /// Get current solution
    std::span<const double> solution() const {
        return std::span{u_current_};
    }

    /// Check if obstacle condition is present
    bool has_obstacle() const {
        return obstacle_.has_value();
    }

    /// Set TR-BDF2 configuration
    void set_config(const TRBDF2Config& config) {
        config_ = config;
    }

    /// Get TR-BDF2 configuration
    const TRBDF2Config& config() const {
        return config_;
    }

    /// Add temporal event to be executed at specific time
    ///
    /// Events are applied AFTER the TR-BDF2 step completes (not before).
    /// This ensures the PDE state is fully updated before event application.
    ///
    /// @param time Time at which to execute event
    /// @param callback Event callback: callback(t, x, u)
    void add_temporal_event(double time, TemporalEventCallback callback) {
        events_.push_back({time, std::move(callback)});
        std::sort(events_.begin(), events_.end(),
                  [](const auto& a, const auto& b) { return a.time < b.time; });
    }

protected:
    // CRTP helper to get derived class instance
    Derived& derived() { return static_cast<Derived&>(*this); }
    const Derived& derived() const { return static_cast<const Derived&>(*this); }

private:
    // Grid and configuration
    std::span<const double> grid_;
    TimeDomain time_;
    TRBDF2Config config_;
    std::optional<ObstacleCallback> obstacle_;

    // Grid size
    size_t n_;

    // Output buffer control
    std::span<double> output_buffer_;  // External buffer if provided

    // Workspace for cache blocking
    std::shared_ptr<PDEWorkspace> workspace_owner_;
    PDEWorkspace* workspace_;

    // Solution storage (spans point into either output_buffer_ or solution_storage_)
    std::vector<double> solution_storage_;  // Backing storage when no external buffer
    std::span<double> u_current_;           // u^{n+1} or u^{n+γ}
    std::span<double> u_old_;               // u^n
    std::vector<double> rhs_;               // n: RHS vector for stages

    // Newton workspace arrays (merged from NewtonWorkspace)
    std::vector<double> jacobian_lower_;      // n-1: Lower diagonal
    std::vector<double> jacobian_diag_;       // n: Main diagonal
    std::vector<double> jacobian_upper_;      // n-1: Upper diagonal
    std::vector<double> residual_;            // n: Residual vector
    std::vector<double> delta_u_;             // n: Newton step
    std::vector<double> newton_u_old_;        // n: Previous Newton iterate
    std::vector<double> tridiag_workspace_;   // 2n: Thomas algorithm workspace

    // ISA target for diagnostic logging
    cpu::ISATarget isa_target_;

    // Temporal event system
    std::vector<TemporalEvent> events_;
    size_t next_event_idx_ = 0;

    PDEWorkspace& acquire_workspace(std::span<const double> grid, PDEWorkspace* external_workspace) {
        if (external_workspace) {
            workspace_ = external_workspace;
            return *workspace_;
        }
        // Create GridSpec - assume uniform spacing
        double x_min = grid.front();
        double x_max = grid.back();
        auto grid_spec = GridSpec<double>::uniform(x_min, x_max, grid.size());
        if (!grid_spec.has_value()) {
            throw std::runtime_error("Failed to create grid spec");
        }

        // Use default memory resource for internal workspace
        auto ws_result = PDEWorkspace::create(grid_spec.value(), std::pmr::get_default_resource());
        if (!ws_result.has_value()) {
            throw std::runtime_error("Failed to create workspace: " + ws_result.error());
        }
        workspace_owner_ = ws_result.value();
        workspace_ = workspace_owner_.get();
        return *workspace_;
    }

    /// Process temporal events in time interval (t_old, t_new]
    ///
    /// Events are applied AFTER the TR-BDF2 step completes.
    /// This ensures proper ordering: PDE evolution happens first,
    /// then events modify the solution (e.g., dividend jumps).
    ///
    /// CRITICAL: After each event, obstacle and boundary conditions
    /// must be re-applied to maintain consistency. Dividend jumps
    /// interpolate the solution, which can violate constraints.
    void process_temporal_events(double t_old, double t_new, [[maybe_unused]] size_t step) {
        while (next_event_idx_ < events_.size()) {
            const auto& event = events_[next_event_idx_];

            if (event.time <= t_old) {
                next_event_idx_++;
                continue;
            }

            if (event.time > t_new) {
                break;
            }

            // Event is in (t_old, t_new] - apply it
            event.callback(event.time, grid_, std::span{u_current_});

            // CRITICAL FIX (Issue #98): Re-apply obstacle and boundary conditions
            // after event to maintain consistency. Dividend jumps interpolate
            // the solution, which can violate:
            // 1. Obstacle condition: u ≥ ψ (early exercise boundary)
            // 2. Boundary conditions: u[0] and u[n-1] values
            //
            // Without this, American puts with discrete dividends produce
            // values exceeding theoretical bounds (e.g., value > strike).
            apply_obstacle(event.time, std::span{u_current_});
            apply_boundary_conditions(std::span{u_current_}, event.time);

            next_event_idx_++;
        }
    }

    /// Apply obstacle condition: u(x,t) ≥ ψ(x,t)
    ///
    /// Projects solution onto obstacle constraint via complementarity:
    /// u[i] = max(u[i], ψ[i]) for all i
    ///
    /// This is called AFTER each Newton update to enforce variational
    /// inequality constraints (e.g., American option early exercise).
    void apply_obstacle(double t, std::span<double> u) {
        if (!obstacle_) return;

        auto psi = workspace_->psi();
        (*obstacle_)(t, grid_, psi);

        // Project: u[i] = max(u[i], psi[i])
        for (size_t i = 0; i < u.size(); ++i) {
            if (u[i] < psi[i]) {
                u[i] = psi[i];
            }
        }
    }

    /// Apply boundary conditions (CRTP version)
    void apply_boundary_conditions(std::span<double> u, double t) {
        auto dx_span = workspace_->dx();

        // Get BCs from derived class via CRTP (use const auto& to avoid copies!)
        const auto& left_bc = derived().left_boundary();
        const auto& right_bc = derived().right_boundary();

        // Left boundary
        double x_left = grid_[0];
        double dx_left = (n_ > 1) ? dx_span[0] : 1.0;
        double u_interior_left = (n_ > 1) ? u[1] : 0.0;
        left_bc.apply(u[0], x_left, t, dx_left, u_interior_left, 0.0, bc::BoundarySide::Left);

        // Right boundary
        double x_right = grid_[n_ - 1];
        double dx_right = (n_ > 1) ? dx_span[n_ - 2] : 1.0;
        double u_interior_right = (n_ > 1) ? u[n_ - 2] : 0.0;
        right_bc.apply(u[n_ - 1], x_right, t, dx_right, u_interior_right, 0.0, bc::BoundarySide::Right);
    }

    /// Apply spatial operator (single-pass evaluation, CRTP version)
    ///
    /// Note: Cache blocking was previously attempted but removed because it was
    /// ineffective. The blocked path still passed full arrays to the stencil,
    /// defeating locality benefits while adding loop overhead. True blocking would
    /// require materializing block-local buffers with halos, which adds complexity
    /// without clear benefit on modern CPUs with large caches.
    void apply_operator_with_blocking(double t,
                                      std::span<const double> u,
                                      std::span<double> Lu) {
        const size_t n = grid_.size();

        // Get spatial operator from derived class via CRTP (use const auto& to avoid copies!)
        const auto& spatial_op = derived().spatial_operator();

        // Direct evaluation (no blocking)
        spatial_op.apply(t, u, Lu);

        // Zero boundary values (BCs will override after)
        Lu[0] = Lu[n-1] = 0.0;
    }

    /// TR-BDF2 Stage 1: Trapezoidal rule
    ///
    /// u^{n+γ} = u^n + (γ·dt/2) · [L(u^n) + L(u^{n+γ})]
    ///
    /// Solved via Newton-Raphson iteration
    std::expected<void, SolverError> solve_stage1(double t_n, double t_stage, double dt) {
        const double w1 = config_.stage1_weight(dt);  // γ·dt/2

        // Compute L(u^n)
        apply_operator_with_blocking(t_n, std::span{u_old_}, workspace_->lu());

        // RHS = u^n + w1·L(u^n)
        // Use FMA for SAXPY-style loop
        for (size_t i = 0; i < n_; ++i) {
            rhs_[i] = std::fma(w1, workspace_->lu()[i], u_old_[i]);
        }

        // Initial guess: u* = u^n
        std::copy(u_old_.begin(), u_old_.end(), u_current_.begin());

        // Solve implicit stage
        auto result = solve_implicit_stage(t_stage, w1,
                                          std::span{u_current_},
                                          std::span{rhs_});

        if (!result.converged) {
            SolverError error{
                .code = SolverErrorCode::Stage1ConvergenceFailure,
                .message = result.failure_reason.value_or("TR-BDF2 stage1 failed to converge"),
                .iterations = result.iterations
            };

            if (error.message == "Singular Jacobian") {
                error.code = SolverErrorCode::LinearSolveFailure;
            }

            return std::unexpected(error);
        }

        return {};
    }

    /// TR-BDF2 Stage 2: BDF2
    ///
    /// Standard TR-BDF2 formulation (Ascher, Ruuth, Wetton 1995):
    /// u^{n+1} - [(1-γ)·dt/(2-γ)]·L(u^{n+1}) = [1/(γ(2-γ))]·u^{n+γ} - [(1-γ)²/(γ(2-γ))]·u^n
    ///
    /// Solved via Newton-Raphson iteration
    std::expected<void, SolverError> solve_stage2([[maybe_unused]] double t_stage, double t_next, double dt) {
        const double gamma = config_.gamma;
        const double one_minus_gamma = 1.0 - gamma;
        const double two_minus_gamma = 2.0 - gamma;
        const double denom = gamma * two_minus_gamma;

        // Correct BDF2 coefficients (Ascher, Ruuth, Wetton 1995)
        const double alpha = 1.0 / denom;  // Coefficient for u^{n+γ}
        const double beta = -(one_minus_gamma * one_minus_gamma) / denom;  // Coefficient for u^n
        const double w2 = config_.stage2_weight(dt);  // (1-γ)·dt/(2-γ)

        // RHS = alpha·u^{n+γ} + beta·u^n (u_current_ currently holds u^{n+γ})
        // Use FMA: alpha*u_current[i] + beta*u_old[i]
        for (size_t i = 0; i < n_; ++i) {
            rhs_[i] = std::fma(alpha, u_current_[i], beta * u_old_[i]);
        }

        // Initial guess: u^{n+1} = u* (already in u_current_)
        // (No need to copy, u_current_ already has u^{n+γ})

        // Solve implicit stage
        auto result = solve_implicit_stage(t_next, w2,
                                          std::span{u_current_},
                                          std::span{rhs_});

        if (!result.converged) {
            SolverError error{
                .code = SolverErrorCode::Stage2ConvergenceFailure,
                .message = result.failure_reason.value_or("TR-BDF2 stage2 failed to converge"),
                .iterations = result.iterations
            };

            if (error.message == "Singular Jacobian") {
                error.code = SolverErrorCode::LinearSolveFailure;
            }

            return std::unexpected(error);
        }

        return {};
    }

    // ========================================================================
    // Newton-Raphson methods (for implicit stage solving)
    // ========================================================================
    //
    // NOTE: These methods are TR-BDF2-specific implementation details and NOT
    // intended as a general-purpose Newton solver. They solve the specific form:
    //   u = rhs + coeff_dt·L(u)
    // which arises from implicit time-stepping in PDEs.
    //
    // For general root-finding needs (e.g., Brent's method for scalar equations),
    // see root_finding.hpp. A truly general Newton solver would use function
    // pointers/callables rather than template parameters for boundary conditions
    // and spatial operators.
    //
    // These methods were previously in a separate NewtonSolver class but were
    // merged into PDESolver to make the design honest about their specific
    // purpose. See PR #97 for the architectural discussion.
    // ========================================================================

    /// Solve implicit stage equation via Newton-Raphson
    ///
    /// Solves: u = rhs + coeff_dt·L(u)
    /// Equivalently: F(u) = u - rhs - coeff_dt·L(u) = 0
    ///
    /// @param t Time at which to evaluate operators
    /// @param coeff_dt TR-BDF2 weight (stage1_weight or stage2_weight)
    /// @param u Solution vector (input: initial guess, output: converged solution)
    /// @param rhs Right-hand side from previous stage
    /// @return Result with convergence status
    NewtonResult solve_implicit_stage(double t, double coeff_dt,
                                      std::span<double> u,
                                      std::span<const double> rhs) {
        const double eps = config_.jacobian_fd_epsilon;

        // Apply BCs to initial guess
        apply_boundary_conditions(u, t);

        // Quasi-Newton: Build Jacobian once and reuse
        build_jacobian(t, coeff_dt, u, eps);

        // Copy initial guess
        std::copy(u.begin(), u.end(), newton_u_old_.begin());

        // Newton iteration
        for (size_t iter = 0; iter < config_.max_iter; ++iter) {
            // Evaluate L(u)
            apply_operator_with_blocking(t, u, workspace_->lu());

            // Compute residual: F(u) = u - rhs - coeff_dt·L(u)
            compute_residual(u, coeff_dt, workspace_->lu(), rhs,
                           residual_);

            // CRITICAL FIX: Pass u explicitly to avoid reading stale workspace
            apply_bc_to_residual(residual_, u, t);

            // Newton method: Solve J·δu = -F(u), then update u ← u + δu
            // Negate residual for RHS
            for (size_t i = 0; i < n_; ++i) {
                residual_[i] = -residual_[i];
            }

            // Solve J·δu = -F(u) using Thomas algorithm
            auto result = solve_thomas<double>(
                jacobian_lower_,
                jacobian_diag_,
                jacobian_upper_,
                residual_,
                delta_u_,
                tridiag_workspace_
            );

            if (!result.ok()) {
                return {false, iter, std::numeric_limits<double>::infinity(),
                       "Singular Jacobian"};
            }

            // Update: u ← u + δu
            for (size_t i = 0; i < n_; ++i) {
                u[i] += delta_u_[i];
            }

            // Apply obstacle projection BEFORE boundary conditions
            // This ensures complementarity: u ≥ ψ
            apply_obstacle(t, u);

            apply_boundary_conditions(u, t);

            // Check convergence via step delta
            double error = compute_step_delta_error(u, newton_u_old_);

            if (error < config_.tolerance) {
                return {true, iter + 1, error, std::nullopt};
            }

            // Prepare for next iteration
            std::copy(u.begin(), u.end(), newton_u_old_.begin());
        }

        return {false, config_.max_iter,
               compute_step_delta_error(u, newton_u_old_),
               "Max iterations reached"};
    }

    void compute_residual(std::span<const double> u, double coeff_dt,
                         std::span<const double> Lu,
                         std::span<const double> rhs,
                         std::span<double> residual) {
        // F(u) = u - rhs - coeff_dt·L(u) = 0
        // We want to solve u = rhs + coeff_dt·L(u)
        // Use FMA: u[i] - rhs[i] - coeff_dt*Lu[i] = (u[i] - rhs[i]) + (-coeff_dt)*Lu[i]
        for (size_t i = 0; i < n_; ++i) {
            residual[i] = std::fma(-coeff_dt, Lu[i], u[i] - rhs[i]);
        }
    }

    double compute_step_delta_error(std::span<const double> u_new,
                                    std::span<const double> u_old) {
        double sum_sq_error = 0.0;
        double sum_sq_norm = 0.0;
        // Use FMA for sum of squares: sum += x*x
        for (size_t i = 0; i < n_; ++i) {
            double diff = u_new[i] - u_old[i];
            sum_sq_error = std::fma(diff, diff, sum_sq_error);
            sum_sq_norm = std::fma(u_new[i], u_new[i], sum_sq_norm);
        }
        double rms_error = std::sqrt(sum_sq_error / n_);
        double rms_norm = std::sqrt(sum_sq_norm / n_);
        const double epsilon = 1e-12;
        return (rms_norm > epsilon) ? rms_error / (rms_norm + epsilon) : rms_error;
    }

    void apply_bc_to_residual(std::span<double> residual,
                              std::span<const double> u,
                              double t) {
        // Get BCs from derived class via CRTP (use const auto& to avoid copies!)
        const auto& left_bc = derived().left_boundary();
        const auto& right_bc = derived().right_boundary();

        // For Dirichlet BC: F(u) = u - g, so residual = u - g
        // Left boundary
        using LeftBCType = std::remove_cvref_t<decltype(left_bc)>;
        if constexpr (std::is_same_v<bc::boundary_tag_t<LeftBCType>, bc::dirichlet_tag>) {
            double g = left_bc.value(t, grid_[0]);
            residual[0] = u[0] - g;  // u - g (we want u = g, so F = u - g = 0)
        }

        // Right boundary
        using RightBCType = std::remove_cvref_t<decltype(right_bc)>;
        if constexpr (std::is_same_v<bc::boundary_tag_t<RightBCType>, bc::dirichlet_tag>) {
            double g = right_bc.value(t, grid_[n_ - 1]);
            residual[n_ - 1] = u[n_ - 1] - g;  // u - g
        }
    }

    void build_jacobian(double t, double coeff_dt,
                       std::span<const double> u, double eps) {
        // Get spatial operator from derived class via CRTP (use const auto& to avoid copies!)
        const auto& spatial_op = derived().spatial_operator();
        using SpatialOpType = std::remove_cvref_t<decltype(spatial_op)>;

        // Dispatch to analytical or finite-difference Jacobian
        if constexpr (HasAnalyticalJacobian<SpatialOpType>) {
            // Analytical Jacobian (O(n) - fast path)
            JacobianView jac(jacobian_lower_,
                           jacobian_diag_,
                           jacobian_upper_);
            spatial_op.assemble_jacobian(coeff_dt, jac);
        } else {
            // Finite-difference Jacobian (O(n²) - fallback for unsupported operators)
            build_jacobian_finite_difference(t, coeff_dt, u, eps);
        }

        // Boundary rows (same for both methods)
        build_jacobian_boundaries(t, coeff_dt, u, eps);
    }

    /// Finite-difference Jacobian (fallback for unsupported operators)
    ///
    /// This is the original O(n²) implementation using finite differences.
    /// Used when spatial operator doesn't provide analytical Jacobian.
    void build_jacobian_finite_difference(double t, double coeff_dt,
                                          std::span<const double> u, double eps) {
        // Initialize u_perturb and compute baseline L(u)
        std::copy(u.begin(), u.end(), workspace_->u_stage().begin());
        apply_operator_with_blocking(t, u, workspace_->lu());

        // Interior points: tridiagonal structure via finite differences
        // J = ∂F/∂u where F(u) = u - rhs - coeff_dt·L(u)
        // So ∂F/∂u = I - coeff_dt·∂L/∂u
        for (size_t i = 1; i < n_ - 1; ++i) {
            // Diagonal: ∂F/∂u_i = 1 - coeff_dt·∂L_i/∂u_i
            workspace_->u_stage()[i] = u[i] + eps;
            apply_operator_with_blocking(t, workspace_->u_stage(), workspace_->rhs());
            double dLi_dui = (workspace_->rhs()[i] - workspace_->lu()[i]) / eps;
            jacobian_diag_[i] = 1.0 - coeff_dt * dLi_dui;
            workspace_->u_stage()[i] = u[i];

            // Lower diagonal: ∂F_i/∂u_{i-1} = -coeff_dt·∂L_i/∂u_{i-1}
            workspace_->u_stage()[i - 1] = u[i - 1] + eps;
            apply_operator_with_blocking(t, workspace_->u_stage(), workspace_->rhs());
            double dLi_duim1 = (workspace_->rhs()[i] - workspace_->lu()[i]) / eps;
            jacobian_lower_[i - 1] = -coeff_dt * dLi_duim1;
            workspace_->u_stage()[i - 1] = u[i - 1];

            // Upper diagonal: ∂F_i/∂u_{i+1} = -coeff_dt·∂L_i/∂u_{i+1}
            workspace_->u_stage()[i + 1] = u[i + 1] + eps;
            apply_operator_with_blocking(t, workspace_->u_stage(), workspace_->rhs());
            double dLi_duip1 = (workspace_->rhs()[i] - workspace_->lu()[i]) / eps;
            jacobian_upper_[i] = -coeff_dt * dLi_duip1;
            workspace_->u_stage()[i + 1] = u[i + 1];
        }
    }

    void build_jacobian_boundaries(double t, double coeff_dt,
                                   std::span<const double> u, double eps) {
        // Get BCs from derived class via CRTP (use const auto& to avoid copies!)
        const auto& left_bc = derived().left_boundary();
        const auto& right_bc = derived().right_boundary();
        using LeftBCType = std::remove_cvref_t<decltype(left_bc)>;
        using RightBCType = std::remove_cvref_t<decltype(right_bc)>;

        // Left boundary
        if constexpr (std::is_same_v<bc::boundary_tag_t<LeftBCType>, bc::dirichlet_tag>) {
            // For Dirichlet: F(u) = u - g, so ∂F/∂u = 1
            jacobian_diag_[0] = 1.0;
            jacobian_upper_[0] = 0.0;
        } else if constexpr (std::is_same_v<bc::boundary_tag_t<LeftBCType>, bc::neumann_tag>) {
            // For Neumann: F(u) = u - rhs - coeff_dt·L(u)
            workspace_->u_stage()[0] = u[0] + eps;
            apply_operator_with_blocking(t, workspace_->u_stage(), workspace_->rhs());
            double dL0_du0 = (workspace_->rhs()[0] - workspace_->lu()[0]) / eps;
            jacobian_diag_[0] = 1.0 - coeff_dt * dL0_du0;
            workspace_->u_stage()[0] = u[0];

            workspace_->u_stage()[1] = u[1] + eps;
            apply_operator_with_blocking(t, workspace_->u_stage(), workspace_->rhs());
            double dL0_du1 = (workspace_->rhs()[0] - workspace_->lu()[0]) / eps;
            jacobian_upper_[0] = -coeff_dt * dL0_du1;
            workspace_->u_stage()[1] = u[1];
        }

        // Right boundary
        if constexpr (std::is_same_v<bc::boundary_tag_t<RightBCType>, bc::dirichlet_tag>) {
            // For Dirichlet: F(u) = u - g, so ∂F/∂u = 1
            jacobian_diag_[n_ - 1] = 1.0;
        } else if constexpr (std::is_same_v<bc::boundary_tag_t<RightBCType>, bc::neumann_tag>) {
            // For Neumann: F(u) = u - rhs - coeff_dt·L(u)
            size_t i = n_ - 1;
            workspace_->u_stage()[i] = u[i] + eps;
            apply_operator_with_blocking(t, workspace_->u_stage(), workspace_->rhs());
            double dLi_dui = (workspace_->rhs()[i] - workspace_->lu()[i]) / eps;
            jacobian_diag_[i] = 1.0 - coeff_dt * dLi_dui;
            workspace_->u_stage()[i] = u[i];
        }
    }

};

}  // namespace mango
