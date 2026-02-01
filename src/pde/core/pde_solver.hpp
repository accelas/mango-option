// SPDX-License-Identifier: MIT
#pragma once

#include "src/pde/core/grid.hpp"
#include "src/pde/core/pde_workspace.hpp"
#include "src/pde/operators/centered_difference_facade.hpp"
#include "src/pde/core/boundary_conditions.hpp"
#include "src/pde/core/time_domain.hpp"
#include "src/pde/core/trbdf2_config.hpp"
#include "src/math/thomas_solver.hpp"
#include "src/math/tridiagonal_matrix_view.hpp"
#include <expected>
#include "src/support/error_types.hpp"
#include <memory>
#include <span>
#include <vector>
#include <functional>
#include <cmath>
#include <algorithm>
#include <limits>

namespace mango {

/// Concept to detect spatial operators with analytical Jacobian capability
template<typename SpatialOp>
concept HasAnalyticalJacobian = requires(const SpatialOp op, double t, double coeff_dt, TridiagonalMatrixView jac) {
    { op.assemble_jacobian(t, coeff_dt, jac) } -> std::same_as<void>;
};

/// Concept to detect if derived class provides obstacle condition
template<typename Derived>
concept HasObstacle = requires(const Derived& d, double t, std::span<const double> x, std::span<double> psi) {
    { d.obstacle(t, x, psi) } -> std::same_as<void>;
};

// Temporal event callback signature
using TemporalEventCallback = std::function<void(double t,
                                                  std::span<const double> x,
                                                  std::span<double> u)>;

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
/// Uses CRTP to obtain boundary conditions, spatial operator, and obstacle
/// from derived class, eliminating redundant constructor parameters.
///
/// Derived classes must implement:
/// - left_boundary() - Returns left boundary condition object
/// - right_boundary() - Returns right boundary condition object
/// - spatial_operator() - Returns spatial operator object
/// - obstacle(t, x, psi) - Optional: computes obstacle constraint (if HasObstacle<Derived>)
///
/// @tparam Derived The derived solver class
template<typename Derived>
class PDESolver {
public:
    /// Constructor (New design: Grid + Workspace separation)
    ///
    /// @param grid Grid with solution storage (outlives solver, passed by shared_ptr)
    /// @param workspace Named spans to caller-managed PMR buffers
    ///
    /// Note: Boundary conditions, spatial operator, and obstacle are obtained from
    ///       derived class via CRTP calls, not passed as constructor arguments
    /// Note: TR-BDF2 configuration uses defaults initially, can be changed via set_config()
    PDESolver(std::shared_ptr<Grid<double>> grid,
              PDEWorkspace workspace)
        : grid_(grid)  // Copy shared_ptr (not move - shared ownership)
        , config_{}  // Default-initialized
        , n_(grid->n_space())
        , workspace_(workspace)
    {
        // Grid owns solution storage, workspace provides temporary buffers
        // No allocation needed in constructor
    }

    /// Initialize with initial condition
    ///
    /// @param ic Initial condition function: ic(x, u)
    template<typename IC>
    void initialize(IC&& ic) {
        auto u_current = grid_->solution();
        ic(grid_->x(), u_current);

        // Apply constraints at t=0 (boundary before obstacle, consistent with Newton iteration)
        double t = grid_->time().t_start();
        apply_boundary_conditions(u_current, t);
        apply_obstacle(t, u_current);

        // Copy initial condition to u_prev for first iteration
        auto u_prev = grid_->solution_prev();
        std::copy(u_current.begin(), u_current.end(), u_prev.begin());
    }

    /// Solve PDE from t_start to t_end
    ///
    /// @return expected success or solver error diagnostic
    std::expected<void, SolverError> solve() {
        const auto& time = grid_->time();
        const auto time_pts = time.time_points();
        double t = time_pts[0];

        auto u_current = grid_->solution();
        auto u_prev = grid_->solution_prev();

        // Record initial condition if requested
        if (grid_->should_record(0)) {
            grid_->record(0, u_current);
        }

        size_t step = 0;
        if (config_.rannacher_startup && time.n_steps() > 0) {
            double dt_step = time.dt_at(0);
            auto rannacher_ok = solve_rannacher_startup(t, dt_step, u_current, u_prev)
                .transform([&] {
                    t = time_pts[1];  // avoid float drift
                    if (grid_->should_record(step + 1)) {
                        grid_->record(step + 1, u_current);
                    }
                    step = 1;
                });
            if (!rannacher_ok) {
                return std::unexpected(rannacher_ok.error());
            }
        }

        for (; step < time.n_steps(); ++step) {
            double t_old = t;
            double dt_step = time.dt_at(step);
            double t_next = time_pts[step + 1];  // read from sequence, not t + dt

            // Copy u_current to u_prev for next iteration
            std::copy(u_current.begin(), u_current.end(), u_prev.begin());

            // Stage 1: Trapezoidal rule to t_n + γ·dt
            double t_stage1 = t + config_.gamma * dt_step;
            auto stage1_ok = solve_stage1(t, t_stage1, dt_step, u_current, u_prev);
            if (!stage1_ok) {
                return std::unexpected(stage1_ok.error());
            }

            // Stage 2: BDF2 from t_n to t_n+1
            auto stage2_ok = solve_stage2(t_stage1, t_next, dt_step, u_current, u_prev);
            if (!stage2_ok) {
                return std::unexpected(stage2_ok.error());
            }

            // Process temporal events AFTER completing the step
            process_temporal_events(t_old, t_next, step, u_current);

            // Update time from sequence (not accumulation)
            t = t_next;

            // Record snapshot AFTER events
            if (grid_->should_record(step + 1)) {
                grid_->record(step + 1, u_current);
            }
        }

        // Final solution is already in grid_->solution()
        return {};
    }

    /// Get current solution
    std::span<const double> solution() const {
        return grid_->solution();
    }

    /// Check if obstacle condition is present (compile-time concept check)
    static constexpr bool has_obstacle() {
        return HasObstacle<Derived>;
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
    // Grid with solution storage (persistent, outlives solver)
    std::shared_ptr<Grid<double>> grid_;

    // Configuration
    TRBDF2Config config_;

    // Grid size
    size_t n_;

    // Workspace spans (caller-managed PMR buffers)
    PDEWorkspace workspace_;

    // Temporal event system
    std::vector<TemporalEvent> events_;
    size_t next_event_idx_ = 0;

    /// Process temporal events in time interval (t_old, t_new]
    ///
    /// Events are applied AFTER the TR-BDF2 step completes.
    /// This ensures proper ordering: PDE evolution happens first,
    /// then events modify the solution (e.g., dividend jumps).
    ///
    /// CRITICAL: After each event, obstacle and boundary conditions
    /// must be re-applied to maintain consistency. Dividend jumps
    /// interpolate the solution, which can violate constraints.
    void process_temporal_events(double t_old, double t_new, [[maybe_unused]] size_t step,
                                  std::span<double> u_current) {
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
            event.callback(event.time, grid_->x(), u_current);

            // CRITICAL FIX (Issue #98): Re-apply obstacle and boundary conditions
            // after event to maintain consistency. Dividend jumps interpolate
            // the solution, which can violate:
            // 1. Obstacle condition: u ≥ ψ (early exercise boundary)
            // 2. Boundary conditions: u[0] and u[n-1] values
            //
            // Without this, American puts with discrete dividends produce
            // values exceeding theoretical bounds (e.g., value > strike).
            // Apply boundary before obstacle (consistent with Newton iteration)
            apply_boundary_conditions(u_current, event.time);
            apply_obstacle(event.time, u_current);

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
    ///
    /// Uses CRTP with concept check - only calls derived().obstacle() if
    /// Derived class satisfies HasObstacle<Derived> concept.
    void apply_obstacle(double t, std::span<double> u) {
        if constexpr (HasObstacle<Derived>) {
            auto psi = workspace_.psi();
            derived().obstacle(t, grid_->x(), psi);

            // Project: u[i] = max(u[i], psi[i])
            for (size_t i = 0; i < u.size(); ++i) {
                if (u[i] < psi[i]) {
                    u[i] = psi[i];
                }
            }
        }
    }

    /// Apply boundary conditions (CRTP version)
    void apply_boundary_conditions(std::span<double> u, double t) {
        auto dx_span = workspace_.dx();

        // Get BCs from derived class via CRTP (use const auto& to avoid copies!)
        const auto& left_bc = derived().left_boundary();
        const auto& right_bc = derived().right_boundary();

        // Left boundary
        double x_left = grid_->x()[0];
        double dx_left = (n_ > 1) ? dx_span[0] : 1.0;
        double u_interior_left = (n_ > 1) ? u[1] : 0.0;
        left_bc.apply(u[0], x_left, t, dx_left, u_interior_left, 0.0, bc::BoundarySide::Left);

        // Right boundary
        double x_right = grid_->x()[n_ - 1];
        double dx_right = (n_ > 1) ? dx_span[n_ - 2] : 1.0;
        double u_interior_right = (n_ > 1) ? u[n_ - 2] : 0.0;
        right_bc.apply(u[n_ - 1], x_right, t, dx_right, u_interior_right, 0.0, bc::BoundarySide::Right);
    }

    /// Apply spatial operator and zero boundary values
    void apply_spatial_operator(double t,
                                      std::span<const double> u,
                                      std::span<double> Lu) {
        const size_t n = grid_->x().size();

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
    std::expected<void, SolverError> solve_stage1(double t_n, double t_stage, double dt,
                                                   std::span<double> u_current,
                                                   std::span<const double> u_prev) {
        const double w1 = config_.stage1_weight(dt);  // γ·dt/2

        // Compute L(u^n)
        apply_spatial_operator(t_n, u_prev, workspace_.lu());

        // RHS = u^n + w1·L(u^n)
        // Use FMA for SAXPY-style loop
        auto rhs = workspace_.rhs();
        for (size_t i = 0; i < n_; ++i) {
            rhs[i] = std::fma(w1, workspace_.lu()[i], u_prev[i]);
        }

        // Initial guess: u* = u^n
        std::copy(u_prev.begin(), u_prev.end(), u_current.begin());

        return solve_implicit_stage(t_stage, w1, u_current, rhs);
    }

    /// TR-BDF2 Stage 2: BDF2
    ///
    /// Standard TR-BDF2 formulation (Ascher, Ruuth, Wetton 1995):
    /// u^{n+1} - [(1-γ)·dt/(2-γ)]·L(u^{n+1}) = [1/(γ(2-γ))]·u^{n+γ} - [(1-γ)²/(γ(2-γ))]·u^n
    ///
    /// Solved via Newton-Raphson iteration
    std::expected<void, SolverError> solve_stage2([[maybe_unused]] double t_stage, double t_next, double dt,
                                                   std::span<double> u_current,
                                                   std::span<const double> u_prev) {
        const double gamma = config_.gamma;
        const double one_minus_gamma = 1.0 - gamma;
        const double two_minus_gamma = 2.0 - gamma;
        const double denom = gamma * two_minus_gamma;

        // Correct BDF2 coefficients (Ascher, Ruuth, Wetton 1995)
        const double alpha = 1.0 / denom;  // Coefficient for u^{n+γ}
        const double beta = -(one_minus_gamma * one_minus_gamma) / denom;  // Coefficient for u^n
        const double w2 = config_.stage2_weight(dt);  // (1-γ)·dt/(2-γ)

        // RHS = alpha·u^{n+γ} + beta·u^n (u_current currently holds u^{n+γ})
        // Use FMA: alpha*u_current[i] + beta*u_prev[i]
        auto rhs = workspace_.rhs();
        for (size_t i = 0; i < n_; ++i) {
            rhs[i] = std::fma(alpha, u_current[i], beta * u_prev[i]);
        }

        // Initial guess: u^{n+1} = u* (already in u_current)
        // (No need to copy, u_current already has u^{n+γ})

        return solve_implicit_stage(t_next, w2, u_current, rhs);
    }

    /// Implicit Euler step (used for Rannacher startup)
    ///
    /// u^{n+1} - dt·L(u^{n+1}) = u^n
    std::expected<void, SolverError> solve_implicit_euler([[maybe_unused]] double t_n, double t_next, double dt,
                                                          std::span<double> u_current,
                                                          std::span<const double> u_prev) {
        auto rhs = workspace_.rhs();
        for (size_t i = 0; i < n_; ++i) {
            rhs[i] = u_prev[i];
        }

        // Initial guess: u^{n+1} = u^n
        std::copy(u_prev.begin(), u_prev.end(), u_current.begin());

        return solve_implicit_stage(t_next, dt, u_current, rhs);
    }

    /// Rannacher startup: two half-step implicit Euler updates for step 0
    std::expected<void, SolverError> solve_rannacher_startup(double t,
                                                             double dt,
                                                             std::span<double> u_current,
                                                             std::span<double> u_prev) {
        double t_old = t;
        double t_next = t + dt;
        const double half_dt = 0.5 * dt;
        const double t_mid = t + half_dt;

        // Copy u_current to u_prev for next iteration
        std::copy(u_current.begin(), u_current.end(), u_prev.begin());

        // First half-step: implicit Euler to t_mid
        auto euler1_ok = solve_implicit_euler(t, t_mid, half_dt, u_current, u_prev);
        if (!euler1_ok) {
            return std::unexpected(euler1_ok.error());
        }

        // Process temporal events in (t, t_mid]
        process_temporal_events(t_old, t_mid, 0, u_current);

        // Prepare for second half-step
        std::copy(u_current.begin(), u_current.end(), u_prev.begin());

        // Second half-step: implicit Euler to t_next
        auto euler2_ok = solve_implicit_euler(t_mid, t_next, half_dt, u_current, u_prev);
        if (!euler2_ok) {
            return std::unexpected(euler2_ok.error());
        }

        // Process temporal events in (t_mid, t_next]
        process_temporal_events(t_mid, t_next, 0, u_current);

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

    /// Solve implicit stage equation: u = rhs + coeff_dt·L(u)
    ///
    /// Dispatches at compile time based on obstacle presence:
    /// - With obstacle → Projected Thomas (Brennan-Schwartz LCP, single pass)
    /// - Without obstacle → Newton iteration (quasi-Newton, iterative)
    std::expected<void, SolverError> solve_implicit_stage(double t, double coeff_dt,
                                                          std::span<double> u,
                                                          std::span<const double> rhs) {
        if constexpr (HasObstacle<Derived>) {
            return solve_implicit_stage_projected(t, coeff_dt, u, rhs);
        } else {
            return solve_implicit_stage_newton(t, coeff_dt, u, rhs);
        }
    }

    /// Solve implicit stage using Projected Thomas (Brennan-Schwartz) algorithm
    ///
    /// **Algorithm Overview:**
    /// Solves the Linear Complementarity Problem (LCP) for American option pricing:
    ///   A·u = rhs,  subject to u ≥ ψ (obstacle constraint)
    /// where A = I - coeff_dt·∂L/∂u is the TR-BDF2 stage matrix.
    ///
    /// **Key Mathematical Insight:**
    /// This method solves the TRUE TR-BDF2 stage equation directly, NOT a Newton correction.
    ///
    /// CORRECT (this implementation):
    ///   A·u = rhs  where A = I - coeff_dt·∂L/∂u
    ///   → Solves for u directly via projected tridiagonal solver
    ///
    /// WRONG (previous Newton-based approach):
    ///   J·δu = -F(u)  where F(u) = u - rhs - coeff_dt·L(u)
    ///   → Solves for correction δu, then u ← u + δu
    ///   → For deep ITM, this lifted values above intrinsic (wrong physics)
    ///
    /// **Brennan-Schwartz Projection:**
    /// Standard tridiagonal solver with obstacle projection during backward substitution:
    ///   u[i] = max(unconstrained_value, ψ[i])
    /// This couples the obstacle constraint directly with the tridiagonal structure,
    /// ensuring u ≥ ψ at every node without iteration.
    ///
    /// **Advantages over Newton + Heuristic Active Set:**
    /// - No dual variables λ → no sensitivity to numerical noise
    /// - No iteration → always converges in single pass for M-matrices
    /// - Direct physics: solves stage equation, not correction equation
    /// - Provably correct for American options (Brennan-Schwartz 1977)
    ///
    /// **Critical Fixes Applied:**
    /// 1. Dirichlet RHS correction: For direct solve A·u = rhs, Dirichlet rows
    ///    need rhs = g(t), not the interior formula rhs = u_old + w·L(u_old)
    /// 2. Deep exercise locking: Nodes deep ITM (ψ > 0.95) are locked to obstacle
    ///    to prevent diffusion from lifting values above intrinsic value
    ///
    /// @param t Current time in PDE (backward from T to 0)
    /// @param coeff_dt Time step coefficient (w1 for stage 1, w2 for stage 2)
    /// @param u Solution vector (input: initial guess, output: solution)
    /// @param rhs Right-hand side computed by caller (interior formula)
    /// @return Success, or LinearSolveFailure if the tridiagonal system is singular
    std::expected<void, SolverError> solve_implicit_stage_projected(
            double t, double coeff_dt,
            std::span<double> u,
            std::span<const double> rhs) {
        // Apply boundary conditions to initial guess
        apply_boundary_conditions(u, t);

        // ═══════════════════════════════════════════════════════════════════════
        // STEP 1: Build the TRUE TR-BDF2 stage matrix A = I - coeff_dt·∂L/∂u
        // ═══════════════════════════════════════════════════════════════════════
        // For TR-BDF2, each implicit stage has the form:
        //   u - coeff_dt·L(u) = rhs
        // Linearizing L(u) ≈ L(u_prev) + ∂L/∂u·(u - u_prev), we get:
        //   (I - coeff_dt·∂L/∂u)·u = rhs'
        // where A = I - coeff_dt·∂L/∂u is the stage matrix (Jacobian).
        //
        // This is FUNDAMENTALLY DIFFERENT from Newton iteration:
        //   Newton: J·δu = -F(u) where F(u) = u - rhs - coeff_dt·L(u)
        //   TR-BDF2 stage: A·u = rhs where A = I - coeff_dt·∂L/∂u
        //
        // The Newton approach solves for a CORRECTION δu and updates u ← u + δu.
        // The TR-BDF2 stage solves for u DIRECTLY. This distinction is critical
        // for American options because the obstacle constraint applies to u, not δu.
        build_jacobian(t, coeff_dt, u, config_.jacobian_fd_epsilon);

        // ═══════════════════════════════════════════════════════════════════════
        // CRITICAL FIX #1: Dirichlet Boundary RHS Correction
        // ═══════════════════════════════════════════════════════════════════════
        // When solving A·u = rhs DIRECTLY (not Newton correction J·δu = -F),
        // Dirichlet boundaries require special RHS treatment.
        //
        // For Dirichlet BC u[0] = g(t), the Jacobian row becomes [0, 1, 0, ...],
        // so the equation is simply: 1·u[0] = rhs[0]
        // Therefore, we MUST have rhs[0] = g(t) (the boundary value).
        //
        // The caller provides rhs with the INTERIOR formula:
        //   rhs[i] = u_old[i] + w1·L(u_old)[i]  (Stage 1)
        //   rhs[i] = α·u_stage1[i] + β·u_old[i]  (Stage 2)
        // This formula is WRONG for Dirichlet rows! We must override it.
        //
        // Why this wasn't needed for Newton: In J·δu = -F(u), the BC rows
        // enforce δu[0] = 0 (no change), so rhs = 0 is correct.
        // But for A·u = rhs, we need rhs = g(t) to get u[0] = g(t).
        auto rhs_with_bc = workspace_.reserved1();
        std::copy(rhs.begin(), rhs.end(), rhs_with_bc.begin());

        // Apply Dirichlet boundary values to RHS
        const auto& left_bc = derived().left_boundary();
        const auto& right_bc = derived().right_boundary();
        using LeftBCType = std::remove_cvref_t<decltype(left_bc)>;
        using RightBCType = std::remove_cvref_t<decltype(right_bc)>;

        if constexpr (std::is_same_v<bc::boundary_tag_t<LeftBCType>, bc::dirichlet_tag>) {
            rhs_with_bc[0] = left_bc.value(t, grid_->x()[0]);
        }
        if constexpr (std::is_same_v<bc::boundary_tag_t<RightBCType>, bc::dirichlet_tag>) {
            rhs_with_bc[n_-1] = right_bc.value(t, grid_->x()[n_-1]);
        }

        // Get obstacle constraint via CRTP
        // Note: This method is only called when HasObstacle<Derived> is true,
        // so we can safely call derived().obstacle()
        auto psi = workspace_.psi();

        // Evaluate obstacle constraint ψ(x,t) at current time
        derived().obstacle(t, grid_->x(), psi);

        // ═══════════════════════════════════════════════════════════════════════
        // CRITICAL FIX #2: Lock Deep Exercise Region to Prevent Diffusion Lift
        // ═══════════════════════════════════════════════════════════════════════
        // **Problem:** For VERY deep ITM options (e.g., S=0.25, K=100), the PDE
        // has dominant diffusion that can lift interior values above intrinsic.
        //
        // **Physics:** Deep ITM, exercise is definitively optimal at ALL nodes in
        // the region. There is NO continuation value - only intrinsic value.
        // The solution should be u(x) = ψ(x) = max(K - S·exp(x), 0) everywhere.
        //
        // **Why Projected Thomas alone isn't enough:**
        // - Projected Thomas enforces u ≥ ψ during backward substitution
        // - But the forward elimination and tridiagonal coupling can still allow
        //   diffusion to "push" values above ψ in the unconstrained solve
        // - When back-substituting, if the unconstrained value is > ψ, projection
        //   doesn't activate, and we get u > ψ (wrong!)
        //
        // **Solution:** Lock deep ITM nodes by converting them to Dirichlet constraints.
        // This ELIMINATES the tridiagonal coupling for those nodes, preventing diffusion
        // from affecting them. The equation becomes simply: u[i] = ψ[i].
        //
        // **Threshold Selection:**
        // - Lock only if ψ > 0.95 (95% of strike in normalized units)
        // - For American puts: ψ = max(1 - exp(x), 0), so:
        //     ψ > 0.95  ⟺  1 - exp(x) > 0.95  ⟺  exp(x) < 0.05  ⟺  x < -3.0
        // - This is ~3 log-moneyness units deep ITM, well separated from ATM (x=0)
        // - ATM/near-ATM nodes (where time value exists) remain free to solve via LCP
        //
        // **Why 0.95 and not 0.99 or 1.0?**
        // - 0.99 would only lock the extreme boundary (x → -∞)
        // - 0.95 captures the entire region where exercise is definitively optimal
        // - Empirically verified: S=0.25, K=100 has ψ ≈ 0.9975 at spot, needs locking
        //
        // **Implementation:** Convert Jacobian row to identity:
        //   Before: [a_lower[i], a_diag[i], a_upper[i]] · [u[i-1], u[i], u[i+1]]ᵀ = rhs[i]
        //   After:  [0, 1, 0] · [u[i-1], u[i], u[i+1]]ᵀ = ψ[i]
        //   Result: u[i] = ψ[i] (Dirichlet constraint)
        constexpr double deep_itm_threshold = 0.95;  // Lock if ψ > 95% of strike
        constexpr double exercise_tolerance = 1e-8;  // Tolerance for "on obstacle"

        for (size_t i = 1; i < n_ - 1; ++i) {
            // Check both conditions:
            // 1. Deep ITM: ψ[i] > 0.95 (far from ATM where time value matters)
            // 2. On obstacle: u[i] - ψ[i] < 1e-8 (solution already at intrinsic)
            //
            // The second condition prevents locking nodes that legitimately have
            // time value > 0 (e.g., at earlier times before convergence to payoff).
            bool deep_itm = (psi[i] > deep_itm_threshold);
            bool at_obstacle = (u[i] - psi[i] < exercise_tolerance);

            if (deep_itm && at_obstacle) {
                // Convert row i to Dirichlet constraint: u[i] = ψ[i]
                // Jacobian row: [0, 1, 0] (identity row)
                // RHS: ψ[i] (intrinsic value)
                auto jac = workspace_.jacobian();
                if (i > 0) jac.lower()[i-1] = 0.0;  // Zero lower diagonal
                jac.diag()[i] = 1.0;                 // Set diagonal to 1
                if (i < n_ - 1) jac.upper()[i] = 0.0; // Zero upper diagonal
                rhs_with_bc[i] = psi[i];                 // RHS = intrinsic value
            }
        }

        // ═══════════════════════════════════════════════════════════════════════
        // STEP 2: Solve the LCP using Projected Thomas (Brennan-Schwartz)
        // ═══════════════════════════════════════════════════════════════════════
        // Solve: A·u = rhs  subject to u ≥ ψ
        //
        // The RHS vector now contains three types of entries:
        //   1. Interior nodes: rhs[i] = u_old[i] + w·L(u_old)[i]  (TR-BDF2 formula)
        //   2. Dirichlet boundaries: rhs[0] = g(t)  (FIX #1 applied)
        //   3. Locked deep ITM: rhs[i] = ψ[i]  (FIX #2 applied)
        //
        // The Jacobian matrix A has three types of rows:
        //   1. Interior nodes: A[i] = I - coeff_dt·∂L/∂u  (standard TR-BDF2)
        //   2. Dirichlet boundaries: A[0] = [0, 1, 0]  (identity row)
        //   3. Locked deep ITM: A[i] = [0, 1, 0]  (identity row, FIX #2)
        //
        // Projected Thomas algorithm:
        //   - Forward elimination: identical to standard Thomas (build c', d' arrays)
        //   - Backward substitution: u[i] = max(unconstrained, ψ[i]) at each step
        // This couples the obstacle constraint u ≥ ψ with the tridiagonal structure,
        // ensuring the constraint is satisfied without iteration.
        //
        // CRITICAL: We pass u (solution), not δu (correction)!
        //   Newton would solve: J·δu = -F, then u ← u + δu
        //   Projected Thomas solves: A·u = rhs directly (u is the solution)
        auto result = solve_thomas_projected<double>(
            workspace_.jacobian(),
            rhs_with_bc,  // Corrected RHS (FIX #1 and #2 applied)
            psi,          // Obstacle constraint ψ(x,t)
            u,            // OUTPUT: solution u (not correction δu)
            workspace_.tridiag_workspace()
        );

        if (!result.ok()) {
            return std::unexpected(SolverError{
                .code = SolverErrorCode::LinearSolveFailure,
                .iterations = 1,
                .residual = std::numeric_limits<double>::infinity()
            });
        }

        // No update step needed - u already contains the solution from Projected Thomas
        // (Unlike Newton where we'd do: u ← u + δu)

        // Re-apply boundary conditions to ensure exact satisfaction
        apply_boundary_conditions(u, t);

        return {};
    }

    std::expected<void, SolverError> solve_implicit_stage_newton(
            double t, double coeff_dt,
            std::span<double> u,
            std::span<const double> rhs) {
        const double eps = config_.jacobian_fd_epsilon;

        // Apply BCs to initial guess
        apply_boundary_conditions(u, t);

        // Quasi-Newton: Build Jacobian once and reuse
        build_jacobian(t, coeff_dt, u, eps);

        // Copy initial guess
        std::copy(u.begin(), u.end(), workspace_.newton_u_old().begin());

        // Newton iteration
        for (size_t iter = 0; iter < config_.max_iter; ++iter) {
            // Evaluate L(u)
            apply_spatial_operator(t, u, workspace_.lu());

            // Compute residual: F(u) = u - rhs - coeff_dt·L(u)
            compute_residual(u, coeff_dt, workspace_.lu(), rhs,
                           workspace_.residual());

            // CRITICAL FIX: Pass u explicitly to avoid reading stale workspace
            apply_bc_to_residual(workspace_.residual(), u, t);

            // Newton method: Solve J·δu = -F(u), then update u ← u + δu
            // Negate residual for RHS
            for (size_t i = 0; i < n_; ++i) {
                workspace_.residual()[i] = -workspace_.residual()[i];
            }

            // Solve J·δu = -F(u) using Thomas algorithm
            auto result = solve_thomas<double>(
                workspace_.jacobian(),
                workspace_.residual(),
                workspace_.delta_u(),
                workspace_.tridiag_workspace()
            );

            if (!result.ok()) {
                return std::unexpected(SolverError{
                    .code = SolverErrorCode::LinearSolveFailure,
                    .iterations = iter,
                    .residual = std::numeric_limits<double>::infinity()
                });
            }

            // Update: u ← u + δu
            for (size_t i = 0; i < n_; ++i) {
                u[i] += workspace_.delta_u()[i];
            }

            // Apply boundary conditions BEFORE obstacle projection
            // This ensures boundary values are set before enforcing complementarity
            apply_boundary_conditions(u, t);

            // Apply obstacle projection AFTER boundary conditions
            // This ensures complementarity: u ≥ ψ
            apply_obstacle(t, u);

            // Check convergence via step delta
            double error = compute_step_delta_error(u, workspace_.newton_u_old());

            if (error < config_.tolerance) {
                return {};
            }

            // Prepare for next iteration
            std::copy(u.begin(), u.end(), workspace_.newton_u_old().begin());
        }

        return std::unexpected(SolverError{
            .code = SolverErrorCode::ConvergenceFailure,
            .iterations = config_.max_iter,
            .residual = compute_step_delta_error(u, workspace_.newton_u_old())
        });
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
            double g = left_bc.value(t, grid_->x()[0]);
            residual[0] = u[0] - g;  // u - g (we want u = g, so F = u - g = 0)
        }

        // Right boundary
        using RightBCType = std::remove_cvref_t<decltype(right_bc)>;
        if constexpr (std::is_same_v<bc::boundary_tag_t<RightBCType>, bc::dirichlet_tag>) {
            double g = right_bc.value(t, grid_->x()[n_ - 1]);
            residual[n_ - 1] = u[n_ - 1] - g;  // u - g
        }
    }

    void build_jacobian(double t, double coeff_dt,
                       std::span<const double> u, double eps) {
        // Get spatial operator from derived class via CRTP (use const auto& to avoid copies!)
        const auto& spatial_op = derived().spatial_operator();
        using SpatialOpType = std::remove_cvref_t<decltype(spatial_op)>;

        auto jac = workspace_.jacobian();

        // Dispatch to analytical or finite-difference Jacobian
        if constexpr (HasAnalyticalJacobian<SpatialOpType>) {
            // Analytical Jacobian (O(n) - fast path)
            spatial_op.assemble_jacobian(t, coeff_dt, jac);
        } else {
            // Finite-difference Jacobian (O(n²) - fallback for unsupported operators)
            build_jacobian_finite_difference(t, coeff_dt, u, eps, jac);
        }

        // Boundary rows (same for both methods)
        build_jacobian_boundaries(t, coeff_dt, u, eps, jac);
    }

    /// Finite-difference Jacobian (fallback for unsupported operators)
    ///
    /// This is the original O(n²) implementation using finite differences.
    /// Used when spatial operator doesn't provide analytical Jacobian.
    void build_jacobian_finite_difference(double t, double coeff_dt,
                                          std::span<const double> u, double eps,
                                          TridiagonalMatrixView jac) {
        // Initialize u_perturb and compute baseline L(u)
        std::copy(u.begin(), u.end(), workspace_.u_stage().begin());
        apply_spatial_operator(t, u, workspace_.lu());

        // Interior points: tridiagonal structure via finite differences
        // J = ∂F/∂u where F(u) = u - rhs - coeff_dt·L(u)
        // So ∂F/∂u = I - coeff_dt·∂L/∂u
        for (size_t i = 1; i < n_ - 1; ++i) {
            // Diagonal: ∂F/∂u_i = 1 - coeff_dt·∂L_i/∂u_i
            workspace_.u_stage()[i] = u[i] + eps;
            apply_spatial_operator(t, workspace_.u_stage(), workspace_.reserved1());
            double dLi_dui = (workspace_.reserved1()[i] - workspace_.lu()[i]) / eps;
            jac.diag()[i] = 1.0 - coeff_dt * dLi_dui;
            workspace_.u_stage()[i] = u[i];

            // Lower diagonal: ∂F_i/∂u_{i-1} = -coeff_dt·∂L_i/∂u_{i-1}
            workspace_.u_stage()[i - 1] = u[i - 1] + eps;
            apply_spatial_operator(t, workspace_.u_stage(), workspace_.reserved1());
            double dLi_duim1 = (workspace_.reserved1()[i] - workspace_.lu()[i]) / eps;
            jac.lower()[i - 1] = -coeff_dt * dLi_duim1;
            workspace_.u_stage()[i - 1] = u[i - 1];

            // Upper diagonal: ∂F_i/∂u_{i+1} = -coeff_dt·∂L_i/∂u_{i+1}
            workspace_.u_stage()[i + 1] = u[i + 1] + eps;
            apply_spatial_operator(t, workspace_.u_stage(), workspace_.reserved1());
            double dLi_duip1 = (workspace_.reserved1()[i] - workspace_.lu()[i]) / eps;
            jac.upper()[i] = -coeff_dt * dLi_duip1;
            workspace_.u_stage()[i + 1] = u[i + 1];
        }
    }

    void build_jacobian_boundaries(double t, double coeff_dt,
                                   std::span<const double> u, double eps,
                                   TridiagonalMatrixView jac) {
        // Get BCs from derived class via CRTP (use const auto& to avoid copies!)
        const auto& left_bc = derived().left_boundary();
        const auto& right_bc = derived().right_boundary();
        using LeftBCType = std::remove_cvref_t<decltype(left_bc)>;
        using RightBCType = std::remove_cvref_t<decltype(right_bc)>;

        // Left boundary
        if constexpr (std::is_same_v<bc::boundary_tag_t<LeftBCType>, bc::dirichlet_tag>) {
            // For Dirichlet: F(u) = u - g, so ∂F/∂u = 1
            jac.diag()[0] = 1.0;
            jac.upper()[0] = 0.0;
        } else if constexpr (std::is_same_v<bc::boundary_tag_t<LeftBCType>, bc::neumann_tag>) {
            // For Neumann: F(u) = u - rhs - coeff_dt·L(u)
            workspace_.u_stage()[0] = u[0] + eps;
            apply_spatial_operator(t, workspace_.u_stage(), workspace_.reserved1());
            double dL0_du0 = (workspace_.reserved1()[0] - workspace_.lu()[0]) / eps;
            jac.diag()[0] = 1.0 - coeff_dt * dL0_du0;
            workspace_.u_stage()[0] = u[0];

            workspace_.u_stage()[1] = u[1] + eps;
            apply_spatial_operator(t, workspace_.u_stage(), workspace_.reserved1());
            double dL0_du1 = (workspace_.reserved1()[0] - workspace_.lu()[0]) / eps;
            jac.upper()[0] = -coeff_dt * dL0_du1;
            workspace_.u_stage()[1] = u[1];
        }

        // Right boundary
        if constexpr (std::is_same_v<bc::boundary_tag_t<RightBCType>, bc::dirichlet_tag>) {
            // For Dirichlet: F(u) = u - g, so ∂F/∂u = 1
            jac.diag()[n_ - 1] = 1.0;
        } else if constexpr (std::is_same_v<bc::boundary_tag_t<RightBCType>, bc::neumann_tag>) {
            // For Neumann: F(u) = u - rhs - coeff_dt·L(u)
            size_t i = n_ - 1;
            workspace_.u_stage()[i] = u[i] + eps;
            apply_spatial_operator(t, workspace_.u_stage(), workspace_.reserved1());
            double dLi_dui = (workspace_.reserved1()[i] - workspace_.lu()[i]) / eps;
            jac.diag()[i] = 1.0 - coeff_dt * dLi_dui;
            workspace_.u_stage()[i] = u[i];
        }
    }

};

}  // namespace mango
