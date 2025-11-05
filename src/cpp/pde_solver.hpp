#pragma once

#include "grid.hpp"
#include "workspace.hpp"
#include "boundary_conditions.hpp"
#include "time_domain.hpp"
#include "trbdf2_config.hpp"
#include "tridiagonal_solver.hpp"
#include "newton_workspace.hpp"
#include "root_finding.hpp"
#include "snapshot.hpp"
#include <span>
#include <vector>
#include <functional>
#include <optional>
#include <cmath>
#include <algorithm>
#include <limits>

namespace mango {

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

/// PDE Solver with TR-BDF2 time stepping and cache blocking
///
/// Solves PDEs of the form: ∂u/∂t = L(u, x, t)
/// where L is a spatial operator (e.g., diffusion, advection, reaction)
///
/// Time stepping uses TR-BDF2 (Two-stage Runge-Kutta with BDF2):
/// - Stage 1: Trapezoidal rule to t_n + γ·dt
/// - Stage 2: BDF2 from t_n to t_n+1
/// where γ = 2 - √2 for L-stability
///
/// Cache blocking is automatically applied for large grids (n ≥ threshold)
/// to improve cache locality and reduce memory bandwidth.
///
/// @tparam BoundaryL Left boundary condition type
/// @tparam BoundaryR Right boundary condition type
/// @tparam SpatialOp Spatial operator type
template<typename BoundaryL, typename BoundaryR, typename SpatialOp>
class PDESolver {
public:
    /// Constructor
    ///
    /// @param grid Spatial grid (x coordinates)
    /// @param time Time domain configuration
    /// @param config TR-BDF2 configuration
    /// @param root_config Root-finding configuration for Newton solver
    /// @param left_bc Left boundary condition
    /// @param right_bc Right boundary condition
    /// @param spatial_op Spatial operator L(u)
    /// @param obstacle Optional obstacle condition ψ(x,t) for u ≥ ψ constraint
    PDESolver(std::span<const double> grid,
              const TimeDomain& time,
              const TRBDF2Config& config,
              const RootFindingConfig& root_config,
              const BoundaryL& left_bc,
              const BoundaryR& right_bc,
              const SpatialOp& spatial_op,
              std::optional<ObstacleCallback> obstacle = std::nullopt)
        : grid_(grid)
        , time_(time)
        , config_(config)
        , root_config_(root_config)
        , left_bc_(left_bc)
        , right_bc_(right_bc)
        , spatial_op_(spatial_op)
        , obstacle_(std::move(obstacle))
        , n_(grid.size())
        , workspace_(n_, grid, config_.cache_blocking_threshold)
        , u_current_(n_)
        , u_old_(n_)
        , rhs_(n_)
        , newton_ws_(n_, workspace_)
    {
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
    /// @return true if converged at all time steps, false otherwise
    bool solve() {
        double t = time_.t_start();
        const double dt = time_.dt();

        for (size_t step = 0; step < time_.n_steps(); ++step) {
            double t_old = t;

            // Store u^n for TR-BDF2
            std::copy(u_current_.begin(), u_current_.end(), u_old_.begin());

            // Stage 1: Trapezoidal rule to t_n + γ·dt
            double t_stage1 = t + config_.gamma * dt;
            bool stage1_ok = solve_stage1(t, t_stage1, dt);
            if (!stage1_ok) {
                return false;  // Convergence failure
            }

            // Stage 2: BDF2 from t_n to t_n+1
            double t_next = t + dt;
            bool stage2_ok = solve_stage2(t_stage1, t_next, dt);
            if (!stage2_ok) {
                return false;  // Convergence failure
            }

            // Update time
            t = t_next;

            // Process temporal events AFTER completing the step
            process_temporal_events(t_old, t_next, step);

            // Process snapshots (CHANGED: pass step index)
            process_snapshots(step, t);
        }

        return true;
    }

    /// Get current solution
    std::span<const double> solution() const {
        return std::span{u_current_};
    }

    /// Check if obstacle condition is present
    bool has_obstacle() const {
        return obstacle_.has_value();
    }

    /// Register snapshot collection at specific step index
    ///
    /// @param step_index Step number (0-based) to collect snapshot
    /// @param user_index User-provided index for matching
    /// @param collector Callback to receive snapshot (must outlive solver)
    void register_snapshot(size_t step_index, size_t user_index, SnapshotCollector* collector) {
        snapshot_requests_.push_back({step_index, user_index, collector});
        // Sort by step index for efficient lookup
        std::sort(snapshot_requests_.begin(), snapshot_requests_.end(),
                 [](const auto& a, const auto& b) { return a.step_index < b.step_index; });
        next_snapshot_idx_ = 0;
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

private:
    // Grid and configuration
    std::span<const double> grid_;
    TimeDomain time_;
    TRBDF2Config config_;
    RootFindingConfig root_config_;
    BoundaryL left_bc_;
    BoundaryR right_bc_;
    SpatialOp spatial_op_;
    std::optional<ObstacleCallback> obstacle_;

    // Grid size
    size_t n_;

    // Workspace for cache blocking
    WorkspaceStorage workspace_;

    // Solution storage
    std::vector<double> u_current_;  // u^{n+1} or u^{n+γ}
    std::vector<double> u_old_;      // u^n
    std::vector<double> rhs_;        // n: RHS vector for stages

    // Newton workspace (for implicit stage solving)
    NewtonWorkspace newton_ws_;

    // Snapshot collection
    struct SnapshotRequest {
        size_t step_index;        // CHANGED: use step index not time
        size_t user_index;
        SnapshotCollector* collector;
    };
    std::vector<SnapshotRequest> snapshot_requests_;
    size_t next_snapshot_idx_ = 0;

    // Temporal event system
    std::vector<TemporalEvent> events_;
    size_t next_event_idx_ = 0;

    // Workspace for derivatives
    std::vector<double> du_dx_;
    std::vector<double> d2u_dx2_;

    /// Process snapshots at current step index
    void process_snapshots(size_t step_idx, double t_current) {
        while (next_snapshot_idx_ < snapshot_requests_.size()) {
            const auto& req = snapshot_requests_[next_snapshot_idx_];

            // Check if this step index matches
            if (req.step_index > step_idx) {
                break;  // Future snapshot
            }

            if (req.step_index != step_idx) {
                ++next_snapshot_idx_;  // Skip missed snapshot
                continue;
            }

            // Allocate derivative storage on first use
            if (du_dx_.empty()) {
                du_dx_.resize(n_);
                d2u_dx2_.resize(n_);
            }

            // Compute derivatives using PDE operator
            spatial_op_.compute_first_derivative(grid_, std::span{u_current_},
                                                std::span{du_dx_}, workspace_.dx());
            spatial_op_.compute_second_derivative(grid_, std::span{u_current_},
                                                  std::span{d2u_dx2_}, workspace_.dx());

            // Build snapshot
            Snapshot snapshot{
                .time = t_current,
                .user_index = req.user_index,
                .spatial_grid = grid_,
                .dx = workspace_.dx(),
                .solution = std::span{u_current_},
                .spatial_operator = workspace_.lu(),
                .first_derivative = std::span{du_dx_},
                .second_derivative = std::span{d2u_dx2_},
                .problem_params = nullptr
            };

            // Call collector
            req.collector->collect(snapshot);

            ++next_snapshot_idx_;
        }
    }

    /// Process temporal events in time interval (t_old, t_new]
    ///
    /// Events are applied AFTER the TR-BDF2 step completes.
    /// This ensures proper ordering: PDE evolution happens first,
    /// then events modify the solution (e.g., dividend jumps).
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

        auto psi = workspace_.psi_buffer();
        (*obstacle_)(t, grid_, psi);

        // Project: u[i] = max(u[i], psi[i])
        for (size_t i = 0; i < u.size(); ++i) {
            if (u[i] < psi[i]) {
                u[i] = psi[i];
            }
        }
    }

    /// Apply boundary conditions
    void apply_boundary_conditions(std::span<double> u, double t) {
        auto dx_span = workspace_.dx();

        // Left boundary
        double x_left = grid_[0];
        double dx_left = (n_ > 1) ? dx_span[0] : 1.0;
        double u_interior_left = (n_ > 1) ? u[1] : 0.0;
        left_bc_.apply(u[0], x_left, t, dx_left, u_interior_left, 0.0, bc::BoundarySide::Left);

        // Right boundary
        double x_right = grid_[n_ - 1];
        double dx_right = (n_ > 1) ? dx_span[n_ - 2] : 1.0;
        double u_interior_right = (n_ > 1) ? u[n_ - 2] : 0.0;
        right_bc_.apply(u[n_ - 1], x_right, t, dx_right, u_interior_right, 0.0, bc::BoundarySide::Right);
    }

    /// Apply spatial operator with cache-blocking for large grids
    void apply_operator_with_blocking(double t,
                                      std::span<const double> u,
                                      std::span<double> Lu) {
        const size_t n = grid_.size();

        // Small grid: use full-array path (no blocking overhead)
        if (workspace_.cache_config().n_blocks == 1) {
            spatial_op_(t, grid_, u, Lu, workspace_.dx());
            return;
        }

        // Large grid: blocked evaluation
        for (size_t block = 0; block < workspace_.cache_config().n_blocks; ++block) {
            auto [interior_start, interior_end] =
                workspace_.get_block_interior_range(block);

            // Skip boundary-only blocks
            if (interior_start >= interior_end) continue;

            // Compute halo sizes (clamped at global boundaries)
            const size_t halo_left = std::min(workspace_.cache_config().overlap,
                                             interior_start);
            const size_t halo_right = std::min(workspace_.cache_config().overlap,
                                              n - interior_end);
            const size_t interior_count = interior_end - interior_start;

            // Build spans with halos
            auto x_halo = std::span{grid_.data() + interior_start - halo_left,
                                   interior_count + halo_left + halo_right};
            auto u_halo = std::span{u.data() + interior_start - halo_left,
                                   interior_count + halo_left + halo_right};
            auto lu_out = std::span{Lu.data() + interior_start, interior_count};

            // Call block-aware operator
            spatial_op_.apply_block(t, interior_start, halo_left, halo_right,
                                   x_halo, u_halo, lu_out, workspace_.dx());
        }

        // Zero boundary values (BCs will override after)
        Lu[0] = Lu[n-1] = 0.0;
    }

    /// TR-BDF2 Stage 1: Trapezoidal rule
    ///
    /// u^{n+γ} = u^n + (γ·dt/2) · [L(u^n) + L(u^{n+γ})]
    ///
    /// Solved via Newton-Raphson iteration
    bool solve_stage1(double t_n, double t_stage, double dt) {
        const double w1 = config_.stage1_weight(dt);  // γ·dt/2

        // Compute L(u^n)
        apply_operator_with_blocking(t_n, std::span{u_old_}, workspace_.lu());

        // RHS = u^n + w1·L(u^n)
        for (size_t i = 0; i < n_; ++i) {
            rhs_[i] = u_old_[i] + w1 * workspace_.lu()[i];
        }

        // Initial guess: u* = u^n
        std::copy(u_old_.begin(), u_old_.end(), u_current_.begin());

        // Solve implicit stage
        auto result = solve_implicit_stage(t_stage, w1,
                                          std::span{u_current_},
                                          std::span{rhs_});

        return result.converged;
    }

    /// TR-BDF2 Stage 2: BDF2
    ///
    /// Standard TR-BDF2 formulation (Ascher, Ruuth, Wetton 1995):
    /// u^{n+1} - [(1-γ)·dt/(2-γ)]·L(u^{n+1}) = [1/(γ(2-γ))]·u^{n+γ} - [(1-γ)²/(γ(2-γ))]·u^n
    ///
    /// Solved via Newton-Raphson iteration
    bool solve_stage2([[maybe_unused]] double t_stage, double t_next, double dt) {
        const double gamma = config_.gamma;
        const double one_minus_gamma = 1.0 - gamma;
        const double two_minus_gamma = 2.0 - gamma;
        const double denom = gamma * two_minus_gamma;

        // Correct BDF2 coefficients (Ascher, Ruuth, Wetton 1995)
        const double alpha = 1.0 / denom;  // Coefficient for u^{n+γ}
        const double beta = -(one_minus_gamma * one_minus_gamma) / denom;  // Coefficient for u^n
        const double w2 = config_.stage2_weight(dt);  // (1-γ)·dt/(2-γ)

        // RHS = alpha·u^{n+γ} + beta·u^n (u_current_ currently holds u^{n+γ})
        for (size_t i = 0; i < n_; ++i) {
            rhs_[i] = alpha * u_current_[i] + beta * u_old_[i];
        }

        // Initial guess: u^{n+1} = u* (already in u_current_)
        // (No need to copy, u_current_ already has u^{n+γ})

        // Solve implicit stage
        auto result = solve_implicit_stage(t_next, w2,
                                          std::span{u_current_},
                                          std::span{rhs_});

        return result.converged;
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
    // For general root-finding needs, see root_finding.hpp for the abstraction
    // layer (RootFindingConfig, RootFindingResult). A truly general Newton
    // solver would use function pointers/callables rather than template
    // parameters for boundary conditions and spatial operators.
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
    RootFindingResult solve_implicit_stage(double t, double coeff_dt,
                                           std::span<double> u,
                                           std::span<const double> rhs) {
        const double eps = root_config_.jacobian_fd_epsilon;

        // Apply BCs to initial guess
        apply_boundary_conditions(u, t);

        // Quasi-Newton: Build Jacobian once and reuse
        build_jacobian(t, coeff_dt, u, eps);

        // Copy initial guess
        std::copy(u.begin(), u.end(), newton_ws_.u_old().begin());

        // Newton iteration
        for (size_t iter = 0; iter < root_config_.max_iter; ++iter) {
            // Evaluate L(u)
            apply_operator_with_blocking(t, u, workspace_.lu());

            // Compute residual: F(u) = u - rhs - coeff_dt·L(u)
            compute_residual(u, coeff_dt, workspace_.lu(), rhs,
                           newton_ws_.residual());

            // CRITICAL FIX: Pass u explicitly to avoid reading stale workspace
            apply_bc_to_residual(newton_ws_.residual(), u, t);

            // Newton method: Solve J·δu = -F(u), then update u ← u + δu
            // Negate residual for RHS
            for (size_t i = 0; i < n_; ++i) {
                newton_ws_.residual()[i] = -newton_ws_.residual()[i];
            }

            // Solve J·δu = -F(u) using Thomas algorithm
            bool success = solve_tridiagonal(
                newton_ws_.jacobian_lower(),
                newton_ws_.jacobian_diag(),
                newton_ws_.jacobian_upper(),
                newton_ws_.residual(),
                newton_ws_.delta_u(),
                newton_ws_.tridiag_workspace()
            );

            if (!success) {
                return {false, iter, std::numeric_limits<double>::infinity(),
                       "Singular Jacobian", std::nullopt};
            }

            // Update: u ← u + δu
            for (size_t i = 0; i < n_; ++i) {
                u[i] += newton_ws_.delta_u()[i];
            }

            // Apply obstacle projection BEFORE boundary conditions
            // This ensures complementarity: u ≥ ψ
            apply_obstacle(t, u);

            apply_boundary_conditions(u, t);

            // Check convergence via step delta
            double error = compute_step_delta_error(u, newton_ws_.u_old());

            if (error < root_config_.tolerance) {
                return {true, iter + 1, error, std::nullopt, std::nullopt};
            }

            // Prepare for next iteration
            std::copy(u.begin(), u.end(), newton_ws_.u_old().begin());
        }

        return {false, root_config_.max_iter,
               compute_step_delta_error(u, newton_ws_.u_old()),
               "Max iterations reached", std::nullopt};
    }

    void compute_residual(std::span<const double> u, double coeff_dt,
                         std::span<const double> Lu,
                         std::span<const double> rhs,
                         std::span<double> residual) {
        // F(u) = u - rhs - coeff_dt·L(u) = 0
        // We want to solve u = rhs + coeff_dt·L(u)
        for (size_t i = 0; i < n_; ++i) {
            residual[i] = u[i] - rhs[i] - coeff_dt * Lu[i];
        }
    }

    double compute_step_delta_error(std::span<const double> u_new,
                                    std::span<const double> u_old) {
        double sum_sq_error = 0.0;
        double sum_sq_norm = 0.0;
        for (size_t i = 0; i < n_; ++i) {
            double diff = u_new[i] - u_old[i];
            sum_sq_error += diff * diff;
            sum_sq_norm += u_new[i] * u_new[i];
        }
        double rms_error = std::sqrt(sum_sq_error / n_);
        double rms_norm = std::sqrt(sum_sq_norm / n_);
        const double epsilon = 1e-12;
        return (rms_norm > epsilon) ? rms_error / (rms_norm + epsilon) : rms_error;
    }

    void apply_bc_to_residual(std::span<double> residual,
                              std::span<const double> u,
                              double t) {
        // For Dirichlet BC: F(u) = u - g, so residual = u - g
        // Left boundary
        if constexpr (std::is_same_v<bc::boundary_tag_t<BoundaryL>, bc::dirichlet_tag>) {
            double g = left_bc_.value(t, grid_[0]);
            residual[0] = u[0] - g;  // u - g (we want u = g, so F = u - g = 0)
        }

        // Right boundary
        if constexpr (std::is_same_v<bc::boundary_tag_t<BoundaryR>, bc::dirichlet_tag>) {
            double g = right_bc_.value(t, grid_[n_ - 1]);
            residual[n_ - 1] = u[n_ - 1] - g;  // u - g
        }
    }

    void build_jacobian(double t, double coeff_dt,
                       std::span<const double> u, double eps) {
        // Initialize u_perturb and compute baseline L(u)
        std::copy(u.begin(), u.end(), newton_ws_.u_perturb().begin());
        apply_operator_with_blocking(t, u, workspace_.lu());

        // Interior points: tridiagonal structure via finite differences
        // J = ∂F/∂u where F(u) = u - rhs - coeff_dt·L(u)
        // So ∂F/∂u = I - coeff_dt·∂L/∂u
        for (size_t i = 1; i < n_ - 1; ++i) {
            // Diagonal: ∂F/∂u_i = 1 - coeff_dt·∂L_i/∂u_i
            newton_ws_.u_perturb()[i] = u[i] + eps;
            apply_operator_with_blocking(t, newton_ws_.u_perturb(), newton_ws_.Lu_perturb());
            double dLi_dui = (newton_ws_.Lu_perturb()[i] - workspace_.lu()[i]) / eps;
            newton_ws_.jacobian_diag()[i] = 1.0 - coeff_dt * dLi_dui;
            newton_ws_.u_perturb()[i] = u[i];

            // Lower diagonal: ∂F_i/∂u_{i-1} = -coeff_dt·∂L_i/∂u_{i-1}
            newton_ws_.u_perturb()[i - 1] = u[i - 1] + eps;
            apply_operator_with_blocking(t, newton_ws_.u_perturb(), newton_ws_.Lu_perturb());
            double dLi_duim1 = (newton_ws_.Lu_perturb()[i] - workspace_.lu()[i]) / eps;
            newton_ws_.jacobian_lower()[i - 1] = -coeff_dt * dLi_duim1;
            newton_ws_.u_perturb()[i - 1] = u[i - 1];

            // Upper diagonal: ∂F_i/∂u_{i+1} = -coeff_dt·∂L_i/∂u_{i+1}
            newton_ws_.u_perturb()[i + 1] = u[i + 1] + eps;
            apply_operator_with_blocking(t, newton_ws_.u_perturb(), newton_ws_.Lu_perturb());
            double dLi_duip1 = (newton_ws_.Lu_perturb()[i] - workspace_.lu()[i]) / eps;
            newton_ws_.jacobian_upper()[i] = -coeff_dt * dLi_duip1;
            newton_ws_.u_perturb()[i + 1] = u[i + 1];
        }

        // Boundary rows
        build_jacobian_boundaries(t, coeff_dt, u, eps);
    }

    void build_jacobian_boundaries(double t, double coeff_dt,
                                   std::span<const double> u, double eps) {
        // Left boundary
        if constexpr (std::is_same_v<bc::boundary_tag_t<BoundaryL>, bc::dirichlet_tag>) {
            // For Dirichlet: F(u) = u - g, so ∂F/∂u = 1
            newton_ws_.jacobian_diag()[0] = 1.0;
            newton_ws_.jacobian_upper()[0] = 0.0;
        } else if constexpr (std::is_same_v<bc::boundary_tag_t<BoundaryL>, bc::neumann_tag>) {
            // For Neumann: F(u) = u - rhs - coeff_dt·L(u)
            newton_ws_.u_perturb()[0] = u[0] + eps;
            apply_operator_with_blocking(t, newton_ws_.u_perturb(), newton_ws_.Lu_perturb());
            double dL0_du0 = (newton_ws_.Lu_perturb()[0] - workspace_.lu()[0]) / eps;
            newton_ws_.jacobian_diag()[0] = 1.0 - coeff_dt * dL0_du0;
            newton_ws_.u_perturb()[0] = u[0];

            newton_ws_.u_perturb()[1] = u[1] + eps;
            apply_operator_with_blocking(t, newton_ws_.u_perturb(), newton_ws_.Lu_perturb());
            double dL0_du1 = (newton_ws_.Lu_perturb()[0] - workspace_.lu()[0]) / eps;
            newton_ws_.jacobian_upper()[0] = -coeff_dt * dL0_du1;
            newton_ws_.u_perturb()[1] = u[1];
        }

        // Right boundary
        if constexpr (std::is_same_v<bc::boundary_tag_t<BoundaryR>, bc::dirichlet_tag>) {
            // For Dirichlet: F(u) = u - g, so ∂F/∂u = 1
            newton_ws_.jacobian_diag()[n_ - 1] = 1.0;
        } else if constexpr (std::is_same_v<bc::boundary_tag_t<BoundaryR>, bc::neumann_tag>) {
            // For Neumann: F(u) = u - rhs - coeff_dt·L(u)
            size_t i = n_ - 1;
            newton_ws_.u_perturb()[i] = u[i] + eps;
            apply_operator_with_blocking(t, newton_ws_.u_perturb(), newton_ws_.Lu_perturb());
            double dLi_dui = (newton_ws_.Lu_perturb()[i] - workspace_.lu()[i]) / eps;
            newton_ws_.jacobian_diag()[i] = 1.0 - coeff_dt * dLi_dui;
            newton_ws_.u_perturb()[i] = u[i];
        }
    }

};

}  // namespace mango
