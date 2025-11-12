#pragma once

#include "src/pde/core/grid.hpp"
#include "src/pde/memory/pde_workspace.hpp"
#include "src/support/cpu/feature_detection.hpp"
#include "src/pde/operators/centered_difference_facade.hpp"
#include "src/pde/core/boundary_conditions.hpp"
#include "src/pde/core/time_domain.hpp"
#include "src/pde/core/trbdf2_config.hpp"
#include "src/pde/core/thomas_solver.hpp"
#include "src/pde/core/newton_workspace.hpp"
#include "src/pde/core/root_finding.hpp"
#include "src/option/snapshot.hpp"
#include "src/pde/core/jacobian_view.hpp"
#include "src/support/expected.hpp"
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
concept HasAnalyticalJacobian = requires(const SpatialOp op, double coeff_dt, JacobianView jac, std::optional<size_t> lane) {
    { op.assemble_jacobian(coeff_dt, jac, lane) } -> std::same_as<void>;
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
              SpatialOp spatial_op,  // Pass by value, move into member
              std::optional<ObstacleCallback> obstacle = std::nullopt,
              PDEWorkspace* external_workspace = nullptr)
        : grid_(grid)
        , time_(time)
        , config_(config)
        , root_config_(root_config)
        , left_bc_(left_bc)
        , right_bc_(right_bc)
        , spatial_op_(std::move(spatial_op))
        , obstacle_(std::move(obstacle))
        , n_(grid.size())
        , workspace_owner_(nullptr)
        , workspace_(nullptr)
        , u_current_(n_)
        , u_old_(n_)
        , rhs_(n_)
        , newton_ws_(n_, acquire_workspace(grid, external_workspace))
        , isa_target_(cpu::select_isa_target())
    {
        #ifndef NDEBUG
        std::cout << "PDESolver ISA target: " << cpu::isa_target_name(isa_target_) << "\n";
        #endif

        // Initialize grid information for legacy operators that need it
        // (e.g., LaplacianOperator) via set_grid() if present
        if constexpr (requires { spatial_op_.set_grid(grid, workspace_->dx()); }) {
            spatial_op_.set_grid(grid, workspace_->dx());
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

        // If batch mode, broadcast initial condition to all lanes
        if (workspace_->has_batch()) {
            const size_t n_lanes = workspace_->batch_width();
            for (size_t lane = 0; lane < n_lanes; ++lane) {
                auto u_lane = workspace_->u_lane(lane);
                std::copy(u_current_.begin(), u_current_.end(), u_lane.begin());
            }
        }
    }

    /// Solve PDE from t_start to t_end
    ///
    /// @return expected success or solver error diagnostic
    expected<void, SolverError> solve() {
        double t = time_.t_start();
        const double dt = time_.dt();

        for (size_t step = 0; step < time_.n_steps(); ++step) {
            double t_old = t;

            // Store u^n for TR-BDF2
            // In batch mode, copy to workspace per-lane buffers
            // In single-contract mode, use the single u_old_ buffer
            const bool is_batched = workspace_->has_batch();
            if (is_batched) {
                const size_t n_lanes = workspace_->batch_width();
                for (size_t lane = 0; lane < n_lanes; ++lane) {
                    auto u_lane = workspace_->u_lane(lane);
                    auto u_old_lane = workspace_->u_old_lane(lane);
                    std::copy(u_lane.begin(), u_lane.end(), u_old_lane.begin());
                }
            } else {
                std::copy(u_current_.begin(), u_current_.end(), u_old_.begin());
            }

            // Stage 1: Trapezoidal rule to t_n + γ·dt
            double t_stage1 = t + config_.gamma * dt;
            auto stage1_ok = solve_stage1(t, t_stage1, dt);
            if (!stage1_ok) {
                return unexpected(stage1_ok.error());
            }

            // Stage 2: BDF2 from t_n to t_n+1
            double t_next = t + dt;
            auto stage2_ok = solve_stage2(t_stage1, t_next, dt);
            if (!stage2_ok) {
                return unexpected(stage2_ok.error());
            }

            // Update time
            t = t_next;

            // Process temporal events AFTER completing the step
            process_temporal_events(t_old, t_next, step);

            // Process snapshots (CHANGED: pass step index)
            process_snapshots(step, t);
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

    /// Register snapshot collection at specific step index (single-contract mode)
    ///
    /// @param step_index Step number (0-based) to collect snapshot
    /// @param user_index User-provided index for matching
    /// @param collector Callback to receive snapshot (must outlive solver)
    void register_snapshot(size_t step_index, size_t user_index, SnapshotCollector* collector) {
        snapshot_requests_.push_back({step_index, user_index, collector, {}});
        // Sort by step index for efficient lookup
        std::sort(snapshot_requests_.begin(), snapshot_requests_.end(),
                 [](const auto& a, const auto& b) { return a.step_index < b.step_index; });
        next_snapshot_idx_ = 0;
    }

    /// Register per-lane snapshot collectors at specific step index (batch mode)
    ///
    /// @param step_index Step number (0-based) to collect snapshot
    /// @param user_index User-provided index for matching
    /// @param lane_collectors Vector of collectors (one per lane, must outlive solver)
    void register_snapshot_batch(size_t step_index, size_t user_index,
                                 std::vector<SnapshotCollector*> lane_collectors) {
        snapshot_requests_.push_back({step_index, user_index, nullptr, std::move(lane_collectors)});
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
    std::unique_ptr<PDEWorkspace> workspace_owner_;
    PDEWorkspace* workspace_;

    // Solution storage
    std::vector<double> u_current_;  // u^{n+1} or u^{n+γ}
    std::vector<double> u_old_;      // u^n
    std::vector<double> rhs_;        // n: RHS vector for stages

    // Newton workspace (for implicit stage solving)
    NewtonWorkspace newton_ws_;

    // ISA target for diagnostic logging (must be after newton_ws_)
    cpu::ISATarget isa_target_;

    // Snapshot collection
    struct SnapshotRequest {
        size_t step_index;        // CHANGED: use step index not time
        size_t user_index;
        SnapshotCollector* collector;  // Single-contract mode
        std::vector<SnapshotCollector*> lane_collectors;  // Batch mode (empty = single-contract)
    };
    std::vector<SnapshotRequest> snapshot_requests_;
    size_t next_snapshot_idx_ = 0;

    // Temporal event system
    std::vector<TemporalEvent> events_;
    size_t next_event_idx_ = 0;

    // Workspace for derivatives (single-contract mode)
    std::vector<double> du_dx_;
    std::vector<double> d2u_dx2_;

    // Per-lane derivative storage (batch mode)
    std::vector<std::vector<double>> du_dx_lanes_;
    std::vector<std::vector<double>> d2u_dx2_lanes_;

    PDEWorkspace& acquire_workspace(std::span<const double> grid, PDEWorkspace* external_workspace) {
        if (external_workspace) {
            workspace_ = external_workspace;
            return *workspace_;
        }
        workspace_owner_ = std::make_unique<PDEWorkspace>(n_, grid);
        workspace_ = workspace_owner_.get();
        return *workspace_;
    }

    /// Process snapshots at current step index
    void process_snapshots(size_t step_idx, double t_current) {
        const bool is_batched = workspace_->has_batch();

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

            if (is_batched && !req.lane_collectors.empty()) {
                // Batch mode: unpack per-lane snapshots
                const size_t n_lanes = workspace_->batch_width();

                // Validate collector count matches batch width
                if (req.lane_collectors.size() != n_lanes) {
                    // Programming error: mismatched collector count
                    assert(false && "Collector count must match batch_width");
                    ++next_snapshot_idx_;
                    continue;
                }

                // Allocate per-lane derivative storage on first use
                if (du_dx_lanes_.empty()) {
                    du_dx_lanes_.resize(n_lanes);
                    d2u_dx2_lanes_.resize(n_lanes);
                    for (size_t lane = 0; lane < n_lanes; ++lane) {
                        du_dx_lanes_[lane].resize(n_);
                        d2u_dx2_lanes_[lane].resize(n_);
                    }
                }

                // Process each lane's snapshot
                for (size_t lane = 0; lane < n_lanes; ++lane) {
                    // Get per-lane data from workspace
                    auto u_lane = workspace_->u_lane(lane);
                    auto lu_lane = workspace_->lu_lane(lane);

                    // Compute derivatives for this lane using the stencil
                    // NOTE: We need to compute derivatives from the per-lane solution
                    // Since spatial_op_ operates on batched data, we'll use the
                    // CenteredDifference stencil directly on per-lane data
                    spatial_op_.compute_first_derivative(u_lane, std::span{du_dx_lanes_[lane]});
                    spatial_op_.compute_second_derivative(u_lane, std::span{d2u_dx2_lanes_[lane]});

                    // Build snapshot for this lane
                    Snapshot snapshot{
                        .time = t_current,
                        .user_index = req.user_index,
                        .spatial_grid = grid_,
                        .dx = workspace_->dx(),
                        .solution = u_lane,
                        .spatial_operator = lu_lane,
                        .first_derivative = std::span{du_dx_lanes_[lane]},
                        .second_derivative = std::span{d2u_dx2_lanes_[lane]},
                        .problem_params = nullptr
                    };

                    // Call per-lane collector
                    req.lane_collectors[lane]->collect(snapshot);
                }
            } else {
                // Single-contract mode: original behavior
                // Allocate derivative storage on first use
                if (du_dx_.empty()) {
                    du_dx_.resize(n_);
                    d2u_dx2_.resize(n_);
                }

                // Compute derivatives using PDE operator
                spatial_op_.compute_first_derivative(std::span{u_current_}, std::span{du_dx_});
                spatial_op_.compute_second_derivative(std::span{u_current_}, std::span{d2u_dx2_});

                // Build snapshot
                Snapshot snapshot{
                    .time = t_current,
                    .user_index = req.user_index,
                    .spatial_grid = grid_,
                    .dx = workspace_->dx(),
                    .solution = std::span{u_current_},
                    .spatial_operator = workspace_->lu(),
                    .first_derivative = std::span{du_dx_},
                    .second_derivative = std::span{d2u_dx2_},
                    .problem_params = nullptr
                };

                // Call collector
                req.collector->collect(snapshot);
            }

            ++next_snapshot_idx_;
        }
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

        auto psi = workspace_->psi_buffer();
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
        auto dx_span = workspace_->dx();

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

    /// Apply spatial operator (single-pass evaluation)
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

        // Direct evaluation (no blocking)
        spatial_op_.apply(t, u, Lu);

        // Zero boundary values (BCs will override after)
        Lu[0] = Lu[n-1] = 0.0;
    }

    /// TR-BDF2 Stage 1: Trapezoidal rule
    ///
    /// u^{n+γ} = u^n + (γ·dt/2) · [L(u^n) + L(u^{n+γ})]
    ///
    /// Solved via Newton-Raphson iteration
    expected<void, SolverError> solve_stage1(double t_n, double t_stage, double dt) {
        const double w1 = config_.stage1_weight(dt);  // γ·dt/2
        const bool is_batched = workspace_->has_batch();

        if (is_batched) {
            // Batch mode: compute RHS per-lane from workspace buffers
            const size_t n_lanes = workspace_->batch_width();

            // Pack u_old into batch_slice for batched operator
            for (size_t lane = 0; lane < n_lanes; ++lane) {
                auto u_old_lane = workspace_->u_old_lane(lane);
                for (size_t i = 0; i < n_; ++i) {
                    workspace_->batch_slice()[i * n_lanes + lane] = u_old_lane[i];
                }
            }

            // Compute L(u^n) in batch (AoS → AoS)
            apply_operator_with_blocking_batch(t_n, workspace_->batch_slice(),
                                              workspace_->lu_batch(),
                                              n_lanes);

            // Scatter lu_batch → lu_lanes and compute RHS per lane
            for (size_t lane = 0; lane < n_lanes; ++lane) {
                auto u_old_lane = workspace_->u_old_lane(lane);
                auto rhs_lane = workspace_->rhs_lane(lane);
                auto u_lane = workspace_->u_lane(lane);

                // Scatter Lu from AoS to SoA
                for (size_t i = 0; i < n_; ++i) {
                    const double lu_i = workspace_->lu_batch()[i * n_lanes + lane];
                    // RHS = u^n + w1·L(u^n)
                    rhs_lane[i] = std::fma(w1, lu_i, u_old_lane[i]);
                }

                // Initial guess: u* = u^n
                std::copy(u_old_lane.begin(), u_old_lane.end(), u_lane.begin());
            }
        } else {
            // Single-contract mode: use single buffers
            // Compute L(u^n)
            apply_operator_with_blocking(t_n, std::span{u_old_}, workspace_->lu());

            // RHS = u^n + w1·L(u^n)
            // Use FMA for SAXPY-style loop
            for (size_t i = 0; i < n_; ++i) {
                rhs_[i] = std::fma(w1, workspace_->lu()[i], u_old_[i]);
            }

            // Initial guess: u* = u^n
            std::copy(u_old_.begin(), u_old_.end(), u_current_.begin());
        }

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

            return unexpected(error);
        }

        return {};
    }

    /// TR-BDF2 Stage 2: BDF2
    ///
    /// Standard TR-BDF2 formulation (Ascher, Ruuth, Wetton 1995):
    /// u^{n+1} - [(1-γ)·dt/(2-γ)]·L(u^{n+1}) = [1/(γ(2-γ))]·u^{n+γ} - [(1-γ)²/(γ(2-γ))]·u^n
    ///
    /// Solved via Newton-Raphson iteration
    expected<void, SolverError> solve_stage2([[maybe_unused]] double t_stage, double t_next, double dt) {
        const double gamma = config_.gamma;
        const double one_minus_gamma = 1.0 - gamma;
        const double two_minus_gamma = 2.0 - gamma;
        const double denom = gamma * two_minus_gamma;

        // Correct BDF2 coefficients (Ascher, Ruuth, Wetton 1995)
        const double alpha = 1.0 / denom;  // Coefficient for u^{n+γ}
        const double beta = -(one_minus_gamma * one_minus_gamma) / denom;  // Coefficient for u^n
        const double w2 = config_.stage2_weight(dt);  // (1-γ)·dt/(2-γ)
        const bool is_batched = workspace_->has_batch();

        if (is_batched) {
            // Batch mode: compute RHS per-lane
            // Note: u_lane contains u^{n+γ} from Stage 1
            const size_t n_lanes = workspace_->batch_width();
            for (size_t lane = 0; lane < n_lanes; ++lane) {
                auto u_lane = workspace_->u_lane(lane);          // u^{n+γ} from Stage 1
                auto u_old_lane = workspace_->u_old_lane(lane);  // u^n stored earlier
                auto rhs_lane = workspace_->rhs_lane(lane);

                // RHS = alpha·u^{n+γ} + beta·u^n
                for (size_t i = 0; i < n_; ++i) {
                    rhs_lane[i] = std::fma(alpha, u_lane[i], beta * u_old_lane[i]);
                }

                // Initial guess: u^{n+1} = u^{n+γ} (already in u_lane)
                // No copy needed
            }
        } else {
            // Single-contract mode: use single buffers
            // RHS = alpha·u^{n+γ} + beta·u^n (u_current_ currently holds u^{n+γ})
            // Use FMA: alpha*u_current[i] + beta*u_old[i]
            for (size_t i = 0; i < n_; ++i) {
                rhs_[i] = std::fma(alpha, u_current_[i], beta * u_old_[i]);
            }

            // Initial guess: u^{n+1} = u* (already in u_current_)
            // (No need to copy, u_current_ already has u^{n+γ})
        }

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

            return unexpected(error);
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

        // Hoist batch detection ONCE per stage
        const bool is_batched = workspace_->has_batch();
        const size_t n_lanes = is_batched ? workspace_->batch_width() : 1;

        // Apply BCs to initial guess
        apply_boundary_conditions(u, t);

        // Quasi-Newton: Build Jacobian once and reuse (single-contract only)
        // In batch mode, Jacobian is built per-lane inside the Newton loop
        if (!is_batched) {
            build_jacobian(t, coeff_dt, u, eps, std::nullopt);
        }

        // Copy initial guess
        std::copy(u.begin(), u.end(), newton_ws_.u_old().begin());

        // Track max error across all lanes (for final return)
        double max_error = 0.0;

        // Per-lane previous state for convergence checking
        std::vector<std::vector<double>> u_old_per_lane;
        if (is_batched) {
            u_old_per_lane.resize(n_lanes);
            for (size_t lane = 0; lane < n_lanes; ++lane) {
                auto u_lane = workspace_->u_lane(lane);
                u_old_per_lane[lane].assign(u_lane.begin(), u_lane.end());
            }
        } else {
            // Single-contract: use existing u_old buffer
            std::copy(u.begin(), u.end(), newton_ws_.u_old().begin());
        }

        // Newton iteration
        for (size_t iter = 0; iter < root_config_.max_iter; ++iter) {
            // Refresh AoS buffer from updated SoA lane buffers (after iteration N-1)
            if (is_batched) {
                workspace_->pack_to_batch_slice();  // SoA → AoS
            }

            // Batched or single-contract stencil
            if (is_batched) {
                apply_operator_with_blocking_batch(t, workspace_->batch_slice(),
                                                  workspace_->lu_batch(),
                                                  workspace_->batch_width());
                workspace_->scatter_from_batch_slice();  // AoS → SoA
            } else {
                apply_operator_with_blocking(t, u, workspace_->lu());
            }

            // Per-lane Newton machinery
            bool all_converged = true;
            max_error = 0.0;  // Reset for this iteration

            for (size_t lane = 0; lane < n_lanes; ++lane) {
                auto u_lane = is_batched ? workspace_->u_lane(lane) : u;
                auto lu_lane = is_batched ? workspace_->lu_lane(lane) : workspace_->lu();
                // Lane-aware RHS: each contract has its own RHS from TR-BDF2 staging
                auto rhs_lane = is_batched ? workspace_->rhs_lane(lane) : rhs;

                // Build per-lane Jacobian in batch mode
                if (is_batched) {
                    build_jacobian(t, coeff_dt, u_lane, eps, lane);
                }

                // Compute residual: F(u) = u - rhs - coeff_dt·L(u)
                compute_residual(u_lane, coeff_dt, lu_lane, rhs_lane,
                               newton_ws_.residual());

                // CRITICAL FIX: Pass u explicitly to avoid reading stale workspace
                apply_bc_to_residual(newton_ws_.residual(), u_lane, t);

                // Newton method: Solve J·δu = -F(u), then update u ← u + δu
                // Negate residual for RHS
                for (size_t i = 0; i < n_; ++i) {
                    newton_ws_.residual()[i] = -newton_ws_.residual()[i];
                }

                // Solve J·δu = -F(u) using Thomas algorithm
                auto result = solve_thomas<double>(
                    newton_ws_.jacobian_lower(),
                    newton_ws_.jacobian_diag(),
                    newton_ws_.jacobian_upper(),
                    newton_ws_.residual(),
                    newton_ws_.delta_u(),
                    newton_ws_.tridiag_workspace()
                );

                if (!result.ok()) {
                    return {false, iter, std::numeric_limits<double>::infinity(),
                           "Singular Jacobian", std::nullopt};
                }

                // Update: u ← u + δu
                for (size_t i = 0; i < n_; ++i) {
                    u_lane[i] += newton_ws_.delta_u()[i];
                }

                // Apply obstacle projection BEFORE boundary conditions
                apply_obstacle(t, u_lane);
                apply_boundary_conditions(u_lane, t);

                // Check convergence via step delta
                double error_lane = is_batched
                    ? compute_step_delta_error(u_lane, u_old_per_lane[lane])
                    : compute_step_delta_error(u_lane, newton_ws_.u_old());
                max_error = std::max(max_error, error_lane);
                all_converged &= (error_lane < root_config_.tolerance);
            }

            // Update old state for next Newton iteration
            if (is_batched) {
                for (size_t lane = 0; lane < n_lanes; ++lane) {
                    auto u_lane = workspace_->u_lane(lane);
                    u_old_per_lane[lane].assign(u_lane.begin(), u_lane.end());
                }
            } else {
                std::copy(u.begin(), u.end(), newton_ws_.u_old().begin());
            }

            // Exit when ALL lanes converged
            if (all_converged) {
                return {true, iter + 1, max_error, std::nullopt, std::nullopt};
            }
        }

        return {false, root_config_.max_iter, max_error,
               "Max iterations reached", std::nullopt};
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
                       std::span<const double> u, double eps,
                       std::optional<size_t> lane) {
        // Dispatch to analytical or finite-difference Jacobian
        if constexpr (HasAnalyticalJacobian<SpatialOp>) {
            // Analytical Jacobian (O(n) - fast path)
            JacobianView jac(newton_ws_.jacobian_lower(),
                           newton_ws_.jacobian_diag(),
                           newton_ws_.jacobian_upper());
            spatial_op_.assemble_jacobian(coeff_dt, jac, lane);
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
        std::copy(u.begin(), u.end(), newton_ws_.u_perturb().begin());
        apply_operator_with_blocking(t, u, workspace_->lu());

        // Interior points: tridiagonal structure via finite differences
        // J = ∂F/∂u where F(u) = u - rhs - coeff_dt·L(u)
        // So ∂F/∂u = I - coeff_dt·∂L/∂u
        for (size_t i = 1; i < n_ - 1; ++i) {
            // Diagonal: ∂F/∂u_i = 1 - coeff_dt·∂L_i/∂u_i
            newton_ws_.u_perturb()[i] = u[i] + eps;
            apply_operator_with_blocking(t, newton_ws_.u_perturb(), newton_ws_.Lu_perturb());
            double dLi_dui = (newton_ws_.Lu_perturb()[i] - workspace_->lu()[i]) / eps;
            newton_ws_.jacobian_diag()[i] = 1.0 - coeff_dt * dLi_dui;
            newton_ws_.u_perturb()[i] = u[i];

            // Lower diagonal: ∂F_i/∂u_{i-1} = -coeff_dt·∂L_i/∂u_{i-1}
            newton_ws_.u_perturb()[i - 1] = u[i - 1] + eps;
            apply_operator_with_blocking(t, newton_ws_.u_perturb(), newton_ws_.Lu_perturb());
            double dLi_duim1 = (newton_ws_.Lu_perturb()[i] - workspace_->lu()[i]) / eps;
            newton_ws_.jacobian_lower()[i - 1] = -coeff_dt * dLi_duim1;
            newton_ws_.u_perturb()[i - 1] = u[i - 1];

            // Upper diagonal: ∂F_i/∂u_{i+1} = -coeff_dt·∂L_i/∂u_{i+1}
            newton_ws_.u_perturb()[i + 1] = u[i + 1] + eps;
            apply_operator_with_blocking(t, newton_ws_.u_perturb(), newton_ws_.Lu_perturb());
            double dLi_duip1 = (newton_ws_.Lu_perturb()[i] - workspace_->lu()[i]) / eps;
            newton_ws_.jacobian_upper()[i] = -coeff_dt * dLi_duip1;
            newton_ws_.u_perturb()[i + 1] = u[i + 1];
        }
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
            double dL0_du0 = (newton_ws_.Lu_perturb()[0] - workspace_->lu()[0]) / eps;
            newton_ws_.jacobian_diag()[0] = 1.0 - coeff_dt * dL0_du0;
            newton_ws_.u_perturb()[0] = u[0];

            newton_ws_.u_perturb()[1] = u[1] + eps;
            apply_operator_with_blocking(t, newton_ws_.u_perturb(), newton_ws_.Lu_perturb());
            double dL0_du1 = (newton_ws_.Lu_perturb()[0] - workspace_->lu()[0]) / eps;
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
            double dLi_dui = (newton_ws_.Lu_perturb()[i] - workspace_->lu()[i]) / eps;
            newton_ws_.jacobian_diag()[i] = 1.0 - coeff_dt * dLi_dui;
            newton_ws_.u_perturb()[i] = u[i];
        }
    }

    /// Apply operator to batch (AoS layout)
    ///
    /// Applies the spatial operator to a batch of contracts using Array-of-Structs
    /// layout where u_batch[i * batch_width + lane] is contract `lane` at grid point `i`.
    ///
    /// Note: Cache blocking was previously attempted but removed because it was
    /// ineffective. The blocked path still passed full arrays to the stencil,
    /// defeating locality benefits while adding loop overhead. True blocking would
    /// require materializing block-local buffers with halos. See CLAUDE.md for details.
    ///
    /// @param t Time at which to evaluate operator
    /// @param u_batch Input solution batch (size: n * batch_width, AoS layout)
    /// @param lu_batch Output operator result (size: n * batch_width, AoS layout)
    /// @param batch_width Number of contracts in batch
    void apply_operator_with_blocking_batch(double t,
                                           std::span<const double> u_batch,
                                           std::span<double> lu_batch,
                                           size_t batch_width) {
        const size_t n = u_batch.size() / batch_width;

        // Direct evaluation (no cache blocking - see CLAUDE.md)
        spatial_op_.apply_interior_batch(t, u_batch, lu_batch, batch_width, 1, n-1);

        // Zero boundary values (BCs will override after)
        for (size_t lane = 0; lane < batch_width; ++lane) {
            lu_batch[lane] = 0.0;  // Left boundary
            lu_batch[(n-1)*batch_width + lane] = 0.0;  // Right boundary
        }
    }

};

}  // namespace mango
