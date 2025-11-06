#ifndef PDE_SOLVER_H
#define PDE_SOLVER_H

#include <stddef.h>
#include <stdbool.h>
#include "common/cubic_spline.h"

// Boundary condition types
typedef enum {
    BC_DIRICHLET,  // u = g(t) at boundary
    BC_NEUMANN,    // du/dx = g(t) at boundary
    BC_ROBIN       // a*u + b*du/dx = g(t) at boundary
} BoundaryType;

// Spatial domain and discretization
typedef struct {
    double x_min;      // Left boundary
    double x_max;      // Right boundary
    size_t n_points;   // Number of spatial grid points
    double dx;         // Spatial step size
    double *x;         // Grid points
} SpatialGrid;

// Time domain
typedef struct {
    double t_start;
    double t_end;
    double dt;         // Time step
    size_t n_steps;    // Number of time steps
} TimeDomain;

// Forward declarations
typedef struct PDESolver PDESolver;
typedef struct PDECallbacks PDECallbacks;

// Callback function types - All callbacks operate on entire vectors for efficiency

// Initial condition: u(x, t=0) for all grid points
// Parameters: x (grid points), n_points (grid size), u0 (output array), user_data
typedef void (*InitialConditionFunc)(const double *x, size_t n_points,
                                     double *u0, void *user_data);

// Boundary condition: value/gradient at boundary for time t
// Parameters:
//   t: current time
//   x_boundary: location of boundary (x_min for left, x_max for right)
//   bc_type: type of boundary condition (Dirichlet/Neumann/Robin)
//   user_data: user-provided context
// Returns:
//   - For Dirichlet: boundary value u(x_boundary, t)
//   - For Neumann: boundary gradient du/dx at x_boundary
//   - For Robin: right-hand side g(t) in equation a*u + b*du/dx = g(t)
typedef double (*BoundaryConditionFunc)(double t, double x_boundary,
                                        BoundaryType bc_type, void *user_data);

// Spatial operator: L(u) in the PDE du/dt = L(u)
// Computes the spatial discretization for all points
// Parameters: x (grid), t (time), u (solution), n_points (size),
//             Lu (output array), user_data
typedef void (*SpatialOperatorFunc)(const double *x, double t, const double *u,
                                    size_t n_points, double *Lu, void *user_data);

// Jump condition: for discontinuous coefficients at interfaces
// Returns true if there's a jump at position x, and sets jump value
// Note: Currently unused by solver, kept for backward compatibility
typedef bool (*JumpConditionFunc)(double x, double *jump_value, void *user_data);

// Obstacle condition: u(x,t) >= psi(x,t) for variational inequalities
// Computes obstacle values for all grid points
// Parameters: x (grid), t (time), n_points (size), psi (output), user_data
typedef void (*ObstacleFunc)(const double *x, double t, size_t n_points,
                             double *psi, void *user_data);

// Temporal event: Handle time-based events (e.g., dividend payments)
// Called by solver when crossing registered event times
// Parameters: t (current time after events), x (grid points), n_points (size),
//             u (solution - writable), event_indices (indices of events that occurred),
//             n_events_triggered (number of events), user_data,
//             workspace (n_points doubles for temporary storage)
// Note: Callback can modify u in-place to apply event effects
typedef void (*TemporalEventFunc)(double t, const double *x, size_t n_points,
                                   double *u, const size_t *event_indices,
                                   size_t n_events_triggered, void *user_data,
                                   double *workspace);

// Callback structure
struct PDECallbacks {
    InitialConditionFunc initial_condition;
    BoundaryConditionFunc left_boundary;
    BoundaryConditionFunc right_boundary;
    SpatialOperatorFunc spatial_operator;
    double diffusion_coeff;                // Diffusion coefficient D for L(u)=D·∂²u/∂x²
                                           // Required for Neumann BC; set to NAN if variable/not applicable
    JumpConditionFunc jump_condition;      // Optional, can be NULL (unused)
    ObstacleFunc obstacle;                 // Optional, can be NULL
    TemporalEventFunc temporal_event;      // Optional, can be NULL
    size_t n_temporal_events;              // Number of temporal events
    double *temporal_event_times;          // Event times (must be sorted ascending)
    void *user_data;                       // User-provided context data
};

// Boundary configuration
typedef struct {
    BoundaryType left_type;
    BoundaryType right_type;
    double left_robin_a;   // For Robin BC: a*u + b*du/dx = g
    double left_robin_b;
    double right_robin_a;
    double right_robin_b;
} BoundaryConfig;

// TR-BDF2 parameters
typedef struct {
    double gamma;      // TR-BDF2 parameter (typically 2 - sqrt(2))
    size_t max_iter;   // Max iterations for implicit solver
    double tolerance;  // Convergence tolerance
} TRBDF2Config;

// Main PDE solver structure
struct PDESolver {
    SpatialGrid grid;
    TimeDomain time;
    BoundaryConfig bc_config;
    TRBDF2Config trbdf2_config;
    PDECallbacks callbacks;

    // Single workspace buffer for all arrays (better cache locality)
    double *workspace;

    // Swappable solution buffers (separate allocations for pointer swapping)
    double *buffer_A;
    double *buffer_B;
    double *buffer_C;

    // Solution storage (pointers to buffers, swappable)
    double *u_current;    // u^n (points to one of buffer_A/B/C)
    double *u_next;       // u^{n+1} (points to one of buffer_A/B/C)
    double *u_stage;      // Intermediate stage for TR-BDF2 (points to one of buffer_A/B/C)
    double *rhs;          // Right-hand side vector (sliced from workspace)

    // Working arrays for linear system (sliced from workspace)
    double *matrix_diag;
    double *matrix_upper;
    double *matrix_lower;

    // Temporary arrays for implicit solver (sliced from workspace)
    double *u_old;        // Previous iteration in fixed-point
    double *Lu;           // Spatial operator result
    double *u_temp;       // Temporary for relaxation

    // Workspace for tridiagonal solver (2n doubles, sliced from workspace)
    double *tridiag_workspace;  // c_prime and d_prime arrays for Thomas algorithm
};

// Core API functions

// Create and initialize solver
// Note: Takes ownership of grid - grid.x will be set to nullptr after this call
PDESolver* pde_solver_create(SpatialGrid *grid,
                              const TimeDomain *time,
                              const BoundaryConfig *bc_config,
                              const TRBDF2Config *trbdf2_config,
                              const PDECallbacks *callbacks);

// Destroy solver and free memory
void pde_solver_destroy(PDESolver *solver);

// Initialize the solution with initial conditions
void pde_solver_initialize(PDESolver *solver);

// Perform one TR-BDF2 time step
int pde_solver_step(PDESolver *solver, double t_current);

// Solve the entire time domain
int pde_solver_solve(PDESolver *solver);

// Get current solution
const double* pde_solver_get_solution(const PDESolver *solver);

// Get grid points
const double* pde_solver_get_grid(const PDESolver *solver);

// Utility functions

// Create spatial grid
SpatialGrid pde_create_grid(double x_min, double x_max, size_t n_points);

// Free spatial grid resources
void pde_free_grid(SpatialGrid *grid);

// Create default TR-BDF2 configuration
TRBDF2Config pde_default_trbdf2_config(void);

// Create default boundary configuration (Dirichlet)
BoundaryConfig pde_default_boundary_config(void);

// Convenience function: Interpolate solution from solver at arbitrary point
// Uses cubic spline interpolation
double pde_solver_interpolate(const PDESolver *solver, double x_eval);

#endif // PDE_SOLVER_H
