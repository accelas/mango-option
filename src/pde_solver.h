#ifndef PDE_SOLVER_H
#define PDE_SOLVER_H

#include <stddef.h>
#include <stdbool.h>

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

// Boundary condition: value at boundary for time t
typedef double (*BoundaryConditionFunc)(double t, void *user_data);

// Spatial operator: L(u) in the PDE du/dt = L(u)
// Computes the spatial discretization for all points
// Parameters: x (grid), t (time), u (solution), n_points (size),
//             Lu (output array), user_data
typedef void (*SpatialOperatorFunc)(const double *x, double t, const double *u,
                                    size_t n_points, double *Lu, void *user_data);

// Jump condition: for discontinuous coefficients at interfaces
// Returns true if there's a jump at position x, and sets jump value
typedef bool (*JumpConditionFunc)(double x, double *jump_value, void *user_data);

// Obstacle condition: u(x,t) >= psi(x,t) for variational inequalities
// Computes obstacle values for all grid points
// Parameters: x (grid), t (time), n_points (size), psi (output), user_data
typedef void (*ObstacleFunc)(const double *x, double t, size_t n_points,
                             double *psi, void *user_data);

// Callback structure
struct PDECallbacks {
    InitialConditionFunc initial_condition;
    BoundaryConditionFunc left_boundary;
    BoundaryConditionFunc right_boundary;
    SpatialOperatorFunc spatial_operator;
    JumpConditionFunc jump_condition;      // Optional, can be NULL
    ObstacleFunc obstacle;                 // Optional, can be NULL
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

    // Solution storage (sliced from workspace)
    double *u_current;    // u^n
    double *u_next;       // u^{n+1}
    double *u_stage;      // Intermediate stage for TR-BDF2
    double *rhs;          // Right-hand side vector

    // Working arrays for linear system (sliced from workspace)
    double *matrix_diag;
    double *matrix_upper;
    double *matrix_lower;

    // Temporary arrays for implicit solver (sliced from workspace)
    double *u_old;        // Previous iteration in fixed-point
    double *Lu;           // Spatial operator result
    double *u_temp;       // Temporary for relaxation
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

// Cubic spline interpolation

// Cubic spline structure
typedef struct {
    size_t n_points;      // Number of data points
    const double *x;      // Grid points (not owned)
    const double *y;      // Function values (not owned)
    double *workspace;    // Single buffer for all coefficients
    double *coeffs_a;     // Spline coefficients (sliced from workspace)
    double *coeffs_b;
    double *coeffs_c;
    double *coeffs_d;
} CubicSpline;

// Create and compute cubic spline interpolation
// Uses natural boundary conditions (second derivative = 0 at endpoints)
CubicSpline* pde_spline_create(const double *x, const double *y, size_t n_points);

// Destroy spline and free memory
void pde_spline_destroy(CubicSpline *spline);

// Evaluate spline at arbitrary point x_eval
// Returns interpolated value
double pde_spline_eval(const CubicSpline *spline, double x_eval);

// Evaluate spline derivative at arbitrary point x_eval
double pde_spline_eval_derivative(const CubicSpline *spline, double x_eval);

// Convenience function: Interpolate solution from solver at arbitrary point
double pde_solver_interpolate(const PDESolver *solver, double x_eval);

#endif // PDE_SOLVER_H
