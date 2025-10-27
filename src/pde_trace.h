/**
 * @file pde_trace.h
 * @brief USDT (User Statically-Defined Tracing) probes for PDE solver library
 *
 * This header provides zero-overhead tracing points that can be dynamically
 * enabled at runtime using tools like bpftrace, systemtap, or perf.
 *
 * When tracing is disabled (default), probes compile to single NOP instructions.
 * When enabled via tracing tools, probes capture structured data without
 * modifying the library binary.
 *
 * Example usage with bpftrace:
 *   # Trace solver lifecycle
 *   sudo bpftrace -e 'usdt:./libpde_solver.so:pde:solver_start { ... }'
 *
 *   # Monitor convergence failures
 *   sudo bpftrace -e 'usdt:./libpde_solver.so:pde:implicit_failed { ... }'
 */

#ifndef PDE_TRACE_H
#define PDE_TRACE_H

#include <stddef.h>

/**
 * USDT Configuration
 *
 * On Linux with systemtap-sdt-dev installed, use sys/sdt.h
 * Otherwise, define no-op macros for compatibility
 */
#ifdef HAVE_SYSTEMTAP_SDT
#include <sys/sdt.h>
#else
// Fallback: define empty macros when SDT is not available
#define DTRACE_PROBE(provider, probe) do {} while(0)
#define DTRACE_PROBE1(provider, probe, arg1) do {} while(0)
#define DTRACE_PROBE2(provider, probe, arg1, arg2) do {} while(0)
#define DTRACE_PROBE3(provider, probe, arg1, arg2, arg3) do {} while(0)
#define DTRACE_PROBE4(provider, probe, arg1, arg2, arg3, arg4) do {} while(0)
#define DTRACE_PROBE5(provider, probe, arg1, arg2, arg3, arg4, arg5) do {} while(0)
#endif

/**
 * Provider name for all PDE solver probes
 */
#define PDE_PROVIDER pde

/**
 * Solver Lifecycle Probes
 * ========================
 * Track the high-level execution flow of PDE solve operations
 */

/**
 * Fired when pde_solver_solve() begins
 * @param t_start: Starting time value
 * @param t_end: Ending time value
 * @param dt: Time step size
 * @param n_steps: Total number of time steps
 */
#define PDE_TRACE_SOLVER_START(t_start, t_end, dt, n_steps) \
    DTRACE_PROBE4(PDE_PROVIDER, solver_start, t_start, t_end, dt, n_steps)

/**
 * Fired periodically during solve to report progress
 * @param step: Current time step number
 * @param n_steps: Total number of time steps
 * @param t_current: Current time value
 */
#define PDE_TRACE_SOLVER_PROGRESS(step, n_steps, t_current) \
    DTRACE_PROBE3(PDE_PROVIDER, solver_progress, step, n_steps, t_current)

/**
 * Fired when pde_solver_solve() completes successfully
 * @param total_steps: Total number of steps executed
 * @param final_time: Final time value reached
 */
#define PDE_TRACE_SOLVER_COMPLETE(total_steps, final_time) \
    DTRACE_PROBE2(PDE_PROVIDER, solver_complete, total_steps, final_time)

/**
 * Convergence Tracking Probes
 * ============================
 * Monitor implicit solver iterations and convergence behavior
 */

/**
 * Fired on each iteration of the implicit solver
 * @param step: Current time step number
 * @param iter: Current iteration number
 * @param error: Current relative error
 * @param tolerance: Convergence tolerance threshold
 */
#define PDE_TRACE_IMPLICIT_ITER(step, iter, error, tolerance) \
    DTRACE_PROBE4(PDE_PROVIDER, implicit_iter, step, iter, error, tolerance)

/**
 * Fired when implicit solver converges successfully
 * @param step: Time step number
 * @param final_iter: Number of iterations required
 * @param final_error: Final error achieved
 */
#define PDE_TRACE_IMPLICIT_CONVERGED(step, final_iter, final_error) \
    DTRACE_PROBE3(PDE_PROVIDER, implicit_converged, step, final_iter, final_error)

/**
 * Fired when implicit solver fails to converge
 * @param step: Time step number where failure occurred
 * @param t_current: Time value where failure occurred
 * @param max_iter: Maximum iterations that were attempted
 */
#define PDE_TRACE_IMPLICIT_FAILED(step, t_current, max_iter) \
    DTRACE_PROBE3(PDE_PROVIDER, implicit_failed, step, t_current, max_iter)

/**
 * Error Event Probes
 * ===================
 * Track error conditions and input validation failures
 */

/**
 * Fired when spline creation fails due to insufficient points
 * @param n_points: Number of points provided
 * @param min_required: Minimum number of points required
 */
#define PDE_TRACE_SPLINE_ERROR(n_points, min_required) \
    DTRACE_PROBE2(PDE_PROVIDER, spline_error, n_points, min_required)

/**
 * Performance Monitoring Probes (Optional)
 * =========================================
 * Fine-grained timing for performance analysis
 */

/**
 * Fired at the start of a time step
 * @param step: Time step number
 * @param t: Current time value
 */
#define PDE_TRACE_STEP_START(step, t) \
    DTRACE_PROBE2(PDE_PROVIDER, step_start, step, t)

/**
 * Fired at the end of a time step
 * @param step: Time step number
 * @param t: Current time value
 */
#define PDE_TRACE_STEP_END(step, t) \
    DTRACE_PROBE2(PDE_PROVIDER, step_end, step, t)

/**
 * Fired before evaluating the spatial operator
 * @param n_points: Number of grid points
 * @param t: Current time value
 */
#define PDE_TRACE_SPATIAL_OP_START(n_points, t) \
    DTRACE_PROBE2(PDE_PROVIDER, spatial_op_start, n_points, t)

/**
 * Fired after evaluating the spatial operator
 * @param n_points: Number of grid points
 * @param t: Current time value
 */
#define PDE_TRACE_SPATIAL_OP_END(n_points, t) \
    DTRACE_PROBE2(PDE_PROVIDER, spatial_op_end, n_points, t)

#endif // PDE_TRACE_H
