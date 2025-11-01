/**
 * @file ivcalc_trace.h
 * @brief USDT (User Statically-Defined Tracing) probes for mango library
 *
 * This header provides zero-overhead tracing points that can be dynamically
 * enabled at runtime using tools like bpftrace, systemtap, or perf.
 *
 * When tracing is disabled (default), probes compile to single NOP instructions.
 * When enabled via tracing tools, probes capture structured data without
 * modifying the library binary.
 *
 * The tracing system is designed to be module-agnostic and work across all
 * components: PDE solver, implied volatility, American options, root finding, etc.
 *
 * Example usage with bpftrace:
 *   # Trace all algorithm executions
 *   sudo bpftrace -e 'usdt:./lib*.so:mango:algo_* { ... }'
 *
 *   # Monitor convergence across all modules
 *   sudo bpftrace -e 'usdt:./lib*.so:mango:convergence_failed { ... }'
 */

#ifndef IVCALC_TRACE_H
#define IVCALC_TRACE_H

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
#define DTRACE_PROBE6(provider, probe, arg1, arg2, arg3, arg4, arg5, arg6) do {} while(0)
#endif

/**
 * Provider name for all mango library probes
 */
#define MANGO_PROVIDER mango

/**
 * Module identifiers for multi-module tracing
 * These are passed as the first parameter to many probes
 */
#define MODULE_PDE_SOLVER       1
#define MODULE_AMERICAN_OPTION  2
#define MODULE_IMPLIED_VOL      3
#define MODULE_BRENT_ROOT       4
#define MODULE_CUBIC_SPLINE     5
#define MODULE_VALIDATION       6
#define MODULE_PRICE_TABLE      7

/**
 * ============================================================================
 * Algorithm Lifecycle Probes
 * ============================================================================
 * Track high-level execution flow of algorithms across all modules
 */

/**
 * Fired when an algorithm begins execution
 * @param module_id: Module identifier (MODULE_* constant)
 * @param param1: Module-specific parameter (e.g., n_steps, max_iter)
 * @param param2: Module-specific parameter (e.g., dt, tolerance)
 * @param param3: Module-specific parameter
 */
#define MANGO_TRACE_ALGO_START(module_id, param1, param2, param3) \
    DTRACE_PROBE4(MANGO_PROVIDER, algo_start, module_id, param1, param2, param3)

/**
 * Fired periodically during algorithm execution to report progress
 * @param module_id: Module identifier
 * @param current: Current progress (e.g., step number, iteration)
 * @param total: Total work (e.g., total steps, max iterations)
 * @param metric: Progress metric (e.g., current time, current error)
 */
#define MANGO_TRACE_ALGO_PROGRESS(module_id, current, total, metric) \
    DTRACE_PROBE4(MANGO_PROVIDER, algo_progress, module_id, current, total, metric)

/**
 * Fired when an algorithm completes successfully
 * @param module_id: Module identifier
 * @param iterations: Number of iterations/steps completed
 * @param final_metric: Final metric value (e.g., final time, final error)
 */
#define MANGO_TRACE_ALGO_COMPLETE(module_id, iterations, final_metric) \
    DTRACE_PROBE3(MANGO_PROVIDER, algo_complete, module_id, iterations, final_metric)

/**
 * ============================================================================
 * Convergence Tracking Probes
 * ============================================================================
 * Monitor iterative solver convergence across all algorithms
 */

/**
 * Fired on each iteration of a convergence loop
 * @param module_id: Module identifier
 * @param step: Outer step/stage number (0 if not applicable)
 * @param iter: Current iteration number
 * @param error: Current error metric
 * @param tolerance: Convergence threshold
 */
#define MANGO_TRACE_CONVERGENCE_ITER(module_id, step, iter, error, tolerance) \
    DTRACE_PROBE5(MANGO_PROVIDER, convergence_iter, module_id, step, iter, error, tolerance)

/**
 * Fired when convergence is achieved
 * @param module_id: Module identifier
 * @param step: Outer step/stage number (0 if not applicable)
 * @param final_iter: Number of iterations required
 * @param final_error: Final error achieved
 */
#define MANGO_TRACE_CONVERGENCE_SUCCESS(module_id, step, final_iter, final_error) \
    DTRACE_PROBE4(MANGO_PROVIDER, convergence_success, module_id, step, final_iter, final_error)

/**
 * Fired when convergence fails
 * @param module_id: Module identifier
 * @param step: Outer step/stage number (0 if not applicable)
 * @param max_iter: Maximum iterations attempted
 * @param final_error: Final error at failure
 */
#define MANGO_TRACE_CONVERGENCE_FAILED(module_id, step, max_iter, final_error) \
    DTRACE_PROBE4(MANGO_PROVIDER, convergence_failed, module_id, step, max_iter, final_error)

/**
 * ============================================================================
 * Validation and Error Probes
 * ============================================================================
 * Track input validation failures and runtime errors
 */

/**
 * Fired when input validation fails
 * @param module_id: Module identifier
 * @param error_code: Error code (module-specific)
 * @param param1: Relevant parameter value
 * @param param2: Relevant parameter value or threshold
 */
#define MANGO_TRACE_VALIDATION_ERROR(module_id, error_code, param1, param2) \
    DTRACE_PROBE4(MANGO_PROVIDER, validation_error, module_id, error_code, param1, param2)

/**
 * Fired when a runtime error occurs
 * @param module_id: Module identifier
 * @param error_code: Error code
 * @param context: Context value (e.g., step number, iteration)
 */
#define MANGO_TRACE_RUNTIME_ERROR(module_id, error_code, context) \
    DTRACE_PROBE3(MANGO_PROVIDER, runtime_error, module_id, error_code, context)

/**
 * ============================================================================
 * Module-Specific Probes: PDE Solver
 * ============================================================================
 */

/**
 * Fired when PDE solve begins
 * @param t_start: Starting time
 * @param t_end: Ending time
 * @param dt: Time step size
 * @param n_steps: Total number of steps
 */
#define MANGO_TRACE_PDE_START(t_start, t_end, dt, n_steps) \
    MANGO_TRACE_ALGO_START(MODULE_PDE_SOLVER, n_steps, dt, (t_end - t_start))

/**
 * Fired during PDE solve progress
 * @param step: Current step
 * @param n_steps: Total steps
 * @param t_current: Current time
 */
#define MANGO_TRACE_PDE_PROGRESS(step, n_steps, t_current) \
    MANGO_TRACE_ALGO_PROGRESS(MODULE_PDE_SOLVER, step, n_steps, t_current)

/**
 * Fired when PDE solve completes
 * @param total_steps: Total steps executed
 * @param final_time: Final time reached
 */
#define MANGO_TRACE_PDE_COMPLETE(total_steps, final_time) \
    MANGO_TRACE_ALGO_COMPLETE(MODULE_PDE_SOLVER, total_steps, final_time)

/**
 * PDE implicit solver iteration
 */
#define MANGO_TRACE_PDE_IMPLICIT_ITER(step, iter, error, tolerance) \
    MANGO_TRACE_CONVERGENCE_ITER(MODULE_PDE_SOLVER, step, iter, error, tolerance)

/**
 * PDE implicit solver converged
 */
#define MANGO_TRACE_PDE_IMPLICIT_CONVERGED(step, final_iter, final_error) \
    MANGO_TRACE_CONVERGENCE_SUCCESS(MODULE_PDE_SOLVER, step, final_iter, final_error)

/**
 * PDE implicit solver failed to converge
 */
#define MANGO_TRACE_PDE_IMPLICIT_FAILED(step, max_iter, final_error) \
    MANGO_TRACE_CONVERGENCE_FAILED(MODULE_PDE_SOLVER, step, max_iter, final_error)

/**
 * ============================================================================
 * Module-Specific Probes: Implied Volatility
 * ============================================================================
 */

/**
 * Fired when IV calculation begins
 * @param spot: Spot price
 * @param strike: Strike price
 * @param time_to_maturity: Time to expiration
 * @param market_price: Market price
 */
#define MANGO_TRACE_IV_START(spot, strike, time_to_maturity, market_price) \
    DTRACE_PROBE4(MANGO_PROVIDER, iv_start, spot, strike, time_to_maturity, market_price)

/**
 * Fired when IV calculation completes
 * @param implied_vol: Calculated implied volatility
 * @param iterations: Number of iterations
 * @param converged: 1 if converged, 0 if failed
 */
#define MANGO_TRACE_IV_COMPLETE(implied_vol, iterations, converged) \
    DTRACE_PROBE3(MANGO_PROVIDER, iv_complete, implied_vol, iterations, converged)

/**
 * Fired when IV validation fails (arbitrage bounds, etc.)
 * @param error_code: Error type (1=spot, 2=strike, 3=time, 4=price, 5=arbitrage)
 * @param param1: Relevant parameter
 * @param param2: Threshold or bound
 */
#define MANGO_TRACE_IV_VALIDATION_ERROR(error_code, param1, param2) \
    MANGO_TRACE_VALIDATION_ERROR(MODULE_IMPLIED_VOL, error_code, param1, param2)

/**
 * ============================================================================
 * Module-Specific Probes: Brent's Method
 * ============================================================================
 */

/**
 * Fired when root finding begins
 * @param a: Lower bound
 * @param b: Upper bound
 * @param tolerance: Convergence tolerance
 * @param max_iter: Maximum iterations
 */
#define MANGO_TRACE_BRENT_START(a, b, tolerance, max_iter) \
    MANGO_TRACE_ALGO_START(MODULE_BRENT_ROOT, max_iter, tolerance, (b - a))

/**
 * Fired on each Brent iteration
 * @param iter: Iteration number
 * @param x: Current point
 * @param fx: Function value at x
 * @param interval_width: Current bracket width
 */
#define MANGO_TRACE_BRENT_ITER(iter, x, fx, interval_width) \
    DTRACE_PROBE4(MANGO_PROVIDER, brent_iter, iter, x, fx, interval_width)

/**
 * Fired when root finding completes
 * @param root: Found root
 * @param iterations: Number of iterations
 * @param converged: 1 if converged, 0 if failed
 */
#define MANGO_TRACE_BRENT_COMPLETE(root, iterations, converged) \
    MANGO_TRACE_ALGO_COMPLETE(MODULE_BRENT_ROOT, iterations, root)

/**
 * ============================================================================
 * Module-Specific Probes: American Options
 * ============================================================================
 */

/**
 * Fired when option pricing begins
 * @param option_type: 0=call, 1=put
 * @param strike: Strike price
 * @param volatility: Volatility
 * @param time_to_maturity: Time to maturity
 */
#define MANGO_TRACE_OPTION_START(option_type, strike, volatility, time_to_maturity) \
    DTRACE_PROBE4(MANGO_PROVIDER, option_start, option_type, strike, volatility, time_to_maturity)

/**
 * Fired when option pricing completes
 * @param status: 0=success, -1=failure
 * @param iterations: Number of PDE steps
 */
#define MANGO_TRACE_OPTION_COMPLETE(status, iterations) \
    DTRACE_PROBE2(MANGO_PROVIDER, option_complete, status, iterations)

/**
 * ============================================================================
 * Module-Specific Probes: Cubic Spline
 * ============================================================================
 */

/**
 * Fired when spline creation fails
 * @param n_points: Points provided
 * @param min_required: Minimum required
 */
#define MANGO_TRACE_SPLINE_ERROR(n_points, min_required) \
    MANGO_TRACE_VALIDATION_ERROR(MODULE_CUBIC_SPLINE, 1, n_points, min_required)

#endif // IVCALC_TRACE_H
