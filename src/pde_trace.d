/**
 * DTrace/SystemTap Provider Definition for PDE Solver Library
 *
 * This file documents all USDT probe points in the iv_calc library.
 * While the actual probes use sys/sdt.h macros, this file serves as
 * formal documentation and can be used with dtrace/systemtap tools.
 *
 * To list available probes in a compiled binary:
 *   $ readelf -n libpde_solver.so | grep -A4 NT_STAPSDT
 *   $ sudo bpftrace -l 'usdt:./libpde_solver.so:*'
 *
 * Provider: pde
 * Library: libpde_solver.so / iv_calc
 */

provider pde {
    /**
     * Solver Lifecycle Probes
     */

    /* Fired when solve begins */
    probe solver_start(double t_start, double t_end, double dt, size_t n_steps);

    /* Fired periodically to report progress */
    probe solver_progress(size_t step, size_t n_steps, double t_current);

    /* Fired when solve completes successfully */
    probe solver_complete(size_t total_steps, double final_time);

    /**
     * Implicit Solver Convergence Probes
     */

    /* Fired on each implicit solver iteration */
    probe implicit_iter(size_t step, size_t iter, double error, double tolerance);

    /* Fired when implicit solver converges */
    probe implicit_converged(size_t step, size_t final_iter, double final_error);

    /* Fired when implicit solver fails to converge */
    probe implicit_failed(size_t step, double t_current, size_t max_iter);

    /**
     * Error Event Probes
     */

    /* Fired when spline creation fails */
    probe spline_error(size_t n_points, size_t min_required);

    /**
     * Performance Monitoring Probes (Optional)
     */

    /* Time step boundaries */
    probe step_start(size_t step, double t);
    probe step_end(size_t step, double t);

    /* Spatial operator evaluation boundaries */
    probe spatial_op_start(size_t n_points, double t);
    probe spatial_op_end(size_t n_points, double t);
};
