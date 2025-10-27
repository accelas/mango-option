/**
 * DTrace/SystemTap Provider Definition for iv_calc Library
 *
 * This file documents all USDT probe points in the iv_calc library.
 * The tracing system is designed to be module-agnostic and work across
 * all library components.
 *
 * To list available probes in a compiled binary:
 *   $ readelf -n lib*.so | grep -A4 NT_STAPSDT
 *   $ sudo bpftrace -l 'usdt:./lib*.so:ivcalc:*'
 *
 * Provider: ivcalc
 * Library: iv_calc (libpde_solver.so, libamerican_option.so, etc.)
 */

provider ivcalc {
    /**
     * Algorithm Lifecycle Probes (General)
     */

    /* Fired when any algorithm begins */
    probe algo_start(int module_id, double param1, double param2, double param3);

    /* Fired periodically during algorithm execution */
    probe algo_progress(int module_id, size_t current, size_t total, double metric);

    /* Fired when algorithm completes successfully */
    probe algo_complete(int module_id, size_t iterations, double final_metric);

    /**
     * Convergence Tracking Probes (General)
     */

    /* Fired on each convergence iteration */
    probe convergence_iter(int module_id, size_t step, size_t iter, double error, double tolerance);

    /* Fired when convergence succeeds */
    probe convergence_success(int module_id, size_t step, size_t final_iter, double final_error);

    /* Fired when convergence fails */
    probe convergence_failed(int module_id, size_t step, size_t max_iter, double final_error);

    /**
     * Validation and Error Probes (General)
     */

    /* Fired when input validation fails */
    probe validation_error(int module_id, int error_code, double param1, double param2);

    /* Fired on runtime errors */
    probe runtime_error(int module_id, int error_code, double context);

    /**
     * Implied Volatility Specific Probes
     */

    /* Fired when IV calculation begins */
    probe iv_start(double spot, double strike, double time_to_maturity, double market_price);

    /* Fired when IV calculation completes */
    probe iv_complete(double implied_vol, int iterations, int converged);

    /**
     * Brent's Method Specific Probes
     */

    /* Fired on each Brent iteration */
    probe brent_iter(int iter, double x, double fx, double interval_width);

    /**
     * American Option Specific Probes
     */

    /* Fired when option pricing begins */
    probe option_start(int option_type, double strike, double volatility, double time_to_maturity);

    /* Fired when option pricing completes */
    probe option_complete(int status, int iterations);
};
