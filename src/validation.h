#ifndef MANGO_VALIDATION_H
#define MANGO_VALIDATION_H

#include <stddef.h>
#include <stdbool.h>
#include "price_table.h"
#include "american_option.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @file validation.h
 * @brief Validation framework for adaptive grid refinement
 *
 * Provides tools to measure interpolation error in implied volatility (IV) space
 * and identify regions requiring grid refinement. This enables the adaptive
 * refinement workflow to achieve <1bp IV error for 95% of validation points.
 *
 * Key features:
 * - IV error computation (not price error) for more meaningful accuracy metrics
 * - Stratified random sampling for comprehensive coverage
 * - Statistical analysis (max, mean, p95, fraction below threshold)
 * - High-error region identification for targeted refinement
 *
 * Typical usage:
 *   ValidationResult result = validate_interpolation_error(
 *       table, &grid_params, 10000, 1.0);  // 10K samples, 1bp target
 *
 *   printf("P95 IV error: %.2f bp\n", result.p95_iv_error);
 *   printf("Fraction below 1bp: %.1f%%\n", result.fraction_below_1bp * 100);
 *
 *   // Refine high-error regions
 *   expand_grid_at_points(table, result.high_error_moneyness, result.n_high_error);
 *
 *   validation_result_free(&result);
 */

/**
 * Validation result structure
 *
 * Contains error statistics and high-error region identification for
 * guiding adaptive refinement.
 */
typedef struct {
    // Error statistics (in basis points)
    double max_iv_error;          ///< Maximum IV error across all samples (bp)
    double mean_iv_error;         ///< Mean absolute IV error (bp)
    double median_iv_error;       ///< Median IV error (bp)
    double p95_iv_error;          ///< 95th percentile IV error (bp)
    double p99_iv_error;          ///< 99th percentile IV error (bp)

    // Coverage metrics
    double fraction_below_1bp;    ///< Fraction of samples with error < 1bp
    double fraction_below_5bp;    ///< Fraction of samples with error < 5bp
    double fraction_below_10bp;   ///< Fraction of samples with error < 10bp
    size_t n_samples;             ///< Number of validation samples

    // High-error regions for refinement (sorted by error magnitude)
    double *high_error_moneyness; ///< Moneyness values needing refinement
    double *high_error_maturity;  ///< Maturity values for high-error points
    double *high_error_values;    ///< Error magnitudes (bp) for sorting
    size_t n_high_error;          ///< Number of high-error points
    size_t high_error_capacity;   ///< Allocated capacity for high-error arrays
} ValidationResult;

/**
 * Validation sample point (4D parameter space)
 */
typedef struct {
    double moneyness;    // m = S/K
    double maturity;     // τ = T - t (years)
    double volatility;   // σ (decimal)
    double rate;         // r (decimal)
} ValidationSample;

/**
 * Validate interpolation error in IV space
 *
 * Compares interpolated prices from table against "true" prices from either:
 * 1. Reference table interpolation (fast, tests interpolation quality)
 * 2. FDM solve on fine grid (slow, tests absolute accuracy)
 *
 * Converts both to implied volatility using Brent's method and measures error
 * in basis points.
 *
 * @param table: Price table to validate
 * @param grid_params: FDM grid parameters (only used if reference_table is NULL)
 * @param reference_table: Optional reference table for fast validation (NULL = use FDM)
 * @param n_samples: Number of random samples to test (recommend 10,000+)
 * @param target_error_bp: Target error threshold for high-error identification (e.g., 1.0 for 1bp)
 * @return Validation result with statistics and high-error regions
 *
 * Performance:
 * - With reference table: ~100 microseconds per 1000 samples (fast)
 * - With FDM: ~2-5 seconds per 1000 samples (slow)
 *
 * Example (fast validation with reference table):
 * @code
 *   // Create reference table with 2× denser grid
 *   OptionPriceTable *ref_table = price_table_create(...);  // Dense grid
 *   price_table_precompute(ref_table, &grid);
 *   price_table_build_interpolation(ref_table);
 *
 *   // Validate test table against reference
 *   ValidationResult result = validate_interpolation_error(
 *       table, NULL, ref_table, 10000, 1.0);
 *
 *   if (result.p95_iv_error > 1.0) {
 *       printf("Need refinement: P95 error = %.2f bp\n", result.p95_iv_error);
 *   }
 *
 *   validation_result_free(&result);
 *   price_table_destroy(ref_table);
 * @endcode
 *
 * Example (absolute accuracy test with FDM):
 * @code
 *   AmericanOptionGrid grid = {.n_points = 201, .n_steps = 2000, ...};
 *   ValidationResult result = validate_interpolation_error(
 *       table, &grid, NULL, 1000, 1.0);  // No reference table = use FDM
 *
 *   validation_result_free(&result);
 * @endcode
 */
ValidationResult validate_interpolation_error(
    const OptionPriceTable *table,
    const AmericanOptionGrid *grid_params,
    const OptionPriceTable *reference_table,
    size_t n_samples,
    double target_error_bp);

/**
 * Free validation result resources
 *
 * @param result: Validation result to free (must not be NULL)
 */
void validation_result_free(ValidationResult *result);

/**
 * Generate stratified random samples in 4D parameter space
 *
 * Uses stratified sampling to ensure comprehensive coverage of the parameter
 * space. Divides each dimension into bins and samples uniformly within each bin.
 *
 * @param table: Price table defining the parameter space bounds
 * @param n_samples: Number of samples to generate
 * @param samples_out: Output array (caller-allocated, size = n_samples)
 *
 * Sampling strategy:
 * - Moneyness: Log-uniform in [0.7, 1.3] (OTM to ITM)
 * - Maturity: Linear in [table.tau_min, table.tau_max]
 * - Volatility: Linear in [table.sigma_min, table.sigma_max]
 * - Rate: Linear in [table.r_min, table.r_max]
 *
 * Example:
 * @code
 *   ValidationSample *samples = malloc(10000 * sizeof(ValidationSample));
 *   generate_stratified_samples(table, 10000, samples);
 *   // Use samples for validation...
 *   free(samples);
 * @endcode
 */
void generate_stratified_samples(
    const OptionPriceTable *table,
    size_t n_samples,
    ValidationSample *samples_out);

/**
 * Compute implied volatility from option price using Brent's method
 *
 * Internal helper exposed for testing. Wraps the IV solver with appropriate
 * error handling.
 *
 * @param price: Market option price
 * @param moneyness: S/K ratio
 * @param maturity: Time to maturity (years)
 * @param rate: Risk-free rate (decimal)
 * @param option_type: OPTION_CALL or OPTION_PUT
 * @return Implied volatility (decimal), or NAN on failure
 *
 * Convergence: Typically 5-10 iterations, tolerance 1e-8
 */
double compute_implied_volatility(
    double price,
    double moneyness,
    double maturity,
    double rate,
    OptionType option_type);

/**
 * Print validation result summary to stdout
 *
 * @param result: Validation result to print
 *
 * Output format:
 *   Validation Results (10000 samples):
 *     Mean IV error:     0.42 bp
 *     Median IV error:   0.31 bp
 *     P95 IV error:      1.23 bp
 *     P99 IV error:      2.87 bp
 *     Max IV error:      5.42 bp
 *     Below 1bp:         94.3%
 *     Below 5bp:         99.1%
 *     High-error points: 573
 */
void validation_result_print(const ValidationResult *result);

/**
 * Identify moneyness refinement points based on validation errors (for adaptive refinement)
 *
 * Analyzes high-error points from validation and generates new moneyness grid points
 * to add for refinement. Uses a clustering strategy to add points where errors are
 * concentrated, inserting midpoints between existing grid points in high-error regions.
 *
 * @param result: Validation result containing high-error points
 * @param table: Price table with current moneyness grid
 * @param n_new_out: Output parameter for number of new points (set by function)
 * @return Dynamically allocated array of new moneyness values (caller must free),
 *         or NULL on error
 *
 * Strategy:
 * 1. Bin high-error points by moneyness (using current grid intervals)
 * 2. For bins with error concentration, add midpoint refinement
 * 3. Limit total new points to avoid grid explosion
 *
 * Example (adaptive refinement loop):
 * @code
 *   ValidationResult result = validate_interpolation_error(table, &grid, 1000, 1.0);
 *
 *   if (result.p95_iv_error > 1.0) {
 *       size_t n_new;
 *       double *new_points = identify_refinement_points(&result, table, &n_new);
 *
 *       if (new_points) {
 *           price_table_expand_grid(table, new_points, n_new);
 *           price_table_precompute(table, &grid);  // Recompute NaN entries
 *           free(new_points);
 *       }
 *   }
 *
 *   validation_result_free(&result);
 * @endcode
 *
 * Implementation notes:
 * - Adds at most n_moneyness new points (doubles grid size max)
 * - Skips intervals with < 3 high-error samples (not statistically significant)
 * - Returns sorted array of unique new points
 *
 * Time complexity: O(n_high_error × log(n_m))
 * Space complexity: O(n_m)
 */
double* identify_refinement_points(
    const ValidationResult *result,
    const OptionPriceTable *table,
    size_t *n_new_out);

/**
 * Pre-compute prices with adaptive grid refinement (for high-accuracy requirements)
 *
 * Iteratively refines the moneyness grid based on validation errors until target
 * accuracy is achieved or iteration limit is reached. Uses the validation framework
 * to identify high-error regions and adds refinement points where needed.
 *
 * @param table: Price table to populate (must use LAYOUT_M_INNER)
 * @param grid: FDM solver grid parameters
 * @param target_iv_error_bp: Target IV error in basis points (e.g., 1.0 for 1bp)
 * @param max_iterations: Maximum refinement iterations (typically 3-5)
 * @param validation_samples: Number of random samples for error validation (e.g., 1000)
 * @return 0 on success (target achieved), 1 on partial success (max iterations),
 *         -1 on error
 *
 * Workflow:
 * 1. Start with coarse grid (e.g., 10-15 moneyness points)
 * 2. Precompute prices on current grid
 * 3. Validate interpolation error via random sampling
 * 4. If P95 error > target:
 *    a. Identify high-error intervals
 *    b. Add midpoint refinement points
 *    c. Expand grid and mark new points as NaN
 *    d. Recompute only NaN entries
 *    e. Repeat
 * 5. Converge when P95 error < target and 95% of points < target
 *
 * Example:
 * @code
 *   // Create table with coarse grid (10 points)
 *   double m_grid[10];
 *   generate_log_spaced(m_grid, 10, 0.7, 1.3);
 *   OptionPriceTable *table = price_table_create(m_grid, 10, ...);
 *
 *   // Adaptive refinement to 1bp accuracy
 *   AmericanOptionGrid grid = {.n_space = 101, .n_time = 1000, .S_max = 200.0};
 *   int status = price_table_precompute_adaptive(
 *       table, &grid,
 *       1.0,    // 1bp target
 *       5,      // max 5 iterations
 *       1000    // validate with 1000 samples
 *   );
 *
 *   if (status == 0) {
 *       printf("Target accuracy achieved: grid size = %zu\n", table->n_moneyness);
 *   }
 * @endcode
 *
 * Performance:
 * - Typical convergence: 2-3 iterations
 * - Grid size: 10 → 15-25 points (50-150% increase)
 * - Time: 300-500ms per iteration (3× slower than non-adaptive, 6× faster than dense uniform)
 * - Accuracy: <1bp for 95% of validation points
 *
 * Requirements:
 * - table->memory_layout must be LAYOUT_M_INNER (unified grid requirement)
 * - Initial grid should be coarse (~10-15 points)
 * - target_iv_error_bp typically 0.5-2.0 bp
 */
int price_table_precompute_adaptive(
    OptionPriceTable *table,
    const AmericanOptionGrid *grid,
    double target_iv_error_bp,
    size_t max_iterations,
    size_t validation_samples);

#ifdef __cplusplus
}
#endif

#endif // MANGO_VALIDATION_H
