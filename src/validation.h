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
 * Compares interpolated prices from table against "true" FDM prices computed
 * on fine grids. Converts both to implied volatility using Brent's method and
 * measures error in basis points.
 *
 * @param table: Price table to validate
 * @param grid_params: FDM grid parameters for computing "true" prices
 * @param n_samples: Number of random samples to test (recommend 10,000+)
 * @param target_error_bp: Target error threshold for high-error identification (e.g., 1.0 for 1bp)
 * @return Validation result with statistics and high-error regions
 *
 * Performance: ~2-5 seconds per 1000 samples (depends on FDM grid resolution)
 *
 * Example:
 * @code
 *   AmericanOptionGrid grid = {.n_points = 201, .n_steps = 2000, ...};
 *   ValidationResult result = validate_interpolation_error(table, &grid, 10000, 1.0);
 *
 *   if (result.p95_iv_error > 1.0) {
 *       printf("Need refinement: P95 error = %.2f bp\n", result.p95_iv_error);
 *   }
 *
 *   validation_result_free(&result);
 * @endcode
 */
ValidationResult validate_interpolation_error(
    const OptionPriceTable *table,
    const AmericanOptionGrid *grid_params,
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

#ifdef __cplusplus
}
#endif

#endif // MANGO_VALIDATION_H
