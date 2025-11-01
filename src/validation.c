#include "validation.h"
#include "implied_volatility.h"
#include "ivcalc_trace.h"
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <stdio.h>

// Comparison function for qsort (ascending order)
static int compare_doubles(const void *a, const void *b) {
    double diff = *(const double *)a - *(const double *)b;
    return (diff > 0) - (diff < 0);
}

// Simple linear congruential generator for reproducible random numbers
static uint64_t rng_state = 12345;

static void rng_seed(uint64_t seed) {
    rng_state = seed;
}

static double rng_uniform(void) {
    // LCG: Xn+1 = (a * Xn + c) mod m
    rng_state = (rng_state * 6364136223846793005ULL + 1442695040888963407ULL);
    return (rng_state >> 11) * (1.0 / 9007199254740992.0);  // [0, 1)
}

void generate_stratified_samples(
    const OptionPriceTable *table,
    size_t n_samples,
    ValidationSample *samples_out)
{
    if (!table || !samples_out || n_samples == 0) return;

    // Seed RNG for reproducibility
    rng_seed(42);

    // Get grid bounds
    double m_min = table->moneyness_grid[0];
    double m_max = table->moneyness_grid[table->n_moneyness - 1];
    double tau_min = table->maturity_grid[0];
    double tau_max = table->maturity_grid[table->n_maturity - 1];
    double sigma_min = table->volatility_grid[0];
    double sigma_max = table->volatility_grid[table->n_volatility - 1];
    double r_min = table->rate_grid[0];
    double r_max = table->rate_grid[table->n_rate - 1];

    // Stratified sampling: divide each dimension into bins
    // For simplicity, use sqrt(n_samples) bins per dimension (4D -> n^(1/4) bins)
    size_t n_bins = (size_t)ceil(pow((double)n_samples, 0.25));
    if (n_bins < 2) n_bins = 2;

    // Generate samples with stratification
    for (size_t i = 0; i < n_samples; i++) {
        // Simple stratified sampling: divide each dimension independently
        // More sophisticated approach would use Latin Hypercube Sampling

        // Moneyness: log-uniform distribution (more samples near ATM)
        double u_m = rng_uniform();
        double log_m_min = log(m_min);
        double log_m_max = log(m_max);
        double log_m = log_m_min + u_m * (log_m_max - log_m_min);
        samples_out[i].moneyness = exp(log_m);

        // Maturity: uniform distribution
        double u_tau = rng_uniform();
        samples_out[i].maturity = tau_min + u_tau * (tau_max - tau_min);

        // Volatility: uniform distribution
        double u_sigma = rng_uniform();
        samples_out[i].volatility = sigma_min + u_sigma * (sigma_max - sigma_min);

        // Rate: uniform distribution
        double u_r = rng_uniform();
        samples_out[i].rate = r_min + u_r * (r_max - r_min);
    }
}

double compute_implied_volatility(
    double price,
    double moneyness,
    double maturity,
    double rate,
    OptionType option_type)
{
    // Set up IV parameters
    IVParams params = {
        .spot_price = moneyness * 100.0,  // Use K=100 for simplicity
        .strike = 100.0,
        .time_to_maturity = maturity,
        .risk_free_rate = rate,
        .dividend_yield = 0.0,
        .market_price = price,
        .option_type = option_type,
        .exercise_type = AMERICAN
    };

    // Use simple IV calculation with default grid
    IVResult result = calculate_iv_simple(&params, NULL);

    if (result.converged) {
        return result.implied_vol;
    } else {
        return NAN;
    }
}

ValidationResult validate_interpolation_error(
    const OptionPriceTable *table,
    const AmericanOptionGrid *grid_params,
    size_t n_samples,
    double target_error_bp)
{
    ValidationResult result = {0};

    if (!table || !grid_params || n_samples == 0) {
        return result;
    }

    MANGO_TRACE_ALGO_START(MODULE_VALIDATION, n_samples, 0, 0.0);

    // Allocate samples
    ValidationSample *samples = malloc(n_samples * sizeof(ValidationSample));
    if (!samples) {
        MANGO_TRACE_RUNTIME_ERROR(MODULE_VALIDATION, 0, 0);
        return result;
    }

    // Generate stratified samples
    generate_stratified_samples(table, n_samples, samples);

    // Allocate error arrays
    double *iv_errors = malloc(n_samples * sizeof(double));
    if (!iv_errors) {
        free(samples);
        MANGO_TRACE_RUNTIME_ERROR(MODULE_VALIDATION, 1, 0);
        return result;
    }

    // Allocate high-error tracking (worst 10% of samples)
    size_t high_error_capacity = n_samples / 10;
    if (high_error_capacity < 100) high_error_capacity = 100;

    result.high_error_moneyness = malloc(high_error_capacity * sizeof(double));
    result.high_error_maturity = malloc(high_error_capacity * sizeof(double));
    result.high_error_values = malloc(high_error_capacity * sizeof(double));

    if (!result.high_error_moneyness || !result.high_error_maturity ||
        !result.high_error_values) {
        free(samples);
        free(iv_errors);
        free(result.high_error_moneyness);
        free(result.high_error_maturity);
        free(result.high_error_values);
        memset(&result, 0, sizeof(result));
        MANGO_TRACE_RUNTIME_ERROR(MODULE_VALIDATION, 2, 0);
        return result;
    }

    result.high_error_capacity = high_error_capacity;

    // Validate each sample
    size_t valid_samples = 0;
    double sum_error = 0.0;
    size_t below_1bp = 0, below_5bp = 0, below_10bp = 0;

    for (size_t i = 0; i < n_samples; i++) {
        if (i % 1000 == 0 && i > 0) {
            size_t progress = (i * 100) / n_samples;
            MANGO_TRACE_ALGO_PROGRESS(MODULE_VALIDATION, i, progress, 0.0);
        }

        ValidationSample *s = &samples[i];

        // 1. Get interpolated price from table
        double price_interp = price_table_interpolate_4d(
            table, s->moneyness, s->maturity, s->volatility, s->rate);

        if (isnan(price_interp) || price_interp <= 0.0) {
            iv_errors[i] = NAN;
            continue;
        }

        // 2. Compute "true" price via FDM on fine grid
        // Use american_option_solve_on_moneyness_grid for unified grid
        OptionData option_data = {
            .strike = 100.0,  // Use K=100 as reference
            .volatility = s->volatility,
            .risk_free_rate = s->rate,
            .time_to_maturity = s->maturity,
            .option_type = table->type,
            .n_dividends = 0,
            .dividend_times = NULL,
            .dividend_amounts = NULL
        };

        // Create fine moneyness grid around sample point
        double m_grid[3] = {
            s->moneyness * 0.95,
            s->moneyness,
            s->moneyness * 1.05
        };

        AmericanOptionResult fdm_result = american_option_solve(
            &option_data, m_grid, 3, grid_params->dt, grid_params->n_steps);

        if (fdm_result.status != 0) {
            iv_errors[i] = NAN;
            american_option_free_result(&fdm_result);
            continue;
        }

        // Extract price at middle point (index 1)
        const double *solution = pde_solver_get_solution(fdm_result.solver);
        double price_fdm = solution[1];

        american_option_free_result(&fdm_result);

        if (isnan(price_fdm) || price_fdm <= 0.0) {
            iv_errors[i] = NAN;
            continue;
        }

        // 3. Convert both prices to IV
        double iv_interp = compute_implied_volatility(
            price_interp, s->moneyness, s->maturity, s->rate, table->type);

        double iv_fdm = compute_implied_volatility(
            price_fdm, s->moneyness, s->maturity, s->rate, table->type);

        if (isnan(iv_interp) || isnan(iv_fdm)) {
            iv_errors[i] = NAN;
            continue;
        }

        // 4. Compute IV error in basis points
        double iv_error_bp = fabs(iv_interp - iv_fdm) * 10000.0;
        iv_errors[i] = iv_error_bp;

        // Update statistics
        sum_error += iv_error_bp;
        valid_samples++;

        if (iv_error_bp < 1.0) below_1bp++;
        if (iv_error_bp < 5.0) below_5bp++;
        if (iv_error_bp < 10.0) below_10bp++;

        // Track high-error points
        if (iv_error_bp > target_error_bp && result.n_high_error < high_error_capacity) {
            result.high_error_moneyness[result.n_high_error] = s->moneyness;
            result.high_error_maturity[result.n_high_error] = s->maturity;
            result.high_error_values[result.n_high_error] = iv_error_bp;
            result.n_high_error++;
        }
    }

    if (valid_samples == 0) {
        free(samples);
        free(iv_errors);
        MANGO_TRACE_RUNTIME_ERROR(MODULE_VALIDATION, 3, 0);
        return result;
    }

    // Compute statistics
    result.n_samples = valid_samples;
    result.mean_iv_error = sum_error / valid_samples;
    result.fraction_below_1bp = (double)below_1bp / valid_samples;
    result.fraction_below_5bp = (double)below_5bp / valid_samples;
    result.fraction_below_10bp = (double)below_10bp / valid_samples;

    // Sort errors for percentiles
    qsort(iv_errors, n_samples, sizeof(double), compare_doubles);

    // Find first non-NAN error (sorted to end)
    size_t first_valid = 0;
    while (first_valid < n_samples && isnan(iv_errors[first_valid])) {
        first_valid++;
    }

    if (first_valid < n_samples) {
        size_t n_valid = n_samples - first_valid;
        result.median_iv_error = iv_errors[first_valid + n_valid / 2];
        result.p95_iv_error = iv_errors[first_valid + (size_t)(n_valid * 0.95)];
        result.p99_iv_error = iv_errors[first_valid + (size_t)(n_valid * 0.99)];
        result.max_iv_error = iv_errors[n_samples - 1];
    }

    free(samples);
    free(iv_errors);

    MANGO_TRACE_ALGO_COMPLETE(MODULE_VALIDATION, valid_samples, 0);

    return result;
}

void validation_result_free(ValidationResult *result) {
    if (!result) return;

    free(result->high_error_moneyness);
    free(result->high_error_maturity);
    free(result->high_error_values);

    memset(result, 0, sizeof(ValidationResult));
}

void validation_result_print(const ValidationResult *result) {
    if (!result) return;

    printf("\nValidation Results (%zu samples):\n", result->n_samples);
    printf("  Mean IV error:     %.2f bp\n", result->mean_iv_error);
    printf("  Median IV error:   %.2f bp\n", result->median_iv_error);
    printf("  P95 IV error:      %.2f bp\n", result->p95_iv_error);
    printf("  P99 IV error:      %.2f bp\n", result->p99_iv_error);
    printf("  Max IV error:      %.2f bp\n", result->max_iv_error);
    printf("  Below 1bp:         %.1f%%\n", result->fraction_below_1bp * 100.0);
    printf("  Below 5bp:         %.1f%%\n", result->fraction_below_5bp * 100.0);
    printf("  Below 10bp:        %.1f%%\n", result->fraction_below_10bp * 100.0);
    printf("  High-error points: %zu\n", result->n_high_error);
}

// Helper: Find grid interval index for a moneyness value
static int find_interval(const double *grid, size_t n, double value) {
    // Binary search for interval
    if (value < grid[0]) return -1;
    if (value >= grid[n-1]) return (int)(n - 2);

    size_t left = 0;
    size_t right = n - 1;

    while (right - left > 1) {
        size_t mid = (left + right) / 2;
        if (value < grid[mid]) {
            right = mid;
        } else {
            left = mid;
        }
    }

    return (int)left;  // Returns i such that grid[i] <= value < grid[i+1]
}

double* identify_refinement_points(
    const ValidationResult *result,
    const OptionPriceTable *table,
    size_t *n_new_out)
{
    // 1. Validate inputs
    if (!result || !table || !n_new_out) {
        MANGO_TRACE_VALIDATION_ERROR(MODULE_VALIDATION, 0, 0, 0.0);
        return NULL;
    }

    if (result->n_high_error == 0 || table->n_moneyness < 2) {
        *n_new_out = 0;
        return NULL;  // Nothing to refine
    }

    const size_t n_m = table->n_moneyness;
    const size_t n_intervals = n_m - 1;

    // 2. Count high-error samples in each moneyness interval
    size_t *interval_counts = calloc(n_intervals, sizeof(size_t));
    double *interval_max_error = calloc(n_intervals, sizeof(double));

    if (!interval_counts || !interval_max_error) {
        free(interval_counts);
        free(interval_max_error);
        MANGO_TRACE_RUNTIME_ERROR(MODULE_VALIDATION, 0, 0);
        return NULL;
    }

    // Bin high-error points by interval
    for (size_t i = 0; i < result->n_high_error; i++) {
        double m = result->high_error_moneyness[i];
        double error = result->high_error_values[i];

        int interval = find_interval(table->moneyness_grid, n_m, m);

        if (interval >= 0 && interval < (int)n_intervals) {
            interval_counts[interval]++;

            if (error > interval_max_error[interval]) {
                interval_max_error[interval] = error;
            }
        }
    }

    // 3. Select intervals for refinement (>= 3 high-error samples)
    const size_t min_samples_per_interval = 3;
    const size_t max_new_points = n_m;  // At most double the grid size

    double *candidate_points = malloc(n_intervals * sizeof(double));
    size_t n_candidates = 0;

    if (!candidate_points) {
        free(interval_counts);
        free(interval_max_error);
        MANGO_TRACE_RUNTIME_ERROR(MODULE_VALIDATION, 1, 0);
        return NULL;
    }

    // Add midpoint of each high-error interval
    for (size_t i = 0; i < n_intervals; i++) {
        if (interval_counts[i] >= min_samples_per_interval) {
            double m_left = table->moneyness_grid[i];
            double m_right = table->moneyness_grid[i + 1];

            // Add midpoint (geometric mean for moneyness)
            double midpoint = sqrt(m_left * m_right);
            candidate_points[n_candidates++] = midpoint;

            if (n_candidates >= max_new_points) {
                break;  // Limit reached
            }
        }
    }

    free(interval_counts);
    free(interval_max_error);

    // If no candidates, return NULL
    if (n_candidates == 0) {
        free(candidate_points);
        *n_new_out = 0;
        return NULL;
    }

    // 4. Sort candidates and remove duplicates
    qsort(candidate_points, n_candidates, sizeof(double), compare_doubles);

    // Remove duplicates with tolerance
    const double tol = 1e-10;
    size_t write_idx = 0;

    for (size_t read_idx = 0; read_idx < n_candidates; read_idx++) {
        // Skip if duplicate of previous value
        if (read_idx > 0 && fabs(candidate_points[read_idx] - candidate_points[write_idx - 1]) < tol) {
            continue;
        }

        candidate_points[write_idx++] = candidate_points[read_idx];
    }

    *n_new_out = write_idx;

    // Trim to actual size
    if (write_idx < n_candidates) {
        double *trimmed = realloc(candidate_points, write_idx * sizeof(double));
        return trimmed ? trimmed : candidate_points;
    }

    return candidate_points;
}

int price_table_precompute_adaptive(
    OptionPriceTable *table,
    const AmericanOptionGrid *grid,
    double target_iv_error_bp,
    size_t max_iterations,
    size_t validation_samples)
{
    // 1. Validate inputs
    if (!table || !grid) {
        MANGO_TRACE_VALIDATION_ERROR(MODULE_VALIDATION, 0, 0, 0.0);
        return -1;
    }

    // Verify LAYOUT_M_INNER (required for unified grid)
    if (table->memory_layout != LAYOUT_M_INNER) {
        MANGO_TRACE_VALIDATION_ERROR(MODULE_VALIDATION, 1, table->memory_layout, LAYOUT_M_INNER);
        return -1;
    }

    // Validate parameters
    if (target_iv_error_bp <= 0.0 || max_iterations == 0 || validation_samples < 100) {
        MANGO_TRACE_VALIDATION_ERROR(MODULE_VALIDATION, 2, max_iterations, target_iv_error_bp);
        return -1;
    }

    printf("\nAdaptive Refinement:\n");
    printf("  Target IV error: %.2f bp\n", target_iv_error_bp);
    printf("  Max iterations:  %zu\n", max_iterations);
    printf("  Validation samples: %zu\n", validation_samples);
    printf("  Initial grid size: %zu moneyness points\n\n", table->n_moneyness);

    // 2. Adaptive refinement loop
    for (size_t iter = 0; iter < max_iterations; iter++) {
        printf("Iteration %zu:\n", iter + 1);

        // 2a. Precompute prices (NaN entries only on subsequent iterations)
        printf("  Precomputing prices...\n");
        int precompute_status = price_table_precompute(table, grid);

        if (precompute_status != 0) {
            MANGO_TRACE_RUNTIME_ERROR(MODULE_VALIDATION, 0, iter);
            return -1;
        }

        // 2b. Validate interpolation error
        printf("  Validating accuracy...\n");
        ValidationResult result = validate_interpolation_error(
            table, grid, validation_samples, target_iv_error_bp
        );

        if (result.n_samples == 0) {
            MANGO_TRACE_RUNTIME_ERROR(MODULE_VALIDATION, 1, iter);
            validation_result_free(&result);
            return -1;
        }

        // Print validation summary
        printf("  Grid size: %zu\n", table->n_moneyness);
        printf("  Mean IV error:   %.2f bp\n", result.mean_iv_error);
        printf("  P95 IV error:    %.2f bp\n", result.p95_iv_error);
        printf("  P99 IV error:    %.2f bp\n", result.p99_iv_error);
        printf("  Below %.1f bp:   %.1f%%\n", target_iv_error_bp, result.fraction_below_1bp * 100.0);
        printf("  High-error pts:  %zu\n\n", result.n_high_error);

        // 2c. Check convergence
        bool converged = (result.p95_iv_error < target_iv_error_bp) &&
                        (result.fraction_below_1bp > 0.95);

        if (converged) {
            printf("✓ Target accuracy achieved!\n");
            printf("  Final grid size: %zu moneyness points\n", table->n_moneyness);
            printf("  P95 error: %.2f bp (target: %.2f bp)\n",
                   result.p95_iv_error, target_iv_error_bp);
            printf("  Coverage: %.1f%% below %.1f bp\n\n",
                   result.fraction_below_1bp * 100.0, target_iv_error_bp);

            validation_result_free(&result);
            return 0;  // Success
        }

        // 2d. Check if this is last iteration
        if (iter == max_iterations - 1) {
            printf("⚠ Maximum iterations reached without full convergence\n");
            printf("  Final P95 error: %.2f bp (target: %.2f bp)\n",
                   result.p95_iv_error, target_iv_error_bp);
            validation_result_free(&result);
            return 1;  // Partial success
        }

        // 2e. Identify refinement points
        printf("  Identifying refinement regions...\n");
        size_t n_new;
        double *new_points = identify_refinement_points(&result, table, &n_new);

        validation_result_free(&result);

        if (!new_points || n_new == 0) {
            printf("  No refinement points identified (might be at numerical limits)\n\n");
            return 1;  // Can't refine further
        }

        printf("  Adding %zu refinement points\n", n_new);

        // 2f. Expand grid
        int expand_status = price_table_expand_grid(table, new_points, n_new);
        free(new_points);

        if (expand_status != 0) {
            MANGO_TRACE_RUNTIME_ERROR(MODULE_VALIDATION, 2, iter);
            return -1;
        }

        printf("  New grid size: %zu moneyness points\n\n", table->n_moneyness);
    }

    // Should not reach here (loop should return from inside)
    return 1;
}
