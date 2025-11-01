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
