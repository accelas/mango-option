#include "price_table.h"
#include "american_option.h"
#include "interp_cubic.h"
#include "ivcalc_trace.h"
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <stdio.h>
#include <time.h>

#ifndef min
#define min(a, b) ((a) < (b) ? (a) : (b))
#endif

// File format constants
#define PRICE_TABLE_MAGIC 0x50545442  // "PTTB"
#define PRICE_TABLE_VERSION 4          // Version 4: adds thetas and rhos

// File header structure (Version 4)
typedef struct {
    uint32_t magic;
    uint32_t version;
    size_t n_moneyness;
    size_t n_maturity;
    size_t n_volatility;
    size_t n_rate;
    size_t n_dividend;
    OptionType type;
    ExerciseType exercise;
    char underlying[32];
    time_t generation_time;
    CoordinateSystem coord_system;    // Version 2+: coordinate transformation
    MemoryLayout memory_layout;       // Version 2+: memory layout strategy
    uint8_t has_gammas;               // Version 3+: 1 if gammas present, 0 otherwise
    uint8_t has_thetas;               // Version 4+: 1 if thetas present, 0 otherwise
    uint8_t has_rhos;                 // Version 4+: 1 if rhos present, 0 otherwise
    uint8_t padding[117];             // Reserved for future use (reduced from 119)
} PriceTableHeader;

// ---------- Helper Functions ----------

/**
 * Transform user coordinates to grid coordinates
 *
 * @param coord_system: Which transformation to apply
 * @param m_raw, tau_raw, sigma_raw, r_raw: User-provided raw coordinates
 * @param m_grid, tau_grid, sigma_grid, r_grid: [OUT] Grid coordinates
 */
void transform_query_to_grid(
    CoordinateSystem coord_system,
    double m_raw, double tau_raw, double sigma_raw, double r_raw,
    double *m_grid, double *tau_grid, double *sigma_grid, double *r_grid)
{
    switch (coord_system) {
        case COORD_RAW:
            *m_grid = m_raw;
            *tau_grid = tau_raw;
            break;

        case COORD_LOG_SQRT:
            *m_grid = log(m_raw);
            *tau_grid = sqrt(tau_raw);
            break;

        case COORD_LOG_VARIANCE:
            // Future implementation
            *m_grid = log(m_raw);
            *tau_grid = sigma_raw * sigma_raw * tau_raw;  // w = σ²T
            break;
    }

    // Volatility and rate always stay raw
    *sigma_grid = sigma_raw;
    *r_grid = r_raw;
}

/**
 * Convert grid point indices to OptionData structure.
 * Extracts grid values at the specified indices and constructs an option
 * with fixed reference strike K_ref = 100.0.
 *
 * Moneyness scaling approach:
 * - All precomputed prices use K_ref = 100.0 as the strike
 * - Moneyness m = S/K is stored in the grid
 * - Actual spot price S will be computed when needed: S = m * K_ref
 * - This allows one table to serve all strikes via moneyness interpolation
 *
 * Note: For continuous dividend yield, we pass it via a dummy discrete
 * dividend at t=0 with amount corresponding to the yield effect.
 * This is a simplification that works for the current implementation.
 */
static OptionData grid_point_to_option(const OptionPriceTable *table,
                                       [[maybe_unused]] size_t i_m,
                                       size_t i_tau,
                                       size_t i_sigma, size_t i_r,
                                       size_t i_q) {
    const double K_ref = 100.0;  // Reference strike for moneyness scaling

    // Reverse transform grid coordinates to raw coordinates for FDM solver
    // Grid stores transformed values, but FDM needs raw T
    // Note: moneyness not needed here - only used when extracting prices after solving
    double tau;
    switch (table->coord_system) {
        case COORD_RAW:
            tau = table->maturity_grid[i_tau];
            break;

        case COORD_LOG_SQRT:
            // Grid stores sqrt(T), reverse to get T
            tau = table->maturity_grid[i_tau] * table->maturity_grid[i_tau];
            break;

        case COORD_LOG_VARIANCE:
            // Grid stores w = σ²T, need to extract T
            // tau = w / σ² = maturity_grid[i_tau] / (sigma² )
            // For now, fall back to raw (this feature is incomplete)
            tau = table->maturity_grid[i_tau];
            break;

        default:
            // Fallback to raw for unknown coordinate systems
            tau = table->maturity_grid[i_tau];
            break;
    }

    double sigma = table->volatility_grid[i_sigma];
    double r = table->rate_grid[i_r];

    // Note: dividend (q) not used in OptionData here
    (void)i_q;

    OptionData option = {
        .strike = K_ref,
        .volatility = sigma,
        .risk_free_rate = r,
        .time_to_maturity = tau,
        .option_type = table->type,
        .n_dividends = 0,
        .dividend_times = NULL,
        .dividend_amounts = NULL
    };

    return option;
}

/**
 * Get batch size for parallel computation from environment variable.
 * Defaults to 100 if IVCALC_PRECOMPUTE_BATCH_SIZE is not set or invalid.
 * Valid range: 1 to 100000.
 */
static size_t get_batch_size(void) {
    size_t batch_size = 100;  // Default

    char *env_batch = getenv("IVCALC_PRECOMPUTE_BATCH_SIZE");
    if (env_batch) {
        long val = atol(env_batch);
        if (val >= 1 && val <= 100000) {
            batch_size = (size_t)val;
        }
    }

    return batch_size;
}

// ---------- Creation and Destruction ----------

/**
 * Compute strides based on memory layout
 */
static void compute_strides(OptionPriceTable *table) {
    size_t n_m = table->n_moneyness;
    size_t n_tau = table->n_maturity;
    size_t n_sigma = table->n_volatility;
    size_t n_r = table->n_rate;
    size_t n_q = table->n_dividend > 0 ? table->n_dividend : 1;

    switch (table->memory_layout) {
        case LAYOUT_M_OUTER:  // Current: [m][tau][sigma][r][q]
            if (table->n_dividend > 0) {
                table->stride_q = 1;
                table->stride_r = n_q;
                table->stride_sigma = n_r * n_q;
                table->stride_tau = n_sigma * n_r * n_q;
                table->stride_m = n_tau * n_sigma * n_r * n_q;
            } else {
                table->stride_q = 0;
                table->stride_r = 1;
                table->stride_sigma = n_r;
                table->stride_tau = n_sigma * n_r;
                table->stride_m = n_tau * n_sigma * n_r;
            }
            break;

        case LAYOUT_M_INNER:  // Optimized: [q][r][sigma][tau][m]
            if (table->n_dividend > 0) {
                table->stride_m = 1;
                table->stride_tau = n_m;
                table->stride_sigma = n_tau * n_m;
                table->stride_r = n_sigma * n_tau * n_m;
                table->stride_q = n_r * n_sigma * n_tau * n_m;
            } else {
                table->stride_m = 1;
                table->stride_tau = n_m;
                table->stride_sigma = n_tau * n_m;
                table->stride_r = n_sigma * n_tau * n_m;
                table->stride_q = 0;
            }
            break;

        case LAYOUT_BLOCKED:
            // Future: fall back to M_INNER for now
            table->memory_layout = LAYOUT_M_INNER;
            compute_strides(table);  // Recursive call
            break;
    }
}

OptionPriceTable* price_table_create_with_strategy(
    const double *moneyness, size_t n_m,
    const double *maturity, size_t n_tau,
    const double *volatility, size_t n_sigma,
    const double *rate, size_t n_r,
    const double *dividend, size_t n_q,
    OptionType type, ExerciseType exercise,
    const InterpolationStrategy *strategy) {

    if (!moneyness || !maturity || !volatility || !rate) return NULL;
    if (n_m == 0 || n_tau == 0 || n_sigma == 0 || n_r == 0) return NULL;
    if (n_q > 0 && !dividend) return NULL;

    // Default to cubic if no strategy specified
    if (!strategy) {
        strategy = &INTERP_CUBIC;
    }

    OptionPriceTable *table = calloc(1, sizeof(OptionPriceTable));
    if (!table) return NULL;

    // Copy dimensions
    table->n_moneyness = n_m;
    table->n_maturity = n_tau;
    table->n_volatility = n_sigma;
    table->n_rate = n_r;
    table->n_dividend = n_q;

    // Allocate and copy grids
    table->moneyness_grid = malloc(n_m * sizeof(double));
    table->maturity_grid = malloc(n_tau * sizeof(double));
    table->volatility_grid = malloc(n_sigma * sizeof(double));
    table->rate_grid = malloc(n_r * sizeof(double));
    table->dividend_grid = (n_q > 0) ? malloc(n_q * sizeof(double)) : NULL;

    if (!table->moneyness_grid || !table->maturity_grid ||
        !table->volatility_grid || !table->rate_grid ||
        (n_q > 0 && !table->dividend_grid)) {
        free(table->moneyness_grid);
        free(table->maturity_grid);
        free(table->volatility_grid);
        free(table->rate_grid);
        free(table->dividend_grid);
        free(table);
        return NULL;
    }

    memcpy(table->moneyness_grid, moneyness, n_m * sizeof(double));
    memcpy(table->maturity_grid, maturity, n_tau * sizeof(double));
    memcpy(table->volatility_grid, volatility, n_sigma * sizeof(double));
    memcpy(table->rate_grid, rate, n_r * sizeof(double));
    if (n_q > 0) {
        memcpy(table->dividend_grid, dividend, n_q * sizeof(double));
    }

    // Allocate price array
    size_t n_points = n_m * n_tau * n_sigma * n_r * (n_q > 0 ? n_q : 1);
    table->prices = malloc(n_points * sizeof(double));
    if (!table->prices) {
        free(table->moneyness_grid);
        free(table->maturity_grid);
        free(table->volatility_grid);
        free(table->rate_grid);
        free(table->dividend_grid);
        free(table);
        return NULL;
    }

    // Initialize prices to NaN
    #pragma omp simd
    for (size_t i = 0; i < n_points; i++) {
        table->prices[i] = NAN;
    }

    // Vega array - not allocated in create, only during precompute
    table->vegas = NULL;

    // Set metadata
    table->type = type;
    table->exercise = exercise;
    memset(table->underlying, 0, sizeof(table->underlying));
    table->generation_time = time(NULL);

    // Set transformation config (defaults for backward compatibility)
    table->coord_system = COORD_RAW;
    table->memory_layout = LAYOUT_M_OUTER;

    // Compute strides based on layout
    compute_strides(table);

    // Set strategy
    table->strategy = strategy;
    size_t dimensions = (n_q > 0) ? 5 : 4;
    size_t grid_sizes[5] = {n_m, n_tau, n_sigma, n_r, n_q};
    table->interp_context = NULL;
    if (strategy->create_context) {
        table->interp_context = strategy->create_context(dimensions, grid_sizes);
    }

    return table;
}

OptionPriceTable* price_table_create_ex(
    const double *moneyness, size_t n_m,
    const double *maturity, size_t n_tau,
    const double *volatility, size_t n_sigma,
    const double *rate, size_t n_r,
    const double *dividend, size_t n_q,
    OptionType type, ExerciseType exercise,
    CoordinateSystem coord_system,
    MemoryLayout memory_layout)
{
    // Validation
    if (!moneyness || !maturity || !volatility || !rate) return NULL;
    if (n_m == 0 || n_tau == 0 || n_sigma == 0 || n_r == 0) return NULL;

    OptionPriceTable *table = calloc(1, sizeof(OptionPriceTable));
    if (!table) return NULL;

    // Set dimensions
    table->n_moneyness = n_m;
    table->n_maturity = n_tau;
    table->n_volatility = n_sigma;
    table->n_rate = n_r;
    table->n_dividend = n_q;

    // Set transformation config
    table->coord_system = coord_system;
    table->memory_layout = memory_layout;

    // Allocate grids
    table->moneyness_grid = malloc(n_m * sizeof(double));
    table->maturity_grid = malloc(n_tau * sizeof(double));
    table->volatility_grid = malloc(n_sigma * sizeof(double));
    table->rate_grid = malloc(n_r * sizeof(double));
    table->dividend_grid = n_q > 0 ? malloc(n_q * sizeof(double)) : NULL;

    if (!table->moneyness_grid || !table->maturity_grid ||
        !table->volatility_grid || !table->rate_grid ||
        (n_q > 0 && !table->dividend_grid)) {
        free(table->moneyness_grid);
        free(table->maturity_grid);
        free(table->volatility_grid);
        free(table->rate_grid);
        free(table->dividend_grid);
        free(table);
        return NULL;
    }

    // Copy grids
    memcpy(table->moneyness_grid, moneyness, n_m * sizeof(double));
    memcpy(table->maturity_grid, maturity, n_tau * sizeof(double));
    memcpy(table->volatility_grid, volatility, n_sigma * sizeof(double));
    memcpy(table->rate_grid, rate, n_r * sizeof(double));
    if (n_q > 0) {
        memcpy(table->dividend_grid, dividend, n_q * sizeof(double));
    }

    // Allocate prices array
    size_t n_total = n_m * n_tau * n_sigma * n_r * (n_q > 0 ? n_q : 1);
    table->prices = malloc(n_total * sizeof(double));
    if (!table->prices) {
        free(table->moneyness_grid);
        free(table->maturity_grid);
        free(table->volatility_grid);
        free(table->rate_grid);
        free(table->dividend_grid);
        free(table);
        return NULL;
    }

    // Initialize prices to NAN
    #pragma omp simd
    for (size_t i = 0; i < n_total; i++) {
        table->prices[i] = NAN;
    }

    // Greeks arrays - not allocated in create, only during precompute
    table->vegas = NULL;
    table->gammas = NULL;
    table->thetas = NULL;
    table->rhos = NULL;

    // Set metadata
    table->type = type;
    table->exercise = exercise;
    memset(table->underlying, 0, sizeof(table->underlying));
    table->generation_time = time(NULL);

    // Compute strides based on layout
    compute_strides(table);

    // Set interpolation strategy to cubic (default)
    table->strategy = &INTERP_CUBIC;
    table->interp_context = NULL;

    return table;
}

OptionPriceTable* price_table_create(
    const double *moneyness, size_t n_m,
    const double *maturity, size_t n_tau,
    const double *volatility, size_t n_sigma,
    const double *rate, size_t n_r,
    const double *dividend, size_t n_q,
    OptionType type, ExerciseType exercise) {
    // Delegate to _ex with default settings
    return price_table_create_ex(
        moneyness, n_m, maturity, n_tau,
        volatility, n_sigma, rate, n_r,
        dividend, n_q, type, exercise,
        COORD_RAW,      // Default: no transformation
        LAYOUT_M_OUTER  // Default: current layout
    );
}

void price_table_destroy(OptionPriceTable *table) {
    if (!table) return;

    // Destroy interpolation context
    if (table->strategy && table->strategy->destroy_context) {
        table->strategy->destroy_context(table->interp_context);
    }

    // Free arrays
    free(table->moneyness_grid);
    free(table->maturity_grid);
    free(table->volatility_grid);
    free(table->rate_grid);
    free(table->dividend_grid);
    free(table->prices);
    free(table->vegas);
    free(table->gammas);
    free(table->thetas);
    free(table->rhos);
    free(table);
}

// ---------- Pre-computation ----------

int price_table_precompute(OptionPriceTable *table,
                            const AmericanOptionGrid *grid) {
    if (!table || !grid || !table->prices) {
        return -1;
    }

    // Calculate total grid points
    size_t n_total = table->n_moneyness * table->n_maturity *
                     table->n_volatility * table->n_rate;
    if (table->n_dividend > 0) {
        n_total *= table->n_dividend;
    }

    // Allocate vega array if not already allocated
    if (!table->vegas) {
        table->vegas = malloc(n_total * sizeof(double));
        if (!table->vegas) {
            return -1;
        }
        // Initialize to NaN
        for (size_t i = 0; i < n_total; i++) {
            table->vegas[i] = NAN;
        }
    }

    // Allocate gamma array if not already allocated
    if (!table->gammas) {
        table->gammas = malloc(n_total * sizeof(double));
        if (!table->gammas) {
            return -1;
        }
        // Initialize to NaN
        for (size_t i = 0; i < n_total; i++) {
            table->gammas[i] = NAN;
        }
    }

    // Allocate theta array if not already allocated
    if (!table->thetas) {
        table->thetas = malloc(n_total * sizeof(double));
        if (!table->thetas) {
            return -1;
        }
        // Initialize to NaN
        for (size_t i = 0; i < n_total; i++) {
            table->thetas[i] = NAN;
        }
    }

    // Allocate rho array if not already allocated
    if (!table->rhos) {
        table->rhos = malloc(n_total * sizeof(double));
        if (!table->rhos) {
            return -1;
        }
        // Initialize to NaN
        for (size_t i = 0; i < n_total; i++) {
            table->rhos[i] = NAN;
        }
    }

    size_t batch_size = get_batch_size();

    // Allocate batch arrays
    OptionData *batch_options = malloc(batch_size * sizeof(OptionData));
    AmericanOptionResult *batch_results = malloc(batch_size * sizeof(AmericanOptionResult));

    if (!batch_options || !batch_results) {
        free(batch_options);
        free(batch_results);
        return -1;
    }

    MANGO_TRACE_ALGO_START(MODULE_PRICE_TABLE, n_total, batch_size, 0);

    const double K_ref = 100.0;  // Reference strike for moneyness scaling
    size_t completed = 0;

    // Process each maturity separately with adaptive time steps
    for (size_t i_tau = 0; i_tau < table->n_maturity; i_tau++) {
        double tau_grid = table->maturity_grid[i_tau];

        // Reverse transform maturity to get raw T for time step calculation
        double tau_raw;
        switch (table->coord_system) {
            case COORD_RAW:
                tau_raw = tau_grid;
                break;
            case COORD_LOG_SQRT:
                tau_raw = tau_grid * tau_grid;  // sqrt(T) → T
                break;
            case COORD_LOG_VARIANCE:
                // Incomplete: need sigma to extract T from w = σ²T
                tau_raw = tau_grid;  // Fall back
                break;
            default:
                tau_raw = tau_grid;  // Fallback to raw
                break;
        }

        // Create adaptive grid for this maturity
        AmericanOptionGrid adaptive_grid = *grid;  // Copy base grid
        adaptive_grid.n_steps = (size_t)(tau_raw / grid->dt);  // Adaptive time steps
        if (adaptive_grid.n_steps < 10) adaptive_grid.n_steps = 10;  // Minimum steps

        // Calculate points for this maturity slice
        size_t points_per_maturity = table->n_moneyness * table->n_volatility * table->n_rate;
        if (table->n_dividend > 0) {
            points_per_maturity *= table->n_dividend;
        }

        // Process this maturity slice in batches
        for (size_t slice_start = 0; slice_start < points_per_maturity; slice_start += batch_size) {
            size_t batch_count = min(batch_size, points_per_maturity - slice_start);

            // Fill batch with points from this maturity slice
            for (size_t i = 0; i < batch_count; i++) {
                size_t slice_idx = slice_start + i;

                // Decompose slice index into other dimensions
                size_t i_m, i_sigma, i_r, i_q;
                if (table->n_dividend > 0) {
                    size_t per_dividend = table->n_moneyness * table->n_volatility * table->n_rate;
                    i_q = slice_idx / per_dividend;
                    slice_idx %= per_dividend;
                } else {
                    i_q = 0;
                }

                size_t per_rate = table->n_moneyness * table->n_volatility;
                i_r = slice_idx / per_rate;
                slice_idx %= per_rate;

                size_t per_sigma = table->n_moneyness;
                i_sigma = slice_idx / per_sigma;
                i_m = slice_idx % per_sigma;

                batch_options[i] = grid_point_to_option(table, i_m, i_tau,
                                                         i_sigma, i_r, i_q);
            }

            // Solve batch with maturity-specific grid
            int status = american_option_price_batch(batch_options, &adaptive_grid,
                                                      batch_count, batch_results);
            if (status != 0) {
                free(batch_options);
                free(batch_results);
                MANGO_TRACE_RUNTIME_ERROR(MODULE_PRICE_TABLE, status, completed);
                return -1;
            }

            // Store results in table and free solvers
            for (size_t i = 0; i < batch_count; i++) {
                size_t slice_idx = slice_start + i;

                // Reconstruct global index
                size_t i_m, i_sigma, i_r, i_q;
                if (table->n_dividend > 0) {
                    size_t per_dividend = table->n_moneyness * table->n_volatility * table->n_rate;
                    i_q = slice_idx / per_dividend;
                    slice_idx %= per_dividend;
                } else {
                    i_q = 0;
                }

                size_t per_rate = table->n_moneyness * table->n_volatility;
                i_r = slice_idx / per_rate;
                slice_idx %= per_rate;

                size_t per_sigma = table->n_moneyness;
                i_sigma = slice_idx / per_sigma;
                i_m = slice_idx % per_sigma;

                // Calculate global index
                size_t idx = i_m * table->stride_m + i_tau * table->stride_tau
                           + i_sigma * table->stride_sigma + i_r * table->stride_r
                           + i_q * table->stride_q;

                // Extract moneyness for this grid point
                // Reverse transform if using coordinate transformations
                double m_grid = table->moneyness_grid[i_m];
                double m_raw;
                switch (table->coord_system) {
                    case COORD_RAW:
                        m_raw = m_grid;
                        break;
                    case COORD_LOG_SQRT:
                    case COORD_LOG_VARIANCE:
                        m_raw = exp(m_grid);  // Grid stores log(m), convert to m
                        break;
                    default:
                        m_raw = m_grid;  // Fallback to raw
                        break;
                }
                double spot_price = m_raw * K_ref;

                // Extract price at the spot price
                double price = american_option_get_value_at_spot(
                    batch_results[i].solver, spot_price, K_ref);

                table->prices[idx] = price;

                // Free the solver
                pde_solver_destroy(batch_results[i].solver);
            }

            completed += batch_count;

            // Progress tracking (every 10 batches)
            if ((completed / batch_size) % 10 == 0) {
                MANGO_TRACE_ALGO_PROGRESS(MODULE_PRICE_TABLE,
                                           completed, n_total,
                                           (double)completed / (double)n_total);
            }
        }
    }

    // Second pass: Compute vega via finite differences
    // Restructured for SIMD vectorization: handle boundary cases separately from interior points

    size_t n_q_effective = table->n_dividend > 0 ? table->n_dividend : 1;

    // Handle lower boundary (i_sigma == 0) with forward differences
    if (table->n_volatility > 1) {
        double sigma_0 = table->volatility_grid[0];
        double sigma_1 = table->volatility_grid[1];
        double h_forward = sigma_1 - sigma_0;

        for (size_t i_m = 0; i_m < table->n_moneyness; i_m++) {
            for (size_t i_tau = 0; i_tau < table->n_maturity; i_tau++) {
                for (size_t i_r = 0; i_r < table->n_rate; i_r++) {
                    #pragma omp simd
                    for (size_t i_q = 0; i_q < n_q_effective; i_q++) {
                        size_t idx = i_m * table->stride_m + i_tau * table->stride_tau
                                   + 0 * table->stride_sigma + i_r * table->stride_r
                                   + i_q * table->stride_q;
                        size_t idx_next = idx + table->stride_sigma;

                        double price_current = table->prices[idx];
                        double price_next = table->prices[idx_next];

                        table->vegas[idx] = (!isnan(price_current) && !isnan(price_next))
                            ? (price_next - price_current) / h_forward
                            : NAN;
                    }
                }
            }
        }
    }

    // Handle interior points (0 < i_sigma < n-1) with centered differences (SIMD-friendly)
    if (table->n_volatility > 2) {
        for (size_t i_m = 0; i_m < table->n_moneyness; i_m++) {
            for (size_t i_tau = 0; i_tau < table->n_maturity; i_tau++) {
                for (size_t i_sigma = 1; i_sigma < table->n_volatility - 1; i_sigma++) {
                    double sigma_minus = table->volatility_grid[i_sigma - 1];
                    double sigma_plus = table->volatility_grid[i_sigma + 1];
                    double h_centered = sigma_plus - sigma_minus;

                    for (size_t i_r = 0; i_r < table->n_rate; i_r++) {
                        #pragma omp simd
                        for (size_t i_q = 0; i_q < n_q_effective; i_q++) {
                            size_t idx = i_m * table->stride_m + i_tau * table->stride_tau
                                       + i_sigma * table->stride_sigma + i_r * table->stride_r
                                       + i_q * table->stride_q;
                            size_t idx_minus = idx - table->stride_sigma;
                            size_t idx_plus = idx + table->stride_sigma;

                            double price_minus = table->prices[idx_minus];
                            double price_plus = table->prices[idx_plus];

                            table->vegas[idx] = (!isnan(price_minus) && !isnan(price_plus))
                                ? (price_plus - price_minus) / h_centered
                                : NAN;
                        }
                    }
                }
            }
        }
    }

    // Handle upper boundary (i_sigma == n-1) with backward differences
    if (table->n_volatility > 1) {
        size_t i_sigma_last = table->n_volatility - 1;
        double sigma_last = table->volatility_grid[i_sigma_last];
        double sigma_prev = table->volatility_grid[i_sigma_last - 1];
        double h_backward = sigma_last - sigma_prev;

        for (size_t i_m = 0; i_m < table->n_moneyness; i_m++) {
            for (size_t i_tau = 0; i_tau < table->n_maturity; i_tau++) {
                for (size_t i_r = 0; i_r < table->n_rate; i_r++) {
                    #pragma omp simd
                    for (size_t i_q = 0; i_q < n_q_effective; i_q++) {
                        size_t idx = i_m * table->stride_m + i_tau * table->stride_tau
                                   + i_sigma_last * table->stride_sigma + i_r * table->stride_r
                                   + i_q * table->stride_q;
                        size_t idx_prev = idx - table->stride_sigma;

                        double price_current = table->prices[idx];
                        double price_prev = table->prices[idx_prev];

                        table->vegas[idx] = (!isnan(price_current) && !isnan(price_prev))
                            ? (price_current - price_prev) / h_backward
                            : NAN;
                    }
                }
            }
        }
    }

    // Third pass: Compute gamma via finite differences on moneyness axis
    // Note: γ = ∂²V/∂S², not ∂²V/∂m². Since m = S/K_ref, we have:
    // ∂²V/∂S² = ∂²V/∂m² · (∂m/∂S)² = ∂²V/∂m² / K_ref²
    const double K_ref_sq = K_ref * K_ref;  // 10000

    // Handle lower boundary (i_m == 0) with forward differences
    if (table->n_moneyness > 2) {
        for (size_t i_tau = 0; i_tau < table->n_maturity; i_tau++) {
            for (size_t i_sigma = 0; i_sigma < table->n_volatility; i_sigma++) {
                for (size_t i_r = 0; i_r < table->n_rate; i_r++) {
                    for (size_t i_q = 0; i_q < n_q_effective; i_q++) {
                        size_t idx0 = 0 * table->stride_m + i_tau * table->stride_tau
                                   + i_sigma * table->stride_sigma + i_r * table->stride_r
                                   + i_q * table->stride_q;
                        size_t idx1 = idx0 + table->stride_m;
                        size_t idx2 = idx0 + 2 * table->stride_m;

                        double V0 = table->prices[idx0];
                        double V1 = table->prices[idx1];
                        double V2 = table->prices[idx2];

                        if (table->coord_system == COORD_LOG_SQRT) {
                            // Transform from log-space to raw space
                            double m0 = exp(table->moneyness_grid[0]);
                            double h = table->moneyness_grid[1] - table->moneyness_grid[0];

                            if (!isnan(V0) && !isnan(V1) && !isnan(V2)) {
                                double d2V = (V2 - 2*V1 + V0) / (h * h);
                                double dV = (V1 - V0) / h;
                                double d2V_dm2 = (d2V - dV) / (m0 * m0);
                                table->gammas[idx0] = d2V_dm2 / K_ref_sq;  // Convert to ∂²V/∂S²
                            }
                        } else {
                            // Raw coordinates - direct computation
                            double h = table->moneyness_grid[1] - table->moneyness_grid[0];
                            if (!isnan(V0) && !isnan(V1) && !isnan(V2)) {
                                double d2V_dm2 = (V2 - 2*V1 + V0) / (h * h);
                                table->gammas[idx0] = d2V_dm2 / K_ref_sq;  // Convert to ∂²V/∂S²
                            }
                        }
                    }
                }
            }
        }
    }

    // Interior points - centered differences with SIMD vectorization
    if (table->n_moneyness > 2) {
        for (size_t i_tau = 0; i_tau < table->n_maturity; i_tau++) {
            for (size_t i_sigma = 0; i_sigma < table->n_volatility; i_sigma++) {
                for (size_t i_r = 0; i_r < table->n_rate; i_r++) {
                    for (size_t i_q = 0; i_q < n_q_effective; i_q++) {
                        #pragma omp simd
                        for (size_t i_m = 1; i_m < table->n_moneyness - 1; i_m++) {
                            size_t idx = i_m * table->stride_m + i_tau * table->stride_tau
                                       + i_sigma * table->stride_sigma + i_r * table->stride_r
                                       + i_q * table->stride_q;
                            size_t idx_minus = idx - table->stride_m;
                            size_t idx_plus = idx + table->stride_m;

                            double V_minus = table->prices[idx_minus];
                            double V = table->prices[idx];
                            double V_plus = table->prices[idx_plus];

                            if (table->coord_system == COORD_LOG_SQRT) {
                                // Transform from log-space to raw space
                                double m = exp(table->moneyness_grid[i_m]);
                                double h = table->moneyness_grid[i_m+1] - table->moneyness_grid[i_m];

                                if (!isnan(V_minus) && !isnan(V_plus)) {
                                    double d2V_dlogm2 = (V_plus - 2*V + V_minus) / (h * h);
                                    double dV_dlogm = (V_plus - V_minus) / (2 * h);
                                    double d2V_dm2 = (d2V_dlogm2 - dV_dlogm) / (m * m);
                                    table->gammas[idx] = d2V_dm2 / K_ref_sq;  // Convert to ∂²V/∂S²
                                }
                            } else {
                                // Raw coordinates - direct computation
                                double h = table->moneyness_grid[i_m+1] - table->moneyness_grid[i_m];
                                if (!isnan(V_minus) && !isnan(V_plus)) {
                                    double d2V_dm2 = (V_plus - 2*V + V_minus) / (h * h);
                                    table->gammas[idx] = d2V_dm2 / K_ref_sq;  // Convert to ∂²V/∂S²
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    // Upper boundary (i_m == n_moneyness-1) with backward differences
    if (table->n_moneyness > 2) {
        size_t i_m_last = table->n_moneyness - 1;
        for (size_t i_tau = 0; i_tau < table->n_maturity; i_tau++) {
            for (size_t i_sigma = 0; i_sigma < table->n_volatility; i_sigma++) {
                for (size_t i_r = 0; i_r < table->n_rate; i_r++) {
                    for (size_t i_q = 0; i_q < n_q_effective; i_q++) {
                        size_t idx = i_m_last * table->stride_m + i_tau * table->stride_tau
                                   + i_sigma * table->stride_sigma + i_r * table->stride_r
                                   + i_q * table->stride_q;
                        size_t idx_minus1 = idx - table->stride_m;
                        size_t idx_minus2 = idx - 2 * table->stride_m;

                        double V = table->prices[idx];
                        double V1 = table->prices[idx_minus1];
                        double V2 = table->prices[idx_minus2];

                        if (table->coord_system == COORD_LOG_SQRT) {
                            // Transform from log-space to raw space
                            double m = exp(table->moneyness_grid[i_m_last]);
                            double h = table->moneyness_grid[i_m_last] - table->moneyness_grid[i_m_last-1];

                            if (!isnan(V) && !isnan(V1) && !isnan(V2)) {
                                double d2V = (V - 2*V1 + V2) / (h * h);
                                double dV = (V - V1) / h;
                                double d2V_dm2 = (d2V - dV) / (m * m);
                                table->gammas[idx] = d2V_dm2 / K_ref_sq;  // Convert to ∂²V/∂S²
                            }
                        } else {
                            // Raw coordinates - direct computation
                            double h = table->moneyness_grid[i_m_last] - table->moneyness_grid[i_m_last-1];
                            if (!isnan(V) && !isnan(V1) && !isnan(V2)) {
                                double d2V_dm2 = (V - 2*V1 + V2) / (h * h);
                                table->gammas[idx] = d2V_dm2 / K_ref_sq;  // Convert to ∂²V/∂S²
                            }
                        }
                    }
                }
            }
        }
    }

    // Fourth pass: Compute theta via finite differences on maturity axis
    // Note: θ = -∂V/∂τ (negative time derivative for time decay)

    // Handle lower boundary (i_tau == 0) with forward differences
    if (table->n_maturity > 1) {
        double tau_0 = table->maturity_grid[0];
        double tau_1 = table->maturity_grid[1];
        double h_forward = tau_1 - tau_0;

        for (size_t i_m = 0; i_m < table->n_moneyness; i_m++) {
            for (size_t i_sigma = 0; i_sigma < table->n_volatility; i_sigma++) {
                for (size_t i_r = 0; i_r < table->n_rate; i_r++) {
                    #pragma omp simd
                    for (size_t i_q = 0; i_q < n_q_effective; i_q++) {
                        size_t idx = i_m * table->stride_m + 0 * table->stride_tau
                                   + i_sigma * table->stride_sigma + i_r * table->stride_r
                                   + i_q * table->stride_q;
                        size_t idx_next = idx + table->stride_tau;

                        double price_current = table->prices[idx];
                        double price_next = table->prices[idx_next];

                        // Theta is negative of time derivative
                        table->thetas[idx] = (!isnan(price_current) && !isnan(price_next))
                            ? -(price_next - price_current) / h_forward
                            : NAN;
                    }
                }
            }
        }
    }

    // Handle interior points (0 < i_tau < n-1) with centered differences (SIMD-friendly)
    if (table->n_maturity > 2) {
        for (size_t i_m = 0; i_m < table->n_moneyness; i_m++) {
            for (size_t i_sigma = 0; i_sigma < table->n_volatility; i_sigma++) {
                for (size_t i_tau = 1; i_tau < table->n_maturity - 1; i_tau++) {
                    double tau_minus = table->maturity_grid[i_tau - 1];
                    double tau_plus = table->maturity_grid[i_tau + 1];
                    double h_centered = tau_plus - tau_minus;

                    for (size_t i_r = 0; i_r < table->n_rate; i_r++) {
                        #pragma omp simd
                        for (size_t i_q = 0; i_q < n_q_effective; i_q++) {
                            size_t idx = i_m * table->stride_m + i_tau * table->stride_tau
                                       + i_sigma * table->stride_sigma + i_r * table->stride_r
                                       + i_q * table->stride_q;
                            size_t idx_minus = idx - table->stride_tau;
                            size_t idx_plus = idx + table->stride_tau;

                            double price_minus = table->prices[idx_minus];
                            double price_plus = table->prices[idx_plus];

                            // Theta is negative of time derivative
                            table->thetas[idx] = (!isnan(price_minus) && !isnan(price_plus))
                                ? -(price_plus - price_minus) / h_centered
                                : NAN;
                        }
                    }
                }
            }
        }
    }

    // Handle upper boundary (i_tau == n-1) with backward differences
    if (table->n_maturity > 1) {
        size_t i_tau_last = table->n_maturity - 1;
        double tau_last = table->maturity_grid[i_tau_last];
        double tau_prev = table->maturity_grid[i_tau_last - 1];
        double h_backward = tau_last - tau_prev;

        for (size_t i_m = 0; i_m < table->n_moneyness; i_m++) {
            for (size_t i_sigma = 0; i_sigma < table->n_volatility; i_sigma++) {
                for (size_t i_r = 0; i_r < table->n_rate; i_r++) {
                    #pragma omp simd
                    for (size_t i_q = 0; i_q < n_q_effective; i_q++) {
                        size_t idx = i_m * table->stride_m + i_tau_last * table->stride_tau
                                   + i_sigma * table->stride_sigma + i_r * table->stride_r
                                   + i_q * table->stride_q;
                        size_t idx_prev = idx - table->stride_tau;

                        double price_current = table->prices[idx];
                        double price_prev = table->prices[idx_prev];

                        // Theta is negative of time derivative
                        table->thetas[idx] = (!isnan(price_current) && !isnan(price_prev))
                            ? -(price_current - price_prev) / h_backward
                            : NAN;
                    }
                }
            }
        }
    }

    // Fifth pass: Compute rho via finite differences on rate axis
    // ρ = ∂V/∂r (interest rate sensitivity)

    // Handle lower boundary (i_r == 0) with forward differences
    if (table->n_rate > 1) {
        double r_0 = table->rate_grid[0];
        double r_1 = table->rate_grid[1];
        double h_forward = r_1 - r_0;

        for (size_t i_m = 0; i_m < table->n_moneyness; i_m++) {
            for (size_t i_tau = 0; i_tau < table->n_maturity; i_tau++) {
                for (size_t i_sigma = 0; i_sigma < table->n_volatility; i_sigma++) {
                    #pragma omp simd
                    for (size_t i_q = 0; i_q < n_q_effective; i_q++) {
                        size_t idx = i_m * table->stride_m + i_tau * table->stride_tau
                                   + i_sigma * table->stride_sigma + 0 * table->stride_r
                                   + i_q * table->stride_q;
                        size_t idx_next = idx + table->stride_r;

                        double price_current = table->prices[idx];
                        double price_next = table->prices[idx_next];

                        table->rhos[idx] = (!isnan(price_current) && !isnan(price_next))
                            ? (price_next - price_current) / h_forward
                            : NAN;
                    }
                }
            }
        }
    }

    // Handle interior points (0 < i_r < n-1) with centered differences (SIMD-friendly)
    if (table->n_rate > 2) {
        for (size_t i_m = 0; i_m < table->n_moneyness; i_m++) {
            for (size_t i_tau = 0; i_tau < table->n_maturity; i_tau++) {
                for (size_t i_sigma = 0; i_sigma < table->n_volatility; i_sigma++) {
                    for (size_t i_r = 1; i_r < table->n_rate - 1; i_r++) {
                        double r_minus = table->rate_grid[i_r - 1];
                        double r_plus = table->rate_grid[i_r + 1];
                        double h_centered = r_plus - r_minus;

                        #pragma omp simd
                        for (size_t i_q = 0; i_q < n_q_effective; i_q++) {
                            size_t idx = i_m * table->stride_m + i_tau * table->stride_tau
                                       + i_sigma * table->stride_sigma + i_r * table->stride_r
                                       + i_q * table->stride_q;
                            size_t idx_minus = idx - table->stride_r;
                            size_t idx_plus = idx + table->stride_r;

                            double price_minus = table->prices[idx_minus];
                            double price_plus = table->prices[idx_plus];

                            table->rhos[idx] = (!isnan(price_minus) && !isnan(price_plus))
                                ? (price_plus - price_minus) / h_centered
                                : NAN;
                        }
                    }
                }
            }
        }
    }

    // Handle upper boundary (i_r == n-1) with backward differences
    if (table->n_rate > 1) {
        size_t i_r_last = table->n_rate - 1;
        double r_last = table->rate_grid[i_r_last];
        double r_prev = table->rate_grid[i_r_last - 1];
        double h_backward = r_last - r_prev;

        for (size_t i_m = 0; i_m < table->n_moneyness; i_m++) {
            for (size_t i_tau = 0; i_tau < table->n_maturity; i_tau++) {
                for (size_t i_sigma = 0; i_sigma < table->n_volatility; i_sigma++) {
                    #pragma omp simd
                    for (size_t i_q = 0; i_q < n_q_effective; i_q++) {
                        size_t idx = i_m * table->stride_m + i_tau * table->stride_tau
                                   + i_sigma * table->stride_sigma + i_r_last * table->stride_r
                                   + i_q * table->stride_q;
                        size_t idx_prev = idx - table->stride_r;

                        double price_current = table->prices[idx];
                        double price_prev = table->prices[idx_prev];

                        table->rhos[idx] = (!isnan(price_current) && !isnan(price_prev))
                            ? (price_current - price_prev) / h_backward
                            : NAN;
                    }
                }
            }
        }
    }

    MANGO_TRACE_ALGO_COMPLETE(MODULE_PRICE_TABLE, n_total, 1.0);

    free(batch_options);
    free(batch_results);

    // Mark table with generation timestamp
    table->generation_time = time(NULL);

    return 0;
}

// Helper: Binary search to find index of value in sorted array
static int find_moneyness_index(const double *grid, size_t n, double value) {
    // Binary search with tolerance for floating point comparison
    const double tol = 1e-10;

    for (size_t i = 0; i < n; i++) {
        if (fabs(grid[i] - value) < tol) {
            return (int)i;
        }
    }
    return -1;  // Not found
}

// Helper: Comparison function for qsort
static int compare_doubles(const void *a, const void *b) {
    double diff = *(const double*)a - *(const double*)b;
    return (diff > 0) - (diff < 0);
}

// Helper: Merge and sort arrays, removing duplicates
static double* merge_and_sort_unique(
    const double *old_grid, size_t n_old,
    const double *new_points, size_t n_new,
    size_t *n_out
) {
    // Allocate temporary array for merging
    double *merged = malloc((n_old + n_new) * sizeof(double));
    if (!merged) return NULL;

    // Copy old grid
    memcpy(merged, old_grid, n_old * sizeof(double));

    // Copy new points
    memcpy(merged + n_old, new_points, n_new * sizeof(double));

    // Sort combined array
    qsort(merged, n_old + n_new, sizeof(double), compare_doubles);

    // Remove duplicates with tolerance
    const double tol = 1e-10;
    size_t write_idx = 0;

    for (size_t read_idx = 0; read_idx < n_old + n_new; read_idx++) {
        // Skip if duplicate of previous value
        if (read_idx > 0 && fabs(merged[read_idx] - merged[write_idx - 1]) < tol) {
            continue;
        }

        merged[write_idx++] = merged[read_idx];
    }

    *n_out = write_idx;
    return merged;
}

int price_table_expand_grid(OptionPriceTable *table,
                            const double *new_m_points,
                            size_t n_new) {
    // 1. Validate inputs
    if (!table || !new_m_points || n_new == 0) {
        MANGO_TRACE_VALIDATION_ERROR(MODULE_PRICE_TABLE, 0, n_new, 0.0);
        return -1;
    }

    // Verify LAYOUT_M_INNER (required for unified grid)
    if (table->memory_layout != LAYOUT_M_INNER) {
        MANGO_TRACE_VALIDATION_ERROR(MODULE_PRICE_TABLE, 1, table->memory_layout, LAYOUT_M_INNER);
        return -1;
    }

    // Validate all new points are positive
    for (size_t i = 0; i < n_new; i++) {
        if (new_m_points[i] <= 0.0) {
            MANGO_TRACE_VALIDATION_ERROR(MODULE_PRICE_TABLE, 2, i, new_m_points[i]);
            return -1;
        }
    }

    // 2. Merge and sort grids, removing duplicates
    size_t n_total;
    double *merged_grid = merge_and_sort_unique(
        table->moneyness_grid, table->n_moneyness,
        new_m_points, n_new,
        &n_total
    );

    if (!merged_grid) {
        MANGO_TRACE_RUNTIME_ERROR(MODULE_PRICE_TABLE, 0, 0);
        return -1;
    }

    // If no new points were actually added (all duplicates), nothing to do
    if (n_total == table->n_moneyness) {
        free(merged_grid);
        return 0;
    }

    // 3. Allocate new arrays with expanded moneyness dimension
    const size_t n_old = table->n_moneyness;
    const size_t n_tau = table->n_maturity;
    const size_t n_sigma = table->n_volatility;
    const size_t n_r = table->n_rate;
    const size_t n_q = (table->n_dividend == 0) ? 1 : table->n_dividend;

    const size_t new_size = n_total * n_tau * n_sigma * n_r * n_q;

    double *new_prices = malloc(new_size * sizeof(double));
    double *new_vegas = table->vegas ? malloc(new_size * sizeof(double)) : NULL;
    double *new_gammas = table->gammas ? malloc(new_size * sizeof(double)) : NULL;
    double *new_thetas = table->thetas ? malloc(new_size * sizeof(double)) : NULL;
    double *new_rhos = table->rhos ? malloc(new_size * sizeof(double)) : NULL;

    if (!new_prices ||
        (table->vegas && !new_vegas) ||
        (table->gammas && !new_gammas) ||
        (table->thetas && !new_thetas) ||
        (table->rhos && !new_rhos)) {
        free(merged_grid);
        free(new_prices);
        free(new_vegas);
        free(new_gammas);
        free(new_thetas);
        free(new_rhos);
        MANGO_TRACE_RUNTIME_ERROR(MODULE_PRICE_TABLE, 1, new_size);
        return -1;
    }

    // Initialize all new arrays to NaN
    for (size_t i = 0; i < new_size; i++) {
        new_prices[i] = NAN;
        if (new_vegas) new_vegas[i] = NAN;
        if (new_gammas) new_gammas[i] = NAN;
        if (new_thetas) new_thetas[i] = NAN;
        if (new_rhos) new_rhos[i] = NAN;
    }

    // 4. Copy existing values to new positions
    // For LAYOUT_M_INNER: [tau][sigma][r][m]
    // Index: ((i_tau * n_sigma + i_sigma) * n_r + i_r) * n_m + i_m

    for (size_t i_tau = 0; i_tau < n_tau; i_tau++) {
        for (size_t i_sigma = 0; i_sigma < n_sigma; i_sigma++) {
            for (size_t i_r = 0; i_r < n_r; i_r++) {
                for (size_t i_q = 0; i_q < n_q; i_q++) {
                    // For each old moneyness point, find its new index
                    for (size_t i_m_old = 0; i_m_old < n_old; i_m_old++) {
                        double m_value = table->moneyness_grid[i_m_old];
                        int i_m_new = find_moneyness_index(merged_grid, n_total, m_value);

                        if (i_m_new < 0) {
                            // Should never happen if merge worked correctly
                            continue;
                        }

                        // Old index (with n_old points)
                        size_t old_idx = ((i_tau * n_sigma + i_sigma) * n_r + i_r) * n_old + i_m_old;
                        if (table->n_dividend > 0) {
                            old_idx = old_idx * n_q + i_q;
                        }

                        // New index (with n_total points)
                        size_t new_idx = ((i_tau * n_sigma + i_sigma) * n_r + i_r) * n_total + i_m_new;
                        if (table->n_dividend > 0) {
                            new_idx = new_idx * n_q + i_q;
                        }

                        // Copy values
                        new_prices[new_idx] = table->prices[old_idx];
                        if (new_vegas && table->vegas) {
                            new_vegas[new_idx] = table->vegas[old_idx];
                        }
                        if (new_gammas && table->gammas) {
                            new_gammas[new_idx] = table->gammas[old_idx];
                        }
                        if (new_thetas && table->thetas) {
                            new_thetas[new_idx] = table->thetas[old_idx];
                        }
                        if (new_rhos && table->rhos) {
                            new_rhos[new_idx] = table->rhos[old_idx];
                        }
                    }
                }
            }
        }
    }

    // 5. Replace table arrays
    free(table->moneyness_grid);
    free(table->prices);
    if (table->vegas) free(table->vegas);
    if (table->gammas) free(table->gammas);
    if (table->thetas) free(table->thetas);
    if (table->rhos) free(table->rhos);

    table->moneyness_grid = merged_grid;
    table->n_moneyness = n_total;
    table->prices = new_prices;
    table->vegas = new_vegas;
    table->gammas = new_gammas;
    table->thetas = new_thetas;
    table->rhos = new_rhos;

    // Update strides for new dimensions
    // For LAYOUT_M_INNER: moneyness is innermost
    table->stride_q = 1;
    table->stride_r = (table->n_dividend > 0) ? table->n_dividend : 1;
    table->stride_sigma = table->stride_r * table->n_rate;
    table->stride_tau = table->stride_sigma * table->n_volatility;
    table->stride_m = 1;  // Innermost dimension

    return 0;
}

// ---------- Data Access ----------

double price_table_get(const OptionPriceTable *table,
                       size_t i_m, size_t i_tau, size_t i_sigma,
                       size_t i_r, size_t i_q) {
    if (!table) return NAN;
    if (i_m >= table->n_moneyness || i_tau >= table->n_maturity ||
        i_sigma >= table->n_volatility || i_r >= table->n_rate) {
        return NAN;
    }
    if (table->n_dividend > 0 && i_q >= table->n_dividend) {
        return NAN;
    }

    size_t idx = i_m * table->stride_m + i_tau * table->stride_tau
               + i_sigma * table->stride_sigma + i_r * table->stride_r
               + i_q * table->stride_q;

    return table->prices[idx];
}

int price_table_set(OptionPriceTable *table,
                    size_t i_m, size_t i_tau, size_t i_sigma,
                    size_t i_r, size_t i_q, double price) {
    if (!table) return -1;
    if (i_m >= table->n_moneyness || i_tau >= table->n_maturity ||
        i_sigma >= table->n_volatility || i_r >= table->n_rate) {
        return -1;
    }
    if (table->n_dividend > 0 && i_q >= table->n_dividend) {
        return -1;
    }

    size_t idx = i_m * table->stride_m + i_tau * table->stride_tau
               + i_sigma * table->stride_sigma + i_r * table->stride_r
               + i_q * table->stride_q;

    table->prices[idx] = price;
    return 0;
}

double price_table_get_vega(const OptionPriceTable *table,
                             size_t i_m, size_t i_tau, size_t i_sigma,
                             size_t i_r, size_t i_q) {
    if (!table || !table->vegas) return NAN;

    // Bounds checking
    if (i_m >= table->n_moneyness || i_tau >= table->n_maturity ||
        i_sigma >= table->n_volatility || i_r >= table->n_rate ||
        i_q >= (table->n_dividend > 0 ? table->n_dividend : 1)) {
        return NAN;
    }

    // Calculate flat index using pre-computed strides
    size_t idx = i_m * table->stride_m + i_tau * table->stride_tau
               + i_sigma * table->stride_sigma + i_r * table->stride_r
               + i_q * table->stride_q;

    return table->vegas[idx];
}

int price_table_set_vega(OptionPriceTable *table,
                         size_t i_m, size_t i_tau, size_t i_sigma,
                         size_t i_r, size_t i_q, double vega) {
    if (!table || !table->vegas) return -1;

    // Bounds checking
    if (i_m >= table->n_moneyness || i_tau >= table->n_maturity ||
        i_sigma >= table->n_volatility || i_r >= table->n_rate ||
        i_q >= (table->n_dividend > 0 ? table->n_dividend : 1)) {
        return -1;
    }

    // Calculate flat index using pre-computed strides
    size_t idx = i_m * table->stride_m + i_tau * table->stride_tau
               + i_sigma * table->stride_sigma + i_r * table->stride_r
               + i_q * table->stride_q;

    table->vegas[idx] = vega;
    return 0;
}

double price_table_get_gamma(const OptionPriceTable *table,
                             size_t i_m, size_t i_tau, size_t i_sigma,
                             size_t i_r, size_t i_q) {
    if (!table || !table->gammas) {
        return NAN;
    }

    // Bounds checking
    size_t n_q_effective = table->n_dividend > 0 ? table->n_dividend : 1;
    if (i_m >= table->n_moneyness || i_tau >= table->n_maturity ||
        i_sigma >= table->n_volatility || i_r >= table->n_rate ||
        i_q >= n_q_effective) {
        return NAN;
    }

    // Calculate flat index using pre-computed strides
    size_t idx = i_m * table->stride_m + i_tau * table->stride_tau
               + i_sigma * table->stride_sigma + i_r * table->stride_r
               + i_q * table->stride_q;

    return table->gammas[idx];
}

int price_table_set_gamma(OptionPriceTable *table,
                          size_t i_m, size_t i_tau, size_t i_sigma,
                          size_t i_r, size_t i_q, double gamma) {
    if (!table || !table->gammas) return -1;

    // Bounds checking
    size_t n_q_effective = table->n_dividend > 0 ? table->n_dividend : 1;
    if (i_m >= table->n_moneyness || i_tau >= table->n_maturity ||
        i_sigma >= table->n_volatility || i_r >= table->n_rate ||
        i_q >= n_q_effective) {
        return -1;
    }

    // Calculate flat index using pre-computed strides
    size_t idx = i_m * table->stride_m + i_tau * table->stride_tau
               + i_sigma * table->stride_sigma + i_r * table->stride_r
               + i_q * table->stride_q;

    table->gammas[idx] = gamma;
    return 0;
}

// ========== Theta Get/Set ==========

double price_table_get_theta(const OptionPriceTable *table,
                              size_t i_m, size_t i_tau, size_t i_sigma,
                              size_t i_r, size_t i_q) {
    if (!table || !table->thetas) return NAN;

    // Bounds checking
    size_t n_q_effective = table->n_dividend > 0 ? table->n_dividend : 1;
    if (i_m >= table->n_moneyness || i_tau >= table->n_maturity ||
        i_sigma >= table->n_volatility || i_r >= table->n_rate ||
        i_q >= n_q_effective) {
        return NAN;
    }

    // Calculate flat index using pre-computed strides
    size_t idx = i_m * table->stride_m + i_tau * table->stride_tau
               + i_sigma * table->stride_sigma + i_r * table->stride_r
               + i_q * table->stride_q;

    return table->thetas[idx];
}

int price_table_set_theta(OptionPriceTable *table,
                           size_t i_m, size_t i_tau, size_t i_sigma,
                           size_t i_r, size_t i_q, double theta) {
    if (!table || !table->thetas) return -1;

    // Bounds checking
    size_t n_q_effective = table->n_dividend > 0 ? table->n_dividend : 1;
    if (i_m >= table->n_moneyness || i_tau >= table->n_maturity ||
        i_sigma >= table->n_volatility || i_r >= table->n_rate ||
        i_q >= n_q_effective) {
        return -1;
    }

    // Calculate flat index using pre-computed strides
    size_t idx = i_m * table->stride_m + i_tau * table->stride_tau
               + i_sigma * table->stride_sigma + i_r * table->stride_r
               + i_q * table->stride_q;

    table->thetas[idx] = theta;
    return 0;
}

// ========== Rho Get/Set ==========

double price_table_get_rho(const OptionPriceTable *table,
                            size_t i_m, size_t i_tau, size_t i_sigma,
                            size_t i_r, size_t i_q) {
    if (!table || !table->rhos) return NAN;

    // Bounds checking
    size_t n_q_effective = table->n_dividend > 0 ? table->n_dividend : 1;
    if (i_m >= table->n_moneyness || i_tau >= table->n_maturity ||
        i_sigma >= table->n_volatility || i_r >= table->n_rate ||
        i_q >= n_q_effective) {
        return NAN;
    }

    // Calculate flat index using pre-computed strides
    size_t idx = i_m * table->stride_m + i_tau * table->stride_tau
               + i_sigma * table->stride_sigma + i_r * table->stride_r
               + i_q * table->stride_q;

    return table->rhos[idx];
}

int price_table_set_rho(OptionPriceTable *table,
                         size_t i_m, size_t i_tau, size_t i_sigma,
                         size_t i_r, size_t i_q, double rho) {
    if (!table || !table->rhos) return -1;

    // Bounds checking
    size_t n_q_effective = table->n_dividend > 0 ? table->n_dividend : 1;
    if (i_m >= table->n_moneyness || i_tau >= table->n_maturity ||
        i_sigma >= table->n_volatility || i_r >= table->n_rate ||
        i_q >= n_q_effective) {
        return -1;
    }

    // Calculate flat index using pre-computed strides
    size_t idx = i_m * table->stride_m + i_tau * table->stride_tau
               + i_sigma * table->stride_sigma + i_r * table->stride_r
               + i_q * table->stride_q;

    table->rhos[idx] = rho;
    return 0;
}

int price_table_build_interpolation(OptionPriceTable *table) {
    if (!table) return -1;

    // Trigger interpolation strategy precomputation (e.g., cubic spline coefficients)
    if (table->strategy && table->strategy->precompute && table->interp_context) {
        int status = table->strategy->precompute(table, table->interp_context);
        if (status != 0) {
            return -1;
        }
    }

    return 0;
}

// ---------- Interpolation ----------

double price_table_interpolate_4d(const OptionPriceTable *table,
                                   double moneyness, double maturity,
                                   double volatility, double rate) {
    if (!table || !table->strategy || !table->strategy->interpolate_4d) {
        return NAN;
    }

    if (table->n_dividend > 0) {
        // Table is 5D, can't use 4D interpolation
        return NAN;
    }

    return table->strategy->interpolate_4d(table, moneyness, maturity,
                                            volatility, rate,
                                            table->interp_context);
}

double price_table_interpolate_5d(const OptionPriceTable *table,
                                   double moneyness, double maturity,
                                   double volatility, double rate,
                                   double dividend) {
    if (!table || !table->strategy || !table->strategy->interpolate_5d) {
        return NAN;
    }

    if (table->n_dividend == 0) {
        // Table is 4D, can't use 5D interpolation
        return NAN;
    }

    return table->strategy->interpolate_5d(table, moneyness, maturity,
                                            volatility, rate, dividend,
                                            table->interp_context);
}

double price_table_interpolate_vega_4d(const OptionPriceTable *table,
                                       double moneyness, double maturity,
                                       double volatility, double rate) {
    if (!table || !table->strategy || !table->vegas) {
        return NAN;
    }

    if (table->n_dividend > 0) {
        // Table is 5D, can't use 4D interpolation
        return NAN;
    }

    // Check if strategy supports vega interpolation
    if (!table->strategy->interpolate_4d) {
        return NAN;
    }

    // Temporarily swap prices with vegas for interpolation
    double *original_prices = table->prices;
    ((OptionPriceTable*)table)->prices = table->vegas;

    // Use price interpolation strategy on vega data
    double result = table->strategy->interpolate_4d(
        table, moneyness, maturity, volatility, rate,
        table->interp_context);

    // Restore original prices pointer
    ((OptionPriceTable*)table)->prices = original_prices;

    return result;
}

double price_table_interpolate_vega_5d(const OptionPriceTable *table,
                                       double moneyness, double maturity,
                                       double volatility, double rate,
                                       double dividend) {
    if (!table || !table->strategy || !table->vegas) {
        return NAN;
    }

    if (table->n_dividend == 0) {
        // Table is 4D, can't use 5D interpolation
        return NAN;
    }

    // Check if strategy supports vega interpolation
    if (!table->strategy->interpolate_5d) {
        return NAN;
    }

    // Temporarily swap prices with vegas for interpolation
    double *original_prices = table->prices;
    ((OptionPriceTable*)table)->prices = table->vegas;

    // Use price interpolation strategy on vega data
    double result = table->strategy->interpolate_5d(
        table, moneyness, maturity, volatility, rate, dividend,
        table->interp_context);

    // Restore original prices pointer
    ((OptionPriceTable*)table)->prices = original_prices;

    return result;
}

double price_table_interpolate_gamma_4d(const OptionPriceTable *table,
                                        double moneyness, double maturity,
                                        double volatility, double rate) {
    if (!table || !table->strategy || !table->gammas) {
        return NAN;
    }

    if (table->n_dividend > 0) {
        // Table is 5D, can't use 4D interpolation
        return NAN;
    }

    // Check if strategy supports gamma interpolation
    if (!table->strategy->interpolate_4d) {
        return NAN;
    }

    // Temporarily swap prices with gammas for interpolation
    double *original_prices = table->prices;
    ((OptionPriceTable*)table)->prices = table->gammas;

    // Use price interpolation strategy on gamma data
    double result = table->strategy->interpolate_4d(
        table, moneyness, maturity, volatility, rate,
        table->interp_context);

    // Restore original prices pointer
    ((OptionPriceTable*)table)->prices = original_prices;

    return result;
}

double price_table_interpolate_gamma_5d(const OptionPriceTable *table,
                                        double moneyness, double maturity,
                                        double volatility, double rate,
                                        double dividend) {
    if (!table || !table->strategy || !table->gammas) {
        return NAN;
    }

    if (table->n_dividend == 0) {
        // Table is 4D, can't use 5D interpolation
        return NAN;
    }

    // Check if strategy supports gamma interpolation
    if (!table->strategy->interpolate_5d) {
        return NAN;
    }

    // Temporarily swap prices with gammas for interpolation
    double *original_prices = table->prices;
    ((OptionPriceTable*)table)->prices = table->gammas;

    // Use price interpolation strategy on gamma data
    double result = table->strategy->interpolate_5d(
        table, moneyness, maturity, volatility, rate, dividend,
        table->interp_context);

    // Restore original prices pointer
    ((OptionPriceTable*)table)->prices = original_prices;

    return result;
}

double price_table_interpolate_theta_4d(const OptionPriceTable *table,
                                         double moneyness, double maturity,
                                         double volatility, double rate) {
    if (!table || !table->strategy || !table->thetas) {
        return NAN;
    }

    if (table->n_dividend > 0) {
        // Table is 5D, can't use 4D interpolation
        return NAN;
    }

    // Check if strategy supports theta interpolation
    if (!table->strategy->interpolate_4d) {
        return NAN;
    }

    // Temporarily swap prices with thetas for interpolation
    double *original_prices = table->prices;
    ((OptionPriceTable*)table)->prices = table->thetas;

    // Use price interpolation strategy on theta data
    double result = table->strategy->interpolate_4d(
        table, moneyness, maturity, volatility, rate,
        table->interp_context);

    // Restore original prices pointer
    ((OptionPriceTable*)table)->prices = original_prices;

    return result;
}

double price_table_interpolate_theta_5d(const OptionPriceTable *table,
                                         double moneyness, double maturity,
                                         double volatility, double rate,
                                         double dividend) {
    if (!table || !table->strategy || !table->thetas) {
        return NAN;
    }

    if (table->n_dividend == 0) {
        // Table is 4D, can't use 5D interpolation
        return NAN;
    }

    // Check if strategy supports theta interpolation
    if (!table->strategy->interpolate_5d) {
        return NAN;
    }

    // Temporarily swap prices with thetas for interpolation
    double *original_prices = table->prices;
    ((OptionPriceTable*)table)->prices = table->thetas;

    // Use price interpolation strategy on theta data
    double result = table->strategy->interpolate_5d(
        table, moneyness, maturity, volatility, rate, dividend,
        table->interp_context);

    // Restore original prices pointer
    ((OptionPriceTable*)table)->prices = original_prices;

    return result;
}

double price_table_interpolate_rho_4d(const OptionPriceTable *table,
                                       double moneyness, double maturity,
                                       double volatility, double rate) {
    if (!table || !table->strategy || !table->rhos) {
        return NAN;
    }

    if (table->n_dividend > 0) {
        // Table is 5D, can't use 4D interpolation
        return NAN;
    }

    // Check if strategy supports rho interpolation
    if (!table->strategy->interpolate_4d) {
        return NAN;
    }

    // Temporarily swap prices with rhos for interpolation
    double *original_prices = table->prices;
    ((OptionPriceTable*)table)->prices = table->rhos;

    // Use price interpolation strategy on rho data
    double result = table->strategy->interpolate_4d(
        table, moneyness, maturity, volatility, rate,
        table->interp_context);

    // Restore original prices pointer
    ((OptionPriceTable*)table)->prices = original_prices;

    return result;
}

double price_table_interpolate_rho_5d(const OptionPriceTable *table,
                                       double moneyness, double maturity,
                                       double volatility, double rate,
                                       double dividend) {
    if (!table || !table->strategy || !table->rhos) {
        return NAN;
    }

    if (table->n_dividend == 0) {
        // Table is 4D, can't use 5D interpolation
        return NAN;
    }

    // Check if strategy supports rho interpolation
    if (!table->strategy->interpolate_5d) {
        return NAN;
    }

    // Temporarily swap prices with rhos for interpolation
    double *original_prices = table->prices;
    ((OptionPriceTable*)table)->prices = table->rhos;

    // Use price interpolation strategy on rho data
    double result = table->strategy->interpolate_5d(
        table, moneyness, maturity, volatility, rate, dividend,
        table->interp_context);

    // Restore original prices pointer
    ((OptionPriceTable*)table)->prices = original_prices;

    return result;
}

OptionGreeks price_table_greeks_4d(const OptionPriceTable *table,
                                    double moneyness, double maturity,
                                    double volatility, double rate) {
    OptionGreeks greeks = {0};

    if (!table) return greeks;

    // Finite difference steps
    const double h_m = 0.001;      // 0.1% for moneyness
    const double h_tau = 1.0/365;  // 1 day for maturity
    const double h_sigma = 0.001;  // 0.1% for volatility
    const double h_r = 0.0001;     // 1 basis point for rate

    // Base price
    double V = price_table_interpolate_4d(table, moneyness, maturity, volatility, rate);

    // Delta: ∂V/∂m (moneyness sensitivity)
    double V_up = price_table_interpolate_4d(table, moneyness + h_m, maturity, volatility, rate);
    double V_down = price_table_interpolate_4d(table, moneyness - h_m, maturity, volatility, rate);
    greeks.delta = (V_up - V_down) / (2 * h_m);

    // Gamma: ∂²V/∂m²
    greeks.gamma = (V_up - 2*V + V_down) / (h_m * h_m);

    // Vega: ∂V/∂σ
    double V_vega_up = price_table_interpolate_4d(table, moneyness, maturity, volatility + h_sigma, rate);
    double V_vega_down = price_table_interpolate_4d(table, moneyness, maturity, volatility - h_sigma, rate);
    greeks.vega = (V_vega_up - V_vega_down) / (2 * h_sigma);

    // Theta: -∂V/∂τ (note negative sign)
    double V_theta_up = price_table_interpolate_4d(table, moneyness, maturity + h_tau, volatility, rate);
    double V_theta_down = price_table_interpolate_4d(table, moneyness, maturity - h_tau, volatility, rate);
    greeks.theta = -(V_theta_up - V_theta_down) / (2 * h_tau);

    // Rho: ∂V/∂r
    double V_rho_up = price_table_interpolate_4d(table, moneyness, maturity, volatility, rate + h_r);
    double V_rho_down = price_table_interpolate_4d(table, moneyness, maturity, volatility, rate - h_r);
    greeks.rho = (V_rho_up - V_rho_down) / (2 * h_r);

    return greeks;
}

OptionGreeks price_table_greeks_5d(const OptionPriceTable *table,
                                    double moneyness, double maturity,
                                    double volatility, double rate,
                                    double dividend) {
    OptionGreeks greeks = {0};

    if (!table) return greeks;

    // Finite difference steps
    const double h_m = 0.001;
    const double h_tau = 1.0/365;
    const double h_sigma = 0.001;
    const double h_r = 0.0001;

    // Base price
    double V = price_table_interpolate_5d(table, moneyness, maturity, volatility, rate, dividend);

    // Delta
    double V_up = price_table_interpolate_5d(table, moneyness + h_m, maturity, volatility, rate, dividend);
    double V_down = price_table_interpolate_5d(table, moneyness - h_m, maturity, volatility, rate, dividend);
    greeks.delta = (V_up - V_down) / (2 * h_m);

    // Gamma
    greeks.gamma = (V_up - 2*V + V_down) / (h_m * h_m);

    // Vega
    double V_vega_up = price_table_interpolate_5d(table, moneyness, maturity, volatility + h_sigma, rate, dividend);
    double V_vega_down = price_table_interpolate_5d(table, moneyness, maturity, volatility - h_sigma, rate, dividend);
    greeks.vega = (V_vega_up - V_vega_down) / (2 * h_sigma);

    // Theta
    double V_theta_up = price_table_interpolate_5d(table, moneyness, maturity + h_tau, volatility, rate, dividend);
    double V_theta_down = price_table_interpolate_5d(table, moneyness, maturity - h_tau, volatility, rate, dividend);
    greeks.theta = -(V_theta_up - V_theta_down) / (2 * h_tau);

    // Rho
    double V_rho_up = price_table_interpolate_5d(table, moneyness, maturity, volatility, rate + h_r, dividend);
    double V_rho_down = price_table_interpolate_5d(table, moneyness, maturity, volatility, rate - h_r, dividend);
    greeks.rho = (V_rho_up - V_rho_down) / (2 * h_r);

    return greeks;
}

int price_table_extract_slice(
    const OptionPriceTable *table,
    SliceDimension dimension,
    const int *fixed_indices,
    double *out_slice,
    bool *is_contiguous)
{
    if (!table || !fixed_indices || !out_slice || !is_contiguous) {
        return -1;
    }

    size_t slice_stride, slice_length;

    // Determine stride and length for requested dimension
    switch (dimension) {
        case SLICE_DIM_MONEYNESS:
            slice_stride = table->stride_m;
            slice_length = table->n_moneyness;
            break;
        case SLICE_DIM_MATURITY:
            slice_stride = table->stride_tau;
            slice_length = table->n_maturity;
            break;
        case SLICE_DIM_VOLATILITY:
            slice_stride = table->stride_sigma;
            slice_length = table->n_volatility;
            break;
        case SLICE_DIM_RATE:
            slice_stride = table->stride_r;
            slice_length = table->n_rate;
            break;
        case SLICE_DIM_DIVIDEND:
            if (table->n_dividend == 0) return -1;
            slice_stride = table->stride_q;
            slice_length = table->n_dividend;
            break;
        default:
            return -1;
    }

    // Calculate base offset from fixed indices
    size_t base_idx = 0;
    if (fixed_indices[0] >= 0) base_idx += fixed_indices[0] * table->stride_m;
    if (fixed_indices[1] >= 0) base_idx += fixed_indices[1] * table->stride_tau;
    if (fixed_indices[2] >= 0) base_idx += fixed_indices[2] * table->stride_sigma;
    if (fixed_indices[3] >= 0) base_idx += fixed_indices[3] * table->stride_r;
    if (fixed_indices[4] >= 0) base_idx += fixed_indices[4] * table->stride_q;

    // Extract: zero-copy if contiguous, strided copy otherwise
    if (slice_stride == 1) {
        *is_contiguous = true;
        memcpy(out_slice, &table->prices[base_idx], slice_length * sizeof(double));
    } else {
        *is_contiguous = false;
        for (size_t i = 0; i < slice_length; i++) {
            out_slice[i] = table->prices[base_idx + i * slice_stride];
        }
    }

    return 0;
}

int price_table_set_strategy(OptionPriceTable *table,
                              const InterpolationStrategy *strategy) {
    if (!table || !strategy) return -1;

    // Destroy old context
    if (table->strategy && table->strategy->destroy_context) {
        table->strategy->destroy_context(table->interp_context);
    }

    // Set new strategy
    table->strategy = strategy;

    // Create new context
    size_t dimensions = (table->n_dividend > 0) ? 5 : 4;
    size_t grid_sizes[5] = {
        table->n_moneyness, table->n_maturity, table->n_volatility,
        table->n_rate, table->n_dividend
    };
    table->interp_context = NULL;
    if (strategy->create_context) {
        table->interp_context = strategy->create_context(dimensions, grid_sizes);
    }

    // Pre-compute if supported
    if (strategy->precompute) {
        strategy->precompute(table, table->interp_context);
    }

    return 0;
}

// ---------- Metadata ----------

void price_table_set_underlying(OptionPriceTable *table, const char *underlying) {
    if (!table || !underlying) return;
    strncpy(table->underlying, underlying, sizeof(table->underlying) - 1);
    table->underlying[sizeof(table->underlying) - 1] = '\0';
}

const char* price_table_get_underlying(const OptionPriceTable *table) {
    return table ? table->underlying : NULL;
}

// ---------- I/O ----------

int price_table_save(const OptionPriceTable *table, const char *filename) {
    if (!table || !filename) return -1;

    FILE *fp = fopen(filename, "wb");
    if (!fp) return -1;

    // Write header
    PriceTableHeader header = {
        .magic = PRICE_TABLE_MAGIC,
        .version = PRICE_TABLE_VERSION,
        .n_moneyness = table->n_moneyness,
        .n_maturity = table->n_maturity,
        .n_volatility = table->n_volatility,
        .n_rate = table->n_rate,
        .n_dividend = table->n_dividend,
        .type = table->type,
        .exercise = table->exercise,
        .generation_time = table->generation_time,
        .coord_system = table->coord_system,
        .memory_layout = table->memory_layout,
        .has_gammas = (table->gammas != NULL) ? 1 : 0,
        .has_thetas = (table->thetas != NULL) ? 1 : 0,
        .has_rhos = (table->rhos != NULL) ? 1 : 0
    };
    memcpy(header.underlying, table->underlying, sizeof(header.underlying));

    if (fwrite(&header, sizeof(header), 1, fp) != 1) {
        fclose(fp);
        return -1;
    }

    // Write grid arrays
    if (fwrite(table->moneyness_grid, sizeof(double), table->n_moneyness, fp) != table->n_moneyness ||
        fwrite(table->maturity_grid, sizeof(double), table->n_maturity, fp) != table->n_maturity ||
        fwrite(table->volatility_grid, sizeof(double), table->n_volatility, fp) != table->n_volatility ||
        fwrite(table->rate_grid, sizeof(double), table->n_rate, fp) != table->n_rate) {
        fclose(fp);
        return -1;
    }

    if (table->n_dividend > 0) {
        if (fwrite(table->dividend_grid, sizeof(double), table->n_dividend, fp) != table->n_dividend) {
            fclose(fp);
            return -1;
        }
    }

    // Write price data
    size_t n_points = table->n_moneyness * table->n_maturity * table->n_volatility
                    * table->n_rate * (table->n_dividend > 0 ? table->n_dividend : 1);
    if (fwrite(table->prices, sizeof(double), n_points, fp) != n_points) {
        fclose(fp);
        return -1;
    }

    // Write vega data (only if allocated)
    if (table->vegas) {
        if (fwrite(table->vegas, sizeof(double), n_points, fp) != n_points) {
            fclose(fp);
            return -1;
        }
    }

    // Write gamma data (only if allocated)
    if (table->gammas) {
        if (fwrite(table->gammas, sizeof(double), n_points, fp) != n_points) {
            fclose(fp);
            return -1;
        }
    }

    // Write theta data (only if allocated)
    if (table->thetas) {
        if (fwrite(table->thetas, sizeof(double), n_points, fp) != n_points) {
            fclose(fp);
            return -1;
        }
    }

    // Write rho data (only if allocated)
    if (table->rhos) {
        if (fwrite(table->rhos, sizeof(double), n_points, fp) != n_points) {
            fclose(fp);
            return -1;
        }
    }

    fclose(fp);
    return 0;
}

OptionPriceTable* price_table_load(const char *filename) {
    if (!filename) return NULL;

    FILE *fp = fopen(filename, "rb");
    if (!fp) return NULL;

    // Read header
    PriceTableHeader header;
    if (fread(&header, sizeof(header), 1, fp) != 1) {
        fclose(fp);
        return NULL;
    }

    // Validate magic and version
    if (header.magic != PRICE_TABLE_MAGIC) {
        fclose(fp);
        return NULL;
    }

    // Support version 1 (without coord_system/memory_layout), version 2, version 3 (adds gammas), and version 4 (adds theta/rho)
    if (header.version < 1 || header.version > 4) {
        fclose(fp);
        return NULL;
    }

    // For version 1, use default values; for version 2+, use header values
    CoordinateSystem coord_system = (header.version >= 2) ? header.coord_system : COORD_RAW;
    MemoryLayout memory_layout = (header.version >= 2) ? header.memory_layout : LAYOUT_M_OUTER;

    // Allocate grid arrays
    double *moneyness = malloc(header.n_moneyness * sizeof(double));
    double *maturity = malloc(header.n_maturity * sizeof(double));
    double *volatility = malloc(header.n_volatility * sizeof(double));
    double *rate = malloc(header.n_rate * sizeof(double));
    double *dividend = (header.n_dividend > 0) ? malloc(header.n_dividend * sizeof(double)) : NULL;

    if (!moneyness || !maturity || !volatility || !rate ||
        (header.n_dividend > 0 && !dividend)) {
        free(moneyness);
        free(maturity);
        free(volatility);
        free(rate);
        free(dividend);
        fclose(fp);
        return NULL;
    }

    // Read grids
    if (fread(moneyness, sizeof(double), header.n_moneyness, fp) != header.n_moneyness ||
        fread(maturity, sizeof(double), header.n_maturity, fp) != header.n_maturity ||
        fread(volatility, sizeof(double), header.n_volatility, fp) != header.n_volatility ||
        fread(rate, sizeof(double), header.n_rate, fp) != header.n_rate) {
        free(moneyness);
        free(maturity);
        free(volatility);
        free(rate);
        free(dividend);
        fclose(fp);
        return NULL;
    }

    if (header.n_dividend > 0) {
        if (fread(dividend, sizeof(double), header.n_dividend, fp) != header.n_dividend) {
            free(moneyness);
            free(maturity);
            free(volatility);
            free(rate);
            free(dividend);
            fclose(fp);
            return NULL;
        }
    }

    // Create table with coordinate system and memory layout
    OptionPriceTable *table = price_table_create_ex(
        moneyness, header.n_moneyness,
        maturity, header.n_maturity,
        volatility, header.n_volatility,
        rate, header.n_rate,
        dividend, header.n_dividend,
        header.type, header.exercise,
        coord_system, memory_layout);

    free(moneyness);
    free(maturity);
    free(volatility);
    free(rate);
    free(dividend);

    if (!table) {
        fclose(fp);
        return NULL;
    }

    // Read price data
    size_t n_points = header.n_moneyness * header.n_maturity * header.n_volatility
                    * header.n_rate * (header.n_dividend > 0 ? header.n_dividend : 1);
    if (fread(table->prices, sizeof(double), n_points, fp) != n_points) {
        price_table_destroy(table);
        fclose(fp);
        return NULL;
    }

    // Read vega data (if available in file)
    // For backward compatibility, check if there's more data
    long current_pos = ftell(fp);
    fseek(fp, 0, SEEK_END);
    long end_pos = ftell(fp);
    fseek(fp, current_pos, SEEK_SET);

    if (end_pos - current_pos >= (long)(n_points * sizeof(double))) {
        // Vega data exists in file - allocate and read
        table->vegas = malloc(n_points * sizeof(double));
        if (!table->vegas) {
            price_table_destroy(table);
            fclose(fp);
            return NULL;
        }
        if (fread(table->vegas, sizeof(double), n_points, fp) != n_points) {
            price_table_destroy(table);
            fclose(fp);
            return NULL;
        }
    } else {
        // Old format without vega data - vegas stays NULL
        // It will be allocated if/when precompute is called
        table->vegas = NULL;
    }

    // Load gamma data (version 3+)
    if (header.version >= 3 && header.has_gammas) {
        table->gammas = malloc(n_points * sizeof(double));
        if (!table->gammas) {
            price_table_destroy(table);
            fclose(fp);
            return NULL;
        }
        if (fread(table->gammas, sizeof(double), n_points, fp) != n_points) {
            price_table_destroy(table);
            fclose(fp);
            return NULL;
        }
    } else {
        // Older version or no gammas - initialize to NULL
        table->gammas = NULL;
    }

    // Load theta data (version 4+)
    if (header.version >= 4 && header.has_thetas) {
        table->thetas = malloc(n_points * sizeof(double));
        if (!table->thetas) {
            price_table_destroy(table);
            fclose(fp);
            return NULL;
        }
        if (fread(table->thetas, sizeof(double), n_points, fp) != n_points) {
            price_table_destroy(table);
            fclose(fp);
            return NULL;
        }
    } else {
        // Older version or no thetas - initialize to NULL
        // Will be computed if/when precompute is called
        table->thetas = NULL;
    }

    // Load rho data (version 4+)
    if (header.version >= 4 && header.has_rhos) {
        table->rhos = malloc(n_points * sizeof(double));
        if (!table->rhos) {
            price_table_destroy(table);
            fclose(fp);
            return NULL;
        }
        if (fread(table->rhos, sizeof(double), n_points, fp) != n_points) {
            price_table_destroy(table);
            fclose(fp);
            return NULL;
        }
    } else {
        // Older version or no rhos - initialize to NULL
        // Will be computed if/when precompute is called
        table->rhos = NULL;
    }

    // Set metadata
    memcpy(table->underlying, header.underlying, sizeof(table->underlying));
    table->generation_time = header.generation_time;

    fclose(fp);
    return table;
}
