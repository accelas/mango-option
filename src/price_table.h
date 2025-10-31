#ifndef IVCALC_PRICE_TABLE_H
#define IVCALC_PRICE_TABLE_H

#include <stddef.h>
#include <time.h>
#include <stdbool.h>
#include "interp_strategy.h"
#include "american_option.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Exercise type enumeration
 */
typedef enum {
    EUROPEAN,
    AMERICAN
} ExerciseType;

/**
 * @file price_table.h
 * @brief Multi-dimensional option price table with pluggable interpolation
 *
 * Pre-computes option prices on a multi-dimensional grid for fast lookup:
 * - Moneyness (m = S/K)
 * - Maturity (τ = T - t)
 * - Volatility (σ)
 * - Interest rate (r)
 * - Dividend yield (q) [optional, 5D mode]
 *
 * Features:
 * - Sub-microsecond queries (4D: ~500ns, 5D: ~2µs)
 * - 40,000x faster than FDM solver (21.7ms → 500ns)
 * - Runtime interpolation strategy selection
 * - Parallel pre-computation via OpenMP
 * - Binary save/load for persistence
 *
 * Typical Usage:
 *   // Create table structure
 *   OptionPriceTable *table = price_table_create(
 *       moneyness, n_m, maturity, n_tau, volatility, n_sigma,
 *       rate, n_r, NULL, 0, OPTION_PUT, AMERICAN);
 *
 *   // Pre-compute all option prices (uses FDM)
 *   price_table_precompute(table, pde_solver_template);
 *
 *   // Save for fast loading later
 *   price_table_save(table, "spx_put_american.bin");
 *
 *   // Fast query (~500ns)
 *   double price = price_table_interpolate(table, 1.05, 0.25, 0.20, 0.05);
 *
 *   // Cleanup
 *   price_table_destroy(table);
 */

/**
 * Coordinate system for grid interpretation
 *
 * User API always accepts raw coordinates (m, T, σ, r, q).
 * Grid storage uses transformed coordinates for numerical stability.
 */
typedef enum {
    COORD_RAW,           // m, T, σ, r, q (current behavior, default)
    COORD_LOG_SQRT,      // log(m), sqrt(T), σ, r, q (recommended)
    COORD_LOG_VARIANCE,  // log(m), σ²T, r, q (future: collapsed dimensions)
} CoordinateSystem;

/**
 * Memory layout for price array
 *
 * Determines dimension ordering in flattened array.
 * LAYOUT_M_INNER optimizes for moneyness slice extraction (cubic interpolation).
 */
typedef enum {
    LAYOUT_M_OUTER,      // [m][tau][sigma][r][q] (current behavior, default)
    LAYOUT_M_INNER,      // [r][sigma][tau][m] (cache-optimized)
    LAYOUT_BLOCKED,      // Future: cache-oblivious tiled layout
} MemoryLayout;

/**
 * Option Price Table data structure
 *
 * Memory layout: row-major with fastest-to-slowest dimensions:
 *   moneyness, maturity, volatility, rate, dividend
 *
 * Index calculation:
 *   idx = i_m * stride_m + i_tau * stride_tau + i_sigma * stride_sigma
 *       + i_r * stride_r + i_q * stride_q
 */
typedef struct OptionPriceTable {
    // Grid definition (4D or 5D)
    size_t n_moneyness;         // S/K dimension
    size_t n_maturity;          // τ dimension
    size_t n_volatility;        // σ dimension
    size_t n_rate;              // r dimension
    size_t n_dividend;          // q dimension (0 for 4D mode)

    double *moneyness_grid;     // Sorted moneyness values
    double *maturity_grid;      // Sorted maturity values
    double *volatility_grid;    // Sorted volatility values
    double *rate_grid;          // Sorted rate values
    double *dividend_grid;      // Sorted dividend values (NULL for 4D)

    // Option prices (flattened multi-dimensional array)
    double *prices;             // n_m × n_tau × n_sigma × n_r × n_q values

    // Metadata
    OptionType type;            // CALL or PUT
    ExerciseType exercise;      // EUROPEAN or AMERICAN
    char underlying[32];        // Underlying symbol
    time_t generation_time;     // When table was computed

    // Transformation configuration (NEW)
    CoordinateSystem coord_system;  // How to interpret grid values
    MemoryLayout memory_layout;     // How prices are stored physically

    // Pre-computed strides for fast indexing
    size_t stride_m;            // = n_tau * n_sigma * n_r * n_q
    size_t stride_tau;          // = n_sigma * n_r * n_q
    size_t stride_sigma;        // = n_r * n_q
    size_t stride_r;            // = n_q
    size_t stride_q;            // = 1

    // Interpolation strategy (dependency injection)
    const InterpolationStrategy *strategy;  // Strategy vtable (not owned)
    InterpContext interp_context;           // Algorithm-specific context (owned)
} OptionPriceTable;

/**
 * Greeks structure for option sensitivities
 */
typedef struct OptionGreeks {
    double delta;       // ∂V/∂S
    double gamma;       // ∂²V/∂S²
    double vega;        // ∂V/∂σ
    double theta;       // -∂V/∂τ (note: negative time derivative)
    double rho;         // ∂V/∂r
} OptionGreeks;

// ---------- Creation and Destruction ----------

/**
 * Create option price table with specified interpolation strategy
 *
 * @param moneyness: array of moneyness values (must be sorted)
 * @param n_m: number of moneyness points
 * @param maturity: array of maturity values (must be sorted)
 * @param n_tau: number of maturity points
 * @param volatility: array of volatility values (must be sorted)
 * @param n_sigma: number of volatility points
 * @param rate: array of rate values (must be sorted)
 * @param n_r: number of rate points
 * @param dividend: array of dividend yield values (NULL for 4D mode)
 * @param n_q: number of dividend points (0 for 4D mode)
 * @param type: CALL or PUT
 * @param exercise: EUROPEAN or AMERICAN
 * @param strategy: interpolation strategy (e.g., &INTERP_MULTILINEAR)
 *                  If NULL, defaults to multilinear
 * @return newly created table, or NULL on error
 *
 * Note: Takes ownership of grid arrays (caller should not free them)
 * Note: Prices are initialized to NaN; call price_table_precompute() to fill
 */
OptionPriceTable* price_table_create_with_strategy(
    const double *moneyness, size_t n_m,
    const double *maturity, size_t n_tau,
    const double *volatility, size_t n_sigma,
    const double *rate, size_t n_r,
    const double *dividend, size_t n_q,
    OptionType type, ExerciseType exercise,
    const InterpolationStrategy *strategy);

/**
 * Create option price table with default (multilinear) interpolation
 */
OptionPriceTable* price_table_create(
    const double *moneyness, size_t n_m,
    const double *maturity, size_t n_tau,
    const double *volatility, size_t n_sigma,
    const double *rate, size_t n_r,
    const double *dividend, size_t n_q,
    OptionType type, ExerciseType exercise);

/**
 * Destroy option price table and free all resources
 */
void price_table_destroy(OptionPriceTable *table);

// ---------- Pre-computation ----------

/**
 * Pre-compute option prices for all grid points
 *
 * Populates the price table by computing option prices at each grid point
 * using the FDM solver via american_option_price_batch(). Uses batch
 * processing with configurable batch size (environment variable
 * IVCALC_PRECOMPUTE_BATCH_SIZE, default 100).
 *
 * Performance:
 * - 300K grid points: ~15-20 minutes on 16-core machine
 * - Throughput: ~300 options/second with parallelization
 * - Memory: ~10 KB per batch (default batch_size=100)
 *
 * Progress tracking via USDT probes (MODULE_PRICE_TABLE):
 * - ALGO_START: Start of pre-computation
 * - ALGO_PROGRESS: Every 10 batches
 * - ALGO_COMPLETE: Completion
 * - RUNTIME_ERROR: Batch computation failures
 *
 * @param table: Option price table to populate (must have allocated prices array)
 * @param grid: Spatial/temporal discretization for FDM solver
 * @return 0 on success, -1 on error (NULL inputs, allocation failure, batch failure)
 *
 * Environment variables:
 * - IVCALC_PRECOMPUTE_BATCH_SIZE: Batch size (1-100000, default 100)
 *
 * Example:
 * @code
 *   OptionPriceTable *table = price_table_create(...);
 *   AmericanOptionGrid grid = { .n_space = 101, .n_time = 1000, .S_max = 200.0 };
 *   int status = price_table_precompute(table, &grid);
 *   if (status == 0) {
 *       price_table_save(table, "table.bin");
 *   }
 * @endcode
 */
int price_table_precompute(OptionPriceTable *table,
                            const AmericanOptionGrid *grid);

// ---------- Data Access ----------

/**
 * Get price at specific grid point
 *
 * @return price at grid point, or NaN if indices out of bounds
 */
double price_table_get(const OptionPriceTable *table,
                       size_t i_m, size_t i_tau, size_t i_sigma,
                       size_t i_r, size_t i_q);

/**
 * Set price at specific grid point
 *
 * @return 0 on success, -1 on error
 */
int price_table_set(OptionPriceTable *table,
                    size_t i_m, size_t i_tau, size_t i_sigma,
                    size_t i_r, size_t i_q, double price);

// ---------- Interpolation ----------

/**
 * Interpolate option price at arbitrary point (4D mode)
 *
 * @param table: option price table
 * @param moneyness: query moneyness (m = S/K)
 * @param maturity: query maturity (τ = T - t, in years)
 * @param volatility: query volatility (σ, as decimal)
 * @param rate: query interest rate (r, as decimal)
 * @return interpolated option price
 *
 * Note: If query is outside grid bounds, clamps to boundary
 * Performance: ~200-500ns (multilinear), ~1-2µs (cubic)
 */
double price_table_interpolate_4d(const OptionPriceTable *table,
                                   double moneyness, double maturity,
                                   double volatility, double rate);

/**
 * Interpolate option price at arbitrary point (5D mode)
 *
 * @param dividend: query dividend yield (q, as decimal)
 */
double price_table_interpolate_5d(const OptionPriceTable *table,
                                   double moneyness, double maturity,
                                   double volatility, double rate,
                                   double dividend);

/**
 * Compute Greeks via finite differences on interpolated prices
 *
 * Uses centered finite differences:
 *   delta = (V(S+h) - V(S-h)) / (2h)
 *   gamma = (V(S+h) - 2V(S) + V(S-h)) / h²
 *
 * @param table: option price table
 * @param moneyness: query moneyness
 * @param maturity: query maturity
 * @param volatility: query volatility
 * @param rate: query rate
 * @return computed Greeks
 *
 * Note: Uses small finite difference step (h = 0.01 for delta/gamma)
 * Performance: ~5-10µs (requires multiple interpolations)
 */
OptionGreeks price_table_greeks_4d(const OptionPriceTable *table,
                                    double moneyness, double maturity,
                                    double volatility, double rate);

OptionGreeks price_table_greeks_5d(const OptionPriceTable *table,
                                    double moneyness, double maturity,
                                    double volatility, double rate,
                                    double dividend);

/**
 * Change interpolation strategy at runtime
 *
 * @return 0 on success, -1 on error
 */
int price_table_set_strategy(OptionPriceTable *table,
                              const InterpolationStrategy *strategy);

// ---------- Metadata ----------

/**
 * Set underlying symbol
 */
void price_table_set_underlying(OptionPriceTable *table, const char *underlying);

/**
 * Get underlying symbol
 */
const char* price_table_get_underlying(const OptionPriceTable *table);

// ---------- I/O ----------

/**
 * Save price table to binary file
 *
 * File format:
 *   - Header (256 bytes): magic, version, dimensions, metadata
 *   - Grid arrays: moneyness[], maturity[], volatility[], rate[], dividend[]
 *   - Price data: prices[]
 *   - Footer: checksum
 *
 * @param table: price table to save
 * @param filename: output file path
 * @return 0 on success, -1 on error
 *
 * File size: 4D with 50×30×20×10 = 2.4MB, 5D adds more
 */
int price_table_save(const OptionPriceTable *table, const char *filename);

/**
 * Load price table from binary file
 *
 * @param filename: input file path
 * @return loaded table, or NULL on error
 *
 * Note: Always loads with default (multilinear) strategy
 *       Call price_table_set_strategy() to change after loading
 */
OptionPriceTable* price_table_load(const char *filename);

#ifdef __cplusplus
}
#endif

#endif // IVCALC_PRICE_TABLE_H
