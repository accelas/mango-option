#define _GNU_SOURCE  // For M_PI
#include "grid_generation.h"
#include <stdlib.h>
#include <math.h>
#include <string.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// Comparison function for qsort
static int compare_doubles(const void *a, const void *b) {
    double da = *(const double*)a;
    double db = *(const double*)b;
    if (da < db) return -1;
    if (da > db) return 1;
    return 0;
}

double* grid_uniform(double min, double max, size_t n) {
    if (n < 2 || max <= min) return NULL;

    double *grid = malloc(n * sizeof(double));
    if (!grid) return NULL;

    const double h = (max - min) / (double)(n - 1);

    for (size_t i = 0; i < n; i++) {
        grid[i] = min + i * h;
    }

    // Ensure exact endpoints
    grid[0] = min;
    grid[n - 1] = max;

    return grid;
}

double* grid_log(double min, double max, size_t n) {
    if (n < 2 || max <= min || min <= 0) return NULL;

    double *grid = malloc(n * sizeof(double));
    if (!grid) return NULL;

    const double log_ratio = log(max / min);

    for (size_t i = 0; i < n; i++) {
        double t = (double)i / (double)(n - 1);
        grid[i] = min * exp(t * log_ratio);
    }

    // Ensure exact endpoints
    grid[0] = min;
    grid[n - 1] = max;

    return grid;
}

double* grid_chebyshev(double min, double max, size_t n) {
    if (n < 2 || max <= min) return NULL;

    double *grid = malloc(n * sizeof(double));
    if (!grid) return NULL;

    const double center = (min + max) / 2.0;
    const double radius = (max - min) / 2.0;

    for (size_t i = 0; i < n; i++) {
        // Chebyshev nodes: x_i = cos((2i + 1)π / (2n))
        double theta = (2.0 * i + 1.0) * M_PI / (2.0 * n);
        grid[i] = center + radius * cos(theta);
    }

    // Sort in ascending order (Chebyshev nodes are naturally descending)
    qsort(grid, n, sizeof(double), compare_doubles);

    // Ensure exact endpoints (Chebyshev nodes don't include endpoints exactly)
    grid[0] = min;
    grid[n - 1] = max;

    return grid;
}

double* grid_tanh_center(double min, double max, size_t n,
                         double center, double strength) {
    if (n < 2 || max <= min || center < min || center > max) return NULL;
    if (strength <= 0.0 || strength > 10.0) return NULL;

    double *grid = malloc(n * sizeof(double));
    if (!grid) return NULL;

    const double alpha = strength;
    const double tanh_alpha_half = tanh(alpha / 2.0);

    for (size_t i = 0; i < n; i++) {
        double t = (double)i / (double)(n - 1);  // [0, 1]
        double s = tanh(alpha * (t - 0.5)) / tanh_alpha_half;  // [-1, 1]

        // Map to [min, max] centered at 'center'
        if (s >= 0) {
            grid[i] = center + s * (max - center);
        } else {
            grid[i] = center + s * (center - min);
        }
    }

    // Ensure exact endpoints
    grid[0] = min;
    grid[n - 1] = max;

    return grid;
}

double* grid_sinh_onesided(double min, double max, size_t n,
                           double strength) {
    if (n < 2 || max <= min) return NULL;
    if (strength <= 0.0 || strength > 10.0) return NULL;

    double *grid = malloc(n * sizeof(double));
    if (!grid) return NULL;

    const double alpha = strength;
    const double sinh_alpha = sinh(alpha);

    for (size_t i = 0; i < n; i++) {
        double t = (double)i / (double)(n - 1);  // [0, 1]
        double s = sinh(alpha * t) / sinh_alpha;  // [0, 1] with concentration near 0
        grid[i] = min + s * (max - min);
    }

    // Ensure exact endpoints
    grid[0] = min;
    grid[n - 1] = max;

    return grid;
}

double* grid_generate(const GridSpec *spec) {
    if (!spec) return NULL;

    switch (spec->type) {
        case GRID_UNIFORM:
            return grid_uniform(spec->min, spec->max, spec->n_points);

        case GRID_LOG:
            return grid_log(spec->min, spec->max, spec->n_points);

        case GRID_CHEBYSHEV:
            return grid_chebyshev(spec->min, spec->max, spec->n_points);

        case GRID_TANH_CENTER:
            return grid_tanh_center(spec->min, spec->max, spec->n_points,
                                   spec->tanh_params.center,
                                   spec->tanh_params.strength);

        case GRID_SINH_ONESIDED:
            return grid_sinh_onesided(spec->min, spec->max, spec->n_points,
                                     spec->sinh_params.strength);

        case GRID_CUSTOM:
            // Custom grids not yet implemented
            return NULL;

        default:
            return NULL;
    }
}

bool grid_validate(const double *grid, size_t n, double min, double max) {
    if (!grid || n < 2 || max <= min) return false;

    const double tol = 1e-10;

    // Check endpoints
    if (fabs(grid[0] - min) > tol) return false;
    if (fabs(grid[n - 1] - max) > tol) return false;

    // Check sorted and no duplicates
    for (size_t i = 0; i < n - 1; i++) {
        if (grid[i] >= grid[i + 1]) return false;  // Not strictly increasing
        if (grid[i] < min - tol || grid[i] > max + tol) return false;  // Out of bounds
    }

    // Check last point in bounds
    if (grid[n - 1] < min - tol || grid[n - 1] > max + tol) return false;

    return true;
}

GridMetrics grid_compute_metrics(const double *grid, size_t n) {
    GridMetrics metrics = {0};

    if (!grid || n < 2) {
        metrics.min_spacing = NAN;
        metrics.max_spacing = NAN;
        metrics.avg_spacing = NAN;
        metrics.spacing_ratio = NAN;
        return metrics;
    }

    // Compute spacings
    double sum_spacing = 0.0;
    metrics.min_spacing = INFINITY;
    metrics.max_spacing = -INFINITY;

    for (size_t i = 0; i < n - 1; i++) {
        double spacing = grid[i + 1] - grid[i];
        sum_spacing += spacing;

        if (spacing < metrics.min_spacing) {
            metrics.min_spacing = spacing;
        }
        if (spacing > metrics.max_spacing) {
            metrics.max_spacing = spacing;
        }
    }

    metrics.avg_spacing = sum_spacing / (double)(n - 1);

    // Compute ratio (uniformity metric)
    if (metrics.min_spacing > 0) {
        metrics.spacing_ratio = metrics.max_spacing / metrics.min_spacing;
    } else {
        metrics.spacing_ratio = INFINITY;
    }

    return metrics;
}
