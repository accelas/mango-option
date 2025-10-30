#include "interp_cubic.h"
#include <stddef.h>

// Helper to find maximum of dimensions
static inline size_t max_size(size_t a, size_t b) {
    return a > b ? a : b;
}

size_t cubic_interp_workspace_size_2d(size_t n_moneyness, size_t n_maturity) {
    size_t max_grid = max_size(n_moneyness, n_maturity);

    // Spline workspace: 4n (coeffs) + 6n (temp)
    size_t spline_ws = 10 * max_grid;

    // Intermediate array for maturity interpolation results
    size_t intermediate = n_maturity;

    // Slice buffer for moneyness extraction
    size_t slice = n_moneyness;

    return spline_ws + intermediate + slice;
}

size_t cubic_interp_workspace_size_4d(size_t n_moneyness, size_t n_maturity,
                                       size_t n_volatility, size_t n_rate) {
    size_t max_grid = n_moneyness;
    max_grid = max_size(max_grid, n_maturity);
    max_grid = max_size(max_grid, n_volatility);
    max_grid = max_size(max_grid, n_rate);

    // Spline workspace
    size_t spline_ws = 10 * max_grid;

    // Intermediate arrays for each stage
    size_t intermediate1 = n_maturity * n_volatility * n_rate;  // After moneyness interp
    size_t intermediate2 = n_volatility * n_rate;               // After maturity interp
    size_t intermediate3 = n_rate;                              // After volatility interp
    size_t total_intermediate = intermediate1 + intermediate2 + intermediate3;

    // Slice buffer (max of all dimensions)
    size_t slice = max_grid;

    return spline_ws + total_intermediate + slice;
}

size_t cubic_interp_workspace_size_5d(size_t n_moneyness, size_t n_maturity,
                                       size_t n_volatility, size_t n_rate,
                                       size_t n_dividend) {
    size_t max_grid = n_moneyness;
    max_grid = max_size(max_grid, n_maturity);
    max_grid = max_size(max_grid, n_volatility);
    max_grid = max_size(max_grid, n_rate);
    max_grid = max_size(max_grid, n_dividend);

    // Spline workspace
    size_t spline_ws = 10 * max_grid;

    // Intermediate arrays for each stage
    size_t intermediate1 = n_maturity * n_volatility * n_rate * n_dividend;
    size_t intermediate2 = n_volatility * n_rate * n_dividend;
    size_t intermediate3 = n_rate * n_dividend;
    size_t intermediate4 = n_dividend;
    size_t total_intermediate = intermediate1 + intermediate2 + intermediate3 + intermediate4;

    // Slice buffer
    size_t slice = max_grid;

    return spline_ws + total_intermediate + slice;
}

int cubic_interp_workspace_init(CubicInterpWorkspace *workspace,
                                 double *buffer,
                                 size_t n_moneyness, size_t n_maturity,
                                 size_t n_volatility, size_t n_rate,
                                 size_t n_dividend) {
    if (workspace == NULL || buffer == NULL) {
        return -1;
    }

    // Determine dimensions
    size_t dimensions = 2;
    if (n_volatility > 0) dimensions = 4;
    if (n_dividend > 0) dimensions = 5;

    // Calculate size based on dimensions
    size_t required_size;
    if (dimensions == 2) {
        required_size = cubic_interp_workspace_size_2d(n_moneyness, n_maturity);
    } else if (dimensions == 4) {
        required_size = cubic_interp_workspace_size_4d(n_moneyness, n_maturity, n_volatility, n_rate);
    } else {
        required_size = cubic_interp_workspace_size_5d(n_moneyness, n_maturity, n_volatility, n_rate, n_dividend);
    }

    // Find max grid size
    size_t max_grid = n_moneyness;
    max_grid = max_size(max_grid, n_maturity);
    if (dimensions >= 4) {
        max_grid = max_size(max_grid, n_volatility);
        max_grid = max_size(max_grid, n_rate);
    }
    if (dimensions == 5) {
        max_grid = max_size(max_grid, n_dividend);
    }

    // Slice workspace into sections
    double *ptr = buffer;

    workspace->spline_coeff_workspace = ptr;
    ptr += 4 * max_grid;

    workspace->spline_temp_workspace = ptr;
    ptr += 6 * max_grid;

    workspace->intermediate_arrays = ptr;
    ptr += (required_size - 10 * max_grid - max_grid); // All intermediate space

    workspace->slice_buffers = ptr;

    workspace->max_grid_size = max_grid;
    workspace->total_size = required_size;

    return 0;
}
