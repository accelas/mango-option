#include "iv_surface.h"
#include "interp_cubic.h"
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <stdio.h>

// File format constants
#define IV_SURFACE_MAGIC 0x49565346  // "IVSF"
#define IV_SURFACE_VERSION 1

// File header structure
typedef struct {
    uint32_t magic;
    uint32_t version;
    size_t n_moneyness;
    size_t n_maturity;
    char underlying[32];
    time_t last_update;
    uint8_t padding[64];  // Reserved for future use
} IVSurfaceHeader;

// ---------- Creation and Destruction ----------

IVSurface* iv_surface_create_with_strategy(
    const double *moneyness, size_t n_m,
    const double *maturity, size_t n_tau,
    const InterpolationStrategy *strategy) {

    if (!moneyness || !maturity || n_m == 0 || n_tau == 0) {
        return NULL;
    }

    // Default to cubic if no strategy specified
    if (!strategy) {
        strategy = &INTERP_CUBIC;
    }

    IVSurface *surface = malloc(sizeof(IVSurface));
    if (!surface) return NULL;

    // Copy grid dimensions
    surface->n_moneyness = n_m;
    surface->n_maturity = n_tau;

    // Allocate and copy grids
    surface->moneyness_grid = malloc(n_m * sizeof(double));
    surface->maturity_grid = malloc(n_tau * sizeof(double));
    surface->iv_surface = malloc(n_m * n_tau * sizeof(double));

    if (!surface->moneyness_grid || !surface->maturity_grid || !surface->iv_surface) {
        free(surface->moneyness_grid);
        free(surface->maturity_grid);
        free(surface->iv_surface);
        free(surface);
        return NULL;
    }

    memcpy(surface->moneyness_grid, moneyness, n_m * sizeof(double));
    memcpy(surface->maturity_grid, maturity, n_tau * sizeof(double));

    // Initialize IV data to NaN
    for (size_t i = 0; i < n_m * n_tau; i++) {
        surface->iv_surface[i] = NAN;
    }

    // Initialize metadata
    memset(surface->underlying, 0, sizeof(surface->underlying));
    surface->last_update = time(NULL);

    // Set strategy
    surface->strategy = strategy;
    size_t grid_sizes[2] = {n_m, n_tau};
    surface->interp_context = NULL;
    if (strategy->create_context) {
        surface->interp_context = strategy->create_context(2, grid_sizes);
    }

    return surface;
}

IVSurface* iv_surface_create(
    const double *moneyness, size_t n_m,
    const double *maturity, size_t n_tau) {
    return iv_surface_create_with_strategy(moneyness, n_m, maturity, n_tau,
                                            &INTERP_CUBIC);
}

void iv_surface_destroy(IVSurface *surface) {
    if (!surface) return;

    // Destroy interpolation context
    if (surface->strategy && surface->strategy->destroy_context) {
        surface->strategy->destroy_context(surface->interp_context);
    }

    // Free arrays
    free(surface->moneyness_grid);
    free(surface->maturity_grid);
    free(surface->iv_surface);
    free(surface);
}

// ---------- Data Access ----------

int iv_surface_set(IVSurface *surface, const double *iv_data) {
    if (!surface || !iv_data) return -1;

    size_t n_points = surface->n_moneyness * surface->n_maturity;
    memcpy(surface->iv_surface, iv_data, n_points * sizeof(double));

    // Pre-compute interpolation coefficients if supported
    if (surface->strategy && surface->strategy->precompute && surface->interp_context) {
        surface->strategy->precompute(surface, surface->interp_context);
    }

    return 0;
}

double iv_surface_get(const IVSurface *surface, size_t i_m, size_t i_tau) {
    if (!surface || i_m >= surface->n_moneyness || i_tau >= surface->n_maturity) {
        return NAN;
    }

    return surface->iv_surface[i_tau * surface->n_moneyness + i_m];
}

int iv_surface_set_point(IVSurface *surface, size_t i_m, size_t i_tau, double iv) {
    if (!surface || i_m >= surface->n_moneyness || i_tau >= surface->n_maturity) {
        return -1;
    }

    surface->iv_surface[i_tau * surface->n_moneyness + i_m] = iv;
    return 0;
}

// ---------- Interpolation ----------

double iv_surface_interpolate(const IVSurface *surface,
                               double moneyness, double maturity) {
    if (!surface || !surface->strategy || !surface->strategy->interpolate_2d) {
        return NAN;
    }

    return surface->strategy->interpolate_2d(surface, moneyness, maturity,
                                              surface->interp_context);
}

int iv_surface_set_strategy(IVSurface *surface,
                             const InterpolationStrategy *strategy) {
    if (!surface || !strategy) return -1;

    // Destroy old context
    if (surface->strategy && surface->strategy->destroy_context) {
        surface->strategy->destroy_context(surface->interp_context);
    }

    // Set new strategy
    surface->strategy = strategy;

    // Create new context
    size_t grid_sizes[2] = {surface->n_moneyness, surface->n_maturity};
    surface->interp_context = NULL;
    if (strategy->create_context) {
        surface->interp_context = strategy->create_context(2, grid_sizes);
    }

    // Pre-compute if supported
    if (strategy->precompute) {
        strategy->precompute(surface, surface->interp_context);
    }

    return 0;
}

// ---------- Metadata ----------

void iv_surface_set_underlying(IVSurface *surface, const char *underlying) {
    if (!surface || !underlying) return;
    strncpy(surface->underlying, underlying, sizeof(surface->underlying) - 1);
    surface->underlying[sizeof(surface->underlying) - 1] = '\0';
}

const char* iv_surface_get_underlying(const IVSurface *surface) {
    return surface ? surface->underlying : NULL;
}

void iv_surface_touch(IVSurface *surface) {
    if (surface) {
        surface->last_update = time(NULL);
    }
}

// ---------- I/O ----------

int iv_surface_save(const IVSurface *surface, const char *filename) {
    if (!surface || !filename) return -1;

    FILE *fp = fopen(filename, "wb");
    if (!fp) return -1;

    // Write header
    IVSurfaceHeader header = {
        .magic = IV_SURFACE_MAGIC,
        .version = IV_SURFACE_VERSION,
        .n_moneyness = surface->n_moneyness,
        .n_maturity = surface->n_maturity,
        .last_update = surface->last_update
    };
    memcpy(header.underlying, surface->underlying, sizeof(header.underlying));

    if (fwrite(&header, sizeof(header), 1, fp) != 1) {
        fclose(fp);
        return -1;
    }

    // Write grid arrays
    if (fwrite(surface->moneyness_grid, sizeof(double), surface->n_moneyness, fp) != surface->n_moneyness) {
        fclose(fp);
        return -1;
    }

    if (fwrite(surface->maturity_grid, sizeof(double), surface->n_maturity, fp) != surface->n_maturity) {
        fclose(fp);
        return -1;
    }

    // Write IV data
    size_t n_points = surface->n_moneyness * surface->n_maturity;
    if (fwrite(surface->iv_surface, sizeof(double), n_points, fp) != n_points) {
        fclose(fp);
        return -1;
    }

    fclose(fp);
    return 0;
}

IVSurface* iv_surface_load(const char *filename) {
    if (!filename) return NULL;

    FILE *fp = fopen(filename, "rb");
    if (!fp) return NULL;

    // Read header
    IVSurfaceHeader header;
    if (fread(&header, sizeof(header), 1, fp) != 1) {
        fclose(fp);
        return NULL;
    }

    // Validate magic and version
    if (header.magic != IV_SURFACE_MAGIC || header.version != IV_SURFACE_VERSION) {
        fclose(fp);
        return NULL;
    }

    // Allocate grid arrays
    double *moneyness = malloc(header.n_moneyness * sizeof(double));
    double *maturity = malloc(header.n_maturity * sizeof(double));

    if (!moneyness || !maturity) {
        free(moneyness);
        free(maturity);
        fclose(fp);
        return NULL;
    }

    // Read grids
    if (fread(moneyness, sizeof(double), header.n_moneyness, fp) != header.n_moneyness ||
        fread(maturity, sizeof(double), header.n_maturity, fp) != header.n_maturity) {
        free(moneyness);
        free(maturity);
        fclose(fp);
        return NULL;
    }

    // Create surface
    IVSurface *surface = iv_surface_create(moneyness, header.n_moneyness,
                                            maturity, header.n_maturity);
    free(moneyness);
    free(maturity);

    if (!surface) {
        fclose(fp);
        return NULL;
    }

    // Read IV data
    size_t n_points = header.n_moneyness * header.n_maturity;
    if (fread(surface->iv_surface, sizeof(double), n_points, fp) != n_points) {
        iv_surface_destroy(surface);
        fclose(fp);
        return NULL;
    }

    // Set metadata
    memcpy(surface->underlying, header.underlying, sizeof(surface->underlying));
    surface->last_update = header.last_update;

    fclose(fp);
    return surface;
}
