#include "grid_presets.h"
#include <stdlib.h>
#include <string.h>

GridConfig grid_preset_get(
    GridPreset preset,
    double m_min, double m_max,
    double tau_min, double tau_max,
    double sigma_min, double sigma_max,
    double r_min, double r_max,
    double q_min, double q_max)
{
    GridConfig config;
    memset(&config, 0, sizeof(GridConfig));

    // Determine if 5D (with dividend) or 4D
    bool has_dividend = (q_max > q_min);

    switch (preset) {
        case GRID_PRESET_UNIFORM: {
            // Uniform baseline: 30×25×15×10 = 112,500 points (4D)
            config.moneyness = (GridSpec){GRID_UNIFORM, m_min, m_max, 30};
            config.maturity = (GridSpec){GRID_UNIFORM, tau_min, tau_max, 25};
            config.volatility = (GridSpec){GRID_UNIFORM, sigma_min, sigma_max, 15};
            config.rate = (GridSpec){GRID_UNIFORM, r_min, r_max, 10};
            if (has_dividend) {
                config.dividend = (GridSpec){GRID_UNIFORM, q_min, q_max, 5};
            }
            break;
        }

        case GRID_PRESET_LOG_STANDARD: {
            // Log-spaced moneyness (current default): 30×25×15×10 = 112,500 points
            config.moneyness = (GridSpec){GRID_LOG, m_min, m_max, 30};
            config.maturity = (GridSpec){GRID_UNIFORM, tau_min, tau_max, 25};
            config.volatility = (GridSpec){GRID_UNIFORM, sigma_min, sigma_max, 15};
            config.rate = (GridSpec){GRID_UNIFORM, r_min, r_max, 10};
            if (has_dividend) {
                config.dividend = (GridSpec){GRID_UNIFORM, q_min, q_max, 5};
            }
            break;
        }

        case GRID_PRESET_ADAPTIVE_FAST: {
            // Fast: 12×10×8×5 = 4,800 points (4D)
            // Target: ~10% error, rapid prototyping
            double m_center = (m_min + m_max) / 2.0;
            double sigma_center = (sigma_min + sigma_max) / 2.0;

            config.moneyness = (GridSpec){
                GRID_TANH_CENTER, m_min, m_max, 12,
                .tanh_params = {m_center, 3.0}
            };
            config.maturity = (GridSpec){
                GRID_SINH_ONESIDED, tau_min, tau_max, 10,
                .sinh_params = {2.0}
            };
            config.volatility = (GridSpec){
                GRID_TANH_CENTER, sigma_min, sigma_max, 8,
                .tanh_params = {sigma_center, 1.5}
            };
            config.rate = (GridSpec){GRID_UNIFORM, r_min, r_max, 5};
            if (has_dividend) {
                config.dividend = (GridSpec){GRID_UNIFORM, q_min, q_max, 3};
            }
            break;
        }

        case GRID_PRESET_ADAPTIVE_BALANCED: {
            // Balanced: 20×15×10×5 = 15,000 points (4D)
            // Target: ~3% error, production-ready
            double m_center = (m_min + m_max) / 2.0;
            double sigma_center = (sigma_min + sigma_max) / 2.0;

            config.moneyness = (GridSpec){
                GRID_TANH_CENTER, m_min, m_max, 20,
                .tanh_params = {m_center, 3.0}
            };
            config.maturity = (GridSpec){
                GRID_SINH_ONESIDED, tau_min, tau_max, 15,
                .sinh_params = {2.5}
            };
            config.volatility = (GridSpec){
                GRID_TANH_CENTER, sigma_min, sigma_max, 10,
                .tanh_params = {sigma_center, 2.0}
            };
            config.rate = (GridSpec){GRID_UNIFORM, r_min, r_max, 5};
            if (has_dividend) {
                config.dividend = (GridSpec){GRID_UNIFORM, q_min, q_max, 4};
            }
            break;
        }

        case GRID_PRESET_ADAPTIVE_ACCURATE: {
            // Accurate: 25×20×12×5 = 30,000 points (4D)
            // Target: ~1% error, high-accuracy applications
            double m_center = (m_min + m_max) / 2.0;
            double sigma_center = (sigma_min + sigma_max) / 2.0;

            config.moneyness = (GridSpec){
                GRID_TANH_CENTER, m_min, m_max, 25,
                .tanh_params = {m_center, 3.5}
            };
            config.maturity = (GridSpec){
                GRID_SINH_ONESIDED, tau_min, tau_max, 20,
                .sinh_params = {3.0}
            };
            config.volatility = (GridSpec){
                GRID_TANH_CENTER, sigma_min, sigma_max, 12,
                .tanh_params = {sigma_center, 2.5}
            };
            config.rate = (GridSpec){GRID_UNIFORM, r_min, r_max, 5};
            if (has_dividend) {
                config.dividend = (GridSpec){GRID_UNIFORM, q_min, q_max, 5};
            }
            break;
        }

        case GRID_PRESET_CUSTOM:
            // User must configure manually
            break;
    }

    return config;
}

GeneratedGrids grid_generate_all(const GridConfig *config) {
    GeneratedGrids grids;
    memset(&grids, 0, sizeof(GeneratedGrids));

    if (!config) return grids;

    // Generate moneyness grid
    grids.moneyness = grid_generate(&config->moneyness);
    grids.n_moneyness = config->moneyness.n_points;

    // Generate maturity grid
    grids.maturity = grid_generate(&config->maturity);
    grids.n_maturity = config->maturity.n_points;

    // Generate volatility grid
    grids.volatility = grid_generate(&config->volatility);
    grids.n_volatility = config->volatility.n_points;

    // Generate rate grid
    grids.rate = grid_generate(&config->rate);
    grids.n_rate = config->rate.n_points;

    // Generate dividend grid (if needed)
    if (config->dividend.n_points > 0) {
        grids.dividend = grid_generate(&config->dividend);
        grids.n_dividend = config->dividend.n_points;
    } else {
        grids.dividend = NULL;
        grids.n_dividend = 0;
    }

    // Compute total points
    grids.total_points = grids.n_moneyness * grids.n_maturity *
                         grids.n_volatility * grids.n_rate;
    if (grids.n_dividend > 0) {
        grids.total_points *= grids.n_dividend;
    }

    // Check if any allocation failed
    if (!grids.moneyness || !grids.maturity || !grids.volatility || !grids.rate) {
        grid_free_all(&grids);
        memset(&grids, 0, sizeof(GeneratedGrids));
    }

    // Check dividend grid if needed
    if (config->dividend.n_points > 0 && !grids.dividend) {
        grid_free_all(&grids);
        memset(&grids, 0, sizeof(GeneratedGrids));
    }

    return grids;
}

void grid_free_all(GeneratedGrids *grids) {
    if (!grids) return;

    free(grids->moneyness);
    free(grids->maturity);
    free(grids->volatility);
    free(grids->rate);
    free(grids->dividend);

    memset(grids, 0, sizeof(GeneratedGrids));
}

const char* grid_preset_name(GridPreset preset) {
    switch (preset) {
        case GRID_PRESET_UNIFORM:
            return "Uniform";
        case GRID_PRESET_LOG_STANDARD:
            return "Log-Spaced";
        case GRID_PRESET_ADAPTIVE_FAST:
            return "Adaptive Fast";
        case GRID_PRESET_ADAPTIVE_BALANCED:
            return "Adaptive Balanced";
        case GRID_PRESET_ADAPTIVE_ACCURATE:
            return "Adaptive Accurate";
        case GRID_PRESET_CUSTOM:
            return "Custom";
        default:
            return "Unknown";
    }
}

const char* grid_preset_description(GridPreset preset) {
    switch (preset) {
        case GRID_PRESET_UNIFORM:
            return "Uniform spacing (baseline, ~112K points)";
        case GRID_PRESET_LOG_STANDARD:
            return "Log-spaced moneyness (current default, ~112K points)";
        case GRID_PRESET_ADAPTIVE_FAST:
            return "Fast: ~5K points, ~10% error, rapid prototyping";
        case GRID_PRESET_ADAPTIVE_BALANCED:
            return "Balanced: ~15K points, ~3% error, production-ready";
        case GRID_PRESET_ADAPTIVE_ACCURATE:
            return "Accurate: ~30K points, ~1% error, high-accuracy";
        case GRID_PRESET_CUSTOM:
            return "User-defined configuration";
        default:
            return "Unknown preset";
    }
}
