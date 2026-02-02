// SPDX-License-Identifier: MIT
#include "src/option/grid_spec_types.hpp"

namespace mango {

GridAccuracyParams make_grid_accuracy(GridAccuracyProfile profile) {
    GridAccuracyParams params;
    switch (profile) {
        case GridAccuracyProfile::Low:
            params.tol = 5e-3;
            params.min_spatial_points = 150;
            params.max_spatial_points = 1500;
            params.max_time_steps = 6000;
            break;
        case GridAccuracyProfile::Medium:
            params.tol = 5e-5;
            params.min_spatial_points = 201;
            params.max_spatial_points = 2500;
            params.max_time_steps = 12000;
            break;
        case GridAccuracyProfile::High:
            params.tol = 1e-5;
            params.min_spatial_points = 301;
            params.max_spatial_points = 3500;
            params.max_time_steps = 16000;
            break;
        case GridAccuracyProfile::Ultra:
            params.tol = 5e-6;
            params.min_spatial_points = 401;
            params.max_spatial_points = 5000;
            params.max_time_steps = 20000;
            break;
    }
    return params;
}

}  // namespace mango
