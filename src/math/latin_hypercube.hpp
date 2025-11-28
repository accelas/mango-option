#pragma once

#include <array>
#include <vector>
#include <algorithm>
#include <random>
#include <numeric>

namespace mango {

/// Generate Latin Hypercube samples in 4D unit hypercube [0,1]^4
///
/// Latin Hypercube ensures each dimension has exactly one sample per
/// stratum (bin), providing better space coverage than random sampling.
///
/// @param n Number of samples
/// @param seed Random seed for reproducibility
/// @return Vector of n 4D points in [0,1]^4
inline std::vector<std::array<double, 4>> latin_hypercube_4d(size_t n, uint64_t seed) {
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<double> uniform(0.0, 1.0);

    std::vector<std::array<double, 4>> samples(n);

    // For each dimension, create stratified samples and shuffle
    for (size_t d = 0; d < 4; ++d) {
        // Create indices 0..n-1
        std::vector<size_t> indices(n);
        std::iota(indices.begin(), indices.end(), 0);

        // Shuffle indices for this dimension
        std::shuffle(indices.begin(), indices.end(), rng);

        // Assign stratified samples
        for (size_t i = 0; i < n; ++i) {
            // Sample uniformly within stratum indices[i]
            double stratum_start = static_cast<double>(indices[i]) / static_cast<double>(n);
            double stratum_width = 1.0 / static_cast<double>(n);
            samples[i][d] = stratum_start + uniform(rng) * stratum_width;
        }
    }

    return samples;
}

/// Scale Latin Hypercube samples from [0,1]^4 to custom bounds
///
/// @param samples Samples in [0,1]^4
/// @param bounds Array of {min, max} pairs for each dimension
/// @return Scaled samples
inline std::vector<std::array<double, 4>> scale_lhs_samples(
    const std::vector<std::array<double, 4>>& samples,
    const std::array<std::pair<double, double>, 4>& bounds)
{
    std::vector<std::array<double, 4>> scaled(samples.size());
    for (size_t i = 0; i < samples.size(); ++i) {
        for (size_t d = 0; d < 4; ++d) {
            double range = bounds[d].second - bounds[d].first;
            scaled[i][d] = bounds[d].first + samples[i][d] * range;
        }
    }
    return scaled;
}

}  // namespace mango
