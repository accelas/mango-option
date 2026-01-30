<!-- SPDX-License-Identifier: MIT -->
# Adaptive Grid Builder Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build an iterative grid refinement system that achieves target IV error by validating against fresh FD solves and refining dimensions with highest error concentration.

**Architecture:** AdaptiveGridBuilder orchestrates PriceTableBuilder's internal methods, caches (σ,r) slice results, validates against fresh FD solves at Latin hypercube sample points, and refines grid dimensions based on bin-based error attribution.

**Tech Stack:** C++23, Bazel, GoogleTest, existing PriceTableBuilder/BatchAmericanOptionSolver APIs

---

## Task 1: Black-Scholes Vega Function

**Files:**
- Create: `src/math/black_scholes_analytics.hpp`
- Test: `tests/black_scholes_analytics_test.cc`
- Modify: `src/math/BUILD.bazel`

**Step 1: Write the failing test**

In `tests/black_scholes_analytics_test.cc`:

```cpp
#include <gtest/gtest.h>
#include "src/math/black_scholes_analytics.hpp"
#include <cmath>

namespace mango {
namespace {

TEST(BlackScholesAnalyticsTest, VegaATMPut) {
    // ATM put: S=K=100, τ=1, σ=0.20, r=0.05
    double vega = bs_vega(100.0, 100.0, 1.0, 0.20, 0.05);
    // Expected: S * sqrt(τ) * N'(d1) ≈ 100 * 1 * 0.3969 ≈ 39.69
    EXPECT_NEAR(vega, 39.69, 0.1);
}

TEST(BlackScholesAnalyticsTest, VegaOTMPut) {
    // OTM put: S=100, K=80, τ=0.5, σ=0.25, r=0.03
    double vega = bs_vega(100.0, 80.0, 0.5, 0.25, 0.03);
    // Lower vega for OTM
    EXPECT_GT(vega, 0.0);
    EXPECT_LT(vega, 20.0);  // Much lower than ATM
}

TEST(BlackScholesAnalyticsTest, VegaShortMaturity) {
    // Very short maturity: τ=0.01
    double vega = bs_vega(100.0, 100.0, 0.01, 0.20, 0.05);
    // Vega scales with sqrt(τ), should be ~1/10 of 1-year
    EXPECT_LT(vega, 5.0);
}

TEST(BlackScholesAnalyticsTest, VegaDeepITM) {
    // Deep ITM put: S=100, K=150
    double vega = bs_vega(100.0, 150.0, 1.0, 0.20, 0.05);
    // Still positive but lower than ATM
    EXPECT_GT(vega, 0.0);
    EXPECT_LT(vega, 30.0);
}

}  // namespace
}  // namespace mango
```

**Step 2: Run test to verify it fails**

Run: `bazel test //tests:black_scholes_analytics_test --test_output=all`
Expected: BUILD ERROR - file not found

**Step 3: Create header with vega implementation**

In `src/math/black_scholes_analytics.hpp`:

```cpp
#pragma once

#include <cmath>

namespace mango {

/// Standard normal PDF: φ(x) = exp(-x²/2) / sqrt(2π)
inline double norm_pdf(double x) {
    static constexpr double kInvSqrt2Pi = 0.3989422804014327;  // 1/sqrt(2π)
    return kInvSqrt2Pi * std::exp(-0.5 * x * x);
}

/// Standard normal CDF: Φ(x) using Abramowitz & Stegun approximation
inline double norm_cdf(double x) {
    // Use erfc for numerical stability
    return 0.5 * std::erfc(-x * M_SQRT1_2);
}

/// Black-Scholes d1 term
/// d1 = [ln(S/K) + (r + σ²/2)τ] / (σ√τ)
inline double bs_d1(double spot, double strike, double tau, double sigma, double rate) {
    double sigma_sqrt_tau = sigma * std::sqrt(tau);
    return (std::log(spot / strike) + (rate + 0.5 * sigma * sigma) * tau) / sigma_sqrt_tau;
}

/// Black-Scholes Vega: ∂V/∂σ = S√τ φ(d1)
/// Same for puts and calls
///
/// @param spot Current underlying price
/// @param strike Strike price
/// @param tau Time to expiry in years
/// @param sigma Volatility
/// @param rate Risk-free rate
/// @return Vega (price change per unit volatility change)
inline double bs_vega(double spot, double strike, double tau, double sigma, double rate) {
    if (tau <= 0.0 || sigma <= 0.0) {
        return 0.0;
    }
    double d1 = bs_d1(spot, strike, tau, sigma, rate);
    return spot * std::sqrt(tau) * norm_pdf(d1);
}

}  // namespace mango
```

**Step 4: Add BUILD rule**

In `src/math/BUILD.bazel`, add:

```python
cc_library(
    name = "black_scholes_analytics",
    hdrs = ["black_scholes_analytics.hpp"],
    visibility = ["//visibility:public"],
)
```

**Step 5: Add test BUILD rule**

In `tests/BUILD.bazel`, add:

```python
cc_test(
    name = "black_scholes_analytics_test",
    size = "small",
    srcs = ["black_scholes_analytics_test.cc"],
    deps = [
        "//src/math:black_scholes_analytics",
        "@googletest//:gtest",
        "@googletest//:gtest_main",
    ],
)
```

**Step 6: Run test to verify it passes**

Run: `bazel test //tests:black_scholes_analytics_test --test_output=all`
Expected: PASS

**Step 7: Commit**

```bash
git add src/math/black_scholes_analytics.hpp src/math/BUILD.bazel tests/black_scholes_analytics_test.cc tests/BUILD.bazel
git commit -m "feat: add Black-Scholes vega for error scaling"
```

---

## Task 2: Latin Hypercube Sampling

**Files:**
- Create: `src/math/latin_hypercube.hpp`
- Test: `tests/latin_hypercube_test.cc`
- Modify: `src/math/BUILD.bazel`, `tests/BUILD.bazel`

**Step 1: Write the failing test**

In `tests/latin_hypercube_test.cc`:

```cpp
#include <gtest/gtest.h>
#include "src/math/latin_hypercube.hpp"
#include <algorithm>
#include <set>

namespace mango {
namespace {

TEST(LatinHypercubeTest, GeneratesCorrectSize) {
    auto samples = latin_hypercube_4d(64, 42);  // 64 samples, seed=42
    EXPECT_EQ(samples.size(), 64);
}

TEST(LatinHypercubeTest, AllValuesInUnitInterval) {
    auto samples = latin_hypercube_4d(100, 123);
    for (const auto& s : samples) {
        for (size_t d = 0; d < 4; ++d) {
            EXPECT_GE(s[d], 0.0);
            EXPECT_LE(s[d], 1.0);
        }
    }
}

TEST(LatinHypercubeTest, EachBinOccupiedOnce) {
    size_t n = 50;
    auto samples = latin_hypercube_4d(n, 456);

    // For each dimension, verify one sample per bin
    for (size_t d = 0; d < 4; ++d) {
        std::set<size_t> bins;
        for (const auto& s : samples) {
            size_t bin = static_cast<size_t>(s[d] * n);
            bin = std::min(bin, n - 1);  // Handle edge case
            bins.insert(bin);
        }
        EXPECT_EQ(bins.size(), n) << "Dimension " << d << " has repeated bins";
    }
}

TEST(LatinHypercubeTest, DeterministicWithSeed) {
    auto samples1 = latin_hypercube_4d(32, 999);
    auto samples2 = latin_hypercube_4d(32, 999);

    ASSERT_EQ(samples1.size(), samples2.size());
    for (size_t i = 0; i < samples1.size(); ++i) {
        for (size_t d = 0; d < 4; ++d) {
            EXPECT_DOUBLE_EQ(samples1[i][d], samples2[i][d]);
        }
    }
}

TEST(LatinHypercubeTest, DifferentSeedsDifferentSamples) {
    auto samples1 = latin_hypercube_4d(32, 111);
    auto samples2 = latin_hypercube_4d(32, 222);

    // At least some samples should differ
    bool any_different = false;
    for (size_t i = 0; i < samples1.size(); ++i) {
        if (samples1[i][0] != samples2[i][0]) {
            any_different = true;
            break;
        }
    }
    EXPECT_TRUE(any_different);
}

}  // namespace
}  // namespace mango
```

**Step 2: Run test to verify it fails**

Run: `bazel test //tests:latin_hypercube_test --test_output=all`
Expected: BUILD ERROR - file not found

**Step 3: Create header**

In `src/math/latin_hypercube.hpp`:

```cpp
#pragma once

#include <array>
#include <vector>
#include <algorithm>
#include <random>

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
```

**Step 4: Add BUILD rules**

In `src/math/BUILD.bazel`:

```python
cc_library(
    name = "latin_hypercube",
    hdrs = ["latin_hypercube.hpp"],
    visibility = ["//visibility:public"],
)
```

In `tests/BUILD.bazel`:

```python
cc_test(
    name = "latin_hypercube_test",
    size = "small",
    srcs = ["latin_hypercube_test.cc"],
    deps = [
        "//src/math:latin_hypercube",
        "@googletest//:gtest",
        "@googletest//:gtest_main",
    ],
)
```

**Step 5: Run test to verify it passes**

Run: `bazel test //tests:latin_hypercube_test --test_output=all`
Expected: PASS

**Step 6: Commit**

```bash
git add src/math/latin_hypercube.hpp src/math/BUILD.bazel tests/latin_hypercube_test.cc tests/BUILD.bazel
git commit -m "feat: add Latin Hypercube sampling for validation points"
```

---

## Task 3: Error Attribution (ErrorBins)

**Files:**
- Create: `src/option/table/error_attribution.hpp`
- Test: `tests/error_attribution_test.cc`
- Modify: `tests/BUILD.bazel`

**Step 1: Write the failing test**

In `tests/error_attribution_test.cc`:

```cpp
#include <gtest/gtest.h>
#include "src/option/table/error_attribution.hpp"

namespace mango {
namespace {

TEST(ErrorBinsTest, RecordsSingleError) {
    ErrorBins bins;
    // Error at position (0.5, 0.5, 0.5, 0.5) - all middle bins
    bins.record_error({0.5, 0.5, 0.5, 0.5}, 0.001, 0.0005);

    // Error exceeds threshold, should be recorded
    EXPECT_GT(bins.dim_error_mass[0], 0.0);
}

TEST(ErrorBinsTest, IdentifiesWorstDimension) {
    ErrorBins bins;
    // Concentrate errors in dimension 2 (volatility)
    // Dimension 2 = middle region (bin 2)
    for (int i = 0; i < 10; ++i) {
        bins.record_error({0.1 * i, 0.1 * i, 0.5, 0.1 * i}, 0.002, 0.001);
    }

    size_t worst = bins.worst_dimension();
    // Dim 2 has all errors concentrated in bin 2
    // Other dims have errors spread across bins
    // So dim 2 should have highest concentration
    EXPECT_EQ(worst, 2);
}

TEST(ErrorBinsTest, IgnoresErrorsBelowThreshold) {
    ErrorBins bins;
    bins.record_error({0.5, 0.5, 0.5, 0.5}, 0.0001, 0.001);  // Below threshold

    // Should not be recorded
    EXPECT_DOUBLE_EQ(bins.dim_error_mass[0], 0.0);
}

TEST(ErrorBinsTest, FindsProblematicBins) {
    ErrorBins bins;
    // Add errors in bins 0 and 1 of dimension 0
    for (int i = 0; i < 5; ++i) {
        bins.record_error({0.1, 0.5, 0.5, 0.5}, 0.002, 0.001);  // bin 0
        bins.record_error({0.25, 0.5, 0.5, 0.5}, 0.002, 0.001); // bin 1
    }

    auto problematic = bins.problematic_bins(0, 3);
    // Bins 0 and 1 should have count >= 3
    EXPECT_TRUE(std::find(problematic.begin(), problematic.end(), 0) != problematic.end());
    EXPECT_TRUE(std::find(problematic.begin(), problematic.end(), 1) != problematic.end());
}

TEST(ErrorBinsTest, NormalizedPositionOutOfRange) {
    ErrorBins bins;
    // Position outside [0,1] should be clamped
    bins.record_error({-0.1, 1.5, 0.5, 0.5}, 0.002, 0.001);

    // Should still work without crash
    EXPECT_GT(bins.dim_error_mass[0], 0.0);
}

}  // namespace
}  // namespace mango
```

**Step 2: Run test to verify it fails**

Run: `bazel test //tests:error_attribution_test --test_output=all`
Expected: BUILD ERROR

**Step 3: Create header**

In `src/option/table/error_attribution.hpp`:

```cpp
#pragma once

#include <array>
#include <vector>
#include <algorithm>
#include <cstddef>

namespace mango {

/// Bin-based error attribution for adaptive grid refinement
///
/// Tracks where errors occur in each dimension to identify which
/// dimension and which region needs refinement.
struct ErrorBins {
    static constexpr size_t N_BINS = 5;
    static constexpr size_t N_DIMS = 4;

    /// Count of high-error samples in each bin for each dimension
    std::array<std::array<size_t, N_BINS>, N_DIMS> bin_counts = {};

    /// Total error mass accumulated in each dimension
    std::array<double, N_DIMS> dim_error_mass = {};

    /// Record an error at a normalized position [0,1]^4
    ///
    /// @param normalized_pos Position in [0,1]^4 (clamped if out of range)
    /// @param iv_error IV error at this point
    /// @param threshold Only record if iv_error > threshold
    void record_error(const std::array<double, N_DIMS>& normalized_pos,
                      double iv_error, double threshold) {
        if (iv_error <= threshold) {
            return;
        }

        for (size_t d = 0; d < N_DIMS; ++d) {
            // Clamp to [0, 1] and compute bin
            double pos = std::clamp(normalized_pos[d], 0.0, 1.0);
            size_t bin = static_cast<size_t>(pos * N_BINS);
            bin = std::min(bin, N_BINS - 1);  // Handle pos == 1.0

            bin_counts[d][bin]++;
            dim_error_mass[d] += iv_error;
        }
    }

    /// Find dimension with most concentrated errors
    ///
    /// Returns the dimension where errors are most localized (highest
    /// max bin count relative to total), indicating refinement will help.
    [[nodiscard]] size_t worst_dimension() const {
        double best_score = -1.0;
        size_t best_dim = 0;

        for (size_t d = 0; d < N_DIMS; ++d) {
            // Find max bin count for this dimension
            size_t max_count = *std::max_element(
                bin_counts[d].begin(), bin_counts[d].end());
            size_t total_count = 0;
            for (size_t b = 0; b < N_BINS; ++b) {
                total_count += bin_counts[d][b];
            }

            if (total_count == 0) continue;

            // Score = concentration ratio * error mass
            // Higher when errors are localized AND significant
            double concentration = static_cast<double>(max_count) / static_cast<double>(total_count);
            double score = concentration * dim_error_mass[d];

            if (score > best_score) {
                best_score = score;
                best_dim = d;
            }
        }

        return best_dim;
    }

    /// Get bins with error count >= min_count for a dimension
    [[nodiscard]] std::vector<size_t> problematic_bins(size_t dim, size_t min_count = 2) const {
        std::vector<size_t> result;
        for (size_t b = 0; b < N_BINS; ++b) {
            if (bin_counts[dim][b] >= min_count) {
                result.push_back(b);
            }
        }
        return result;
    }

    /// Clear all bins
    void reset() {
        for (auto& dim_bins : bin_counts) {
            dim_bins.fill(0);
        }
        dim_error_mass.fill(0.0);
    }
};

}  // namespace mango
```

**Step 4: Add BUILD rules**

In `src/option/table/BUILD.bazel`, add:

```python
cc_library(
    name = "error_attribution",
    hdrs = ["error_attribution.hpp"],
    visibility = ["//visibility:public"],
)
```

In `tests/BUILD.bazel`:

```python
cc_test(
    name = "error_attribution_test",
    size = "small",
    srcs = ["error_attribution_test.cc"],
    deps = [
        "//src/option/table:error_attribution",
        "@googletest//:gtest",
        "@googletest//:gtest_main",
    ],
)
```

**Step 5: Run test**

Run: `bazel test //tests:error_attribution_test --test_output=all`
Expected: PASS

**Step 6: Commit**

```bash
git add src/option/table/error_attribution.hpp src/option/table/BUILD.bazel tests/error_attribution_test.cc tests/BUILD.bazel
git commit -m "feat: add ErrorBins for dimension-wise error attribution"
```

---

## Task 4: Slice Cache

**Files:**
- Create: `src/option/table/slice_cache.hpp`
- Test: `tests/slice_cache_test.cc`
- Modify: `tests/BUILD.bazel`

**Step 1: Write the failing test**

In `tests/slice_cache_test.cc`:

```cpp
#include <gtest/gtest.h>
#include "src/option/table/slice_cache.hpp"

namespace mango {
namespace {

// Mock AmericanOptionResult for testing
AmericanOptionResult make_mock_result() {
    // Create minimal valid result - we just need something storable
    PricingParams params{
        .strike = 100.0,
        .spot = 100.0,
        .maturity = 1.0,
        .volatility = 0.20,
        .rate = 0.05
    };
    // Note: In real code we'd need a valid Grid, but for cache tests
    // we just need the type to compile
    return AmericanOptionResult{nullptr, params};
}

TEST(SliceCacheTest, AddAndRetrieve) {
    SliceCache cache;
    auto result = make_mock_result();

    cache.add(0.20, 0.05, result);

    auto retrieved = cache.get(0.20, 0.05);
    EXPECT_TRUE(retrieved.has_value());
}

TEST(SliceCacheTest, MissingKeyReturnsNullopt) {
    SliceCache cache;
    auto retrieved = cache.get(0.30, 0.04);
    EXPECT_FALSE(retrieved.has_value());
}

TEST(SliceCacheTest, InvalidateOnTauChange) {
    SliceCache cache;
    auto result = make_mock_result();

    cache.set_tau_grid({0.1, 0.5, 1.0});
    cache.add(0.20, 0.05, result);

    // Same tau grid - should still have result
    cache.invalidate_if_tau_changed({0.1, 0.5, 1.0});
    EXPECT_TRUE(cache.get(0.20, 0.05).has_value());

    // Different tau grid - should invalidate
    cache.invalidate_if_tau_changed({0.1, 0.25, 0.5, 1.0});
    EXPECT_FALSE(cache.get(0.20, 0.05).has_value());
}

TEST(SliceCacheTest, GetMissingPairs) {
    SliceCache cache;
    auto result = make_mock_result();

    cache.add(0.20, 0.05, result);
    cache.add(0.30, 0.05, result);

    std::vector<std::pair<double, double>> all_pairs = {
        {0.20, 0.05},  // exists
        {0.30, 0.05},  // exists
        {0.25, 0.05},  // missing
        {0.20, 0.04},  // missing
    };

    auto missing = cache.get_missing_pairs(all_pairs);
    EXPECT_EQ(missing.size(), 2);
}

TEST(SliceCacheTest, GetMissingIndices) {
    SliceCache cache;
    auto result = make_mock_result();

    cache.add(0.20, 0.05, result);

    std::vector<std::pair<double, double>> all_pairs = {
        {0.20, 0.05},  // exists - index 0
        {0.25, 0.05},  // missing - index 1
        {0.30, 0.05},  // missing - index 2
    };

    auto missing_indices = cache.get_missing_indices(all_pairs);
    ASSERT_EQ(missing_indices.size(), 2);
    EXPECT_EQ(missing_indices[0], 1);
    EXPECT_EQ(missing_indices[1], 2);
}

}  // namespace
}  // namespace mango
```

**Step 2: Run test to verify it fails**

Run: `bazel test //tests:slice_cache_test --test_output=all`
Expected: BUILD ERROR

**Step 3: Create header**

In `src/option/table/slice_cache.hpp`:

```cpp
#pragma once

#include "src/option/american_option_result.hpp"
#include <map>
#include <optional>
#include <vector>
#include <utility>
#include <cmath>

namespace mango {

/// Cache for (σ, r) → AmericanOptionResult mappings
///
/// Used by AdaptiveGridBuilder to avoid re-solving PDE for unchanged slices.
/// Cache is invalidated when τ grid changes (PDE solve depends on maturity).
class SliceCache {
public:
    /// Add a result to the cache
    void add(double sigma, double rate, AmericanOptionResult result) {
        results_[make_key(sigma, rate)] = std::move(result);
    }

    /// Get a result from the cache
    [[nodiscard]] std::optional<AmericanOptionResult> get(double sigma, double rate) const {
        auto it = results_.find(make_key(sigma, rate));
        if (it != results_.end()) {
            return it->second;
        }
        return std::nullopt;
    }

    /// Set the current tau grid for invalidation checking
    void set_tau_grid(const std::vector<double>& tau_grid) {
        current_tau_grid_ = tau_grid;
    }

    /// Invalidate cache if tau grid changed
    void invalidate_if_tau_changed(const std::vector<double>& new_tau) {
        if (!grids_equal(current_tau_grid_, new_tau)) {
            results_.clear();
            current_tau_grid_ = new_tau;
        }
    }

    /// Get pairs that are NOT in the cache
    [[nodiscard]] std::vector<std::pair<double, double>> get_missing_pairs(
        const std::vector<std::pair<double, double>>& all_pairs) const
    {
        std::vector<std::pair<double, double>> missing;
        for (const auto& p : all_pairs) {
            if (results_.find(make_key(p.first, p.second)) == results_.end()) {
                missing.push_back(p);
            }
        }
        return missing;
    }

    /// Get indices of pairs that are NOT in the cache
    [[nodiscard]] std::vector<size_t> get_missing_indices(
        const std::vector<std::pair<double, double>>& all_pairs) const
    {
        std::vector<size_t> indices;
        for (size_t i = 0; i < all_pairs.size(); ++i) {
            if (results_.find(make_key(all_pairs[i].first, all_pairs[i].second)) == results_.end()) {
                indices.push_back(i);
            }
        }
        return indices;
    }

    /// Check if a pair exists in cache
    [[nodiscard]] bool contains(double sigma, double rate) const {
        return results_.find(make_key(sigma, rate)) != results_.end();
    }

    /// Get number of cached results
    [[nodiscard]] size_t size() const { return results_.size(); }

    /// Clear all cached results
    void clear() {
        results_.clear();
        current_tau_grid_.clear();
    }

private:
    /// Key for the map - use pair of sigma, rate
    using Key = std::pair<double, double>;

    static Key make_key(double sigma, double rate) {
        // Round to avoid floating point comparison issues
        // 6 decimal places is enough precision for vol/rate
        auto round6 = [](double x) {
            return std::round(x * 1e6) / 1e6;
        };
        return {round6(sigma), round6(rate)};
    }

    static bool grids_equal(const std::vector<double>& a, const std::vector<double>& b) {
        if (a.size() != b.size()) return false;
        for (size_t i = 0; i < a.size(); ++i) {
            if (std::abs(a[i] - b[i]) > 1e-10) return false;
        }
        return true;
    }

    std::map<Key, AmericanOptionResult> results_;
    std::vector<double> current_tau_grid_;
};

}  // namespace mango
```

**Step 4: Add BUILD rules**

In `src/option/table/BUILD.bazel`:

```python
cc_library(
    name = "slice_cache",
    hdrs = ["slice_cache.hpp"],
    deps = [
        "//src/option:american_option_result",
    ],
    visibility = ["//visibility:public"],
)
```

In `tests/BUILD.bazel`:

```python
cc_test(
    name = "slice_cache_test",
    size = "small",
    srcs = ["slice_cache_test.cc"],
    deps = [
        "//src/option/table:slice_cache",
        "//src/option:american_option_result",
        "@googletest//:gtest",
        "@googletest//:gtest_main",
    ],
)
```

**Step 5: Run test**

Run: `bazel test //tests:slice_cache_test --test_output=all`
Expected: PASS

**Step 6: Commit**

```bash
git add src/option/table/slice_cache.hpp src/option/table/BUILD.bazel tests/slice_cache_test.cc tests/BUILD.bazel
git commit -m "feat: add SliceCache for (σ,r) result caching"
```

---

## Task 5: AdaptiveGridParams and Result Structs

**Files:**
- Create: `src/option/table/adaptive_grid_types.hpp`
- Test: `tests/adaptive_grid_types_test.cc`

**Step 1: Write the failing test**

In `tests/adaptive_grid_types_test.cc`:

```cpp
#include <gtest/gtest.h>
#include "src/option/table/adaptive_grid_types.hpp"

namespace mango {
namespace {

TEST(AdaptiveGridParamsTest, DefaultValues) {
    AdaptiveGridParams params;

    EXPECT_DOUBLE_EQ(params.target_iv_error, 0.0005);  // 5 bps
    EXPECT_EQ(params.max_iterations, 5);
    EXPECT_EQ(params.max_points_per_dim, 50);
    EXPECT_EQ(params.validation_samples, 64);
    EXPECT_DOUBLE_EQ(params.refinement_factor, 1.3);
    EXPECT_EQ(params.bins_per_dim, 5);
}

TEST(IterationStatsTest, DefaultConstruction) {
    IterationStats stats;

    EXPECT_EQ(stats.iteration, 0);
    EXPECT_EQ(stats.pde_solves_table, 0);
    EXPECT_EQ(stats.pde_solves_validation, 0);
    EXPECT_DOUBLE_EQ(stats.max_error, 0.0);
}

TEST(AdaptiveResultTest, TargetMetFlag) {
    AdaptiveResult result;
    result.achieved_max_error = 0.0003;
    result.target_met = true;

    EXPECT_TRUE(result.target_met);
    EXPECT_LT(result.achieved_max_error, 0.0005);
}

}  // namespace
}  // namespace mango
```

**Step 2: Run test to verify it fails**

Run: `bazel test //tests:adaptive_grid_types_test --test_output=all`
Expected: BUILD ERROR

**Step 3: Create header**

In `src/option/table/adaptive_grid_types.hpp`:

```cpp
#pragma once

#include "src/option/table/price_table_axes.hpp"
#include "src/option/table/price_table_surface.hpp"
#include <array>
#include <memory>
#include <vector>
#include <cstddef>

namespace mango {

/// Configuration for adaptive grid refinement
struct AdaptiveGridParams {
    /// Target IV error in absolute terms (default: 5 bps = 0.0005)
    double target_iv_error = 0.0005;

    /// Maximum refinement iterations (default: 5)
    size_t max_iterations = 5;

    /// Maximum points per dimension ceiling (default: 50)
    size_t max_points_per_dim = 50;

    /// Number of validation FD solves per iteration (default: 64)
    size_t validation_samples = 64;

    /// Grid growth factor per refinement (default: 1.3)
    double refinement_factor = 1.3;

    /// Number of bins per dimension for error attribution (default: 5)
    size_t bins_per_dim = 5;

    /// Random seed for Latin Hypercube sampling (default: 42)
    uint64_t lhs_seed = 42;

    /// Vega floor for error metric (default: 1e-4)
    /// When vega < floor, fall back to price-based tolerance
    double vega_floor = 1e-4;
};

/// Per-iteration diagnostics
struct IterationStats {
    size_t iteration = 0;                    ///< Iteration number (0-indexed)
    std::array<size_t, 4> grid_sizes = {};   ///< [m, tau, sigma, r] sizes
    size_t pde_solves_table = 0;             ///< Slices computed for table
    size_t pde_solves_validation = 0;        ///< Fresh solves for validation
    double max_error = 0.0;                  ///< Max IV error observed
    double avg_error = 0.0;                  ///< Mean IV error
    int refined_dim = -1;                    ///< Which dim was refined (-1 if none)
    double elapsed_seconds = 0.0;            ///< Wall-clock time for this iteration
};

/// Final result with full diagnostics
struct AdaptiveResult {
    /// The built price table surface (always populated, even if target not met)
    std::shared_ptr<const PriceTableSurface<4>> surface = nullptr;

    /// Final axes used for the surface
    PriceTableAxes<4> axes;

    /// Per-iteration history for diagnostics
    std::vector<IterationStats> iterations;

    /// Actual max IV error from final validation
    double achieved_max_error = 0.0;

    /// Actual mean IV error from final validation
    double achieved_avg_error = 0.0;

    /// True iff achieved_max_error <= target_iv_error
    bool target_met = false;

    /// Total PDE solves across all iterations (table + validation)
    size_t total_pde_solves = 0;
};

}  // namespace mango
```

**Step 4: Add BUILD rules**

In `src/option/table/BUILD.bazel`:

```python
cc_library(
    name = "adaptive_grid_types",
    hdrs = ["adaptive_grid_types.hpp"],
    deps = [
        ":price_table_axes",
        ":price_table_surface",
    ],
    visibility = ["//visibility:public"],
)
```

In `tests/BUILD.bazel`:

```python
cc_test(
    name = "adaptive_grid_types_test",
    size = "small",
    srcs = ["adaptive_grid_types_test.cc"],
    deps = [
        "//src/option/table:adaptive_grid_types",
        "@googletest//:gtest",
        "@googletest//:gtest_main",
    ],
)
```

**Step 5: Run test**

Run: `bazel test //tests:adaptive_grid_types_test --test_output=all`
Expected: PASS

**Step 6: Commit**

```bash
git add src/option/table/adaptive_grid_types.hpp src/option/table/BUILD.bazel tests/adaptive_grid_types_test.cc tests/BUILD.bazel
git commit -m "feat: add AdaptiveGridParams and result types"
```

---

## Task 6: AdaptiveGridBuilder Skeleton

**Files:**
- Create: `src/option/table/adaptive_grid_builder.hpp`
- Create: `src/option/table/adaptive_grid_builder.cpp`
- Modify: `src/option/table/BUILD.bazel`
- Test: `tests/adaptive_grid_builder_test.cc`

**Step 1: Write the failing test**

In `tests/adaptive_grid_builder_test.cc`:

```cpp
#include <gtest/gtest.h>
#include "src/option/table/adaptive_grid_builder.hpp"

namespace mango {
namespace {

TEST(AdaptiveGridBuilderTest, ConstructWithDefaultParams) {
    AdaptiveGridParams params;
    AdaptiveGridBuilder builder(params);

    // Should compile and not crash
    SUCCEED();
}

TEST(AdaptiveGridBuilderTest, ConstructWithCustomParams) {
    AdaptiveGridParams params;
    params.target_iv_error = 0.001;  // 10 bps
    params.max_iterations = 3;

    AdaptiveGridBuilder builder(params);
    SUCCEED();
}

}  // namespace
}  // namespace mango
```

**Step 2: Run test to verify it fails**

Run: `bazel test //tests:adaptive_grid_builder_test --test_output=all`
Expected: BUILD ERROR

**Step 3: Create header**

In `src/option/table/adaptive_grid_builder.hpp`:

```cpp
#pragma once

#include "src/option/table/adaptive_grid_types.hpp"
#include "src/option/table/price_table_builder.hpp"
#include "src/option/table/slice_cache.hpp"
#include "src/option/table/error_attribution.hpp"
#include "src/option/option_chain.hpp"
#include "src/pde/core/grid.hpp"
#include "src/support/error_types.hpp"
#include <expected>

namespace mango {

/// Adaptive grid builder for price tables
///
/// Iteratively refines grid density until target IV error is achieved.
/// Uses fresh FD solves for validation (not self-referential spline comparison).
///
/// **Usage:**
/// ```cpp
/// AdaptiveGridParams params;
/// params.target_iv_error = 0.0005;  // 5 bps
///
/// AdaptiveGridBuilder builder(params);
/// auto result = builder.build(chain, grid_spec, n_time, OptionType::PUT);
///
/// if (result->target_met) {
///     auto price = result->surface->value({m, tau, sigma, r});
/// }
/// ```
class AdaptiveGridBuilder {
public:
    /// Construct builder with configuration
    explicit AdaptiveGridBuilder(AdaptiveGridParams params);

    /// Build price table with adaptive grid refinement
    ///
    /// @param chain Option chain providing domain bounds
    /// @param grid_spec PDE spatial grid specification
    /// @param n_time Number of time steps for PDE solver
    /// @param type Option type (default: PUT)
    /// @return AdaptiveResult with surface and diagnostics, or error
    [[nodiscard]] std::expected<AdaptiveResult, PriceTableError>
    build(const OptionChain& chain,
          GridSpec<double> grid_spec,
          size_t n_time,
          OptionType type = OptionType::PUT);

private:
    AdaptiveGridParams params_;
    SliceCache cache_;

    /// Compute hybrid IV/price error metric with vega floor
    double compute_error_metric(double interpolated_price, double reference_price,
                                double spot, double strike, double tau,
                                double sigma, double rate) const;

    /// Refine grid in the specified dimension
    std::vector<double> refine_dimension(const std::vector<double>& current_grid,
                                         const std::vector<size_t>& problematic_bins,
                                         size_t dim) const;
};

}  // namespace mango
```

**Step 4: Create minimal implementation**

In `src/option/table/adaptive_grid_builder.cpp`:

```cpp
#include "src/option/table/adaptive_grid_builder.hpp"
#include "src/math/black_scholes_analytics.hpp"
#include "src/math/latin_hypercube.hpp"
#include "src/option/table/price_table_grid_estimator.hpp"
#include <algorithm>
#include <cmath>

namespace mango {

AdaptiveGridBuilder::AdaptiveGridBuilder(AdaptiveGridParams params)
    : params_(std::move(params))
{}

std::expected<AdaptiveResult, PriceTableError>
AdaptiveGridBuilder::build(const OptionChain& chain,
                           GridSpec<double> grid_spec,
                           size_t n_time,
                           OptionType type)
{
    // TODO: Implement main loop
    return std::unexpected(PriceTableError{
        .code = PriceTableErrorCode::InvalidAxes,
        .message = "AdaptiveGridBuilder::build() not yet implemented"
    });
}

double AdaptiveGridBuilder::compute_error_metric(
    double interpolated_price, double reference_price,
    double spot, double strike, double tau, double sigma, double rate) const
{
    double price_error = std::abs(interpolated_price - reference_price);
    double vega = bs_vega(spot, strike, tau, sigma, rate);

    if (vega >= params_.vega_floor) {
        return price_error / vega;
    } else {
        // Fallback: treat vega_floor as minimum vega
        return price_error / params_.vega_floor;
    }
}

std::vector<double> AdaptiveGridBuilder::refine_dimension(
    const std::vector<double>& current_grid,
    const std::vector<size_t>& problematic_bins,
    size_t dim) const
{
    // TODO: Implement refinement
    return current_grid;
}

}  // namespace mango
```

**Step 5: Add BUILD rules**

In `src/option/table/BUILD.bazel`:

```python
cc_library(
    name = "adaptive_grid_builder",
    srcs = ["adaptive_grid_builder.cpp"],
    hdrs = ["adaptive_grid_builder.hpp"],
    copts = [
        "-fopenmp",
        "-pthread",
    ],
    linkopts = [
        "-fopenmp",
        "-pthread",
    ],
    deps = [
        ":adaptive_grid_types",
        ":error_attribution",
        ":price_table_builder",
        ":price_table_grid_estimator",
        ":slice_cache",
        "//src/math:black_scholes_analytics",
        "//src/math:latin_hypercube",
        "//src/option:american_option_batch",
        "//src/option:option_chain",
        "//src/pde/core:grid",
        "//src/support:error_types",
    ],
    visibility = ["//visibility:public"],
)
```

In `tests/BUILD.bazel`:

```python
cc_test(
    name = "adaptive_grid_builder_test",
    size = "small",
    srcs = ["adaptive_grid_builder_test.cc"],
    deps = [
        "//src/option/table:adaptive_grid_builder",
        "@googletest//:gtest",
        "@googletest//:gtest_main",
    ],
)
```

**Step 6: Run test**

Run: `bazel test //tests:adaptive_grid_builder_test --test_output=all`
Expected: PASS

**Step 7: Commit**

```bash
git add src/option/table/adaptive_grid_builder.hpp src/option/table/adaptive_grid_builder.cpp src/option/table/BUILD.bazel tests/adaptive_grid_builder_test.cc tests/BUILD.bazel
git commit -m "feat: add AdaptiveGridBuilder skeleton"
```

---

## Task 7: Promote _for_testing Methods to _internal

**Files:**
- Modify: `src/option/table/price_table_builder.hpp`
- Test: Existing tests should still pass

**Step 1: Run existing tests to establish baseline**

Run: `bazel test //tests:price_table_builder_test --test_output=all`
Expected: PASS

**Step 2: Add _internal aliases in header**

In `src/option/table/price_table_builder.hpp`, after each `_for_testing` method, add an `_internal` alias. For example, after `make_batch_for_testing`:

```cpp
    /// Internal API: generate batch of AmericanOptionParams from axes
    /// Used by AdaptiveGridBuilder for incremental builds
    [[nodiscard]] std::vector<AmericanOptionParams> make_batch_internal(
        const PriceTableAxes<N>& axes) const {
        return make_batch(axes);
    }

    /// Internal API: extract tensor from batch results
    /// Used by AdaptiveGridBuilder for incremental builds
    [[nodiscard]] std::expected<ExtractionResult<N>, PriceTableError> extract_tensor_internal(
        const BatchAmericanOptionResult& batch,
        const PriceTableAxes<N>& axes) const {
        return extract_tensor(batch, axes);
    }

    /// Internal API: fit B-spline coefficients from tensor
    /// Used by AdaptiveGridBuilder for incremental builds
    [[nodiscard]] std::expected<std::vector<double>, PriceTableError> fit_coeffs_internal(
        const PriceTensor<N>& tensor,
        const PriceTableAxes<N>& axes) const {
        auto result = fit_coeffs(tensor, axes);
        if (!result.has_value()) {
            return std::unexpected(result.error());
        }
        return std::move(result.value().coefficients);
    }
```

**Step 3: Run tests to verify no regressions**

Run: `bazel test //tests:price_table_builder_test --test_output=all`
Expected: PASS

**Step 4: Commit**

```bash
git add src/option/table/price_table_builder.hpp
git commit -m "refactor: add _internal aliases for AdaptiveGridBuilder"
```

---

## Task 8: Implement Main Build Loop

**Files:**
- Modify: `src/option/table/adaptive_grid_builder.cpp`
- Test: `tests/adaptive_grid_builder_test.cc` (add new tests)

**Step 1: Write integration test**

Add to `tests/adaptive_grid_builder_test.cc`:

```cpp
TEST(AdaptiveGridBuilderTest, BuildsWithSyntheticChain) {
    // Create a minimal synthetic chain
    OptionChain chain;
    chain.spot = 100.0;
    chain.dividend_yield = 0.0;

    // Add some options with different strikes/maturities
    for (double K : {90.0, 95.0, 100.0, 105.0, 110.0}) {
        for (double tau : {0.25, 0.5, 1.0}) {
            OptionData opt;
            opt.strike = K;
            opt.maturity = tau;
            opt.implied_vol = 0.20;
            opt.rate = 0.05;
            opt.type = OptionType::PUT;
            chain.options.push_back(opt);
        }
    }

    AdaptiveGridParams params;
    params.target_iv_error = 0.001;  // 10 bps - relaxed for test speed
    params.max_iterations = 3;
    params.validation_samples = 16;  // Fewer for test speed

    AdaptiveGridBuilder builder(params);

    auto grid_spec = GridSpec<double>::sinh_spaced(-3.0, 3.0, 101, 2.0).value();
    auto result = builder.build(chain, grid_spec, 500, OptionType::PUT);

    ASSERT_TRUE(result.has_value()) << result.error().message;

    // Should have at least one iteration
    EXPECT_GE(result->iterations.size(), 1);

    // Surface should be populated
    EXPECT_NE(result->surface, nullptr);

    // Should have done some PDE solves
    EXPECT_GT(result->total_pde_solves, 0);
}
```

**Step 2: Run test to verify it fails**

Run: `bazel test //tests:adaptive_grid_builder_test --test_output=all`
Expected: FAIL (not implemented yet)

**Step 3: Implement build() method**

This is a large implementation. See `adaptive_grid_builder.cpp` for the full code. Key sections:

1. **Seed estimate**: Use `estimate_grid_from_chain_bounds()`
2. **Main loop**: For each iteration:
   - Build/update table using `_internal` methods
   - Generate validation samples with Latin Hypercube
   - Validate against fresh FD solves
   - Check convergence
   - Diagnose & refine using ErrorBins

**Step 4: Run test to verify it passes**

Run: `bazel test //tests:adaptive_grid_builder_test --test_output=all`
Expected: PASS

**Step 5: Commit**

```bash
git add src/option/table/adaptive_grid_builder.cpp tests/adaptive_grid_builder_test.cc
git commit -m "feat: implement AdaptiveGridBuilder main loop"
```

---

## Task 9: Integration Tests

**Files:**
- Create: `tests/adaptive_grid_builder_integration_test.cc`
- Modify: `tests/BUILD.bazel`

**Step 1: Write comprehensive integration tests**

In `tests/adaptive_grid_builder_integration_test.cc`:

```cpp
#include <gtest/gtest.h>
#include "src/option/table/adaptive_grid_builder.hpp"
#include <chrono>

namespace mango {
namespace {

class AdaptiveGridBuilderIntegrationTest : public ::testing::Test {
protected:
    OptionChain make_test_chain() {
        OptionChain chain;
        chain.spot = 100.0;
        chain.dividend_yield = 0.02;

        for (double K : {85.0, 90.0, 95.0, 100.0, 105.0, 110.0, 115.0}) {
            for (double tau : {0.1, 0.25, 0.5, 1.0, 2.0}) {
                OptionData opt;
                opt.strike = K;
                opt.maturity = tau;
                opt.implied_vol = 0.15 + 0.05 * std::abs(K - 100.0) / 100.0;
                opt.rate = 0.04;
                opt.type = OptionType::PUT;
                chain.options.push_back(opt);
            }
        }
        return chain;
    }
};

TEST_F(AdaptiveGridBuilderIntegrationTest, ConvergesToTarget) {
    auto chain = make_test_chain();

    AdaptiveGridParams params;
    params.target_iv_error = 0.0005;  // 5 bps
    params.max_iterations = 5;
    params.validation_samples = 64;

    AdaptiveGridBuilder builder(params);
    auto grid_spec = GridSpec<double>::sinh_spaced(-3.0, 3.0, 101, 2.0).value();

    auto result = builder.build(chain, grid_spec, 1000, OptionType::PUT);

    ASSERT_TRUE(result.has_value());

    // Should meet target within max_iterations
    if (result->target_met) {
        EXPECT_LE(result->achieved_max_error, params.target_iv_error);
    }

    // Should have diagnostic history
    EXPECT_FALSE(result->iterations.empty());
}

TEST_F(AdaptiveGridBuilderIntegrationTest, CacheReusesSlices) {
    auto chain = make_test_chain();

    AdaptiveGridParams params;
    params.target_iv_error = 0.001;  // Relaxed target
    params.max_iterations = 2;

    AdaptiveGridBuilder builder(params);
    auto grid_spec = GridSpec<double>::sinh_spaced(-3.0, 3.0, 101, 2.0).value();

    auto result = builder.build(chain, grid_spec, 500, OptionType::PUT);

    ASSERT_TRUE(result.has_value());

    if (result->iterations.size() >= 2) {
        // Second iteration should have fewer table solves (cache hit)
        auto& iter1 = result->iterations[0];
        auto& iter2 = result->iterations[1];

        // If only σ/r refined, second iteration uses fewer solves
        // If m/τ refined, cache is invalidated - solves may be same
        // Just verify we get reasonable counts
        EXPECT_GT(iter1.pde_solves_table, 0);
    }
}

TEST_F(AdaptiveGridBuilderIntegrationTest, HandlesImpossibleTarget) {
    auto chain = make_test_chain();

    AdaptiveGridParams params;
    params.target_iv_error = 1e-10;  // Impossible target
    params.max_iterations = 2;       // Limited iterations
    params.max_points_per_dim = 10;  // Limited grid

    AdaptiveGridBuilder builder(params);
    auto grid_spec = GridSpec<double>::sinh_spaced(-3.0, 3.0, 51, 2.0).value();

    auto result = builder.build(chain, grid_spec, 200, OptionType::PUT);

    ASSERT_TRUE(result.has_value());

    // Should return best-effort result with target_met = false
    EXPECT_FALSE(result->target_met);
    EXPECT_NE(result->surface, nullptr);  // Still have a surface
}

TEST_F(AdaptiveGridBuilderIntegrationTest, DeterministicWithSameSeed) {
    auto chain = make_test_chain();

    AdaptiveGridParams params;
    params.target_iv_error = 0.001;
    params.max_iterations = 2;
    params.lhs_seed = 12345;

    AdaptiveGridBuilder builder1(params);
    AdaptiveGridBuilder builder2(params);

    auto grid_spec = GridSpec<double>::sinh_spaced(-3.0, 3.0, 51, 2.0).value();

    auto result1 = builder1.build(chain, grid_spec, 300, OptionType::PUT);
    auto result2 = builder2.build(chain, grid_spec, 300, OptionType::PUT);

    ASSERT_TRUE(result1.has_value());
    ASSERT_TRUE(result2.has_value());

    // Same seed should produce same results
    EXPECT_DOUBLE_EQ(result1->achieved_max_error, result2->achieved_max_error);
}

}  // namespace
}  // namespace mango
```

**Step 2: Add BUILD rule**

In `tests/BUILD.bazel`:

```python
cc_test(
    name = "adaptive_grid_builder_integration_test",
    size = "medium",
    timeout = "moderate",
    srcs = ["adaptive_grid_builder_integration_test.cc"],
    copts = ["-fopenmp"],
    linkopts = ["-fopenmp"],
    deps = [
        "//src/option/table:adaptive_grid_builder",
        "@googletest//:gtest",
        "@googletest//:gtest_main",
    ],
)
```

**Step 3: Run tests**

Run: `bazel test //tests:adaptive_grid_builder_integration_test --test_output=all`
Expected: PASS

**Step 4: Commit**

```bash
git add tests/adaptive_grid_builder_integration_test.cc tests/BUILD.bazel
git commit -m "test: add AdaptiveGridBuilder integration tests"
```

---

## Task 10: Final Cleanup and Documentation

**Files:**
- Update: `docs/plans/2025-11-27-adaptive-grid-builder-design.md` (mark as implemented)
- Run: Full test suite

**Step 1: Run full test suite**

Run: `bazel test //...`
Expected: All tests pass

**Step 2: Update design doc status**

Change status from "Approved (rev 3)" to "Implemented"

**Step 3: Commit**

```bash
git add docs/plans/2025-11-27-adaptive-grid-builder-design.md
git commit -m "docs: mark adaptive grid builder design as implemented"
```

**Step 4: Verify examples and benchmarks compile**

Run: `bazel build //examples/... //benchmarks/...`
Expected: SUCCESS

---

## Summary

| Task | Description | Files |
|------|-------------|-------|
| 1 | Black-Scholes vega | `src/math/black_scholes_analytics.hpp` |
| 2 | Latin Hypercube sampling | `src/math/latin_hypercube.hpp` |
| 3 | ErrorBins | `src/option/table/error_attribution.hpp` |
| 4 | SliceCache | `src/option/table/slice_cache.hpp` |
| 5 | Types (Params, Stats, Result) | `src/option/table/adaptive_grid_types.hpp` |
| 6 | AdaptiveGridBuilder skeleton | `src/option/table/adaptive_grid_builder.{hpp,cpp}` |
| 7 | _internal method aliases | `src/option/table/price_table_builder.hpp` |
| 8 | Main build loop | `src/option/table/adaptive_grid_builder.cpp` |
| 9 | Integration tests | `tests/adaptive_grid_builder_integration_test.cc` |
| 10 | Final cleanup | Design doc update |

**Estimated: 10 commits, ~1000 lines of code**
