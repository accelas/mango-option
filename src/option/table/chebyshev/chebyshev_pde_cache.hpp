// SPDX-License-Identifier: MIT
#pragma once

#include "mango/math/cubic_spline_solver.hpp"
#include "mango/option/table/pde_cache.hpp"
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <map>
#include <span>
#include <utility>
#include <vector>

namespace mango {

/// (sigma, rate) -> per-tau_idx CubicSpline cache for Chebyshev adaptive builder.
///
/// Wraps PairCacheCore with per-tau-index spline storage and cumulative PDE
/// solve tracking.  Uses 1e12 quantization for CC-level node reuse stability.
class ChebyshevPDECache {
public:
    ChebyshevPDECache() : core_(PairKeyCodec{1e12}) {}

    /// Store a spline for (sigma, rate, tau_idx).
    void store_slice(double sigma, double rate, size_t tau_idx,
                     std::span<const double> x_grid,
                     std::span<const double> values) {
        auto& tau_map = core_.upsert(sigma, rate);
        auto& entry = tau_map[tau_idx];
        auto err = entry.spline.build(x_grid, values);
        entry.valid = !err.has_value();

        // [STORE] sanity-check: values fed in, then spline.eval at a fixed
        // point. If values are clean but eval is NaN, spline build is buggy.
        if (entry.valid) {
            size_t n_nan_in = 0;
            for (double v : values) if (std::isnan(v)) ++n_nan_in;
            // Probe the spline at the first grid point (must equal values[0]
            // for a natural cubic spline interpolating those values).
            double probe_at_first = entry.spline.eval(x_grid.front());
            double probe_at_mid = entry.spline.eval(
                (x_grid.front() + x_grid.back()) * 0.5);
            if (std::isnan(probe_at_first) || std::isnan(probe_at_mid) || n_nan_in > 0) {
                std::fprintf(stderr,
                    "[STORE-NaN] sigma=%.6f rate=%.6f tau_idx=%zu x_grid.size=%zu "
                    "values_nan=%zu probe_at_first=%.6g (expected %.6g) probe_mid=%.6g\n",
                    sigma, rate, tau_idx, x_grid.size(), n_nan_in,
                    probe_at_first, values.empty() ? 0.0 : values.front(),
                    probe_at_mid);
            }
        } else {
            std::fprintf(stderr,
                "[STORE-FAIL] sigma=%.6f rate=%.6f tau_idx=%zu — spline.build error\n",
                sigma, rate, tau_idx);
        }
    }

    /// Check if any tau slices exist for (sigma, rate).
    [[nodiscard]] bool has_slice(double sigma, double rate) const {
        return core_.contains(sigma, rate);
    }

    /// Retrieve the spline for (sigma, rate, tau_idx), or nullptr.
    [[nodiscard]] const CubicSpline<double>*
    get_slice(double sigma, double rate, size_t tau_idx) const {
        auto* tau_map = core_.get(sigma, rate);
        if (!tau_map) return nullptr;
        auto jt = tau_map->find(tau_idx);
        if (jt == tau_map->end() || !jt->second.valid) return nullptr;
        return &jt->second.spline;
    }

    /// Number of distinct (sigma, rate) pairs cached.
    [[nodiscard]] size_t num_cached_pairs() const { return core_.size(); }

    /// Number of tau slices stored for a given (sigma, rate) pair.
    [[nodiscard]] size_t num_tau_slices(double sigma, double rate) const {
        auto* tau_map = core_.get(sigma, rate);
        return tau_map ? tau_map->size() : 0;
    }

    /// Given wanted sigma and rate arrays, return (sigma_idx, rate_idx)
    /// pairs that are NOT yet cached.
    [[nodiscard]] std::vector<std::pair<size_t, size_t>>
    missing_pairs(std::span<const double> sigma_values,
                  std::span<const double> rate_values) const {
        return core_.missing_pairs(sigma_values, rate_values);
    }

    /// Record cumulative PDE solve count.
    void record_pde_solves(size_t count) { total_pde_solves_ += count; }

    /// Total PDE solves performed to populate this cache.
    [[nodiscard]] size_t total_pde_solves() const { return total_pde_solves_; }

    /// Clear all cached slices (preserves cumulative PDE solve count).
    void clear() { core_.clear(); }

private:
    struct SliceEntry {
        CubicSpline<double> spline;
        bool valid = false;
    };

    PairCacheCore<std::map<size_t, SliceEntry>> core_;
    size_t total_pde_solves_ = 0;
};

}  // namespace mango
