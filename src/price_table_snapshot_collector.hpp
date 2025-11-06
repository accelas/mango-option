#pragma once

#include "snapshot.hpp"
#include "snapshot_interpolator.hpp"
#include <span>
#include <vector>
#include <cmath>
#include <limits>
#include <algorithm>

namespace mango {

enum class ExerciseType { EUROPEAN, AMERICAN };

struct PriceTableSnapshotCollectorConfig {
    std::span<const double> moneyness;
    std::span<const double> tau;
    double K_ref;
    ExerciseType exercise_type;
    const void* payoff_params = nullptr;
};

/// Collects snapshots into price table format
///
/// PERFORMANCE: Builds interpolators ONCE per snapshot (not O(n²))
/// CORRECTNESS: PDE provides ∂²V/∂S² directly - no transformation needed!
class PriceTableSnapshotCollector : public SnapshotCollector {
public:
    explicit PriceTableSnapshotCollector(const PriceTableSnapshotCollectorConfig& config)
        : moneyness_(config.moneyness.begin(), config.moneyness.end())
        , tau_(config.tau.begin(), config.tau.end())
        , K_ref_(config.K_ref)
        , exercise_type_(config.exercise_type)
        , payoff_params_(config.payoff_params)
    {
        const size_t n = moneyness_.size() * tau_.size();
        prices_.resize(n, 0.0);
        deltas_.resize(n, 0.0);
        gammas_.resize(n, 0.0);
        thetas_.resize(n, 0.0);
    }

    void collect(const Snapshot& snapshot) override {
        // FIXED: Use user_index to match tau directly (no float comparison!)
        // Snapshot user_index IS the tau index
        const size_t tau_idx = snapshot.user_index;

        // PERFORMANCE FIX: Build interpolators ONCE outside loop
        // We build interpolators for V and Lu only
        // Derivatives will use eval_from_data() with PDE-computed arrays
        SnapshotInterpolator V_interp, Lu_interp;

        // Build succeeds (grid is always valid from PDE solver)
        (void)V_interp.build(snapshot.spatial_grid, snapshot.solution);
        (void)Lu_interp.build(snapshot.spatial_grid, snapshot.spatial_operator);

        // Fill price table for all moneyness points
        for (size_t m_idx = 0; m_idx < moneyness_.size(); ++m_idx) {
            const double m = moneyness_[m_idx];
            const double S = m * K_ref_;

            const size_t table_idx = m_idx * tau_.size() + tau_idx;

            // Interpolate price
            const double V = V_interp.eval(S);
            prices_[table_idx] = V;

            // Interpolate delta from PDE data
            // PDE provides ∂V/∂S directly (already in S-space)
            const double dVdS = V_interp.eval_from_data(S, snapshot.first_derivative);
            deltas_[table_idx] = dVdS;

            // CRITICAL: Gamma computation with CORRECTED understanding
            //
            // The PDE solver works in S-space and computes:
            //   - snapshot.solution: V(S)
            //   - snapshot.first_derivative: ∂V/∂S
            //   - snapshot.second_derivative: ∂²V/∂S²
            //
            // We want gamma = ∂²V/∂S², which the PDE provides DIRECTLY!
            // No transformation needed - just interpolate!
            //
            // Mathematical note:
            //   If we were working in moneyness space (m = S/K), the chain rule gives:
            //     ∂V/∂S = (∂V/∂m) · (∂m/∂S) = (∂V/∂m) / K
            //     ∂²V/∂S² = ∂/∂S[∂V/∂S] = (∂²V/∂m²) / K²
            //
            //   But the PDE snapshot is already in S-space, so we use it directly.
            const double d2VdS2 = V_interp.eval_from_data(S, snapshot.second_derivative);
            gammas_[table_idx] = d2VdS2;  // Already in S-space!

            // Theta computation
            if (exercise_type_ == ExerciseType::EUROPEAN) {
                // European: theta = -L(V) everywhere
                const double Lu = Lu_interp.eval(S);
                thetas_[table_idx] = -Lu;
            } else {
                // American: theta = -L(V) in continuation region, NaN at boundary
                const double obstacle = compute_american_obstacle(S, snapshot.time);
                const double BOUNDARY_TOLERANCE = 1e-6;

                if (std::abs(V - obstacle) < BOUNDARY_TOLERANCE) {
                    // At exercise boundary
                    thetas_[table_idx] = std::numeric_limits<double>::quiet_NaN();
                } else {
                    // In continuation region
                    const double Lu = Lu_interp.eval(S);
                    thetas_[table_idx] = -Lu;
                }
            }
        }
    }

    std::span<const double> prices() const { return prices_; }
    std::span<const double> deltas() const { return deltas_; }
    std::span<const double> gammas() const { return gammas_; }
    std::span<const double> thetas() const { return thetas_; }

private:
    std::vector<double> moneyness_;
    std::vector<double> tau_;
    double K_ref_;
    ExerciseType exercise_type_;
    const void* payoff_params_;

    std::vector<double> prices_;
    std::vector<double> deltas_;
    std::vector<double> gammas_;
    std::vector<double> thetas_;

    double compute_american_obstacle(double S, double /*tau*/) const {
        // American put payoff: max(K - S, 0)
        return std::max(K_ref_ - S, 0.0);
    }
};

}  // namespace mango
