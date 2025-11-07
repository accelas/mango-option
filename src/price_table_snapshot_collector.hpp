#pragma once

#include "snapshot.hpp"
#include "snapshot_interpolator.hpp"
#include <span>
#include <vector>
#include <cmath>
#include <limits>
#include <algorithm>
#include <cassert>

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

        // Build interpolators (grid should always be valid from PDE solver)
        auto V_error = V_interp.build(snapshot.spatial_grid, snapshot.solution);
        auto Lu_error = Lu_interp.build(snapshot.spatial_grid, snapshot.spatial_operator);

        // Assert on failure - indicates programming error in PDE solver
        assert(!V_error.has_value() && "Failed to build value interpolator");
        assert(!Lu_error.has_value() && "Failed to build spatial operator interpolator");

        // Fill price table for all moneyness points
        for (size_t m_idx = 0; m_idx < moneyness_.size(); ++m_idx) {
            const double m = moneyness_[m_idx];

            // CRITICAL: The PDE works in log-moneyness space x = ln(S/K)
            // snapshot.spatial_grid contains x values, NOT dollar spots
            // snapshot.solution contains NORMALIZED prices V_norm = V_dollar / K

            // Convert moneyness to log-moneyness: x = ln(m) = ln(S/K_ref)
            const double x = std::log(m);
            const double S = m * K_ref_;  // For later use in chain rule

            const size_t table_idx = m_idx * tau_.size() + tau_idx;

            // Interpolate NORMALIZED price at log-moneyness x
            const double V_norm = V_interp.eval(x);

            // Convert to DOLLAR price: V_dollar = K_ref * V_norm
            prices_[table_idx] = K_ref_ * V_norm;

            // Interpolate normalized delta from PDE data: dV_norm/dx
            const double dVnorm_dx = V_interp.eval_from_data(x, snapshot.first_derivative);

            // Transform to dollar delta using chain rule:
            // V_dollar(S) = K_ref * V_norm(x(S)) where x = ln(S/K_ref)
            // ∂V_dollar/∂S = K_ref * ∂V_norm/∂x * ∂x/∂S
            //              = K_ref * dVnorm/dx * (1/S)
            //              = (K_ref/S) * dVnorm/dx
            const double delta_dollar = (K_ref_ / S) * dVnorm_dx;
            deltas_[table_idx] = delta_dollar;

            // Interpolate normalized second derivative: d²V_norm/dx²
            const double d2Vnorm_dx2 = V_interp.eval_from_data(x, snapshot.second_derivative);

            // Transform to dollar gamma using chain rule:
            // gamma = ∂²V_dollar/∂S²
            //       = ∂/∂S[(K_ref/S) * dV_norm/dx]
            //       = K_ref * ∂/∂S[(1/S) * dV_norm/dx]
            //       = K_ref * [(-1/S²) * dV_norm/dx + (1/S) * d(dV_norm/dx)/dS]
            //       = K_ref * [(-1/S²) * dV_norm/dx + (1/S) * d²V_norm/dx² * dx/dS]
            //       = K_ref * [(-1/S²) * dV_norm/dx + (1/S) * d²V_norm/dx² * (1/S)]
            //       = (K_ref/S²) * [d²V_norm/dx² - dV_norm/dx]
            const double gamma_dollar = (K_ref_ / (S * S)) * (d2Vnorm_dx2 - dVnorm_dx);
            gammas_[table_idx] = gamma_dollar;

            // Theta computation
            if (exercise_type_ == ExerciseType::EUROPEAN) {
                // European: theta = -L(V) everywhere
                // L(V) is also in normalized space
                const double Lu_norm = Lu_interp.eval(x);
                const double Lu_dollar = K_ref_ * Lu_norm;
                thetas_[table_idx] = -Lu_dollar;
            } else {
                // American: theta = -L(V) in continuation region, NaN at boundary
                const double obstacle = compute_american_obstacle(S, snapshot.time);
                const double BOUNDARY_TOLERANCE = 1e-6;

                if (std::abs(prices_[table_idx] - obstacle) < BOUNDARY_TOLERANCE) {
                    // At exercise boundary
                    thetas_[table_idx] = std::numeric_limits<double>::quiet_NaN();
                } else {
                    // In continuation region
                    const double Lu_norm = Lu_interp.eval(x);
                    const double Lu_dollar = K_ref_ * Lu_norm;
                    thetas_[table_idx] = -Lu_dollar;
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
