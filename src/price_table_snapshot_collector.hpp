#pragma once

#include "snapshot.hpp"
#include "snapshot_interpolator.hpp"
#include "american_option.hpp"  // For OptionType enum
#include <span>
#include <vector>
#include <cmath>
#include <limits>
#include <algorithm>
#include <cassert>

namespace mango {

struct PriceTableSnapshotCollectorConfig {
    std::span<const double> moneyness;
    std::span<const double> tau;
    double K_ref;
    OptionType option_type = OptionType::CALL;  // Used for obstacle computation
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
        , option_type_(config.option_type)
        , payoff_params_(config.payoff_params)
    {
        const size_t n = moneyness_.size() * tau_.size();
        prices_.resize(n, 0.0);
        deltas_.resize(n, 0.0);
        gammas_.resize(n, 0.0);
        thetas_.resize(n, 0.0);

        // PERFORMANCE: Precompute log-moneyness and scaling factors
        // These are constant across all snapshots, so cache them to avoid
        // repeated transcendentals and divisions in the hot path
        log_moneyness_.resize(moneyness_.size());
        spot_values_.resize(moneyness_.size());
        inv_spot_.resize(moneyness_.size());
        inv_spot_sq_.resize(moneyness_.size());

        for (size_t i = 0; i < moneyness_.size(); ++i) {
            const double m = moneyness_[i];
            log_moneyness_[i] = std::log(m);           // x = ln(m)
            spot_values_[i] = m * K_ref_;               // S = m * K_ref
            inv_spot_[i] = 1.0 / spot_values_[i];      // 1/S
            inv_spot_sq_[i] = inv_spot_[i] * inv_spot_[i];  // 1/S²
        }
    }

    void collect(const Snapshot& snapshot) override {
        // FIXED: Use user_index to match tau directly (no float comparison!)
        // Snapshot user_index IS the tau index
        const size_t tau_idx = snapshot.user_index;

        // Build interpolators (grid should always be valid from PDE solver)
        auto V_error = value_interp_.build(snapshot.spatial_grid, snapshot.solution);
        auto Lu_error = lu_interp_.build(snapshot.spatial_grid, snapshot.spatial_operator);

        // Assert on failure - indicates programming error in PDE solver
        assert(!V_error.has_value() && "Failed to build value interpolator");
        assert(!Lu_error.has_value() && "Failed to build spatial operator interpolator");

        // Fill price table for all moneyness points
        for (size_t m_idx = 0; m_idx < moneyness_.size(); ++m_idx) {
            // PERFORMANCE: Use precomputed values instead of recomputing
            const double x = log_moneyness_[m_idx];      // Cached ln(m)
            const double S = spot_values_[m_idx];         // Cached m * K_ref
            const double inv_S = inv_spot_[m_idx];        // Cached 1/S
            const double inv_S2 = inv_spot_sq_[m_idx];    // Cached 1/S²

            const size_t table_idx = m_idx * tau_.size() + tau_idx;

            // Interpolate NORMALIZED price at log-moneyness x
            const double V_norm = value_interp_.eval(x);

            // Convert to DOLLAR price: V_dollar = K_ref * V_norm
            prices_[table_idx] = K_ref_ * V_norm;

            // Interpolate normalized delta from PDE data: dV_norm/dx
            const double dVnorm_dx = value_interp_.eval_from_data(x, snapshot.first_derivative);

            // Transform to dollar delta using chain rule:
            // PERFORMANCE: Use FMA for better precision and potential FMA instruction
            const double delta_scale = K_ref_ * inv_S;
            deltas_[table_idx] = delta_scale * dVnorm_dx;

            // Interpolate normalized second derivative: d²V_norm/dx²
            const double d2Vnorm_dx2 = value_interp_.eval_from_data(x, snapshot.second_derivative);

            // Transform to dollar gamma using chain rule:
            // gamma = (K_ref/S²) * [d²V_norm/dx² - dV_norm/dx]
            // PERFORMANCE: Use FMA to reduce rounding and enable fused instructions
            const double gamma_scale = K_ref_ * inv_S2;
            gammas_[table_idx] = std::fma(gamma_scale, d2Vnorm_dx2, -gamma_scale * dVnorm_dx);

            // Theta computation
            // American exercise: theta = -L(V) in continuation region, NaN at boundary
            const double obstacle = compute_american_obstacle(S, snapshot.time);
            const double BOUNDARY_TOLERANCE = 1e-6;

            if (std::abs(prices_[table_idx] - obstacle) < BOUNDARY_TOLERANCE) {
                // At exercise boundary
                thetas_[table_idx] = std::numeric_limits<double>::quiet_NaN();
            } else {
                // In continuation region
                const double Lu_norm = lu_interp_.eval(x);
                thetas_[table_idx] = -(K_ref_ * Lu_norm);
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
    OptionType option_type_;
    const void* payoff_params_;

    std::vector<double> prices_;
    std::vector<double> deltas_;
    std::vector<double> gammas_;
    std::vector<double> thetas_;

    // PERFORMANCE: Precomputed values to avoid repeated transcendentals
    std::vector<double> log_moneyness_;  ///< Cached ln(m) for each moneyness point
    std::vector<double> spot_values_;    ///< Cached S = m * K_ref
    std::vector<double> inv_spot_;       ///< Cached 1/S
    std::vector<double> inv_spot_sq_;    ///< Cached 1/S²

    SnapshotInterpolator value_interp_;
    SnapshotInterpolator lu_interp_;

    double compute_american_obstacle(double S, double /*tau*/) const {
        // American option intrinsic value (exercise boundary)
        if (option_type_ == OptionType::CALL) {
            return std::max(S - K_ref_, 0.0);  // Call: max(S - K, 0)
        } else {
            return std::max(K_ref_ - S, 0.0);  // Put: max(K - S, 0)
        }
    }
};

}  // namespace mango
