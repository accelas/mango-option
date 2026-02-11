// SPDX-License-Identifier: MIT
#pragma once

#include "mango/option/table/dimensionless/dimensionless_builder.hpp"
#include "mango/option/table/eep/eep_decomposer.hpp"

#include <cmath>
#include <cstddef>
#include <vector>

namespace mango {

/// EEP accessor for a 3D dimensionless grid (x, tau', ln_kappa).
///
/// Wraps the flat V/K vector from solve_dimensionless_pde() and maps
/// dimensionless coordinates to canonical physical parameters for
/// AnalyticalEEP: spot = K_ref*exp(x), strike = K_ref,
/// tau = tau', sigma = sqrt(2), rate = exp(ln_kappa).
///
/// Layout: [x][tau'][ln_kappa] row-major, ln_kappa innermost.
class Dimensionless3DAccessor {
public:
    Dimensionless3DAccessor(std::vector<double>& values,
                            const DimensionlessAxes& axes,
                            double K_ref)
        : values_(values), axes_(axes), K_ref_(K_ref),
          Nm_(axes.log_moneyness.size()),
          Nt_(axes.tau_prime.size()),
          Nk_(axes.ln_kappa.size()) {}

    size_t size() const { return Nm_ * Nt_ * Nk_; }
    double strike() const { return K_ref_; }

    /// Dollar American price: V/K * K_ref
    double american_price(size_t i) const {
        return K_ref_ * values_[i];
    }

    /// Spot = K_ref * exp(x)
    double spot(size_t i) const {
        return std::exp(axes_.log_moneyness[to_3d(i).mi]) * K_ref_;
    }

    /// tau = tau' (dimensionless time IS physical tau when sigma=sqrt(2))
    double tau(size_t i) const {
        return axes_.tau_prime[to_3d(i).ti];
    }

    /// sigma = sqrt(2) always (the PDE's effective sigma)
    double sigma(size_t /*i*/) const { return std::sqrt(2.0); }

    /// rate = kappa = exp(ln_kappa)
    double rate(size_t i) const {
        return std::exp(axes_.ln_kappa[to_3d(i).ki]);
    }

    /// Store value directly (dollar EEP after eep_decompose)
    void set_value(size_t i, double v) { values_[i] = v; }

private:
    struct Idx3D { size_t mi, ti, ki; };

    Idx3D to_3d(size_t flat) const {
        size_t ki = flat % Nk_;  flat /= Nk_;
        size_t ti = flat % Nt_;
        size_t mi = flat / Nt_;
        return {mi, ti, ki};
    }

    std::vector<double>& values_;
    const DimensionlessAxes& axes_;
    double K_ref_;
    size_t Nm_, Nt_, Nk_;
};

}  // namespace mango
