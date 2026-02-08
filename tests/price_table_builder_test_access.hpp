// SPDX-License-Identifier: MIT
#pragma once
#include "mango/option/table/price_table_builder.hpp"

namespace mango::testing {

template <size_t N>
struct PriceTableBuilderAccess {
    static inline auto make_batch(const PriceTableBuilderND<N>& b, const PriceTableAxesND<N>& a) {
        return b.make_batch(a);
    }
    static inline auto solve_batch(const PriceTableBuilderND<N>& b,
                                    const std::vector<PricingParams>& batch,
                                    const PriceTableAxesND<N>& a) {
        return b.solve_batch(batch, a);
    }
    static inline auto extract_tensor(const PriceTableBuilderND<N>& b,
                                       const BatchAmericanOptionResult& batch,
                                       const PriceTableAxesND<N>& a) {
        return b.extract_tensor(batch, a);
    }
    // Note: fit_coeffs returns FitCoeffsResult (private struct).
    // The _for_testing methods extracted .coefficients. We replicate that.
    static inline auto fit_coeffs(const PriceTableBuilderND<N>& b,
                                   const PriceTensorND<N>& tensor,
                                   const PriceTableAxesND<N>& a) {
        auto result = b.fit_coeffs(tensor, a);
        if (!result.has_value()) {
            return std::expected<std::vector<double>, PriceTableError>(
                std::unexpected(result.error()));
        }
        return std::expected<std::vector<double>, PriceTableError>(
            std::move(result.value().coefficients));
    }
    static inline auto find_nearest_valid_neighbor(const PriceTableBuilderND<N>& b,
                                                    size_t s_idx, size_t r_idx,
                                                    size_t Ns, size_t Nr,
                                                    const std::vector<bool>& valid) {
        return b.find_nearest_valid_neighbor(s_idx, r_idx, Ns, Nr, valid);
    }
    static inline auto repair_failed_slices(const PriceTableBuilderND<N>& b,
                                             PriceTensorND<N>& tensor,
                                             const std::vector<size_t>& failed_pde,
                                             const std::vector<std::tuple<size_t, size_t, size_t>>& failed_spline,
                                             const PriceTableAxesND<N>& axes) {
        return b.repair_failed_slices(tensor, failed_pde, failed_spline, axes);
    }
};

}  // namespace mango::testing
