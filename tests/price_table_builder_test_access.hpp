// SPDX-License-Identifier: MIT
#pragma once
#include "src/option/table/price_table_builder.hpp"

namespace mango::testing {

template <size_t N>
struct PriceTableBuilderAccess {
    static inline auto make_batch(const PriceTableBuilder<N>& b, const PriceTableAxes<N>& a) {
        return b.make_batch(a);
    }
    static inline auto solve_batch(const PriceTableBuilder<N>& b,
                                    const std::vector<PricingParams>& batch,
                                    const PriceTableAxes<N>& a) {
        return b.solve_batch(batch, a);
    }
    static inline auto extract_tensor(const PriceTableBuilder<N>& b,
                                       const BatchAmericanOptionResult& batch,
                                       const PriceTableAxes<N>& a) {
        return b.extract_tensor(batch, a);
    }
    // Note: fit_coeffs returns FitCoeffsResult (private struct).
    // The _for_testing methods extracted .coefficients. We replicate that.
    static inline auto fit_coeffs(const PriceTableBuilder<N>& b,
                                   const PriceTensor<N>& tensor,
                                   const PriceTableAxes<N>& a) {
        auto result = b.fit_coeffs(tensor, a);
        if (!result.has_value()) {
            return std::expected<std::vector<double>, PriceTableError>(
                std::unexpected(result.error()));
        }
        return std::expected<std::vector<double>, PriceTableError>(
            std::move(result.value().coefficients));
    }
    static inline auto find_nearest_valid_neighbor(const PriceTableBuilder<N>& b,
                                                    size_t s_idx, size_t r_idx,
                                                    size_t Ns, size_t Nr,
                                                    const std::vector<bool>& valid) {
        return b.find_nearest_valid_neighbor(s_idx, r_idx, Ns, Nr, valid);
    }
    static inline auto repair_failed_slices(const PriceTableBuilder<N>& b,
                                             PriceTensor<N>& tensor,
                                             const std::vector<size_t>& failed_pde,
                                             const std::vector<std::tuple<size_t, size_t, size_t>>& failed_spline,
                                             const PriceTableAxes<N>& axes) {
        return b.repair_failed_slices(tensor, failed_pde, failed_spline, axes);
    }
};

}  // namespace mango::testing
