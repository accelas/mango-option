// SPDX-License-Identifier: MIT
#include "src/option/table/segmented_multi_kref_builder.hpp"

#include <algorithm>
#include <cmath>

namespace mango {

std::expected<SegmentedMultiKRefSurface, ValidationError>
SegmentedMultiKRefBuilder::build(const Config& config) {
    // Determine K_ref values
    std::vector<double> K_refs = config.kref_config.K_refs;

    if (K_refs.empty()) {
        // Auto selection: log-spaced from spot*(1-span) to spot*(1+span)
        // Log spacing naturally concentrates points near ATM.
        const int count = config.kref_config.K_ref_count;
        const double span = config.kref_config.K_ref_span;

        if (count < 1) {
            return std::unexpected(ValidationError{ValidationErrorCode::InvalidGridSize, static_cast<double>(count)});
        }
        if (span <= 0.0) {
            return std::unexpected(ValidationError{ValidationErrorCode::InvalidBounds, span});
        }

        const double log_lo = std::log(1.0 - span);
        const double log_hi = std::log(1.0 + span);

        K_refs.reserve(static_cast<size_t>(count));
        if (count == 1) {
            K_refs.push_back(config.spot);
        } else {
            for (int i = 0; i < count; ++i) {
                double t = static_cast<double>(i) / static_cast<double>(count - 1);
                K_refs.push_back(config.spot * std::exp(log_lo + t * (log_hi - log_lo)));
            }
        }
    }

    // Build per-K_ref surfaces
    std::vector<SegmentedMultiKRefSurface::Entry> entries;
    entries.reserve(K_refs.size());

    for (double K_ref : K_refs) {
        SegmentedPriceTableBuilder::Config seg_config{
            .K_ref = K_ref,
            .option_type = config.option_type,
            .dividend_yield = config.dividend_yield,
            .dividends = config.dividends,
            .moneyness_grid = config.moneyness_grid,
            .maturity = config.maturity,
            .vol_grid = config.vol_grid,
            .rate_grid = config.rate_grid,
        };

        auto surface = SegmentedPriceTableBuilder::build(seg_config);
        if (!surface.has_value()) {
            return std::unexpected(surface.error());
        }

        entries.push_back({K_ref, std::move(*surface)});
    }

    return SegmentedMultiKRefSurface::create(std::move(entries));
}

}  // namespace mango
