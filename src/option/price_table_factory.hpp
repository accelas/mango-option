// SPDX-License-Identifier: MIT
#pragma once

#include "mango/option/interpolated_iv_solver.hpp"
#include "mango/option/iv_result.hpp"
#include "mango/option/option_spec.hpp"
#include "mango/option/table/adaptive_grid_types.hpp"
#include "mango/option/table/greek_types.hpp"
#include "mango/option/table/serialization/price_table_data.hpp"
#include "mango/support/error_types.hpp"

#include <expected>
#include <filesystem>
#include <memory>
#include <optional>
#include <string>

namespace mango {

enum class PriceTableCompression {
    NONE,
    SNAPPY,
    ZSTD,
};

class AnyPriceTable {
public:
    [[nodiscard]] std::string surface_type() const;
    [[nodiscard]] OptionType option_type() const noexcept;
    [[nodiscard]] double dividend_yield() const noexcept;

    [[nodiscard]] std::expected<void, ValidationError>
    validate_pricing_params(const PricingParams& params) const;

    [[nodiscard]] double price(const PricingParams& params) const;
    [[nodiscard]] double vega(const PricingParams& params) const;
    [[nodiscard]] std::expected<double, GreekError> delta(const PricingParams& params) const;
    [[nodiscard]] std::expected<double, GreekError> gamma(const PricingParams& params) const;
    [[nodiscard]] std::expected<double, GreekError> theta(const PricingParams& params) const;
    [[nodiscard]] std::expected<double, GreekError> rho(const PricingParams& params) const;

    /// @param build_dividends Discrete schedule for validate_query.
    ///        nullopt = use the table's stored build-time schedule when
    ///        known (set by make_price_table for freshly built tables),
    ///        otherwise infer from table type: segmented (MultiKRef)
    ///        tables get "unknown" (checks skipped — schedules are not
    ///        persisted to Parquet, so tables loaded via load_price_table
    ///        lack this provenance), all others get known-empty.
    [[nodiscard]] std::expected<AnyInterpIVSolver, ValidationError>
    make_iv_solver(const InterpolatedIVSolverConfig& config = {},
                   std::optional<std::vector<Dividend>> build_dividends =
                       std::nullopt) const;

    /// Diagnostics from adaptive grid refinement (spec D7).  `nullopt` for
    /// manually-gridded tables and tables loaded from Parquet -- diagnostics
    /// never enter serialization.
    [[nodiscard]] std::optional<BuildDiagnostics> build_diagnostics() const;

    /// Convenience one-shot IV solve. Constructs a fresh solver (bounds
    /// extraction + variant dispatch) on every call — not intended for
    /// repeated or hot-path queries. For those, create the solver once
    /// via make_iv_solver() and reuse it.
    [[nodiscard]] std::expected<IVSuccess, IVError>
    solve_iv(const IVQuery& query,
             const InterpolatedIVSolverConfig& config = {}) const;

    [[nodiscard]] PriceTableData to_data() const;

    [[nodiscard]] std::expected<void, PriceTableError>
    save(const std::filesystem::path& path,
         PriceTableCompression compression = PriceTableCompression::ZSTD) const;

    struct Impl;
    explicit AnyPriceTable(std::unique_ptr<Impl> impl);
    AnyPriceTable(AnyPriceTable&&) noexcept;
    AnyPriceTable& operator=(AnyPriceTable&&) noexcept;
    ~AnyPriceTable();

private:
    std::unique_ptr<Impl> impl_;
};

[[nodiscard]] std::expected<AnyPriceTable, ValidationError>
make_price_table(const IVSolverFactoryConfig& config);

[[nodiscard]] std::expected<AnyPriceTable, PriceTableError>
load_price_table(const std::filesystem::path& path);

}  // namespace mango
