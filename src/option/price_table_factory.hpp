// SPDX-License-Identifier: MIT
#pragma once

#include "mango/option/interpolated_iv_solver.hpp"
#include "mango/option/iv_result.hpp"
#include "mango/option/option_spec.hpp"
#include "mango/option/table/greek_types.hpp"
#include "mango/option/table/serialization/price_table_data.hpp"
#include "mango/support/error_types.hpp"

#include <expected>
#include <filesystem>
#include <memory>
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

    [[nodiscard]] double price(const PricingParams& params) const;
    [[nodiscard]] double vega(const PricingParams& params) const;
    [[nodiscard]] std::expected<double, GreekError> delta(const PricingParams& params) const;
    [[nodiscard]] std::expected<double, GreekError> gamma(const PricingParams& params) const;
    [[nodiscard]] std::expected<double, GreekError> theta(const PricingParams& params) const;
    [[nodiscard]] std::expected<double, GreekError> rho(const PricingParams& params) const;

    [[nodiscard]] std::expected<AnyInterpIVSolver, ValidationError>
    make_iv_solver(const InterpolatedIVSolverConfig& config = {}) const;

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
