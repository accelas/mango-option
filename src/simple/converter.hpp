/**
 * @file converter.hpp
 * @brief Type-safe converter traits for data sources
 */

#pragma once

#include "src/simple/price.hpp"
#include "src/simple/timestamp.hpp"
#include "src/simple/option_types.hpp"
#include "src/option/option_spec.hpp"
#include <concepts>
#include <stdexcept>

namespace mango::simple {

// Source tag types
struct YFinanceSource {};
struct DatabentSource {};
struct IBKRSource {};

/// Conversion error exception
class ConversionError : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

/// Converter trait - must be specialized for each source
template<typename Source>
struct Converter;

/// Concept for valid converter
template<typename T>
concept ValidConverter = requires {
    // Must have price conversion
    { T::to_price(std::declval<double>()) } -> std::same_as<Price>;
    // Must have option type conversion
    { T::to_option_type(std::declval<const char*>()) } -> std::same_as<mango::OptionType>;
};

}  // namespace mango::simple
