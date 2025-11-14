#pragma once

#include <cstdint>
#include <cstddef>

namespace mango {

/// CRC64-ECMA implementation for data integrity checking
/// Uses polynomial 0x42F0E1EBA9EA3693 (standardized by ECMA)
class CRC64 {
public:
    /// Compute CRC64-ECMA checksum for double array
    ///
    /// @param data Pointer to double array
    /// @param count Number of doubles in array
    /// @return 64-bit checksum
    static uint64_t compute(const double* data, size_t count);

private:
    /// CRC64-ECMA polynomial (reversed)
    static constexpr uint64_t POLY = 0xC96C5795D7870F42ULL;

    /// Lookup table for fast CRC64 computation (initialized on first use)
    static uint64_t table_[256];
    static bool table_initialized_;

    /// Initialize lookup table (called automatically on first use)
    static void init_table();
};

}  // namespace mango
