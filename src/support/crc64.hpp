// SPDX-License-Identifier: MIT
#pragma once

#include <cstdint>
#include <cstddef>
#include <mutex>

namespace mango {

/// CRC64-ECMA-182 implementation for data integrity checking
/// Uses polynomial 0x42F0E1EBA9EA3693 (ECMA-182 standard)
/// Initial value: 0x0, Final XOR: 0x0
class CRC64 {
public:
    /// Compute CRC64-ECMA-182 checksum for double array
    ///
    /// @param data Pointer to double array
    /// @param count Number of doubles in array
    /// @return 64-bit checksum
    static uint64_t compute(const double* data, size_t count);

    /// Compute CRC64-ECMA-182 checksum for raw byte array
    static uint64_t compute_bytes(const uint8_t* data, size_t byte_count);

    /// Incrementally update a running CRC with more bytes.
    static uint64_t update(uint64_t crc, const uint8_t* data, size_t byte_count);

private:
    /// CRC64-ECMA-182 polynomial (reversed for LSB-first processing)
    static constexpr uint64_t POLY = 0xC96C5795D7870F42ULL;

    /// Lookup table for fast CRC64 computation (initialized on first use)
    static uint64_t table_[256];
    static std::once_flag init_flag_;

    /// Initialize lookup table (called automatically on first use, thread-safe)
    static void init_table();
};

}  // namespace mango
