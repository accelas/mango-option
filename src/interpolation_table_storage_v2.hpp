#ifndef MANGO_INTERPOLATION_TABLE_STORAGE_V2_HPP
#define MANGO_INTERPOLATION_TABLE_STORAGE_V2_HPP

#include <memory>
#include <string>
#include <vector>
#include "expected.hpp"

namespace mango {

// Forward declarations
class BSpline4D_FMA;

/**
 * Memory-mapped storage format for 4D interpolation tables
 *
 * Binary format layout:
 * - Header (256 bytes, aligned)
 * - Moneyness knots array
 * - Maturity knots array
 * - Volatility knots array
 * - Rate knots array
 * - Coefficients array (4D tensor, row-major)
 *
 * Features:
 * - Memory-mapped loading for instant access
 * - Zero-copy deserialization
 * - Compact binary format (~2-5 MB for typical tables)
 * - Self-describing with version header
 *
 * File format compatible with Apache Arrow philosophy (columnar, aligned)
 * but uses a simpler custom format for easier integration.
 */

#pragma pack(push, 1)
struct InterpolationTableHeader {
    uint32_t magic;              // 0x4D494E54 ('MINT' = Mango INTerpolation)
    uint32_t version;            // Format version (currently 1)
    double K_ref;                // Reference strike price
    uint32_t option_type;        // 0=PUT, 1=CALL
    uint32_t spline_degree;      // B-spline degree (typically 3)

    // Grid dimensions
    uint64_t n_moneyness;
    uint64_t n_maturity;
    uint64_t n_volatility;
    uint64_t n_rate;
    uint64_t n_coefficients;

    // Data offsets from file start (for alignment)
    uint64_t moneyness_offset;
    uint64_t maturity_offset;
    uint64_t volatility_offset;
    uint64_t rate_offset;
    uint64_t coefficients_offset;

    char option_type_str[16];    // "PUT" or "CALL"
    char reserved[136];          // Reserved for future use
};
#pragma pack(pop)

static_assert(sizeof(InterpolationTableHeader) == 256, "Header must be exactly 256 bytes");

class InterpolationTableStorage {
public:
    /**
     * Save interpolation table to disk in memory-mapped format
     *
     * @param filepath Path to save the table
     * @param moneyness_knots Knot vector for moneyness dimension
     * @param maturity_knots Knot vector for maturity dimension
     * @param volatility_knots Knot vector for volatility dimension
     * @param rate_knots Knot vector for rate dimension
     * @param coefficients Flattened 4D coefficient array (row-major order)
     * @param K_ref Reference strike price
     * @param option_type "PUT" or "CALL"
     * @param spline_degree Degree of B-spline basis (typically 3 for cubic)
     * @return expected<void, string> Success or error message
     */
    static expected<void, std::string> save(
        const std::string& filepath,
        const std::vector<double>& moneyness_knots,
        const std::vector<double>& maturity_knots,
        const std::vector<double>& volatility_knots,
        const std::vector<double>& rate_knots,
        const std::vector<double>& coefficients,
        double K_ref,
        const std::string& option_type,
        int spline_degree = 3
    );

    /**
     * Load interpolation table from disk using memory mapping
     *
     * @param filepath Path to the saved table
     * @return expected<unique_ptr<BSpline4D_FMA>, string> Loaded evaluator or error
     *
     * The returned evaluator uses memory-mapped data for zero-copy access.
     * The memory mapping is managed internally and persists as long as the
     * evaluator exists.
     */
    static expected<std::unique_ptr<BSpline4D_FMA>, std::string> load(
        const std::string& filepath
    );

    /**
     * Metadata extracted from a saved table
     */
    struct TableMetadata {
        double K_ref;
        std::string option_type;
        int spline_degree;
        size_t n_moneyness;
        size_t n_maturity;
        size_t n_volatility;
        size_t n_rate;
        size_t n_coefficients;
        uint64_t file_size_bytes;
    };

    /**
     * Read metadata without loading the full table
     *
     * @param filepath Path to the saved table
     * @return expected<TableMetadata, string> Metadata or error
     */
    static expected<TableMetadata, std::string> read_metadata(
        const std::string& filepath
    );

private:
    // Helper to validate header
    static expected<void, std::string> validate_header(const InterpolationTableHeader& header);
};

} // namespace mango

#endif // MANGO_INTERPOLATION_TABLE_STORAGE_V2_HPP
