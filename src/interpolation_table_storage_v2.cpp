#include "interpolation_table_storage_v2.hpp"
#include "bspline_4d.hpp"
#include <cstring>
#include <fstream>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <memory>

namespace mango {

namespace {

constexpr uint32_t MAGIC = 0x4D494E54; // 'MINT'
constexpr uint32_t VERSION = 1;

// RAII wrapper for memory-mapped file
class MappedFile {
public:
    MappedFile(int fd, void* addr, size_t length)
        : fd_(fd), addr_(addr), length_(length) {}

    ~MappedFile() {
        if (addr_ != MAP_FAILED && addr_ != nullptr) {
            munmap(addr_, length_);
        }
        if (fd_ >= 0) {
            close(fd_);
        }
    }

    void* data() const { return addr_; }
    size_t size() const { return length_; }

    // Non-copyable
    MappedFile(const MappedFile&) = delete;
    MappedFile& operator=(const MappedFile&) = delete;

private:
    int fd_;
    void* addr_;
    size_t length_;
};

} // anonymous namespace

expected<void, std::string> InterpolationTableStorage::validate_header(
    const InterpolationTableHeader& header
) {
    if (header.magic != MAGIC) {
        return unexpected("Invalid magic number - not a valid interpolation table file");
    }
    if (header.version != VERSION) {
        return unexpected("Unsupported file version");
    }
    if (header.spline_degree < 1 || header.spline_degree > 10) {
        return unexpected("Invalid spline degree");
    }
    if (header.n_moneyness < 2 || header.n_maturity < 2 ||
        header.n_volatility < 2 || header.n_rate < 2) {
        return unexpected("Grid dimensions too small");
    }

    size_t expected_coeffs = header.n_moneyness * header.n_maturity *
                            header.n_volatility * header.n_rate;
    if (header.n_coefficients != expected_coeffs) {
        return unexpected("Coefficient count mismatch with grid dimensions");
    }

    return {};
}

expected<void, std::string> InterpolationTableStorage::save(
    const std::string& filepath,
    const std::vector<double>& moneyness_knots,
    const std::vector<double>& maturity_knots,
    const std::vector<double>& volatility_knots,
    const std::vector<double>& rate_knots,
    const std::vector<double>& coefficients,
    double K_ref,
    const std::string& option_type,
    int spline_degree
) {
    // Validate inputs
    if (moneyness_knots.empty() || maturity_knots.empty() ||
        volatility_knots.empty() || rate_knots.empty()) {
        return unexpected("Empty knot vectors not allowed");
    }

    size_t expected_coeffs = moneyness_knots.size() * maturity_knots.size() *
                            volatility_knots.size() * rate_knots.size();
    if (coefficients.size() != expected_coeffs) {
        return unexpected("Coefficient array size mismatch");
    }

    if (option_type != "PUT" && option_type != "CALL") {
        return unexpected("option_type must be 'PUT' or 'CALL'");
    }

    // Create header
    InterpolationTableHeader header{};
    header.magic = MAGIC;
    header.version = VERSION;
    header.K_ref = K_ref;
    header.option_type = (option_type == "PUT") ? 0 : 1;
    header.spline_degree = static_cast<uint32_t>(spline_degree);

    header.n_moneyness = moneyness_knots.size();
    header.n_maturity = maturity_knots.size();
    header.n_volatility = volatility_knots.size();
    header.n_rate = rate_knots.size();
    header.n_coefficients = coefficients.size();

    std::strncpy(header.option_type_str, option_type.c_str(), 15);
    header.option_type_str[15] = '\0';

    // Calculate offsets (all arrays are 64-byte aligned for cache efficiency)
    uint64_t offset = sizeof(InterpolationTableHeader);
    auto align_offset = [](uint64_t off) -> uint64_t {
        return (off + 63) & ~63ULL; // Round up to 64-byte boundary
    };

    header.moneyness_offset = align_offset(offset);
    offset = header.moneyness_offset + moneyness_knots.size() * sizeof(double);

    header.maturity_offset = align_offset(offset);
    offset = header.maturity_offset + maturity_knots.size() * sizeof(double);

    header.volatility_offset = align_offset(offset);
    offset = header.volatility_offset + volatility_knots.size() * sizeof(double);

    header.rate_offset = align_offset(offset);
    offset = header.rate_offset + rate_knots.size() * sizeof(double);

    header.coefficients_offset = align_offset(offset);

    // Open file for writing
    std::ofstream file(filepath, std::ios::binary | std::ios::trunc);
    if (!file) {
        return unexpected("Failed to open file for writing: " + filepath);
    }

    // Write header
    file.write(reinterpret_cast<const char*>(&header), sizeof(header));
    if (!file) {
        return unexpected("Failed to write header");
    }

    // Helper to write aligned array
    auto write_aligned_array = [&](uint64_t target_offset, const std::vector<double>& data) -> bool {
        uint64_t current_pos = file.tellp();
        if (current_pos < target_offset) {
            // Write padding zeros
            std::vector<char> padding(target_offset - current_pos, 0);
            file.write(padding.data(), padding.size());
        }
        file.write(reinterpret_cast<const char*>(data.data()), data.size() * sizeof(double));
        return file.good();
    };

    // Write all arrays
    if (!write_aligned_array(header.moneyness_offset, moneyness_knots)) {
        return unexpected("Failed to write moneyness knots");
    }
    if (!write_aligned_array(header.maturity_offset, maturity_knots)) {
        return unexpected("Failed to write maturity knots");
    }
    if (!write_aligned_array(header.volatility_offset, volatility_knots)) {
        return unexpected("Failed to write volatility knots");
    }
    if (!write_aligned_array(header.rate_offset, rate_knots)) {
        return unexpected("Failed to write rate knots");
    }
    if (!write_aligned_array(header.coefficients_offset, coefficients)) {
        return unexpected("Failed to write coefficients");
    }

    file.close();
    if (!file) {
        return unexpected("Failed to close file properly");
    }

    return {};
}

expected<std::unique_ptr<BSpline4D_FMA>, std::string> InterpolationTableStorage::load(
    const std::string& filepath
) {
    // Open file
    int fd = open(filepath.c_str(), O_RDONLY);
    if (fd < 0) {
        return unexpected("Failed to open file: " + filepath);
    }

    // Get file size
    struct stat st;
    if (fstat(fd, &st) < 0) {
        close(fd);
        return unexpected("Failed to stat file");
    }
    size_t file_size = st.st_size;

    if (file_size < sizeof(InterpolationTableHeader)) {
        close(fd);
        return unexpected("File too small to contain valid header");
    }

    // Memory map the file
    void* addr = mmap(nullptr, file_size, PROT_READ, MAP_PRIVATE, fd, 0);
    if (addr == MAP_FAILED) {
        close(fd);
        return unexpected("Failed to memory map file");
    }

    // Create RAII wrapper (will clean up on any error return)
    auto mapped_file = std::make_shared<MappedFile>(fd, addr, file_size);

    // Read and validate header
    const auto* header = static_cast<const InterpolationTableHeader*>(mapped_file->data());
    auto validation = validate_header(*header);
    if (!validation) {
        return unexpected(validation.error());
    }

    // Extract data pointers from mapped memory
    const auto* base = static_cast<const char*>(mapped_file->data());

    auto extract_array = [&](uint64_t offset, size_t count) -> std::vector<double> {
        const double* ptr = reinterpret_cast<const double*>(base + offset);
        return std::vector<double>(ptr, ptr + count);
    };

    auto moneyness_knots = extract_array(header->moneyness_offset, header->n_moneyness);
    auto maturity_knots = extract_array(header->maturity_offset, header->n_maturity);
    auto volatility_knots = extract_array(header->volatility_offset, header->n_volatility);
    auto rate_knots = extract_array(header->rate_offset, header->n_rate);
    auto coefficients = extract_array(header->coefficients_offset, header->n_coefficients);

    // Construct BSpline4D_FMA
    // Note: BSpline4D_FMA copies the data, so we don't need to keep the mapped file alive
    return std::make_unique<BSpline4D_FMA>(
        std::move(moneyness_knots),
        std::move(maturity_knots),
        std::move(volatility_knots),
        std::move(rate_knots),
        std::move(coefficients)
    );
}

expected<InterpolationTableStorage::TableMetadata, std::string>
InterpolationTableStorage::read_metadata(const std::string& filepath) {
    // Open file and read just the header
    std::ifstream file(filepath, std::ios::binary);
    if (!file) {
        return unexpected("Failed to open file: " + filepath);
    }

    InterpolationTableHeader header;
    file.read(reinterpret_cast<char*>(&header), sizeof(header));
    if (!file) {
        return unexpected("Failed to read header");
    }

    auto validation = validate_header(header);
    if (!validation) {
        return unexpected(validation.error());
    }

    // Get file size
    file.seekg(0, std::ios::end);
    uint64_t file_size = file.tellg();

    TableMetadata meta;
    meta.K_ref = header.K_ref;
    meta.option_type = std::string(header.option_type_str);
    meta.spline_degree = header.spline_degree;
    meta.n_moneyness = header.n_moneyness;
    meta.n_maturity = header.n_maturity;
    meta.n_volatility = header.n_volatility;
    meta.n_rate = header.n_rate;
    meta.n_coefficients = header.n_coefficients;
    meta.file_size_bytes = file_size;

    return meta;
}

} // namespace mango
