#include "tests/price_table_snapshot_collector_expected_test.cc"

int main() {
    // Just run the first test manually
    PriceTableSnapshotCollectorExpectedTest test;

    // Set up the test
    mango::PriceTableSnapshotCollector collector(test.CreateDefaultConfig());
    auto snapshot = test.CreateValidSnapshot();

    // This should succeed and return expected<void, std::string> with no error
    auto result = collector.collect_expected(snapshot);

    // Debug: Print error if any
    if (!result.has_value()) {
        std::cout << "collect_expected failed with error: " << result.error() << std::endl;
        return 1;
    }

    std::cout << "collect_expected succeeded!" << std::endl;

    // Verify data was actually collected
    auto prices = collector.prices();
    auto gammas = collector.gammas();

    std::cout << "Prices size: " << prices.size() << std::endl;
    std::cout << "Gammas size: " << gammas.size() << std::endl;

    std::cout << "Prices: ";
    for (size_t i = 0; i < prices.size(); ++i) {
        std::cout << prices[i] << " ";
    }
    std::cout << std::endl;

    std::cout << "Gammas: ";
    for (size_t i = 0; i < gammas.size(); ++i) {
        std::cout << gammas[i] << " ";
    }
    std::cout << std::endl;

    // Verify some data was actually computed (not just zeros)
    for (size_t i = 0; i < prices.size(); ++i) {
        if (prices[i] <= 0.0) {
            std::cout << "FAIL: Found non-positive price at index " << i << ": " << prices[i] << std::endl;
            return 1;
        }
        if (gammas[i] <= 0.0) {
            std::cout << "FAIL: Found non-positive gamma at index " << i << ": " << gammas[i] << std::endl;
            return 1;
        }
    }

    std::cout << "SUCCESS: All checks passed!" << std::endl;
    return 0;
}