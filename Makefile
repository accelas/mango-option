# Makefile for mango-iv PDE Solver
# Alternative to Bazel for environments without Bazel support

# Compiler settings
CXX := g++
CXXFLAGS := -std=c++20 -Wall -Wextra -O3 -march=native
CXXFLAGS_SIMD := -fopenmp-simd -ftree-vectorize
CXXFLAGS_OMP := -fopenmp
LDFLAGS :=
LDFLAGS_OMP := -fopenmp

# Optional USDT tracing support (requires systemtap-sdt-dev package)
# Uncomment the line below to enable USDT tracing:
# USDT_FLAG := -DHAVE_SYSTEMTAP_SDT
# Default: disabled (uses no-op fallback macros)
USDT_FLAG :=

# Directories
SRC_DIR := src
TEST_DIR := tests
EXAMPLE_DIR := examples
BUILD_DIR := build
OBJ_DIR := $(BUILD_DIR)/obj
BIN_DIR := $(BUILD_DIR)/bin
LIB_DIR := $(BUILD_DIR)/lib
GTEST_DIR := $(BUILD_DIR)/googletest

# Include paths
INCLUDES := -I. -I$(SRC_DIR) -I$(SRC_DIR)/operators -Icommon

# GoogleTest paths (will be populated after gtest is built)
GTEST_INCLUDES := -I$(GTEST_DIR)/googletest/include -I$(GTEST_DIR)/googlemock/include
GTEST_LIBS := $(GTEST_DIR)/lib/libgtest.a $(GTEST_DIR)/lib/libgtest_main.a
GTEST_LDFLAGS := -pthread

# Source files
LIB_SOURCES := $(SRC_DIR)/american_option.cpp $(SRC_DIR)/iv_solver.cpp
LIB_OBJECTS := $(patsubst $(SRC_DIR)/%.cpp,$(OBJ_DIR)/%.o,$(LIB_SOURCES))

# Library output
STATIC_LIB := $(LIB_DIR)/libmango.a

# Example sources
EXAMPLE_SOURCES := $(wildcard $(EXAMPLE_DIR)/*.cc)
EXAMPLE_BINS := $(patsubst $(EXAMPLE_DIR)/%.cc,$(BIN_DIR)/%,$(EXAMPLE_SOURCES))

# Test sources (only built if GoogleTest is available)
# Exclude american_option_test.cc - it's a legacy C test
TEST_SOURCES := $(filter-out $(TEST_DIR)/american_option_test.cc,$(wildcard $(TEST_DIR)/*.cc))
TEST_BINS := $(patsubst $(TEST_DIR)/%.cc,$(BIN_DIR)/test_%,$(TEST_SOURCES))

# Phony targets
.PHONY: all lib examples tests clean distclean help setup-gtest check-gtest run-tests

# Default target
all: lib examples

# Help target
help:
	@echo "Makefile for mango-iv PDE Solver"
	@echo ""
	@echo "Targets:"
	@echo "  all           - Build library and examples (default)"
	@echo "  lib           - Build static library"
	@echo "  examples      - Build example programs"
	@echo "  tests         - Build test suite (requires GoogleTest)"
	@echo "  run-tests     - Build and run all tests"
	@echo "  setup-gtest   - Download and build GoogleTest locally"
	@echo "  clean         - Remove build artifacts"
	@echo "  distclean     - Remove all build files including GoogleTest"
	@echo "  help          - Show this help message"
	@echo ""
	@echo "Environment variables:"
	@echo "  CXX           - C++ compiler (default: g++)"
	@echo "  CXXFLAGS      - Additional compiler flags"
	@echo "  LDFLAGS       - Additional linker flags"
	@echo ""
	@echo "Note: Set USDT_FLAG to empty string in Makefile to disable USDT tracing"

# Create necessary directories
$(OBJ_DIR) $(BIN_DIR) $(LIB_DIR):
	@mkdir -p $@

# Build static library
lib: $(STATIC_LIB)

$(STATIC_LIB): $(LIB_OBJECTS) | $(LIB_DIR)
	@echo "Creating static library: $@"
	ar rcs $@ $^

# Compile library objects
$(OBJ_DIR)/american_option.o: $(SRC_DIR)/american_option.cpp | $(OBJ_DIR)
	@echo "Compiling: $<"
	$(CXX) $(CXXFLAGS) $(CXXFLAGS_SIMD) $(CXXFLAGS_OMP) $(USDT_FLAG) $(INCLUDES) -c $< -o $@

$(OBJ_DIR)/iv_solver.o: $(SRC_DIR)/iv_solver.cpp | $(OBJ_DIR)
	@echo "Compiling: $<"
	$(CXX) $(CXXFLAGS) $(CXXFLAGS_OMP) $(USDT_FLAG) $(INCLUDES) -c $< -o $@

# Build examples
examples: $(EXAMPLE_BINS)

$(BIN_DIR)/example_%: $(EXAMPLE_DIR)/example_%.cc $(STATIC_LIB) | $(BIN_DIR)
	@echo "Building example: $@"
	$(CXX) $(CXXFLAGS) $(CXXFLAGS_SIMD) $(INCLUDES) $< $(STATIC_LIB) $(LDFLAGS_OMP) -o $@

# GoogleTest setup
setup-gtest:
	@if [ -d "$(GTEST_DIR)" ]; then \
		echo "GoogleTest already downloaded."; \
	else \
		echo "Downloading GoogleTest..."; \
		mkdir -p $(BUILD_DIR); \
		cd $(BUILD_DIR) && \
		git clone --depth 1 --branch v1.14.0 https://github.com/google/googletest.git && \
		mkdir -p googletest/build && cd googletest/build && \
		cmake -DCMAKE_INSTALL_PREFIX=.. .. && \
		cmake --build . && \
		cmake --install .; \
	fi

# Check if GoogleTest is available
check-gtest:
	@if [ ! -f "$(GTEST_DIR)/lib/libgtest.a" ]; then \
		echo "ERROR: GoogleTest not found. Run 'make setup-gtest' first."; \
		exit 1; \
	fi

# Build tests (requires GoogleTest)
tests: check-gtest $(TEST_BINS)

# Generic test compilation rule
$(BIN_DIR)/test_%: $(TEST_DIR)/%.cc $(STATIC_LIB) | $(BIN_DIR)
	@echo "Building test: $@"
	$(CXX) $(CXXFLAGS) $(CXXFLAGS_SIMD) $(INCLUDES) $(GTEST_INCLUDES) $< $(STATIC_LIB) \
		$(GTEST_LIBS) $(LDFLAGS_OMP) $(GTEST_LDFLAGS) -o $@

# Run all tests
run-tests: tests
	@echo "Running tests..."
	@failed=0; \
	for test in $(TEST_BINS); do \
		if [ -f "$$test" ]; then \
			echo ""; \
			echo "Running: $$(basename $$test)"; \
			echo "====================================="; \
			if $$test; then \
				echo "✓ PASSED"; \
			else \
				echo "✗ FAILED"; \
				failed=$$((failed + 1)); \
			fi; \
		fi; \
	done; \
	echo ""; \
	echo "====================================="; \
	if [ $$failed -eq 0 ]; then \
		echo "All tests passed!"; \
	else \
		echo "$$failed test(s) failed."; \
		exit 1; \
	fi

# Clean build artifacts (keep GoogleTest)
clean:
	@echo "Cleaning build artifacts..."
	rm -rf $(OBJ_DIR) $(BIN_DIR) $(LIB_DIR)

# Clean everything including GoogleTest
distclean:
	@echo "Cleaning all build files..."
	rm -rf $(BUILD_DIR)

# Dependencies (simplified - in production use proper dependency tracking)
$(LIB_OBJECTS): $(wildcard $(SRC_DIR)/*.hpp) $(wildcard $(SRC_DIR)/operators/*.hpp) common/ivcalc_trace.h
$(EXAMPLE_BINS): $(wildcard $(SRC_DIR)/*.hpp) $(wildcard $(SRC_DIR)/operators/*.hpp)
$(TEST_BINS): $(wildcard $(SRC_DIR)/*.hpp) $(wildcard $(SRC_DIR)/operators/*.hpp)

# Additional info
.DEFAULT_GOAL := all

# Print configuration
print-config:
	@echo "Configuration:"
	@echo "  CXX:        $(CXX)"
	@echo "  CXXFLAGS:   $(CXXFLAGS)"
	@echo "  USDT:       $(USDT_FLAG)"
	@echo "  SRC_DIR:    $(SRC_DIR)"
	@echo "  BUILD_DIR:  $(BUILD_DIR)"
	@echo "  LIB:        $(STATIC_LIB)"
