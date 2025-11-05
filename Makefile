# Makefile for mango-iv PDE solver
# Temporary workaround for environments without Bazel support

# Compiler configuration
CC := gcc
CXX := g++
AR := ar

# Compiler flags
# Note: Use -std=c2x for GCC < 14 (c2x is the C23 draft standard)
CFLAGS := -std=c2x -Wall -Wextra -O3 -march=native -fopenmp-simd -ftree-vectorize -I.
CFLAGS_OPENMP := -std=c2x -Wall -Wextra -O3 -march=native -fopenmp -ftree-vectorize -I.
CXXFLAGS := -std=c++20 -Wall -Wextra -O3 -I.
CXXFLAGS_17 := -std=c++17 -Wall -Wextra -O3 -I.

# USDT support (optional - gracefully falls back if not available)
# Uncomment the following lines if systemtap-sdt-dev is installed:
# CFLAGS += -DHAVE_SYSTEMTAP_SDT
# CFLAGS_OPENMP += -DHAVE_SYSTEMTAP_SDT

# Linker flags
LDFLAGS := -lm
LDFLAGS_OPENMP := -lm -fopenmp

# GoogleTest flags (adjust if installed in non-standard location)
# Check if GoogleTest is available
GTEST_AVAILABLE := $(shell pkg-config --exists gtest && echo yes || echo no)
GTEST_CXXFLAGS := $(shell pkg-config --cflags gtest 2>/dev/null || echo "-I/usr/include")
GTEST_LDFLAGS := $(shell pkg-config --libs gtest gtest_main 2>/dev/null || echo "-lgtest -lgtest_main -lpthread")

# Directories
SRC_DIR := src
CPP_DIR := src/cpp
EXAMPLES_DIR := examples
TESTS_DIR := tests
BUILD_DIR := build
OBJ_DIR := $(BUILD_DIR)/obj
BIN_DIR := $(BUILD_DIR)/bin
LIB_DIR := $(BUILD_DIR)/lib

# Create directories
$(shell mkdir -p $(OBJ_DIR) $(BIN_DIR) $(LIB_DIR))

# ============================================================================
# C Library Source Files
# ============================================================================

# Core library objects
CUBIC_SPLINE_OBJS := $(OBJ_DIR)/cubic_spline.o
GRID_GEN_OBJS := $(OBJ_DIR)/grid_generation.o
GRID_PRESETS_OBJS := $(OBJ_DIR)/grid_presets.o $(GRID_GEN_OBJS)
GRID_TRANSFORM_OBJS := $(OBJ_DIR)/grid_transform.o
PDE_SOLVER_OBJS := $(OBJ_DIR)/pde_solver.o $(CUBIC_SPLINE_OBJS)
AMERICAN_OPTION_OBJS := $(OBJ_DIR)/american_option.o $(PDE_SOLVER_OBJS)
LETS_BE_RATIONAL_OBJS := $(OBJ_DIR)/lets_be_rational.o
INTERP_CUBIC_OBJS := $(OBJ_DIR)/interp_cubic.o $(OBJ_DIR)/interp_cubic_workspace.o $(CUBIC_SPLINE_OBJS)
IV_SURFACE_OBJS := $(OBJ_DIR)/iv_surface.o $(INTERP_CUBIC_OBJS)
PRICE_TABLE_OBJS := $(OBJ_DIR)/price_table.o $(INTERP_CUBIC_OBJS) $(AMERICAN_OPTION_OBJS)
IMPLIED_VOL_OBJS := $(OBJ_DIR)/implied_volatility.o $(AMERICAN_OPTION_OBJS) $(LETS_BE_RATIONAL_OBJS) $(PRICE_TABLE_OBJS)
VALIDATION_OBJS := $(OBJ_DIR)/validation.o $(AMERICAN_OPTION_OBJS) $(IMPLIED_VOL_OBJS) $(PRICE_TABLE_OBJS)

# All library objects
ALL_LIB_OBJS := $(OBJ_DIR)/cubic_spline.o \
                $(OBJ_DIR)/grid_generation.o \
                $(OBJ_DIR)/grid_presets.o \
                $(OBJ_DIR)/grid_transform.o \
                $(OBJ_DIR)/pde_solver.o \
                $(OBJ_DIR)/american_option.o \
                $(OBJ_DIR)/lets_be_rational.o \
                $(OBJ_DIR)/interp_cubic.o \
                $(OBJ_DIR)/interp_cubic_workspace.o \
                $(OBJ_DIR)/iv_surface.o \
                $(OBJ_DIR)/price_table.o \
                $(OBJ_DIR)/implied_volatility.o \
                $(OBJ_DIR)/validation.o

# Static library
LIBMANGO := $(LIB_DIR)/libmango.a

# ============================================================================
# Example Programs
# ============================================================================

EXAMPLES := \
    $(BIN_DIR)/example_heat_equation \
    $(BIN_DIR)/example_american_option \
    $(BIN_DIR)/example_american_option_dividend \
    $(BIN_DIR)/example_implied_volatility \
    $(BIN_DIR)/example_interpolation_engine \
    $(BIN_DIR)/test_cubic_4d_5d \
    $(BIN_DIR)/example_precompute_table \
    $(BIN_DIR)/example_newton_solver \
    $(BIN_DIR)/test_iv_accuracy

# ============================================================================
# Test Programs
# ============================================================================

TESTS := \
    $(BIN_DIR)/grid_test \
    $(BIN_DIR)/multigrid_test \
    $(BIN_DIR)/boundary_conditions_test \
    $(BIN_DIR)/cache_config_test \
    $(BIN_DIR)/workspace_test \
    $(BIN_DIR)/root_finding_test \
    $(BIN_DIR)/newton_workspace_test \
    $(BIN_DIR)/newton_solver_test \
    $(BIN_DIR)/spatial_operators_test \
    $(BIN_DIR)/pde_solver_test \
    $(BIN_DIR)/cubic_spline_test \
    $(BIN_DIR)/grid_generation_test \
    $(BIN_DIR)/grid_presets_test \
    $(BIN_DIR)/stability_test \
    $(BIN_DIR)/brent_test \
    $(BIN_DIR)/lets_be_rational_test \
    $(BIN_DIR)/implied_volatility_test \
    $(BIN_DIR)/tridiagonal_test \
    $(BIN_DIR)/american_option_test \
    $(BIN_DIR)/unified_grid_test \
    $(BIN_DIR)/adaptive_accuracy_test \
    $(BIN_DIR)/interpolation_test \
    $(BIN_DIR)/cubic_interp_4d_5d_test \
    $(BIN_DIR)/interpolation_workspace_test \
    $(BIN_DIR)/price_table_test \
    $(BIN_DIR)/price_table_slow_test \
    $(BIN_DIR)/coordinate_transform_test \
    $(BIN_DIR)/memory_layout_test \
    $(BIN_DIR)/diagnostic_interp_test \
    $(BIN_DIR)/integration_5d_price_table_test \
    $(BIN_DIR)/time_domain_test \
    $(BIN_DIR)/trbdf2_config_test \
    $(BIN_DIR)/tridiagonal_solver_test

# Fast tests (excluding manual/slow tagged tests)
FAST_TESTS := \
    $(BIN_DIR)/grid_test \
    $(BIN_DIR)/multigrid_test \
    $(BIN_DIR)/boundary_conditions_test \
    $(BIN_DIR)/cache_config_test \
    $(BIN_DIR)/workspace_test \
    $(BIN_DIR)/root_finding_test \
    $(BIN_DIR)/newton_workspace_test \
    $(BIN_DIR)/newton_solver_test \
    $(BIN_DIR)/spatial_operators_test \
    $(BIN_DIR)/pde_solver_test \
    $(BIN_DIR)/cubic_spline_test \
    $(BIN_DIR)/grid_generation_test \
    $(BIN_DIR)/grid_presets_test \
    $(BIN_DIR)/stability_test \
    $(BIN_DIR)/brent_test \
    $(BIN_DIR)/lets_be_rational_test \
    $(BIN_DIR)/tridiagonal_test \
    $(BIN_DIR)/american_option_test \
    $(BIN_DIR)/unified_grid_test \
    $(BIN_DIR)/interpolation_test \
    $(BIN_DIR)/cubic_interp_4d_5d_test \
    $(BIN_DIR)/interpolation_workspace_test \
    $(BIN_DIR)/price_table_test \
    $(BIN_DIR)/coordinate_transform_test \
    $(BIN_DIR)/memory_layout_test \
    $(BIN_DIR)/diagnostic_interp_test \
    $(BIN_DIR)/integration_5d_price_table_test \
    $(BIN_DIR)/time_domain_test \
    $(BIN_DIR)/trbdf2_config_test \
    $(BIN_DIR)/tridiagonal_solver_test

# ============================================================================
# Phony Targets
# ============================================================================

.PHONY: all clean lib examples tests test fast-test help check-gtest

all: lib examples

lib: $(LIBMANGO)

examples: $(EXAMPLES)

tests: $(TESTS)

# Run fast tests (excludes slow tests like implied_volatility_test, price_table_slow_test, adaptive_accuracy_test)
fast-test: check-gtest $(FAST_TESTS)
	@echo "Running fast tests..."
	@failed=0; \
	for test in $(FAST_TESTS); do \
		echo "Running $$test..."; \
		if $$test; then \
			echo "  ✓ PASSED"; \
		else \
			echo "  ✗ FAILED"; \
			failed=$$((failed + 1)); \
		fi; \
	done; \
	if [ $$failed -eq 0 ]; then \
		echo "All fast tests passed!"; \
	else \
		echo "$$failed test(s) failed."; \
		exit 1; \
	fi

# Run all tests (including slow ones)
test: check-gtest $(TESTS)
	@echo "Running all tests..."
	@failed=0; \
	for test in $(TESTS); do \
		echo "Running $$test..."; \
		if $$test; then \
			echo "  ✓ PASSED"; \
		else \
			echo "  ✗ FAILED"; \
			failed=$$((failed + 1)); \
		fi; \
	done; \
	if [ $$failed -eq 0 ]; then \
		echo "All tests passed!"; \
	else \
		echo "$$failed test(s) failed."; \
		exit 1; \
	fi

# Check for GoogleTest availability
.PHONY: check-gtest
check-gtest:
	@if [ "$(GTEST_AVAILABLE)" != "yes" ]; then \
		echo ""; \
		echo "ERROR: GoogleTest is not installed!"; \
		echo ""; \
		echo "To build and run tests, install GoogleTest:"; \
		echo "  Ubuntu/Debian: sudo apt-get install libgtest-dev googletest"; \
		echo "  Fedora/RHEL:   sudo dnf install gtest gtest-devel"; \
		echo ""; \
		echo "Note: You can still build the library and examples without GoogleTest."; \
		echo "      Use 'make lib' or 'make examples' instead."; \
		echo ""; \
		exit 1; \
	fi

clean:
	rm -rf $(BUILD_DIR)

help:
	@echo "Makefile for mango-iv PDE solver"
	@echo ""
	@echo "Targets:"
	@echo "  all         - Build library and examples (tests require GoogleTest)"
	@echo "  lib         - Build static library only"
	@echo "  examples    - Build example programs"
	@echo "  tests       - Build test programs (requires GoogleTest)"
	@echo "  fast-test   - Build and run fast tests (requires GoogleTest)"
	@echo "  test        - Build and run all tests (requires GoogleTest)"
	@echo "  clean       - Remove all build artifacts"
	@echo "  help        - Show this help message"
	@echo ""
	@echo "Output directories:"
	@echo "  Examples:   $(BIN_DIR)/"
	@echo "  Library:    $(LIB_DIR)/"
	@echo ""
	@echo "Requirements:"
	@echo "  - GCC/G++ (C2x/C++20 support)"
	@echo "  - GoogleTest (for tests only): sudo apt-get install libgtest-dev googletest"
	@echo "  - systemtap-sdt-dev (optional, for USDT tracing)"

# ============================================================================
# Library Build Rules
# ============================================================================

$(LIBMANGO): $(ALL_LIB_OBJS)
	@echo "Creating static library $@"
	$(AR) rcs $@ $^

# C source files (most use standard CFLAGS)
$(OBJ_DIR)/cubic_spline.o: $(SRC_DIR)/cubic_spline.c $(SRC_DIR)/cubic_spline.h $(SRC_DIR)/tridiagonal.h $(SRC_DIR)/ivcalc_trace.h
	$(CC) $(CFLAGS) -c $< -o $@

$(OBJ_DIR)/grid_generation.o: $(SRC_DIR)/grid_generation.c $(SRC_DIR)/grid_generation.h
	$(CC) $(CFLAGS) -c $< -o $@

$(OBJ_DIR)/grid_presets.o: $(SRC_DIR)/grid_presets.c $(SRC_DIR)/grid_presets.h
	$(CC) $(CFLAGS) -c $< -o $@

$(OBJ_DIR)/grid_transform.o: $(SRC_DIR)/grid_transform.c $(SRC_DIR)/grid_transform.h
	$(CC) $(CFLAGS) -c $< -o $@

$(OBJ_DIR)/pde_solver.o: $(SRC_DIR)/pde_solver.c $(SRC_DIR)/pde_solver.h $(SRC_DIR)/ivcalc_trace.h
	$(CC) $(CFLAGS) -c $< -o $@

# american_option.c requires full OpenMP (not just simd)
$(OBJ_DIR)/american_option.o: $(SRC_DIR)/american_option.c $(SRC_DIR)/american_option.h $(SRC_DIR)/ivcalc_trace.h
	$(CC) $(CFLAGS_OPENMP) -c $< -o $@

$(OBJ_DIR)/lets_be_rational.o: $(SRC_DIR)/lets_be_rational.c $(SRC_DIR)/lets_be_rational.h
	$(CC) $(CFLAGS) -c $< -o $@

$(OBJ_DIR)/implied_volatility.o: $(SRC_DIR)/implied_volatility.c $(SRC_DIR)/implied_volatility.h
	$(CC) $(CFLAGS) -c $< -o $@

$(OBJ_DIR)/validation.o: $(SRC_DIR)/validation.c $(SRC_DIR)/validation.h
	$(CC) $(CFLAGS) -c $< -o $@

$(OBJ_DIR)/interp_cubic.o: $(SRC_DIR)/interp_cubic.c $(SRC_DIR)/interp_cubic.h
	$(CC) $(CFLAGS) -c $< -o $@

$(OBJ_DIR)/interp_cubic_workspace.o: $(SRC_DIR)/interp_cubic_workspace.c $(SRC_DIR)/interp_cubic.h
	$(CC) $(CFLAGS) -c $< -o $@

$(OBJ_DIR)/iv_surface.o: $(SRC_DIR)/iv_surface.c $(SRC_DIR)/iv_surface.h
	$(CC) $(CFLAGS) -c $< -o $@

$(OBJ_DIR)/price_table.o: $(SRC_DIR)/price_table.c $(SRC_DIR)/price_table.h
	$(CC) $(CFLAGS) -c $< -o $@

# ============================================================================
# Example Programs
# ============================================================================

$(BIN_DIR)/example_heat_equation: $(EXAMPLES_DIR)/example_heat_equation.c $(LIBMANGO)
	$(CC) $(CFLAGS) $< -o $@ -L$(LIB_DIR) -lmango $(LDFLAGS)

$(BIN_DIR)/example_american_option: $(EXAMPLES_DIR)/example_american_option.c $(LIBMANGO)
	$(CC) $(CFLAGS) $< -o $@ -L$(LIB_DIR) -lmango $(LDFLAGS_OPENMP)

$(BIN_DIR)/example_american_option_dividend: $(EXAMPLES_DIR)/example_american_option_dividend.c $(LIBMANGO)
	$(CC) $(CFLAGS) $< -o $@ -L$(LIB_DIR) -lmango $(LDFLAGS_OPENMP)

$(BIN_DIR)/example_implied_volatility: $(EXAMPLES_DIR)/example_implied_volatility.c $(LIBMANGO)
	$(CC) $(CFLAGS) $< -o $@ -L$(LIB_DIR) -lmango $(LDFLAGS_OPENMP)

$(BIN_DIR)/example_interpolation_engine: $(EXAMPLES_DIR)/example_interpolation_engine.c $(LIBMANGO)
	$(CC) $(CFLAGS) $< -o $@ -L$(LIB_DIR) -lmango $(LDFLAGS_OPENMP)

$(BIN_DIR)/test_cubic_4d_5d: $(EXAMPLES_DIR)/test_cubic_4d_5d.c $(LIBMANGO)
	$(CC) $(CFLAGS) $< -o $@ -L$(LIB_DIR) -lmango $(LDFLAGS_OPENMP)

$(BIN_DIR)/example_precompute_table: $(EXAMPLES_DIR)/example_precompute_table.c $(LIBMANGO)
	$(CC) $(CFLAGS) $< -o $@ -L$(LIB_DIR) -lmango $(LDFLAGS_OPENMP)

$(BIN_DIR)/example_newton_solver: $(EXAMPLES_DIR)/example_newton_solver.cc
	$(CXX) $(CXXFLAGS) -I$(CPP_DIR) -I$(SRC_DIR) $< -o $@ $(LDFLAGS)

$(BIN_DIR)/test_iv_accuracy: $(EXAMPLES_DIR)/test_iv_accuracy.cc $(LIBMANGO)
	$(CXX) $(CXXFLAGS_17) -I$(SRC_DIR) $< -o $@ -L$(LIB_DIR) -lmango $(LDFLAGS_OPENMP)

# ============================================================================
# Test Programs (C++20)
# ============================================================================

# Header-only C++ tests
$(BIN_DIR)/grid_test: $(TESTS_DIR)/grid_test.cc
	$(CXX) $(CXXFLAGS) -I$(CPP_DIR) $< -o $@ $(GTEST_LDFLAGS)

$(BIN_DIR)/multigrid_test: $(TESTS_DIR)/multigrid_test.cc
	$(CXX) $(CXXFLAGS) -I$(CPP_DIR) $< -o $@ $(GTEST_LDFLAGS)

$(BIN_DIR)/boundary_conditions_test: $(TESTS_DIR)/boundary_conditions_test.cc
	$(CXX) $(CXXFLAGS) -I$(CPP_DIR) $< -o $@ $(GTEST_LDFLAGS)

$(BIN_DIR)/cache_config_test: $(TESTS_DIR)/cache_config_test.cc
	$(CXX) $(CXXFLAGS) -I$(CPP_DIR) $< -o $@ $(GTEST_LDFLAGS)

$(BIN_DIR)/workspace_test: $(TESTS_DIR)/workspace_test.cc
	$(CXX) $(CXXFLAGS) -I$(CPP_DIR) $< -o $@ $(GTEST_LDFLAGS)

$(BIN_DIR)/root_finding_test: $(TESTS_DIR)/root_finding_test.cc
	$(CXX) $(CXXFLAGS) -I$(CPP_DIR) $< -o $@ $(GTEST_LDFLAGS)

$(BIN_DIR)/newton_workspace_test: $(TESTS_DIR)/newton_workspace_test.cc
	$(CXX) $(CXXFLAGS) -I$(CPP_DIR) $< -o $@ $(GTEST_LDFLAGS)

$(BIN_DIR)/newton_solver_test: $(TESTS_DIR)/newton_solver_test.cc
	$(CXX) $(CXXFLAGS) -I$(CPP_DIR) $< -o $@ $(GTEST_LDFLAGS)

$(BIN_DIR)/spatial_operators_test: $(TESTS_DIR)/spatial_operators_test.cc
	$(CXX) $(CXXFLAGS) -I$(CPP_DIR) $< -o $@ $(GTEST_LDFLAGS)

$(BIN_DIR)/pde_solver_test: $(TESTS_DIR)/pde_solver_test.cc
	$(CXX) $(CXXFLAGS) -I$(CPP_DIR) $< -o $@ $(GTEST_LDFLAGS)

$(BIN_DIR)/time_domain_test: $(TESTS_DIR)/time_domain_test.cc
	$(CXX) $(CXXFLAGS) -I$(CPP_DIR) $< -o $@ $(GTEST_LDFLAGS)

$(BIN_DIR)/trbdf2_config_test: $(TESTS_DIR)/trbdf2_config_test.cc
	$(CXX) $(CXXFLAGS) -I$(CPP_DIR) $< -o $@ $(GTEST_LDFLAGS)

$(BIN_DIR)/tridiagonal_solver_test: $(TESTS_DIR)/tridiagonal_solver_test.cc
	$(CXX) $(CXXFLAGS) -I$(CPP_DIR) $< -o $@ $(GTEST_LDFLAGS)

$(BIN_DIR)/integration_5d_price_table_test: $(TESTS_DIR)/integration_5d_price_table_test.cc
	$(CXX) $(CXXFLAGS) -I$(CPP_DIR) $< -o $@ $(GTEST_LDFLAGS)

# C library tests (C++17)
$(BIN_DIR)/cubic_spline_test: $(TESTS_DIR)/cubic_spline_test.cc $(LIBMANGO)
	$(CXX) $(CXXFLAGS_17) $(GTEST_CXXFLAGS) -I$(SRC_DIR) $< -o $@ -L$(LIB_DIR) -lmango $(GTEST_LDFLAGS) $(LDFLAGS)

$(BIN_DIR)/grid_generation_test: $(TESTS_DIR)/grid_generation_test.cc $(LIBMANGO)
	$(CXX) $(CXXFLAGS_17) $(GTEST_CXXFLAGS) -I$(SRC_DIR) $< -o $@ -L$(LIB_DIR) -lmango $(GTEST_LDFLAGS) $(LDFLAGS)

$(BIN_DIR)/grid_presets_test: $(TESTS_DIR)/grid_presets_test.cc $(LIBMANGO)
	$(CXX) $(CXXFLAGS_17) $(GTEST_CXXFLAGS) -I$(SRC_DIR) $< -o $@ -L$(LIB_DIR) -lmango $(GTEST_LDFLAGS) $(LDFLAGS)

$(BIN_DIR)/stability_test: $(TESTS_DIR)/stability_test.cc $(LIBMANGO)
	$(CXX) $(CXXFLAGS_17) $(GTEST_CXXFLAGS) -I$(SRC_DIR) $< -o $@ -L$(LIB_DIR) -lmango $(GTEST_LDFLAGS) $(LDFLAGS)

$(BIN_DIR)/brent_test: $(TESTS_DIR)/brent_test.cc $(LIBMANGO)
	$(CXX) $(CXXFLAGS_17) $(GTEST_CXXFLAGS) -I$(SRC_DIR) $< -o $@ -L$(LIB_DIR) -lmango $(GTEST_LDFLAGS) $(LDFLAGS_OPENMP)

$(BIN_DIR)/lets_be_rational_test: $(TESTS_DIR)/lets_be_rational_test.cc $(LIBMANGO)
	$(CXX) $(CXXFLAGS_17) $(GTEST_CXXFLAGS) -I$(SRC_DIR) $< -o $@ -L$(LIB_DIR) -lmango $(GTEST_LDFLAGS) $(LDFLAGS)

$(BIN_DIR)/implied_volatility_test: $(TESTS_DIR)/implied_volatility_test.cc $(LIBMANGO)
	$(CXX) $(CXXFLAGS_17) $(GTEST_CXXFLAGS) -I$(SRC_DIR) $< -o $@ -L$(LIB_DIR) -lmango $(GTEST_LDFLAGS) $(LDFLAGS_OPENMP)

$(BIN_DIR)/tridiagonal_test: $(TESTS_DIR)/tridiagonal_test.cc $(LIBMANGO)
	$(CXX) $(CXXFLAGS_17) $(GTEST_CXXFLAGS) -I$(SRC_DIR) $< -o $@ -L$(LIB_DIR) -lmango $(GTEST_LDFLAGS) $(LDFLAGS)

$(BIN_DIR)/american_option_test: $(TESTS_DIR)/american_option_test.cc $(LIBMANGO)
	$(CXX) $(CXXFLAGS_17) $(GTEST_CXXFLAGS) -I$(SRC_DIR) $< -o $@ -L$(LIB_DIR) -lmango $(GTEST_LDFLAGS) $(LDFLAGS_OPENMP)

$(BIN_DIR)/unified_grid_test: $(TESTS_DIR)/unified_grid_test.cc $(LIBMANGO)
	$(CXX) $(CXXFLAGS_17) $(GTEST_CXXFLAGS) -I$(SRC_DIR) $< -o $@ -L$(LIB_DIR) -lmango $(GTEST_LDFLAGS) $(LDFLAGS_OPENMP)

$(BIN_DIR)/adaptive_accuracy_test: $(TESTS_DIR)/adaptive_accuracy_test.cc $(LIBMANGO)
	$(CXX) $(CXXFLAGS_17) $(GTEST_CXXFLAGS) -I$(SRC_DIR) $< -o $@ -L$(LIB_DIR) -lmango $(GTEST_LDFLAGS) $(LDFLAGS_OPENMP)

$(BIN_DIR)/interpolation_test: $(TESTS_DIR)/interpolation_test.cc $(LIBMANGO)
	$(CXX) $(CXXFLAGS_17) $(GTEST_CXXFLAGS) -I$(SRC_DIR) $< -o $@ -L$(LIB_DIR) -lmango $(GTEST_LDFLAGS) $(LDFLAGS_OPENMP)

$(BIN_DIR)/cubic_interp_4d_5d_test: $(TESTS_DIR)/cubic_interp_4d_5d_test.cc $(LIBMANGO)
	$(CXX) $(CXXFLAGS_17) $(GTEST_CXXFLAGS) -I$(SRC_DIR) $< -o $@ -L$(LIB_DIR) -lmango $(GTEST_LDFLAGS) $(LDFLAGS_OPENMP)

$(BIN_DIR)/interpolation_workspace_test: $(TESTS_DIR)/interpolation_workspace_test.cc $(LIBMANGO)
	$(CXX) $(CXXFLAGS_17) $(GTEST_CXXFLAGS) -I$(SRC_DIR) $< -o $@ -L$(LIB_DIR) -lmango $(GTEST_LDFLAGS) $(LDFLAGS_OPENMP)

$(BIN_DIR)/price_table_test: $(TESTS_DIR)/price_table_test.cc $(LIBMANGO)
	$(CXX) $(CXXFLAGS_17) $(GTEST_CXXFLAGS) -I$(SRC_DIR) $< -o $@ -L$(LIB_DIR) -lmango $(GTEST_LDFLAGS) $(LDFLAGS_OPENMP)

$(BIN_DIR)/price_table_slow_test: $(TESTS_DIR)/price_table_slow_test.cc $(LIBMANGO)
	$(CXX) $(CXXFLAGS_17) $(GTEST_CXXFLAGS) -I$(SRC_DIR) $< -o $@ -L$(LIB_DIR) -lmango $(GTEST_LDFLAGS) $(LDFLAGS_OPENMP)

$(BIN_DIR)/coordinate_transform_test: $(TESTS_DIR)/coordinate_transform_test.cc $(LIBMANGO)
	$(CXX) $(CXXFLAGS_17) $(GTEST_CXXFLAGS) -I$(SRC_DIR) $< -o $@ -L$(LIB_DIR) -lmango $(GTEST_LDFLAGS) $(LDFLAGS_OPENMP)

$(BIN_DIR)/memory_layout_test: $(TESTS_DIR)/memory_layout_test.cc $(LIBMANGO)
	$(CXX) $(CXXFLAGS_17) $(GTEST_CXXFLAGS) -I$(SRC_DIR) $< -o $@ -L$(LIB_DIR) -lmango $(GTEST_LDFLAGS) $(LDFLAGS_OPENMP)

$(BIN_DIR)/diagnostic_interp_test: $(TESTS_DIR)/diagnostic_interp_test.cc $(LIBMANGO)
	$(CXX) $(CXXFLAGS_17) $(GTEST_CXXFLAGS) -I$(SRC_DIR) $< -o $@ -L$(LIB_DIR) -lmango $(GTEST_LDFLAGS) $(LDFLAGS_OPENMP)
