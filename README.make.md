# Makefile Build System for mango-iv

This is an alternative build system using standard Make, designed as a workaround for environments where Bazel is not available (e.g., Claude Code web interface).

## Quick Start

### Build the library and examples:
```bash
make
```

### Build specific targets:
```bash
make lib        # Build static library only
make examples   # Build example programs
```

### Run an example:
```bash
./build/bin/example_newton_solver
```

## Building Tests

Tests require GoogleTest. You have two options:

### Option 1: Automatic Setup (Recommended)
```bash
# Download and build GoogleTest locally (one-time setup)
make setup-gtest

# Build all tests
make tests

# Build and run all tests
make run-tests
```

### Option 2: System GoogleTest
If you have GoogleTest installed system-wide, you'll need to modify the Makefile to use system paths instead of the local build.

## Available Targets

| Target | Description |
|--------|-------------|
| `all` | Build library and examples (default) |
| `lib` | Build static library (`build/lib/libmango.a`) |
| `examples` | Build example programs |
| `tests` | Build test suite (requires GoogleTest) |
| `run-tests` | Build and run all tests |
| `setup-gtest` | Download and build GoogleTest locally |
| `clean` | Remove build artifacts (keeps GoogleTest) |
| `distclean` | Remove all build files including GoogleTest |
| `help` | Show help message |
| `print-config` | Display build configuration |

## Build Output

All build artifacts are placed in the `build/` directory:

```
build/
├── obj/              # Object files (.o)
├── lib/              # Static library (libmango.a)
├── bin/              # Executables (examples and tests)
└── googletest/       # GoogleTest installation (if using setup-gtest)
```

## Compiler Configuration

### Default Settings
- Compiler: `g++`
- Standard: C++20
- Optimization: `-O3 -march=native`
- SIMD: `-fopenmp-simd -ftree-vectorize`
- OpenMP: `-fopenmp` (for parallelization)

### Customization

You can override compiler settings using environment variables:

```bash
# Use clang++ instead of g++
make CXX=clang++

# Add extra flags
make CXXFLAGS="-std=c++20 -Wall -O2"

# Disable native optimization (for portability)
make CXXFLAGS="-std=c++20 -Wall -O3"
```

### USDT Tracing

USDT tracing is **disabled by default** (uses no-op fallback macros). To enable it:

1. Install systemtap-sdt-dev:
   ```bash
   sudo apt-get install systemtap-sdt-dev
   ```

2. Edit the Makefile and uncomment the USDT flag:
   ```makefile
   USDT_FLAG := -DHAVE_SYSTEMTAP_SDT
   ```

3. Rebuild:
   ```bash
   make clean
   make
   ```

## Project Structure

The Makefile builds the following components:

### Core Library
- `src/american_option.cpp` - American option pricing
- `src/iv_solver.cpp` - Implied volatility solver
- Header-only components (automatically included)

### Examples
- `examples/example_newton_solver.cc` - Newton-Raphson solver example

### Tests
- All `tests/*.cc` files are built as individual test executables
- Test binaries are prefixed with `test_` (e.g., `test_pde_solver_test`)

## Comparison with Bazel

| Feature | Bazel | Makefile |
|---------|-------|----------|
| Dependency management | Automatic (Bzlmod) | Manual |
| External deps | Integrated | Manual download |
| Incremental builds | Excellent | Good |
| Cross-platform | Excellent | Platform-specific |
| Setup complexity | High | Low |
| Build speed | Fast (cached) | Moderate |

The Makefile system is intended as a **temporary workaround** for environments without Bazel support. For production use, Bazel is recommended.

## Troubleshooting

### Missing Headers
If you encounter missing header errors, ensure all header files are present:
```bash
ls src/*.hpp
ls src/operators/*.hpp
ls common/*.h
```

### Compilation Errors
1. Check your compiler version:
   ```bash
   g++ --version  # Should support C++20
   ```

2. Verify C++20 support:
   ```bash
   make CXX=g++-11  # Use a specific version if needed
   ```

### Test Failures
If tests fail to build:
```bash
# Clean and rebuild GoogleTest
make distclean
make setup-gtest
make tests
```

### Performance Issues
If the build is too slow:
```bash
# Reduce optimization level
make clean
make CXXFLAGS="-std=c++20 -O2"

# Disable native optimization
make clean
make CXXFLAGS="-std=c++20 -O3"  # Remove -march=native
```

## Integration with Development Workflow

### Clean Rebuild
```bash
make clean && make -j4
```

### Specific Test
```bash
make tests
./build/bin/test_pde_solver_test
```

### Debugging Build
```bash
make clean
make CXXFLAGS="-std=c++20 -g -O0"
gdb ./build/bin/example_newton_solver
```

## Limitations

1. **No automatic dependency tracking** - If you modify headers, run `make clean` first
2. **No external dependency management** - Dependencies like GoogleTest must be manually managed
3. **Limited parallelism** - Use `make -jN` for parallel builds, but dependency ordering is manual
4. **Platform-specific** - Tested on Linux, may need adjustments for macOS/Windows

## Future Improvements

Potential enhancements to this Makefile system:

- [ ] Automatic header dependency generation (`.d` files)
- [ ] Support for system-installed GoogleTest
- [ ] Cross-platform compatibility (macOS, MSYS2)
- [ ] Benchmark targets
- [ ] Installation targets (`make install`)
- [ ] Package generation (`.deb`, `.rpm`)

## Migrating Back to Bazel

When Bazel becomes available, simply use the standard Bazel commands:

```bash
# Clean Makefile artifacts
make distclean

# Use Bazel
bazel build //...
bazel test //...
```

The Bazel build files are authoritative and always kept up-to-date.
