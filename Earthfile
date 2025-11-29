VERSION 0.8

# =============================================================================
# Earthfile for mango-option fuzz testing
#
# This provides an isolated Clang + libc++ environment for running FuzzTest,
# which is incompatible with the main GCC 14 + libstdc++ build.
#
# Usage:
#   earthly +fuzz-test              # Run fuzz tests
#   earthly +fuzz-test-interactive  # Interactive shell for debugging
#
# Requirements:
#   - Install earthly: https://earthly.dev/get-earthly
# =============================================================================

# Base image with Clang and libc++
fuzz-base:
    FROM ubuntu:24.04

    # Install Clang, libc++, and build tools
    RUN apt-get update && apt-get install -y \
        clang-19 \
        libc++-19-dev \
        libc++abi-19-dev \
        lld-19 \
        curl \
        git \
        python3 \
        && rm -rf /var/lib/apt/lists/*

    # Install Bazelisk (manages Bazel versions)
    RUN curl -fsSL https://github.com/bazelbuild/bazelisk/releases/download/v1.19.0/bazelisk-linux-amd64 \
        -o /usr/local/bin/bazel && chmod +x /usr/local/bin/bazel

    # Set up Clang + libc++ environment
    ENV CC=clang-19
    ENV CXX=clang++-19

    WORKDIR /workspace

# Build target for fuzz tests
fuzz-build:
    FROM +fuzz-base

    # Copy source files
    COPY . /workspace

    # Create a .bazelrc.user for libc++ configuration
    RUN echo 'build --config=fuzz' >> .bazelrc.user

    # Build fuzz tests with Clang + libc++
    RUN bazel build --config=fuzz //tests:batch_solver_fuzz_test

# Run fuzz tests
fuzz-test:
    FROM +fuzz-build

    # Run fuzz tests (unit test mode by default)
    RUN bazel test --config=fuzz //tests:batch_solver_fuzz_test --test_output=all

# Interactive shell for debugging
fuzz-test-interactive:
    FROM +fuzz-build

    # Drop into shell for manual testing
    RUN echo "Run: bazel test --config=fuzz //tests:batch_solver_fuzz_test"
    ENTRYPOINT ["/bin/bash"]
