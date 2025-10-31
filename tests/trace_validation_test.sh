#!/usr/bin/env bash
#
# trace_validation_test.sh - Validate USDT probes are working correctly
#
# This script verifies that:
# 1. Binaries have USDT notes embedded
# 2. bpftrace can list the probes
# 3. Probes actually fire when the program runs
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

TESTS_PASSED=0
TESTS_FAILED=0

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[PASS]${NC} $1"
    ((TESTS_PASSED++))
}

log_error() {
    echo -e "${RED}[FAIL]${NC} $1"
    ((TESTS_FAILED++))
}

log_warning() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

echo "========================================"
echo "USDT Trace Validation Test"
echo "========================================"
echo

# Check if running as root
if [[ $EUID -ne 0 ]]; then
    log_warning "Not running as root - some tests will be skipped"
    log_warning "Run with sudo for full validation"
    SKIP_BPFTRACE=1
else
    SKIP_BPFTRACE=0
fi

# Build test binaries
log_info "Building test binaries..."
cd "$PROJECT_ROOT"

if ! bazel build //examples:example_heat_equation //examples:example_implied_volatility 2>&1 | tail -5; then
    log_error "Failed to build binaries"
    exit 1
fi

log_success "Binaries built successfully"

# Find built binaries
HEAT_EQ_BIN="$(bazel info bazel-bin)/examples/example_heat_equation"
IV_BIN="$(bazel info bazel-bin)/examples/example_implied_volatility"

if [[ ! -f "$HEAT_EQ_BIN" ]]; then
    log_error "Binary not found: $HEAT_EQ_BIN"
    exit 1
fi

# Test 1: Check for USDT notes in ELF
log_info "Test 1: Checking for USDT notes in binaries..."

if readelf -n "$HEAT_EQ_BIN" 2>/dev/null | grep -q NT_STAPSDT; then
    log_success "USDT notes found in example_heat_equation"
else
    log_error "No USDT notes in example_heat_equation"
    log_warning "Did you install systemtap-sdt-dev and rebuild?"
fi

if readelf -n "$IV_BIN" 2>/dev/null | grep -q NT_STAPSDT; then
    log_success "USDT notes found in example_implied_volatility"
else
    log_error "No USDT notes in example_implied_volatility"
fi

# Test 2: Check bpftrace can list probes
if [[ $SKIP_BPFTRACE -eq 0 ]]; then
    log_info "Test 2: Checking bpftrace can list probes..."

    if ! command -v bpftrace &> /dev/null; then
        log_warning "bpftrace not installed - skipping bpftrace tests"
        SKIP_BPFTRACE=1
    else
        # Check expected probes exist
        PROBE_COUNT=$(bpftrace -l "usdt:$HEAT_EQ_BIN:mango:*" 2>/dev/null | wc -l)

        if [[ $PROBE_COUNT -gt 0 ]]; then
            log_success "bpftrace can list probes (found $PROBE_COUNT probes)"
        else
            log_error "bpftrace cannot list probes"
        fi

        # Check for specific expected probes
        EXPECTED_PROBES=(
            "algo_start"
            "algo_complete"
            "convergence_iter"
            "convergence_success"
            "convergence_failed"
        )

        for probe in "${EXPECTED_PROBES[@]}"; do
            if bpftrace -l "usdt:$HEAT_EQ_BIN:mango:$probe" 2>/dev/null | grep -q "$probe"; then
                log_success "Found probe: $probe"
            else
                log_error "Missing probe: $probe"
            fi
        done
    fi
fi

# Test 3: Verify probes fire during execution
if [[ $SKIP_BPFTRACE -eq 0 ]]; then
    log_info "Test 3: Verifying probes fire during execution..."

    # Create a simple bpftrace script to catch probes
    TEST_SCRIPT=$(mktemp)
    trap "rm -f $TEST_SCRIPT" EXIT

    cat > "$TEST_SCRIPT" << 'EOF'
BEGIN { @algo_start = 0; @algo_complete = 0; @convergence = 0; }

usdt::mango:algo_start /@algo_start < 5/ {
    @algo_start++;
}

usdt::mango:algo_complete /@algo_complete < 5/ {
    @algo_complete++;
}

usdt::mango:convergence_iter /@convergence < 10/ {
    @convergence++;
}

usdt::mango:convergence_success {
    @success = 1;
}

END {
    printf("algo_start=%d algo_complete=%d convergence_iter=%d success=%d\n",
           @algo_start, @algo_complete, @convergence, @success);
}
EOF

    # Run bpftrace with the test program
    log_info "Running bpftrace with example_heat_equation..."
    OUTPUT=$(timeout 10s bpftrace "$TEST_SCRIPT" -c "$HEAT_EQ_BIN" 2>/dev/null | tail -1)

    if echo "$OUTPUT" | grep -q "algo_start=[1-9]"; then
        log_success "algo_start probe fired"
    else
        log_error "algo_start probe did not fire"
    fi

    if echo "$OUTPUT" | grep -q "algo_complete=[1-9]"; then
        log_success "algo_complete probe fired"
    else
        log_error "algo_complete probe did not fire"
    fi

    if echo "$OUTPUT" | grep -q "convergence_iter=[1-9]"; then
        log_success "convergence_iter probe fired"
    else
        log_error "convergence_iter probe did not fire"
    fi

    if echo "$OUTPUT" | grep -q "success=1"; then
        log_success "convergence_success probe fired"
    else
        log_error "convergence_success probe did not fire"
    fi
fi

# Test 4: Validate helper tool works
log_info "Test 4: Validating helper tool..."

if [[ ! -x "$PROJECT_ROOT/scripts/mango-trace" ]]; then
    log_error "Helper tool not executable: $PROJECT_ROOT/scripts/mango-trace"
else
    log_success "Helper tool is executable"

    if [[ $SKIP_BPFTRACE -eq 0 ]]; then
        # Test check command
        if "$PROJECT_ROOT/scripts/mango-trace" check "$HEAT_EQ_BIN" 2>&1 | grep -q "USDT support: OK"; then
            log_success "Helper tool 'check' command works"
        else
            log_error "Helper tool 'check' command failed"
        fi

        # Test list command
        if "$PROJECT_ROOT/scripts/mango-trace" list "$HEAT_EQ_BIN" 2>&1 | grep -q "mango"; then
            log_success "Helper tool 'list' command works"
        else
            log_error "Helper tool 'list' command failed"
        fi
    fi
fi

# Test 5: Validate bpftrace scripts exist and are executable
log_info "Test 5: Validating bpftrace scripts..."

SCRIPTS=(
    "monitor_all.bt"
    "convergence_watch.bt"
    "debug_failures.bt"
    "performance_profile.bt"
    "pde_detailed.bt"
    "iv_detailed.bt"
)

for script in "${SCRIPTS[@]}"; do
    SCRIPT_PATH="$PROJECT_ROOT/scripts/tracing/$script"
    if [[ -x "$SCRIPT_PATH" ]]; then
        log_success "Script exists and is executable: $script"
    else
        log_error "Script missing or not executable: $script"
    fi
done

# Summary
echo
echo "========================================"
echo "Test Summary"
echo "========================================"
echo -e "${GREEN}Passed:${NC} $TESTS_PASSED"
echo -e "${RED}Failed:${NC} $TESTS_FAILED"

if [[ $TESTS_FAILED -eq 0 ]]; then
    echo
    echo -e "${GREEN}✓ All tests passed!${NC}"
    exit 0
else
    echo
    echo -e "${RED}✗ Some tests failed${NC}"
    exit 1
fi
