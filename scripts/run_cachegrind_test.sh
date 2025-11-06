#!/usr/bin/env bash
#
# Cachegrind test harness for cache-blocking verification
#
# This script runs the cachegrind_harness binary twice:
#   1. With cache blocking enabled
#   2. Without cache blocking (disabled)
#
# It compares L1 data cache miss rates to verify that cache blocking
# provides measurable benefit.
#
# Requirements:
#   - valgrind (with cachegrind tool)
#   - bazel (to build the harness)
#
# Usage:
#   ./scripts/run_cachegrind_test.sh

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
HARNESS="bazel-bin/tests/cachegrind_harness"
OUT_DIR="cachegrind_results"

echo -e "${BLUE}=== Cachegrind Cache-Blocking Test ===${NC}\n"

# Step 1: Build the harness
echo -e "${BLUE}[1/5] Building cachegrind_harness...${NC}"
bazel build //tests:cachegrind_harness
echo -e "${GREEN}✓ Build complete${NC}\n"

# Step 2: Create output directory
mkdir -p "$OUT_DIR"

# Step 3: Run WITHOUT cache blocking
echo -e "${BLUE}[2/5] Running WITHOUT cache blocking...${NC}"
valgrind --tool=cachegrind \
    --cachegrind-out-file="$OUT_DIR/cachegrind.out.no-blocking" \
    --log-file="$OUT_DIR/valgrind.log.no-blocking" \
    "$HARNESS" --without-blocking
echo -e "${GREEN}✓ Run complete${NC}\n"

# Step 4: Run WITH cache blocking
echo -e "${BLUE}[3/5] Running WITH cache blocking...${NC}"
valgrind --tool=cachegrind \
    --cachegrind-out-file="$OUT_DIR/cachegrind.out.with-blocking" \
    --log-file="$OUT_DIR/valgrind.log.with-blocking" \
    "$HARNESS" --with-blocking
echo -e "${GREEN}✓ Run complete${NC}\n"

# Step 5: Parse and compare results
echo -e "${BLUE}[4/5] Analyzing results...${NC}"

# Extract L1 data cache statistics
extract_cache_stats() {
    local file=$1
    local output_var_prefix=$2

    # Parse cachegrind output
    # Format: D refs:          12,345,678
    #         D1  misses:       1,234,567
    #         LLd misses:         123,456

    local d_refs=$(grep "D refs:" "$file" | awk '{print $3}' | tr -d ',')
    local d1_misses=$(grep "D1  misses:" "$file" | awk '{print $3}' | tr -d ',')
    local lld_misses=$(grep "LLd misses:" "$file" | awk '{print $3}' | tr -d ',')

    # Calculate miss rate
    local d1_miss_rate=$(echo "scale=4; $d1_misses * 100.0 / $d_refs" | bc)

    echo "$d_refs $d1_misses $lld_misses $d1_miss_rate"
}

# Get stats for both runs
no_block_stats=$(extract_cache_stats "$OUT_DIR/cachegrind.out.no-blocking" "no_block")
with_block_stats=$(extract_cache_stats "$OUT_DIR/cachegrind.out.with-blocking" "with_block")

# Parse results
read no_block_refs no_block_d1miss no_block_llmiss no_block_d1rate <<< "$no_block_stats"
read with_block_refs with_block_d1miss with_block_llmiss with_block_d1rate <<< "$with_block_stats"

# Step 6: Display results
echo -e "\n${BLUE}[5/5] Results:${NC}\n"

echo "┌─────────────────────────────────────────────────────────────┐"
echo "│                    L1 Data Cache Statistics                 │"
echo "├─────────────────────────────────────────────────────────────┤"
printf "│ %-25s │ %15s │ %15s │\n" "Metric" "No Blocking" "With Blocking"
echo "├─────────────────────────────────────────────────────────────┤"
printf "│ %-25s │ %'15d │ %'15d │\n" "D refs (total)" "$no_block_refs" "$with_block_refs"
printf "│ %-25s │ %'15d │ %'15d │\n" "D1 misses" "$no_block_d1miss" "$with_block_d1miss"
printf "│ %-25s │ %'15d │ %'15d │\n" "LLd misses" "$no_block_llmiss" "$with_block_llmiss"
printf "│ %-25s │ %14.2f%% │ %14.2f%% │\n" "D1 miss rate" "$no_block_d1rate" "$with_block_d1rate"
echo "└─────────────────────────────────────────────────────────────┘"

# Calculate improvement
d1_miss_reduction=$(echo "scale=2; 100.0 * (1.0 - $with_block_d1miss / $no_block_d1miss)" | bc)
ll_miss_reduction=$(echo "scale=2; 100.0 * (1.0 - $with_block_llmiss / $no_block_llmiss)" | bc)

echo ""
echo -e "${BLUE}Improvement Summary:${NC}"
printf "  L1 miss reduction: %.2f%%\n" "$d1_miss_reduction"
printf "  LL miss reduction: %.2f%%\n" "$ll_miss_reduction"
echo ""

# Verdict
if (( $(echo "$d1_miss_reduction > 5.0" | bc -l) )); then
    echo -e "${GREEN}✓ SUCCESS: Cache blocking reduces L1 misses by ${d1_miss_reduction}%${NC}"
    exit 0
elif (( $(echo "$d1_miss_reduction > 0.0" | bc -l) )); then
    echo -e "${YELLOW}⚠ MARGINAL: Cache blocking reduces L1 misses by only ${d1_miss_reduction}%${NC}"
    echo -e "${YELLOW}  This may indicate hardware with large L1 cache or other bottlenecks${NC}"
    exit 0
else
    echo -e "${RED}✗ FAILURE: Cache blocking does not improve L1 cache performance${NC}"
    echo -e "${RED}  D1 miss reduction: ${d1_miss_reduction}%${NC}"
    exit 1
fi
