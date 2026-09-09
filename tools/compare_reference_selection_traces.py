#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Compare public evaluator traces exactly, excluding only measured runtime."""
import argparse
import difflib
import json
from pathlib import Path
import re


def normalize(path):
    return [re.sub(r" seconds=[^ ]+", "", line.rstrip())
            for line in Path(path).read_text().splitlines() if line.strip()]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", help="Baseline reference_selection_probe stdout")
    parser.add_argument("candidate", help="Candidate reference_selection_probe stdout")
    args = parser.parse_args()
    before, after = normalize(args.baseline), normalize(args.candidate)
    if not all(any(row.startswith(prefix) for row in before)
               for prefix in ("sequence,", "refs=", "provider_work ")):
        parser.error("baseline must contain numerical sequences and request/provider summaries")
    if before != after:
        print("\n".join(difflib.unified_diff(before, after,
            fromfile=args.baseline, tofile=args.candidate, lineterm="")))
        return 1
    sequences = [row for row in before if row.startswith("sequence,")]
    print(json.dumps(dict(exact_match=True, sequence_rows=len(sequences),
        grid_price_or_derivative_scalars=9 * len(sequences),
        provider_work=[row for row in before if row.startswith("provider_work ")],
        request_counts=[row for row in before if row.startswith("refs=")]), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
