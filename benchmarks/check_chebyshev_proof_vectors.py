#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Check characterization enclosures with exact rational polynomial arithmetic.

Usage: bazel run -c opt //benchmarks:chebyshev_polynomial_characterization > report
       python3 benchmarks/check_chebyshev_proof_vectors.py report
These finite checks supplement the inclusion argument; they do not replace it.
"""
from fractions import Fraction
import sys

count = 0
with open(sys.argv[1], encoding="utf-8") as report:
    for line in report:
        if not line.startswith("proof_vector "):
            continue
        u, lower, upper, d_lower, d_upper = (
            Fraction(float.fromhex(item)) for item in line.split()[1:]
        )
        x = 2 * u - 1
        tm, t, dm, derivative = Fraction(1), x, Fraction(0), Fraction(1)
        value = Fraction(-8, 1024) + Fraction(-7, 1024) * x
        slope = Fraction(-7, 1024)
        for degree in range(2, 257):
            next_t = 2 * x * t - tm
            next_d = 2 * t + 2 * x * derivative - dm
            coefficient = Fraction(degree % 17 - 8, 1024)
            value += coefficient * next_t
            slope += coefficient * next_d
            tm, t, dm, derivative = t, next_t, derivative, next_d
        slope /= 2  # physical domain [-2,2]
        assert lower <= value <= upper, (u, "value enclosure")
        assert d_lower <= slope <= d_upper, (u, "partial enclosure")
        count += 1
assert count == 65, count
print(f"PASS: {count} exact-rational degree-256 value and partial enclosures")
