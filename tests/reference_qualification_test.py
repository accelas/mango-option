# SPDX-License-Identifier: MIT
"""Numerical/reference-ledger invariants; no candidate surface is involved."""
import math
import unittest

from tools import reference_qualification as rq


class ConvergenceTest(unittest.TestCase):
    def test_three_level_second_order_sequence(self):
        result = rq.convergence([1.016, 1.004, 1.001], 1e-14)
        self.assertTrue(result["stable"])
        self.assertAlmostEqual(result["uncertainty"], 0.002, places=12)

    def test_stalled_and_oscillating_sequences_are_unresolved(self):
        for values in ([1, 1.1, 1.2], [1, 1.1, 1.09], [1, math.nan, 1]):
            self.assertFalse(rq.convergence(values, 1e-14)["stable"])

    def test_mesh_limited_bumps_are_not_called_roundoff(self):
        result = rq.convergence([37.9, 37.90001, 37.900015], 0.001, floor_kind="mesh")
        self.assertTrue(result["stable"])
        self.assertEqual(result["reason"], "mesh-limited")

    def test_flat_sequence_keeps_roundoff_floor(self):
        result = rq.convergence([7, 7, 7], 1e-12)
        self.assertTrue(result["stable"])
        self.assertGreaterEqual(result["uncertainty"], 1e-12)


class EligibilityTest(unittest.TestCase):
    def test_price_budget_does_not_qualify_iv_budget(self):
        result = rq.iv_eligibility(7.0, 0.0, 100.0, 0.0009, 10.0, 0.001)
        self.assertEqual(result["status"], "oracle-unresolved")
        self.assertEqual(result["reason"], "vega-scaled-price-budget")

    def test_filtered_is_not_a_zero_error_iv_measurement(self):
        result = rq.iv_eligibility(0.005, 0.0, 100.0, 1e-8, 1.0, 1e-6)
        self.assertEqual(result["status"], "tv-filtered")
        self.assertIsNone(result["oracle_iv_resolution"])

    def test_vega_floor_straddle_stays_unresolved(self):
        result = rq.iv_eligibility(7.0, 0.0, 100.0, 1e-10, 1e-4, 1e-5)
        self.assertEqual(result["status"], "oracle-unresolved")
        self.assertEqual(result["reason"], "vega-floor-straddle")

    def test_measurable_uses_lower_vega_endpoint(self):
        result = rq.iv_eligibility(7.0, 0.0, 100.0, 1e-6, 20.0, 0.01)
        self.assertEqual(result["status"], "reference-measurable")
        self.assertAlmostEqual(result["oracle_iv_resolution"], 1e-6 / 19.99)


class AdmissionAndResumeTest(unittest.TestCase):
    def test_put_no_exercise_region_preserves_cash_restriction(self):
        row = {"option_type": "PUT", "rate": -0.05, "dividend_yield": 0.02,
               "rolled_dividends": []}
        self.assertTrue(rq.analytic_eligible(row))
        self.assertFalse(rq.analytic_eligible(dict(row, rate=0.01)))
        self.assertFalse(rq.analytic_eligible(dict(row, dividend_yield=-0.06)))
        self.assertFalse(rq.analytic_eligible(dict(row, rolled_dividends=[{"calendar_time": 0.1, "amount": 1}])))

    def test_known_intrinsic_violation_is_not_a_qualified_price(self):
        self.assertFalse(rq.price_qualified(9.0, 10.0, 1e-5))
        self.assertTrue(rq.price_qualified(10.0, 10.0, 1e-5))

    def test_requesting_new_crosscheck_reopens_saved_case(self):
        from types import SimpleNamespace
        args = SimpleNamespace(quantlib=True, audit_analytic=False, rounds=1)
        row = {"option_type": "PUT", "maturity": 1.0, "dividend_yield": 0.02,
               "rate": 0.05, "rolled_dividends": []}
        old = {"status": "qualified", "rounds_requested": 1, "quantlib_requested": False}
        self.assertFalse(rq.should_skip(old, args, row))
        old["quantlib_requested"] = True
        self.assertTrue(rq.should_skip(old, args, row))

    def test_more_budget_reopens_unresolved_analytic_fd_audit(self):
        from types import SimpleNamespace
        args = SimpleNamespace(quantlib=False, audit_analytic=True, rounds=2)
        row = {"option_type": "CALL", "maturity": 1.0, "dividend_yield": 0.0,
               "rate": 0.05, "rolled_dividends": []}
        old = {"status": "qualified", "rounds_requested": 1,
               "fd_audit": {"status": "oracle-unresolved"}}
        self.assertFalse(rq.should_skip(old, args, row))

    def test_event_regularity_is_separate_from_price_admission(self):
        self.assertTrue(rq.exact_cash_event({"events": [{"offset_in_tau": 0.0}]}))
        self.assertFalse(rq.exact_cash_event({"events": [{"offset_in_tau": 1e-6}]}))


if __name__ == "__main__":
    unittest.main()
