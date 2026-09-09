# SPDX-License-Identifier: MIT
"""Completed evidence must take precedence over an accepted cached prefix."""
import copy
import json
from pathlib import Path
import unittest

from tools import reference_consumption as rc


class ConsumptionTest(unittest.TestCase):
    def sample(self):
        fixture = Path(__file__).parent / "data/reference_evidence/earlier_prefixes.json"
        return json.loads(fixture.read_text())[0][0]

    def test_later_qualified_overlapping_evidence_can_recover(self):
        record = self.sample()
        early = record["case"]["history"][-1]
        later = copy.deepcopy(early)
        later["round"] += 2
        later["price"]["price"] += early["price"]["uncertainty"] / 2
        middle = copy.deepcopy(later)
        middle["round"] -= 1
        middle["price"].update(stable=False, uncertainty=None)
        record["case"]["history"].extend([middle, later])
        result = rc.reconcile_reference([record])
        self.assertTrue(result["price_ready"])
        self.assertTrue(result["iv_ready"])
        self.assertEqual(result["price"], later["price"]["price"])

    def test_disjoint_qualified_allowances_do_not_recover(self):
        record = self.sample()
        early = record["case"]["history"][-1]
        later = copy.deepcopy(early)
        later["round"] += 1
        later["price"]["price"] += 4 * early["price"]["uncertainty"]
        record["case"]["history"].append(later)
        result = rc.reconcile_reference([record])
        self.assertFalse(result["price_ready"])
        self.assertEqual(result["reason"], "disjoint-qualified-allowances")
        self.assertEqual(result["disjoint_qualified_allowances"],
                         [[early["round"], later["round"]]])

    def test_latest_price_reclassifies_iv_instead_of_reusing_filter(self):
        record = self.sample()
        record["case"]["row"]["spot"] = 100.0
        early = record["case"]["history"][-1]
        early["price"].update(price=0.00999, uncertainty=0.000005, stable=True)
        early["vega"].update(value=None, uncertainty=None)
        later = copy.deepcopy(early)
        later["round"] += 1
        later["price"].update(price=0.009996, uncertainty=0.00002)
        record["case"]["history"] = [early]
        self.assertTrue(rc.reconcile_reference([record])["iv_ready"])
        record["case"]["history"].append(later)
        result = rc.reconcile_reference([record])
        self.assertTrue(result["price_ready"])
        self.assertFalse(result["iv_ready"])
        self.assertEqual(result["iv"]["status"], "oracle-unresolved")
        self.assertEqual(result["observational_contradictions"],
                         [[early["round"], later["round"]]])
        self.assertEqual(result["disjoint_qualified_allowances"], [])

    def test_conflicting_same_round_assessments_refused(self):
        first = self.sample()
        second = copy.deepcopy(first)
        second["case"]["history"][-1]["price"]["price"] += 1e-8
        with self.assertRaisesRegex(ValueError, "same completed round"):
            rc.reconcile_reference([first, second])

    def test_empty_and_incomplete_evidence_refused(self):
        with self.assertRaises(ValueError):
            rc.reconcile_reference([])
        record = self.sample()
        record["case"]["history"] = []
        with self.assertRaises(ValueError):
            rc.reconcile_reference([record])

    def test_all_twenty_later_failures_block_early_filtered_passes(self):
        fixture = Path(__file__).parent / "data/reference_evidence/earlier_prefixes.json"
        cases = json.loads(fixture.read_text())
        contradictions = []
        for pair in cases:
            with self.subTest(id=pair[0]["case"]["id"]):
                self.assertTrue(rc.reconcile_reference(pair[:1])["price_ready"])
                result = rc.reconcile_reference(pair)
                self.assertFalse(result["price_ready"])
                self.assertFalse(result["iv_ready"])
                self.assertEqual(result["reason"], "latest-price-evidence-unresolved")
                self.assertEqual(result["selected_round"], 2)
                self.assertEqual(result, rc.reconcile_reference(list(reversed(pair))))
                if result["observational_contradictions"]:
                    contradictions.append(pair[0]["case"]["id"])
        self.assertEqual(contradictions, ["C2-PUT-8c0e3eb1686ff95288c5",
                                         "CS-PUT-db700bb90c1b614326f8",
                                         "DQ-PUT-3d3dc231f5341828292e"])

    def test_mixed_worker_or_contract_is_refused(self):
        fixture = Path(__file__).parent / "data/reference_evidence/earlier_prefixes.json"
        original = json.loads(fixture.read_text())[0]
        for field in ("worker", "contract", "budget"):
            pair = copy.deepcopy(original)
            if field == "worker":
                pair[1]["policy"]["effective_worker_hash"] = "different"
            elif field == "budget":
                pair[1]["policy"]["price_budget"] *= 2
            else:
                pair[1]["case"]["row"]["spot"] += 1
            with self.subTest(field=field), self.assertRaises(ValueError):
                rc.reconcile_reference(pair)


if __name__ == "__main__":
    unittest.main()
