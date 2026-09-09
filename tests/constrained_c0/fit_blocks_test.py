# SPDX-License-Identifier: MIT
import unittest
import numpy as np
from fit_blocks import constrained_least_squares

class PhysicalDerivativeFitTest(unittest.TestCase):
    def test_total_vega_constraint_allows_decreasing_premium(self):
        # Premium observations 1,.8 at sigma0,1 favor slope-.2. European
        # vega .1 permits premium slope-.1, whose LS intercept is .95.
        fit = constrained_least_squares(np.array([[1.,0.],[1.,1.]]),
            np.array([1.,.8]), np.array([[0.,1.]]), np.array([-.1]))
        self.assertEqual(fit['status'], 'solved')
        np.testing.assert_allclose(fit['coefficients'], [.95,-.1], atol=1e-12)

    def test_iteration_exhaustion_does_not_claim_zero_fit_error(self):
        fit = constrained_least_squares(np.eye(2), np.ones(2),
            np.eye(2), np.zeros(2), max_iterations=0)
        self.assertEqual(fit['status'], 'iteration_limit')
        self.assertNotIn('coefficients', fit)
        self.assertNotIn('sample_rms', fit)

    def test_nonfinite_observation_is_retained_as_failure(self):
        fit = constrained_least_squares(np.eye(2), np.array([1.,np.nan]),
            np.eye(2), np.zeros(2))
        self.assertEqual(fit['status'], 'nonfinite_input')
        self.assertNotIn('coefficients', fit)

if __name__ == '__main__': unittest.main()
