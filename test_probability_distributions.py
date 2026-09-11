"""Regression checks for nonnegative initial and propagated probabilities."""
import unittest
import numpy as np
from scipy import sparse
from spatiotemporal_clustering import propagate_distributions, compute_spatial_eigenvectors
from reduced_spatiotemporal_clustering import compute_reduced_spatial_eigenvectors


class ProbabilityTests(unittest.TestCase):
    def test_zeros_are_regularized_at_initial_and_later_snapshots(self):
        transitions = [sparse.csr_matrix([[1., 0.], [1., 0.]])] * 3
        mu = propagate_distributions(transitions, initial=[0., 1.])
        self.assertTrue(np.all(mu > 0))
        np.testing.assert_allclose(mu.sum(axis=1), 1.)
        self.assertLess(mu[0, 0], 2e-15)
        self.assertLess(mu[1, 1], 2e-15)

    def test_zero_initial_probability_produces_finite_eigenpairs(self):
        for solve in (compute_spatial_eigenvectors, compute_reduced_spatial_eigenvectors):
            kwargs = {'basis_dimension': 2} if solve is compute_reduced_spatial_eigenvectors else {}
            with np.errstate(divide='raise', invalid='raise'):
                result = solve([np.eye(2)] * 3, initial_distribution=[1., 0.],
                               n_eigenvectors=1, **kwargs)
            self.assertTrue(np.isfinite(result.eigenvalues).all())
            self.assertTrue(np.isfinite(result.eigenvectors).all())
            np.testing.assert_allclose(result.eigenvalues, [1.], atol=1e-10)

    def test_invalid_distributions_are_rejected(self):
        for initial in ([-1., 2.], [0., 0.], [np.nan, 1.], [np.inf, 1.]):
            with self.assertRaises(ValueError):
                propagate_distributions([sparse.eye(2)], initial=initial)

    def test_positive_input_propagates_without_material_change(self):
        transition = sparse.csr_matrix([[.8, .2], [.3, .7]])
        mu = propagate_distributions([transition]*2, initial=[2., 3.])
        np.testing.assert_allclose(mu[0], [.4, .6])
        np.testing.assert_allclose(mu[1], mu[0] @ transition, atol=1e-15)


if __name__ == '__main__':
    unittest.main()
