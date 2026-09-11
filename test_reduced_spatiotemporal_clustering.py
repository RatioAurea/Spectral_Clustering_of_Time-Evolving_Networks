"""Numerical checks of Section 4 against an independent dense construction."""
import unittest
import numpy as np
from scipy.linalg import block_diag
from reduced_spatiotemporal_clustering import compute_reduced_spatial_eigenvectors


class ReducedModelTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(12)
        self.adjacencies = []
        for _ in range(3):
            a = rng.uniform(.1, 1, (6, 6))
            self.adjacencies.append(a + a.T)

    def full_operator(self, result):
        transitions = [a / a.sum(axis=1)[:, None] for a in self.adjacencies]
        m, n = result.distributions.shape
        ch = np.zeros((m*n, m*n))
        for t in range(m):
            product = np.eye(n)
            for s in range(t+1, m):
                product = product @ transitions[s-1]
                ch[t*n:(t+1)*n, s*n:(s+1)*n] = result.coupling[t,s] * product
                ch[s*n:(s+1)*n, t*n:(t+1)*n] = result.coupling[s,t] * (
                    product.T * result.distributions[t][None,:] / result.distributions[s][:,None])
        return ch

    def check_projection(self, **kwargs):
        r = compute_reduced_spatial_eigenvectors(self.adjacencies, n_eigenvectors=100, **kwargs)
        nu = (r.stationary[:,None] * r.distributions).ravel()
        lift = block_diag(*[w[:,1:] / np.sqrt(p) for w,p in zip(r.bases,r.stationary)])
        expected = lift.T @ (nu[:,None] * self.full_operator(r)) @ lift
        np.testing.assert_allclose(r.spatial_operator, expected, atol=1e-12)
        np.testing.assert_allclose(r.eigenvalues, np.linalg.eigvalsh(expected)[::-1], atol=1e-12)
        f = r.eigenvectors.reshape(len(nu), -1)
        np.testing.assert_allclose(f.T @ (nu[:,None]*f), np.eye(f.shape[1]), atol=1e-12)
        np.testing.assert_allclose(np.einsum('tn,tnk->tk',r.distributions,r.eigenvectors),0,atol=1e-12)
        for t in range(len(r.bases)):
            np.testing.assert_allclose(r.bases[t] @ r.coefficients[t],r.eigenvectors[t],atol=1e-12)
        return r

    def test_variable_dimensions_and_custom_reversible_coupling(self):
        self.check_projection(basis_dimension=[2,3,4], custom_coupling=np.array([[0.,2,0],[2,0,1],[0,1,0]]))

    def test_full_basis_recovers_full_spatial_spectrum(self):
        r = self.check_projection(basis_dimension=6)
        # All 15 spatial modes, including negative modes, remain; no temporal zeros.
        f = r.eigenvectors.reshape(18,15)
        np.testing.assert_allclose(self.full_operator(r) @ f, f*r.eigenvalues, atol=1e-11)
        self.assertLess(r.eigenvalues[-1],0)

    def test_custom_dependent_basis_and_directed_network(self):
        self.adjacencies[0][0,1] += .3
        rng = np.random.default_rng(9)
        bases=[np.column_stack((np.ones(6),np.ones(6),rng.normal(size=(6,2)))) for _ in range(3)]
        r=self.check_projection(bases=bases, initial_distribution=np.arange(1,7))
        self.assertEqual([w.shape[1] for w in r.bases],[3,3,3])
        with self.assertRaisesRegex(ValueError,'undirected'):
            compute_reduced_spatial_eigenvectors(self.adjacencies)

    def test_basis_method_selection(self):
        self.check_projection(basis_method='exact', basis_dimension=3)
        self.check_projection(basis_method='custom', bases=[np.eye(6)]*3)
        for kwargs in ({'basis_method':'unknown'}, {'basis_method':'custom'},
                       {'basis_method':'exact', 'bases':[np.eye(6)]*3}):
            with self.assertRaises(ValueError):
                compute_reduced_spatial_eigenvectors(self.adjacencies, **kwargs)

    def test_invalid_inputs_and_constant_only_basis(self):
        for kwargs in ({'basis_dimension':1}, {'basis_dimension':7}, {'n_eigenvectors':0},
                       {'initial_distribution':[-1,1,1,1,1,1]}, {'bases':[np.ones((6,1))]*3}):
            with self.assertRaises(ValueError):
                compute_reduced_spatial_eigenvectors(self.adjacencies, **kwargs)


if __name__ == '__main__':
    unittest.main()
