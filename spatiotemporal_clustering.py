"""Spectral clustering of temporal networks via spatio-temporal random walks.

Implements Algorithm 1 and Appendix D of Blaskovic et al. using a matrix-free
representation.  Arrays are ordered snapshot-major: (snapshot, node).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import LinearOperator, eigsh
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score


def _natural_key(path: Path):
    return [int(x) if x.isdigit() else x for x in re.split(r"(\d+)", path.name)]


def load_temporal_network(network_dir, pattern="adj_*"):
    """Load numerically ordered CSV, NPY, or NPZ adjacency matrices."""
    network_dir = Path(network_dir)
    supported = {".csv", ".npy", ".npz"}
    files = sorted(
        (path for path in network_dir.glob(pattern) if path.suffix.lower() in supported),
        key=_natural_key,
    )
    if not files:
        raise FileNotFoundError(
            f"No CSV, NPY, or NPZ files matching {pattern!r} in {network_dir}"
        )
    stems = [path.stem for path in files]
    duplicates = sorted({stem for stem in stems if stems.count(stem) > 1})
    if duplicates:
        raise ValueError(
            "Multiple file formats were found for the same snapshot(s): "
            + ", ".join(duplicates)
        )
    matrices = []
    shape = None
    for file in files:
        suffix = file.suffix.lower()
        if suffix == ".csv":
            a = np.loadtxt(file, delimiter=",")
        elif suffix == ".npy":
            a = np.load(file, allow_pickle=False)
        else:
            try:
                a = sparse.load_npz(file)
            except (ValueError, KeyError):
                with np.load(file, allow_pickle=False) as archive:
                    if len(archive.files) != 1:
                        raise ValueError(
                            f"{file.name} must contain exactly one adjacency array; "
                            f"found keys {archive.files}"
                        )
                    a = archive[archive.files[0]]
        if a.ndim != 2 or a.shape[0] != a.shape[1]:
            raise ValueError(f"{file.name} is not a square matrix: {a.shape}")
        if shape is not None and a.shape != shape:
            raise ValueError(f"All snapshots must have one shape; {file.name} has {a.shape}, expected {shape}")
        values = a.data if sparse.issparse(a) else np.asarray(a)
        if not np.all(np.isfinite(values)) or np.any(values < 0):
            raise ValueError(f"{file.name} contains non-finite or negative weights")
        shape = a.shape
        matrices.append(sparse.csr_matrix(a))
    return matrices, files


def load_ground_truth(example_dir, expected_shape=None):
    """Load *_ground_truth.npy or CSV; return None when neither exists."""
    root = Path(example_dir)
    candidates = sorted(root.glob("*_ground_truth.npy")) or sorted(root.glob("*_ground_truth.csv"))
    if not candidates:
        return None
    labels = np.load(candidates[0]) if candidates[0].suffix == ".npy" else np.loadtxt(candidates[0], delimiter=",")
    labels = np.asarray(labels)
    if expected_shape is not None and labels.shape != tuple(expected_shape):
        raise ValueError(f"Ground truth shape {labels.shape}; expected {tuple(expected_shape)}")
    return labels.astype(int)


def transition_matrices(adjacencies, isolated="self_loop"):
    """Equation (1): row-stochastic S_t = D_t^-1 A_t.

    Isolated nodes receive a self-loop by default, keeping every row stochastic.
    """
    transitions = []
    for a in adjacencies:
        a = sparse.csr_matrix(a, dtype=float).copy()
        degree = np.asarray(a.sum(axis=1)).ravel()
        isolated_nodes = np.flatnonzero(degree <= 0)
        if isolated_nodes.size:
            if isolated != "self_loop":
                raise ValueError("Zero-degree node encountered; use isolated='self_loop'")
            a[isolated_nodes, isolated_nodes] = 1.0
            degree = np.asarray(a.sum(axis=1)).ravel()
        transitions.append(sparse.diags(1.0 / degree) @ a)
    return transitions


def propagate_distributions(transitions, initial=None, floor=1e-15):
    """Propagate nonnegative distributions with positive total mass.

    Normalize the input, then floor entries below ``floor`` and renormalize
    at every snapshot, including the initial one. This numerical regularization
    prevents division by zero in observable coordinates. Where the floor is
    active, distributions approximate rather than exactly satisfy the paper's
    propagation equation. Negative, nonfinite, and all-zero inputs are invalid.
    """
    if not np.isfinite(floor) or not 0 < floor < 1:
        raise ValueError("floor must be finite and strictly between zero and one")
    n = transitions[0].shape[0]
    mu = np.full(n, 1.0 / n) if initial is None else np.asarray(initial, dtype=float).copy()
    if mu.shape != (n,):
        raise ValueError(f"Initial distribution must have shape ({n},), received {mu.shape}")
    if not np.all(np.isfinite(mu)) or np.any(mu < 0) or mu.sum() <= 0:
        raise ValueError("Initial distribution must be finite, nonnegative, and have positive mass")
    # Scaling first also avoids overflow when normalizing large finite weights.
    mu /= mu.max()
    mu /= mu.sum()
    mu = np.maximum(mu, floor)
    mu /= mu.sum()
    distributions = [mu]
    for s in transitions[:-1]:
        mu = np.asarray(mu @ s).ravel()
        mu = np.maximum(mu, floor)
        mu /= mu.sum()
        distributions.append(mu)
    return np.asarray(distributions)


def coupling_matrix(m, alpha=0.03, cyclic=False, bandwidth=None):
    """Equation (12) and its cyclic variant, row-normalized as in equation (9)."""
    if alpha < 0:
        raise ValueError("alpha must be non-negative")
    idx = np.arange(m)
    distance = np.abs(idx[:, None] - idx[None, :]).astype(float)
    if cyclic:
        distance = np.minimum(distance, m - distance)
    weights = np.exp(-alpha * distance**2)
    np.fill_diagonal(weights, 0.0)
    if bandwidth is not None:
        weights[distance > bandwidth] = 0.0
    totals = weights.sum(axis=1)
    if np.any(totals == 0):
        raise ValueError("Coupling network has an isolated snapshot; increase bandwidth or reduce alpha")
    return weights / totals[:, None], weights


def coupling_stationary_distribution(weights):
    """Stationary distribution of H for symmetric unnormalized weights."""
    degree = weights.sum(axis=1)
    return degree / degree.sum()


def prepare_custom_coupling(matrix, m, tolerance=1e-10):
    """Validate a custom H or normalize a custom symmetric weight matrix W.

    A row-stochastic input is interpreted as H and must be reversible. Otherwise,
    a symmetric nonnegative input is interpreted as the weights W from equation
    (9) and is normalized row-wise. In both cases the diagonal must be zero.
    """
    matrix = np.asarray(matrix, dtype=float)
    if matrix.shape != (m, m):
        raise ValueError(f"Custom coupling must have shape ({m}, {m}), received {matrix.shape}")
    if not np.all(np.isfinite(matrix)) or np.any(matrix < 0):
        raise ValueError("Custom coupling must contain finite, nonnegative entries")
    if not np.allclose(np.diag(matrix), 0.0, atol=tolerance, rtol=0):
        raise ValueError("Custom coupling must have a zero diagonal, as assumed in the paper")

    row_sums = matrix.sum(axis=1)
    if np.any(row_sums <= tolerance):
        raise ValueError("Custom coupling cannot contain an isolated snapshot (a zero row)")

    if np.allclose(row_sums, 1.0, atol=tolerance, rtol=tolerance):
        h = matrix.copy()
        # Solve pi^T H = pi^T together with sum(pi) = 1.
        system = np.vstack((h.T - np.eye(m), np.ones((1, m))))
        target = np.concatenate((np.zeros(m), [1.0]))
        pi = np.linalg.lstsq(system, target, rcond=None)[0]
        if np.any(pi <= tolerance) or not np.allclose(pi @ h, pi, atol=1e-8, rtol=1e-8):
            raise ValueError("Custom H must have a strictly positive stationary distribution")
        pi /= pi.sum()
        weights = pi[:, None] * h
        if not np.allclose(weights, weights.T, atol=1e-8, rtol=1e-8):
            raise ValueError("Custom H must be reversible (diag(pi) @ H must be symmetric)")
        return h, weights, pi

    if not np.allclose(matrix, matrix.T, atol=tolerance, rtol=tolerance):
        raise ValueError(
            "A custom matrix that is not row-stochastic must be symmetric so it can be interpreted as W"
        )
    weights = matrix.copy()
    h = weights / row_sums[:, None]
    pi = coupling_stationary_distribution(weights)
    return h, weights, pi


@dataclass
class SpectralResult:
    eigenvalues: np.ndarray
    eigenvectors: np.ndarray  # (M, N, k), observables f = D_nu^-1/2 g
    coupling: np.ndarray
    coupling_weights: np.ndarray
    distributions: np.ndarray
    stationary: np.ndarray


class SpatioTemporalOperator:
    """Matrix-free symmetric representation of Q_perp D_nu^1/2 C_H D_nu^-1/2 Q_perp."""
    def __init__(self, transitions, mu, coupling, pi):
        self.s = transitions
        self.mu = np.asarray(mu)
        self.h = np.asarray(coupling)
        self.pi = np.asarray(pi)
        self.m, self.n = self.mu.shape
        self.sqrt_nu = np.sqrt(self.pi[:, None] * self.mu)
        self.shape = (self.m * self.n, self.m * self.n)

    def project(self, x):
        x = np.asarray(x).reshape(self.m, self.n).copy()
        # In symmetric coordinates, constants in snapshot t are sqrt(mu_t).
        q = np.sqrt(self.mu)
        x -= (x * q).sum(axis=1)[:, None] * q
        return x

    def apply_ch(self, f):
        """Apply equation (11), without materializing its MN by MN blocks."""
        f = np.asarray(f).reshape(self.m, self.n)
        out = np.zeros_like(f)
        for t in range(self.m):
            # K_ts f_s = S_t ... S_(s-1) f_s, computed backwards in s.
            propagated = f[t].copy()
            for sidx in range(t - 1, -1, -1):
                propagated = self.s[sidx] @ propagated
                out[sidx] += self.h[sidx, t] * propagated
            # T_ts f_t = D_mu_s^-1 (S_t...S_(s-1))^T D_mu_t f_t.
            transported = self.mu[t] * f[t]
            for sidx in range(t + 1, self.m):
                transported = self.s[sidx - 1].T @ transported
                out[sidx] += self.h[sidx, t] * (transported / self.mu[sidx])
        return out

    def matvec(self, vector):
        g = self.project(vector)
        f = g / self.sqrt_nu
        y = self.sqrt_nu * self.apply_ch(f)
        return self.project(y).ravel()

    def as_linear_operator(self):
        return LinearOperator(self.shape, matvec=self.matvec, rmatvec=self.matvec, dtype=float)


def compute_spatial_eigenvectors(adjacencies, alpha=0.03, cyclic=False,
                                 n_eigenvectors=8, bandwidth=None,
                                 tolerance=1e-8, maxiter=None,
                                 random_state=0, initial_distribution=None,
                                 custom_coupling=None):
    """Execute Algorithm 1 through extraction of the leading spatial eigenpairs."""
    transitions = transition_matrices(adjacencies)
    mu = propagate_distributions(transitions, initial=initial_distribution)
    if custom_coupling is None:
        h, weights = coupling_matrix(len(adjacencies), alpha, cyclic, bandwidth=bandwidth)
        pi = coupling_stationary_distribution(weights)
    else:
        h, weights, pi = prepare_custom_coupling(custom_coupling, len(adjacencies))
    op = SpatioTemporalOperator(transitions, mu, h, pi)
    dimension = op.shape[0]
    k = min(int(n_eigenvectors), dimension - len(adjacencies) - 1)
    if k < 1:
        raise ValueError("Need at least two nodes and two snapshots")
    rng = np.random.default_rng(random_state)
    v0 = op.project(rng.standard_normal((op.m, op.n))).ravel()
    values, symmetric_vectors = eigsh(op.as_linear_operator(), k=k, which="LA", v0=v0,
                                      tol=tolerance, maxiter=maxiter)
    order = np.argsort(values)[::-1]
    values, symmetric_vectors = values[order], symmetric_vectors[:, order]
    observables = (symmetric_vectors.reshape(op.m, op.n, k) / op.sqrt_nu[:, :, None])
    # Coordinate conversion above is required to obtain C_H observables.
    # Preserve their computed scale and sign: no post-solve normalization.
    return SpectralResult(values, observables, h, weights, mu, pi)


def singular_vector_heuristics(eigenvectors):
    """Appendix D SVD diagnostics for every spatial eigenvector F_k."""
    vectors = np.asarray(eigenvectors)
    diagnostics = []
    for k in range(vectors.shape[2]):
        u, singular_values, vt = np.linalg.svd(vectors[:, :, k], full_matrices=False)
        diagnostics.append({
            "singular_values": singular_values,
            "left_vector": u[:, 0],
            "right_vector": vt[0],
            "rank_one": singular_values[0] * np.outer(u[:, 0], vt[0]),
            "dominance": singular_values[0] / max(singular_values[1] if len(singular_values) > 1 else 0, 1e-15),
        })
    return diagnostics


def cluster_embedding(eigenvectors, selected, n_clusters, random_state=0, n_init=50, row_normalize=False):
    """Cluster selected 1-based spatial eigenvectors; return (M,N) labels and embedding."""
    indices = np.asarray(selected, dtype=int) - 1
    if indices.size == 0 or np.any(indices < 0) or np.any(indices >= eigenvectors.shape[2]):
        raise ValueError(f"selected must contain 1-based indices in 1..{eigenvectors.shape[2]}")
    embedding = np.asarray(eigenvectors)[:, :, indices].reshape(-1, len(indices))
    if row_normalize:
        norms = np.linalg.norm(embedding, axis=1, keepdims=True)
        embedding = embedding / np.maximum(norms, 1e-15)
    labels = KMeans(n_clusters=int(n_clusters), random_state=random_state, n_init=n_init).fit_predict(embedding)
    return labels.reshape(eigenvectors.shape[:2]), embedding


def adjusted_rand(labels, truth):
    """ARI over all space-time nodes (cluster-label permutations do not matter)."""
    if truth is None:
        return None
    if np.shape(labels) != np.shape(truth):
        raise ValueError(f"Label shape {np.shape(labels)} differs from truth shape {np.shape(truth)}")
    return adjusted_rand_score(np.ravel(truth), np.ravel(labels))
