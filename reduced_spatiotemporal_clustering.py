"""Section 4 Galerkin reduction; no full space-time matrix is constructed."""
from dataclasses import dataclass

import numpy as np
from scipy import sparse
from scipy.linalg import eigh
from scipy.sparse.linalg import eigsh

from spatiotemporal_clustering import (
    SpectralResult, transition_matrices, propagate_distributions,
    coupling_matrix, coupling_stationary_distribution, prepare_custom_coupling,
)


@dataclass
class ReducedSpectralResult(SpectralResult):
    bases: list  # mu_t-orthonormal bases, constant first
    coefficients: list  # f_t = bases[t] @ coefficients[t]
    spatial_operator: np.ndarray  # symmetric reduced spatial operator


def _orthonormal_basis(candidates, mu):
    """Preserve the supplied span plus constants, removing dependent columns."""
    candidates = np.asarray(candidates, dtype=float)
    n = len(mu)
    if candidates.ndim != 2 or candidates.shape[0] != n or not np.all(np.isfinite(candidates)):
        raise ValueError('Each basis must be a finite N x d real array')
    q = np.sqrt(mu)
    centered = q[:, None] * candidates
    centered -= q[:, None] * (q @ centered)[None, :]
    # Use the uncentered scale so a numerically constant column is discarded.
    scale = np.linalg.norm(q[:, None] * candidates)
    u, values, _ = np.linalg.svd(centered, full_matrices=False)
    rank = np.count_nonzero(values > 1e-12 * max(scale, np.finfo(float).tiny))
    return np.column_stack((np.ones(n), u[:, :rank] / q[:, None]))


def snapshot_bases(adjacencies, mu, basis_dimension=4, random_state=0):
    """Exact dominant random-walk eigenspaces for undirected snapshots.

    basis_dimension includes the constant; constants are explicitly included
    even for disconnected graphs. The dimension may be a scalar or length M.
    Directed networks require user-supplied bases instead.
    """
    m, n = mu.shape
    dimensions = np.broadcast_to(np.asarray(basis_dimension), (m,))
    if np.any(dimensions != dimensions.astype(int)) or np.any(dimensions < 2) or np.any(dimensions > n):
        raise ValueError('basis_dimension must contain integers between 2 and N')
    rng = np.random.default_rng(random_state)
    bases = []
    for t, adjacency in enumerate(adjacencies):
        a = sparse.csr_matrix(adjacency, dtype=float).copy()
        delta = a - a.T
        if delta.nnz and np.max(np.abs(delta.data)) > 1e-12:
            raise ValueError('Automatic snapshot bases require undirected adjacency matrices; supply bases for directed networks')
        degree = np.asarray(a.sum(axis=1)).ravel()
        a = a + sparse.diags((degree == 0).astype(float))
        degree = np.asarray(a.sum(axis=1)).ravel()
        inv_sqrt = 1 / np.sqrt(degree)
        symmetric = sparse.diags(inv_sqrt) @ a @ sparse.diags(inv_sqrt)
        d = int(dimensions[t])
        if d == n:
            bases.append(_orthonormal_basis(np.eye(n), mu[t]))
            continue
        # Request d modes, then retain d-1 independent centered directions.
        values, vectors = eigsh(symmetric, k=d, which='LA', v0=rng.standard_normal(n))
        candidates = inv_sqrt[:, None] * vectors[:, np.argsort(values)[::-1]]
        # Sequential orthogonalization preserves dominance ordering and handles
        # arbitrary rotations within the eigenvalue-1 space of disconnected graphs.
        columns = [np.ones(n)]
        for candidate in candidates.T:
            v = candidate.copy()
            for _ in range(2):
                for column in columns:
                    v -= column * np.dot(mu[t] * column, v)
            norm = np.sqrt(np.dot(mu[t], v * v))
            if norm > 1e-10 * np.sqrt(np.dot(mu[t], candidate * candidate)):
                columns.append(v / norm)
            if len(columns) == d:
                break
        if len(columns) != d:
            raise ValueError('Could not construct the requested independent snapshot modes')
        bases.append(np.column_stack(columns))
    return bases


def compute_reduced_spatial_eigenvectors(
        adjacencies, alpha=0.03, cyclic=False, n_eigenvectors=8,
        basis_dimension=4, bases=None, bandwidth=None, random_state=0,
        initial_distribution=None, custom_coupling=None, basis_method=None):
    """Equations (18)--(25): project, remove temporal modes, solve, and lift.

    basis_method selects 'exact' or 'custom'. None preserves the
    original API: custom when bases are supplied, otherwise exact.
    Optional bases may have different dimensions at each snapshot. Constants
    are added and dependencies removed before mu-weighted orthonormalization.
    Cross-covariances use exact sparse transition applications to thin bases,
    never products of projected one-step operators. The dense reduced solve
    uses O((sum_t (d_t-1))**2) memory.
    """
    if len(adjacencies) < 2:
        raise ValueError('Need at least two snapshots')
    n = adjacencies[0].shape[0]
    for a in adjacencies:
        a = sparse.csr_matrix(a)
        if a.shape != (n, n) or not np.all(np.isfinite(a.data)) or np.any(a.data < 0):
            raise ValueError('Adjacencies must share a square shape and have finite nonnegative weights')
    transitions = transition_matrices(adjacencies)
    mu = propagate_distributions(transitions, initial=initial_distribution)
    if custom_coupling is None:
        h, weights = coupling_matrix(len(adjacencies), alpha, cyclic, bandwidth=bandwidth)
        pi = coupling_stationary_distribution(weights)
    else:
        h, weights, pi = prepare_custom_coupling(custom_coupling, len(adjacencies))
    method = ('custom' if bases is not None else 'exact') if basis_method is None else basis_method
    if method not in {'exact', 'custom'}:
        raise ValueError("basis_method must be 'exact' or 'custom'")
    if method != 'custom' and bases is not None:
        raise ValueError("Set basis_method='custom' to use supplied bases")
    if method == 'custom' and bases is None:
        raise ValueError("basis_method='custom' requires one basis array per snapshot")
    if method != 'custom':
        bases = snapshot_bases(adjacencies, mu, basis_dimension, random_state)
    else:
        if len(bases) != len(adjacencies):
            raise ValueError('Supply one basis per snapshot')
        bases = [_orthonormal_basis(w, mass) for w, mass in zip(bases, mu)]
    spatial = [w[:, 1:] for w in bases]
    offsets = np.cumsum([0] + [w.shape[1] for w in spatial])
    dimension = int(offsets[-1])
    if int(n_eigenvectors) != n_eigenvectors or n_eigenvectors < 1 or dimension == 0:
        raise ValueError('Need a positive integer n_eigenvectors and a nonempty spatial basis')
    operator = np.zeros((dimension, dimension))
    for s in range(1, len(bases)):
        propagated = spatial[s].copy()
        targets = np.flatnonzero(h[:s, s])
        if not len(targets):
            continue
        for t in range(s - 1, int(targets[0]) - 1, -1):
            propagated = transitions[t] @ propagated
            if h[t, s] == 0:
                continue
            covariance = spatial[t].T @ (mu[t, :, None] * propagated)
            block = np.sqrt(pi[t] / pi[s]) * h[t, s] * covariance
            ts, ss = slice(offsets[t], offsets[t+1]), slice(offsets[s], offsets[s+1])
            operator[ts, ss] = block
            operator[ss, ts] = block.T
    k = min(int(n_eigenvectors), dimension)
    values, vectors = eigh(operator, subset_by_index=(dimension-k, dimension-1))
    values, vectors = values[::-1], vectors[:, ::-1]
    coefficients = [np.vstack((np.zeros((1, k)), vectors[offsets[t]:offsets[t+1]] / np.sqrt(pi[t])))
                    for t in range(len(bases))]
    observables = np.stack([w @ a for w, a in zip(bases, coefficients)])
    # Lift coefficients without any post-solve normalization or sign changes.
    # The pi factors convert symmetric coordinates to reduced observables.
    return ReducedSpectralResult(values, observables, h, weights, mu, pi,
                                 bases, coefficients, operator)
