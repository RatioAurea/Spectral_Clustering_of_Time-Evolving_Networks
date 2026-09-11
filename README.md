# Spatio-temporal random-walk spectral clustering

Code for **“Spectral Clustering of Time-Evolving Networks Using Spatio-Temporal Random Walks”** by F. Blaskovic, T. Conrad, S. Klus, and N. Djurdjevac Conrad.

The notebook `spatio_temporal_spectral_clustering.ipynb` implements Algorithm 1 and the singular-vector diagnostics in Appendix D. It loads the included examples or any user dataset stored as numerically named, square adjacency matrices (`adj_0.csv`, `adj_1.npy`, `adj_2.npz`, …). Dense CSV/NPY/NPZ and SciPy sparse NPZ matrices are supported and may be mixed. The optional ground truth must be an `(snapshots, nodes)` array named `*_ground_truth.npy` or `*_ground_truth.csv` in the parent directory.

## Run

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
jupyter lab spatio_temporal_spectral_clustering.ipynb
```

Edit only the configuration cell, then run all cells. Spatial eigenvector indices are **1-based**, matching the paper and plot titles. The bundled `example_3` has no ground-truth file, so ARI is skipped for it.

## Configuration details

### Reduced model (Section 4)

Open `reduced_spatio_temporal_spectral_clustering.ipynb` for the same plotting,
Appendix D diagnostics, manual clustering, and ARI workflow using the Galerkin
model. It defaults to Example 2 with alpha 0.03, noncyclic coupling, selected
lifted modes `[1, 2]`, and three clusters. `BASIS_DIMENSION` (default 4) includes
the constant and may be an integer or one dimension per snapshot. Automatic
bases use exact dominant random-walk eigenvectors for undirected snapshots;
leave `CUSTOM_BASES = None` for dominant eigenvectors, or supply a list of
basis arrays to use custom bases. Python calls may also explicitly select
`basis_method='exact'` or `'custom'`.
`CUSTOM_BASES` accepts one real basis array per snapshot. 

The helper `reduced_spatiotemporal_clustering.py` projects exact transition
products onto these bases, removes temporal coordinates, solves the symmetric
reduced eigenproblem, and lifts modes to node observables before clustering.
Initial distributions must be finite, nonnegative, and have positive total mass.
A probability floor of 1e-15 is applied before renormalizing the initial and
propagated distributions to avoid division by zero. When active, this is a
numerical regularization of the paper’s exact propagation equation. No full space-time matrix is
constructed, but the reduced spatial matrix is dense. Increase the basis size
to assess convergence; a basis dimension of N recovers the full search space.
Set `COMPARE_WITH_FULL = True` for an optional eigenvalue comparison. The
existing dependencies suffice for both notebooks.

`ROW_NORMALIZE_EMBEDDING` controls preprocessing immediately before k-means:

- `False` (the paper-style default): cluster the selected spatial-eigenvector values directly.
- `True`: divide every space-time node's feature vector by its Euclidean norm before clustering.

Row normalization removes feature-vector magnitude and retains only its direction. It is common in some variants of spectral clustering, but it can also discard meaningful magnitude information; leave it disabled when reproducing the paper's results.

`CUSTOM_COUPLING_MATRIX` optionally replaces the paper's exponential coupling, `w_ts = exp(-alpha * distance(t, s)^2)`. Its default value is `None`, so the notebook constructs the coupling from `ALPHA`, `CYCLIC_COUPLING`, and `COUPLING_BANDWIDTH`. To use a custom coupling, assign or load an `M × M` NumPy array in the configuration cell. The array may be either a row-stochastic reversible coupling matrix `H`, which is used directly, or a symmetric nonnegative weight matrix `W`, which is normalized according to equation (9). It must have a zero diagonal and no zero rows. For example:

```python
CUSTOM_COUPLING_MATRIX = np.load('my_coupling_matrix.npy')
```

## Repository contents

- `spatio_temporal_spectral_clustering.ipynb` — documented analysis and visualization workflow.
- `spatiotemporal_clustering.py` — local implementation imported by the notebook.
- `PAPER_EXAMPLES/` — the four bundled temporal-network datasets and available labels.
- `requirements.txt` — all third-party Python dependencies.
- `figure_8.ipynb` — plots trajectories from Figure 8.

No package from elsewhere in the original project is required. Clone or download this directory, keep these files together, start Jupyter from this directory, and run the notebook.

Paper configurations include: `example_1` with `alpha=0.01`, cyclic coupling, eigenvectors `[1,3,4]`, and 6 clusters; `example_2` with `alpha=0.03`, eigenvectors `[1,2]`, and 3 clusters; and `example_3` with `alpha=0.01`, eigenvectors `[1,3]` (the paper uses its reduced model there). The guiding example uses eigenvectors `[1,3]` and 3 clusters. Because k-means is stochastic and equivalent partitions can have different integer labels, the implementation fixes a random seed and evaluates with permutation-invariant ARI.

The implementation is matrix-free: it applies the exact full operator from equation (11) without allocating its dense `MN × MN` matrix. Isolated nodes are assigned self-loops. Input matrices must be nonnegative, share one square shape, and use the same node ordering at every snapshot.

## Figure 8: opinion trajectories

Run `figure_8.ipynb` to cluster the existing adjacency matrices in
`PAPER_EXAMPLES/example_3/network` (all 503 nodes). The reduced model uses
alpha 0.01, spatial modes 1 and 3, and four clusters, with configurable basis
dimension. `trajectories.npy` is used only for panels (a) and (b): trajectories
of 500 voters and three parties, and voter dots colored by the network labels.
Panel (c) shows computed eigenvectors for all network nodes. 
Set `SAVE_FIGURE = True` to save outputs to `figure_8_output/` as PNG, PDF, and a compressed archive
of all-node labels, eigenvectors, and network filenames.
