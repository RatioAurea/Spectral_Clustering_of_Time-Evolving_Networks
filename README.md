# Spatio-temporal spectral clustering

Code for the paper **“Spectral Clustering of Time-Evolving Networks Using Spatio-Temporal Random Walks”** by F. Blaskovic, T. Conrad, S. Klus, and N. Djurdjevac Conrad.

The repository contains notebooks for applying the full and reduced spectral-clustering methods to time-evolving networks. Example datasets used in the paper are included.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Start the main notebook with:

```bash
jupyter lab spatio_temporal_spectral_clustering.ipynb
```

Edit the configuration cell and run all cells. Eigenvector indices are 1-based, as in the paper.

## Notebooks

- `spatio_temporal_spectral_clustering.ipynb`: full method from Algorithm 1.
- `reduced_spatio_temporal_spectral_clustering.ipynb`: reduced model from Section 4.
- `figure_8.ipynb`: reproduces the opinion-dynamics plots from Figure 8.

## Input data

Store the snapshots as numerically named square adjacency matrices, for example:

```text
adj_0.csv
adj_1.csv
adj_2.csv
```

CSV, NumPy (`.npy`), and SciPy sparse (`.npz`) matrices are supported. All snapshots must have the same size and node ordering.

Optional reference labels can be provided in a file named `*_ground_truth.npy` or `*_ground_truth.csv`. The expected shape is `(snapshots, nodes)`.

## Repository contents

- `spatiotemporal_clustering.py`: implementation of the full method.
- `reduced_spatiotemporal_clustering.py`: implementation of the reduced method.
- `PAPER_EXAMPLES/`: example datasets and available reference labels.
- `requirements.txt`: Python dependencies.

