"""Spectral Clustering from scratch (Ng, Jordan & Weiss, 2002): KNN-Ähnlichkeitsgraph mit
Gauß-Kernel-Gewichten, normalisierter Graph-Laplace, Eigenzerlegung, Zeilennormierung,
k-Means auf der Einbettung. sklearn.cluster.SpectralClustering ist in tests/ nur ein
Test-only-Kreuzvergleich (Partitions-Übereinstimmung, kein exaktes Zahlen-Match - Eigensolver
und k-Means-Init unterscheiden sich zwischen Implementierungen)."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SpectralResult:
    similarity: np.ndarray  # (n, n) gewichtete Adjazenzmatrix
    eigenvalues: tuple  # aufsteigend sortiert, alle n
    embedding: np.ndarray  # (n, k), zeilennormiert
    labels: tuple  # finale Clusterzuordnung, 0..k-1


def build_similarity_graph(data, n_neighbors):
    """k-nächste-Nachbarn-Graph (OHNE Selbsteinschluss - Standard-KNN-Konvention, anders
    als DBSCANs `min_samples`), symmetrisiert über die "ODER"-Regel (Kante, wenn j unter
    is nächsten Nachbarn liegt ODER umgekehrt), Kantengewichte über einen Gauß-Kernel mit
    einer aus den erhaltenen Kanten abgeleiteten Bandbreite (Median-Distanz-Heuristik,
    gängige Praxis - keine zusätzliche Reglergröße nötig)."""
    n = len(data)
    dist = np.sqrt(((data[:, None, :] - data[None, :, :]) ** 2).sum(axis=2))

    order = np.argsort(dist, axis=1)
    neighbor_mask = np.zeros((n, n), dtype=bool)
    for i in range(n):
        nn = order[i, 1 : n_neighbors + 1]  # Spalte 0 ist der Punkt selbst (Distanz 0)
        neighbor_mask[i, nn] = True
    adjacency = neighbor_mask | neighbor_mask.T
    np.fill_diagonal(adjacency, False)

    retained_distances = dist[adjacency]
    sigma = np.median(retained_distances) if retained_distances.size > 0 else 1.0
    sigma = max(sigma, 1e-6)

    weights = np.exp(-(dist ** 2) / (2 * sigma ** 2))
    similarity = np.where(adjacency, weights, 0.0)
    return similarity


def normalized_laplacian(similarity):
    """Symmetrisch normalisierter Graph-Laplace (Ng-Jordan-Weiss): L_sym = I - D^-1/2 W
    D^-1/2. Isolierte Punkte (Gradsumme 0) werden mit einer winzigen Regularisierung
    abgefangen, damit die Division nie durch exakt 0 teilt."""
    n = similarity.shape[0]
    degree = similarity.sum(axis=1)
    degree_safe = np.maximum(degree, 1e-12)
    d_inv_sqrt = 1.0 / np.sqrt(degree_safe)
    laplacian = np.eye(n) - (d_inv_sqrt[:, None] * similarity * d_inv_sqrt[None, :])
    return laplacian


def spectral_embedding(laplacian, k):
    """Die k Eigenvektoren zu den k KLEINSTEN Eigenwerten von L_sym, zeilennormiert
    (NJW-Schritt). Gibt zusätzlich ALLE (aufsteigend sortierten) Eigenwerte zurück - für
    den Scree-Plot und die Zusammenhangskomponenten-Verifikation in den Tests."""
    eigenvalues, eigenvectors = np.linalg.eigh(laplacian)  # eigh sortiert bereits aufsteigend
    u = eigenvectors[:, :k]
    row_norms = np.linalg.norm(u, axis=1, keepdims=True)
    row_norms = np.maximum(row_norms, 1e-12)
    embedding = u / row_norms
    return eigenvalues, embedding


def simple_kmeans(data, k, seed, max_iter=100):
    """Kleines, frisches Lloyd's (kein Cross-Repo-Import aus kmeans-demo) für den letzten
    Schritt der Pipeline - Zufalls-Init aus k verschiedenen Datenpunkten."""
    rng = np.random.default_rng(seed)
    n = len(data)
    idx = rng.choice(n, size=k, replace=False)
    centers = data[idx].copy()

    labels = None
    for _iteration in range(max_iter):
        dists = ((data[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
        new_labels = dists.argmin(axis=1)
        if labels is not None and np.array_equal(new_labels, labels):
            break
        labels = new_labels
        for c in range(k):
            mask = labels == c
            if mask.any():
                centers[c] = data[mask].mean(axis=0)
    return labels


def run(data, k, n_neighbors, seed):
    """Führt die vollständige NJW-Pipeline aus: Ähnlichkeitsgraph -> normalisierter
    Laplace -> Eigenzerlegung -> Zeilennormierung -> k-Means auf der Einbettung. Anders
    als k-Means/EM/Gibbs-Sampling ist das ein EINMALIGER Durchlauf, keine Iteration -
    siehe app.py's 4-Phasen-Ansicht statt eines Iterations-Reglers."""
    data = np.asarray(data, dtype=float)
    similarity = build_similarity_graph(data, n_neighbors)
    laplacian = normalized_laplacian(similarity)
    eigenvalues, embedding = spectral_embedding(laplacian, k)
    labels = simple_kmeans(embedding, k, seed)

    return SpectralResult(
        similarity=similarity,
        eigenvalues=tuple(float(v) for v in eigenvalues),
        embedding=embedding,
        labels=tuple(int(l) for l in labels),
    )
