"""Zufällige 2D-Punktwolken für die Spectral-Clustering-Demo: entweder k Gauß-Cluster auf
einem Ring ("blobs", wie kmeans-demo) oder k nicht-konvexe Halbkreis-Bögen ("moons") - bei
k=2 das klassische Zwei-Halbmonde-Beispiel, bei k>2 wie Blütenblätter auf einem Ring
angeordnet (dieselbe Generalisierung, die dbscan-demo nachträglich brauchte - hier von
Anfang an, damit `k` bei BEIDEN Formen wirkt, siehe
feedback_no_mode_conditional_dead_controls)."""

from dataclasses import dataclass

import numpy as np

from sp_constants import ARC_RADIUS, ARC_RING_RADIUS, RING_RADIUS


@dataclass(frozen=True)
class ClusteringInstance:
    points: tuple
    true_labels: tuple
    shape: str
    k: int

    @property
    def n_points(self):
        return len(self.points)

    def as_array(self):
        return np.array(self.points, dtype=float)


def _generate_blobs(n_points, k, spread, rng):
    angles = np.linspace(0, 2 * np.pi, k, endpoint=False) + rng.uniform(-0.15, 0.15, size=k)
    centers = np.stack([RING_RADIUS * np.cos(angles), RING_RADIUS * np.sin(angles)], axis=1)
    std = max(spread, 0.05) * RING_RADIUS

    counts = np.full(k, n_points // k)
    counts[-1] += n_points - counts.sum()

    points_per_group, labels_per_group = [], []
    for i in range(k):
        pts = rng.normal(loc=centers[i], scale=std, size=(counts[i], 2))
        points_per_group.append(pts)
        labels_per_group.append(np.full(counts[i], i))
    return np.concatenate(points_per_group, axis=0), np.concatenate(labels_per_group, axis=0)


def _generate_moons(n_points, k, spread, rng):
    """k=2: klassisches Zwei-Halbmonde-Beispiel. k>2: k Halbkreis-Bögen wie
    Blütenblätter auf einem Ring, konkave Seite zum Zentrum (identische Konstruktion wie
    in dbscan-demo, hier frisch nachgebaut - kein Cross-Repo-Import)."""
    counts = np.full(k, n_points // k)
    counts[-1] += n_points - counts.sum()
    noise_std = max(spread, 0.05) * ARC_RADIUS * 0.3

    if k == 2:
        t1 = rng.uniform(0, np.pi, counts[0])
        x1, y1 = ARC_RADIUS * np.cos(t1), ARC_RADIUS * np.sin(t1)
        t2 = rng.uniform(0, np.pi, counts[1])
        x2, y2 = ARC_RADIUS * (1 - np.cos(t2)), ARC_RADIUS * (0.5 - np.sin(t2))
        pts1 = np.stack([x1, y1], axis=1) + rng.normal(scale=noise_std, size=(counts[0], 2))
        pts2 = np.stack([x2, y2], axis=1) + rng.normal(scale=noise_std, size=(counts[1], 2))
        points = np.concatenate([pts1, pts2], axis=0)
        labels = np.concatenate([np.zeros(counts[0], dtype=int), np.ones(counts[1], dtype=int)])
        return points, labels

    layout_angles = np.linspace(0, 2 * np.pi, k, endpoint=False) + rng.uniform(-0.1, 0.1, size=k)
    arc_centers = np.stack(
        [ARC_RING_RADIUS * np.cos(layout_angles), ARC_RING_RADIUS * np.sin(layout_angles)], axis=1
    )

    points_per_group, labels_per_group = [], []
    for i in range(k):
        t = rng.uniform(0, np.pi, counts[i])
        local_x, local_y = ARC_RADIUS * np.cos(t), ARC_RADIUS * np.sin(t)
        rot = layout_angles[i] + np.pi
        cos_r, sin_r = np.cos(rot), np.sin(rot)
        rx = cos_r * local_x - sin_r * local_y
        ry = sin_r * local_x + cos_r * local_y
        pts = np.stack([rx, ry], axis=1) + arc_centers[i] + rng.normal(scale=noise_std, size=(counts[i], 2))
        points_per_group.append(pts)
        labels_per_group.append(np.full(counts[i], i))
    return np.concatenate(points_per_group, axis=0), np.concatenate(labels_per_group, axis=0)


def generate_instance(n_points, k, spread, shape, seed):
    """spread steuert je nach shape entweder die Cluster-Streuung ("blobs") oder das
    Rauschen um die ideale Bogen-Kurve ("moons"). k wirkt bei BEIDEN Formen."""
    rng = np.random.default_rng(seed)

    if shape == "moons":
        points, labels = _generate_moons(n_points, k, spread, rng)
    else:
        points, labels = _generate_blobs(n_points, k, spread, rng)

    return ClusteringInstance(
        points=tuple(map(tuple, points.tolist())),
        true_labels=tuple(int(l) for l in labels),
        shape=shape,
        k=k,
    )
