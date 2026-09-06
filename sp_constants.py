"""Defaults, Regler-Grenzen, Sicherheitsgrenzen und Presets für die Spectral-Clustering-
Demo."""

DEFAULT_N_POINTS = 150
DEFAULT_K = 2
DEFAULT_TARGET_K = 2
DEFAULT_SPREAD = 0.15
DEFAULT_SEED = 1
DEFAULT_SHAPE = "blobs"
DEFAULT_N_NEIGHBORS = 8

N_POINTS_MIN, N_POINTS_MAX = 30, 300
K_MIN, K_MAX = 2, 6
TARGET_K_MIN, TARGET_K_MAX = 2, 6
SPREAD_MIN, SPREAD_MAX = 0.05, 0.9
N_NEIGHBORS_MIN, N_NEIGHBORS_MAX = 2, 30

SHAPES = ("blobs", "moons")
SHAPE_LABELS = {"blobs": "Gruppen (Blobs)", "moons": "Halbmonde"}

RING_RADIUS = 3.0
ARC_RADIUS = 2.5
ARC_RING_RADIUS = 6.5

# Sicherheitsgrenze fuer die naive O(n^2)-Aehnlichkeitsberechnung.
N_POINTS_HARD_MAX = 300

# Fester Sampler-/Init-Seed fuer den k-Means-Schritt auf der Einbettung sowie fuer die
# from-scratch k-Means-Baseline im Vergleichsabschnitt - unabhaengig vom Szenario-Seed
# (Lehre aus gmm-demo/dpmm-demo: Vergleichs-Randomness nie an den Szenario-Seed koppeln).
COMPARISON_SEED = 1

PRESETS = {
    "Einfaches Beispiel": {
        "n_points": 90, "k": 3, "target_k": 3, "spread": 0.15, "shape": "blobs",
        "n_neighbors": 8, "seed": 1,
    },
    "Nicht-konvexe Formen (Halbmonde)": {
        "n_points": 150, "k": 2, "target_k": 2, "spread": 0.08, "shape": "moons",
        "n_neighbors": 10, "seed": 2,
    },
    "Falsches k": {
        "n_points": 150, "k": 2, "target_k": 4, "spread": 0.08, "shape": "moons",
        "n_neighbors": 10, "seed": 2,
    },
    "Schwierige Graph-Konstruktion": {
        "n_points": 150, "k": 2, "target_k": 2, "spread": 0.08, "shape": "moons",
        "n_neighbors": 2, "seed": 2,
    },
}
