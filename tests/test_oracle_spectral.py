"""Unabhängige Orakel für den Spectral-Clustering-Kern: Ähnlichkeitsgraph (sklearn-Nachbarn),
normalisierter Laplace (scipy.sparse.csgraph.laplacian), Eigenwerte über die ANDERE Matrix
D^-1/2 W D^-1/2 (lambda = 1 - mu), Eigenvektor-Residuum, Lloyd-Fixpunkt des k-Means-Schritts,
Rand-Index (sklearn) und Eigenwert-Lücke (numpy.diff)."""

import numpy as np
import pytest

from sp_algorithm import build_similarity_graph, normalized_laplacian, run, simple_kmeans, spectral_embedding
from sp_evaluation import eigenvalue_gap, rand_index
from sp_scenario import generate_instance

neighbors = pytest.importorskip("sklearn.neighbors")
metrics = pytest.importorskip("sklearn.metrics")
csgraph = pytest.importorskip("scipy.sparse.csgraph")
linalg = pytest.importorskip("scipy.linalg")

CASES = [
    dict(n=40 + 6 * i, k=2 + i % 4, spread=0.08 + 0.05 * (i % 5), shape="blobs" if i % 2 else "moons", seed=i,
         nb=4 + i % 9)
    for i in range(12)
]


def _case(c):
    inst = generate_instance(c["n"], c["k"], c["spread"], c["shape"], seed=c["seed"])
    return inst, inst.as_array()


def test_similarity_laplacian_and_spectrum_against_independent_routes():
    for c in CASES:
        _, x = _case(c)
        n = len(x)
        w = build_similarity_graph(x, c["nb"])
        _, ix = neighbors.NearestNeighbors(n_neighbors=c["nb"] + 1).fit(x).kneighbors(x)
        edges = {(min(i, j), max(i, j)): np.linalg.norm(x[i] - x[j]) for i in range(n) for j in ix[i, 1:]}
        sigma = max(np.median(list(edges.values())), 1e-6)
        expect = np.zeros((n, n))
        for (a, b), d in edges.items():
            expect[a, b] = expect[b, a] = np.exp(-d ** 2 / (2 * sigma ** 2))
        assert np.allclose(w, expect, atol=1e-12)

        lap = normalized_laplacian(w)
        assert np.allclose(lap, csgraph.laplacian(w, normed=True), atol=1e-12)
        deg = w.sum(axis=1)
        d_inv = 1 / np.sqrt(deg)
        mu = np.sort(linalg.eigvalsh(d_inv[:, None] * w * d_inv[None, :]))[::-1]
        eigenvalues, embedding = spectral_embedding(lap, c["k"])
        assert np.allclose(eigenvalues, 1 - mu, atol=1e-9)
        # Einbettung = Zeilen der k kleinsten Eigenvektoren (Residuum L u = lambda u), zeilennormiert
        raw = embedding * np.linalg.norm(np.linalg.eigh(lap)[1][:, : c["k"]], axis=1, keepdims=True)
        assert np.allclose(lap @ raw, raw * eigenvalues[: c["k"]], atol=1e-9)
        norms = np.linalg.norm(embedding, axis=1)
        assert np.all((np.abs(norms - 1) < 1e-9) | (norms < 1e-9))  # Einheitslänge (oder Nullzeile bei k < Komponenten)


def test_kmeans_step_reaches_a_lloyd_fixed_point():
    rng = np.random.default_rng(0)
    for c in CASES:
        _, x = _case(c)
        labels = np.asarray(simple_kmeans(x, c["k"], c["seed"]))
        centers = {k: x[labels == k].mean(axis=0) for k in set(labels.tolist())}
        for i, p in enumerate(x):
            d = {k: np.sum((p - m) ** 2) for k, m in centers.items()}
            assert d[labels[i]] <= min(d.values()) + 1e-12
        data = rng.normal(size=(30, 2))
        assert len(simple_kmeans(data, 3, 1)) == 30


def test_run_returns_consistent_result_and_metrics_match_sklearn_and_numpy():
    rng = np.random.default_rng(1)
    for c in CASES:
        inst, x = _case(c)
        res = run(x, c["k"], c["nb"], 1)
        assert len(res.labels) == len(x) and np.asarray(res.embedding).shape == (len(x), c["k"])
        pred = rng.integers(0, c["k"], size=len(x))
        assert rand_index(inst.true_labels, pred) == pytest.approx(metrics.rand_score(inst.true_labels, pred))
        vals = sorted(rng.uniform(0, 2, size=12).tolist())
        assert eigenvalue_gap(vals, c["k"]) == pytest.approx(np.diff(vals[: c["k"] + 1]))


def test_kmeans_step_random_init_limits_documented_in_readme():
    """README "Grenze des k-Means-Schritts": im App-Ablauf (Init-Seed = Szenario-Seed) ist das Ergebnis bei k = 2 praktisch immer perfekt, bei k = 3/4 bleibt ein Teil der Seeds unter 0.99,
    obwohl die Einbettung selbst die Gruppen trennt (bestes von 10 Starts: perfekt)."""
    share = {}
    for k in (2, 3, 4):
        bad = 0
        for seed in range(60):
            inst = generate_instance(150, k, 0.08, "moons", seed=seed)
            x = inst.as_array()
            labels = run(x, k, 10, seed).labels
            bad += metrics.rand_score(inst.true_labels, labels) < 0.99
            if seed < 15:
                emb = spectral_embedding(normalized_laplacian(build_similarity_graph(x, 10)), k)[1]
                assert max(metrics.rand_score(inst.true_labels, simple_kmeans(emb, k, s)) for s in range(10)) > 0.999
        share[k] = bad / 60
    assert share[2] <= 0.05 and 0.1 <= share[3] <= 0.5 and 0.35 <= share[4] <= 0.8
