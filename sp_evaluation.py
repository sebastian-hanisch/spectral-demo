"""Rand-Index (from scratch), eine k-Means-Baseline (from scratch, kein Cross-Repo-Import)
für den direkten Vergleich, sowie die Eigenwert-Lücke-Heuristik zur k-Wahl."""

import numpy as np

from sp_algorithm import run, simple_kmeans


def rand_index(true_labels, pred_labels):
    true_arr = np.asarray(true_labels)
    pred_arr = np.asarray(pred_labels)
    n = len(true_arr)
    if n < 2:
        return 1.0

    iu = np.triu_indices(n, k=1)
    same_true = (true_arr[:, None] == true_arr[None, :])[iu]
    same_pred = (pred_arr[:, None] == pred_arr[None, :])[iu]
    return float(np.mean(same_true == same_pred))


def kmeans_baseline_labels(data, k, seed):
    """Direkte k-Means-Baseline auf den ROHEN Koordinaten (nicht der Spektral-
    Einbettung) - macht den Kontrast zu Spectral Clustering sichtbar: dieselbe
    k-Means-Implementierung, einmal direkt auf die Daten angewendet, einmal (in
    sp_algorithm.run) auf die Spektral-Einbettung."""
    return simple_kmeans(np.asarray(data, dtype=float), k, seed)


def method_comparison(data, true_labels, k, n_neighbors, seed):
    """Rand-Index von Spectral Clustering vs. der reinen k-Means-Baseline, beide beim
    AKTUELL EINGESTELLTEN k - macht direkt sichtbar, wo Spectral Clustering etwas bringt
    (Nicht-Konvexität). Zusätzlich Spectral Clustering beim WAHREN k als Referenz - eine
    reine Spectral-vs-Baseline-Zahl kann bei falschem k zufällig ähnlich ausfallen (Rand-
    Index bestraft eine Aufspaltung eines wahren Clusters nur milde, siehe project-memory),
    der direkte Vergleich "aktuelles k vs. wahres k" macht die "falsches k schadet auch
    Spectral Clustering"-Botschaft dagegen unabhängig von der Baseline sichtbar."""
    spectral_result = run(data, k, n_neighbors, seed)
    baseline_labels = kmeans_baseline_labels(data, k, seed)
    true_k = len(set(true_labels))
    spectral_at_true_k = (
        rand_index(true_labels, spectral_result.labels)
        if k == true_k
        else rand_index(true_labels, run(data, true_k, n_neighbors, seed).labels)
    )
    return {
        "spectral": rand_index(true_labels, spectral_result.labels),
        "kmeans_baseline": rand_index(true_labels, baseline_labels),
        "spectral_at_true_k": spectral_at_true_k,
    }


def eigenvalue_gap(eigenvalues, max_k):
    """Differenzen aufeinanderfolgender (aufsteigend sortierter) Eigenwerte bis
    max_k+1 - die Standard-Heuristik zur k-Wahl bei Spectral Clustering: eine große
    Lücke nach dem k-ten Eigenwert deutet auf k gut getrennte Cluster hin (Analogon zu
    dbscan-demos k-Distanz-Knick)."""
    values = list(eigenvalues[: max_k + 1])
    return [values[i + 1] - values[i] for i in range(len(values) - 1)]
