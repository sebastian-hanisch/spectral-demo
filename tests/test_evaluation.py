import pytest

import sp_constants as C
from sp_evaluation import kmeans_baseline_labels, method_comparison, rand_index
from sp_scenario import generate_instance


def test_rand_index_is_one_for_identical_partitions():
    assert rand_index([0, 0, 1, 1], [0, 0, 1, 1]) == pytest.approx(1.0)


def test_rand_index_is_one_up_to_relabeling():
    assert rand_index([0, 0, 1, 1], [5, 5, 9, 9]) == pytest.approx(1.0)


def test_rand_index_penalizes_disagreement():
    assert rand_index([0, 0, 1, 1], [0, 1, 0, 1]) < 0.6


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_spectral_clustering_solves_moons_where_kmeans_baseline_fails(seed):
    """Kern-Nachweis 1: bei den nicht-konvexen Halbmonden trennt Spectral Clustering
    sauber, waehrend eine reine k-Means-Baseline auf denselben Rohdaten (die dieselbe
    Konvexitaetsannahme macht wie k-Means selbst) nachweislich scheitert."""
    instance = generate_instance(150, 2, 0.08, "moons", seed=seed)
    scores = method_comparison(instance.as_array(), instance.true_labels, k=2, n_neighbors=10, seed=1)
    assert scores["spectral"] > 0.9
    assert scores["kmeans_baseline"] < 0.75


def test_kmeans_baseline_is_fine_on_convex_blobs():
    """Baseline-Gegenprobe: bei konvexen Gruppen gibt es keinen Vorteil - beide Verfahren
    sollen dort etwa gleich gut abschneiden."""
    instance = generate_instance(90, 3, 0.15, "blobs", seed=1)
    scores = method_comparison(instance.as_array(), instance.true_labels, k=3, n_neighbors=8, seed=1)
    assert scores["spectral"] > 0.9
    assert scores["kmeans_baseline"] > 0.9


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_wrong_k_hurts_spectral_clustering_too(seed):
    """Kern-Nachweis 2 (die ehrliche Botschaft der Demo): Spectral Clustering erbt
    k-Means' 'k muss vorab feststehen'-Schwaeche NICHT weg - bei falsch gewaehltem k
    scheitert es genauso wie k-Means."""
    instance = generate_instance(150, 2, 0.08, "moons", seed=seed)
    correct_k = method_comparison(
        instance.as_array(), instance.true_labels, k=2, n_neighbors=10, seed=1
    )
    wrong_k = method_comparison(
        instance.as_array(), instance.true_labels, k=4, n_neighbors=10, seed=1
    )
    assert correct_k["spectral"] > 0.9
    assert wrong_k["spectral"] < correct_k["spectral"]


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_spectral_at_true_k_reference_reveals_wrong_k_degradation(seed):
    """Die App zeigt zusaetzlich 'Spectral Clustering beim wahren k' als Referenzsaeule,
    gerade weil ein reiner Spectral-vs-Baseline-Vergleich bei falschem k nicht zuverlaessig
    zeigt, dass Spectral Clustering selbst leidet (der Rand-Index bestraft eine
    Aufspaltung eines wahren Clusters nur milde - die Baseline kann durch Zufall der
    Geometrie sogar aehnlich oder besser abschneiden). Der direkte 'aktuelles k vs.
    wahres k'-Vergleich ist deshalb die robustere Kennzahl fuer diese Botschaft."""
    instance = generate_instance(150, 2, 0.08, "moons", seed=seed)
    scores = method_comparison(instance.as_array(), instance.true_labels, k=4, n_neighbors=10, seed=1)
    assert scores["spectral_at_true_k"] > 0.9
    assert scores["spectral_at_true_k"] - scores["spectral"] > 0.15


def test_spectral_at_true_k_equals_spectral_when_k_is_already_correct():
    instance = generate_instance(150, 2, 0.08, "moons", seed=1)
    scores = method_comparison(instance.as_array(), instance.true_labels, k=2, n_neighbors=10, seed=1)
    assert scores["spectral_at_true_k"] == pytest.approx(scores["spectral"])


def _run_preset_as_app_would(preset_name):
    """Repliziert app.py's `_compute_run`/`_compute_method_comparison` exakt: `k` erzeugt
    die Szenario-Wahrheit, `target_k` ist die tatsaechlich an den Algorithmus uebergebene
    Clusterzahl - beide MUESSEN fuer das 'Falsches k'-Preset unterschiedlich sein, sonst
    generiert das Preset unbemerkt ein Szenario, dessen wahre Gruppenzahl zufaellig mit dem
    'falschen' k uebereinstimmt (genau dieser Fehler ist beim ersten Entwurf dieses Presets
    passiert und wurde erst live im Browser bemerkt, nicht von einem Test)."""
    p = C.PRESETS[preset_name]
    instance = generate_instance(p["n_points"], p["k"], p["spread"], p["shape"], p["seed"])
    scores = method_comparison(
        instance.as_array(), instance.true_labels, p["target_k"], p["n_neighbors"], C.COMPARISON_SEED
    )
    return instance, p, scores


def test_wrong_k_preset_actually_requests_a_different_k_than_the_scenarios_truth():
    """Die eigentliche Regression: 'Falsches k' muss k != target_k setzen - sonst ist die
    ganze Preset-Idee ein Bug, unabhaengig davon, was die Rand-Index-Zahlen zeigen."""
    p = C.PRESETS["Falsches k"]
    assert p["k"] != p["target_k"]


def test_wrong_k_preset_matches_actual_app_behavior():
    instance, p, scores = _run_preset_as_app_would("Falsches k")
    assert p["target_k"] != instance.k
    assert scores["spectral_at_true_k"] > 0.9
    assert scores["spectral_at_true_k"] - scores["spectral"] > 0.15


def test_moons_preset_matches_actual_app_behavior():
    instance, p, scores = _run_preset_as_app_would("Nicht-konvexe Formen (Halbmonde)")
    assert p["target_k"] == instance.k
    assert scores["spectral"] > 0.9
    assert scores["spectral"] - scores["kmeans_baseline"] > 0.2


def test_simple_preset_matches_actual_app_behavior():
    instance, p, scores = _run_preset_as_app_would("Einfaches Beispiel")
    assert scores["spectral"] > 0.9
    assert scores["kmeans_baseline"] > 0.9


def test_kmeans_baseline_labels_returns_k_distinct_labels():
    instance = generate_instance(60, 3, 0.15, "blobs", seed=1)
    labels = kmeans_baseline_labels(instance.as_array(), k=3, seed=1)
    assert len(set(labels.tolist())) == 3
