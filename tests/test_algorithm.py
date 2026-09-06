import numpy as np
import pytest
from sklearn.cluster import SpectralClustering

from sp_algorithm import build_similarity_graph, normalized_laplacian, run, spectral_embedding
from sp_scenario import generate_instance


def test_similarity_graph_and_laplacian_hand_computed():
    """Zwei weit entfernte Paare, je n_neighbors=1: A=(0,0),B=(1,0) sind gegenseitig
    naechste Nachbarn (Distanz 1), C=(10,0),D=(11,0) ebenso - keine Kreuzkanten. sigma =
    Median der erhaltenen Distanzen = 1, Kantengewicht = exp(-1/2) fuer beide Kanten.
    Da jeder Knoten genau EINE Kante mit demselben Gewicht wie sein eigener Grad hat,
    vereinfacht sich der normalisierte Laplace pro Block exakt zu [[1,-1],[-1,1]] - mit
    den bekannten Eigenwerten 0 und 2."""
    data = np.array([[0.0, 0.0], [1.0, 0.0], [10.0, 0.0], [11.0, 0.0]])

    similarity = build_similarity_graph(data, n_neighbors=1)
    expected_weight = np.exp(-0.5)
    assert similarity[0, 1] == pytest.approx(expected_weight)
    assert similarity[2, 3] == pytest.approx(expected_weight)
    assert similarity[0, 2] == 0.0
    assert similarity[0, 3] == 0.0
    assert similarity[1, 3] == 0.0
    assert np.allclose(similarity, similarity.T)

    laplacian = normalized_laplacian(similarity)
    assert laplacian[0, 0] == pytest.approx(1.0)
    assert laplacian[0, 1] == pytest.approx(-1.0)
    assert laplacian[2, 3] == pytest.approx(-1.0)
    assert laplacian[0, 2] == pytest.approx(0.0)

    eigenvalues, _ = spectral_embedding(laplacian, k=2)
    assert sorted(round(v, 6) for v in eigenvalues) == [0.0, 0.0, 2.0, 2.0]


def _block_diagonal_similarity(block_sizes, rng):
    """Synthetische, GARANTIERT block-diagonale Aehnlichkeitsmatrix mit exakt
    len(block_sizes) Zusammenhangskomponenten - unabhaengig von build_similarity_graph,
    fuer die Zusammenhangskomponenten-Verifikation unten."""
    n = sum(block_sizes)
    similarity = np.zeros((n, n))
    start = 0
    for size in block_sizes:
        block = rng.uniform(0.1, 1.0, size=(size, size))
        block = (block + block.T) / 2
        np.fill_diagonal(block, 0.0)
        similarity[start : start + size, start : start + size] = block
        start += size
    return similarity


@pytest.mark.parametrize(
    "block_sizes", [(3, 3), (2, 2, 2), (5, 3, 2), (2, 2, 2, 2), (4,)]
)
def test_number_of_zero_eigenvalues_equals_number_of_connected_components(block_sizes):
    """Kern-Verifikation, unabhaengig von jeder Clustering-Semantik: ein klassischer Satz
    der Spektralgraphentheorie besagt, dass die Vielfachheit des Eigenwerts 0 des
    normalisierten Laplace GENAU der Anzahl der Zusammenhangskomponenten des Graphen
    entspricht. Hier auf einer KUENSTLICH block-diagonal konstruierten Matrix mit exakt
    bekannter Komponentenzahl direkt nachgerechnet. Blockgroesse >= 2 gewaehlt, weil das
    Theorem in dieser Form (fuer den NORMALISIERTEN Laplace) echte, degree>0-Knoten
    voraussetzt - ein isolierter Einzelknoten (Blockgroesse 1) traegt beim normalisierten
    Laplace einen Eigenwert von 1 bei, nicht 0 (der Sonderfall wird separat unten
    behandelt, siehe test_isolated_node_contributes_eigenvalue_one_not_zero)."""
    rng = np.random.default_rng(42)
    similarity = _block_diagonal_similarity(block_sizes, rng)
    laplacian = normalized_laplacian(similarity)
    eigenvalues, _ = spectral_embedding(laplacian, k=len(block_sizes))

    n_zero = sum(1 for v in eigenvalues if abs(v) < 1e-9)
    assert n_zero == len(block_sizes)


def test_isolated_node_contributes_eigenvalue_one_not_zero():
    """Dokumentiert den Sonderfall explizit, statt ihn nur stillschweigend aus dem
    Parametrisierungs-Test oben auszuschliessen: L_sym[i,i] = 1 - D^-1/2_ii * W_ii *
    D^-1/2_ii = 1 - 0 = 1 fuer JEDEN isolierten Knoten (W_ii ist immer 0, unabhaengig von
    der D^-1/2-Konvention bei Grad 0) - der isolierte Knoten traegt also einen
    Eigenwert von 1 bei, nicht 0. Das Theorem "Vielfachheit von Eigenwert 0 = Anzahl
    Komponenten" gilt fuer den normalisierten Laplace deshalb nur unter der (in der
    Literatur ueblichen) Annahme, dass jeder Knoten Grad > 0 hat - in der eigentlichen
    App nie ein Problem, da der KNN-Graph mit "ODER"-Symmetrisierung ab n_neighbors>=1
    keine isolierten Punkte erzeugt."""
    similarity = np.zeros((3, 3))
    similarity[0, 1] = similarity[1, 0] = 0.7  # Punkt 2 bleibt isoliert
    laplacian = normalized_laplacian(similarity)
    eigenvalues, _ = spectral_embedding(laplacian, k=3)
    assert sorted(round(v, 6) for v in eigenvalues) == [0.0, 1.0, 2.0]


def test_laplacian_eigenvalues_are_within_theoretical_bounds():
    """Die Eigenwerte des symmetrisch normalisierten Laplace liegen immer im Intervall
    [0, 2] - eine bekannte Eigenschaft, hier ueber mehrere Zufallsinstanzen geprueft."""
    for seed in range(1, 6):
        instance = generate_instance(60, 3, 0.2, "blobs", seed=seed)
        similarity = build_similarity_graph(instance.as_array(), n_neighbors=8)
        laplacian = normalized_laplacian(similarity)
        eigenvalues, _ = spectral_embedding(laplacian, k=3)
        assert min(eigenvalues) > -1e-9
        assert max(eigenvalues) < 2.0 + 1e-9


def test_embedding_rows_have_unit_norm():
    instance = generate_instance(60, 3, 0.2, "blobs", seed=1)
    similarity = build_similarity_graph(instance.as_array(), n_neighbors=8)
    laplacian = normalized_laplacian(similarity)
    _, embedding = spectral_embedding(laplacian, k=3)
    norms = np.linalg.norm(embedding, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-6)


def test_similarity_graph_is_symmetric_nonnegative_zero_diagonal():
    instance = generate_instance(60, 3, 0.2, "blobs", seed=2)
    similarity = build_similarity_graph(instance.as_array(), n_neighbors=8)
    assert np.allclose(similarity, similarity.T)
    assert (similarity >= 0).all()
    assert np.allclose(np.diag(similarity), 0.0)


def _agreement_fraction(labels_a, labels_b):
    """Anteil der Punktpaare, bei denen beide Partitionen uebereinstimmen - toleranz-
    basierter Kreuzvergleich statt exaktem Zahlen-Match, da Eigensolver und k-Means-Init
    zwischen den Implementierungen variieren koennen (analog zu hdbscan-demos sklearn-
    Kreuzvergleich)."""
    a, b = np.asarray(labels_a), np.asarray(labels_b)
    n = len(a)
    iu = np.triu_indices(n, k=1)
    same_a = (a[:, None] == a[None, :])[iu]
    same_b = (b[:, None] == b[None, :])[iu]
    return float(np.mean(same_a == same_b))


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_matches_sklearn_spectral_clustering_partition(seed):
    instance = generate_instance(120, 2, 0.08, "moons", seed=seed)
    data = instance.as_array()

    result = run(data, k=2, n_neighbors=10, seed=1)

    sk_model = SpectralClustering(
        n_clusters=2, affinity="nearest_neighbors", n_neighbors=10, random_state=1,
        assign_labels="kmeans",
    )
    sk_labels = sk_model.fit_predict(data)

    agreement = _agreement_fraction(result.labels, sk_labels)
    assert agreement > 0.85
