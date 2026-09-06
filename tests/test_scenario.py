from sp_scenario import generate_instance


def test_reproducible_with_same_seed():
    a = generate_instance(100, 3, 0.2, "blobs", seed=7)
    b = generate_instance(100, 3, 0.2, "blobs", seed=7)
    assert a.points == b.points
    assert a.true_labels == b.true_labels


def test_different_seed_gives_different_points():
    a = generate_instance(100, 3, 0.2, "blobs", seed=1)
    b = generate_instance(100, 3, 0.2, "blobs", seed=2)
    assert a.points != b.points


def test_n_points_and_k_respected_for_blobs():
    instance = generate_instance(97, 4, 0.2, "blobs", seed=5)
    assert instance.n_points == 97
    assert instance.k == 4
    assert set(instance.true_labels) == {0, 1, 2, 3}


def test_moons_k_equals_two_produces_two_balanced_groups():
    instance = generate_instance(100, 2, 0.08, "moons", seed=1)
    labels = instance.true_labels
    assert set(labels) == {0, 1}
    assert abs(labels.count(0) - labels.count(1)) <= 1


def test_moons_k_greater_than_two_produces_k_balanced_arcs():
    """k muss bei BEIDEN Formen wirken - die dbscan-demo-Lehre, hier von Anfang an."""
    instance = generate_instance(200, 4, 0.08, "moons", seed=2)
    labels = instance.true_labels
    assert set(labels) == {0, 1, 2, 3}
    counts = [labels.count(i) for i in range(4)]
    assert max(counts) - min(counts) <= 1
