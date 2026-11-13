import numpy as np
import pandas as pd

from src.supervised import train_logistic_classifier
from src.unsupervised import run_kmeans, run_pca


def make_separable_data(n: int = 200, seed: int = 0):
    """Two well-separated groups, so a working classifier should score near 100%."""
    rng = np.random.default_rng(seed)
    low = rng.normal(loc=0.0, scale=0.5, size=(n // 2, 3))
    high = rng.normal(loc=5.0, scale=0.5, size=(n // 2, 3))
    x = pd.DataFrame(np.vstack([low, high]), columns=["f1", "f2", "f3"])
    y = pd.Series([0] * (n // 2) + [1] * (n // 2), name="y")
    return x, y


def test_logistic_classifier_learns_separable_data():
    x, y = make_separable_data()

    result = train_logistic_classifier(x, y)

    assert result["accuracy"] >= 0.95
    assert {"model", "accuracy", "report"} <= set(result)


def test_logistic_classifier_is_reproducible():
    x, y = make_separable_data()

    first = train_logistic_classifier(x, y, random_state=7)
    second = train_logistic_classifier(x, y, random_state=7)

    assert first["accuracy"] == second["accuracy"]
    assert np.allclose(first["model"].coef_, second["model"].coef_)


def test_pca_returns_requested_components_on_same_index():
    x, _ = make_separable_data()
    x.index = [f"area_{i}" for i in range(len(x))]

    components = run_pca(x, n_components=2)

    assert list(components.columns) == ["pc1", "pc2"]
    assert components.index.equals(x.index)


def test_kmeans_finds_the_two_groups():
    x, y = make_separable_data()

    labels = run_kmeans(x, n_clusters=2)

    assert labels.name == "cluster"
    assert labels.nunique() == 2
    # Each true group should land entirely in one cluster.
    assert labels[y == 0].nunique() == 1
    assert labels[y == 1].nunique() == 1


def test_kmeans_is_deterministic():
    x, _ = make_separable_data()

    assert run_kmeans(x, n_clusters=3).equals(run_kmeans(x, n_clusters=3))
