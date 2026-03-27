import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def run_pca(features: pd.DataFrame, n_components: int = 2) -> pd.DataFrame:
    """Scale features and project to principal components."""
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(features)
    pca = PCA(n_components=n_components, random_state=42)
    components = pca.fit_transform(x_scaled)
    cols = [f"pc{i+1}" for i in range(n_components)]
    return pd.DataFrame(components, columns=cols, index=features.index)


def run_kmeans(features: pd.DataFrame, n_clusters: int = 4) -> pd.Series:
    """Fit KMeans and return cluster labels."""
    model = KMeans(n_clusters=n_clusters, random_state=42, n_init="auto")
    labels = model.fit_predict(features)
    return pd.Series(labels, index=features.index, name="cluster")
