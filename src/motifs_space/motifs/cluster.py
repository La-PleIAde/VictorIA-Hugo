import os.path
from typing import Optional, Union

import numpy as np
from joblib import dump, load
from matplotlib import pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score

from motifs.constants import DEFAULT_CLUSTER_PARAMS, DEFAULT_PROJECTION_PARAMS


def create_config_params(
    projection: Union[str, dict], cluster: Union[str, dict]
) -> dict:
    config = {}

    if projection is not None:
        if isinstance(projection, dict):
            projection_name = projection["name"]
            params = projection.get(
                "params", DEFAULT_PROJECTION_PARAMS[projection_name]
            )
        elif isinstance(projection, str):
            projection_name = projection
            params = DEFAULT_PROJECTION_PARAMS[projection_name]
        else:
            raise NotImplementedError(type(projection))
        config["projection"] = {"name": projection_name, "params": params}

    if isinstance(cluster, dict):
        cluster_name = cluster["name"]
        params = cluster.get("params", DEFAULT_CLUSTER_PARAMS[cluster_name])
    elif isinstance(cluster, str):
        cluster_name = cluster
        params = DEFAULT_CLUSTER_PARAMS[cluster_name]
    else:
        raise NotImplementedError(type(cluster))
    config["cluster"] = {"name": cluster_name, "params": params}
    return config


class Cluster:
    """
    A clustering pipeline applied to the data matrix X

    :param cluster: name of the clustering method
    :param embedding: name
    :param load_from_dir: Path to a previously fitted object. If provided then
    the estimators are loaded from the directory

    :Example:

    Cluster the Iris dataset

    >>> from motifs.cluster import Cluster
    >>> from sklearn import datasets

    >>> iris = datasets.load_iris()
    >>> cluster_params = {
    >>>     "name": "DBSCAN",
    >>>     "params": {"eps": 15, "min_samples": 10, "metric": "euclidean"}
    >>> }
    >>> cluster = Cluster(
    >>>     projection="TSNE",
    >>>     cluster=cluster_params,
    >>> )
    >>> cluster_labels = cluster.fit_predict(iris.data)
    >>> print(cluster_labels)
    >>> cluster.plot()

    """

    def __init__(
        self,
        projection: Union[str, dict] = "TSNE",
        cluster: Union[str, dict] = "DBSCAN",
    ):
        self.config = create_config_params(projection, cluster)
        projection = self.config.get("projection")
        if projection is None:
            self.projector = None
        else:
            if projection["name"] == "TSNE":
                self.projector = TSNE(**projection["params"])
            else:
                raise NotImplementedError(projection)

        cluster = self.config.get("cluster")
        if cluster["name"] == "DBSCAN":
            self.cluster = DBSCAN(**cluster["params"])
        else:
            raise NotImplementedError(projection)
        self.embedding = None
        self.is_fitted = False

    def fit_predict(self, X: np.ndarray):
        """

        :param X: Data array of shape (n_samples, n_features)
        :return:
        """
        self.is_fitted = True
        if self.projector is not None:
            self.embedding = self.projector.fit_transform(X)
            return self.cluster.fit_predict(self.embedding)
        else:
            return self.cluster.fit_predict(X)

    def plot(self, X: Optional[np.ndarray] = None):
        """

        :param X:
        :return:
        """
        assert self.is_fitted, "You must fit the clusters first!"
        if self.embedding is None:
            if X is None:
                raise AssertionError(
                    "You must give X if projector is not " "given!"
                )
            pca = PCA(n_components=2)
            embedding = pca.fit_transform(X)
            projection = "PCA"
        else:
            projection = self.config["projection"]["name"]
            embedding = self.embedding

        cluster_labels = self.cluster.labels_
        n = len(cluster_labels)
        clustered_points = np.array(range(n))[cluster_labels != -1]
        if len(set(cluster_labels[clustered_points])) > 1:
            silhouette_avg = silhouette_score(
                embedding[clustered_points, :],
                cluster_labels[clustered_points],
                metric="euclidean",
            )
        else:
            silhouette_avg = np.nan

        fig, axs = plt.subplots(1, 2, figsize=(10, 5))
        axs[0].scatter(x=embedding[:, 0], y=embedding[:, 1], alpha=0.5)
        axs[0].set_title(f"{projection} embedding space")
        axs[1].scatter(
            x=embedding[cluster_labels != -1, 0],
            y=embedding[cluster_labels != -1, 1],
            c=cluster_labels[cluster_labels != -1],
            cmap="tab20",
            alpha=0.5,
        )
        axs[1].scatter(
            x=embedding[cluster_labels == -1, 0],
            y=embedding[cluster_labels == -1, 1],
            marker="x",
            c="black",
            alpha=0.5,
        )
        axs[1].set_title(
            "Clusters on the embedding space\nSilhouette score = "
            f"{silhouette_avg:.2f}"
        )
        plt.show()

    def save(self, path: str):
        if os.path.isfile(path):
            raise ValueError(
                f"The provided path {path} already exsits! "
                "Choose a different one"
            )
        dump(self, path)

    @staticmethod
    def load(path):
        return load(path)
