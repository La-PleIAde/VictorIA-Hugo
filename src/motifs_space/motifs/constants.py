AVAILABLE_TOKEN_TYPES = ["text", "lemma", "pos", "motif"]
AVAILABLE_FEATURES = ["tfidf", "freq"]
AVAILABLE_METHODS = ["pca"]

DEFAULT_PROJECTION_PARAMS = {
    "TSNE": {
        "n_components": 2,
        "random_state": 42,
        "init": "random",
        "perplexity": 5,
    }
}
DEFAULT_CLUSTER_PARAMS = {
    "DBSCAN": {"eps": 4, "min_samples": 5, "metric": "euclidean"}
}
