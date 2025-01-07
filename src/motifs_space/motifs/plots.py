import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_samples, silhouette_score

from motifs.config import LOGGER

PLOT_TYPES = ["group", "sep"]


def plot_tf_idf(
    tf_idf: pd.DataFrame,
    n_tokens: int = 20,
    plot_type: str = "sep",
    col_wrap: int = 3,
):
    if plot_type not in PLOT_TYPES:
        LOGGER.error(
            f"{plot_type} is not implemented. For now you can use "
            f"only {PLOT_TYPES}"
        )
        raise NotImplementedError

    temp = tf_idf.copy()
    temp = temp.groupby("doc").head(n_tokens)
    temp.sort_values(by="tfidf", ascending=False, inplace=True)

    if plot_type == "sep":
        g = sns.FacetGrid(
            temp,
            col="doc",
            sharey=False,
            col_wrap=col_wrap,
        )
        g.map(sns.barplot, "tfidf", "token")
        g.set_titles("{col_name}")
    elif plot_type == "group":
        g = sns.barplot(temp, y="token", x="tfidf", hue="doc")
    g.set(xlabel=None, ylabel=None)
    plt.show()


def pca_variable_plot_old(pca: PCA):
    t = np.linspace(0, np.pi * 2, 100)
    fig, ax = plt.subplots(1, 1)

    corr_ = pd.DataFrame(
        [
            [
                np.corrcoef(pca.transformed_data[:, i], pca.factors["comp_0"])[
                    0, 1
                ]
                for i in range(3)
            ],
            [
                np.corrcoef(pca.transformed_data[:, i], pca.factors["comp_1"])[
                    0, 1
                ]
                for i in range(3)
            ],
        ],
        columns=pca.loadings.index,
        index=["comp_0", "comp_1"],
    ).T

    for i in range(3):
        ax.annotate(
            corr_.index[i],
            xy=(corr_["comp_0"].values[i], corr_["comp_1"].values[i]),
            xytext=(corr_["comp_0"].values[i], corr_["comp_1"].values[i]),
        )

    for i in range(3):
        ax.annotate(
            "",
            xy=(corr_["comp_0"].values[i], corr_["comp_1"].values[i]),
            xytext=(0, 0),
            arrowprops=dict(arrowstyle="->"),
        )

    ax.plot(np.cos(t), np.sin(t), linewidth=1, c="black")
    # ax.set_box_aspect(1)
    ax.set_aspect("equal")
    ax.grid(True, which="both")
    ax.axhline(y=0, color="k")
    ax.axvline(x=0, color="k")
    plt.show()
    # plt.close()


def facet_bar_plot(
    data: pd.DataFrame,
    plot_type: str = "sep",
    col_wrap: int = 2,
    heigth: int = 4,
    y="token",
    x="count",
    ylabel="token",
    xlabel="count",
):
    """
    Helper function to create a barplot using Seaborn.

    :param data: DataFrame of ngrams tokens with columns: [x, y, "doc"]
    where x has type supported by sns.barplot such as float.
    :param plot_type: "group" or "sep"
    :param col_wrap:
    :param heigth:
    :param y: Variable on the y axis
    :param x: Variables on the x axis
    :param ylabel:
    :param xlabel:
    :return:
    """

    if plot_type not in PLOT_TYPES:
        LOGGER.error(
            f"{plot_type} is not implemented. For now you can use "
            f"only {PLOT_TYPES}"
        )
        raise NotImplementedError

    if plot_type == "sep":
        g = sns.FacetGrid(
            data,
            col="doc",
            sharey=False,
            col_wrap=col_wrap,
            height=heigth,
            aspect=5 / heigth,
        )
        g.map(sns.barplot, x, y)
        g.set_titles("{col_name}")
    elif plot_type == "group":
        plt.figure(figsize=(5, len(data) / 5))
        g = sns.barplot(data, x="count", y="token", hue="doc")
    g.set(xlabel=xlabel, ylabel=ylabel)
    plt.show()


def plot_motif_histogram(
    ngrams: pd.DataFrame,
    stat: str = "count",
    n_tokens: int = 15,
    plot_type: str = "group",
):
    """
    Create a histogram plot of the n-grams based on different statistics such
    as "count" (default), "proportion", or "percent". The n_tokens
    corresponds to the maximum number of tokens to represent on
    the graph.

    :param ngrams: DataFrame of ngrams tokens with columns: ["token", "doc"]
    :param stat: One of ["count", "proportion", "percent"]
    :param n_tokens: Max number of tokens to plot
    :param plot_type: "group" or "sep"
    :return:
    """
    freq = ngrams.groupby("doc").token.value_counts()
    if stat == "proportion" or stat == "percent":
        freq = freq / freq.groupby("doc").sum()
        if stat == "percent":
            freq *= 100
    freq = (
        freq.sort_values(ascending=False)
        .groupby("doc")
        .head(n_tokens)
        .reset_index()
    )

    facet_bar_plot(freq, plot_type=plot_type, xlabel=stat)


def plot_clusters(X, cluster_labels, ax=None, title=None):
    n_clusters = len(set(cluster_labels))
    noise = cluster_labels == -1
    # 2nd Plot showing the actual clusters formed
    colors = cm.nipy_spectral(cluster_labels.astype(float) / n_clusters)
    if ax is None:
        ax = plt
    if sum(noise) > 0:
        ax.scatter(
            X[~noise, 0],
            X[~noise, 1],
            lw=0,
            alpha=0.7,
            c=colors[~noise],
            edgecolor="k",
        )
        ax.scatter(
            X[noise, 0],
            X[noise, 1],
            marker="x",
            alpha=0.5,
            s=30,
            c="grey",
        )
    else:
        ax.scatter(
            X[:, 0], X[:, 1], lw=0, alpha=0.7, c=colors[~noise], edgecolor="k"
        )

    # Labeling the clusters
    centers = np.array(
        [
            np.mean(X[cluster_labels == i, :], axis=0)
            for i in set(cluster_labels)
            if i != -1
        ]
    )
    # Draw white circles at cluster centers
    ax.scatter(
        centers[:, 0],
        centers[:, 1],
        marker="o",
        c="white",
        alpha=1,
        s=200,
        edgecolor="k",
    )

    for i, c in enumerate(centers):
        label = list(set(cluster_labels))[i]
        if label != -1:
            ax.scatter(
                c[0],
                c[1],
                marker="$%d$" % label,
                alpha=1,
                s=50,
                edgecolor="k",
            )
    if title is not None:
        ax.set_title(title)


def silhouette_plot(X, cluster_labels, metric: str = "euclidean"):
    """
    cf: https://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_silhouette_analysis.html
    :param X:
    :param cluster_labels:
    :return:
    """
    if -1 in cluster_labels:
        n_clusters = len(set(cluster_labels)) - 1
    else:
        n_clusters = len(set(cluster_labels))

    # Create a subplot with 1 row and 2 columns
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.set_size_inches(18, 7)

    # The 1st subplot is the silhouette plot
    # The (n_clusters+1)*10 is for inserting blank space between silhouette
    # plots of individual clusters, to demarcate them clearly.
    ax1.set_ylim([0, len(X) + (n_clusters + 1) * 10])

    # Compute the silhouette scores for each sample
    sample_silhouette_values = silhouette_samples(X, cluster_labels)
    y_lower = 10
    for i in set(cluster_labels):
        if i != -1:
            # Aggregate the silhouette scores for samples belonging to
            # cluster i, and sort them
            ith_cluster_silhouette_values = sample_silhouette_values[
                cluster_labels == i
            ]

            ith_cluster_silhouette_values.sort()

            size_cluster_i = ith_cluster_silhouette_values.shape[0]
            y_upper = y_lower + size_cluster_i

            color = cm.nipy_spectral(float(i) / n_clusters)
            ax1.fill_betweenx(
                np.arange(y_lower, y_upper),
                0,
                ith_cluster_silhouette_values,
                facecolor=color,
                edgecolor=color,
                alpha=0.7,
            )
            # Label the silhouette plots with their cluster numbers at the
            # middle
            ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
            # Compute the new y_lower for next plot
            y_lower = y_upper + 10

    ax1.set_title("Silhouette plot for the various clusters.")
    ax1.set_xlabel("Silhouette coefficient values")
    ax1.set_ylabel("Cluster label")

    # The vertical line for average silhouette score of all the values
    noise = cluster_labels == -1
    silhouette_avg = silhouette_score(
        X[~noise, :], cluster_labels[~noise], metric=metric
    )
    ax1.axvline(x=silhouette_avg, color="red", linestyle="--")
    ax1.set_yticks([])  # Clear the yaxis labels / ticks
    # ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

    # 2nd Plot showing the actual clusters formed
    plot_clusters(X, cluster_labels, ax=ax2, title="Clustered Data")
    plt.suptitle(
        "Silhouette analysis for DBSCAN clustering on sample data with "
        "n_clusters = %d" % n_clusters,
        fontsize=14,
        fontweight="bold",
    )
    plt.show()

    return sample_silhouette_values


def scatter_plot_one_cluster(
    embedding: np.ndarray,
    cluster_labels: np.ndarray,
    label: int,
    feature: pd.Series,
    figsize=(15, 5),
):
    """
    Scatter plot of the embedding with multiple colors:
    - colored individuals are the one belonging to the identified cluster
    with label=`label` and colors matching the given categories in
    the given `feature`
    - individuals that do not belong to the cluster are not colored

    :param embedding:
    :param cluster_labels:
    :param label:
    :param feature:
    :param figsize:
    :return:
    """
    authors_cat_mapping = dict(enumerate(feature.cat.categories))

    assert isinstance(feature, pd.Series)
    assert type(feature.dtype) is pd.CategoricalDtype
    cluster_feature = feature.loc[cluster_labels == label]
    features_cat = cluster_feature.cat.categories[
        cluster_feature.cat.categories.isin(cluster_feature)
    ]

    fig, axs = plt.subplots(1, 3, figsize=figsize)
    axs[0].scatter(
        embedding[:, 0],
        embedding[:, 1],
        c=["red" if i else "grey" for i in cluster_labels == label],
        alpha=0.5,
    )
    axs[0].set_title("Feature space")
    axs[1].scatter(
        embedding[cluster_labels != label, 0],
        embedding[cluster_labels != label, 1],
        c="grey",
        alpha=0.5,
    )
    scatter = axs[1].scatter(
        embedding[cluster_labels == label][:, 0],
        embedding[cluster_labels == label][:, 1],
        c=cluster_feature.cat.codes,
        alpha=1,
        cmap=None if len(cluster_feature.cat.codes.unique()) < 10 else "tab20",
    )
    legend = axs[1].legend(
        scatter.legend_elements()[0],
        [
            a.split(",")[0]
            for a in [
                authors_cat_mapping[i]
                for i in sorted(cluster_feature.cat.codes.unique())
            ]
        ],
        title="Author",
    )
    axs[1].add_artist(legend)
    axs[1].set_title("Authors in the feature space within cluster %d" % label)

    bars = cluster_feature.value_counts()
    bars = bars[bars.index.isin(features_cat)]
    axs[2].bar(features_cat, bars, color="grey")
    axs[2].set_xticks(
        range(len(features_cat)),
        [a.split(",")[0] for a in features_cat],
        rotation=90,
    )
    axs[2].set_title("Author frequency within cluster %d" % label)
    plt.show()
