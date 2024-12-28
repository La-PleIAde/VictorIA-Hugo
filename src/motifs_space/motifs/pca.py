import itertools
import math

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from motifs.config import LOGGER


def pca_transform(data, plot: bool = False):
    """

    :param features_data:
    :return:
    """

    # Normalize the input data
    docs = data.columns
    scaler = StandardScaler()
    data = scaler.fit_transform(data)
    data = pd.DataFrame(data, columns=docs)

    pca = PCA(n_components=data.shape[-1])
    pca.fit(data)

    if plot:
        plot_explained_variance_ratio(pca)
        plot_pca_projection(pca, docs)
        pca_variable_plot(data, pca, colwrap=4)

    return pca


def plot_explained_variance_ratio(pca: PCA):
    plots = sns.barplot(pca.explained_variance_ratio_ * 100)
    plots.set(ylabel="Explained variance ratio", ylim=[0, 100])
    for bar in plots.patches:
        plots.annotate(
            format(bar.get_height(), ".0f") + "%",
            (bar.get_x() + bar.get_width() / 2, bar.get_height()),
            ha="center",
            va="center",
            xytext=(0, 8),
            textcoords="offset points",
        )
    plt.title("PC explained variance ratio")
    plt.show()


def plot_pca_projection(pca, var_names):
    loadings = pd.DataFrame(pca.components_.T, index=var_names)
    sns.heatmap(
        loadings,
        cmap="bwr",
        square=pca.components_.shape[0] == len(var_names),
        vmax=1,
        vmin=-1,
        center=0,
    )
    plt.title("PC loadings")
    plt.show()


def pca_variable_plot(data, pca, n_components=None, colwrap=3, max_plots=50):
    names = list(data.columns)
    if len(names) > 10:
        cm = plt.get_cmap("gist_rainbow")
        colors = [cm(1.0 * i / len(names)) for i in range(len(names))]
    else:
        colors = list(mcolors.TABLEAU_COLORS.values())

    factors = pca.transform(data)
    if n_components:
        factors = factors[:, :n_components]
    pairs = list(itertools.combinations(list(range(factors.shape[-1])), 2))
    if len(pairs) > max_plots:
        LOGGER.error(
            f"Number of plots larger than max_plots={max_plots}. If you "
            f"really want to produce {len(pairs)} plots, then increase "
            f"max_plots. Aborting."
        )
        return
    if len(pairs) >= colwrap:
        ncol = colwrap
    else:
        ncol = len(pairs)
    nrow = math.ceil(len(pairs) / colwrap)

    row = 0
    height = 5
    fig, axs = plt.subplots(nrow, ncol, figsize=(height * ncol, height * nrow))

    if nrow * ncol == 1:
        pca_variable_2dplot(data, factors, ax=axs)
        ax = axs
    else:
        if nrow > 1:
            for i, pair in enumerate(pairs):
                if i % colwrap == 0 and i >= 3:
                    row += 1
                col = i % colwrap
                pca_variable_2dplot(
                    data, factors[:, [pair[0], pair[1]]], ax=axs[row, col]
                )
                axs[row, col].set_title(f"{pair[1]} vs {pair[0]}")

            # Remove extra empty axes
            if (nrow * ncol - len(pairs)) > 0:
                for i in range(colwrap - (nrow * ncol - len(pairs)), colwrap):
                    axs[row, i].set_axis_off()

            ax = axs[0, 0]
        else:
            for col, pair in enumerate(pairs):
                pca_variable_2dplot(
                    data, factors[:, [pair[0], pair[1]]], ax=axs[col]
                )
                axs[col].set_title(f"PC{pair[1]} vs PC{pair[0]}")
            ax = axs[0]

    for i, name in enumerate(names):
        y = -i / height + 1
        ax.text(-5, y, name, fontsize=12, color=colors[i])

    fig.suptitle("PCA Variable plot")
    plt.show()


def pca_variable_2dplot(data: pd.DataFrame, factors, ax):
    assert factors.shape[-1] == 2
    t = np.linspace(0, np.pi * 2, 100)
    corr_ = pd.DataFrame(
        [
            [
                np.corrcoef(data.values[:, i], factors[:, 0])[0, 1]
                for i in range(data.shape[-1])
            ],
            [
                np.corrcoef(data.values[:, i], factors[:, 1])[0, 1]
                for i in range(data.shape[-1])
            ],
        ],
        columns=data.columns,
        index=["comp_0", "comp_1"],
    ).T

    num_vars = data.shape[-1]
    if num_vars > 10:
        cm = plt.get_cmap("gist_rainbow")
        colors = [cm(1.0 * i / num_vars) for i in range(num_vars)]
    else:
        colors = list(mcolors.TABLEAU_COLORS.values())

    for i in range(num_vars):
        ax.annotate(
            "",
            xy=(corr_["comp_0"].values[i], corr_["comp_1"].values[i]),
            xytext=(0, 0),
            arrowprops=dict(arrowstyle="->", color=colors[i]),
        )

    ax.plot(np.cos(t), np.sin(t), linewidth=1, c="black")
    ax.set_aspect("equal")
    ax.grid(True, which="both")
    ax.axhline(y=0, color="k")
    ax.axvline(x=0, color="k")
