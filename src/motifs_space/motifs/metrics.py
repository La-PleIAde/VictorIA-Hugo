import numpy as np
import pandas as pd
from scipy.sparse import find, triu

from motifs.config import LOGGER
from motifs.features import build_cooccurrence_matrix


def find_top_n_cooccurrence(
    data: pd.DataFrame, n: int, by: str = "window"
) -> pd.DataFrame:
    """
    Return the top n cooccurrent tokens in the data in a bottom up fashion.

    :param data: DataFrame with columns ["token", by]
    :param n: Maximum number of cooccurrence to return
    :param by: Name of the variable defining the window on which the
    cooccurrence is computed
    :return: a DataFrame with columns ["token1", "token2", "count"]
    """
    cooc, _, cols, _, _ = build_cooccurrence_matrix(data, by=by)
    # Extract non-zeros elements from the upper triangular matrix
    cooc = triu(cooc, k=0)
    r, c, v = find(cooc)
    # order by cooccurrence
    order_ = np.argsort(v)
    # get the tokens
    non_zeros = np.array([cols[r][order_], cols[c][order_], v[order_]])
    return pd.DataFrame(
        non_zeros[:, -n:], index=["token1", "token2", "count"]
    ).T


def corpus_top_n_cooccurence(
    data: pd.DataFrame, n: int, by: str = "window"
) -> pd.DataFrame:
    """
    Wrapper function of find_top_n_cooccurrence where data contains a corpus of
     text. Apply find_top_n_cooccurrence for each document in the data.

    :param data: DataFrame with columns ["token", "doc", by]
    :param n:
    :param by:
    :return:
    """
    cooc = pd.DataFrame()
    for doc in data.doc.unique():
        LOGGER.debug(f"Build cooccurrence matrix for {doc}...")
        temp = data.loc[data["doc"] == doc, ["token", by]]
        temp = find_top_n_cooccurrence(temp, n, by=by)

        temp["token"] = temp.apply(
            lambda x: '"' + x["token1"] + '" ' + '"' + x["token2"] + '"',
            axis=1,
        )

        temp["doc"] = doc
        cooc = pd.concat([cooc, temp], ignore_index=True)

    return cooc


def find_cooccurrent_tokens(
    token: str, data: pd.DataFrame, n: int, by: str = "window"
) -> pd.DataFrame:
    """
    Returns the top n cooccurrent tokens of the input token in a top down
    fashion.
    :param token:
    :param data: DataFrame with columns ["token", by]
    :param n: Maximum number of cooccurrence to return
    :param by: Name of the variable defining the window on which the
    cooccurrence is computed
    :return: a DataFrame with columns ["token", "count"]
    """
    cooc, _, cols, _, _ = build_cooccurrence_matrix(data, by=by)
    token_id_mapper = {c: i for i, c in enumerate(cols)}

    # Extract upper triangular matrix
    cooc = triu(cooc, k=0)

    if token in token_id_mapper:
        r, c, v = find(cooc.getrow(token_id_mapper[token]))
        # order by cooccurrence
        order_ = np.argsort(v)
        # get the tokens
        non_zeros = np.array([cols[c][order_], v[order_]])
        cooc = pd.DataFrame(non_zeros[:, -n:], index=["token", "count"]).T
        cooc["target"] = token
        return cooc[["target", "token", "count"]]
    else:
        return pd.DataFrame(columns=["target", "token", "count"])


def corpus_cooccurrent_tokens(
    token: str, data: pd.DataFrame, n: int, by: str = "window"
) -> pd.DataFrame:
    """
    Wrapper function of find_cooccurrent_tokens where data contains a corpus of
     text. Apply find_cooccurrent_tokens for each document in the data.

    :param token:
    :param data:
    :param n:
    :param by:
    :return:
    """
    cooccurent_tokens = pd.DataFrame()
    for doc in data.doc.unique():
        t = find_cooccurrent_tokens(token, data[data["doc"] == doc], n, by=by)
        t["doc"] = doc
        cooccurent_tokens = pd.concat(
            [cooccurent_tokens, t], ignore_index=True
        )

    return cooccurent_tokens
