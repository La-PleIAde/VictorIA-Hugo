import os
from typing import List, Optional

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from motifs.config import LOGGER


def load_tokens_from_directory(dir_: str, docs: List[str] = None):
    if docs is None:
        docs = os.listdir(dir_)
    files = [f for f in os.listdir(dir_) if f.endswith(".csv") and f in docs]

    def worker(f):
        path = f"{dir_}/{f}"
        try:
            return pd.read_csv(path)
        except Exception as _exc:
            LOGGER.error(f"Could not load file {path}")
            LOGGER.exception(_exc)

    tokens = Parallel(n_jobs=os.cpu_count(), prefer="threads")(
        delayed(worker)(f) for f in files
    )
    tokens = pd.concat(tokens, ignore_index=True)
    return tokens


def build_window_corpus(
    ngrams: pd.DataFrame,
    seq_length: int,
    overlap: bool = True,
    n: Optional[int] = None,
) -> pd.DataFrame:
    """

    :param ngrams: DataFrame of n-grams with columns ["doc", "text", "token"]
    :param seq_length: Length of the window
    :param overlap: If true, it returns all n-grams contained in the
    window, otherwise it returns non-overlapping n-grams only.
    context
    :param n: n-gram length, required if `overlap=False`
    :return:
    """
    if not overlap:
        assert n is not None, "You must provide n if overlap is False!"
    window_ngram = pd.DataFrame()
    for doc in ngrams.doc.unique():
        temp = ngrams[ngrams["doc"] == doc]
        if not overlap:
            temp = temp.iloc[list(range(0, len(temp), n))]
        window = build_window_data(temp, seq_length)
        window["doc"] = doc
        window_ngram = pd.concat([window_ngram, window], ignore_index=True)

    return window_ngram


def build_window_data(ngrams: pd.DataFrame, seq_length: int):
    p = ngrams[["text", "token"]].values
    window = np.zeros((p.shape[0], 3), dtype=object)
    for i in range(0, len(ngrams) - 1, seq_length):
        window[i : i + seq_length, 0] = str(i)
        window[i : i + seq_length, 1] = p[i : i + seq_length, 0]
        window[i : i + seq_length, 2] = p[i : i + seq_length, 1]

    window = pd.DataFrame(window, columns=["window", "text", "token"])
    window["window"] = window["window"].astype(int)

    # Remove extra 0s
    if len(ngrams) % seq_length != 0:
        window = window.iloc[: -(len(ngrams) % seq_length), :]

    return window


def return_to_text_from_token_better_try(
    ngrams: pd.DataFrame, token: str, n: int, context_len: int
) -> pd.DataFrame:
    ngrams = ngrams.reset_index()
    words = ngrams["word"].values
    ids = ngrams.index[ngrams["token"] == token]

    start_of_text = [id_ for id_ in ids if id_ - context_len < 0]
    end_of_text = [id_ for id_ in ids if id_ + n + context_len >= len(words)]
    ids = list(set(ids) - set(start_of_text) - set(end_of_text))

    l_c = words[[range(id_ - context_len, id_) for id_ in ids]]
    l_c = np.apply_along_axis(lambda x: " ".join(x), 1, l_c)
    r_c = words[[range(id_ + n, id_ + n + context_len) for id_ in ids]]
    r_c = np.apply_along_axis(lambda x: " ".join(x), 1, r_c)

    context = pd.DataFrame(
        [l_c, ngrams["text"].values[ids], r_c], columns=ids
    ).T
    no_left_c = []
    for id_ in start_of_text:
        no_left_c.append(
            [
                " ".join(words[0:id_]),
                ngrams["text"].values[id_],
                " ".join(words[range(id_ + n, id_ + n + context_len)]),
            ]
        )
    if len(no_left_c):
        context = pd.concat(
            [context, pd.DataFrame(no_left_c, index=start_of_text)],
            ignore_index=True,
        )

    no_right_c = []
    for id_ in end_of_text:
        no_right_c.append(
            [
                " ".join(words[range(id_ - context_len, id_)]),
                ngrams["text"].values[id_],
                None,
            ]
        )
    if len(no_right_c):
        context = pd.concat(
            [context, pd.DataFrame(no_right_c, index=end_of_text)],
            ignore_index=True,
        )
    context["token"] = token
    context["doc"] = ngrams.loc[context.index, "doc"].values

    return context


def return_to_text_from_token(
    ngrams: pd.DataFrame, token: str, n: int, context_len: int
) -> pd.DataFrame:
    """
    From a token, such as a motif, returns the left, right context and
    text corresponding to the input token within the original text.

    :param ngrams: a DataFrame containing the original text and
    corresponding tokens in n-grams with columns ["token", "text", "doc"]
    :param token: a string representing the token
    :param n: the n-gram length
    :param context_len: the context length (left and right)
    :return: a DataFrame with columns ["left_context", "righ_context",
    "doc"] and, as index, the different texts corresponding to the token
    """
    l_context = []
    r_context = []
    token_text = []

    ids = ngrams.index[ngrams["token"] == token]
    for id_ in ids:
        token_text.append(ngrams["text"].loc[id_])
        l_c = ngrams["word"].loc[id_ - context_len : id_ - 1]
        l_c = " ".join(l_c.tolist())
        r_c = ngrams["word"].loc[id_ + n : id_ + n + context_len - 1]
        r_c = " ".join(r_c.tolist())

        l_context.append(l_c)
        r_context.append(r_c)

    context = pd.DataFrame(l_context, columns=["left_context"])
    context["text"] = token_text
    context["right_context"] = r_context
    context["doc"] = ngrams.loc[ids, "doc"].values
    context["token"] = token

    return context


def return_to_text_from_spec(
    ngrams: pd.DataFrame,
    spec: str,
    n: int,
    context_len: int,
    min_spec: int,
    min_freq: int = 2,
) -> pd.DataFrame:
    """

    :param ngrams: a DataFrame containing the original text and
    corresponding tokens in n-grams with columns ["token", "text", "doc"]
    :param spec:
    :param n: the n-gram length
    :param context_len: the context length (left and right)
    :param min_spec:
    :param min_freq:

    :return: a DataFrame with columns ["left_context", "righ_context",
    "doc"] and, as index, the different texts corresponding to the token
    """

    output = pd.DataFrame()
    for doc in list(spec.columns):
        s = spec.loc[spec["doc"] == doc, :]
        tokens = s.loc[(s[doc] >= min_spec) & (s["f"] >= min_freq), doc].index
        if len(tokens):
            temp = pd.concat(
                [
                    return_to_text_from_token(
                        ngrams[ngrams["doc"] == doc], token, n, context_len
                    )
                    for token in tokens
                ],
                ignore_index=True,
            )
            output = pd.concat([output, temp], ignore_index=True)
    if len(output):
        output = (
            output.set_index("token")
            .join(spec[["spec", "f", "t"]])
            .sort_values(by=["spec", "doc"], ascending=False)
        )
    return output


def filter_token_by_freq(
    tokens: pd.DataFrame, f: int, groupby: Optional[str] = None
) -> pd.DataFrame:
    """
    Filter out tokens that appear only $f$ times. If groupby is give,
    then the count is computed at the groupby level.

    :param tokens: a DataFrame with column "token" and `groupby` if given
    :param f: the frequency
    :param groupby: Level at which to compute the frequency
    :return:
    """
    if groupby:
        freq = tokens.groupby(groupby)["token"].value_counts()
    else:
        freq = tokens["token"].value_counts()
    if groupby:
        index_col = [groupby, "token"]
    else:
        index_col = "token"
    tokens = (
        tokens.set_index(index_col).loc[freq[freq > f].index].reset_index()
    )
    return tokens
