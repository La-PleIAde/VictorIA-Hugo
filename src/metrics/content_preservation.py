from typing import Sequence
import numpy as np
from numpy import floating
from src.metrics.vec import scaled_cosine_similarity


def mean_similarity(
        source_embeddings: Sequence[np.ndarray],
        target_embeddings: Sequence[np.ndarray]
) -> floating:
    """
    Computes the mean similarity between two corpora of embeddings.

    Parameters:
    ----------
    source_embeddings : Sequence[np.ndarray]
        The sequence of source embeddings (e.g., from the source author).
    target_embeddings : Sequence[np.ndarray]
        The sequence of target embeddings (e.g., from the target author).

    Returns:
    -------
    floating
        The mean similarity score between the two corpora.

    Raises:
    ------
    ValueError
        If the two corpora do not have the same length.
    """
    if len(source_embeddings) != len(target_embeddings):
        raise ValueError("The two corpora must have the same length.")

    # Calculate the mean similarity score between the two corpora
    similarity_scores = [
        scaled_cosine_similarity(a, b) for a, b in zip(source_embeddings, target_embeddings)
    ]
    return np.mean(similarity_scores)


def compute_content_preservation_score(
        source_embeddings: Sequence[np.ndarray],
        target_embeddings: Sequence[np.ndarray],
        style_transferred_embeddings: Sequence[np.ndarray]
) -> float:
    """
    Computes the content preservation (SIM) score for style-transferred embeddings.

    Parameters:
    ----------
    source_embeddings : Sequence[np.ndarray]
        Sequence of source author's corpus embeddings.
    target_embeddings : Sequence[np.ndarray]
        Sequence of target author's corpus embeddings.
    style_transferred_embeddings : Sequence[np.ndarray]
        Sequence of style-transferred corpus embeddings.

    Returns:
    -------
    float
        The computed SIM score.

    Raises:
    ------
    ValueError
        If the MIS between the target and source is equal to or greater than 1.
    """
    mis_style = mean_similarity(style_transferred_embeddings, source_embeddings)
    mis_target = mean_similarity(target_embeddings, source_embeddings)

    # Ensure no division by zero
    if mis_target == 1:
        raise ValueError("MIS between the target and source must be less than 1 to avoid division by zero.")

    # Calculate the SIM score
    content_preservation_score = max(mis_style - mis_target, 0) / (1 - mis_target)
    return content_preservation_score
