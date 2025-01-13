import numpy as np


def scaled_cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """
    Computes the cosine similarity between two vectors and scales the result to [0, 1].

    Parameters:
    ----------
    vec1 : np.ndarray
        The first input vector.
    vec2 : np.ndarray
        The second input vector.

    Returns:
    -------
    float
        The scaled cosine similarity between the two vectors in the range [0, 1].

    Raises:
    ------
    ValueError
        If the vectors do not have the same shape.

    Example:
    --------
    >>> v1 = np.array([1, 2, 3])
    >>> v2 = np.array([4, 5, 6])
    >>> scaled_cosine_similarity(v1, v2)
    0.975
    """
    if vec1.shape != vec2.shape:
        raise ValueError("Vectors must have the same shape.")

    # Compute the dot product and magnitudes of the vectors
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        raise ValueError("Vectors must not be zero vectors.")

    # Compute cosine similarity
    cosine_sim = dot_product / (norm1 * norm2)

    # Scale cosine similarity from [-1, 1] to [0, 1]
    scaled_similarity = (cosine_sim + 1) / 2

    return scaled_similarity


def scaled_cosine_similarity_complement(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """
    Computes the complement of the cosine similarity between two vectors
    and scales the result to [0, 1].

    Parameters:
    ----------
    vec1 : np.ndarray
        The first input vector.
    vec2 : np.ndarray
        The second input vector.

    Returns:
    -------
    float
        The complement of the scaled cosine similarity between the two vectors,
        in the range [0, 1].

    Raises:
    ------
    ValueError
        If the vectors do not have the same shape.

    Example:
    --------
    >>> v1 = np.array([1, 2, 3])
    >>> v2 = np.array([4, 5, 6])
    >>> scaled_cosine_similarity_complement(v1, v2)
    0.025
    """
    # Compute the scaled cosine similarity
    similarity = scaled_cosine_similarity(vec1, vec2)

    # Compute the complement
    complement = 1 - similarity

    return complement
