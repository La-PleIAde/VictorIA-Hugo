import numpy as np

from src.metrics.vec import scaled_cosine_similarity as similarity
from src.metrics.vec import scaled_cosine_similarity_complement as dissimilarity


def geometric_mean(*values: float) -> float:
    """
    Computes the geometric mean of a sequence of values.

    Parameters:
    ----------
    *values : float
        A sequence of numeric values passed as individual arguments.
        All values must be non-negative.

    Returns:
    -------
    float
        The geometric mean of the values.

    Raises:
    ------
    ValueError
        If no values are provided or if any value is negative.
    """
    if not values:
        raise ValueError("At least one value must be provided.")

    if any(value < 0 for value in values):
        raise ValueError("All values must be non-negative to compute the geometric mean.")

    # Compute the geometric mean
    product = np.prod(values)
    return product ** (1 / len(values))


def away(Rs, Rt, Rst):
    """
    Away score

    :param Rs: UAR of the source author's corpus
    :param Rt: UAR of the target author's corpus
    :param Rst: UAR of style transferred corpus
    :return: Away score
    """
    return min(dissimilarity(Rst, Rs), dissimilarity(Rt, Rs)) / dissimilarity(Rt, Rs)

def towards(Rs, Rt, Rst):
    """
    Towards  score

    :param Rs: UAR of the source author's corpus
    :param Rt: UAR of the target author's corpus
    :param Rst: UAR of style transferred corpus
    :return: Towards score
    """
    return max(similarity(Rst, Rt) - similarity(Rs, Rt), 0) / dissimilarity(Rs, Rt)


def joint(away, towards, sim):
    """Joint score: geometric mean of Away, Towards, and SIM scores"""
    return geometric_mean(
        geometric_mean(away, towards), sim
    )
