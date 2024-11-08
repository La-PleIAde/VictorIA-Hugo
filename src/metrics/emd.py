from typing import List

from scipy.optimize import linear_sum_assignment
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import pairwise_distances


def emd_score(references: List[str], hypotheses: List[str]) -> float:
    """
    Calculate the Earth Mover's Distance (EMD) between two text corpora
    using multidimensional distributions.

    :param references: List of reference texts.
    :param hypotheses: List of hypothesis texts.
    :return: EMD score (approximate).
    """
    # Vectorize texts using TF-IDF to get multidimensional embeddings
    vectorizer = TfidfVectorizer()
    ref_vectors = vectorizer.fit_transform(references).toarray()
    hyp_vectors = vectorizer.transform(hypotheses).toarray()

    # Compute the pairwise Euclidean distance matrix between distributions
    distance_matrix = pairwise_distances(ref_vectors, hyp_vectors, metric='euclidean')

    # Use the linear sum assignment to find the minimum cost matching
    row_ind, col_ind = linear_sum_assignment(distance_matrix)

    # Calculate EMD as the average minimum cost based on optimal assignment
    emd_distance = distance_matrix[row_ind, col_ind].sum() / len(row_ind)

    return emd_distance
