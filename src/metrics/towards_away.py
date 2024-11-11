import numpy as np
import torch


def uar(embeddings: list):
    """Universal Authorship Representation
    Averages embeddings of the texts of a given author.

    :param embeddings: List of embeddings.
    :return: averaged embedding vector.
    """
    embeddings = torch.stack(embeddings)
    author_embedding = embeddings.mean(dim=0)
    return author_embedding.flatten()


def similarity(vec1, vec2):
    """Vector similarity"""
    sim = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
    return (sim + 1) / 2  # Scaling sim to the range [0, 1]


def dissimilarity(vec1, vec2):
    """Complement to vector similarity"""
    return 1 - similarity(vec1, vec2)


def mis(Pa, Pb):
    """
    Mutual Implication Score (MIS)

    :param Pa: Corpus of the first author's embeddings
    :param Pb: Corpus of the second author's embeddings
    :return: Mutual Implication Score (MIS)
    """
    return np.mean([np.dot(pa, pb) / (np.linalg.norm(pa) * np.linalg.norm(pb)) for pa, pb in zip(Pa, Pb)])


def geometric_mean(values):
    return np.prod(values) ** (1.0 / len(values))


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


def sim(Ps, Pt, Pst):
    """
    SIM score

    :param Ps: source author's corpus embeddings
    :param Pt: target author's corpus embeddings
    :param Pst: style transferred corpus embeddings
    :return: SIM score
    """
    return max(mis(Pst, Ps) - mis(Pt, Ps), 0) / (1 - mis(Pt, Ps))


def joint(away, towards, sim):
    """Joint score: geometric mean of Away, Towards, and SIM scores"""
    return geometric_mean(
        [ geometric_mean([away, towards]), sim ]
    )
