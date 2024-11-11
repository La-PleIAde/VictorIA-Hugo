from enum import Enum
from typing import List

import nltk
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from pydantic import BaseModel
from rouge import Rouge

# Ensure you have the nltk data downloaded for tokenization
nltk.download('punkt')
nltk.download('punkt_tab')


class SmoothingMethod(Enum):
    no_smoothing = SmoothingFunction().method0
    avg_ngram_smoothing = SmoothingFunction().method5  # for longer sentences
    interpolated_smoothing = SmoothingFunction().method7  # recommended by default


class PRF(BaseModel):
    """Precision, Recall, F1 score"""
    p: float  # Precision
    r: float  # Recall
    f: float  # F1 score


class RougeScore(BaseModel):
    """Rouge scores for unigrams (rouge1), bigrams (rouge2) and longest common sequences (rougeL)"""
    rouge1: PRF  # Unigrams
    rouge2: PRF  # Bigrams
    rougeL: PRF  # Longest common sequences


def bleu_score(
    references: List[str],
    hypotheses: List[str],
    smoothing_function: SmoothingFunction = SmoothingMethod.avg_ngram_smoothing
    ) -> float:
    """
    Calculate a single corpus-level BLEU score.

    :param references: List of references (gold samples).
    :param hypotheses: List of hypotheses (predicted samples).
    :param smoothing_function: Smoothing function; choose from `SmoothingMethod`.
    :return: BLEU score.
    """
    references_tokens = [[nltk.word_tokenize(reference, language="french")] for reference in references]
    hypotheses_tokens = [nltk.word_tokenize(hypothesis, language="french") for hypothesis in hypotheses]
    return corpus_bleu(references_tokens, hypotheses_tokens, smoothing_function=smoothing_function)


def rouge_score(
        references: List[str],
        hypotheses: List[str],
    ) -> RougeScore:
    """
    Calculate a single corpus-level ROUGE score.

    :param references: List of references (gold samples).
    :param hypotheses: List of hypotheses (predicted samples).
    :return: ROUGE score as a `RougeScore` model.
    """
    scores =  Rouge().get_scores(references, hypotheses, avg=True)
    return RougeScore(
        rouge1 = PRF.model_validate(scores['rouge-1']),
        rouge2 = PRF.model_validate(scores['rouge-2']),
        rougeL = PRF.model_validate(scores['rouge-l']),
    )
