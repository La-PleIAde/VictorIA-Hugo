from enum import Enum
from typing import Union

import torch
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer


class ModelName(Enum):
    camembert: str = "camembert/camembert-large"
    flaubert: str = "flaubert-large-cased"


class SModelName(Enum):
    camembert: str = "dangvantuan/sentence-camembert-large"
    flaubert: str = "Lajavaness/sentence-flaubert-base"


class Embedder:
    def __init__(self, model_name: ModelName):
        """
        Initialize with the model name for either CamemBERT or FlauBERT.

        :param model_name: Model name as per Hugging Face's model hub.
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, do_lower_case=False)
        self.model = AutoModel.from_pretrained(model_name)

    def encode(self, text) -> torch.Tensor:
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)

        with torch.no_grad():
            outputs = self.model(**inputs)
        last_hidden_state = outputs.last_hidden_state

        return last_hidden_state.mean(dim=1).flatten()


EmbedderType = Union[Embedder, SentenceTransformer]
