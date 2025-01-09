from typing import Union

import torch
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification
from sentence_transformers import SentenceTransformer


class Embedder:
    def __init__(self, model_name: str, finetuned: bool = False):
        """
        Initialize with the model name for either CamemBERT or FlauBERT.

        :param model_name: Model name as per Hugging Face's model hub.
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, do_lower_case=False)

        cls = AutoModelForSequenceClassification if finetuned else AutoModel
        self.model = cls.from_pretrained(model_name)

    def encode(self, text) -> torch.Tensor:
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)

        with torch.no_grad():
            outputs = self.model(**inputs)
        last_hidden_state = outputs.last_hidden_state

        return last_hidden_state.mean(dim=1)

    def get_probs(self, text) -> torch.Tensor:
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)

        with torch.no_grad():
            outputs = self.model(**inputs)
        probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)

        return probabilities


EmbedderType = Union[Embedder, SentenceTransformer]
