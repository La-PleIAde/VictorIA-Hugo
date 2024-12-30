import json
import os
import re
import time
import unicodedata
from typing import Any, Dict, Iterator, List, Optional

import pandas as pd
import spacy_udpipe
from spacy.matcher import Matcher
from spacy_udpipe.utils import LANGUAGES as UDPIPE_LANGUAGES
from spacy_udpipe.utils import MODELS_DIR as UDPIPE_MODELS_DIR

from motifs.config import LOGGER, PKG_DATA_PATH
from motifs.constants import AVAILABLE_TOKEN_TYPES

BASE_MOTIFS = json.load(open(f"{PKG_DATA_PATH}/fr_motifs.json", "r"))


def load_txt(path) -> str:
    """
    :param path: file path
    :return: content of file
    """
    try:
        with open(path, mode="r") as f:
            content = f.read()
            if len(content) > 0:
                return content
            else:
                LOGGER.warning(f"{path} seems to be empty! Ignoring it")
    except Exception as exc:
        LOGGER.exception(f"Error while loading {path}...")
        raise exc


def split_text(text: str, max_length: int = 1000000) -> str:
    """

    :param text: Text to split
    :param pattern: Regex pattern to define where to split in windows,
    not applied by default (None)
    :param max_length:
    :return:
    """
    for i in range(0, len(text), max_length):
        yield text[i : i + max_length]


def verify_token_type(token_type: str):
    isinstance(token_type, str)
    if token_type not in AVAILABLE_TOKEN_TYPES:
        LOGGER.error(
            f"This token_type is not implemented! Available targets are"
            f" {AVAILABLE_TOKEN_TYPES}"
        )
        raise NotImplementedError


def preprocess_generator(text: Iterator[str], patterns: Dict) -> Iterator[str]:
    # Combine patterns into a single regular expression pattern
    combined_pattern = re.compile("|".join(map(re.escape, patterns.keys())))

    for string in text:
        # Handle unicode errors
        string = unicodedata.normalize("NFKD", string)
        string = combined_pattern.sub(
            lambda match: patterns[match.group(0)], string
        )
        yield string.strip()


class Tokenizer:
    """
    This pipeline transforms a corpus of documents to tokens with linguistic
    informations and motifs.

    :param corpus_dir: The folder where the corpus is located. The corpus
    must contain at least one document as a .txt file.
    :param docs: List of documents names with the  tokens_dir or corpus_dir.
    Provide a docs list to only load data from the specified documents with the
     directory.
    :param token_type: name of the token to obtain should be one of ["text",
    "lemma", "pos", "motif"]
    :param motifs: A dictionary of motif with the following structure
    dict[list[dict[any]], that is for example: {"motif1": pattern1,
    "motif2": pattern2}, where each pattern is a rule for the token Matcher
    (see https://spacy.io/usage/rule-based-matching for more details).
    A simple example with one motif "ADJ" would be:
    `motif = {"ADJ": [{"POS": "ADJ"}]}`. Each token with `pos` attribute
    equal to "ADJ" will be annotated with the motif "ADJ".
    You can check the `BASE_MOTIFS` for more examples and spacy Matcher to
    create your own motif.
    Should be given if token_type is "motif".
    :param output_dir: The folder where to save the outputs.
    :param lang: language for the udpipe model (default is "fr")

    :Example:

    >>> tokenizer = Tokenizer(
    >>>                 path, token_type="motif", motifs=BASE_MOTIFS,
    >>>                 output_dir="output_pipeline"
    >>>             )
    >>> data = tokenizer.transform(save=True)
    """

    def __init__(
        self,
        corpus_dir: str,
        docs: Optional[List] = None,
        token_type: str = "motif",
        motifs: Optional[dict[list[dict[Any]]]] = BASE_MOTIFS,
        output_dir: Optional[str] = None,
        lang: str = "fr",
    ):
        self.corpus_dir = corpus_dir
        self.docs = docs
        self.token_type = token_type
        verify_token_type(token_type)
        if token_type == "motif" and motifs is None:
            LOGGER.error(
                "You must give the list of motifs with token_type='motif'"
            )
            raise AssertionError
        self.output_dir = output_dir
        if self.output_dir is not None:
            if not os.path.isdir(self.output_dir):
                LOGGER.debug(
                    f"Creating output destination at {self.output_dir}"
                )
                os.makedirs(self.output_dir)
            else:
                LOGGER.debug(
                    f"The destination folder {self.output_dir} already "
                    f"exists, outputs will be overwritten!"
                )
        if docs is None:
            docs = os.listdir(self.corpus_dir)
        self.corpus_path = {
            f: os.path.join(self.corpus_dir, f)
            for f in filter(
                lambda p: p.endswith("txt") and p in docs,
                os.listdir(self.corpus_dir),
            )
        }
        self.motifs = motifs

        if os.path.isdir(UDPIPE_MODELS_DIR):
            if not UDPIPE_LANGUAGES[lang] in os.listdir(UDPIPE_MODELS_DIR):
                spacy_udpipe.download(lang)
        else:
            spacy_udpipe.download(lang)
        self.nlp = spacy_udpipe.load(lang)

    @staticmethod
    def preprocessing(text: str) -> str:
        REGEX = json.load(
            open(f"{PKG_DATA_PATH}/patterns_to_replace.json", "r")
        )
        # Replacements
        for k in REGEX:
            text = re.sub(k, REGEX[k], text)

        return text.strip()

    def annotate_text(self, text: str, validate: bool = False) -> pd.DataFrame:
        doc = self.nlp(text)
        if self.token_type == "motif":
            # Initialized dataframe
            # Initialized motif column with lemma
            data = pd.DataFrame(
                (
                    (
                        token.text,
                        token.lemma_,
                        token.pos_,
                        token.morph,
                        token.dep_,
                        token.n_lefts,
                        token.n_rights,
                        token.is_sent_start,
                        token.lemma_,
                    )
                    for token in self.nlp(text)
                ),
                columns=[
                    "text",
                    "lemma",
                    "pos",
                    "morph",
                    "dep",
                    "n_lefts",
                    "n_rights",
                    "is_sent_start",
                    "motif",
                ],
            )
            # Initialize matcher
            matcher = Matcher(self.nlp.vocab, validate=validate)
            for m in self.motifs:
                matcher.add(m, [self.motifs[m]])
            # Apply it to the doc
            matches = matcher(doc)
            # Get the motif for each match
            motif_match = [
                [start, end, self.nlp.vocab.strings[match_id]]
                for match_id, start, end in matches
            ]
            motif_match = pd.DataFrame(
                motif_match, columns=["start", "end", "motif"]
            )
            # We must be sure that the matches correspond to one token
            matches_length = motif_match["end"] - motif_match["start"]
            errors = motif_match[matches_length > 1]
            if len(errors) > 0:
                to_print = [
                    f"span: ({start}, {end}), text: {doc[start:end]}, motif: "
                    f"{m}"
                    for _, (start, end, m) in errors.iterrows()
                ]
                "\n".join(to_print)
                raise AssertionError(
                    "There is a problem with the motif matches. The matchers "
                    "returned more than one token at the following spans"
                    f"{to_print}!"
                )
            motif_match.set_index("start", inplace=True)
            # Modify motif column if we found a match
            data.loc[motif_match.index, "motif"] = motif_match.loc[:, "motif"]
        else:
            # Initialized dataframe
            # Initialized motif column with lemma
            data = pd.DataFrame(
                (
                    (
                        token.text,
                        token.lemma_,
                        token.pos_,
                        token.morph,
                        token.dep_,
                        token.n_lefts,
                        token.n_rights,
                        token.is_sent_start,
                    )
                    for token in self.nlp(text)
                ),
                columns=[
                    "text",
                    "lemma",
                    "pos",
                    "morph",
                    "dep",
                    "n_lefts",
                    "n_rights",
                    "is_sent_start",
                ],
            )

        data["sent_id"] = data["is_sent_start"].cumsum() - 1

        return data

    def transform_text(self, text: str, validate: bool = False):
        """

        Transform a text to tokens with linguistic information and motifs
        :param text: a text
        :param validate: Validate Matcher pattern, see Spacy
        :return: data, a DataFrame with columns ["text", "lemma", "pos",
        "morph", "dep", "n_lefts", "n_rights", "motif"]. See token Spacy
        documentation for more information.
        """
        text = self.preprocessing(text)
        # Text is too long, split it and annotate
        if len(text) > self.nlp.max_length:
            LOGGER.debug("Text is too long! Split and annotate...")
            data = pd.DataFrame()
            c = 0
            for sub_text in split_text(text, max_length=self.nlp.max_length):
                # self.nlp.max_length
                LOGGER.debug(
                    f"Annotate split {c} of length {len(sub_text)}..."
                )
                temp = self.annotate_text(sub_text, validate=validate)
                if c > 0:
                    last_send_id = data.sent_id.max()
                    temp["sent_id"] = temp["sent_id"] + data.sent_id.max() + 1
                    assert temp.sent_id.min() == last_send_id + 1
                data = pd.concat([data, temp], ignore_index=True)
                LOGGER.debug("Done.")
                c += 1
            return data
        else:
            return self.annotate_text(text, validate=validate)

    def transform_corpus(self, save: bool = False, **kwargs):
        errors = []
        for i, file in enumerate(self.corpus_path):
            LOGGER.debug(
                f"Steps to go {len(self.corpus_path) - i}: tokenizing"
                f" {file}..."
            )
            t1 = time.time()
            try:
                data = self.transform_text(
                    load_txt(self.corpus_path[file]), **kwargs
                )
                # Add doc columns
                filename = file.split(".txt")[0]
                data["doc"] = filename
                if save:
                    assert self.output_dir is not None
                    data.to_csv(
                        f"{self.output_dir}/{filename}.csv", index=False
                    )
                t2 = time.time()
                LOGGER.debug(
                    f"Done with {file} in {round(t2 - t1, 2)} seconds."
                )
                yield data
            except Exception as _exc:
                LOGGER.exception(f"Exception with file {file}...\n{_exc}")
                errors.append(file)
        if len(errors) > 0:
            LOGGER.warning(
                "There were errors while annotating the following texts: "
                f"{errors}"
            )
            if save:
                json.dump(errors, open(f"{self.output_dir}/errors.json", "w"))

    def transform(self, save: bool = False, **kwargs):
        return pd.concat(
            [d for d in self.transform_corpus(save=save, **kwargs)],
            ignore_index=True,
        )
