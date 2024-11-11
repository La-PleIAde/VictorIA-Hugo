import os
from enum import Enum
from typing import List

from root import ROOT_DIR


class Stage(Enum):
    """Data pipeline stages."""
    raw: str = "raw"
    silver: str = "silver"
    gold: str = "gold"


def to_absolute(path: str) -> str:
    """Convert local path to absolute path"""
    return os.path.join(ROOT_DIR, path)


def list_sources() -> List[str]:
    """List sources used in the corpus"""
    raw_stage = to_absolute("__data/stage=raw")
    return [source.split('=')[1] for source in os.listdir(raw_stage)]


def list_authors() -> List[str]:
    """List authors in the corpus"""
    raw_stage = to_absolute("__data/stage=raw")
    authors = [
        # authors from different sources
        os.listdir(os.path.join(raw_stage, f"source={source}")) for source in list_sources()
    ]
    authors = set().union(*authors)  # preserving unique authors
    return list(authors)


def get_relative_path(stage: str | Stage, source: str = None, ds_type: str = None, author: str = None,
                      force_exist: bool = True, create_new: bool=False) -> str:
    """
    Returns the relative path of the targeted dataset or directory in the datalake (creates when necessary).

    :param stage: data pipeline stage
    :param source: (for `raw` stage) - the source of the data
    :param ds_type: (for silver and gold stages) - the name of the dataset
    :param author: author name
    :param force_exist: if True, raise an error if the directory does not exist
    :param create_new: if True, create the directory if it does not exist
    :return: relative path of the targeted dataset
    """
    if isinstance(stage, str):
        try:
            stage = Stage(stage)
        except ValueError:
            raise ValueError(f"Invalid stage: {stage}. Must be one of {[s.value for s in Stage]}")

    base_path = f"__data/stage={stage.value}"

    if source:
        if not source in list_sources():
            raise ValueError(f"Invalid source: {source}. Must be one of {list_sources()}")
        base_path = os.path.join(base_path, f"source={source}")
    if ds_type:
        base_path = os.path.join(base_path, f"type={ds_type}")
    if author:
        if not author in list_authors():
            raise ValueError(f"Invalid author: {author}. Must be one of {list_authors()}")
        base_path = os.path.join(base_path, author)

    if not os.path.exists(base_path):
        if force_exist:
            raise FileNotFoundError(base_path)
        if create_new:
            os.mkdir(base_path)

    return base_path


def get_absolute_path(stage: str | Stage, source: str = None, ds_type: str = None, author: str = None,
                      force_exist: bool = True, create_new: bool=False) -> str:
    """
    Returns the absolute path of the targeted dataset or directory in the datalake (creates when necessary).

    :param stage: data pipeline stage
    :param source: (for `raw` stage) - the source of the data
    :param ds_type: (for silver and gold stages) - the name of the dataset
    :param author: author name
    :param force_exist: if True, raise an error if the directory does not exist
    :param create_new: if True, create the directory if it does not exist
    :return: absolute path of the targeted dataset
    """
    local_path = get_relative_path(stage, source, ds_type, author, False, False)
    absolute_path = to_absolute(local_path)

    if not os.path.exists(absolute_path):
        if force_exist:
            raise FileNotFoundError(absolute_path)
        if create_new:
            os.makedirs(absolute_path, exist_ok=True)
    return absolute_path
