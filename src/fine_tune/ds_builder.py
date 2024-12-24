import os
from collections import defaultdict
from typing import Tuple

import pandas as pd
from datasets import DatasetDict, Dataset, Features, Value, ClassLabel
from rich.console import Console

from utils import split_dataset

# Initialize a console for rich outputs
console = Console()

MIN_WORDS = 50
MAX_WORDS = 400


class ClassificationDatasetBuilder:
    def __init__(
        self,
        data_dir: str,
        target_author: str = "hugo",
        binary_classification: bool = True,
        split_ratios: Tuple[float, float, float] = (0.8, 0.1, 0.1),
    ):
        """
        Initialize the dataset builder with configuration.

        Args:
            data_dir (str): Path to the directory containing subdirectories for each author.
            target_author (str): The author to use as the positive class in binary classification.
            binary_classification (bool): Whether to create a binary classification dataset.
            split_ratios (Tuple[float, float, float]): Ratios for train, validation, and test splits.
        """
        self.data_dir = data_dir
        self.target_author = target_author.lower()
        self.binary_classification = binary_classification
        self.split_ratios = split_ratios
        self.authors = self._get_authors()
        self.author_to_label = self._map_authors_to_labels()

    def _get_authors(self) -> list:
        """Get a list of author directories (case-insensitive)."""
        authors = sorted(
            [author.lower() for author in os.listdir(self.data_dir) if os.path.isdir(os.path.join(self.data_dir, author))]
        )
        if not authors:
            raise ValueError(f"No author directories found in the specified path: {self.data_dir}")
        return authors

    def _map_authors_to_labels(self) -> dict:
        """Create a mapping from authors to labels based on classification mode."""
        return {
            author: 1 if self.binary_classification and author == self.target_author else 0
            for author in self.authors
        }

    def _read_author_data(self, author: str) -> list:
        """Read all text data for an author from the given directory."""
        paragraphs = []
        author_dir = os.path.join(self.data_dir, author)
        file_list = os.listdir(author_dir)

        for file_name in file_list:
            file_path = os.path.join(author_dir, file_name)
            if not os.path.isfile(file_path):
                continue
            try:
                with open(file_path, "r", encoding="utf-8") as file:
                    lines = [line.strip() for line in file.readlines() if line.strip()]
                    lines = [line for line in lines if MIN_WORDS <= len(line.split()) <= MAX_WORDS]
                    paragraphs.extend(lines)
            except (OSError, UnicodeDecodeError) as e:
                console.print(f"[bold red]Error reading file {file_path}:[/bold red] {e}")
        return paragraphs

    @staticmethod
    def _balance_dataset(df: pd.DataFrame) -> pd.DataFrame:
        """
        Balance the dataset by undersampling the overrepresented label.

        Args:
            df (pd.DataFrame): The input dataset containing 'text' and 'labels' columns.

        Returns:
            pd.DataFrame: The balanced dataset with an equal number of samples for each label.
        """
        console.print("[bold blue]Balancing the dataset...[/bold blue]")

        # Get the count of each label
        label_counts = df['labels'].value_counts()
        min_count = label_counts.min()  # The number of samples for the underrepresented label

        # Group by labels and sample min_count items from each group
        balanced_df = (
            df.groupby('labels', group_keys=False)
            .apply(lambda x: x.sample(n=min_count, random_state=42))
            .reset_index(drop=True)
        )

        console.print(
            f"[bold green]Dataset balanced: {len(balanced_df)} total samples ({min_count} per label).[/bold green]")
        return balanced_df

    def _prepare_dataset(self) -> pd.DataFrame:
        """Prepare the dataset dictionary by reading files and assigning labels."""
        dataset_dict = defaultdict(list)
        console.print("[bold yellow]Preparing dataset...[/bold yellow]")

        for author in self.authors:
            paragraphs = self._read_author_data(author)
            label = self.author_to_label[author] if self.binary_classification else self.authors.index(author)
            dataset_dict["text"].extend(paragraphs)
            dataset_dict["labels"].extend([label] * len(paragraphs))

        if not dataset_dict["text"]:
            console.print("[bold red]Error:[/bold red] No valid data found to create the dataset.")
            raise ValueError("No valid data found to create the dataset.")
        console.print("[bold green]Dataset preparation complete.[/bold green]")
        return pd.DataFrame(dataset_dict)

    def build(self, balance: bool = True) -> DatasetDict:
        """
        Builds and returns the classification dataset as a DatasetDict.

        Returns:
            DatasetDict: A dictionary containing train, validation, and test splits.
        """
        # Step 1: Prepare dataset
        df = self._prepare_dataset()

        # Step 2: Balance dataset if needed
        if balance:
            df = self._balance_dataset(df)

        # Step 3: Split dataset
        train_df, val_df, test_df = split_dataset(df, self.split_ratios)

        # Step 4: Define dataset features
        features = Features({
            "text": Value("string"),
            "labels": ClassLabel(
                num_classes=2 if self.binary_classification else len(self.authors),
                names=["other", self.target_author] if self.binary_classification else self.authors,
            ),
        })

        # Step 5: Create and return DatasetDict
        console.print("[bold blue]Building the DatasetDict...[/bold blue]")
        dataset_dict = DatasetDict({
            "train": Dataset.from_pandas(train_df.reset_index(drop=True), features=features),
            "validation": Dataset.from_pandas(val_df.reset_index(drop=True), features=features),
            "test": Dataset.from_pandas(test_df.reset_index(drop=True), features=features),
        })
        console.print("[bold green]DatasetDict successfully built![/bold green]")
        return dataset_dict


def display_label_distribution(dataset: DatasetDict):
    """
    Display the number of items per label in each dataset split.

    Args:
        dataset (DatasetDict): The dataset containing train, validation, and test splits.
    """
    for split_name, split_data in dataset.items():
        # Convert to Pandas DataFrame for easier aggregation
        df = split_data.to_pandas()
        label_counts = df['labels'].value_counts()  # Count items by label
        print(f"\n[Split: {split_name}]")
        for label, count in label_counts.items():
            print(f"  Label {label}: {count} items")


def display_text_length_stats(dataset: DatasetDict):
    """
    Display text length statistics (min, max, average) for each dataset split.

    Args:
        dataset (DatasetDict): The dataset containing train, validation, and test splits.
    """
    for split_name, split_data in dataset.items():
        # Convert to Pandas DataFrame for easier analysis
        df = split_data.to_pandas()

        # Compute text lengths
        df['word_count'] = df['text'].apply(lambda x: len(x.split()))  # Word count
        df['char_count'] = df['text'].apply(len)  # Character count

        # Calculate statistics for word count
        min_words = df['word_count'].min()
        max_words = df['word_count'].max()
        avg_words = df['word_count'].mean()

        # Calculate statistics for character count
        min_chars = df['char_count'].min()
        max_chars = df['char_count'].max()
        avg_chars = df['char_count'].mean()

        # Print results for the current split
        print(f"\n[Split: {split_name}]")
        print(f"  Words - Min: {min_words}, Max: {max_words}, Average: {avg_words:.2f}")
        print(f"  Characters - Min: {min_chars}, Max: {max_chars}, Average: {avg_chars:.2f}")
