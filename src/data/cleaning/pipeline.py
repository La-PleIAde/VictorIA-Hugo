import argparse
import logging
from collections.abc import Iterable
from pathlib import Path

from src.data.path import get_absolute_path, list_authors
from src.data.cleaning.un_gutenberg import un_gutenberg
from src.data.cleaning.clean_up import clean_up


pipeline_steps = [
    un_gutenberg,
    clean_up
]


def pipeline(text: str, steps: Iterable[callable]) -> str:
    for step in steps:
        text = step(text)
    return text



def process_author_files(author: str, input_ds_name: str, output_ds_name: str):
    """Process all files for a given author."""
    author_path = get_absolute_path('silver', name=input_ds_name, author=author)
    save_dir = get_absolute_path('silver', name=output_ds_name, author=author, force_exist=False, create_new=True)

    Path(save_dir).mkdir(parents=True, exist_ok=True)

    for file in Path(author_path).iterdir():
        if file.is_file():
            with file.open('r', encoding='utf-8') as f:
                content = f.read()
            try:
                content = pipeline(content, steps=pipeline_steps)
                save_path = Path(save_dir) / file.name
                with save_path.open('w', encoding='utf-8') as f:
                    f.write(content)
                logging.info(f"Processed {file.name} by {author}")
            except Exception as e:
                logging.error(f"Failed to process {file.name} by {author}. Error: {e}")


def process_dataset(input_ds_name: str = 'unsourced', output_ds_name: str = 'clean'):
    """Process entire raw dataset."""
    for author in list_authors():
        process_author_files(author, input_ds_name, output_ds_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process Project Gutenberg texts.")
    parser.add_argument(
        "-i",
        type=str,
        default="unsourced",
        help="The name of the input dataset to load the files from."
    )
    parser.add_argument(
        "-o",
        type=str,
        default="clean",
        help="The name of the output dataset to save the processed files."
    )

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    process_dataset(args.i, args.o)
