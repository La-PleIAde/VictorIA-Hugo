import argparse
import logging
import re
from pathlib import Path
from typing import Tuple

from src.data.path import get_absolute_path, list_authors


class UnGutenbergError(Exception):
    """Raised when there is an error processing the Gutenberg text."""


def find_gutenberg_bounds(text: str) -> Tuple[int, int]:
    """Find start and end boundaries in a Gutenberg text."""
    start_pattern = r"\*\*\* START OF THE PROJECT GUTENBERG EBOOK .* \*\*\*"
    end_pattern = r"\*\*\* END OF THE PROJECT GUTENBERG EBOOK .* \*\*\*"

    start_match = re.search(start_pattern, text)
    end_match = re.search(end_pattern, text)

    if not start_match:
        raise UnGutenbergError("Start pattern not found in the provided text")
    if not end_match:
        raise UnGutenbergError("End pattern not found in the provided text")
    if end_match.start() <= start_match.end():
        raise UnGutenbergError("End position is before the start position, invalid text boundaries.")

    return start_match.end(), end_match.start()


def un_gutenberg(text: str) -> str:
    """Remove the Project Gutenberg intro and outro with custom error handling."""
    start, end = find_gutenberg_bounds(text)

    # Extract and return the cleaned text
    result = text[start:end].strip()
    if not result:
        raise UnGutenbergError("Text extraction failed, resulting in an empty string.")
    return result


def process_author_files(author: str, output_ds_name: str):
    """Process all files for a given author."""
    author_path = get_absolute_path('raw', source='gutenberg', author=author)
    save_dir = get_absolute_path('silver', name=output_ds_name, author=author, force_exist=False, create_new=True)

    Path(save_dir).mkdir(parents=True, exist_ok=True)

    for file in Path(author_path).iterdir():
        if file.is_file():
            with file.open('r', encoding='utf-8') as f:
                content = f.read()
            try:
                content = un_gutenberg(content)
                save_path = Path(save_dir) / file.name
                with save_path.open('w', encoding='utf-8') as f:
                    f.write(content)
                logging.info(f"Processed {file.name} by {author}")
            except UnGutenbergError as e:
                logging.error(f"Failed to process {file.name} by {author}. Error: {e}")
            except Exception as e:
                logging.error(f"Unexpected error processing {file.name} by {author}. Error: {e}")


def process_raw_dataset(output_ds_name: str = 'unsourced'):
    """Process entire raw dataset."""
    for author in list_authors():
        process_author_files(author, output_ds_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process Project Gutenberg texts.")
    parser.add_argument(
        "-o",
        type=str,
        default="unsourced",
        help="The name of the output dataset to save the processed files."
    )

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    process_raw_dataset(args.o)
