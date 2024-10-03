import argparse
import logging
import re
from pathlib import Path

from src.data.path import get_absolute_path, list_authors


def clean_up(text: str) -> str:
    beginnings = ["Chapitre I\n", "CHAPITRE I\n", "I\n", "LIVRE PREMIER\n", "Livre premier\n", "Droit de traduction réservé\n"]
    endings = ["NOTES", "NOTE", "Notes", "TABLE", "\nFIN\n"]

    # Remove the part before the beginning marker
    for start in beginnings:
        if start in text:
            text = text[text.find(start) + len(start):]

    # Remove the part after the ending marker
    for end in endings:
        if end in text:
            text = text[:text.find(end)]

    # Remove junk
    text = re.sub(r"_", "", text)  # Remove underscores
    text = re.sub(r'\[.*?\]', "", text)  # Remove bracketed text like [1]
    text = re.sub(r"\.\s*\.", ".", text)  # Replace . . . with .
    text = re.sub(r"\*\s*\*", "", text)  # Remove * * *
    text = re.sub(r"\[Illustration:\s*.*?\s*]", "", text, flags=re.DOTALL)  # Remove illustrations


    # Remove divisions
    text = re.sub(r"^\s*(CHAPITRE\s+[IVXLCDM]+|LIVRE\s+[IVXLCDM]+|LETTRE\s+[IVXLCDM]+|\d+)\s*$", "",
                  text, flags=re.MULTILINE)  # Remove chapter titles
    text = re.sub(r"^\s*[IVXLCDM]+\s*$", "", text, flags=re.MULTILINE)  # Remove lines with Roman numerals
    text = re.sub(r"^\s*\d+\s*$", "", text, flags=re.MULTILINE)  # Remove lines with regular digits


    # Standardize punctuation
    text = re.sub(r"--", "—", text)  # Substitute -- for —
    text = re.sub(r"(^|\s)([-–—])(\s)", r"\1—\3", text)  # Standardize dashes
    text = text.replace("—", "— ") # Ensure the space after the dash
    text = text.replace('«', '"').replace('»', '"')  # Replace guillemets with double quotes
    text = text.replace('“', '"').replace('”', '"')  # Replace quotes
    text = re.sub(r'"(.*?)"', lambda m: m.group(0).replace("\n", ""), text)  # Preserve quotes and dialogues


    # Standardize linebreaks and spaces
    text = re.sub(r"(?<![.!?])\n(?!\n)", " ", text)  # Remove single line breaks inside sentences
    lines = text.splitlines()
    lines = [line.strip() for line in lines if line]
    lines = [re.sub(r'\s+', ' ', line) for line in lines]  # Replace multiple spaces with a single space

    text = '\n'.join(lines)
    return text.strip()


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
                content = clean_up(content)
                save_path = Path(save_dir) / file.name
                with save_path.open('w', encoding='utf-8') as f:
                    f.write(content)
                logging.info(f"Processed {file.name} by {author}")
            except Exception as e:
                logging.error(f"Failed to process {file.name} by {author}. Error: {e}")


def process_raw_dataset(input_ds_name: str = 'unsourced', output_ds_name: str = 'clean'):
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
    process_raw_dataset(args.i, args.o)
