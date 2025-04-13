import os
import re
import sys
from typing import Tuple

# --- Exception classes ---
class UnGutenbergError(Exception):
    """Raised when there is an error processing the Gutenberg text."""
    pass

class UnWikisourceError(Exception):
    """Raised when there is an error processing the Wikisource text."""
    pass

# --- Gutenberg cleaning functions ---
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
    """Remove the Project Gutenberg intro and outro."""
    start, end = find_gutenberg_bounds(text)
    result = text[start:end].strip()
    if not result:
        raise UnGutenbergError("Text extraction failed, resulting in an empty string.")
    return result

# --- Wikisource cleaning functions ---
def un_wikisource(text: str) -> str:
    """
    Remove Wikisource metadata and headers.
    Assumes that the text contains a line with 'Exporté de Wikisource' and removes everything before
    the first non-empty line following that marker.
    """
    if "Exporté de Wikisource" not in text:
        raise UnWikisourceError("Wikisource marker not found in text")
    lines = text.splitlines()
    start_idx = 0
    for i, line in enumerate(lines):
        if "Exporté de Wikisource" in line:
            start_idx = i + 1
            break
    while start_idx < len(lines) and not lines[start_idx].strip():
        start_idx += 1
    cleaned_text = "\n".join(lines[start_idx:]).strip()
    if not cleaned_text:
        raise UnWikisourceError("Text extraction failed after Wikisource header removal.")
    return cleaned_text

# --- Further cleaning function ---
def clean_up(text: str) -> str:
    """Clean up and standardize the text by removing extraneous markers and metadata.
    - Beginning markers are searched for in the first 20% of the text.
    - Ending markers are searched for in the last 20% of the text.
    - A header block produced by publication metadata is removed.
    - French chapter headings using ordinal words (e.g. 'Chapitre Deuxième') are removed.
    - The markers 'prologue' and 'épilogue' are removed entirely.
    """
    total_length = len(text)

    # --- Revised Header Block Removal ---
    header_indicators = re.compile(r"(Produced by|Translated by)", re.IGNORECASE)
    first_20 = text[:int(total_length * 0.2)]
    if header_indicators.search(first_20):
        lines = text.splitlines()
        uppercase_occurrences = {}
        for i, line in enumerate(lines):
            cline = line.strip()
            if cline and cline.isupper() and 3 < len(cline) < 100:
                uppercase_occurrences.setdefault(cline, []).append(i)
        candidate_index = None
        for candidate, indices in uppercase_occurrences.items():
            if len(indices) >= 2:
                candidate_index = indices[1]  # second occurrence
                break
        if candidate_index is not None:
            text = "\n".join(lines[candidate_index+1:])
            total_length = len(text)

    # --- Remove common metadata lines ---
    metadata_patterns = [
        r"^Title:.*$", r"^Author:.*$", r"^Release date:.*$", r"^Language:.*$",
        r"^Original publication:.*$"
    ]
    for pattern in metadata_patterns:
        text = re.sub(pattern, "", text, flags=re.MULTILINE | re.IGNORECASE)

    total_length = len(text)

    # --- Remove word joiner characters ---
    text = text.replace('\u2060', '')

    # --- Remove beginning markers (search in first 20% of the text) ---
    first_20 = text[:int(total_length * 0.2)]
    beginnings = [
        # English markers
        "PART ONE", "PART 1", "PART I",
        "CHAPTER", "CHAPTER I", "CHAPTER 1", "Volume", "VOL.", "BOOK ONE", "BOOK I",
        "DEDICATION", "INTRODUCTION", "PREFACE",
        # French markers
        "PREMIÈRE PARTIE", "PARTIE 1", "PARTIE I",
        "CHAPITRE", "CHAPITRE I", "CHAPITRE 1", "CHAPITRE PREMIER",
        "TOME", "DEDICACE", "PRÉFACE"
    ]
    for marker in beginnings:
        pattern = r"(^|\n)" + re.escape(marker) + r"(?=[\n\.:])"
        match = re.search(pattern, first_20, flags=re.IGNORECASE)
        if match:
            text = text[match.end():]
            break

    # --- Remove ending markers (search in last 20% of the text) ---
    last_20 = text[int(total_length * 0.8):]
    endings = [
        "NOTES", "NOTE", "TABLE OF CONTENTS", "INDEX", "END OF THE TEXT",
        "ADDENDUM", "AFTERWORD", "APPENDIX", "FINALE", "THE END",
        # French endings
        "NOTES", "TABLE DES MATIÈRES", "INDEX", "ANNEXE", "FIN", "À propos de cette édition électronique"
    ]
    for marker in endings:
        if marker.strip().upper() == "FIN":
            pattern = r"(?i)(?:^|\n)" + re.escape(marker) + r"\b.*"
        else:
            pattern = r"(?i)(?:^|\n)" + re.escape(marker) + r"(?=[\n\.:])"
        matches = list(re.finditer(pattern, last_20))
        if matches:
            last_match = matches[-1]
            cutoff_index = int(total_length * 0.8) + last_match.start()
            text = text[:cutoff_index]
            break

    # --- Remove junk characters and unwanted patterns ---
    text = re.sub(r"_", "", text)
    text = re.sub(r"[─|—]+\n", "", text)
    text = re.sub(r"\[.*?\]", "", text)
    text = re.sub(r"\.\s*\.", ".", text)
    text = re.sub(r"\*\s*\*", "", text)
    text = re.sub(r"\[Illustration:\s*.*?\s*]", "", text, flags=re.DOTALL)
    text = re.sub(r"\[Footnotes:\s*.*?\s*]", "", text, flags=re.DOTALL)
    text = re.sub(r"\.\s{2,}\d+\n", ".\n", text)

    # --- Remove isolated chapter/section headings ---
    text = re.sub(
        r"^\s*(?:(CHAPTER|BOOK|VOLUME|LIVRE|LETTER|CHAPITRE|TOME)\s+[IVXLCDM1234567890]+\.?|\d+?)\s*[:\.\s-]*(.*)\s*$",
        "",
        text, flags=re.MULTILINE | re.IGNORECASE
    )
    text = re.sub(
        r"(?mi)^\s*(chapitre|tome|livre|volume)\s+(premier|première|deuxième|troisième|quatrième|cinquième|sixième|septième|huitième|neuvième|dixième|onzième|douzième|treizième|quatorzième|quinzième|seizième|dix[-\s]?septième|dix[-\s]?huitième|dix[-\s]?neuvième|vingtième)\s*[:\.\s-]*(.*)$",
        "",
        text
    )
    text = re.sub(r"^\s*[IVXLCDM]+\.?\s*$", "", text, flags=re.MULTILINE)
    text = re.sub(r"^\s*\d+\.?\s*$", "", text, flags=re.MULTILINE)

    # --- Standardize punctuation ---
    text = re.sub(r"--", "—", text)
    text = re.sub(r"(^|\s)([-–—])(\s)", r"\1—\3", text)
    text = text.replace("—", "— ")
    text = text.replace('«', '"').replace('»', '"')
    text = text.replace('“', '"').replace('”', '"')
    text = re.sub(r'"(.*?)"', lambda m: m.group(0).replace("\n", ""), text)

    # --- Standardize line breaks and whitespace ---
    text = re.sub(r"(?<![.!?])\n(?!\n)", " ", text)
    lines = text.splitlines()
    lines = [re.sub(r"\s+", " ", line.strip()) for line in lines if line.strip()]
    text = "\n".join(lines)

    # --- Remove chapter markers (e.g., "Chapter I:" or "Chapitre II:") and keep trailing text ---
    text = re.sub(r"(?mi)^(?:chapter|chapitre)\s+(?:[ivxlcdm]+|premier)[:\.\s-]+(.*)$", r"\1", text)

    # --- Remove any lines that consist solely of asterisks and whitespace ---
    text = re.sub(r"(?m)^\s*\*[\s\*]*\s*$\n?", "", text)

    # --- Remove lines that consist solely of punctuation (e.g. a lone period) ---
    text = re.sub(r"(?m)^\s*[\.,:;\-]+\s*$\n?", "", text)

    return text.strip()

# --- Main processing functions ---
def process_file(filepath: str) -> str:
    """Read, process, and clean a file."""
    with open(filepath, encoding='utf-8') as f:
        text = f.read()

    # Decide which header removal process to use
    cleaned = text
    try:
        if "START OF THE PROJECT GUTENBERG" in text:
            print(f"Processing Gutenberg file: {os.path.basename(filepath)}")
            cleaned = un_gutenberg(text)
        elif "Exporté de Wikisource" in text:
            print(f"Processing Wikisource file: {os.path.basename(filepath)}")
            cleaned = un_wikisource(text)
        else:
            print(f"No specific header found in {os.path.basename(filepath)}. Proceeding with generic cleaning.")
    except (UnGutenbergError, UnWikisourceError) as e:
        print(f"Warning: {e}. Proceeding with the original text.")

    cleaned = clean_up(cleaned)
    return cleaned

def process_directory(input_dir: str, output_dir: str) -> None:
    """
    Recursively process all .txt files in input_dir.
    The cleaned text for each file is written to output_dir, preserving the relative directory structure.
    """
    for root, _, files in os.walk(input_dir):
        # Calculate relative path from the input directory
        rel_path = os.path.relpath(root, input_dir)
        # Create the corresponding output directory
        target_dir = os.path.join(output_dir, rel_path)
        if not os.path.exists(target_dir):
            os.makedirs(target_dir)

        for filename in files:
            if filename.lower().endswith(".txt"):
                input_file = os.path.join(root, filename)
                try:
                    cleaned_text = process_file(input_file)
                    output_file = os.path.join(target_dir, filename)
                    with open(output_file, 'w', encoding='utf-8') as f_out:
                        f_out.write(cleaned_text)
                    print(f"Cleaned file saved to: {output_file}")
                except Exception as e:
                    print(f"Error processing {input_file}: {e}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python clean_corpus.py <input_directory> <output_directory>")
        sys.exit(1)

    input_directory = sys.argv[1]
    output_directory = sys.argv[2]
    process_directory(input_directory, output_directory)