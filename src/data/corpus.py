import os
import re
import logging
import argparse


def un_gutenberg(text: str):
    """Remove the Project Gutenberg intro and outro"""
    new_text = re.search("[*][*][*] [a-zA-ZÀ-ú.:;,!-? ]+ [*][*][*]", text) #removes "***START OF PROJECT GUTENBERG <...>***"
    no_intro = text[new_text.end():].strip()

    new_text = re.search("[*][*][*] [a-zA-ZÀ-ú.:;,!-? ]+ [*][*][*]", no_intro) #removes "***END OF PROJECT GUTENBERG <...>***"
    no_outro = no_intro[:new_text.start()].strip()

    return no_outro


def clean_up(text: str):
    """
    Clean up the text, removing the beginning notes,
    the ending notes and the unnecessary characters in between
    """
    beginnings = ["Chapitre I\n", "CHAPITRE I\n", "I\n", "LIVRE PREMIER\n", "Livre premier\n"]
    endings = ["NOTES", "NOTE", "Notes"]

    for item in beginnings:
        if item in text:
            new_doc = text[text.find(item):]
    for item in endings:
        if item in new_doc:
            new_doc = new_doc[:new_doc.find(item)]
    cleaned_new_doc = re.sub("--", "—", new_doc) #substitutes -- for — in direct speech
    cleaned_new_doc = re.sub("_", "", cleaned_new_doc) #cleans up strings like _ANANKÊ_
    cleaned_new_doc = re.sub(' +', ' ', cleaned_new_doc) #cleans up redundant spaces
    cleaned_new_doc = re.sub('(\[.*\])', "", cleaned_new_doc) #cleans up things like [1]
    cleaned_new_doc = re.sub("[. ]+[.]", "", cleaned_new_doc) #cleans up . . .
    cleaned_new_doc = re.sub('[* ]+[*]', "", cleaned_new_doc) #cleans up * * *
    cleaned_new_doc = re.sub('\n\n+', "\n", cleaned_new_doc) #cleans up leftover blank lines
    return cleaned_new_doc



def process_file(filepath, source="Gutenberg"):
    """Cleans up the text in a given file and saves it into the silver stage"""
    with open(filepath, "r", encoding="utf-8") as f:
        text = f.read()

    if source == "Gutenberg":
        text = un_gutenberg(text)

    new_filepath = filepath.replace("/stage=raw/", "/stage=silver/")
    os.makedirs(os.path.dirname(new_filepath), exist_ok=True)

    with open(new_filepath, "w", encoding="utf-8") as f:
        f.write(text)

    logging.info(f"Successfully processed {filepath} and saved it to {new_filepath}")


def main(folder, source="Gutenberg"):
    """Process all files in the folder in a loop"""
    for file in os.listdir(folder):
        process_file(os.path.join(folder, file), source=source)
    logging.info("Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--folder", required=True, help="Folder to process")
    parser.add_argument("-s", "--source", default="Gutenberg", help="Source of text")
    args = parser.parse_args()

    main(args.folder, args.source)
