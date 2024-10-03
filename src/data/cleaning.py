import os
import re
import argparse


def un_gutenberg(text):
    """Remove the Project Gutenberg intro and outro"""
    start_pattern = r"\*\*\* START OF THE PROJECT GUTENBERG EBOOK .* \*\*\*"
    start_matches = re.finditer(start_pattern, text)
    start = [substr.end() for substr in start_matches][0]

    end_pattern = r"\*\*\* END OF THE PROJECT GUTENBERG EBOOK .* \*\*\*"
    end_matches = re.finditer(end_pattern, text)
    end = [substr.start() for substr in end_matches][0]

    result = text[start:end].strip()
    return result


for root, dirs, files in os.walk("__data/stage=raw/source=gutenberg"):
    for file in files:
        if file.endswith(".txt"):
            with open(root + "/" + file, "r", encoding="utf-8") as f:
                content = f.read()
        else:
            continue

        try:
            un_gutenberg(content)
        except Exception as e:
            print(root, file, e)


# def clean_up(text):
#     """Clean up the text, removing the beginning notes,
#     the ending notes and the unnecessary characters in between"""
#     beginnings = ["Chapitre I\n", "CHAPITRE I\n", "I\n", "LIVRE PREMIER\n", "Livre premier\n"]
#     endings = ["NOTES", "NOTE", "Notes"]
#
#     new_doc = text
#
#     for item in beginnings:
#         if item in text:
#             new_doc = text[text.find(item):]
#     for item in endings:
#         if item in new_doc:
#             new_doc = new_doc[:new_doc.find(item)]
#     cleaned_new_doc = re.sub("--", "—", new_doc)  # substitutes -- for — in direct speech
#     cleaned_new_doc = re.sub("_", "", cleaned_new_doc)  # cleans up strings like _ANANKÊ_
#     cleaned_new_doc = re.sub(' +', ' ', cleaned_new_doc)  # cleans up redundant spaces
#     cleaned_new_doc = re.sub('(\[.*\])', "", cleaned_new_doc)  # cleans up things like [1]
#     cleaned_new_doc = re.sub("[. ]+[.]", "", cleaned_new_doc)  # cleans up . . .
#     cleaned_new_doc = re.sub('[* ]+[*]', "", cleaned_new_doc)  # cleans up * * *
#     cleaned_new_doc = re.sub('\n\n+', "\n", cleaned_new_doc)  # cleans up leftover blank lines
#     return cleaned_new_doc
#
#
# def lines_concat(lines: list[str]):
#     lines = [line.replace('\n', ' ') for line in lines]
#     lines = ['\n' + line if (line.startswith('-') or line.startswith('—')) else line for line in lines]
#     return ' '.join(lines)
#
#
# def main(filepath: str, source: str):
#     with open(filepath, 'r', encoding='utf-8') as f:
#         text = f.read()
#
#     if source == 'gutenberg':
#         text = un_gutenberg(text)
#
#     text = clean_up(text)
#     text = lines_concat(text.splitlines())
#
#     with open(f'{filepath[:-4]}_cleaned.txt', 'w', encoding='utf-8') as f:
#         f.write(text)
#
#
# if __name__ == '__main__':
#     parser = argparse.ArgumentParser(
#         prog='VictorIA Hugo data cleaner', description='Clean up the dataset'
#     )
#     parser.add_argument('-f', '--filepath', action='store', help='Path to the file')
#     parser.add_argument('-s', '--source', action='store', default='gutenberg', help='Source of the dataset')
#     args = parser.parse_args()
#
#     main(args.filepath, args.source)
