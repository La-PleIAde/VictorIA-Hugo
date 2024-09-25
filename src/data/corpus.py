import re

document = str(input("Please insert the full path to the text (without quotation marks): "))

with open(document, "r", encoding="utf-8") as f:
    text = f.read()


def un_gutenberg(text: str):
    """Removes the Project Gutenberg intro and outro"""
    new_text = re.search("[*][*][*] [a-zA-ZÀ-ú.:;,!-? ]+ [*][*][*]", text) #removes "***START OF PROJECT GUTENBERG <...>***"
    no_intro = text[new_text.end():].strip()

    new_text = re.search("[*][*][*] [a-zA-ZÀ-ú.:;,!-? ]+ [*][*][*]", no_intro) #removes "***END OF PROJECT GUTENBERG <...>***"
    no_outro = no_intro[:new_text.start()].strip()

    return no_outro


def clean_up(text):
    """Clean up the text, removing the beginning notes,
    the ending notes and the unnecessary characters in between"""
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


if "Gutenberg" in text:
    text = un_gutenberg(text)

print(clean_up(text))
