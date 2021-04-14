import re
import spacy 

special_replacer = re.compile(r"[^a-zA-Z0-9\s\.\?\,\$\!%\&\]\['\"\(\)\@#\-\:/\\\<\>\{\};]")
space_compressor = re.compile(r"\s+")
quote_fixer = re.compile(r"[“”’]")
link_replacer = re.compile(r"http\S+")
ellipsis_replacer = re.compile(r'\.{3,}')
number_replacer = re.compile(r'\b\d+[\d,\.]+\b')
squasher = re.compile(r'(.)\1(\1+)')
CHARS = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOP0123456789 ;:(){}[]"

nlp = spacy.load('en_core_web_sm')

def purify(s):
    ns = special_replacer.sub("|", quote_fixer.sub("'",  ellipsis_replacer.sub(".", link_replacer.sub("", space_compressor.sub(" ", s.replace("`", "'")))))).lower()
    return ns

def spacy_ner_replace(sentence):
    s = nlp(sentence)
    for ent in s.ents:
        if len(ent.text) > 1:
            sentence = re.sub(r"\b%s\b" % re.escape(ent.text) , ent.label_, sentence)
    return sentence

def spacy_tokenize(sentence):
    return [token.text for token in nlp(sentence, disable=['parser', 'tagger', 'ner'])]

if __name__ == "__main__":
    while 1:
        i = input("Enter text to purify: ")
        print(f"{purify(i)}")
