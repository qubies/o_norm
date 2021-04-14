import numpy as np  # linear algebra
import pickle
import os
import spacy


from o_norm.preprocessing import purify, spacy_ner_replace
from o_norm.resources import (
    CURSES,
    CURSE_BERTA
)
from o_norm.print_utils import print_banner_completion_wrapper, print_error
from o_norm.fragmenter import Get_Sequences, add_word
from o_norm.model_funcs import load_model

from functools import lru_cache

on = None
nlp = None


def init_o_norm():
    global on
    if on == None:
        on = O_Norm()


def init_spacy():
    global nlp
    if nlp == None:
        nlp = spacy.load("en_core_web_sm")


@print_banner_completion_wrapper("Loading Tokenizer", banner_token=" ")
def get_tokenizer():
    with open(TOKENIZER_FILE, "rb") as f:
        tokenizer = pickle.load(f)
    return tokenizer, len(tokenizer.word_index)

def softmax(w, t = 1.0):
    e = np.exp(w / t)
    dist = e / np.sum(e)
    return dist

class O_Norm:
    @print_banner_completion_wrapper("Building Normalizer", banner_token="=")
    def __init__(self, classification_threshold=0.8, model_type="transformer"):
        self.threshold = classification_threshold
        self.model_type = model_type.lower()
        if self.model_type == "transformer":
            self.model = load_model(CURSE_BERTA)
            self.CURSES = set(CURSES)
            self.CURSES.remove("none")
            self.CURSES = sorted(list(self.CURSES))
            self.CURSES.insert(0, "none")
        else:
            print_error("Invalid model_type specified")
            os.exit(1)

        self.curse_count = len(self.CURSES)
        print("Number of curses", self.curse_count)
        self.translator = {x:i for i,x in enumerate(self.CURSES)}

    def decode(self, n):
        return self.translator[n]


    @lru_cache(None)
    def predict_curse(self, text):
        full_word = "".join(text).strip()
        if len(full_word) < 3:
            return 0.0, full_word
        if full_word in CURSES:
            return 1.0, full_word
        text=text[:50]
        if len(text.strip()) < 3:
            return [["none", 1.0],["none", 0.0]]
        predictions, raw_outputs = self.model.predict([text])
        raw_outputs = softmax(raw_outputs)
        ind = list(raw_outputs.argsort().squeeze())[::-1][:5]
        return [[self.CURSES[x],raw_outputs[0][x]] for x in ind]

def normalize(s, why=False, ner=False):
    if ner:
        init_spacy()
        doc = nlp(s)
        for ent in doc.ents:
            add_word(ent.text.lower())
    init_o_norm()
    s = purify(s)
    oov_blocks = Get_Sequences(s)
    #  print(oov_blocks)
    if why:
        print("oov blocks", oov_blocks)
    olds = s
    replaced = False
    relacement = []
    for oov_block in oov_blocks:
        best_prediction = ""
        to_replace = ""
        best_prediction_score = 0
        for token in oov_block:
            if len(token.strip()) <= 2:
                continue
            scores = on.predict_curse(token)
            top_curse, top_score = scores[0]
            second_curse, second_score = scores[1]
            if why:
                print(f"token: {token}, scores: {scores}")
            if top_score > best_prediction_score and top_curse != "none":
                best_prediction_score = top_score 
                to_replace = token
                best_prediction = top_curse
            elif second_score != "none" and second_score > best_prediction_score:
                best_prediction_score = second_score 
                to_replace = token
                best_prediction = second_curse
        if len(oov_block) > 1:
            full_word = "".join(oov_block).strip()
            if len(full_word) >= 3:
                scores = on.predict_curse(full_word)
                top_curse, top_score = scores[0]
                second_curse, second_score = scores[1]
                if why:
                    print(f"token: {full_word}, scores: {scores}")
                if top_score > best_prediction_score and top_curse != "none":
                    best_prediction_score = top_score
                    to_replace = full_word
                    best_prediction = top_curse
                elif second_score != "none" and second_score > best_prediction_score:
                    best_prediction_score = second_score 
                    to_replace = full_word
                    best_prediction = second_curse

        if (len(to_replace) > 3 and best_prediction_score > 0.4) or (len(to_replace) == 3 and best_prediction_score > 0.98):
            if why:
                print(
                    f"Replacing {to_replace} with {best_prediction}. Score: {best_prediction_score}"
                )
            s = s.replace(to_replace.strip(), best_prediction)
            replaced = True
            relacement.append([to_replace, best_prediction])

    return s, replaced, relacement


def set_options(classification_threshold=0.5, model_type="full"):
    global on
    if on == None or model_type != on.model_type:
        on = O_Norm(classification_threshold, model_type)
    else:
        on.threshold = classification_threshold


if __name__ == "__main__":
    while 1:
        mod = normalize(input("Please Enter a Test Phrase: "), why=True)
        print(mod)

