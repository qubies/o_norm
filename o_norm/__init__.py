import numpy as np  # linear algebra
import pickle
import json
import sys
import spacy
from os import path


from o_norm.preprocessing import purify, spacy_ner_replace
from o_norm.resources import RESOURCES
from o_norm.print_utils import print_banner_completion_wrapper, print_error
from o_norm.fragmenter import Get_Sequences
from o_norm.model_funcs import load_o_norm_model

from functools import lru_cache

nlp = None


def init_spacy():
    global nlp
    if nlp == None:
        nlp = spacy.load("en_core_web_sm")


# from https://gist.github.com/stober/1946926
def softmax(w, t=1.0):
    e = np.exp(w / t)
    dist = e / np.sum(e)
    return dist


class O_Norm:
    @print_banner_completion_wrapper("Building Normalizer", banner_token="=")
    def __init__(
        self,
        model_path,
        classification_threshold=0.5,
        classification_threshold_short=0.95,
        vocabulary=RESOURCES["VOCABULARY"],
        max_length=50,
        silent=True,
        emojii_dict=RESOURCES["EMOTICONS"],
    ):
        if isinstance(vocabulary, str):
            # assume it is a path ...
            with open(vocabulary) as f:
                vocabulary = json.load(f)
        # load up the model and curse file
        self.max_length = max_length
        self.load(model_path, silent)
        # add the curses.....
        vocabulary += [x for x in RESOURCES["CURSES"]]
        # add emojis
        vocabulary += emojii_dict

        self.set_vocabulary(vocabulary)
        self.threshold = classification_threshold
        self.short_threshold = classification_threshold_short

    def decode(self, n):
        return RESOURCES["TRANSLATOR"][n]

    @lru_cache(None)
    def predict_curse(self, text):
        full_word = "".join(text).strip()
        if len(full_word) < 3:
            return 0.0, full_word
        if full_word in RESOURCES["CURSES"]:
            return 1.0, full_word
        text = text[:50]
        if len(text.strip()) < 3:
            return [["none", 1.0], ["none", 0.0]]
        _, raw_outputs = self.model.predict([text])
        raw_outputs = softmax(raw_outputs)
        ind = list(raw_outputs.argsort().squeeze())[::-1][:5]
        return [[RESOURCES["TRANSLATOR"][x], raw_outputs[0][x]] for x in ind]

    def change_threshold(self, short=-1, normal=-1):
        if short != -1:
            if short < 0.0 or short > 1.0:
                raise ValueError(
                    f"Short threshold outside range 0.0 - 1.0. Recieved {short}"
                )
            self.short_threshold = short

        if normal != -1:
            if normal < 0.0 or normal > 1.0:
                raise ValueError(
                    f"normal threshold outside range 0.0 - 1.0. Recieved {normal}"
                )
            self.threshold = normal

    def load(self, model_path, silent):
        print(f"Loading '{model_path}'")
        try:
            self.model = load_o_norm_model(model_path, silent=silent)
        except Exception as e:
            print(f"Invalid model_path '{model_path}' specified:", e, file=sys.stderr)
            print(
                "Please ensure that the folder contains a valid o_norm model",
                file=sys.stderr,
            )
            sys.exit(1)
        self.model_name = path.basename(path.normpath(model_path))
        self.model_path = model_path

        # build internal datastructures
        self.common_curses = [
            x.lower() for x in RESOURCES["CURSES"].keys() if RESOURCES["CURSES"][x] > 0
        ]
        self.really_common_curses = [
            x.lower() for x in RESOURCES["CURSES"].keys() if RESOURCES["CURSES"][x] > 1
        ]
        self.curse_count = len(RESOURCES["CURSES"])
        print("Number of curses", self.curse_count)

    def set_vocabulary(self, vocab):
        RESOURCES["VOCABULARY"] = {x.lower() for x in vocab}

    def normalize(self, s, why=False, ner=False):
        if ner:
            init_spacy()
            doc = nlp(s)
            for ent in doc.ents:
                RESOURCES["VOCABULARY"].add(ent.text.lower())
        s = purify(s)
        oov_blocks = Get_Sequences(s)
        if why:
            print("oov blocks", oov_blocks)
        olds = s
        replaced = False
        relacement = []
        best_prediction_score = 0
        for oov_block in oov_blocks:
            best_prediction = ""
            to_replace = ""
            best_prediction_score = 0
            for token in oov_block:
                if len(token.strip()) <= 2:
                    continue
                scores = self.predict_curse(token)
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
                    scores = self.predict_curse(full_word)
                    top_curse, top_score = scores[0]
                    second_curse, second_score = scores[1]
                    if why:
                        print(f"token: {full_word}, scores: {scores}")
                    if top_score > best_prediction_score and top_curse != "none":
                        best_prediction_score = top_score
                        to_replace = full_word
                        best_prediction = top_curse
                    elif (
                        second_score != "none" and second_score > best_prediction_score
                    ):
                        best_prediction_score = second_score
                        to_replace = full_word
                        best_prediction = second_curse

            if (len(to_replace) > 3 and best_prediction_score > self.threshold) or (
                len(to_replace) == 3 and best_prediction_score > self.short_threshold
            ):
                if why:
                    print(
                        f"Replacing {to_replace} with {best_prediction}. Score: {best_prediction_score}"
                    )
                s = s.replace(to_replace.strip(), best_prediction)
                replaced = True
                relacement.append([to_replace, best_prediction])

        return {
            "Normalized": s,
            "Obscene": replaced,
            "Replacement made": relacement,
            "Score": best_prediction_score,
        }


if __name__ == "__main__":
    while 1:
        mod = normalize(input("Please Enter a Test Phrase: "), why=True)
        print(mod)
