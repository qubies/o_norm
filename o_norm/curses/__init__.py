from o_norm.preprocessing import spacy_tokenize, spacy_ner_replace, purify
from o_norm.obfuscator import obfuscate
from o_norm.print_utils import print_banner
from o_norm.resources import CURSES, VOCABULARY
from tqdm import tqdm

import spacy
import json
import sys
import random
import difflib

e_dup=set()

class example_generator():
    def __init__(self, curse_dict, vocabulary, max_length=50):
        self.curses = [x.lower() for x in curse_dict.keys()]
        self.common_curses = [x.lower() for x in curse_dict.keys() if curse_dict[x] > 0]
        self.really_common_curses =[x.lower() for x in curse_dict.keys() if curse_dict[x] > 1]
        self.vocabulary = [x.lower() for x in vocabulary]
        self.short_vocabulary = [x.lower() for x in vocabulary if len(x) < 5]
        self.max_length=max_length

    def get_random_curse(self):
        return random.choice(self.curses)

    def get_random_common_curse(self):
        return random.choice(self.common_curses)

    def get_random_really_common_curse(self):
        return random.choice(self.really_common_curses)

    def get_random_word(self):
        return random.choice(self.vocabulary)

    def get_random_short_word(self):
        return random.choice(self.short_vocabulary)

    def get_random_number_string(self):
        num = random.choice([
            round(random.random()*100, 2)*(10**random.randint(0,20)), 
            float(random.randint(1, 10)), 
            ])
        if random.random() > 0.5:
            num = f"{num:.2f}"
        elif random.random() > 0.5:
            num = f"{int(num)}"
        else:
            num = f"{num}"
        if random.random() > 0.7:
            num = f"${num}"
        if random.random() > 0.7:
            num = f"({num})"
        return num

    def create_curse_examples(self, number_of_examples):
        return create_from_fn("Full Curse Dictionary", self.get_random_curse, number_of_examples, self.max_length, contains_curses=True)

    def create_common_curse_examples(self, number_of_examples):
        return create_from_fn("Common Curse Dictionary", self.get_random_common_curse, number_of_examples, self.max_length, contains_curses=True)

    def create_really_common_curse_examples(self, number_of_examples):
        return create_from_fn("Really Common Curse List", self.get_random_really_common_curse, number_of_examples, self.max_length, contains_curses=True)

    def create_word_examples(self, number_of_examples):
        return create_from_fn("Good Word Dictionary", self.get_random_word, number_of_examples, self.max_length, contains_curses=False)

    def create_number_examples(self, number_of_examples):
        return create_from_fn("Random Number Strings", self.get_random_number_string, number_of_examples, self.max_length, contains_curses= False, obfus=False)

    def create_short_word_examples(self, number_of_examples):
        return create_from_fn("Short Good Word Dictionary", self.get_random_short_word, number_of_examples, self.max_length, contains_curses=False)

def create_from_fn(name, fn, number_of_examples, max_length, contains_curses=True, obfus=True):
    examples = []
    ys = []
    global e_dup
    duplicates_rejected = 0
    print_banner(f"Creating examples for {name}")
    with tqdm(total=number_of_examples) as pbar:
        while len(examples) < number_of_examples:
            word_to_obfuscate = fn()
            if obfus: 
                obfuscated = purify(obfuscate(word_to_obfuscate))
            else:
                obfuscated = purify(word_to_obfuscate)

            if len(obfuscated) == 0: continue
            obfuscated = obfuscated[:max_length]
            if obfuscated not in e_dup:
                e_dup.add(obfuscated)
                if contains_curses: 
                    ys.append((1.0, [word_to_obfuscate]))
                else:
                    ys.append((0.0, []))
                examples.append(obfuscated)
                pbar.update(1)
            else: 
                duplicates_rejected += 1
    print()
    print(f"{name} Examples Created: {len(examples)}")
    print()
    print_banner("Done")
    assert len(examples) == len(ys)
    return examples, ys


if __name__ == '__main__':
    test = example_generator(CURSES, VOCABULARY)
    print(test.create_curse_examples(number_of_examples=10))
    print(test.create_common_curse_examples(number_of_examples=10))
    print(test.create_really_common_curse_examples(number_of_examples=10))
    print(test.create_word_examples(number_of_examples=10))
    print(test.create_short_word_examples(number_of_examples=10))
    print(test.create_number_examples(number_of_examples=10))

