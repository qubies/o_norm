"""
creates obfuscated text from an input
"""
from functools import partial
import string
import json
import random
from o_norm.resources import RESOURCES

ciel_prob = 0.5


def default(word):
    """
    default return on failed initial probability
    """
    return word


def probability_wrapper(*args, **kwargs):
    """
    decorator function that wraps a subsequent call to an obfuscation function
    with an initial probability to return the initial word
    """

    def inner(func):
        if random.random() < kwargs["initial_probability"]:
            return func
        return default

    return inner


class Viper:
    """
    modifies a text string by randomly replacing chars with a visually
    similar glyph
    """

    def __init__(self):
        self.viper = RESOURCES["VIPER_DCES"]

    def obfuscate(self, word, local_probability=lambda: random.random() * ciel_prob):
        """
        returns the obfuscated word
        """
        if callable(local_probability):
            local_probability = local_probability()
        word = list(word)
        for i, letter in enumerate(word):
            if random.random() < local_probability:
                try:
                    word[i] = random.choice(self.viper[letter])
                except KeyError:
                    continue
        return "".join(word)


def equal_spacing(word):
    """
    returns the string with a random (but constant) number of spaces inserted
    between each of the letters e  x  a  m  p  l  e
    """
    num_spaces = random.randint(1, 3)
    return (" " * num_spaces).join(word)


def random_casing(word, local_probability=lambda: random.random() * ciel_prob):
    """
    returns the strign with random letters capitalized. ExaMPlE
    """
    if callable(local_probability):
        local_probability = local_probability()
    result = ""
    for letter in word:
        # stops it from printing before the first character
        if random.random() < local_probability:
            result += letter.upper()
        else:
            result += letter
    return result


def switch_2_letters(word):
    if len(word) < 3:
        return word
    target = random.randint(0, len(word) - 2)
    w = list(word)
    w[target], w[target + 1] = w[target + 1], w[target]
    return "".join(w)


def remove_one_letter(word):
    if len(word) < 3:
        return word
    target = random.randint(1, len(word) - 1)
    w = list(word)
    del w[target]
    return "".join(w)


leet_dict = {
    "a": ["@", "4", "^"],
    "b": ["6"],
    "c": ["("],
    "e": ["3"],
    "g": ["9", "6"],
    "h": ["#"],
    "i": ["1", "!"],
    "l": ["1"],
    "o": ["0"],
    "s": ["5", "$"],
    "t": ["7"],
    "z": ["2"],
}


def leet_speak(word, local_probability=lambda: random.random() * ciel_prob):
    """
    returns a string with random leet_speak substitutions
    """
    if callable(local_probability):
        local_probability = local_probability()
    result = []
    word = word.lower()
    for letter in word[:]:
        if random.random() < local_probability:
            if letter.lower() in leet_dict:
                result.append(random.choice(leet_dict[letter]))
            else:
                result.append(letter.lower())
        else:
            result.append(letter)
    return "".join(result)


def unequal_spacing(word, local_probability=lambda: random.random() * ciel_prob):
    """
    returns the string with a randomized number of spaces injected
    in between the letters: e    x   amp  l e
    """
    if callable(local_probability):
        local_probability = local_probability()
    mark = random.choice(string.punctuation)
    result = str(word[0])
    for letter in word[1:]:
        # stops it from printing before the first character
        if random.random() < local_probability:
            result += " " * random.randint(1, 3)
        result += letter
    return result


def insert_punctuation(word, local_probability=lambda: random.random() * ciel_prob * 2):
    """
    Inserts a random punctuation mark in a word ex.am.ple
    """
    if callable(local_probability):
        local_probability = local_probability()
    mark = random.choice(string.punctuation)
    if random.random() > 0.5:
        mark = "."
    result = str(word[0])
    for letter in word[1:]:
        # stops it from printing before the first character
        if random.random() < local_probability:
            result += mark
        result += letter
    return result


def replace_QWERTYSMASH(
    word, local_probability=lambda: random.random() * ciel_prob * 2
):
    """
    Inserts a random punctuation mark in a word e@#$^$$e
    """
    if callable(local_probability):
        local_probability = local_probability()
    result = ""
    for i, letter in enumerate(word):
        # stops it from printing before the first character
        if i == 0 and random.random() * 3 < local_probability:
            mark = random.choice(["!", "@", "#", "$", "%", "^", "&", "*", "(", ")"])
            result += mark
        elif random.random() < local_probability:
            mark = random.choice(["!", "@", "#", "$", "%", "^", "&", "*", "(", ")"])
            result += mark
        else:
            result += letter
    return result


class Obfuscator:
    """
    The principal external class.
    Call without arguments, or set initial_probability
    and/or local_probability using kwargs
    """

    def __init__(self, initial_probability=0.1, local_probability=None):
        self.viper = Viper()
        self.initial_probability = initial_probability
        self.local_probability = local_probability
        # methods are called using a single argument (word).
        # They are later wrapped with the probability_wrapper decorator.
        # methods that require local_probability must be passed in a partial
        # (curried) function.
        if local_probability == None:
            self.methods = [
                self.viper.obfuscate,
                equal_spacing,
                unequal_spacing,
                insert_punctuation,
                random_casing,
                replace_QWERTYSMASH,
                switch_2_letters,
                remove_one_letter,
                leet_speak,
            ]
        else:
            self.methods = [
                partial(self.viper.obfuscate, local_probability=local_probability),
                equal_spacing,
                partial(unequal_spacing, local_probability=self.local_probability),
                partial(insert_punctuation, local_probability=self.local_probability),
                partial(random_casing, local_probability=self.local_probability),
                replace_QWERTYSMASH,
                switch_2_letters,
                remove_one_letter,
                partial(leet_speak, local_probability=self.local_probability),
            ]

    def obfuscate(self, word):
        """
        the principal function that returns the obfuscated result
        """
        return probability_wrapper(initial_probability=self.initial_probability)(
            random.choice(self.methods)
        )(word)


GLOBAL_OBFUSCTOR = Obfuscator(initial_probability=1.0)


def obfuscate(word):
    return GLOBAL_OBFUSCTOR.obfuscate(word)


if __name__ == "__main__":
    while 1:
        print(obfuscate(input("Please enter the phrase to obfuscate: ")))
