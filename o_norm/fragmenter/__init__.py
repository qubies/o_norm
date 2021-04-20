import re
from o_norm.resources import RESOURCES

splitter = re.compile(r'(\s|\-|\/|\.|"|\'|\,)')
simple_punct = "’.!,?'\"(){}[]\\;:/|* "

WORD_CHAR_SIZE_THRESHOLD = 1
IV_INCLUDSION_THRESHOLD = 1

stride = 2


class vocab_token:
    def __init__(self, token):
        self.no_punct = token.strip()
        self.text = token
        self.used = False
        #  print(f"Token {token}")
        if not IV(token):
            while len(self.no_punct) > 0 and self.no_punct[-1] in simple_punct:
                self.no_punct = self.no_punct[:-1]
            while len(self.no_punct) > 0 and self.no_punct[0] in simple_punct:
                self.no_punct = self.no_punct[1:]
        self.oov = OOV(self.no_punct)
        self.iv = IV(self.no_punct)

    def use(self):
        self.used = True

    def is_used(self):
        return self.used


def OOV(token):
    t = token.strip().lower()
    return len(t) > 0 and t not in RESOURCES["VOCABULARY"]


def IV(token):
    return token.strip().lower() in RESOURCES["VOCABULARY"]


def tokenize(sentence):
    return [vocab_token(x) for x in splitter.split(sentence)]


def Get_Sequences(sentence):
    tokens = tokenize(sentence)
    seqs = []
    for i, token in enumerate(tokens):
        if token.oov:
            token.use()
            for x in range(i - stride, i + stride):
                if (
                    x < len(tokens)
                    and x >= 0
                    and len(tokens[x].text) <= IV_INCLUDSION_THRESHOLD
                ):  # check for IVs below the threshold
                    tokens[x].use()
    last_used = False
    for token in tokens:
        if (
            token.used and not last_used and token.text != "a"
        ):  # special case for when you say a shit or a f u c k
            seqs.append([token.text])
            last_used = True
        elif token.used and last_used:
            seqs[-1].append(token.text)
        else:
            last_used = False
    return seqs


if __name__ == "__main__":
    print(
        Get_Sequences(
            "wow what a tool!!! :), (:, :-) sjw bulllllshit!!!, this.that th1s-that and sh1t a s h i t"
        )
    )
    while True:
        print(Get_Sequences(input("Enter a test sequence: ")))
