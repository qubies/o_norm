from o_norm.curses import *
from o_norm.print_utils import print_banner_completion_wrapper

import argparse
import sys
import numpy as np

parser = argparse.ArgumentParser(description="Build Curse Training Set", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--examples', type=int, help="The base number of examples to generate", default=10_000)
parser.add_argument('--max_length', type=int, help="The maximum length of an obfuscated curse", default=50)
parser.add_argument('--negative_percent', type=float, help="The percentage of negative examples, between 0.0 and 0.99", default=0.5)
parser.add_argument('--all_curses_percent', type=float, help="The percentage of the number of examples that is generated using the full curse dictionary", default=0.25)
parser.add_argument('--strong_curses_percent', type=float, help="The percentage of the number of examples that is generated using the strong curse dictionary", default=0.25)
parser.add_argument('--all_negative_words_percent', type=float, help="The percentage of the number of examples that is generated using the full in the vocabulary dictionary", default=0.25)
parser.add_argument('--short_negative_words_percent', type=float, help="The percentage of the number of examples that is generated using words with a length of 4 or less in the vocabulary dictionary", default=0.25)
parser.add_argument('--seed', type=int, help="Seed the random number generator for repeatable results -- -1 is random", default=-1)
parser.add_argument('--output_file', type=str, help="The file to write the curses to in json format. if empty they will print to stdout.", default="")

args = parser.parse_args()
num_examples = args.examples
MAX_LENGTH = args.max_length
output_file = args.output_file
seed=args.seed

if seed != -1:
    np.random.seed(seed)

generated_single_curse_word_examples = num_examples*args.all_curses_percent
generated_strong_curse_examples = num_examples*args.strong_curses_percent
generated_single_good_word_examples = num_examples*args.all_negative_words_percent
generated_small_clean_word_examples = num_examples*args.short_negative_words_percent

@print_banner_completion_wrapper("Building Data Set")
def build_data(MAX_LENGTH, generated_single_good_word_examples, generated_single_curse_word_examples, generated_common_curse_examples, generated_small_clean_word_examples):
    global curse_percent
    x_train = []
    y_train = []
    
    new_x, new_y = create_really_common_curse_examples(
        number_of_examples=generated_strong_curse_examples//2, window_size=MAX_LENGTH)
    x_train += new_x
    y_train += new_y

    new_x, new_y = create_common_curse_examples(
            number_of_examples=generated_strong_curse_examples, window_size=MAX_LENGTH)
    x_train += new_x
    y_train += new_y

    new_x, new_y = create_full_curse_examples(
        number_of_examples=generated_single_curse_word_examples, window_size=MAX_LENGTH)
    x_train += new_x
    y_train += new_y

    new_x, new_y = create_good_word_examples(
        number_of_examples=generated_single_good_word_examples, window_size=MAX_LENGTH)
    x_train += new_x
    y_train += new_y

    new_x, new_y = create_small_good_word_examples(
        number_of_examples=generated_small_clean_word_examples, window_size=MAX_LENGTH)
    x_train += new_x
    y_train += new_y

    new_x, new_y = create_number_examples(number_of_examples=generated_small_clean_word_examples, window_size=MAX_LENGTH)
    x_train += new_x
    y_train += new_y

    curse_count=0
    for y in y_train:
        curse_count += y[0]

    curse_percent = round(curse_count/len(y_train)*100, 1)

    return x_train, y_train


x_train, y_train = build_data(MAX_LENGTH, generated_single_good_word_examples, generated_single_curse_word_examples,
                            generated_strong_curse_examples, generated_small_clean_word_examples)

for i, x in enumerate(x_train):
    #check for errors
    if y_train[i][0] == 1 and y_train[i][1] == []:
        print(f"Sorry: 'Example: {i} is x:'{x}' is a curse, and its not in y, y:'{y_train[i]}'", file=sys.stderr)
        sys.exit(2)
    elif y_train[i][0] == 0 and y_train[i][1] != []:
        print(f"Sorry: 'Example: {i} is x:'{x}' is not a curse, and it is in y, y:'{y_train[i]}'", file=sys.stderr)
    #set none to 'none'
    elif y_train[i][0] == 0:
        y_train[i][1] == ["none"]

#write out the file
if output_file != "":
    with open(output_file, "w+") as f:
        json.dump([x_train, y_train], f)

# or display it
# note that dispaly is not the same as output just due to the way the trainer consumes the data.
# this could cause problems with redirect, but its mostly intended for debugging as its designed
else:
    print(json.dumps({x:y for x, y in zip(x_train, y_train)}, indent=4))

print(f"Curse percent: {curse_percent}%")
