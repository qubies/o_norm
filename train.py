#!/usr/bin/env python3
import argparse
from o_norm import resources

parser = argparse.ArgumentParser(description=f"Train a new classier using a json file generated with o_norm. This trainer uses the resource files for vocabulary, which are in:\n\tMain Vocabulary: '{resources.VOCABULARY_FILE}'\n\tMain Curse List: '{resources.CURSES_FILE}'", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('training_file', type=str, help="The training data to load")
parser.add_argument('model_name', type=str, help="The name of your model") 
parser.add_argument('--model_directory', type=str, help="The directory where your models will be saved", default="models")
parser.add_argument('--max_length', type=int, help="The maximum length of an obfuscated curse", default=50)
parser.add_argument('--num_epochs', type=int, help="The number of epochs to train for", default=2)
parser.add_argument('--batch_size', type=int, help="The size of training batches", default=24)
parser.add_argument('--eval_percentage', type=float, help="The percentage of training data to use for evaluation (between 0.0 and 0.90)", default=0.2)
parser.add_argument('--use_cuda', type=bool, help="Use CUDA or not", default=False)


args = parser.parse_args()
eval_fraction=args.eval_percentage
model_directory = args.model_directory
tokenizer_dir = resources.TOKENIZER_FILES
base_file = "base_transformer_model"
model_name = args.model_name
num_epochs = args.num_epochs
train_batch_size = args.batch_size
eval_batch_size = args.batch_size
use_cuda = args.use_cuda
training_data=args.training_file
max_length = args.max_length

import pandas as pd
import json
import sys
from os import path, mkdir
from distutils.dir_util import copy_tree
from functools import partial
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score

from o_norm.model_funcs import copy_model, load_model, build_transformer_model



#verify args
if not (eval_fraction>=0.0 and eval_fraction<=0.9):
    print(f"eval_percentage must be between 0.0 and 0.9 inclusive", file=sys.stderr)
    sys.exit(2)

if not path.exists(model_directory):
    try:
        os.mkdir(model_directory)
    except:
        print(f"Unable to open model directory:'{model_directory}'", file=sys.stderr)
        sys.exit(2)

if not path.exists(training_data):
    print(f"Unable to find training data at:'{training_data}'", file=sys.stderr)
    sys.exit(2)

if len(model_name) < 1:
    print(f"Model name is required, recieved name: '{model_name}'", file=sys.stderr)
    sys.exit(2)


evals = []

def load_training_data():
    print("Loading Training Data...")
    with open(training_data) as f:
        global curses
        global train_df
        global test_df
        x_train, x_test, y_train, y_test = train_test_split(*json.load(f), test_size=0.2, random_state=24)
        print(f"Loaded {len(x_train)} training examples and {len(x_test)} test examples")

        curses = set((y[1][0] for y in y_train if len(y[1]) > 0))
        for y in y_test:
            if len(y[1]) > 0:
                curses.add(y[1][0])

        curses = sorted(list(curses))
        curses.insert(0, "none")
        translator = {x:i for i,x in enumerate(curses)}
        assert(curses[0] == "none")

        print(f"output size: {len(curses)}")

        y_train = [translator[x[1][0]] if x[0]==1 else translator["none"] for x in y_train ] 
        train_df = pd.DataFrame({"text":x_train, "labels":y_train})
        print(train_df.head())

        y_test = [translator[x[1][0]] if x[0]==1 else translator["none"] for x in y_test ] 
        test_df = pd.DataFrame({"text":x_test, "labels":y_test})
        train_df["labels"]=train_df["labels"].astype(int)
        print(f"Max: {train_df['labels'].max()} Min: {train_df['labels'].min()}")
        test_df["labels"]=test_df["labels"].astype(int)
        print(f"Max: {test_df['labels'].max()} Min: {test_df['labels'].min()}")
        assert(train_df.isnull().values.any() == False)
        assert(test_df.isnull().values.any() == False)
    print("Done!")


def evaluate_model(model, test):
    print("Evaluating model....")
    result, model_outputs, wrong_predictions = model.eval_model(test, f1_micro=partial(f1_score, average='micro'), f1_macro=partial(f1_score, average='macro'), precision_micro=partial(precision_score, average='micro'), precision_macro=partial(precision_score, average='macro'), recall_micro=partial(recall_score, average='micro'), recall_macro=partial(recall_score, average="macro"))
    print(f"Result: {result}")
    evals.append(result)
    print("Done Evaluation")

#create a base model with the char level tokenizer
if not path.exists(path.join(model_directory, base_file)):
    build_transformer_model(base_file)

#get the training data
load_training_data()

copy_model(base_file, model_name)
model = load_model(model_name, save_name=f"{path.join(model_name, 'checkpoints')}", num_labels=len(curses), use_cuda=use_cuda, train_batch_size=train_batch_size, eval_batch_size=eval_batch_size, silent=False, max_length=max_length+2, num_epochs=num_epochs)
model.train_model(train_df)
evaluate_model(model, test_df)

with open(path.join(model_directory, "evaluation.json"), "w+") as f:
    json.dump(evals, f)

