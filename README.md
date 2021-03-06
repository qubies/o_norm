## Warning
Repository is in Alpha state, there are likely many bugs. Issues are welcome at this point, as are PRs. 

# Introduction
O_Norm is a library designed to enable de-obfuscation of offensive words in text. O_Norm operates only on words that are out of vocabulary (OOV). The two models provided are trained with different percentages of negative samples (25 % and 50 %) where the model trained on 50% negative examples is more precise, and less likely to produce false positives than the 25% model, however this is at the expense of recall.

To use the library you need a model, you can either use a pretrained model or genereate data, and train your own model using the included scripts (train.py and generateTrainingData.py).
model_path is the path to the model directory, as downloaded or created.

# Installation
``` 
git clone https://github.com/qubies/o_norm
```

#### If pretrained models are desired:
download models from: https://drive.google.com/drive/folders/1zCyeeRKgcP5O1u00Bfj-Q7LU7xhexa43 
and extract

Note that the archive contains two models, one trained on 25% negative data, and another trained on 50%. 
The 25% negative model will detect more offensive text at the expense of more false positives.
The 50% model is more conservative and will be more accurate, but detect fewer obfuscations. 

```
pip3 install --user -r o_norm/requirements.txt ./o_norm 
```

# Use
```python
from o_norm import O_Norm
onorm = O_Norm(model_path, # Replace with local path to model (note that the pretrained archive contains two models, only one can be loaded at a time)
    classification_threshold=0.5,
    classification_threshold_short=0.95,
    silent=True,
 )
s = "f u c k"
s = onorm.normalize(s)
print(s)
# {'Normalized': 'fuck', 'Obscene': True, 'Replacement made': [['f u c k', 'fuck']], 'Score': 0.9905826269259627}
```

# Options
The vocabulary files default to o_norm's resouces directory. If you wish to specify a different vocabulary, indicate that when  loading the model with the keyword arg: vocabulary=[path].
This file is expected to be a json list of ALL IV words.
You can alter the vocabulary on an existing model to alter the OOV/IV methods, but this does not alter the trained model or its predictions. You can use this to add or remove words from the standard dictionary if they are causing false positives or negatives.

# Generate Training Data
The script provided for generating training data is generateTrainingData.py:
```
usage: generateTrainingData.py [-h] [--examples EXAMPLES] [--max_length MAX_LENGTH]
                               [--all_curses_percent ALL_CURSES_PERCENT]
                               [--common_curses_percent COMMON_CURSES_PERCENT]
                               [--really_common_curses_percent REALLY_COMMON_CURSES_PERCENT]
                               [--all_negative_words_percent ALL_NEGATIVE_WORDS_PERCENT]
                               [--short_negative_words_percent SHORT_NEGATIVE_WORDS_PERCENT]
                               [--numeric_percent NUMERIC_PERCENT] [--seed SEED] [--output_dir OUTPUT_DIR]
                               [--curse_file CURSE_FILE] [--vocab_file VOCAB_FILE]

Build Curse Training Set

optional arguments:
  -h, --help            show this help message and exit
  --examples EXAMPLES   The base number of examples to generate (default: 10000)
  --max_length MAX_LENGTH
                        The maximum length of an obfuscated curse (default: 50)
  --all_curses_percent ALL_CURSES_PERCENT
                        The percentage of the number of examples that is generated using the full curse dictionary
                        (default: 0.15)
  --common_curses_percent COMMON_CURSES_PERCENT
                        The percentage of the number of examples that is generated using values greater than 1 in
                        the curse dictionary (default: 0.25)
  --really_common_curses_percent REALLY_COMMON_CURSES_PERCENT
                        The percentage of the number of examples that is generated using values of 2 in the curse
                        dictionary (default: 0.15)
  --all_negative_words_percent ALL_NEGATIVE_WORDS_PERCENT
                        The percentage of the number of examples that is generated using the full in the vocabulary
                        dictionary (default: 0.1)
  --short_negative_words_percent SHORT_NEGATIVE_WORDS_PERCENT
                        The percentage of the number of examples that is generated using words with a length of 4
                        or less in the vocabulary dictionary (default: 0.2)
  --numeric_percent NUMERIC_PERCENT
                        The percentage of the number of examples that is generated using numeric strings (default:
                        0.15)
  --seed SEED           Seed the random number generator for repeatable results -- -1 is random (default: -1)
  --output_dir OUTPUT_DIR
                        The directory to write the curses to in json format. if empty they will print to stdout.
                        (default: )
  --curse_file CURSE_FILE
                        A custom json file of curses to train on (default: o_norm)
  --vocab_file VOCAB_FILE
                        A custom vocabulary file used for negative examples (default: o_norm)
```

# Build Model and Train
The script provided for training is train.py:
```
positional arguments:
  training_file         The training data to load
  model_name            The name of your model

optional arguments:
  -h, --help            show this help message and exit
  --model_directory MODEL_DIRECTORY
                        The directory where your models will be saved (default: models)
  --max_length MAX_LENGTH
                        The maximum length of an obfuscated curse (default: 50)
  --num_epochs NUM_EPOCHS
                        The number of epochs to train for (default: 2)
  --batch_size BATCH_SIZE
                        The size of training batches (default: 24)
  --eval_percentage EVAL_PERCENTAGE
                        The percentage of training data to use for evaluation (between 0.0 and 0.90) (default: 0.2)
  --use_cuda USE_CUDA   Use CUDA or not (default: False)
```
