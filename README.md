# installation

### clone the repo
``` pip3 install --user -r o_norm/requirements.txt ./o_norm ```

# use
```python
import o_norm
s = "f u c k"
s = o_norm.normalize(s)
```

# options
You can configure o_norm's sensitivity in two ways, either by changing the model, or changing the threshold value.
Both are set using `set_options(classification_threshold=0.5, model_type="common")`, where classification_threshold is a value between 0 and 1 where 1 indicates absolute certainty of a curse, and 0 means everything is a curse.
if the classification_threshold is below 0.5, then a match that is less than the top score can be returned (the second place match for the OOV input token)
The model_type can be either "common" or "full", with "full" containing 519 curses, and "common" containing 141. The default is common as the reduction in output space makes the model more conservative on replacements. 

If there are particular tokens you would like to include in the vocabulary, the vocabulary file is in the resources folder, and is expected to be a json list. 

# Train
The script provided for generating training data is generateTrainingData.py:
```
usage: generateTrainingData.py [-h] [--examples EXAMPLES] [--max_length MAX_LENGTH]
                               [--negative_percent NEGATIVE_PERCENT] [--all_curses_percent ALL_CURSES_PERCENT]
                               [--strong_curses_percent STRONG_CURSES_PERCENT]
                               [--all_negative_words_percent ALL_NEGATIVE_WORDS_PERCENT]
                               [--short_negative_words_percent SHORT_NEGATIVE_WORDS_PERCENT] [--seed SEED]
                               [--output_file OUTPUT_FILE]

Build Curse Training Set

optional arguments:
  -h, --help            show this help message and exit
  --examples EXAMPLES   The base number of examples to generate (default: 10000)
  --max_length MAX_LENGTH
                        The maximum length of an obfuscated curse (default: 50)
  --negative_percent NEGATIVE_PERCENT
                        The percentage of negative examples, between 0.0 and 0.99 (default: 0.5)
  --all_curses_percent ALL_CURSES_PERCENT
                        The percentage of the number of examples that is generated using the full curse dictionary
                        (default: 0.25)
  --strong_curses_percent STRONG_CURSES_PERCENT
                        The percentage of the number of examples that is generated using the strong curse
                        dictionary (default: 0.25)
  --all_negative_words_percent ALL_NEGATIVE_WORDS_PERCENT
                        The percentage of the number of examples that is generated using the full in the
                        vocabulary dictionary (default: 0.25)
  --short_negative_words_percent SHORT_NEGATIVE_WORDS_PERCENT
                        The percentage of the number of examples that is generated using words with a length of 4
                        or less in the vocabulary dictionary (default: 0.25)
  --seed SEED           Seed the random number generator for repeatable results -- -1 is random (default: -1)
  --output_file OUTPUT_FILE
                        The file to write the curses to in json format. if empty they will print to stdout.
                        (default: )
```
