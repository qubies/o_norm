import json
from os.path import join, exists
import pkg_resources


RESOURCES = pkg_resources.resource_filename("o_norm", "resources")
VIPER_DCES_FILE = join(RESOURCES, "viper_DCES.json")
CURSE_BERTA = join(RESOURCES, "models/CurseBERTa")
CURSES_FILE = join(RESOURCES, "vocab/curses.json")
VOCABULARY_FILE = join(RESOURCES, "vocab/vocabulary.json")
EMOTICONS_FILE = join(RESOURCES, "vocab/emoticons.json")
TOKENIZER_FILES = join(RESOURCES, "tokenizer_files")
MODEL_DIR = join(RESOURCES, "models")

all_resources = [RESOURCES, VIPER_DCES_FILE, CURSE_BERTA, CURSES_FILE, VOCABULARY_FILE, EMOTICONS_FILE, TOKENIZER_FILES, MODEL_DIR]
for resource in all_resources:
    assert exists(resource)

with open(VIPER_DCES_FILE) as f:
    VIPER_DCES = json.load(f)

with open(CURSES_FILE) as f:
    CURSES = json.load(f)

with open(VOCABULARY_FILE) as f:
    VOCABULARY = json.load(f)

with open(EMOTICONS_FILE) as f:
    EMOTICONS = json.load(f)

# NORMALIZE THE DICTIONARIES
CURSES = {x.strip().lower():y for x,y in CURSES.items()}
VOCABULARY = [x.strip().lower() for x in VOCABULARY if len(x.strip()) > 1]
VOCABULARY.extend(["a", "i"]) # add to the dict
EMOTICONS = [x.strip().lower() for x in EMOTICONS if len(x.strip()) > 1]
