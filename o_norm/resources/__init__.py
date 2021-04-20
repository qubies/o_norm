import json
from os.path import join, exists
import pkg_resources


resource_dir = pkg_resources.resource_filename("o_norm", "resources")
RESOURCES = {
    "VIPER_DCES_FILE": join(resource_dir, "viper_DCES.json"),
    "CURSE_BERTA": join(resource_dir, "models/CurseBERTa"),
    "TWENTY_PERCENT": join(resource_dir, "models/20percent"),
    "ZERO_PERCENT": join(resource_dir, "models/0percent"),
    "CURSES_FILE": join(resource_dir, "vocab/curses.json"),
    "VOCABULARY_FILE": join(resource_dir, "vocab/vocabulary.json"),
    "EMOTICONS_FILE": join(resource_dir, "vocab/emoticons.json"),
    "TOKENIZER_FILES": join(resource_dir, "tokenizer_files"),
    "MODEL_DIR": join(resource_dir, "models"),
}


for resource in RESOURCES.values():
    print(resource)
    assert exists(resource)

with open(RESOURCES["VIPER_DCES_FILE"]) as f:
    RESOURCES["VIPER_DCES"] = json.load(f)

with open(RESOURCES["CURSES_FILE"]) as f:
    RESOURCES["CURSES"] = json.load(f)

with open(RESOURCES["VOCABULARY_FILE"]) as f:
    RESOURCES["VOCABULARY"] = json.load(f)

with open(RESOURCES["EMOTICONS_FILE"]) as f:
    RESOURCES["EMOTICONS"] = json.load(f)

# NORMALIZE THE DICTIONARIES
RESOURCES["CURSES"] = {x.strip().lower(): y for x, y in RESOURCES["CURSES"].items()}
RESOURCES["VOCABULARY"] = [
    x.strip().lower() for x in RESOURCES["VOCABULARY"] if len(x.strip()) > 1
]
RESOURCES["EMOTICONS"] = [
    x.strip().lower() for x in RESOURCES["EMOTICONS"] if len(x.strip()) > 1
]
