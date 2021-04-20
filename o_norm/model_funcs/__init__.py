from simpletransformers.classification import ClassificationModel, ClassificationArgs
import json
from transformers import RobertaConfig, RobertaForMaskedLM
from os import path
from distutils.dir_util import copy_tree
from o_norm.resources import RESOURCES
from torch import cuda

# Creates a generic, untrained model that can be loaded by the trainer.
def build_new_transformer_model(
    model_name,
    directory=RESOURCES["MODEL_DIR"],
    tokenizer_dir=RESOURCES["TOKENIZER_FILES"],
    max_length=50,
    num_heads=12,
    num_layers=6,
    curses=RESOURCES["CURSES"],
):
    global MODEL_CURSES
    MODEL_CURSES = curses
    model_path = path.join(directory, model_name)
    if path.exists(model_path):
        response = input(
            f"Path '{model_path}' already exists for model, overwrite with untrained model (y/n)? "
        )
        if response.lower()[0] != "y":
            return
    config = RobertaConfig(
        vocab_size=128,
        max_position_embeddings=max_length + 2,
        num_attention_heads=num_heads,
        num_hidden_layers=num_layers,
        type_vocab_size=1,
    )

    model = RobertaForMaskedLM(config=config)
    model.save_pretrained(path.join(directory, file_name))
    # We assume that the character level tokenizer is wanted for now, but it might be nice to be able to use different tokenizers in the future.
    copy_tree(tokenizer_dir, model_path)
    ca = ClassificationArgs()
    ca.output_dir = model_path
    ca.max_sequence_length = max_length + 2
    ca.tokenizer_name = tokenizer_dir
    model = ClassificationModel(
        "roberta", model_path, num_labels=len(curses) + 1, args=ca
    )
    model.save_model(model_path)
    with open(path.join(model_path, "curses.json"), "w+") as f:
        json.dump(curses, f)
    print(f"Model {model_path} Created, with output size of {len(curses)+1}")

    return


def copy_model(src_file, dest_file):
    src_path = path.join(RESOURCES["MODEL_DIR"], src_file)
    dest_path = path.join(RESOURCES["MODEL_DIR"], dest_file)
    if not path.exists(dest_path):
        copy_tree(src_path, dest_path)
        return
    else:
        print(
            f"WARN: Copy destination '{dest_path}' already exists, please specify a new model destination to copy"
        )


#  def save_model(model, vocab, curses, save_path):


def load_o_norm_model(
    model_path,
    batch_size=12,
    num_epochs=5,
    save_steps=10_000,
    num_gpus=1,
    use_multiprocessing=False,  # Caused some problems during testing
    silent=False,
):
    print(f"Loading O-Norm Model at '{model_path}'...")
    checkpoint_path = path.join(model_path, "checkpoints")
    if not path.exists(model_path):
        raise FileNotFoundError("Unable to open '{model_path}': ")
    curse_path = path.join(model_path, "curses.json")
    if not path.exists(curse_path):
        print(
            f"Unable to find curse file for this model, please put the curse file in '{model_path}' as 'curses.json'"
        )

    with open(curse_path) as f:
        RESOURCES["CURSES"] = json.load(f)
        RESOURCES["TRANSLATOR"] = sorted({x for x in RESOURCES["CURSES"]})
        RESOURCES["TRANSLATOR"].insert(0, "none")

    ca = ClassificationArgs()
    ca.num_train_epochs = num_epochs
    ca.output_dir = checkpoint_path
    ca.train_batch_size = batch_size
    ca.logging_steps = 1
    ca.save_steps = save_steps
    ca.eval_batch_size = batch_size
    ca.n_gpu = num_gpus
    ca.use_multiprocessing = use_multiprocessing
    ca.use_multiprocessing_for_evaluation = use_multiprocessing
    ca.silent = silent

    return ClassificationModel(
        "roberta", model_path, use_cuda=cuda.is_available(), args=ca
    )
