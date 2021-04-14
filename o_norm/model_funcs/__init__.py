
from simpletransformers.classification import ClassificationModel, ClassificationArgs
from transformers import RobertaConfig, RobertaForMaskedLM
from os import path, mkdir
from distutils.dir_util import copy_tree
from o_norm.resources import MODEL_DIR as model_directory, TOKENIZER_FILES as tokenizer_dir

def build_transformer_model(file_name):
    model_path = path.join(model_directory, file_name)
    if path.exists(model_path):
        response = input(f"Path '{model_path}' already exists for model, overwrite with untrained model (y/n)? ")
        if response.lower()[0] != "y":
            return
    config = RobertaConfig(
        vocab_size=128,
        max_position_embeddings=64+2,
        num_attention_heads=12,
        num_hidden_layers=6,
        type_vocab_size=1,
    )

    model = RobertaForMaskedLM(config=config)
    model.save_pretrained(path.join(model_directory,file_name))
    copy_tree(tokenizer_dir, model_path)
    return 

def copy_model(src_file, dest_file):
    src_path = path.join(model_directory, src_file)
    dest_path = path.join(model_directory, dest_file)
    if not path.exists(dest_path):
        copy_tree(src_path, dest_path)
        return
    else: 
        print(f"WARN: Copy destination '{dest_path}' already exists, please specify a new model destination to copy")

def load_model(file_name, save_name="", model_directory=model_directory, use_cuda=False, num_epochs=2, train_batch_size=128, eval_batch_size=128, tokenizer_dir=tokenizer_dir):
    model_path = path.join(model_directory, file_name)
    save_path = path.join(model_directory, save_name)
    if not path.exists(model_path):
        raise FileNotFoundError("Unable to open '{model_path}': ")
    ca = ClassificationArgs()
    ca.num_train_epochs=num_epochs
    ca.output_dir=save_path
    ca.max_sequence_length=64
    ca.train_batch_size=train_batch_size
    ca.logging_steps=1
    ca.save_steps=-1
    ca.eval_batch_size=eval_batch_size
    ca.tokenizer_name = tokenizer_dir
    ca.n_gpu = 1
    ca.use_multiprocessing=False
    ca.use_multiprocessing_for_evaluation=False
    ca.silent=True

    return ClassificationModel('roberta', model_path, num_labels=141, use_cuda=use_cuda, args=ca)
