import torch
import json
from llama_index.core.evaluation import EmbeddingQAFinetuneDataset
from llama_index.finetuning import SentenceTransformersFinetuneEngine
from json_utils import create_corpus, create_json
from sftb import TBSTFE
import os

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def create_json_input(filename, output_file, negative_size):
    with open(filename, 'r', encoding='utf-8') as fi:
        dict_sample = json.load(fi)
    content = create_corpus(dict_sample, negative_size)
    create_json(content, output_file)
    print('JSON data created and saved.')


def finetune(json_file, model_id, model_output_path, epochs=3):
    print('Initiating fine tuning.....')
    train_dataset = EmbeddingQAFinetuneDataset.from_json(json_file)
    finetuner = SentenceTransformersFinetuneEngine(
      dataset=train_dataset,
      model_id=model_id,
      model_output_path=model_output_path,
      epochs=epochs
    )
    finetuner.finetune()
    finetuned_model = finetuner.get_finetuned_model()
    finetuned_model.to_json()


def finetune2(json_file, model_id, model_output_path, epochs=3, w_path=os.getcwd()):
    print('Initiating fine tuning.....')
    train_dataset = EmbeddingQAFinetuneDataset.from_json(json_file)
    finetuner = TBSTFE(
      dataset=train_dataset,
      model_id=model_id,
      model_output_path=model_output_path,
      epochs=epochs,
      writer_path=w_path
    )
    finetuner.finetune()
    finetuned_model = finetuner.get_finetuned_model()
    finetuned_model.to_json()
