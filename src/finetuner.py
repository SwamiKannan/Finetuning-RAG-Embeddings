import os
from finetune_utils import finetune, create_json_input

MODEL_ID = 'BAAI/bge-small-en'
MODEL_INPUT_PATH = MODEL_ID
MODEL_OUTPUT = 'physics_3'
MODEL_OUTPUT_PATH = os.path.join(EMBEDDINGS_PATH, MODEL_OUTPUT)

TEMPLATED_JSON_PATH = "..//processed_data//templated_json"

if __name__ == "__main__":
    finetune(TEMPLATED_PATH, MODEL_INPUT_PATH, MODEL_OUTPUT_PATH,)
