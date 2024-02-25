import os
from finetune_utils import finetune, create_json_input

MODEL_ID = 'BAAI/bge-small-en'
MODEL_INPUT_PATH = MODEL_ID
MODEL_OUTPUT = 'physics_3'
MODEL_OUTPUT_PATH = os.path.join(EMBEDDINGS_PATH, MODEL_OUTPUT)

BASE_JSON_PATH = "..//processed_data//base_data_json"
TEMPLATED_JSON_PATH = "..//processed_data//templated_json"

INPUT_FILE = "camelai//camel-ai_physics_train.json"
if 'camelai' not in os.listdir(TEMPLATED_JSON_PATH):
    os.makedirs(os.path.join(TEMPLATED_JSON_PATH, 'camelAI'))
OUTPUT_JSON_FILE = 'camelai//camel_train.json'

BASE_INPUT_PATH = os.path.join(BASE_JSON_PATH, INPUT_FILE)
TEMPLATED_PATH = os.path.join(TEMPLATED_JSON_PATH, OUTPUT_JSON_FILE)

NUMBER_OF_CORPUS_ANSWERS = 10

if __name__ == "__main__":
    finetune(TEMPLATED_PATH, MODEL_INPUT_PATH, MODEL_OUTPUT_PATH,)
