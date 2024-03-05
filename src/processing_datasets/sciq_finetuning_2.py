import os
# from sklearn.model_selection import train_test_split

import sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))
from process_data import HFJSONCreator
from finetuner import final_finetune

MODEL_ID1 = "embeddings_camelai"
MODEL_ID2 = "BAAI/bge-small-en-v1.5"
DATASET = 'camel-ai/physics'
FILE_PATH = 'final_json.json'

CORPUS_COUNT = 5
MODEL_OUTPUT1 = 'camelai_sciq'
MODEL_OUTPUT2 = 'embeddings_sciq'


def transform_df(df):
    df['answer'] = df['topic;'] + ' ' + df['sub_topic'] + ' '+df['message_2']
    df['question'] = df['message_1']


camelai = HFJSONCreator(dataset_name=DATASET, transform_data=transform_df,
                        test_ratio=0.2, to_disk=True)

camelai.create_all_dicts()
train_path, test_path, main_dir = camelai.write_dict()

for model_id, model_output in zip([MODEL_ID1, MODEL_ID2],
                                  [MODEL_OUTPUT1, MODEL_OUTPUT2]):
    final_finetune(train_path, os.path.join('..', main_dir, FILE_PATH),
                   CORPUS_COUNT, model_id, model_output)

















'''
import sys
import os
sys.path.insert(1, os.path.join(sys.path[0], '..'))
from src.process_data import HFJSONCreator
from src.finetuner import finetune, create_json_input, EMBEDDINGS_PATH

# data = save_data('sciq')

MODEL_ID = 'physics_2'
MODEL_INPUT_PATH = os.path.join(EMBEDDINGS_PATH, MODEL_ID)
MODEL_OUTPUT = 'physics_3'
MODEL_OUTPUT_PATH = os.path.join(EMBEDDINGS_PATH, MODEL_OUTPUT)


def transform_df(df):
    df['answer'] = df['support']
    return df


source = 'sciq'

sciq_test_json = HFJSONCreator(source, transform_df, split='validation',
                               test_ratio=0, to_disk=True)
sciq_test_json.create_all_dicts()
sciq_test_json.write_dict()
'''