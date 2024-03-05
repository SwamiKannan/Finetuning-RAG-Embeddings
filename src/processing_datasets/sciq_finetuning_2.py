import os
# from sklearn.model_selection import train_test_split

import sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))
from process_data import HFJSONCreator
# from finetuner import final_finetune2
from dataset_setup import Data_setup
from finetune_utils import create_json_input, final_finetune2


MAIN_PATH1 = '..//sciq_physics'
MAIN_PATH2 = '..//sciq_physics2'

MODEL_ID1 = "..//camelai_physics//data//embeddings//embeddings_camelai"
MODEL_ID2 = "BAAI/bge-small-en-v1.5"
DATASET = 'allenai/sciq'

CORPUS_COUNT = 5
MODEL_OUTPUT1 = 'camelai_sciq'
MODEL_OUTPUT2 = 'embeddings_sciq'
TEST_RATIO = 0

SPLITS = ['train', 'validate','test']
def transform_df(df):
    df['answer'] = df['correct_answer'] + ' ' + df['support']


model_dict = {MODEL_ID1: (MODEL_OUTPUT1, MAIN_PATH1), MODEL_ID2: (MODEL_OUTPUT2, MAIN_PATH2)}

for k, v in model_dict.items():
    output = v[0]
    main_path=v[1]
    print(f'Running {k}')
    for split in SPLITS:
        data_sciq = Data_setup(main_path, model_outputs=output, split=split)
        sciq = HFJSONCreator(dataset_name=DATASET, transform_data=transform_df,
                         test_ratio=TEST_RATIO, to_disk=True, data_path=data_sciq.data_path)
        sciq.create_all_dicts()
        train_path, test_path, main_dir = sciq.write_dict()
        if split == 'train':
            final_finetune2(input_path=train_path, output_path=data_sciq.final_json,
                            corpus_count=CORPUS_COUNT, model_input=k,
                            model_output=data_sciq.model_path, w_path=data_sciq.log_path)
            if test_path:
                create_json_input(filename=test_path, output_file=data_sciq.final_json[:-4]+'_test.json',
                                  negative_size=CORPUS_COUNT)
        
        else:
            create_json_input(filename=train_path, output_file=data_sciq.final_json',
                              negative_size=CORPUS_COUNT)
        
















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
