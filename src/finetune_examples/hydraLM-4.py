import os
# from sklearn.model_selection import train_test_split

import sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))
from process_data import HFJSONCreator
from dataset_setup import Data_setup
from finetune_utils import create_json_input, final_finetune2

# Create folder names and filenames

MAIN_PATH1 = '..//hydraLM'
MAIN_PATH2 = '..//hydraLM2'
# EMBEDDINGS_PATH = 'embeddings'
# DATA_PATH = 'data'
# LOG_PATH = 'logs'

MODEL_ID1 = "..//arxiv-30k_physics//data//embeddings//camelai_sciq_arxiv"
MODEL_ID2 = "BAAI/bge-small-en-v1.5"
DATASET = 'HydraLM/physics_dataset_alpaca'
# INTERIM_JSON = 'interim_json.json'
# FINAL_JSON = "final_json.json"

CORPUS_COUNT = 5
MODEL_OUTPUT1 = 'camelai_sciq_arxiv_hydraLM'
MODEL_OUTPUT2 = 'embeddings_hydraLM'
TEST_RATIO = 0.2

EPOCHS = 4

SPLITS = ['train']

def transform_df(df):
    df['question'] = df['input'].str.replace("### Instruction:\n",'')
    df['answer'] = df['output'].str.replace("### Response:\n",'')

# # Creating paths
# data_path = os.path.join(MAIN_PATH, DATA_PATH)
# if not os.path.exists(data_path):
#     os.makedirs(data_path)
# embeddings_path = os.path.join(MAIN_PATH, EMBEDDINGS_PATH)
# if not os.path.exists(embeddings_path):
#     os.makedirs(embeddings_path)
# model1_path = os.path.join(embeddings_path, MODEL_OUTPUT1)
# if not os.path.exists(model1_path):
#     os.makedirs(model1_path)
# model2_path = os.path.join(embeddings_path, MODEL_OUTPUT2)
# if not os.path.exists(model2_path):
#     os.makedirs(model2_path)
# final_json_path = os.path.join(data_path, FINAL_JSON)
# if not os.path.exists(final_json_path):
#     os.makedirs(final_json_path)
# log_path = os.path.join(MAIN_PATH, LOG_PATH)
# if not os.path.exists(log_path):
#     os.makedirs(log_path)

model_dict = {MODEL_ID1: (MODEL_OUTPUT1, MAIN_PATH1), MODEL_ID2: (MODEL_OUTPUT2, MAIN_PATH2)}


model_dict = {MODEL_ID1: (MODEL_OUTPUT1, MAIN_PATH1), MODEL_ID2: (MODEL_OUTPUT2, MAIN_PATH2)}

for k, v in model_dict.items():
    output = v[0]
    main_path = v[1]
    print(f'Running {k}')
    for split in SPLITS:
        data_arxiv = Data_setup(main_path, model_outputs=output, split=split)
        arxiv = HFJSONCreator(dataset_name=DATASET, transform_data=transform_df,
                             test_ratio=TEST_RATIO, to_disk=True,
                             data_path=data_arxiv.data_path)
        arxiv.create_all_dicts()
        train_path, test_path, main_dir = arxiv.write_dict()
        if split == 'train':
            final_finetune2(input_path=train_path, output_path=data_arxiv.final_json,
                            corpus_count=CORPUS_COUNT, model_input=k,
                            model_output=data_arxiv.model_path, w_path=data_arxiv.log_path,epochs=EPOCHS)
            if test_path:
                create_json_input(filename=test_path,
                                  output_file=data_arxiv.final_json.replace('train','test'), # Remove train
                                  negative_size=CORPUS_COUNT)
        else:
            create_json_input(filename=train_path,
                              output_file=data_arxiv.final_json,
                              negative_size=CORPUS_COUNT)

'''
for k, v in model_dict.items():
    output = v[0]
    main_path = v[1]
    print(f'Running {k}')
    data_arxiv = Data_setup(main_path, model_outputs=output)
    arxiv = HFJSONCreator(dataset_name=DATASET, transform_data=None,
                          test_ratio=0.2, to_disk=True, data_path=data_arxiv.data_path)
    arxiv.create_all_dicts()
    train_path, test_path, main_dir = arxiv.write_dict()
    final_finetune2(input_path=train_path, output_path=data_arxiv.final_json,
                    corpus_count=CORPUS_COUNT, model_input=k,
                    model_output=data_arxiv.model_path, w_path=data_arxiv.log_path)
'''x