import os
# from sklearn.model_selection import train_test_split

import sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))
from process_data import HFJSONCreator
from finetuner import final_finetune


MODEL_ID = 'BAAI/bge-small-en-v1.5'
DATASET = 'camel-ai/physics'
FILE_PATH = 'final_json.json'

CORPUS_COUNT = 5
MODEL_OUTPUT = 'embeddings_camelai'


def transform_df(df):
    df['answer'] = df['topic;'] + ' ' + df['sub_topic'] + ' '+df['message_2']
    df['question'] = df['message_1']


camelai = HFJSONCreator(dataset_name=DATASET, transform_data=transform_df,
                        test_ratio=0.2, to_disk=True)

camelai.create_all_dicts()
train_path, test_path, main_dir = camelai.write_dict()

final_finetune(train_path, os.path.join('..', main_dir, FILE_PATH),
               CORPUS_COUNT, MODEL_ID, MODEL_OUTPUT)






# def create_dict(source):
#     dict_train, dict_test = {}, {}
#     df = load_dataset(source, split='train').to_pandas()
#     target_name = source.replace('/', '_').replace('datasets_', '')
#     df['answer'] = df['topic;'] + ' ' + df['sub_topic'] + ' '+df['message_2']
#     train, test = train_test_split(df, test_size=0.2)
#     X_train, y_train = train['message_1'], train['answer']
#     X_test, y_test = test['message_1'], test['answer']
#     for i, (q, a) in enumerate(zip(list(X_train), list(y_train))):
#         dict_train[i] = {'Question': q, 'Reference': a}
#     for i, (q, a) in enumerate(zip(list(X_test), list(y_test))):
#         dict_test[i] = {'Question': q, 'Reference': a}
#     with open(target_name+'_train.json', 'w', encoding='utf-8') as f:
#         json.dump(dict_train, f)
#     with open(target_name+'_test.json', 'w', encoding='utf-8') as f:
#         json.dump(dict_test, f)


# filenames = ['camel-ai/physics']


# dict_f = create_dict(filenames[0])
