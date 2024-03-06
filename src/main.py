import os
# from sklearn.model_selection import train_test_split

from process_data import HFJSONCreator
from dataset_setup import Data_setup
from finetune_utils import create_json_input, final_finetune2

# *************************** UPDATE ALL PARAMETERS BELOW *************************************************

# Path for the folder where you want all the data and embeddings to be stored
MAIN_PATH = '..//camelai_physics' 

# Based embedding model
MODEL_ID = "BAAI/bge-small-en-v1.5"

# Dataset to be trained on
DATASET = 'camel-ai/physics'

# Corpus_count is basically used to mix multiple outputs including the correct response to help tune the embeddings model
CORPUS_COUNT = 5

# Output model name
MODEL_OUTPUT = 'embeddings_camelai'

# If only train set is present, then this ratio splits the data into train and test based on the ratio provided
TEST_RATIO = 0.2

# If the dataset has multiple splits, then you can specify all the splits that you want to process here.
SPLITS = ['train']

# No. of epochs to train
EPOCHS = 3

# Transform function to ensure that the question is stored in df['question'] and the response is stored in df['answer]
def transform_df(df):
    df['answer'] = df['topic;'] + ' ' + df['sub_topic'] + ' '+df['message_2']
    df['question'] = df['message_1']

# creating a model dictionary as {reference_model_name : (trained_model_name, folder_name)}
model_dict = {MODEL_ID: (MODEL_OUTPUT, MAIN_PATH)}  # Just a general structure used if we want to train multiple models on the same dataset


# Once all parameters above are updated, run the python file

for k, v in model_dict.items():
    output = v[0]
    main_path=v[1]
    print(f'Running {k}')
    for split in SPLITS:
        data_camelai = Data_setup(main_path, model_outputs=output, split=split)
        camelai = HFJSONCreator(dataset_name=DATASET,
                                transform_data=transform_df,
                                test_ratio=TEST_RATIO, to_disk=True,
                                data_path=data_camelai.data_path)
        camelai.create_all_dicts()
        train_path, test_path, main_dir = camelai.write_dict()
        if split == 'train':
            final_finetune2(input_path=train_path, output_path=data_camelai.final_json,
                            corpus_count=CORPUS_COUNT, model_input=k,
                            model_output=data_camelai.model_path, w_path=data_camelai.log_path,epochs=EPOCHS)
            if test_path:
                create_json_input(filename=test_path, 
                                  output_file=data_camelai.final_json[:-4]+'test.json',
                                  negative_size=CORPUS_COUNT)
        
        else:
            create_json_input(filename=train_path,
                              output_file=data_camelai.final_json,
                              negative_size=CORPUS_COUNT)




'''
data_camelai = Data_setup(MAIN_PATH, model_outputs=[MODEL_OUTPUT])

camelai = HFJSONCreator(dataset_name=DATASET, transform_data=transform_df,
                        test_ratio=0.2, to_disk=True, data_path=data_camelai.data_path)

camelai.create_all_dicts()
train_path, test_path, main_dir = camelai.write_dict()

final_finetune2(input_path=train_path, output_path=data_camelai.final_json,
                corpus_count=CORPUS_COUNT, model_input=MODEL_ID, 
                model_output=data_camelai.model_path, w_path=data_camelai.log_path)    


MODEL_ID = 'BAAI/bge-small-en-v1.5'
DATASET = 'camel-ai/physics'
FILE_PATH = 'final_json.json'

CORPUS_COUNT = 5
MODEL_OUTPUT = '..//embeddings//embeddings_camelai'
if not os.path.exists(MODEL_OUTPUT):
    os.makedirs(MODEL_OUTPUT)


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
'''