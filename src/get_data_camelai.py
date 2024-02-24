from datasets import load_dataset
import os
import json
from sklearn.model_selection import train_test_split


def save_data(filename):
    data = load_dataset(filename, split='train')
    data.save_to_disk(os.path.join("datasets", filename))
    return data


def create_dict(source):
    dict_train, dict_test = {}, {}
    df = load_dataset(source, split='train').to_pandas()
    target_name = source.replace('/', '_').replace('datasets_', '')
    df['answer'] = df['topic;'] + ' ' + df['sub_topic'] + ' '+df['message_2']
    train, test = train_test_split(df, test_size=0.2)
    X_train, y_train = train['message_1'], train['answer']
    X_test, y_test = test['message_1'], test['answer']
    for i, (q, a) in enumerate(zip(list(X_train), list(y_train))):
        dict_train[i] = {'Question': q, 'Reference': a}
    for i, (q, a) in enumerate(zip(list(X_test), list(y_test))):
        dict_test[i] = {'Question': q, 'Reference': a}
    with open(target_name+'_train.json', 'w', encoding='utf-8') as f:
        json.dump(dict_train, f)
    with open(target_name+'_test.json', 'w', encoding='utf-8') as f:
        json.dump(dict_test, f)


filenames = ['camel-ai/physics']


dict_f = create_dict(filenames[0])
