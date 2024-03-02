from datasets import load_dataset, load_from_disk
import os
from sklearn.model_selection import train_test_split
from abc import ABC, abstractmethod
import json


class JSONAbstractCreator(ABC):
    def __init__(self, filename, split='train'):
        self.filename = filename
        self.split = split

    @abstractmethod
    def load_data(self):
        pass

    @abstractmethod
    def create_dict(self):
        pass


class HFJSONCreator(JSONAbstractCreator):
    def __init__(self, dataset_name, data_path, transform_data=None, split='train',
                 test_ratio=0, to_disk=True, ):
        self.dataset_name = dataset_name
        self.download_filename = dataset_name.replace('/', '')
        self.data_path = data_path
        self.save_download = 'dataset'
        self.split = split
        self.df = None
        self.test_ratio = test_ratio
        self.to_disk = to_disk
        self.transform_data = transform_data
        self.data_path = data_path
        self.INTERIM_JSON_FILE_NAME = 'interim_json'
        self.interim_json_path = os.path.join(self.data_path, self.INTERIM_JSON_FILE_NAME)
        self.save_download_path = os.path.join(self.data_path, self.save_download)
        if not os.path.exists(self.save_download_path):
            os.makedirs(self.save_download_path)

        self.load_data()

    def load_data(self):
        print('Downloading data.....')
        self.data = load_dataset(self.dataset_name, split=self.split)
        if self.to_disk:
            print('Data downloading... saving data')
            self.filename = self.dataset_name+'_'+self.split
            self.data.save_to_disk(os.path.join(self.save_download_path, self.download_filename))
        else:
            print('Data downloaded...')
        self.df = self.data.to_pandas()

    def split_data(self):
        train, test = train_test_split(self.df, test_size=self.test_ratio)
        return train, test

    def create_dict(self, df):
        self.setup_data(df)
        dict_qa = {i: {'Question': q, 'Reference': a} for i, (q, a) in
                   enumerate(zip(list(df['question']),
                                 list(df['answer'])))
                   if a != ""}
        return dict_qa

    def setup_data(self, df):
        if self.transform_data:
            self.transform_data(df)
            assert 'question' in df.columns
            assert 'answer' in df.columns
            assert isinstance(df.iloc[0]['question'], str)

    def create_all_dicts(self):
        if self.test_ratio == 0:
            dict_qa_train = self.create_dict(self.df)
            dict_qa_test = None
        else:
            train, test = self.split_data()
            dict_qa_train = self.create_dict(train)
            dict_qa_test = self.create_dict(test)
        return dict_qa_train, dict_qa_test

    def write_dict(self):
        output_filename = self.INTERIM_JSON_FILE_NAME+'_'+self.split
        qa_train, qa_test = self.create_all_dicts()
        train_path = os.path.join(self.data_path, output_filename+'.json')
        test_path = os.path.join(self.data_path,
                                 output_filename.replace(self.split, '') +
                                 '_test.json')
        with open(os.path.join(train_path), 'w',
                  encoding='utf-8') as f:
            json.dump(qa_train, f)
        if qa_test:
            with open(os.path.join(test_path), 'w', encoding='utf-8') as f:
                json.dump(qa_test, f)
        print(f'Interim json: {output_filename}.json file written in {self.data_path}')
        return train_path, test_path, self.data_path


class LocalJSONCreator(HFJSONCreator):
    def __init__(self, data_path_name, load_data_fn, transform_data,
                 split='train', test_ratio=0):
        self.path_name = data_path_name
        self.load_data_fn = load_data_fn
        to_disk = False
        dataset_name = None
        super().__init__(dataset_name, transform_data, split='train',
                         test_ratio=0, to_disk=to_disk)

        def load_data(self):
            '''
            if it is a downloaded Huggingface dataset, use:
                dataset = load_from_disk(self.path_name)
                self.df = dataset.to_pandas()
            '''
            self.df = self.load_data_fn(self.path_name)
