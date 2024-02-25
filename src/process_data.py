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
    def __init__(self, dataset_name, transform_data=None, split='train',
                 test_ratio=0, to_disk=True):
        self.dataset_name = dataset_name
        self.split = split
        self.df = None
        self.test_ratio = test_ratio
        self.to_disk = to_disk
        self.transform_data = transform_data
        self.path = '..//processed_data//templated_json'
        self.data_path = '..//datasets'
        try:
            os.makedirs(self.path)
        except Exception:
            pass
        try:
            os.mkdir(self.data_path)
        except Exception:
            pass
        self.load_data()

    def load_data(self):
        self.data = load_dataset(self.dataset_name, split=self.split)
        if self.to_disk:
            self.filename = self.dataset_name+'_'+self.split
            self.data.save_to_disk(os.path.join(self.data_path, self.filename))
        self.df = self.data.to_pandas()

    def split_data(self):
        self.train, self.test = train_test_split(self.df, self.test_ratio)

    def create_dict(self, df):
        self.setup_data()
        dict_qa = {i: {'Question': q, 'Reference': a} for i, (q, a) in
                   enumerate(zip(list(df['question']), list(df['answer'])))
                   if a != ""}
        return dict_qa

    def setup_data(self):
        if self.transform_data:
            self.transform_data(self.df)

    def create_all_dicts(self):
        if self.test_ratio == 0:
            dict_qa_train = self.create_dict(self.df)
            dict_qa_test = None
        else:
            self.split_data()
            dict_qa_train = self.create_dict(self.train)
            dict_qa_test = self.create_dict(self.test)
        return dict_qa_train, dict_qa_test

    def write_dict(self):
        if self.test_ratio == 0:
            output_filename = self.dataset_name.replace('/', '')+'_'+self.split
            qa_train, qa_test = self.create_all_dicts()
            with open(os.path.join(self.path, output_filename+'.json'), 'w',
                      encoding='utf-8') as f:
                json.dump(qa_train, f)
            if qa_test:
                with open(os.path.join(self.path, output_filename.replace(
                    self.split, ''+'_test.json')),
                          'w', encoding='utf-8') as f:
                    json.dump(qa_test, f)
        print(f'{output_filename}.json file written in {self.path}')


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
