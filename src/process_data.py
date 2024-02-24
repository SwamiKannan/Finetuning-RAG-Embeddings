from datasets import load_dataset
import os
from sklearn.model_selection import train_test_split
from abc import ABC, abstractmethod


class JSONAbstractCreator(ABC):
    def __init__(self, filename, split='train'):
        self.filename = filename
        self.split = split

    @abstractmethod
    def save_data(self):
        pass

    @abstractmethod
    def load_data(self):
        pass

    @abstractmethod
    def create_dict(self):
        pass


class JSONCreator(JSONAbstractCreator):
    def __init__(self, dataset_name, transform_data, split='train', test_ratio=1,
                 to_disk=True, ):
        self.dataset_name = dataset_name
        self.split = split
        self.df = None
        self.test_ratio = test_ratio
        self.to_disk = to_disk
        self.transform_data = transform_data

    def load_data(self):
        self.data = load_dataset(self.dataset_name, split=self)
        if self.to_disk:
            self.filename = self.dataset_name+'_'+self.split
            self.data.save_to_disk(os.path.join("datasets", self.filename))
        self.df = self.data.to_pandas()

    def split_data(self):
        self.train, self.test = train_test_split(self.df, self.test_ratio)

    def create_dict(self):
        dict_qa = {}
        for i in self.df.iterrows():
            dict_qa[i] = {'Question': self.df['question'], 'Reference': 
                          self.df['answer']}


