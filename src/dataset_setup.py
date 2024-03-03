import os


class Data_setup():
    def __init__(self, main_path, model_outputs=None, corpus_count=5):
        self.main_path = main_path
        self.EMBEDDINGS_PATH = 'embeddings'
        self.DATA_PATH = 'data'
        self.LOG_PATH = ''
        self.FINAL_JSON = "final_json.json"
        self.CORPUS_COUNT = corpus_count
        self.data_path = self.make_dirs(self.DATA_PATH)
        self.embeddings_path = self.make_dirs(folder_name=self.EMBEDDINGS_PATH, path=self.data_path)
        self.log_path = self.make_dirs(folder_name=self.LOG_PATH, path=self.data_path)
        self.final_json = os.path.join(self.data_path, self.FINAL_JSON)
        self.corpus_count = self.CORPUS_COUNT
        self.model_path = self.make_dirs(folder_name=model_outputs, path=self.embeddings_path)

    def make_dirs(self, folder_name, path=None):
        if path:
            temp_path = os.path.join(path, folder_name)
        else:
            temp_path = os.path.join(self.main_path, folder_name)
        if not os.path.exists(temp_path):
            os.makedirs(temp_path)
        return temp_path
