from process_data import JSONCreator

# data = save_data('sciq')


class SciQ(JSONCreator):
    def __init__(self, dataset_name, split, test_ratio, to_disk):
        super().__init__(dataset_name, split, test_ratio, to_disk)
        self.df = None
        self.final_dict = {}
        self.to_disk = to_disk
        self.load_data()
        
    def create_dict(self, source):
        if self.save_data:
        self.data = self.load_data()

    def create_json(self):
        for i in range(self.df.shape[0]):
            self.final_dict[i]
    