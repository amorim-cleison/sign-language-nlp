from torch.utils.data import Dataset

from dataset.builder import DatasetBuilder


class AslDataset(Dataset):
    def __init__(self,
                 dataset_dir,
                 fields,
                 samples_min_freq,
                 composition_strategy,
                 batch_first,
                 device=None,
                 debug=False,
                 **kwargs):
        super(AslDataset).__init__()
        dataset_objs = DatasetBuilder().build(
            dataset_dir=dataset_dir,
            fields=fields,
            samples_min_freq=samples_min_freq,
            composition_strategy=composition_strategy,
            debug=debug,
            batch_first=batch_first)
        dataset = dataset_objs["dataset"]
        self.__examples = dataset.examples
        self.__fields = dataset.fields
        self.__vocabs = dataset_objs["vocabs"]
        self.__batch_first = batch_first
        self.__device = device

    @property
    def src_vocab(self):
        return self.__vocabs["src"]

    @property
    def tgt_vocab(self):
        return self.__vocabs["tgt"]

    @property
    def batch_first(self):
        return self.__batch_first

    def collate(self, data):
        X, y = zip(*data)
        X, lengths = self.collate_input(X)
        y = self.collate_target(y)
        return {"X": X, "lengths": lengths, "y": y}, y

    def collate_input(self, X):
        return self.__process_value(X, "src")

    def collate_target(self, y):
        return self.__process_value(y, "tgt").squeeze()

    def __getitem__(self, idx):
        item = self.__examples[idx]
        return item.src, item.tgt

    def __len__(self):
        return len(self.__examples)

    def __process_value(self, values, field_name):
        field = self.__fields[field_name]
        return field.process(values, device=self.__device)
