import numpy as np
import torch
from skorch.helper import SliceDataset
from torch.utils.data import Dataset

from dataset.builder import DatasetBuilder


class AslDataset(Dataset):

    IDX_X, IDX_Y = 0, 1

    def __init__(self,
                 batch_first=False,
                 dataset_args={},
                 dataset=None,
                 device=None,
                 data=None,
                 **kwargs):
        super(AslDataset).__init__()

        if isinstance(dataset, AslDataset):
            fields = dataset.__fields
            vocabs = dataset.__vocabs
            device = dataset.__device
            batch_first = dataset.__batch_first

            if data is None:
                data = dataset.__data
            elif isinstance(data, tuple):
                if len(data) == 2:
                    assert (len(data[self.IDX_X]) == len(data[self.IDX_Y])), \
                        f"Indexes {self.IDX_X} and {self.IDX_Y} must have "\
                        f"the same length."
                    data = list(zip(*data))
            elif isinstance(data, Dataset):
                data = list(data)

            # Fix X format:
            data = [(self.__ensure_list(X), y) for (X, y) in data]

        else:
            dataset, src_vocab, tgt_vocab, = self.__load_data(
                batch_first=batch_first, dataset_args=dataset_args)
            fields = (dataset.fields["src"], dataset.fields["tgt"])
            vocabs = (src_vocab, tgt_vocab)
            data = list(zip(dataset.src, dataset.tgt))

        self.__data = data
        self.__device = device
        self.__batch_first = batch_first
        self.__fields = fields
        self.__vocabs = vocabs

    def __load_data(self, batch_first, dataset_args):
        dataset_objs = DatasetBuilder().build(batch_first=batch_first,
                                              **dataset_args)
        return dataset_objs["dataset"], dataset_objs[
            "src_vocab"], dataset_objs["tgt_vocab"]

    @property
    def vocab_X(self):
        return self.__vocabs[self.IDX_X]

    @property
    def vocab_y(self):
        return self.__vocabs[self.IDX_Y]

    @property
    def batch_first(self):
        return self.__batch_first

    def __getitem__(self, idx):
        return self.__data[idx]

    def __len__(self):
        return len(self.__data)

    def X(self, fmt=None, collated=False):
        def as_array(data):
            length = max(map(len, data))
            return np.array([x + [None] * (length - len(x)) for x in data])

        data = SliceDataset(self, self.IDX_X)

        if fmt is not None:
            fn = {"array": as_array}
            assert (fmt in fn), "Invalid format"
            data = fn[fmt](data)

        if collated:
            data = self.collate_X(data)
        return data

    def y(self, fmt=None, collated=False):
        def as_array(data):
            return np.array(data)

        data = SliceDataset(self, self.IDX_Y)

        if fmt is not None:
            fn = {"array": as_array}
            assert (fmt in fn), "Invalid format"
            data = fn[fmt](data)

        if collated:
            data = self.__collate_y(data)
        return data

    def collate(self, data):
        X, y = zip(*data)
        X, lengths = self.__collate_X(X)
        y = self.__collate_y(y)
        return {"X": X, "lengths": lengths, "y": y}, y

    def __collate_X(self, X):
        field = self.__fields[self.IDX_X]
        return self.__process_value(X, field)

    def __collate_y(self, y):
        field = self.__fields[self.IDX_Y]
        return self.__process_value(y, field).squeeze(-1)

    def __process_value(self, values, field):
        values = (self.__ensure_list(x) for x in values)
        return field.process(values, device=self.__device)

    def __ensure_list(self, x):
        # Fix type:
        if isinstance(x, (str, int, float, bool)):
            x = [x]
        elif torch.is_tensor(x):
            if (x.ndim < 1):
                x = x.unsqueeze(-1)
            x = x.tolist()
        elif isinstance(x, np.ndarray):
            x = x.tolist()

        # Remove nulls:
        x = [o for o in x if (o is not None)]

        return x

    def truncated(self, length):
        return AslDataset(dataset=self, data=self.__data[0:length])

    def split(self, lengths):
        from torch.utils.data import random_split

        def parse_int(len, total_len):
            if isinstance(len, float):
                len = round(len * total_len)
            assert isinstance(len, int)
            return len

        if not isinstance(lengths, list):
            lengths = [lengths]

        total_len = len(self)
        lengths = [parse_int(len, total_len) for len in lengths]
        sum_lengths = sum(lengths)
        assert (sum_lengths <= total_len)
        remainder = total_len - sum(lengths)

        if remainder > 0:
            lengths.append(remainder)

        splits = random_split(self, lengths)
        return [AslDataset(dataset=self, data=s) for s in splits]
