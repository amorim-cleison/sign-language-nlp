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
                    data = zip(*data)
            elif isinstance(data, Dataset):
                data = data

        else:
            dataset, src_vocab, tgt_vocab, = self.__load_data(
                batch_first=batch_first, dataset_args=dataset_args)
            fields = (dataset.fields["src"], dataset.fields["tgt"])
            vocabs = (src_vocab, tgt_vocab)
            data = zip(dataset.src, dataset.tgt)

        # Fix formats:
        data = ((self.__ensure_list(o[self.IDX_X]),
                 self.__ensure_not_list(o[self.IDX_Y])) for o in data)

        self.__data = list(data)
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

        data = AslSliceDataset(self, self.IDX_X)

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

        data = AslSliceDataset(self, self.IDX_Y)

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
        return self._process_value(X, self.IDX_X)

    def __collate_y(self, y):
        return self._process_value(y, self.IDX_Y).squeeze(-1)

    def _process_value(self, values, field_idx):
        field = self.__fields[field_idx]
        values = (self.__ensure_list(x) for x in values)
        return field.process(values, device=self.__device)

    def __ensure_list(self, o):
        # Fix type:
        if isinstance(o, (str, int, float, bool)):
            o = [o]
        elif torch.is_tensor(o):
            if (o.ndim < 1):
                o = o.unsqueeze(-1)
            o = o.tolist()
        elif isinstance(o, np.ndarray):
            o = o.tolist()

        # Remove nulls:
        return [n for n in o if (n is not None)]

    def __ensure_not_list(self, o):
        if isinstance(o, (list, np.ndarray)):
            o = o[0]
        return o

    def truncated(self, length):
        return AslDataset(dataset=self, data=self.__data[0:length])

    def split(self, lengths, return_indices=False, seed=None):
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

        generator = torch.Generator().manual_seed(seed) if seed else None
        splits = random_split(dataset=self,
                              lengths=lengths,
                              generator=generator)

        def get_item(split, return_indices):
            item = AslDataset(dataset=self, data=split)
            if return_indices:
                item = (item, split.indices)
            return item

        return [get_item(s, return_indices) for s in splits]


class AslSliceDataset(SliceDataset):
    def collated(self):
        data = [self.dataset[x][self.idx] for x in self.indices]
        return self.dataset._process_value(data, self.idx)

    def __getitem__(self, i):
        item = super().__getitem__(i)

        if isinstance(item, SliceDataset):
            return AslSliceDataset(dataset=item.dataset,
                                   idx=item.idx,
                                   indices=item.indices)
        else:
            return item
