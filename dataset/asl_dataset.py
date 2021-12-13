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
                 vocab_fmt="itos",
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

        # Vocab format:
        if vocab_fmt == "stoi":
            data = self.__data_stoi(data, fields, device)

        # Fix formats:
        data = self.__fix_formats(data)

        # Compose data:
        data = self.__fix_compose_data(data)

        self.__data = list(data)
        self.__device = device
        self.__batch_first = batch_first
        self.__fields = fields
        self.__vocabs = vocabs
        self.__vocab_fmt = vocab_fmt

    def __load_data(self, batch_first, dataset_args):
        dataset_objs = DatasetBuilder().build(batch_first=batch_first,
                                              **dataset_args)
        return dataset_objs["dataset"], dataset_objs[
            "src_vocab"], dataset_objs["tgt_vocab"]

    def __fix_formats(self, data):
        def fix_y(y):
            return self.__ensure_not_list(y)

        def fix_X(X):
            if isinstance(X, tuple):
                if len(X) == 1:
                    _X = self.__ensure_list(X)
                elif len(X) == 2:
                    X, lengths = X
                    _X = self.__ensure_list(X), self.__ensure_not_list(lengths)
                else:
                    raise Exception("Invalid `X` length")
            else:
                _X = self.__ensure_list(X)
            return _X

        return ((fix_X(o[self.IDX_X]), fix_y(o[self.IDX_Y])) for o in data)

    def __fix_compose_data(self, data):
        def compose(X, y):
            if isinstance(X, (list, tuple)) and \
               (len(X) == 2) and \
               (X[1] == y):
                X = X[0]
            _X = (X, y)
            return (_X, y)

        return (compose(o[self.IDX_X], o[self.IDX_Y]) for o in data)

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
        if isinstance(idx, (list, tuple, np.ndarray)):
            return [self.__data[i] for i in idx]
        else:
            return self.__data[idx]

    def __len__(self):
        return len(self.__data)

    def X(self):
        return AslSliceDataset(self, self.IDX_X)

    def y(self):
        return AslSliceDataset(self, self.IDX_Y)

    def __data_stoi(self, data, fields, device):
        X, y = zip(*data)
        X = [x[0] for x in X]
        data_X = self.__field_stoi(values=X,
                                   field_idx=self.IDX_X,
                                   fields=fields,
                                   device=device)
        data_y = self.__field_stoi(values=y,
                                   field_idx=self.IDX_Y,
                                   fields=fields,
                                   device=device).squeeze(-1)
        if len(data_X) > 1:
            data_X = zip(*data_X)
        return zip(data_X, data_y)

    def __field_stoi(self, values, field_idx, fields=None, device=None):
        if fields is None:
            fields = self.__fields
        if device is None:
            device = self.__device
        field = fields[field_idx]
        values = (self.__ensure_list(x) for x in values)
        return field.process(values, device=device)

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
        elif torch.is_tensor(o):
            if (o.ndim < 1):
                o = o.item()
            else:
                raise Exception("Unexpected ndim for o.")
        return o

    def stoi(self):
        return AslDataset(dataset=self, data=self.__data, vocab_fmt="stoi")

    def truncated(self, length):
        return AslDataset(dataset=self, data=self.__data[0:length])

    def split(self, lengths, indices_only=False, seed=None):
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

        def get_item(split, indices_only):
            if indices_only:
                item = split.indices
            else:
                item = AslDataset(dataset=self, data=split)
            return item

        return [get_item(s, indices_only) for s in splits]


class AslSliceDataset(SliceDataset):
    def __init__(self, dataset, idx=0, indices=None, cpu=False):
        self.__cpu = cpu
        super().__init__(dataset, idx=idx, indices=indices)

    def __getitem__(self, i):
        item = super().__getitem__(i)

        if isinstance(item, SliceDataset):
            item = AslSliceDataset(dataset=item.dataset,
                                   idx=item.idx,
                                   indices=item.indices)
        else:
            if self.__cpu:
                item = self.__ensure_cpu(item)
        return item

    def __ensure_cpu(self, item):
        if torch.is_tensor(item):
            return item.cpu()
        return item

    def cpu(self):
        return AslSliceDataset(dataset=self.dataset,
                               idx=self.idx,
                               indices=self.indices,
                               cpu=True)
