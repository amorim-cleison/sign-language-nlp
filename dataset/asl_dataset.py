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
                 collated=False,
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

        # Collation:
        if collated:
            data_X, data_y = zip(*data)
            data_X = self.__collate_X(data_X, fields, device)
            data_y = self.__collate_y(data_y, fields, device)

            if len(data_X) > 1:
                data_X = zip(*data_X)
            data = zip(data_X, data_y)

        self.__data = list(data)
        self.__device = device
        self.__batch_first = batch_first
        self.__fields = fields
        self.__vocabs = vocabs
        self.__collated = collated

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

    @property
    def collated(self):
        return self.__collated

    def __getitem__(self, idx):
        if isinstance(idx, list):
            return [self.__data[i] for i in idx]
        else:
            return self.__data[idx]

    def __len__(self):
        return len(self.__data)

    def X(self, fmt=None):
        def as_array(data):
            length = max(map(len, data))
            return np.array([x + [None] * (length - len(x)) for x in data])

        data = AslSliceDataset(self, self.IDX_X)

        if fmt is not None:
            fn = {"array": as_array}
            assert (fmt in fn), "Invalid format"
            data = fn[fmt](data)
        return data

    def y(self, fmt=None):
        def as_array(data):
            return np.array(data)

        data = AslSliceDataset(self, self.IDX_Y)

        if fmt is not None:
            fn = {"array": as_array}
            assert (fmt in fn), "Invalid format"
            data = fn[fmt](data)
        return data

    def __collate_X(self, X, fields=None, device=None):
        return self._process_value(values=X,
                                   field_idx=self.IDX_X,
                                   fields=fields,
                                   device=device)

    def __collate_y(self, y, fields=None, device=None):
        return self._process_value(values=y,
                                   field_idx=self.IDX_Y,
                                   fields=fields,
                                   device=device).squeeze(-1)

    def _process_value(self, values, field_idx, fields=None, device=None):
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
        return o

    def collated(self):
        return AslDataset(dataset=self, data=self.__data, collated=True)

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
