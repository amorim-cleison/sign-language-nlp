import numpy as np
import torch
from skorch.helper import SliceDataset
from torch.utils.data import Dataset

from dataset.builder import DatasetBuilder


class AslDataset(Dataset):

    IDX_X, IDX_Y = 0, 1

    def __init__(self,
                 X=None,
                 y=None,
                 batch_first=False,
                 dataset_args={},
                 dataset=None,
                 device=None,
                 vocab_fmt="itos",
                 **kwargs):
        super(AslDataset).__init__()

        if isinstance(dataset, AslDataset):
            fields = dataset.__fields
            vocabs = dataset.__vocabs
            device = dataset.__device
            batch_first = dataset.__batch_first

            if X is None:
                X, y = dataset.__data
            elif isinstance(X, Dataset):
                X, y = zip(*X)

        elif isinstance(X, SliceDataset) or isinstance(y, SliceDataset):
            if isinstance(X, SliceDataset):
                dataset = X.dataset
                indices = X.indices
            else:
                dataset = y.dataset
                indices = y.dataset
            _X, _y = zip(*(dataset[i] for i in indices))
            self.__init__(dataset=dataset,
                          X=_X,
                          y=_y,
                          batch_first=batch_first,
                          dataset_args=dataset_args,
                          device=device,
                          vocab_fmt=vocab_fmt,
                          **kwargs)
            return

        else:
            dataset, src_vocab, tgt_vocab, = self.__init_data(
                batch_first=batch_first, dataset_args=dataset_args)
            fields = (dataset.fields["src"], dataset.fields["tgt"])
            vocabs = (src_vocab, tgt_vocab)
            X, y = dataset.src, dataset.tgt

        # Vocab format:
        if vocab_fmt == "stoi":
            X, y = self.__init_as_stoi(X=X, y=y, fields=fields, device=device)

        # Compose data:
        X, y = self.__init_as_composed_fixed(X=X, y=y)

        # Setup data:
        data = [None] * (max([self.IDX_X, self.IDX_Y]) + 1)
        data[self.IDX_X] = list(X)
        data[self.IDX_Y] = list(y)
        self.__data = tuple(data)

        # Setup other fields:
        self.__device = device
        self.__batch_first = batch_first
        self.__fields = fields
        self.__vocabs = vocabs
        self.__vocab_fmt = vocab_fmt

    def __getitem__(self, idx):
        def fetch(i):
            return (self.__data[self.IDX_X][i], self.__data[self.IDX_Y][i])

        if isinstance(idx, (list, tuple, np.ndarray)):
            return [fetch(i) for i in idx]
        else:
            return fetch(idx)

    def __len__(self):
        return len(self.__data[self.IDX_X])

    @property
    def batch_first(self):
        return self.__batch_first

    @property
    def vocab_X(self):
        return self.__vocabs[self.IDX_X]

    @property
    def vocab_y(self):
        return self.__vocabs[self.IDX_Y]

    def X(self):
        return AslSliceDataset(self, self.IDX_X)

    def y(self):
        return AslSliceDataset(self, self.IDX_Y)

    def __init_data(self, batch_first, dataset_args):
        dataset_objs = DatasetBuilder().build(batch_first=batch_first,
                                              **dataset_args)
        return dataset_objs["dataset"], dataset_objs[
            "src_vocab"], dataset_objs["tgt_vocab"]

    def __init_as_composed_fixed(self, X, y):
        def compose_X(X, y):
            data, length, label = None, None, y

            # Compose data:
            if isinstance(X, (list, tuple, np.ndarray)):
                if isinstance(X[0], str):
                    data = X
                    length = len(data)
                elif len(X) == 3:
                    data, length, label = X
                elif len(X) == 2:
                    data, length = X
                elif len(X) == 1:
                    data = X
            else:
                data = X

            # Fix formats:
            data = self.__ensure_list(data)
            length = self.__ensure_not_list(length)
            label = self.__ensure_not_list(label)

            # Compose output:
            return (data, length), label

        return zip(*(compose_X(_X, _y) for (_X, _y) in zip(X, y)))

    def __init_as_stoi(self, X, y, fields, device):
        X = [x[0] for x in X]
        data_X = self.__process_to_stoi(values=X,
                                        fields=fields,
                                        field_idx=self.IDX_X,
                                        device=device)
        data_y = self.__process_to_stoi(values=y,
                                        fields=fields,
                                        field_idx=self.IDX_Y,
                                        device=device).squeeze(-1)
        if len(data_X) > 1:
            data_X = zip(*data_X)
        return data_X, data_y

    def __process_to_stoi(self, values, field_idx, fields=None, device=None):
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
        return AslDataset(dataset=self,
                          X=self.X(),
                          y=self.y(),
                          vocab_fmt="stoi")

    def truncated(self, length):
        X = self.__data[self.IDX_X][:length]
        y = self.__data[self.IDX_Y][:length]
        return AslDataset(dataset=self, X=X, y=y)

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
