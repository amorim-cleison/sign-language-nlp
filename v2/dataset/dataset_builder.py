import json
import tempfile

from commons.log import auto_log_progress, log
from commons.util import (delete_file, exists, filename, filter_files,
                          read_json, save_items)
from torchtext.data import Field, TabularDataset

from .tokens import BOS_WORD, EOS_WORD, PAD_WORD, UNK_WORD

import pandas as pd
import numpy as np


class DatasetBuilder():

    BATCH_SIZE = 10

    def __init__(self):
        pass

    def build(self, debug, dataset_args, **kwargs):
        try:
            return self.do_build(debug, **dataset_args)
        except Exception as e:
            raise Exception(f"Failed to build dataset: {repr(e)}")

    def do_build(self,
                 debug,
                 dataset_dir,
                 fields,
                 samples_min_freq,
                 composition_strategy,
                 train_split_ratio,
                 val_split_ratio=None,
                 **kwargs):
        tmp = tempfile.NamedTemporaryFile(delete=False)

        try:
            # Write transient working file:
            self.write_working_file(path=tmp.name,
                                    dataset_dir=dataset_dir,
                                    min_freq=samples_min_freq,
                                    debug=debug)

            # Dataset:
            dataset, src_vocab, tgt_vocab, file_vocab = \
                self.create_dataset(
                    path=tmp.name,
                    fields=fields,
                    composition_strategy=composition_strategy)

            # Splits:
            train_data, val_data, test_data = self.split_dataset(
                dataset=dataset,
                train_split_ratio=train_split_ratio,
                val_split_ratio=val_split_ratio)

            return {
                "dataset": dataset,
                "src_vocab": src_vocab,
                "tgt_vocab": tgt_vocab,
                "file_vocab": file_vocab,
                "train_data": train_data,
                "val_data": val_data,
                "test_data": test_data
            }
        finally:
            tmp.close()
            delete_file(tmp.name)

    def write_working_file(self, path, dataset_dir, min_freq, debug):
        def prefix(f):
            return f.stem.split('-')[0]

        def prepare_row(p):
            data = read_json(p)
            data["file"] = filename(p)
            return json.dumps(data).replace('null', '""')

        assert exists(dataset_dir), "Invalid dataset directory"
        files = filter_files(dataset_dir, ext="json", path_as_str=False)

        if debug:
            files = files[:100]

        # Group and filter data:
        df = pd.DataFrame({"file": files})
        df["prefix"] = df["file"].apply(prefix)
        df = df\
            .groupby(["prefix"])\
            .filter(lambda x: x["file"].count() >= min_freq)

        # Separate data chunks:
        n_samples = len(df)
        n_batches = n_samples / self.BATCH_SIZE
        batches = np.array_split(df, n_batches)
        msg = "Processing dataset... "

        for batch in auto_log_progress(batches, message=msg):
            rows = batch["file"].apply(prepare_row)
            save_items(rows, path, True)

    def create_dataset(self, path, fields, composition_strategy):
        def preprocess_src(rows):
            return self.preprocess_src(rows, fields, composition_strategy)

        # FIX_LENGTH = 10  # FIXME: verify this -> check if it's possible to count dinamically

        # Fields:
        SRC = Field(pad_token=PAD_WORD,
                    unk_token=UNK_WORD,
                    preprocessing=preprocess_src,
                    fix_length=None)
        TGT = Field(
            is_target=True,
            pad_first=True,
            init_token=BOS_WORD,
            #  eos_token=EOS_WORD,
            pad_token=PAD_WORD,
            fix_length=None)
        FILE = Field()

        # Dataset:
        dataset = TabularDataset(path=path,
                                 format="json",
                                 fields={
                                     'frames.phonology': ('src', SRC),
                                     'label': ('tgt', TGT),
                                     'file': ('file', FILE)
                                 })

        # Vocabs:
        SRC.build_vocab(dataset.src)
        TGT.build_vocab(dataset.tgt)
        FILE.build_vocab(dataset.file)
        return dataset, SRC.vocab, TGT.vocab, FILE.vocab

    def split_dataset(self, dataset, train_split_ratio, val_split_ratio=None):
        if not val_split_ratio:
            val_split_ratio = 0
        assert (train_split_ratio + val_split_ratio <=
                1), "Invalid train/val split ratios."
        test_split_ratio = 1 - train_split_ratio - val_split_ratio

        # ratios (parameter): [ train, test, val]
        # output: (train, [val,] test)
        splits = dataset.split(
            split_ratio=[train_split_ratio, test_split_ratio, val_split_ratio])

        if len(splits) == 3:
            train, val, test = splits
        else:
            train, test = splits
            val = None
        return train, val, test

    def preprocess_src(self, rows, fields, composition_strategy):
        STRATEGY_FN = {
            "all_values": self.compose_all_values,
            "as_words": self.compose_as_words,
            "as_words_norm": self.compose_as_words_norm,
            "as_sep_feat": self.compose_sep_feat,
        }
        assert (composition_strategy in STRATEGY_FN
                ), f"Unknown composition strategy: '{composition_strategy}'"

        try:
            fn = STRATEGY_FN[composition_strategy]
            return fn(rows, fields)
        except Exception as e:
            raise Exception(
                f"There was an error while running strategy "
                f"'{composition_strategy}' in FieldComposer: {repr(e)}")

    def compose_all_values(self, rows, fields):
        """
        Example:
        `
        left_back           -                    -left_down_front     -                    -L                   -                    
        `
        """
        return list(
            map(
                lambda row: "-".join([
                    f"{(row[x]['value'] if row[x] else ''):<20}"
                    for x in fields
                ]), rows))

    def compose_as_words(self, rows, fields):
        """
        Example:
        `
        lb--ldf--L-
        `
        """
        def compose_field(data):
            return ''.join([k[0] for k in str(data['value']).split('_')
                            ]) if data else ''

        return list(
            map(lambda row: "-".join([compose_field(row[f]) for f in fields]),
                rows))

    def compose_as_words_norm(self, rows, fields):
        """
        Example:
        `
        l_b-___-ldf-___-L-
        `
        """
        def compose_field(field, data):
            values = str(data['value']) if data else ''

            if field.startswith("orientation") or field.startswith("movement"):
                values = values.split('_')
                return ''.join([("l" if "left" in values else
                                 "r" if "right" in values else "_"),
                                ("u" if "up" in values else
                                 "d" if "down" in values else "_"),
                                ("f" if "front" in values else
                                 "b" if "back" in values else "_")])
            else:
                return values

        return list(
            map(
                lambda row: "-".join(
                    [compose_field(f, row[f]) for f in fields]), rows))

    def compose_sep_feat(self, rows, fields):
        """
        Example:
        `
        ['lb', '', 'ldf', '', 'L', '']
        `
        """
        def compose_field(data):
            return ''.join([k[0] for k in str(data['value']).split('_')
                            ]) if data else ''

        return list(
            map(lambda row: str([compose_field(row[f]) for f in fields]),
                rows))
