import json
import tempfile

from os.path import normpath
import numpy as np
import pandas as pd
from commons.log import auto_log_progress, log
from commons.util import (exists, filename, filter_files, read_json,
                          save_items, get_hash)
from dataset.constant import PAD_WORD, UNK_WORD
from torchtext.data import Field, TabularDataset, interleave_keys


class DatasetBuilder():
    def __init__(self):
        pass

    def build(self,
              dataset_dir,
              fields,
              samples_min_freq,
              batch_first,
              composition_strategy="as_words"):
        log("Loading dataset...")

        # Temp name:
        _name = get_hash({
            "dir": dataset_dir,
            "fields": fields,
            "min_freq": samples_min_freq,
            "strategy": composition_strategy
        })
        path = normpath(f"{tempfile.gettempdir()}/{_name}.tmp")

        # Write transient working file:
        if exists(path):
            log(f"Reusing working data file found at '{path}'...")
        else:
            log("Creating working data file...")
            self.write_working_file(path=path,
                                    dataset_dir=dataset_dir,
                                    min_freq=samples_min_freq)

        # Dataset:
        dataset, src_vocab, tgt_vocab, file_vocab = \
            self.create_dataset(
                path=path,
                fields=fields,
                composition_strategy=composition_strategy,
                batch_first=batch_first)

        return {
            "dataset": dataset,
            "src_vocab": src_vocab,
            "tgt_vocab": tgt_vocab
        }

    def write_working_file(self, path, dataset_dir, min_freq):
        def prefix(f):
            return f.stem.split('-')[0]

        def prepare_row(p):
            data = read_json(p)
            data["file"] = filename(p)
            return json.dumps(data).replace('null', '""')

        assert exists(dataset_dir), "Invalid dataset directory"
        files = filter_files(dataset_dir, ext="json", path_as_str=False)

        # Group and filter data:
        df = pd.DataFrame({"file": files})
        df["prefix"] = df["file"].apply(prefix)
        df = df\
            .groupby(["prefix"])\
            .filter(lambda x: x["file"].count() >= min_freq)

        # Separate data chunks:
        n_samples = len(df)
        n_batches = n_samples / 10
        batches = np.array_split(df, n_batches)
        msg = "Processing data... "

        for batch in auto_log_progress(batches, message=msg):
            rows = batch["file"].apply(prepare_row)
            save_items(rows, path, True)

    def create_dataset(self, path, fields, composition_strategy, batch_first):
        def preprocess_src(rows):
            return self.preprocess_src(rows, fields, composition_strategy)

        # Fields:
        SRC = Field(pad_token=PAD_WORD,
                    unk_token=UNK_WORD,
                    preprocessing=preprocess_src,
                    include_lengths=True,
                    batch_first=batch_first)
        TGT = Field(is_target=True,
                    pad_first=True,
                    pad_token=PAD_WORD,
                    batch_first=batch_first)
        FILE = Field(batch_first=batch_first)

        # Dataset:
        dataset = TabularDataset(path=path,
                                 format="json",
                                 fields={
                                     'frames.phonology': ('src', SRC),
                                     'label': ('tgt', TGT),
                                     'file': ('file', FILE)
                                 })
        dataset.sort_key = lambda x: interleave_keys(len(x.src), len(x.tgt))

        # Vocabs:
        SRC.build_vocab(dataset.src)
        TGT.build_vocab(dataset.tgt)
        FILE.build_vocab(dataset.file)
        return dataset, SRC.vocab, TGT.vocab, FILE.vocab

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
