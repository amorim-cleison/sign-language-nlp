from commons.log import log, log_progress
from commons.util import exists, normpath, filename
from torchtext.data import Field, TabularDataset

from .field_composer import FieldComposer
from .custom_iterator import CustomIterator
from .tokens import BOS_WORD, PAD_WORD, UNK_WORD, EOS_WORD


def build_dataset(path, train_split_ratio, val_split_ratio=None, **kwargs):
    dataset = __provide_dataset(path=path, **kwargs)

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


def __provide_dataset(path, dataset_dir, samples_min_freq, max_len_sentence,
                      fields, composition_strategy, vocab_min_freq, **kwargs):
    # Create dataset if needed:
    path = normpath(path)

    if not exists(path):
        log(f"Creating dataset to '{path}'...")
        __make_dataset(dataset_dir, path, samples_min_freq)
        log("Finished")

    # Create fields:
    composer = FieldComposer(fields, composition_strategy)

    FIX_LENGTH = None  # FIXME: verify this

    SRC = Field(pad_token=PAD_WORD,
                unk_token=UNK_WORD,
                preprocessing=composer.run,
                fix_length=10)  # FIXME: verify this
    TGT = Field(
        is_target=True,
        pad_first=True,
        init_token=BOS_WORD,
        #  eos_token=EOS_WORD,
        pad_token=PAD_WORD,
        fix_length=FIX_LENGTH)  # FIXME: verify this
    FILE = Field()

    # Create dataset:
    dataset = TabularDataset(
        path=path,
        format="json",
        fields={
            'frames.phonology': ('src', SRC),
            'label': ('tgt', TGT),
            'file': ('file', FILE)
        },
        filter_pred=lambda x: len(vars(x)['src']) <= max_len_sentence)

    SRC.build_vocab(dataset.src, min_freq=vocab_min_freq)
    TGT.build_vocab(dataset.tgt, min_freq=vocab_min_freq)
    FILE.build_vocab(dataset.file)

    return dataset


def __make_dataset(dataset_dir, tgt_path, min_count):
    import json

    from commons.util import filter_files, read_json, save_items

    assert exists(dataset_dir), "Invalid attributes directory"
    files = filter_files(dataset_dir, ext="json", path_as_str=False)
    processed = list()

    def prefix(file):
        return file.stem.split('-')[0]

    def prepare_sample(path):
        data = read_json(path)
        data["file"] = filename(path)
        return json.dumps(data).replace('null', '""')

    for file in files:
        if file not in processed:
            file_prefix = prefix(file)
            log_progress(len(processed), len(files), file_prefix)
            similars = [x for x in files if prefix(x) == file_prefix]
            processed.extend(similars)

            if len(similars) >= min_count:
                samples = [prepare_sample(path) for path in similars]
                save_items(samples, tgt_path, True)


def build_iterator(dataset, batch_size, device, train):
    return CustomIterator(dataset,
                          batch_size=batch_size,
                          device=device,
                          repeat=False,
                          sort_key=lambda x: (len(x.src), len(x.tgt)),
                          train=train)
