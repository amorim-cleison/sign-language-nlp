from commons.log import log, log_progress
from commons.util import exists, normpath
from torchtext import data

from .field_composer import FieldComposer
from .iterator import CustomIterator

BOS_WORD = '<bos>'
EOS_WORD = '<eos>'
UNK_WORD = '<unk>'
PAD_WORD = '<pad>'


def build_dataset(path, train_split_ratio, **kwargs):
    dataset = __provide_dataset(path=path, **kwargs)

    # ratios (parameter): [ train, test, val]
    # output: (train, [val,] test)
    splits = dataset.split(
        split_ratio=[train_split_ratio, 1 - train_split_ratio])

    if len(splits) == 3:
        train, val, test = splits
    else:
        train, test = splits
        val = None
    return train, val, test


def __provide_dataset(path, attributes_dir, samples_min_freq, max_len_sentence,
                      fields, composition_strategy, vocab_min_freq, **kwargs):
    # Create dataset if needed:
    path = normpath(path)

    if not exists(path):
        log(f"Creating dataset to '{path}'...")
        __make_dataset(attributes_dir, path, samples_min_freq)
        log("Finished")

    # Create fields:
    composer = FieldComposer(fields, composition_strategy)

    SRC = data.Field(sequential=True,
                     pad_token=PAD_WORD,
                     preprocessing=composer.run)
    TGT = data.Field(sequential=True,
                     is_target=True,
                     pad_first=True,
                     init_token=BOS_WORD,
                     eos_token=EOS_WORD,
                     unk_token=UNK_WORD,
                     pad_token=PAD_WORD)

    # Create dataset:
    dataset = data.TabularDataset(
        path=path,
        format="json",
        fields={
            'phonos': ('src', SRC),
            'label': ('trg', TGT)
        },
        filter_pred=lambda x: len(vars(x)['src']) <= max_len_sentence)

    SRC.build_vocab(dataset.src, min_freq=vocab_min_freq)
    TGT.build_vocab(dataset.trg, min_freq=vocab_min_freq)

    return dataset


def __make_dataset(attributes_dir, tgt_path, min_count):
    import json

    from commons.util import filter_files, read_json, save_items

    assert exists(attributes_dir), "Invalid attributes directory"
    files = filter_files(attributes_dir, ext="json", path_as_str=False)
    processed = list()

    def prefix(file):
        return file.stem.split('-')[0]

    for file in files:
        if file not in processed:
            file_prefix = prefix(file)
            log_progress(len(processed), len(files), file_prefix)

            similar = [x for x in files if prefix(x) == file_prefix]
            processed.extend(similar)

            if len(similar) >= min_count:
                samples = [
                    json.dumps(read_json(x)).replace('null', '""')
                    for x in similar
                ]
                save_items(samples, tgt_path, True)


def build_iterator(dataset, batch_size, device, train):
    return CustomIterator(dataset,
                          batch_size=batch_size,
                          device=device,
                          repeat=False,
                          sort_key=lambda x: (len(x.src), len(x.trg)),
                          train=train)
