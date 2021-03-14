from commons.log import log
from v2.dataset import PAD_WORD


def save_step(phase, dir, **data):
    from commons.util import save_csv, create_if_missing, normpath
    create_if_missing(dir)
    path = normpath(f"{dir}/epochs_log.csv")

    from datetime import datetime
    new_data = {
        "datetime": datetime.now().strftime("%Y/%m/%d %H:%M:%S"),
        "phase": phase,
        "epoch": data.get("epoch"),
        "batch": data.get("batch"),
        "acc": data.get("acc"),
        "loss": data.get("loss"),
        "ppl": data.get("ppl"),
        "lr": data.get("lr")
    }
    # new_data.update(data)
    save_csv([new_data], path, append=True)


def log_step(phase, sep: str = None, **data):
    def separator():
        if sep is not None:
            log(sep * 100)

    msg = " | ".join([f"{k} {v}" for k, v in data.items()])
    msg = f"| {phase.upper()} | {msg} |"
    separator()
    log(msg)
    separator()


def save_eval_outputs(outputs,
                      targets,
                      files,
                      tgt_vocab,
                      file_vocab,
                      dir,
                      epoch=None):
    import torch
    from datetime import datetime
    from commons.util import save_csv, create_if_missing, normpath

    def greedy_decode(indexes, vocab):
        d = [vocab.itos[i] for i in indexes]
        return d[0] if len(d) == 1 else d

    create_if_missing(dir)
    path = normpath(f"{dir}/outputs_log.csv")
    now = datetime.now().strftime("%Y/%m/%d %H:%M:%S")

    outputs = torch.argmax(outputs, dim=-1).transpose(0, 1)
    targets = targets.transpose(0, 1)
    files = files.transpose(0, 1)

    data = [{
        "datetime": now,
        "epoch": epoch if epoch else "",
        "file": greedy_decode(f, file_vocab),
        "output": greedy_decode(o, tgt_vocab),
        "target": greedy_decode(t, tgt_vocab),
        "correct": all(o == t)
    } for (o, t, f) in zip(outputs, targets, files)]

    save_csv(data, path, append=True)


def generate_mask(data, model):
    """
    Mask ensures that position i is allowed to attend the unmasked
    positions. If a ByteTensor is provided, the non-zero positions are
    not allowed to attend while the zero positions will be unchanged.
    If a BoolTensor is provided, positions with ``True`` are not
    allowed to attend while ``False`` values will be unchanged.
    If a FloatTensor is provided, it will be added to the attention
    weight.
    """
    def generate_square_subsequent_mask(sz: int):
        r"""
        Generate a square mask for the sequence. The masked positions are
        filled with float('-inf').
        Unmasked positions are filled with float(0.0).

        Extracted from `torch.nn.Transformer.generate_square_subsequent_mask`.
        """
        import torch
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(
            mask == 1, float(0.0))
        return mask

    mask = generate_square_subsequent_mask(data.size(0))
    mask = (mask != float(0.0)).bool()
    return mask


def generate_padding_mask(data, vocab):
    """
    Padding mask provides specified elements in the key to be ignored
    by the attention. If a ByteTensor is provided, the non-zero
    positions will be ignored while the zero positions will be
    unchanged. If a BoolTensor is provided, the positions with the
    value of ``True`` will be ignored while the position with the
    value of ``False`` will be unchanged.
    """
    pad_idx = vocab.stoi[PAD_WORD]
    mask = (data == pad_idx).transpose(0, 1).bool()
    return mask
