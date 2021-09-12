from commons.log import log


def save_step(phase, dir, **data):
    from commons.util import create_if_missing, normpath, save_csv
    create_if_missing(dir)
    path = normpath(f"{dir}/epoch_log.csv")

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
    from datetime import datetime

    import torch
    from commons.util import create_if_missing, normpath, save_csv

    def greedy_decode(indexes, vocab):
        return vocab.itos[indexes]
        # d = [vocab.itos[i] for i in indexes]
        # return d[0] if len(d) == 1 else d

    create_if_missing(dir)
    path = normpath(f"{dir}/evaluation_log.csv")
    now = datetime.now().strftime("%Y/%m/%d %H:%M:%S")

    outputs = torch.argmax(outputs, dim=-1)
    outputs = outputs.transpose(0, 1) if len(outputs.size()) > 2 else outputs
    targets = targets.transpose(0, 1) if len(targets.size()) > 1 else targets
    files = files.transpose(0, 1) if len(files.size()) > 1 else files

    data = [{
        "datetime": now,
        "epoch": epoch if epoch else "",
        "file": greedy_decode(f, file_vocab),
        "output": greedy_decode(o, tgt_vocab),
        "target": greedy_decode(t, tgt_vocab),
        "correct": (o == t),
    } for (o, t, f) in zip(outputs, targets, files)]

    save_csv(data, path, append=True)


def log_data(data):
    log("-" * 100)
    log("DATA")
    log("-" * 100)

    for name, value in data.items():
        log(f"| {name.upper():15} | {len(value):10d} items |")
    log("-" * 100)
    log("")


def log_model(model):
    for name, value in model.items():
        log("-" * 100)
        log(f"{name.upper()}")
        log("-" * 100)
        log(value)
        log("")
    log("-" * 100)
    log("")
