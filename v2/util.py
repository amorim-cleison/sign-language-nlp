from commons.log import log


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
