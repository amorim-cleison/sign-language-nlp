from os.path import normpath


def filter_files(dir_path, name="*", ext="*"):
    import glob
    assert (dir_path is not None), "`dir_path` is required"
    _ext = ext if ext.startswith(".") else f".{ext}"
    _path = normpath(f"{dir_path}/{name}{_ext}")
    return glob.glob(_path)


