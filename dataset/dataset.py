from commons.log import log, log_progress


def create_dataset(attributes_dir, tgt_path, min_count):
    from commons.util import save_items, filter_files, read_json, exists
    import json

    log(f"Creating dataset to '{tgt_path}'...")
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
    log("Finished")
