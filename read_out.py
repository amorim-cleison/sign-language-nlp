import re
from commons.util import read_items, save_csv

PATTERN = r'\[CV (?P<cv>\d)\/5\] END lr=(?P<lr>[\d.]+), module__dropout=(?P<dropout>[\d.]+), module__embedding_size=(?P<emb_size>[\d.]+), module__hidden_size=(?P<hidden_size>\d+), module__num_heads=(?P<heads>\d+), module__num_layers=(?P<layers>\d+);, score=(?P<score>[\d.-]+) total time=(.+)min'


def run(src_path, tgt_path):
    items = read_items(src_path)
    items = filter(lambda x: x.startswith('[CV '), items)

    p = re.compile(PATTERN)

    data = [p.search(x).groupdict() for x in items]
    save_csv(data, tgt_path)


if __name__ == "__main__":
    run('C:\\Users\\ngscl\\Documents\\UFPE\\work-checkpoints\\2022-05-27 - grid transformer\\grid-transformer.out',
        'C:\\Users\\ngscl\\Documents\\UFPE\\work-checkpoints\\2022-05-27 - grid transformer\\grid-transformer.csv')
