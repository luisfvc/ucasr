from pathlib import Path
import shutil
import random
import os
import yaml


def copy_only_unrolled_scores(origin_root, target_root):

    for p in Path(origin_root).rglob('*unrolled_score.png'):
        os.makedirs(Path(target_root, *p.parts[-5:-1]))
        shutil.copy(p, Path(target_root, *p.parts[-5:]))


if __name__ == '__main__':

    random.seed(345)

    db_path = '/home/luis/data/score_db_unrolled'
    pieces = sorted([p for p in Path(db_path).glob('*/*')])

    factor = 0.95
    n_pieces = len(pieces)

    n_train = int(factor * n_pieces)
    n_val = n_pieces - n_train

    train_split = sorted(random.sample(pieces, n_train))
    val_split = sorted(set(pieces).difference(train_split))

    train_split = [str(Path(*p.parts[-2:])) for p in train_split]
    val_split = [str(Path(*p.parts[-2:])) for p in val_split]

    split = {'train': train_split, 'valid': val_split}

    with open('../splits/score_split.yaml', 'w') as f:
        yaml.dump(split, f)






