"""
This script visualizes and saves plots of some snippets from the validation split of MAESTRO after processed
by a pipeline of augmentation transforms
"""
import argparse
import os
from pathlib import Path
import random

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset

from ucasr.models.custom_modules import ScoreAugModule
from ucasr.score_snippet_dataset import load_score_dataset
from ucasr.utils.utils import load_yaml, set_remote_paths, make_dir

matplotlib.use('TkAgg')

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--save_plots', help='dump plot images', action='store_true', default=True)
    parser.add_argument('--n_snippets', help='number of snippets to visualize', type=int, default=32)

    configs = load_yaml('../config/score_config.yaml')

    for k, v in configs.items():
        parser.add_argument(f"--{k}", default=v, type=type(v))

    args = set_remote_paths(parser.parse_args(), score_pretrain=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ensuring reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = True

    # creating dataloader
    score_split_file = os.path.join(args.split_root, 'score_split_small.yaml')
    split = load_yaml(score_split_file)

    pieces = sorted([str(p) for p in Path(args.scores_root).rglob('*unrolled_score.png') if
                     any(s in str(p) for s in split['train'])])

    dataset = load_score_dataset(pieces, args)
    samples = np.linspace(0, len(dataset) - 1, args.n_snippets).astype(int)
    loader = DataLoader(dataset=Subset(dataset, indices=samples), batch_size=args.n_snippets,
                        num_workers=1, drop_last=False, shuffle=False)

    # get model
    score_aug_module = ScoreAugModule(args)
    score_aug_module.to(device)

    view_1, view_2 = next(iter(loader))
    view_1, view_2 = view_1.to(device), view_2.to(device)

    score_1 = score_aug_module(view_1)
    score_2 = score_aug_module(view_2)

    score_1 = score_1.squeeze().detach().cpu().numpy()
    score_2 = score_2.squeeze().detach().cpu().numpy()

    for i, (x, y) in enumerate(zip(score_1, score_2)):
        plt.figure(figsize=(14, 4))
        plt.clf()
        plt.subplot(1, 2, 1)
        plt.imshow(x, cmap='gray')
        plt.subplot(1, 2, 2)
        plt.imshow(y, cmap='gray')
        plt.savefig(f'{i}.png') if args.save_plots else plt.show()
