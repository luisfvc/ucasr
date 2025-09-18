"""
This script visualizes and saves plots of some snippets from the validation split of MAESTRO after processed
by a pipeline of augmentation transforms
"""
import argparse
import os
import random

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset

from ucasr.models.custom_modules import LogSpectrogramModule
from ucasr.maestro_snippet_dataset import load_maestro_dataset
from ucasr.utils.utils import load_yaml, set_remote_paths, make_dir

matplotlib.use('TkAgg')


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--save_plots', help='dump plot images', action='store_true', default=False)
    parser.add_argument('--audio_context', help='audio snippet size', type=str, default='mc')
    parser.add_argument('--n_snippets', help='number of snippets to visualize', type=int, default=8)

    configs = load_yaml('../config/maestro_config.yaml')

    for k, v in configs.items():
        parser.add_argument(f"--{k}", default=v, type=type(v))

    args = set_remote_paths(parser.parse_args(), audio_pretrain=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ensuring reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = True

    # creating dataloader
    maestro_split_file = os.path.join(args.split_root, 'maestro_split_10.csv')
    df = pd.read_csv(maestro_split_file, sep=",")

    pieces = [os.path.join(args.maestro_root, (piece['audio_filename'])) for _, piece in df.iterrows() if
              piece['split'] == 'validation']
    dataset = load_maestro_dataset(pieces, args)
    samples = np.linspace(0, len(dataset) - 1, args.n_snippets).astype(int)
    loader = DataLoader(dataset=Subset(dataset, indices=samples), batch_size=args.n_snippets,
                        num_workers=args.n_workers, drop_last=False, shuffle=False)

    # get model
    logspec = LogSpectrogramModule(args)
    logspec.to(device)

    view_1, view_2 = next(iter(loader))
    view_1, view_2 = view_1.to(device), view_2.to(device)

    specs_1 = logspec(view_1)
    specs_2 = logspec(view_2)

    specs_1 = specs_1.squeeze().detach().cpu().numpy()
    specs_2 = specs_2.squeeze().detach().cpu().numpy()

    for i, (x, y) in enumerate(zip(specs_1, specs_2)):

        plt.figure(figsize=(14, 4))
        plt.clf()
        plt.subplot(1, 2, 1)
        plt.imshow(x, cmap='viridis', origin='lower', aspect='auto')
        plt.subplot(1, 2, 2)
        plt.imshow(y, cmap='viridis', origin='lower', aspect='auto')
        plt.savefig(f'{i}.png') if args.save_plots else plt.show()
