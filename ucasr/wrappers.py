from functools import partial
import os
from pathlib import Path
import random

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset, Sampler

from ucasr.models.vgg_model import VGGModel, VGGPathModel
from ucasr.maestro_snippet_dataset import load_maestro_dataset
from ucasr.snippet_dataset import load_msmd_dataset, load_umc_dataset
from ucasr.score_snippet_dataset import load_score_dataset
from ucasr.utils.utils import load_yaml


class CustomSampler(Sampler):
    """ Custom sampler that samples a new subset every epoch.
        Reference: https://discuss.pytorch.org/t/new-subset-every-epoch/85018 """

    def __init__(self, data_source, num_samples=None):
        super().__init__(data_source)
        self.data_source = data_source
        self._num_samples = num_samples

        if not isinstance(self.num_samples, int) or self.num_samples <= 0:
            raise ValueError(
                "num_samples should be a positive integer "
                "value, but got num_samples={}".format(self.num_samples)
            )

    @property
    def num_samples(self):
        # dataset size might change at runtime
        if self._num_samples is None:
            return len(self.data_source)
        return self._num_samples

    def __iter__(self):
        n = len(self.data_source)
        return iter(torch.randperm(n, dtype=torch.int64)[: self.num_samples].tolist())

    def __len__(self):
        return self.num_samples


class MixedDataset(torch.utils.data.Dataset):
    def __init__(self, *datasets):
        self.datasets = datasets

    def __getitem__(self, i):
        return tuple(d[i] for d in self.datasets)

    def __len__(self):
        return min(len(d) for d in self.datasets)


def create_maestro_loaders(args):
    print('Loading and preparing data...\n')
    maestro_split_file = os.path.join(args.split_root, 'maestro_split.csv')
    df = pd.read_csv(maestro_split_file, sep=",")

    tr_pieces = [os.path.join(args.maestro_root, (piece['audio_filename'])) for _, piece in df.iterrows() if
                 piece['split'] == 'train']

    va_pieces = [os.path.join(args.maestro_root, (piece['audio_filename'])) for _, piece in df.iterrows() if
                 piece['split'] == 'validation']

    tr_dataset = load_maestro_dataset(tr_pieces, args)
    print(f'Training dataset: {len(tr_dataset)} snippets')

    tr_loader = DataLoader(dataset=tr_dataset,
                           batch_size=args.batch_size,
                           num_workers=args.n_workers,
                           drop_last=True,
                           sampler=CustomSampler(data_source=tr_dataset, num_samples=args.n_train))

    va_dataset = load_maestro_dataset(va_pieces, args)
    print(f'Validation dataset: {len(va_dataset)} snippets\n')

    n_valid = np.min([args.n_valid, len(va_dataset)])
    va_samples = np.linspace(0, len(va_dataset) - 1, n_valid).astype(int)
    va_loader = DataLoader(dataset=Subset(va_dataset, indices=va_samples), batch_size=args.batch_size,
                           num_workers=args.n_workers, drop_last=False, shuffle=False)

    # additional dataloader for evaluating on a subset of the train set
    tr_eval_samples = np.linspace(0, len(tr_dataset) - 1, n_valid).astype(int)
    tr_eval_loader = DataLoader(dataset=Subset(tr_dataset, indices=tr_eval_samples), batch_size=args.batch_size,
                                num_workers=args.n_workers, drop_last=False, shuffle=False)

    return tr_loader, va_loader, tr_eval_loader


def create_score_loaders(args):
    print('Loading and preparing data...\n')
    score_split_file = os.path.join(args.split_root, 'score_split.yaml')
    split = load_yaml(score_split_file)

    tr_pieces = sorted([str(p) for p in Path(args.scores_root).rglob('*unrolled_score.png') if
                        any(s in str(p) for s in split['train'])])

    va_pieces = sorted([str(p) for p in Path(args.scores_root).rglob('*unrolled_score.png') if
                        any(s in str(p) for s in split['valid'])])

    tr_dataset = load_score_dataset(tr_pieces, args)
    print(f'Training dataset: {len(tr_dataset)} snippets')

    tr_loader = DataLoader(dataset=tr_dataset,
                           batch_size=args.batch_size,
                           num_workers=args.n_workers,
                           drop_last=True,
                           sampler=CustomSampler(data_source=tr_dataset, num_samples=args.n_train))

    va_dataset = load_score_dataset(va_pieces, args)
    print(f'Validation dataset: {len(va_dataset)} snippets\n')

    n_valid = np.min([args.n_valid, len(va_dataset)])
    va_samples = np.linspace(0, len(va_dataset) - 1, n_valid).astype(int)
    va_loader = DataLoader(dataset=Subset(va_dataset, indices=va_samples), batch_size=args.batch_size,
                           num_workers=args.n_workers, drop_last=False, shuffle=False)

    # additional dataloader for evaluating on a subset of the train set
    tr_eval_samples = np.linspace(0, len(tr_dataset) - 1, n_valid).astype(int)
    tr_eval_loader = DataLoader(dataset=Subset(tr_dataset, indices=tr_eval_samples), batch_size=args.batch_size,
                                num_workers=args.n_workers, drop_last=False, shuffle=False)

    return tr_loader, va_loader, tr_eval_loader


def create_mixed_loaders(args):
    print('Loading and preparing data...\n')

    # loading sheet music dataset
    score_split_file = os.path.join(args.split_root, 'score_split_small.yaml')
    score_split = load_yaml(score_split_file)

    tr_score_pieces = sorted([str(p) for p in Path(args.scores_root).rglob('*unrolled_score.png') if
                              any(s in str(p) for s in score_split['train'])])

    va_score_pieces = sorted([str(p) for p in Path(args.scores_root).rglob('*unrolled_score.png') if
                              any(s in str(p) for s in score_split['valid'])])

    tr_score_dataset = load_score_dataset(tr_score_pieces, args)
    print(f'Score training dataset: {len(tr_score_dataset)} snippets')

    va_score_dataset = load_score_dataset(va_score_pieces, args)
    print(f'Score validation dataset: {len(va_score_dataset)} snippets\n')

    # loading maestro dataset splits
    maestro_split_file = os.path.join(args.split_root, 'maestro_split_10.csv')
    df = pd.read_csv(maestro_split_file, sep=",")

    tr_maestro_pieces = [os.path.join(args.maestro_root, (piece['audio_filename'])) for _, piece in df.iterrows() if
                         piece['split'] == 'train']

    va_maestro_pieces = [os.path.join(args.maestro_root, (piece['audio_filename'])) for _, piece in df.iterrows() if
                         piece['split'] == 'validation']

    tr_maestro_dataset = load_maestro_dataset(tr_maestro_pieces, args)
    print(f'Audio training dataset: {len(tr_maestro_dataset)} snippets')

    va_maestro_dataset = load_maestro_dataset(va_maestro_pieces, args)
    print(f'Audio validation dataset: {len(va_maestro_dataset)} snippets\n')

    # mixing the training datasets
    tr_mixed_dataset = MixedDataset(tr_score_dataset, tr_maestro_dataset)
    tr_mixed_loader = DataLoader(dataset=tr_mixed_dataset,
                                 batch_size=args.batch_size,
                                 num_workers=args.n_workers,
                                 drop_last=True,
                                 sampler=CustomSampler(data_source=tr_mixed_dataset, num_samples=args.n_train))

    # mixing the validation datasets
    va_mixed_dataset = MixedDataset(va_score_dataset, va_maestro_dataset)
    n_valid = np.min([args.n_valid, len(va_mixed_dataset)])
    va_samples = np.linspace(0, len(va_mixed_dataset) - 1, n_valid).astype(int)
    va_mixed_loader = DataLoader(dataset=Subset(va_mixed_dataset, indices=va_samples), batch_size=args.batch_size,
                                 num_workers=args.n_workers, drop_last=False, shuffle=False)

    # additional dataloader for evaluating on a subset of the train set
    tr_eval_mixed_samples = np.linspace(0, len(tr_mixed_dataset) - 1, n_valid).astype(int)
    tr_eval_mixed_loader = DataLoader(dataset=Subset(tr_mixed_dataset, indices=tr_eval_mixed_samples),
                                      batch_size=args.batch_size, num_workers=args.n_workers, drop_last=False,
                                      shuffle=False)

    return tr_mixed_loader, va_mixed_loader, tr_eval_mixed_loader


def create_train_loaders(args, only_refine=False):
    print('Loading and preparing data...\n')

    ds_ratio = f'_{args.ds_ratio}' if args.ds_ratio else ''

    splits = load_yaml(os.path.join(args.split_root, f'msmd_split{ds_ratio}.yaml'))

    tr_dataset = load_msmd_dataset(args.msmd_root, splits['train'], args, aug='full_aug')

    if only_refine:
        tr_samples = random.sample(range(len(tr_dataset)), k=args.n_refine)
        tr_loader = DataLoader(dataset=Subset(tr_dataset, indices=tr_samples),
                               batch_size=args.batch_size,
                               shuffle=False,
                               drop_last=False,
                               num_workers=args.n_workers)
        return tr_loader
    print(f'Training dataset: {len(tr_dataset)} snippet pairs')

    va_dataset = load_msmd_dataset(args.msmd_root, splits['valid'], args, aug='no_aug')
    print(f'Validation dataset: {len(va_dataset)} snippet pairs\n')

    n_valid = np.min([args.n_valid, len(va_dataset)])
    va_samples = np.linspace(0, len(va_dataset) - 1, n_valid).astype(int)

    tr_loader = DataLoader(dataset=tr_dataset,
                           batch_size=args.batch_size,
                           num_workers=args.n_workers,
                           drop_last=True,
                           sampler=CustomSampler(data_source=tr_dataset, num_samples=args.n_train))

    va_loader = DataLoader(dataset=Subset(va_dataset, indices=va_samples), batch_size=args.batch_size,
                           num_workers=args.n_workers, drop_last=False, shuffle=False)

    # additional dataloader for evaluating on a subset of the train set
    tr_eval_samples = np.linspace(0, len(tr_dataset) - 1, n_valid).astype(int)
    tr_eval_loader = DataLoader(dataset=Subset(tr_dataset, indices=tr_eval_samples), batch_size=args.batch_size,
                                num_workers=args.n_workers, drop_last=False, shuffle=False)

    return tr_loader, va_loader, tr_eval_loader


def create_test_loader(args):
    datasets = {'MSMD': 'msmd_split.yaml', 'RealScores_Synth': 'db_scanned_synth.yaml',
                'RealScores_Rec': 'db_scanned_recording.yaml'}

    splits = load_yaml(os.path.join(args.split_root, datasets[args.dataset]))

    if args.dataset == 'MSMD':
        te_dataset = load_msmd_dataset(args.msmd_root, splits['test'], args, aug='test_aug')
    else:
        te_dataset = load_umc_dataset(args.umc_root, splits['test'], args)

    print(f'Test dataset: {len(te_dataset)} snippet pairs\n')
    n_test = min([args.n_test, len(te_dataset)])
    te_samples = np.linspace(0, len(te_dataset) - 1, n_test).astype(int)
    te_loader = DataLoader(dataset=Subset(te_dataset, indices=te_samples),
                           batch_size=args.batch_size,
                           drop_last=False,
                           shuffle=False,
                           num_workers=args.n_workers)

    return te_loader


def get_model(args, mode='train'):
    if mode == 'train':
        from ucasr.utils.losses import triplet_loss
        loss_function = partial(triplet_loss, margin=args.loss_margin)
        return VGGModel(args), loss_function

    if mode == 'audio_pretrain':
        from ucasr.utils.losses import nt_xent_loss
        loss_function = partial(nt_xent_loss, temperature=args.loss_margin)
        return VGGPathModel(args, is_audio=True, pretrain=True), loss_function

    if mode == 'score_pretrain':
        from ucasr.utils.losses import nt_xent_loss
        loss_function = partial(nt_xent_loss, temperature=args.loss_margin)
        return VGGPathModel(args, is_audio=False, pretrain=True), loss_function

    if mode == 'mixed_pretrain':
        from ucasr.utils.losses import nt_xent_loss
        loss_function = partial(nt_xent_loss, temperature=args.loss_margin)
        return VGGModel(args, use_cca=False, pretrain=True), loss_function
