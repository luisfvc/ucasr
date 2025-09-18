import argparse
import math
from multiprocessing import Pool
import os
from pathlib import Path
import random

import PIL.Image
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import cv2
from elasticdeform import deform_random_grid
from skimage.transform import rotate
import torch
import torchvision.transforms.functional as tf
from torch.utils.data import Dataset as TorchDataset
from tqdm import tqdm

from ucasr.utils.utils import load_yaml, set_remote_paths, generate_perlin_noise_2d

# matplotlib.use('TkAgg')


class ScoreSnippetDataset(TorchDataset):

    def __init__(self, snippet_samples, ur_scores, args):

        # resize snippets
        self.r = 2  # 1: augs before resizing, 2: augs after resizing

        self.sheet_context = args.sheet_context // self.r
        self.staff_height = args.staff_height // self.r

        self.snippet_samples = snippet_samples
        self.ur_scores = ur_scores
        self.pxl_avg = [0, 0.4]
        self.pxl_var = [0.0, 0.3]
        self.max_angle = 4
        self.gb_ks = [3, 5, 7, 9, 11]
        self.gb_sigma = [0.4, 1.0]

        self.args = args
        self.sc = args.sheet_scaling
        self.x_shift = int((1 - args.xshift_overlap) * args.sheet_context) // self.r

        self.p_xshift = args.p_xshift
        self.p_yshift = args.p_yshift
        self.p_scale = args.p_scale
        self.p_rotate = args.p_rotate
        self.p_2dnoise = args.p_2dnoise
        self.p_blur = args.p_blur
        self.p_perlin = args.p_perlin
        self.p_et_small = args.p_et_small
        self.p_et_large = args.p_et_large

    def get_sub_sheet(self, index):

        # fetch sample from the data
        sample = self.snippet_samples[index]

        sheet = self.ur_scores[sample[1]]

        target_coord = sample[0]

        sheet_pad_beg = max(0, 2 * self.sheet_context * self.r - target_coord)
        sheet_pad_end = max(0, target_coord + 2 * self.sheet_context * self.r - sheet.shape[1])
        if sheet_pad_beg or sheet_pad_end:
            sheet_pad_beg_array = np.ones((sheet.shape[0], sheet_pad_beg), dtype=np.float32) * 1.0
            sheet_pad_end_array = np.ones((sheet.shape[0], sheet_pad_end), dtype=np.float32) * 1.0
            sheet = np.hstack((sheet_pad_beg_array, sheet, sheet_pad_end_array))
            target_coord += sheet_pad_beg

        # get sub-image (with coordinate fixing)
        # this is done since we do not want to do the augmentation
        # on the whole sheet image
        c0 = max(0, target_coord - 2 * self.sheet_context * self.r)
        c1 = min(c0 + 4 * self.sheet_context * self.r, sheet.shape[1])
        c0 = max(0, c1 - 4 * self.sheet_context * self.r)
        sheet = sheet[:, c0:c1]

        return cv2.resize(sheet, (sheet.shape[1] // self.r, sheet.shape[0] // self.r), interpolation=cv2.INTER_LINEAR)

    def add_gaussian_noise(self, x):

        rd_avg = (self.pxl_avg[1] - self.pxl_avg[0]) * np.random.random() + self.pxl_avg[0]
        rd_var = (self.pxl_var[1] - self.pxl_var[0]) * np.random.random() + self.pxl_var[0]

        noisy_sheet = x + torch.randn_like(x) * rd_var + rd_avg

        # return random_noise(sub_sheet, mode='gaussian', mean=rd_avg, var=rd_var, clip=True)
        return torch.clip(noisy_sheet, 0.0, 1.0)

    def apply_gaussian_blur(self, x):

        ks = int(np.random.choice(self.gb_ks))
        sigma = (self.gb_sigma[1] - self.gb_sigma[0]) * np.random.random() + self.gb_sigma[0]

        return tf.gaussian_blur(x, kernel_size=ks, sigma=sigma)

    def rotate_img(self, x):

        angle = 2 * self.max_angle * np.random.random() - self.max_angle
        return rotate(x, angle=angle, cval=1.0)

    def apply_elastic_transform_small(self, x):

        alpha = int((200 - 50) * np.random.random() + 50)
        sigma = 0.2
        x = deform_random_grid(x, points=alpha, sigma=sigma, cval=1.0, prefilter=False)
        return x

    def apply_elastic_transform_big(self, x):

        alpha = int((10 - 2) * np.random.random() + 2)
        sigma = 0.5
        x = deform_random_grid(x, points=alpha, sigma=sigma, cval=1.0, prefilter=False)
        return x

    def add_perlin_noise(self, x):

        factor = np.random.random() * 0.6
        r_x = random.choice([1, 2, 4, 5, 8])
        r_y = random.choice([1, 2, 4, 5])
        perlin_noise = (generate_perlin_noise_2d(x.shape, res=(r_x, r_y)) + 0.0) * factor

        return np.clip(x + perlin_noise, 0.0, 1.0)

    def get_snippet(self, sub_sheet, do_x_shift=False):

        # rescaling
        if self.p_scale > np.random.random():
            scale = (self.sc[1] - self.sc[0]) * np.random.random_sample() + self.sc[0]
            new_size = (int(sub_sheet.shape[1] * scale), int(sub_sheet.shape[0] * scale))
            sub_sheet = cv2.resize(sub_sheet, new_size, interpolation=cv2.INTER_NEAREST)

        # horizontal shift
        x = sub_sheet.shape[1] // 2
        if do_x_shift and self.p_xshift > np.random.random():
            x += np.random.randint(low=-self.x_shift, high=self.x_shift + 1)

        # compute sliding window coordinates
        x0 = int(np.max([x - self.sheet_context // 2, 0]))
        x1 = int(np.min([x0 + self.sheet_context, sub_sheet.shape[1] - 1]))
        x0 = int(x1 - self.sheet_context)

        # vertical shift
        y0 = sub_sheet.shape[0] // 2 - self.staff_height // 2
        if self.p_yshift > np.random.random():
            y_shift = math.floor((sub_sheet.shape[0] - self.staff_height) // 2)
            y0 += np.random.randint(low=-y_shift, high=y_shift + 1)
        y1 = y0 + self.staff_height

        sheet_snippet = sub_sheet[y0:y1, x0:x1]

        # # rotating
        # if self.p_rotate > np.random.random():
        #     sheet_snippet = self.rotate_img(sheet_snippet)
        #
        # if self.p_perlin > np.random.random():
        #     sheet_snippet = self.add_perlin_noise(sheet_snippet)
        #
        # if self.p_et_small > np.random.random():
        #     sheet_snippet = self.apply_elastic_transform_small(sheet_snippet)
        # if self.p_et_large > np.random.random():
        #     sheet_snippet = self.apply_elastic_transform_big(sheet_snippet)
        #
        # # get sheet snippet
        # sheet_snippet = tf.to_tensor(sheet_snippet)
        #
        # if self.p_2dnoise > np.random.random():
        #     sheet_snippet = self.add_gaussian_noise(sheet_snippet)
        # if self.p_blur > np.random.random():
        #     sheet_snippet = self.apply_gaussian_blur(sheet_snippet)

        # return tf.resize(sheet_snippet, [sheet_snippet.shape[1] // 2, sheet_snippet.shape[2] // 2])
        return sheet_snippet

    def plot_snippets(self, item, save_plot=False):

        snippet_1, snippet_2 = self[item]
        snippet_1 = snippet_1.squeeze().numpy()
        snippet_2 = snippet_2.squeeze().numpy()
        px = 1 / plt.rcParams['figure.dpi']  # pixel in inches
        plt.figure(figsize=(490*px*0.8, 200*px*0.8))
        plt.clf()
        plt.subplot(1, 2, 1)
        plt.imshow(snippet_1, cmap='gray')
        plt.xticks([])
        plt.yticks([])
        plt.subplot(1, 2, 2)
        plt.imshow(snippet_2, cmap='gray')
        plt.xticks([])
        plt.yticks([])
        plt.tight_layout()
        plt.savefig(f'{item}.png') if save_plot else plt.show()

    def __getitem__(self, index):

        # get a larger sub-sheet snippet to construct the two views from it
        sub_sheet = self.get_sub_sheet(index)

        snippet_1 = self.get_snippet(sub_sheet, do_x_shift=False)
        snippet_2 = self.get_snippet(sub_sheet, do_x_shift=True)

        # return sub_sheet
        return snippet_1, snippet_2

    def __len__(self):
        return len(self.snippet_samples)


def load_score_dataset(score_paths, args):

    hop = args.sheet_context // 2

    params = []
    for sid, sp in enumerate(score_paths):
        params.append(dict(
            sp=sp,
            sid=sid,
            hop=hop,
            sc=args.sheet_context
        ))

    with Pool(8) as pool:
        results = list(tqdm(pool.imap_unordered(prepare_unrolled_scores, params), total=len(params), ncols=70,
                            leave=False))

    all_samples = [sample for score_samples in results for sample in score_samples[0]]
    all_unrolled_scores = {x[0][0][1]: x[1] for x in results}

    return ScoreSnippetDataset(all_samples, all_unrolled_scores, args)


def prepare_unrolled_scores(p):

    unrolled_score = np.array(PIL.Image.open(p['sp'])) / 255
    us_len = unrolled_score.shape[1]

    # computing the central samples for the snippets
    # todo: compute this in a stochastic way
    seqs = list(range(0, int(us_len - p['sc'] // 2), p['hop']))
    samples = [(s, p['sid']) for s in seqs]

    # return samples, unrolled_score
    return samples, unrolled_score.astype(np.float32)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    configs = load_yaml('config/score_config.yaml')
    for k, v in configs.items():
        parser.add_argument(f"--{k}", default=v, type=type(v))

    args_main = set_remote_paths(parser.parse_args(), score_pretrain=True)

    root = '/home/luis/data/score_db/'
    uw_scores = sorted([str(p) for p in Path(root).rglob('*unrolled_score.png')])

    dataset = load_score_dataset(uw_scores[:60], args_main)

    for i in range(2000, 6000, 38):
        dataset.plot_snippets(i, save_plot=True)
