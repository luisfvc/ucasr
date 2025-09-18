import argparse
import os

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import Dataset as TorchDataset
from tqdm import tqdm

from ucasr.utils.utils import load_yaml

# matplotlib.use('TkAgg')


class SnippetDataset(TorchDataset):

    def __init__(self, scores, performances, pieces, args):

        self.scores = scores
        self.performances = performances
        self.pieces = pieces

        self.spec_context = int(args.snippet_len[args.audio_context] * args.fps)
        self.spec_bins = args.spec_bins
        self.sheet_context = args.sheet_context
        self.staff_height = args.staff_height

        self.system_translation = args.aug_configs[args.aug]['system_translation']
        self.sheet_scaling = args.aug_configs[args.aug]['sheet_scaling']
        self.onset_translation = args.aug_configs[args.aug]['onset_translation']

        self.train_entities = []
        self.prepare_train_entities()

    def prepare_train_entities(self):

        for i_score, score in tqdm(enumerate(self.scores), total=len(self.scores), ncols=80, leave=False):
            for i_spec, perf in enumerate(self.performances[i_score]):
                spectrogram, o2c_maps = perf['spec'], perf['o2c']
                for i_note, (onset_frame, x_coord) in enumerate(o2c_maps):

                    o_start = onset_frame - self.spec_context // 2
                    o_stop = o_start + self.spec_context

                    c_start = x_coord - self.sheet_context // 2
                    c_stop = c_start + self.sheet_context

                    # only select samples which lie within the valid edges
                    if o_start >= 0 and o_stop < spectrogram.shape[1] and c_start >= 0 and c_stop < score.shape[1]:
                        self.train_entities.append({'i_score': i_score, 'i_spec': i_spec, 'i_note': i_note})

    def prepare_image_snippet(self, i_score, i_spec, i_note):

        sheet = self.scores[i_score]

        target_coord = self.performances[i_score][i_spec]['o2c'][i_note][1]

        sheet_pad_beg = max(0, 2 * self.sheet_context - target_coord)
        sheet_pad_end = max(0, target_coord + 2 * self.sheet_context - sheet.shape[1])
        if sheet_pad_beg or sheet_pad_end:
            sheet_pad_beg_array = np.ones((sheet.shape[0], sheet_pad_beg), dtype=np.float32) * 255
            sheet_pad_end_array = np.ones((sheet.shape[0], sheet_pad_end), dtype=np.float32) * 255
            sheet = np.hstack((sheet_pad_beg_array, sheet, sheet_pad_end_array))
            target_coord += sheet_pad_beg

        # get sub-image (with coordinate fixing)
        # this is done since we do not want to do the augmentation
        # on the whole sheet image
        c0 = max(0, target_coord - 2 * self.sheet_context)
        c1 = min(c0 + 4 * self.sheet_context, sheet.shape[1])
        c0 = max(0, c1 - 4 * self.sheet_context)
        sheet = sheet[:, c0:c1]

        # sheet scaling augmentation
        if (sc := self.sheet_scaling) != [1.00, 1.00]:
            scale = (sc[1] - sc[0]) * np.random.random_sample() + sc[0]
            new_size = (int(sheet.shape[1] * scale), int(sheet.shape[0] * scale))
            sheet = cv2.resize(sheet, new_size, interpolation=cv2.INTER_NEAREST)

        # target coordinate
        x = sheet.shape[1] // 2

        # compute sliding window coordinates
        x0 = int(np.max([x - self.sheet_context // 2, 0]))
        x1 = int(np.min([x0 + self.sheet_context, sheet.shape[1] - 1]))
        x0 = int(x1 - self.sheet_context)

        # system translation augmentation
        r0 = sheet.shape[0] // 2 - self.staff_height // 2
        if t := self.system_translation:
            r0 += np.random.randint(low=-t, high=t + 1)
        r1 = r0 + self.staff_height

        # get sheet snippet
        sheet_snippet = sheet[r0:r1, x0:x1]

        return sheet_snippet

    def prepare_audio_snippet(self, i_score, i_spec, i_note):

        # get spectrogram and onset
        spec = self.performances[i_score][i_spec]['spec']
        sel_onset = int(self.performances[i_score][i_spec]['o2c'][i_note][0])

        # data augmentation note position
        if t := self.onset_translation:
            sel_onset += np.random.randint(low=-t, high=t + 1)

        # compute sliding window coordinates
        start = np.max([sel_onset - self.spec_context // 2, 0])
        stop = start + self.spec_context

        stop = np.min([stop, spec.shape[1] - 1])
        start = stop - self.spec_context

        spec_snippet = spec[:, start:stop]

        return spec_snippet

    def plot_snippet_pair(self, item, save_plot=True):

        score_snippet, spec_snippet = self[item]
        plt.figure()
        plt.clf()
        plt.subplot(1, 2, 1)
        plt.imshow(score_snippet[0], cmap='gray')
        plt.subplot(1, 2, 2)
        plt.imshow(spec_snippet[0], cmap='viridis', origin='lower')
        plt.savefig(f'{item}.png') if save_plot else plt.show()

    @property
    def snippet_ids(self):

        snippet_ids = [self.train_entities[item]['i_score'] for item in range(len(self))]
        return np.array(snippet_ids)

    @property
    def id_to_piece_name(self):
        return {pid: p_name for pid, p_name in enumerate(self.pieces)}

    def __len__(self):
        return len(self.train_entities)

    def __getitem__(self, item):
        o2c: dict = self.train_entities[item]
        i_score, i_spec, i_note = o2c['i_score'], o2c['i_spec'], o2c['i_note']

        score_snippet = self.prepare_image_snippet(i_score, i_spec, i_note)
        spec_snippet = self.prepare_audio_snippet(i_score, i_spec, i_note)

        # resize score snippet to half of its original dimension
        score_snippet = cv2.resize(score_snippet / 255, (score_snippet.shape[1] // 2, score_snippet.shape[0] // 2))
        score_snippet = score_snippet.astype(np.float32)
        return np.expand_dims(score_snippet, axis=0), np.expand_dims(spec_snippet, axis=0)


def load_msmd_piece(path, piece, args, aug):

    npz_content = np.load(os.path.join(path, f'{piece}.npz'), allow_pickle=True)

    # check which performances match the augmentation pattern
    aug_config = args.aug_configs[aug]
    piece_valid_performances = []
    for perf in npz_content['performances']:
        tempo, synth = perf['perf'].split("tempo-")[1].split("_", 1)
        tempo = float(tempo) / 1000

        if synth not in aug_config["synths"] or tempo < aug_config["tempo_range"][0] \
                or tempo > aug_config["tempo_range"][1]:
            continue
        piece_valid_performances.append(perf)

    return npz_content['unrolled_score'], piece_valid_performances


def load_msmd_dataset(path, pieces, args, aug='full_aug'):

    scores = []
    performances = []
    for piece in pieces:
        unrolled_score, piece_valid_performances = load_msmd_piece(path, piece, args, aug)
        scores.append(unrolled_score)
        performances.append(piece_valid_performances)

    return SnippetDataset(scores, performances, pieces, args)


def load_umc_dataset(path, pieces, args):

    scores = []
    performances = []
    for piece in pieces:
        npz_content = np.load(os.path.join(path, f'{piece}.npz'), allow_pickle=True)
        scores.append(npz_content['unrolled_score'])
        performances.append(npz_content['performances'])

    return SnippetDataset(scores, performances, pieces, args)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--audio_context', help='audio snippet size', type=str, default='mc')

    configs = load_yaml('config/msmd_config.yaml')
    for k, v in configs.items():
        parser.add_argument(f"--{k}", default=v, type=type(v))

    prepared_msmd_path = '/home/luis/data/prepared_umc_alignments'
    split = load_yaml('splits/db_scanned_recording.yaml')['test']

    dataset = load_umc_dataset(prepared_msmd_path, split, parser.parse_args())
    n_test = 2_000
    te_samples = np.linspace(0, len(dataset) - 1, n_test).astype(int)
    for i in te_samples[:100]:
        dataset.plot_snippet_pair(i, save_plot=True)
