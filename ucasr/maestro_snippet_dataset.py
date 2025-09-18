import argparse
import math
from multiprocessing import Pool
import os

from audiomentations import (Compose, AddGaussianSNR, Gain, TimeMask, SevenBandParametricEQ, PolarityInversion,
                             ApplyImpulseResponse)
import librosa.effects
import numpy as np
import pandas as pd
from scipy.io.wavfile import write
from torch.utils.data import Dataset as TorchDataset
from tqdm import tqdm

from ucasr.utils.utils import load_yaml, set_remote_paths


class MAESTROSnippetDataset(TorchDataset):

    def __init__(self, snippet_samples, id_to_recordings, recordings, args):

        snippet_len = args.snippet_len[args.audio_context]

        self.snippet_samples = snippet_samples
        self.id_to_recordings = id_to_recordings
        self.recordings = recordings

        # if there's temporal stretch augmentation, the final snippet size is increased to accommodate temporal
        # variations
        self.padded_snippet_len_sec = snippet_len * args.tstretch_range[1] if args.p_tstretch else snippet_len
        self.padded_snippet_len_samples = math.ceil(self.padded_snippet_len_sec * args.sr)
        self.snippet_len_sec = snippet_len
        self.snippet_len_samples = int(self.snippet_len_sec * args.sr)

        self.sr = args.sr
        self.max_shift = int(snippet_len * (1 - args.tshift_overlap) * args.sr)
        self.tmask = args.tmask_max

        self.p_tshift = args.p_tshift
        self.augment = Compose([
                                ApplyImpulseResponse(ir_path=args.rir_root, p=args.p_rir_cpu),
                                PolarityInversion(p=args.p_polarity),
                                Gain(p=args.p_gain),
                                AddGaussianSNR(p=args.p_noise),
                                TimeMask(min_band_part=0.01, max_band_part=args.tmask_max, p=args.p_tmask),
                                SevenBandParametricEQ(p=args.p_eq)
                                ])

    def get_waveform_snippet(self, sample, shift=True):

        shift_size = np.random.randint(low=0, high=self.max_shift) if shift else 0
        start_sample = sample['start_sample'] + shift_size
        end_sample = start_sample + self.padded_snippet_len_samples

        wf = self.recordings[sample['piece_id']][start_sample:end_sample]

        # pad with zeroes in the end if the snippet is smaller than the expected size
        wf = np.pad(wf, (0, self.padded_snippet_len_samples - wf.shape[0]), mode='constant', constant_values=0)
        return wf

    def render_snippets(self, index):

        wf_1, wf_2 = self[index]
        write(f'audio_snippet_item_{index}_1.wav', self.sr, wf_1)
        write(f'audio_snippet_item_{index}_2.wav', self.sr, wf_2)

    def __getitem__(self, index):

        # fetch sample from the data
        sample = self.snippet_samples[index]

        # loading the two views already with time shift augmentation
        snippet_wf1 = self.get_waveform_snippet(sample, shift=False)

        do_shift_wf2 = self.p_tshift > np.random.uniform()
        snippet_wf2 = self.get_waveform_snippet(sample, do_shift_wf2)

        snippet_wf1 = self.augment(snippet_wf1, sample_rate=self.sr)
        snippet_wf2 = self.augment(snippet_wf2, sample_rate=self.sr)

        return snippet_wf1, snippet_wf2

    def __len__(self):
        return len(self.snippet_samples)


def load_maestro_dataset(recordings_paths, args):

    snippet_size_samples = int(args.snippet_len[args.audio_context] * args.sr)
    hop_size_samples = int(snippet_size_samples * args.hop)

    params = []
    for pid, rec_path in enumerate(recordings_paths):
        params.append(dict(
            rec_path=rec_path,
            pid=pid,
            hop=hop_size_samples,
            snippet_size_samples=snippet_size_samples
        ))
    with Pool(10) as pool:
        results = list(tqdm(pool.imap_unordered(prepare_maestro_rec, params), total=len(params), ncols=70, leave=False))

    id_to_recordings = {pid: str(perf_path) for pid, perf_path in enumerate(recordings_paths)}

    all_samples = [sample for piece_samples in results for sample in piece_samples[0]]
    all_recordings = {x[0][0]['piece_id']: x[1] for x in results}

    return MAESTROSnippetDataset(all_samples, id_to_recordings, all_recordings, args)


def prepare_maestro_rec(p):

    waveform, _ = librosa.load(p['rec_path'], sr=None)
    signal_len = waveform.shape[0]

    # computing the starting samples for the snippets
    seqs = list(range(0, int(signal_len - p['snippet_size_samples']), p['hop']))
    samples = [{'start_sample': s, 'piece_id': p['pid']} for s in seqs]

    return samples, waveform


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--audio_context', help='audio snippet size', type=str, default='mc')

    configs = load_yaml('config/maestro_config.yaml')
    for k, v in configs.items():
        parser.add_argument(f"--{k}", default=v, type=type(v))

    args_main = set_remote_paths(parser.parse_args(), audio_pretrain=True)

    maestro_split_file = os.path.join(args_main.split_root, 'maestro_split_10.csv')
    df = pd.read_csv(maestro_split_file, sep=",")

    tr_recordings = [os.path.join(args_main.maestro_root, (piece['audio_filename'])) for _, piece in df.iterrows()
                     if piece['split'] == 'train']

    dataset = load_maestro_dataset(tr_recordings, args_main)
    for i in range(10, 100, 5):
        dataset.render_snippets(i)
