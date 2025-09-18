import math
import random

import elasticdeform.torch as etorch
import torch
import torch.nn as nn
import torchaudio
import torchvision.transforms.functional as tf
from torch_audiomentations import Compose, ApplyImpulseResponse
from madmom.audio.stft import fft_frequencies
from madmom.audio.spectrogram import LogarithmicFilterbank

from ucasr.utils.utils import generate_perlin_noise_2d_torch

torchaudio.set_audio_backend("sox_io")


class LogSpectrogramModule(nn.Module):
    """
    credits:
    https://github.com/CPJKU/cyolo_score_following/blob/eusipco-2021/cyolo_score_following/models/custom_modules.py
    """

    def __init__(self, args):
        super(LogSpectrogramModule, self).__init__()

        self.sr = args.sr
        self.fps = args.fps
        self.n_fft = args.fft_frame_size
        self.hop_length = int(self.sr / self.fps) + 1
        self.min_rate = args.tstretch_range[0]
        self.max_rate = args.tstretch_range[1]
        self.p_tstretch = args.p_tstretch
        self.p_fmask = args.p_fmask
        self.tc = int(args.fps * args.snippet_len[args.audio_context])  # temporal context

        fbank = LogarithmicFilterbank(fft_frequencies(self.n_fft // 2 + 1, self.sr),
                                      num_bands=16, fmin=30, fmax=6000, norm_filters=True, unique_filters=True)
        fbank = torch.from_numpy(fbank)
        phase_advance = torch.linspace(0, math.pi * self.hop_length, self.n_fft // 2 + 1)[..., None]

        self.register_buffer('window', torch.hann_window(self.n_fft))
        self.register_buffer('fbank', fbank.unsqueeze(0))
        self.register_buffer('phase_advance', phase_advance)

        max_freq_mask = int(fbank.shape[1] * args.fmask_max)
        self.freq_masking = torchaudio.transforms.FrequencyMasking(freq_mask_param=max_freq_mask, iid_masks=True)

        # self.apply_impulse_response = Compose(transforms=[
        #     ApplyImpulseResponse(ir_paths=args.rir_root, p=args.p_rir_gpu, sample_rate=self.sr, target_rate=self.sr)
        # ])

    def compute_batch_spectrograms(self, x):

        # x = self.apply_impulse_response(x.unsqueeze(dim=1)).squeeze()

        x_stft_batch = torch.stft(x.squeeze(), n_fft=self.n_fft, hop_length=self.hop_length, window=self.window,
                                  center=True, return_complex=True)

        if self.p_tstretch:
            specs = []
            for spec in x_stft_batch:
                if self.p_tstretch > random.random():
                    rate = random.uniform(self.min_rate, self.max_rate)
                    spec = torchaudio.functional.phase_vocoder(spec, rate, self.phase_advance)
                spec = spec[:, (spec.shape[1] - self.tc) // 2:(spec.shape[1] - self.tc) // 2 + self.tc]
                specs.append(spec)
            specs = torch.stack(specs, dim=0)
        else:
            specs = x_stft_batch

        assert self.tc == specs.shape[-1]

        specs = torch.view_as_real(specs).pow(2).sum(-1).sqrt().permute(0, 2, 1)
        specs = torch.log10(torch.matmul(specs, self.fbank) + 1).permute(0, 2, 1).unsqueeze(dim=1)

        # apply freq masking
        if random.random() < self.p_fmask:
            specs = self.freq_masking(specs)

        return specs

    def forward(self, x):

        return self.compute_batch_spectrograms(x)


class ScoreAugModule(nn.Module):

    def __init__(self, args):
        super(ScoreAugModule, self).__init__()

        self.sheet_context = args.sheet_context // 2
        self.staff_height = args.staff_height // 2

        self.pxl_avg = [0, 0.5]
        self.pxl_var = [0.0, 0.4]
        self.max_angle = 4
        self.gb_ks = [7, 9, 11, 13, 15]
        # self.gb_ks = [3, 5, 7, 9, 11]
        self.gb_sigma = [0, 0.6]

        self.args = args
        self.sc = args.sheet_scaling
        self.x_shift = int((1 - args.xshift_overlap) * args.sheet_context) // 2

        self.p_xshift = args.p_xshift
        self.p_yshift = args.p_yshift
        self.p_scale = args.p_scale
        self.p_rotate = args.p_rotate
        self.p_2dnoise = args.p_2dnoise
        self.p_blur = args.p_blur
        self.p_perlin = args.p_perlin
        self.p_et_small = args.p_et_small
        self.p_et_large = args.p_et_large

    def rotate_img(self, sub_sheet):

        angle = 2 * self.max_angle * random.random() - self.max_angle
        return tf.rotate(sub_sheet, angle=angle, fill=[1.0, ])

    def add_gaussian_noise(self, x):

        rd_avg = (self.pxl_avg[1] - self.pxl_avg[0]) * random.random() + self.pxl_avg[0]
        rd_var = (self.pxl_var[1] - self.pxl_var[0]) * random.random() + self.pxl_var[0]

        noise = torch.randn(1, x.shape[-2] * 2, x.shape[-1] * 2, device=x.device) * rd_var + rd_avg
        noisy_sheet = x + tf.resize(noise, [x.shape[-2], x.shape[-1]])

        return torch.clip(noisy_sheet, 0.0, 1.0)

    def add_gaussian_noise_(self, x):

        rd_avg = (self.pxl_avg[1] - self.pxl_avg[0]) * random.random() + self.pxl_avg[0]
        rd_var = (self.pxl_var[1] - self.pxl_var[0]) * random.random() + self.pxl_var[0]

        noisy_sheet = x + torch.randn_like(x) * rd_var + rd_avg

        return torch.clip(noisy_sheet, 0.0, 1.0)

    def apply_gaussian_blur(self, x):

        ks = random.choice(self.gb_ks)
        sigma = (self.gb_sigma[1] - self.gb_sigma[0]) * random.random() + self.gb_sigma[0]

        return tf.gaussian_blur(x, kernel_size=ks, sigma=sigma)

    def add_perlin_noise(self, x):

        factor = random.random() * 0.6
        r_x = random.choice([1, 2, 4, 5, 8])
        r_y = random.choice([1, 2, 4, 5])
        perlin_noise = (generate_perlin_noise_2d_torch(x[0].shape, res=(r_x, r_y)) + 0.0) * factor

        return torch.clip(x + perlin_noise.unsqueeze(0), 0.0, 1.0)

    def apply_elastic_transform_small(self, x):

        alpha = int((200 - 50) * random.random() + 50)
        sigma = 0.2
        displacement = torch.randn(2, alpha, alpha) * sigma
        x = etorch.deform_grid(x, displacement=displacement, cval=1.0, prefilter=False, axis=(1, 2))
        return x

    def apply_elastic_transform_big(self, x):

        alpha = int((10 - 2) * random.random() + 2)
        sigma = 0.5
        displacement = torch.randn(2, alpha, alpha) * sigma
        x = etorch.deform_grid(x, displacement=displacement, cval=1.0, prefilter=False, axis=(1, 2))
        return x

    def forward(self, subsheets_batch, do_x_shift=True):

        snippets = []
        for sheet_snippet in subsheets_batch:
            sheet_snippet = sheet_snippet.unsqueeze(0)

            # # rescaling
            # if self.p_scale > random.random():
            #     scale = (self.sc[1] - self.sc[0]) * random.random() + self.sc[0]
            #     new_size = [int(sub_sheet.shape[1] * scale), int(sub_sheet.shape[2] * scale)]
            #     sub_sheet = tf.resize(sub_sheet, new_size)
            #
            # # horizontal shift
            # x = sub_sheet.shape[2] // 2
            # if do_x_shift and self.p_xshift > random.random():
            #     x += random.randint(-self.x_shift, self.x_shift)
            #
            # # compute sliding window coordinates
            # x0 = x - self.sheet_context // 2
            # x1 = x0 + self.sheet_context
            #
            # # vertical shift
            # y0 = sub_sheet.shape[1] // 2 - self.staff_height // 2
            # if self.p_yshift > random.random():
            #     y_shift = math.floor((sub_sheet.shape[1] - self.staff_height) // 2)
            #     y0 += random.randint(-y_shift, y_shift)
            # y1 = y0 + self.staff_height
            #
            # sheet_snippet = sub_sheet[:, y0:y1, x0:x1]

            # rotate
            if self.p_rotate > random.random():
                sheet_snippet = self.rotate_img(sheet_snippet)

            # perlin noise
            if self.p_perlin > random.random():
                sheet_snippet = self.add_perlin_noise(sheet_snippet)

            # elastic deformations
            if self.p_et_small > random.random():
                sheet_snippet = self.apply_elastic_transform_small(sheet_snippet)
            if self.p_et_large > random.random():
                sheet_snippet = self.apply_elastic_transform_big(sheet_snippet)

            # adding gaussian noise
            if self.p_2dnoise > random.random():
                sheet_snippet = self.add_gaussian_noise(sheet_snippet)

            # gaussian blur
            if self.p_blur > random.random():
                sheet_snippet = self.apply_gaussian_blur(sheet_snippet)

            # sheet_snippet = tf.resize(sheet_snippet, [sheet_snippet.shape[1] // 2, sheet_snippet.shape[2] // 2])
            snippets.append(sheet_snippet)

        snippets = torch.stack(snippets, dim=0)

        return snippets


if __name__ == '__main__':
    pass
