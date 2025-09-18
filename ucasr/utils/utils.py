import os
from pathlib import Path

import librosa
import numpy as np
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter
import soundfile as sf
import torch
import yaml


def load_yaml(yaml_fn):
    with open(yaml_fn, 'rb') as f:
        content = yaml.load(f, Loader=yaml.FullLoader)
    return content


def make_dir(dir_path):
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)


def get_maestro_splits(args):
    pass


def set_remote_paths(args, audio_pretrain=False, score_pretrain=False, mixed_pretrain=False):
    hostname = os.uname()[1]
    rks = ["rechenknecht%d.cp.jku.at" % i for i in range(9)] + ["rechenknecht%d" % i for i in range(9)]

    args.exp_root = args.remote_exp_root if hostname in rks else args.local_exp_root
    args.split_root = args.remote_split_root if hostname in rks else args.local_split_root

    if hostname in rks:
        args.rir_root = '/share/cp/datasets/impulse_response_filters/mcdermottlab_22k'
    else:
        args.rir_root = '/home/luis/dev/impulse_response_filters/resampled_rirs/mcdermottlab_22k'

    if audio_pretrain:
        args.maestro_root = args.remote_maestro_root if hostname in rks else args.local_maestro_root
        return args

    if score_pretrain:
        args.scores_root = args.remote_scores_root if hostname in rks else args.local_scores_root
        return args

    if mixed_pretrain:
        args.maestro_root = args.remote_maestro_root if hostname in rks else args.local_maestro_root
        args.scores_root = args.remote_scores_root if hostname in rks else args.local_scores_root
        return args

    args.msmd_root = args.remote_msmd_root if hostname in rks else args.local_msmd_root
    args.umc_root = args.remote_umc_root if hostname in rks else args.local_umc_root

    return args


def load_pretrained_model(args, network, audio=True):
    model_dict = network.state_dict()

    encoder = 'audio' if audio else 'score'
    net = 'y' if audio else 'x'

    audio = True if args.finetune_mixed else audio
    encoder = 'mixed' if args.finetune_mixed else encoder

    params_dir = os.path.join(args.exp_root, 'pretrained_models')
    if args.ft_audio_path and audio:
        params_dir = args.ft_audio_path
    if args.ft_score_path and not audio:
        params_dir = args.ft_score_path
    pretrained_model_path = os.path.join(params_dir, f"params_{encoder}{f'_{args.audio_context}' if audio else''}.pt")
    pretrained_model_path = pretrained_model_path.replace('.pt', '_lm.pt') if args.ft_last_run else pretrained_model_path
    pretrained_dict = torch.load(pretrained_model_path)['model_params']

    # making sure the names of the layers will match
    if args.finetune_mixed:
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and 'encoder' in k}
    else:
        pretrained_dict = {f'{net}_net.{k}': v for k, v in pretrained_dict.items()
                           if f'{net}_net.{k}' in model_dict and 'encoder' in k}

    # overwrite entries in the existing state dict
    model_dict.update(pretrained_dict)
    # load the new state dict
    network.load_state_dict(model_dict)
    print(f'Pretrained {encoder} encoder loaded from {pretrained_model_path}')
    return network


def prepare_rirs(sr):
    rir_root = '/home/luis/dev/impulse_response_filters/resampled_rirs/mcdermottlab'
    target_root = '/home/luis/dev/impulse_response_filters/resampled_rirs/mcdermottlab_22k'
    make_dir(target_root)

    rir_paths = Path(rir_root).rglob('*.wav')
    for rir in rir_paths:
        x, _ = librosa.load(str(rir), sr=sr)
        sf.write(os.path.join(target_root, rir.name), x, sr)


def elastic_transform(image, alpha, sigma, random_state=None):
    """Elastic deformation of images as described in [Simard2003].

    .. [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
       Convolutional Neural Networks applied to Visual Document Analysis", in
       Proc. of the International Conference on Document Analysis and
       Recognition, 2003.
    """
    if random_state is None:
        random_state = np.random.RandomState(None)

    h, w = image.shape[:2]
    x, y = np.meshgrid(np.arange(w), np.arange(h))
    dx = gaussian_filter((random_state.rand(h, w) * 2 - 1), sigma, mode="constant", cval=0) * alpha
    dy = gaussian_filter((random_state.rand(h, w) * 2 - 1), sigma, mode="constant", cval=0) * alpha
    indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1))

    if len(image.shape) > 2:
        c = image.shape[2]
        distorted_image = [map_coordinates(image[:, :, j], indices, order=1, mode='reflect') for j in range(c)]
        distorted_image = np.concatenate(distorted_image, axis=1)
    else:
        distorted_image = map_coordinates(image, indices, order=1, mode='reflect')

    return distorted_image.reshape(image.shape)


def interpolant(t):
    return t * t * t * (t * (t * 6 - 15) + 10)


def generate_perlin_noise_2d(
        shape, res, tileable=(False, False), interpolant=interpolant
):
    """
    Credits: https://github.com/pvigier/perlin-numpy/blob/master/perlin_numpy/perlin2d.py
    Generate a 2D numpy array of perlin noise.
    Args:
        shape: The shape of the generated array (tuple of two ints).
            This must be a multple of res.
        res: The number of periods of noise to generate along each
            axis (tuple of two ints). Note shape must be a multiple of
            res.
        tileable: If the noise should be tileable along each axis
            (tuple of two bools). Defaults to (False, False).
        interpolant: The interpolation function, defaults to
            t*t*t*(t*(t*6 - 15) + 10).
    Returns:
        A numpy array of shape shape with the generated noise.
    Raises:
        ValueError: If shape is not a multiple of res.
    """
    delta = (res[0] / shape[0], res[1] / shape[1])
    d = (shape[0] // res[0], shape[1] // res[1])
    grid = np.mgrid[0:res[0]:delta[0], 0:res[1]:delta[1]] \
               .transpose(1, 2, 0) % 1
    # Gradients
    angles = 2 * np.pi * np.random.rand(res[0] + 1, res[1] + 1)
    gradients = np.dstack((np.cos(angles), np.sin(angles)))
    if tileable[0]:
        gradients[-1, :] = gradients[0, :]
    if tileable[1]:
        gradients[:, -1] = gradients[:, 0]
    gradients = gradients.repeat(d[0], 0).repeat(d[1], 1)
    g00 = gradients[:-d[0], :-d[1]]
    g10 = gradients[d[0]:, :-d[1]]
    g01 = gradients[:-d[0], d[1]:]
    g11 = gradients[d[0]:, d[1]:]
    # Ramps
    n00 = np.sum(np.dstack((grid[:, :, 0], grid[:, :, 1])) * g00, 2)
    n10 = np.sum(np.dstack((grid[:, :, 0] - 1, grid[:, :, 1])) * g10, 2)
    n01 = np.sum(np.dstack((grid[:, :, 0], grid[:, :, 1] - 1)) * g01, 2)
    n11 = np.sum(np.dstack((grid[:, :, 0] - 1, grid[:, :, 1] - 1)) * g11, 2)
    # Interpolation
    t = interpolant(grid)
    n0 = n00 * (1 - t[:, :, 0]) + t[:, :, 0] * n10
    n1 = n01 * (1 - t[:, :, 0]) + t[:, :, 0] * n11
    return (np.sqrt(2) * ((1 - t[:, :, 1]) * n0 + t[:, :, 1] * n1)).astype(np.float32)


def generate_perlin_noise_2d_torch(shape, res, fade=interpolant):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    delta = (res[0] / shape[0], res[1] / shape[1])
    d = (shape[0] // res[0], shape[1] // res[1])

    grid = (torch.stack(torch.meshgrid(torch.arange(0, res[0], delta[0]), torch.arange(0, res[1], delta[1])),
                        dim=-1) % 1).to(device)
    angles = (2 * torch.pi * torch.rand(res[0] + 1, res[1] + 1)).to(device)
    gradients = (torch.stack((torch.cos(angles), torch.sin(angles)), dim=-1)).to(device)

    tile_grads = lambda slice1, slice2: gradients[slice1[0]:slice1[1], slice2[0]:slice2[1]].repeat_interleave(d[0],
                                                                                                              0).repeat_interleave(
        d[1], 1)
    dot = lambda grad, shift: (
            torch.stack((grid[:shape[0], :shape[1], 0] + shift[0], grid[:shape[0], :shape[1], 1] + shift[1]),
                        dim=-1) * grad[:shape[0], :shape[1]]).sum(dim=-1)

    n00 = dot(tile_grads([0, -1], [0, -1]), [0, 0])
    n10 = dot(tile_grads([1, None], [0, -1]), [-1, 0])
    n01 = dot(tile_grads([0, -1], [1, None]), [0, -1])
    n11 = dot(tile_grads([1, None], [1, None]), [-1, -1])
    t = fade(grid[:shape[0], :shape[1]])
    return torch.sqrt(torch.tensor([2])).to(device) * torch.lerp(torch.lerp(n00, n10, t[..., 0]),
                                                                 torch.lerp(n01, n11, t[..., 0]), t[..., 1])


if __name__ == '__main__':
    pass
