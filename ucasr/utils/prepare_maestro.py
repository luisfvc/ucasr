from multiprocessing import Pool
from functools import partial
import os
from pathlib import Path

import librosa
import soundfile as sf
from tqdm import tqdm

from utils import make_dir


def prepare_maestro(filepath, sample_rate, root):

    year, filename = filepath.parts[-2:]
    waveform, _ = librosa.load(str(filepath), sr=sample_rate)

    make_dir(os.path.join(root, year))
    sf.write(os.path.join(root, year, filepath.name), waveform, sr)


if __name__ == '__main__':

    maestro_root = '/home/luis/mounts/home@fs/cp/datasets/maestro-v3.0.0'
    target_root = '/home/luis/data/maestro-v3.0.0'
    make_dir(target_root)

    sr = 22050

    mp_func = partial(prepare_maestro, sample_rate=sr, root=target_root)
    with Pool(10) as pool:
        list(tqdm(pool.imap_unordered(mp_func, Path(maestro_root).rglob('*.wav')), ncols=80))
