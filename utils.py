from __future__ import print_function, division

from hyperparams import Hyperparams as hp
import numpy as np
import librosa
import copy
import matplotlib
matplotlib.use('pdf')
import matplotlib.pyplot as plt
from scipy import signal
import os

def get_spectrograms(fpath):
    # Cargamos el archivo de audio
    y, sample_r = librosa.load(fpath, sr=hp.sr)

    # Recortamos el inicio y final del silencio
    y, _ = librosa.effects.trim(y)

    # Pre-enfasis
    y = np.append(y[0], y[1:] - hp.preemphasis * y[:-1])

    # Short Time Fourier Transform
    linear = librosa.stft(y=y,
                          n_fft=hp.n_fft,
                          hop_length=hp.hop_length,
                          win_length=hp.win_length)
    # Magnitud
    mag = np.abs(linear)

    # Mel Spectogram
    mel_base = librosa.filters.mel(hp.sr, hp.n_fft, hp.n_mels)
    mel = np.dot(mel_base, mag)

    # En decibeles
    mel = 20 * np.log10(np.maximum(1e-5, mel))
    mag = 20 * np.log10(np.maximum(1e-5, mag))

    mel = np.clip((mel - hp.ref_db + hp.max_db) / hp.max_db, 1e-8, 1)
    mag = np.clip((mag - hp.ref_db + hp.max_db) / hp.max_db, 1e-8, 1)

    # Transpose
    mel = mel.T.astype(np.float32)
    mag = mag.T.astype(np.float32)

    return mel, mag