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
import tensorflow as tf

def log10(x):
  numerator = tf.math.log(x)
  denominator = tf.math.log(tf.constant(10, dtype=numerator.dtype))
  return numerator / denominator

def decode_audio(audio_binary):
    audio, _ = tf.audio.decode_wav(audio_binary)
    return tf.squeeze(audio, axis=-1)

def get_waveform(file_path):

    audio_binary = tf.io.read_file(file_path)
    waveform = decode_audio(audio_binary)
    return waveform


def get_spectrogram(waveform):
    # Padding for files with less than 16000 samples
    zero_padding = tf.zeros([16000] - tf.shape(waveform), dtype=tf.float32)

    # Concatenate audio with padding so that all audio clips will be of the
    # same length
    waveform = tf.cast(waveform, tf.float32)
    equal_length = tf.concat([waveform, zero_padding], 0)
    spectrogram = tf.signal.stft(
        equal_length, frame_length=hp.win_length, frame_step=hp.hop_length)

    mag = tf.abs(spectrogram)
    mel_basis = tf.signal.linear_to_mel_weight_matrix(num_mel_bins=hp.n_mels, sample_rate=hp.sr, num_spectrogram_bins=hp.n_fft)
    mel = tf.tensordot(mel_basis, mag, axes = 1)

    # A decibeles
    mel = 20 * log10(tf.math.maximum(1e-15, mel))
    mag = 20 * log10(tf.math.maximum(1e-15, mag))

    # Normalizar
    mel = tf.clip_by_value((mel - hp.ref_db + hp.max_db) / hp.max_db, 1e-8, 1)
    mag = tf.clip_by_value((mag - hp.ref_db + hp.max_db) / hp.max_db, 1e-8, 1)

    mel = tf.transpose(mel)
    mag = tf.transpose(mag)

    return mel, mag



