from __future__ import print_function
from hyperparams import Hyperparams as hp
import numpy as np
import codecs
import re
import os
import unicodedata
import tensorflow as tf
from utils import *

def load_vocab():
    char2index = {char:idx for idx, char in enumerate(hp.vocab)}
    idx2char = {idx:char for idx, char in enumerate(hp.vocab)}

    return char2index, idx2char

def text_normalize(text):

    accents = ('COMBINING ACUTE ACCENT', 'COMBINING GRAVE ACCENT')
    accents = set(map(unicodedata.lookup, accents))
    chars = [c for c in unicodedata.normalize('NFD', text) if c not in accents]

    text = unicodedata.normalize('NFC', ''.join(chars))
    text = text.lower()
    text = re.sub("[^{}]".format(hp.vocab), " ", text)
    text = re.sub("[ ]+", " ", text)
    return text

def load_data(mode = "train"):
    char2index, index2char = load_vocab()

    if mode in ("train", "eval"):
        fpaths, text_lengths, texts = [], [], []
        transcript = os.path.join(hp.data, 'transcript.csv')
        lines = codecs.open(transcript, 'r', 'utf-8').readlines()
        total_hours = 0

        if mode == "train":
            lines = lines[:]

        for line in lines:
            fname, _, text = line.strip().split("|")

            fpath = os.path.join(hp.data, "wavs", fname + ".wav")
            fpaths.append(fpath)

            text = text_normalize(text) + "E"  # E: EOS
            text = [char2index[char] for char in text]

            text_lengths.append(len(text))
            texts.append(np.array(text, np.int32))

        return fpaths, text_lengths, texts

def get_batch():
    with tf.device('/cpu:0'):
        fpaths, text_lengths, texts = load_data()
        max_len, min_len = max(text_lengths), min(text_lengths)
        print(fpaths)
        print(type(fpaths[0]))
        # Cantidad de batchs
        num_batch = len(fpaths) // hp.batch_size

        fpaths = tf.convert_to_tensor(fpaths, dtype=tf.string)
        text_lengths = tf.convert_to_tensor(text_lengths)
        texts = tf.convert_to_tensor(texts)

        print(fpaths)
        print(type(fpaths[0]))

        tf.data.Dataset.from_tensor_slices(tuple([fpaths, text_lengths, texts]))
        text = tf.io.decode_raw(texts, tf.int32)

        fname, mel, mag = tf.py_function(load_spectrograms, [fpaths], [tf.string, tf.float32, tf.float32])

