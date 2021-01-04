from __future__ import print_function
from tacotron.hyperparams import Hyperparams as hp
import numpy as np
import tensorflow as tf
import codecs
import re
import os
import unicodedata

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
