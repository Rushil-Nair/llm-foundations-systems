import os
import random

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def load_words(path=os.path.join(HERE, 'names.txt')):
    with open(path) as f:
        return [w for w in f.read().splitlines() if w]


def build_vocab(words):
    # '.' is both the start and end token, everything else is a-z
    chars = sorted(set(''.join(words)))
    stoi = {s: i + 1 for i, s in enumerate(chars)}
    stoi['.'] = 0
    itos = {i: s for s, i in stoi.items()}
    return stoi, itos


def split_words(words, seed=42):
    # 80/10/10 train/val/test, split on whole words so no name leaks across splits
    words = list(words)
    random.Random(seed).shuffle(words)
    n1, n2 = int(0.8 * len(words)), int(0.9 * len(words))
    return words[:n1], words[n1:n2], words[n2:]


def build_dataset(words, stoi, block_size):
    # every position in every word becomes one example:
    # the previous block_size chars (padded with '.') -> the next char
    X, Y = [], []
    for w in words:
        context = [0] * block_size
        for ch in w + '.':
            ix = stoi[ch]
            X.append(context)
            Y.append(ix)
            context = context[1:] + [ix]
    return np.array(X, dtype=np.int64), np.array(Y, dtype=np.int64)
