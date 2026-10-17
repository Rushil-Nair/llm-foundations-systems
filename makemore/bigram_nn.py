# the same bigram model, but learned with gradient descent on tensorgrad instead of counted.
# a single 27x27 weight matrix W holds log-counts ("logits"), softmax turns a row into P(next | current)

import os
import sys
import time

import numpy as np

sys.path.insert(1, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'tensorgrad'))

from tensor import Tensor, cross_entropy
from data import load_words, build_vocab, split_words, build_dataset
from bigram_counts import count_bigrams, probs_from_counts, nll


def softmax_np(x):
    e = np.exp(x - x.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


if __name__ == '__main__':
    words = load_words()
    stoi, itos = build_vocab(words)
    train_w, val_w, _ = split_words(words)
    Xtr, Ytr = build_dataset(train_w, stoi, block_size=1)
    Xva, Yva = build_dataset(val_w, stoi, block_size=1)
    V = len(stoi)
    print(f"{len(Xtr)} training bigrams")

    rng = np.random.default_rng(2147483647)
    W = Tensor(rng.standard_normal((V, V)), dtype=np.float32)

    steps, lr, reg = 300, 50.0, 0.01
    t0 = time.perf_counter()
    for k in range(steps):
        # onehot(x) @ W just selects row x of W, so index instead of building a
        # (n, 27) one-hot matrix and doing a matmul. same math, way less work
        logits = W[Xtr[:, 0]]
        # the regularizer pulls W toward 0 = uniform probabilities,
        # the gradient-descent version of add-one smoothing
        loss = cross_entropy(logits, Ytr) + reg * (W ** 2).mean()

        W.grad = np.zeros_like(W.data)
        loss.backward()
        W.data -= lr * W.grad

        if k % 50 == 0 or k == steps - 1:
            print(f"step {k:>3} loss {loss.data:.4f}")
    print(f"trained in {time.perf_counter() - t0:.1f}s")

    P_nn = softmax_np(W.data.astype(np.float64))
    P_counts = probs_from_counts(count_bigrams(Xtr, Ytr, V))

    print(f"\ntrain nll  neural {nll(P_nn, Xtr, Ytr):.4f}   counting {nll(P_counts, Xtr, Ytr):.4f}")
    print(f"val nll    neural {nll(P_nn, Xva, Yva):.4f}   counting {nll(P_counts, Xva, Yva):.4f}")
    print(f"max |P_nn - P_counts| = {np.abs(P_nn - P_counts).max():.4f}")
    # both models can only express a 27x27 table, so gradient descent ends up
    # (almost) exactly where counting started
