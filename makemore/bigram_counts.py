# bigram language model by counting: P(next char | current char) straight from the data

import numpy as np

from data import load_words, build_vocab, split_words, build_dataset


def count_bigrams(X, Y, vocab_size):
    N = np.zeros((vocab_size, vocab_size), dtype=np.int64)
    np.add.at(N, (X[:, 0], Y), 1)
    return N


def probs_from_counts(N, smoothing=1):
    # add-one smoothing so no bigram has probability 0 (that would be infinite loss)
    P = (N + smoothing).astype(np.float64)
    return P / P.sum(axis=1, keepdims=True)


def nll(P, X, Y):
    # average negative log likelihood: the number we're trying to minimize
    return -np.log(P[X[:, 0], Y]).mean()


def sample(P, itos, n=10, seed=2147483647):
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        ix, name = 0, []
        while True:
            ix = rng.choice(len(P), p=P[ix])
            if ix == 0:
                break
            name.append(itos[ix])
        out.append(''.join(name))
    return out


def plot_counts(N, itos, path='bigram_counts.png'):
    import matplotlib.pyplot as plt

    V = len(itos)
    plt.figure(figsize=(14, 14))
    plt.imshow(N, cmap='Blues')
    for i in range(V):
        for j in range(V):
            plt.text(j, i, itos[i] + itos[j], ha='center', va='bottom', color='gray', fontsize=7)
            plt.text(j, i, N[i, j], ha='center', va='top', color='gray', fontsize=7)
    plt.axis('off')
    plt.savefig(path, dpi=80, bbox_inches='tight')
    print(f"saved {path}")


if __name__ == '__main__':
    words = load_words()
    stoi, itos = build_vocab(words)
    train_w, val_w, test_w = split_words(words)
    print(f"{len(words)} words, vocab size {len(stoi)}, split {len(train_w)}/{len(val_w)}/{len(test_w)}")

    Xtr, Ytr = build_dataset(train_w, stoi, block_size=1)
    Xva, Yva = build_dataset(val_w, stoi, block_size=1)

    N = count_bigrams(Xtr, Ytr, len(stoi))
    P = probs_from_counts(N)

    print(f"train nll {nll(P, Xtr, Ytr):.4f}")
    print(f"val nll   {nll(P, Xva, Yva):.4f}")
    print(f"(uniform guessing would be {np.log(len(stoi)):.4f})")

    print("\nsamples:")
    for s in sample(P, itos):
        print(" ", s)

    plot_counts(N, itos)
