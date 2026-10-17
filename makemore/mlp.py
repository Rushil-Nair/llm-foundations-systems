# character-level MLP language model (Bengio et al. 2003) trained on tensorgrad.
# look at the previous 3 chars, embed each one, concat, tanh hidden layer, predict the next char

import os
import sys
import time

import numpy as np

sys.path.insert(1, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'tensorgrad'))

from tensor import Tensor, cross_entropy
from data import load_words, build_vocab, split_words, build_dataset

BLOCK_SIZE = 3     # how many previous chars we look at
N_EMBD = 10        # size of each char embedding
N_HIDDEN = 200
BATCH_SIZE = 32
STEPS = 200_000
SEED = 2147483647


class CharMLP:

    def __init__(self, vocab_size, block_size=BLOCK_SIZE, n_embd=N_EMBD, n_hidden=N_HIDDEN, seed=SEED):
        rng = np.random.default_rng(seed)
        fan_in = block_size * n_embd
        self.block_size = block_size
        self.C = Tensor(rng.standard_normal((vocab_size, n_embd)), dtype=np.float32)
        # tanh gain 5/3 over sqrt(fan_in) so the hidden layer doesn't start saturated
        self.W1 = Tensor(rng.standard_normal((fan_in, n_hidden)) * (5 / 3) / fan_in ** 0.5, dtype=np.float32)
        self.b1 = Tensor(rng.standard_normal(n_hidden) * 0.01, dtype=np.float32)
        # tiny output layer -> logits start near 0 -> initial loss near log(27) instead of huge
        self.W2 = Tensor(rng.standard_normal((n_hidden, vocab_size)) * 0.01, dtype=np.float32)
        self.b2 = Tensor(np.zeros(vocab_size), dtype=np.float32)

    def parameters(self):
        return [self.C, self.W1, self.b1, self.W2, self.b2]

    def __call__(self, X):
        emb = self.C[X]                                   # (B, block_size, n_embd)
        x = emb.reshape(emb.shape[0], -1)                 # (B, block_size * n_embd)
        h = (x @ self.W1 + self.b1).tanh()                # (B, n_hidden)
        return h @ self.W2 + self.b2                      # (B, vocab_size)

    def logits_np(self, X):
        # plain numpy forward for eval / sampling, no graph needed
        x = self.C.data[X].reshape(len(X), -1)
        h = np.tanh(x @ self.W1.data + self.b1.data)
        return h @ self.W2.data + self.b2.data


def split_loss(model, X, Y):
    logits = model.logits_np(X).astype(np.float64)
    logits -= logits.max(axis=1, keepdims=True)
    logp = logits - np.log(np.exp(logits).sum(axis=1, keepdims=True))
    return -logp[np.arange(len(Y)), Y].mean()


def sample(model, itos, n=20, seed=SEED + 10):
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        context, name = [0] * model.block_size, []
        while True:
            logits = model.logits_np(np.array([context]))[0].astype(np.float64)
            p = np.exp(logits - logits.max())
            ix = rng.choice(len(p), p=p / p.sum())
            if ix == 0:
                break
            name.append(itos[ix])
            context = context[1:] + [ix]
        out.append(''.join(name))
    return out


def plot_loss(losses, path):
    import matplotlib.pyplot as plt

    # raw minibatch loss is noisy, average every 1000 steps
    k = 1000
    m = len(losses) // k * k
    smooth = np.array(losses[:m]).reshape(-1, k).mean(axis=1)
    plt.figure(figsize=(7, 4))
    plt.plot(np.arange(len(smooth)) * k, smooth)
    plt.xlabel('step')
    plt.ylabel('train loss (avg over 1000 steps)')
    plt.title('char MLP on tensorgrad')
    plt.grid(alpha=0.3)
    plt.savefig(path, dpi=120, bbox_inches='tight')
    print(f"saved {path}")


def plot_embeddings(model, itos, path):
    import matplotlib.pyplot as plt

    # project the 10-d embeddings down to 2-d with PCA (via SVD) so we can look at them
    E = model.C.data.astype(np.float64)
    E = E - E.mean(axis=0)
    _, _, Vt = np.linalg.svd(E, full_matrices=False)
    xy = E @ Vt[:2].T

    plt.figure(figsize=(6, 6))
    plt.scatter(xy[:, 0], xy[:, 1], s=200, c='tab:orange', alpha=0.4)
    for i in range(len(xy)):
        plt.text(xy[i, 0], xy[i, 1], itos[i], ha='center', va='center', fontsize=11)
    plt.title('learned char embeddings (PCA to 2d)')
    plt.grid(alpha=0.3)
    plt.savefig(path, dpi=120, bbox_inches='tight')
    print(f"saved {path}")


if __name__ == '__main__':
    words = load_words()
    stoi, itos = build_vocab(words)
    train_w, val_w, test_w = split_words(words)
    Xtr, Ytr = build_dataset(train_w, stoi, BLOCK_SIZE)
    Xva, Yva = build_dataset(val_w, stoi, BLOCK_SIZE)
    Xte, Yte = build_dataset(test_w, stoi, BLOCK_SIZE)
    print(f"train {Xtr.shape}, val {Xva.shape}, test {Xte.shape}")

    model = CharMLP(len(stoi))
    params = model.parameters()
    print("number of parameters", sum(p.data.size for p in params))
    print(f"loss at init {split_loss(model, Xtr, Ytr):.4f} (uniform would be {np.log(len(stoi)):.4f})")

    rng = np.random.default_rng(SEED)
    losses = []
    t0 = time.perf_counter()
    for k in range(STEPS):
        ix = rng.integers(0, len(Xtr), BATCH_SIZE)
        loss = cross_entropy(model(Xtr[ix]), Ytr[ix])

        for p in params:
            p.grad = np.zeros_like(p.data)
        loss.backward()

        lr = 0.1 if k < STEPS // 2 else 0.01  # step decay halfway through
        for p in params:
            p.data -= lr * p.grad

        losses.append(float(loss.data))
        if k % 10_000 == 0 or k == STEPS - 1:
            print(f"step {k:>6} loss {loss.data:.4f}  ({time.perf_counter() - t0:.0f}s)")

    elapsed = time.perf_counter() - t0
    print(f"trained {STEPS} steps in {elapsed:.0f}s ({1e3 * elapsed / STEPS:.2f} ms/step)")

    print(f"\ntrain {split_loss(model, Xtr, Ytr):.4f}")
    print(f"val   {split_loss(model, Xva, Yva):.4f}")
    print(f"test  {split_loss(model, Xte, Yte):.4f}")

    print("\nsamples:")
    for s in sample(model, itos):
        print(" ", s)

    plot_loss(losses, 'mlp_loss.png')
    plot_embeddings(model, itos, 'mlp_embeddings.png')
