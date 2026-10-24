# makemore part 5: wavenet-style hierarchical model, trained on tensorgrad.
# with 8 chars of context, a flat MLP squashes all of them into the first layer at once.
# the hierarchical version fuses neighbouring pairs first, then pairs of pairs, then
# pairs of those (8 -> 4 -> 2 -> 1), so information is combined gradually like a tree.
# usage: python wavenet.py                 (3 models, 200k steps each, ~25 min)
#        python wavenet.py --steps 5000    (quick run)

import argparse
import os
import sys
import time

os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')  # small matmuls, see deep_mlp.py

import numpy as np

sys.path.insert(1, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'tensorgrad'))

from tensor import Tensor, cross_entropy
from layers import Linear, Tanh, Embedding, Flatten, FlattenConsecutive, BatchNorm1d, Sequential
from data import load_words, build_vocab, split_words, build_dataset

BLOCK_SIZE = 8
BATCH_SIZE = 32
SEED = 42


def flat_mlp(vocab_size, n_embd=10, n_hidden=200):
    return Sequential(
        Embedding(vocab_size, n_embd), Flatten(),
        Linear(n_embd * BLOCK_SIZE, n_hidden, bias=False, gain=5 / 3), BatchNorm1d(n_hidden), Tanh(),
        Linear(n_hidden, vocab_size),
    )


def wavenet(vocab_size, n_embd, n_hidden):
    # (B, 8, n_embd) -> (B, 4, n_hidden) -> (B, 2, n_hidden) -> (B, n_hidden)
    return Sequential(
        Embedding(vocab_size, n_embd),
        FlattenConsecutive(2), Linear(n_embd * 2, n_hidden, bias=False, gain=5 / 3), BatchNorm1d(n_hidden), Tanh(),
        FlattenConsecutive(2), Linear(n_hidden * 2, n_hidden, bias=False, gain=5 / 3), BatchNorm1d(n_hidden), Tanh(),
        FlattenConsecutive(2), Linear(n_hidden * 2, n_hidden, bias=False, gain=5 / 3), BatchNorm1d(n_hidden), Tanh(),
        Linear(n_hidden, vocab_size),
    )


MODELS = {
    'flat MLP (context 8)': lambda V: flat_mlp(V, n_embd=10, n_hidden=200),
    'wavenet, same size': lambda V: wavenet(V, n_embd=10, n_hidden=68),
    'wavenet, bigger': lambda V: wavenet(V, n_embd=24, n_hidden=128),
}


def build(make, vocab_size):
    np.random.seed(SEED)
    model = make(vocab_size)
    model.layers[-1].w.data *= 0.1  # less confident output layer at init
    return model


def split_loss(model, X, Y, chunk=20_000):
    model.eval()
    total = 0.0
    for i in range(0, len(X), chunk):
        total += float(cross_entropy(model(X[i:i + chunk]), Y[i:i + chunk]).data) * len(X[i:i + chunk])
    model.train()
    return total / len(X)


def sample(model, itos, n=15, seed=SEED):
    rng = np.random.default_rng(seed)
    model.eval()
    out = []
    for _ in range(n):
        context, name = [0] * BLOCK_SIZE, []
        while True:
            logits = model(np.array([context])).data[0].astype(np.float64)
            p = np.exp(logits - logits.max())
            ix = rng.choice(len(p), p=p / p.sum())
            if ix == 0:
                break
            name.append(itos[ix])
            context = context[1:] + [ix]
        out.append(''.join(name))
    model.train()
    return out


def train(name, model, data, steps):
    Xtr, Ytr, Xva, Yva = data
    params = model.parameters()
    rng = np.random.default_rng(SEED)
    losses = []
    t0 = time.perf_counter()
    for k in range(steps):
        ix = rng.integers(0, len(Xtr), BATCH_SIZE)
        loss = cross_entropy(model(Xtr[ix]), Ytr[ix])
        model.zero_grad()
        loss.backward()
        lr = 0.1 if k < steps * 3 // 4 else 0.01
        for p in params:
            p.data -= lr * p.grad
        losses.append(float(loss.data))
        if k % (steps // 5) == 0:
            print(f"  [{name}] step {k:>6} loss {loss.data:.4f}", flush=True)
    elapsed = time.perf_counter() - t0
    return dict(
        params=sum(p.data.size for p in params),
        train=split_loss(model, Xtr, Ytr),
        val=split_loss(model, Xva, Yva),
        ms_per_step=1e3 * elapsed / steps,
        losses=losses,
    )


def plot_losses(results, path):
    import matplotlib.pyplot as plt

    k = 1000
    plt.figure(figsize=(7, 4))
    for name, r in results.items():
        m = len(r['losses']) // k * k
        smooth = np.array(r['losses'][:m]).reshape(-1, k).mean(axis=1)
        plt.plot(np.arange(len(smooth)) * k, smooth, label=f"{name} ({r['params']:,} params)")
    plt.xlabel('step')
    plt.ylabel('train loss (avg over 1000 steps)')
    plt.title('flat vs hierarchical, context of 8 chars')
    plt.grid(alpha=0.3)
    plt.legend()
    plt.savefig(path, dpi=120, bbox_inches='tight')
    print(f"saved {path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--steps', type=int, default=200_000)
    args = parser.parse_args()

    words = load_words()
    stoi, itos = build_vocab(words)
    train_w, val_w, _ = split_words(words)
    Xtr, Ytr = build_dataset(train_w, stoi, BLOCK_SIZE)
    Xva, Yva = build_dataset(val_w, stoi, BLOCK_SIZE)
    data = (Xtr, Ytr, Xva, Yva)

    results, samples = {}, {}
    for name, make in MODELS.items():
        model = build(make, len(stoi))
        print(f"{name}: {model}", flush=True)
        results[name] = train(name, model, data, args.steps)
        samples[name] = sample(model, itos)
        r = results[name]
        print(f"  [{name}] {r['params']:,} params, train {r['train']:.4f}, val {r['val']:.4f}, "
              f"{r['ms_per_step']:.2f} ms/step", flush=True)

    print(f"\ncontext {BLOCK_SIZE}, {args.steps} steps, batch {BATCH_SIZE}\n")
    print("| model | params | train | val | ms/step |")
    print("|---|---:|---:|---:|---:|")
    for name, r in results.items():
        print(f"| {name} | {r['params']:,} | {r['train']:.4f} | {r['val']:.4f} | {r['ms_per_step']:.2f} |")

    for name, s in samples.items():
        print(f"\n{name} samples: {'  '.join(s)}")

    plot_losses(results, 'wavenet_loss.png')
