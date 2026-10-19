# makemore part 3: a deeper tanh MLP, and why init / batchnorm matter.
# trains the same 5-hidden-layer network under 4 setups and records what the
# activations, gradients and updates look like.
# usage: python deep_mlp.py            (all 4 configs, ~10 min on a laptop cpu)
#        python deep_mlp.py --steps 2000   (quick run)

import argparse
import os
import sys
import time

# has to be set before numpy loads. with the default (one thread per core) openblas
# splits even a (32,100)@(100,100) matmul across 12 threads and the sync overhead
# makes it ~17x slower than single threaded (313us vs 18us on an i7-9750H)
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')

import numpy as np

sys.path.insert(1, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'tensorgrad'))

from tensor import cross_entropy
from layers import Linear, Tanh, Embedding, Flatten, BatchNorm1d, Sequential
from data import load_words, build_vocab, split_words, build_dataset

BLOCK_SIZE = 3
N_EMBD = 10
N_HIDDEN = 100
N_LAYERS = 5
BATCH_SIZE = 32
SEED = 2147483647

CONFIGS = {
    # every weight ~ N(0, 1): way too big for fan_in 30-100, tanh saturates
    'naive init':           dict(gain='unit', batchnorm=False),
    # (5/3) / sqrt(fan_in) and a small output layer
    'scaled init':          dict(gain=5 / 3, batchnorm=False),
    'scaled init + bn':     dict(gain=5 / 3, batchnorm=True),
    # batchnorm should rescue the bad init on its own
    'naive init + bn':      dict(gain='unit', batchnorm=True),
}


def build_model(vocab_size, gain, batchnorm, seed=SEED):
    np.random.seed(seed)
    layers = [Embedding(vocab_size, N_EMBD), Flatten()]
    fan_in = BLOCK_SIZE * N_EMBD
    for _ in range(N_LAYERS):
        g = np.sqrt(fan_in) if gain == 'unit' else gain  # 'unit' -> std 1
        # bias is pointless right before batchnorm (beta does the same job)
        layers.append(Linear(fan_in, N_HIDDEN, bias=not batchnorm, gain=g))
        if batchnorm:
            layers.append(BatchNorm1d(N_HIDDEN))
        layers.append(Tanh())
        fan_in = N_HIDDEN
    out = Linear(N_HIDDEN, vocab_size, gain=np.sqrt(N_HIDDEN))
    if gain != 'unit':
        out.w.data *= 0.1  # confident-and-wrong logits at init make the first loss huge
    layers.append(out)
    return Sequential(*layers)


def weight_matrices(model):
    return [layer.w for layer in model.layers if isinstance(layer, Linear)]


def tanh_layers(model):
    return [layer for layer in model.layers if isinstance(layer, Tanh)]


def split_loss(model, X, Y, chunk=20_000):
    # eval mode (batchnorm uses running stats), in chunks to keep memory down
    model.eval()
    total = 0.0
    for i in range(0, len(X), chunk):
        xb, yb = X[i:i + chunk], Y[i:i + chunk]
        total += float(cross_entropy(model(xb), yb).data) * len(xb)
    model.train()
    return total / len(X)


def init_stats(model, Xtr, Ytr, rng):
    # one forward + backward at init, snapshot the tanh activations and their gradients
    ix = rng.integers(0, len(Xtr), 512)
    loss = cross_entropy(model(Xtr[ix]), Ytr[ix])
    model.zero_grad()
    for t in tanh_layers(model):
        t.out.grad = np.zeros_like(t.out.data)
    loss.backward()
    acts = [t.out.data.copy() for t in tanh_layers(model)]
    grads = [t.out.grad.copy() for t in tanh_layers(model)]
    return float(loss.data), acts, grads


def train(name, cfg, data, steps, log_every=100):
    Xtr, Ytr, Xva, Yva = data
    model = build_model(27, **cfg)
    params = model.parameters()
    rng = np.random.default_rng(SEED)

    init_loss, acts, grads = init_stats(model, Xtr, Ytr, rng)
    saturated = [float((np.abs(a) > 0.97).mean()) for a in acts]

    ratios = []  # log10(update std / weight std) per weight matrix, every log_every steps
    t0 = time.perf_counter()
    for k in range(steps):
        ix = rng.integers(0, len(Xtr), BATCH_SIZE)
        loss = cross_entropy(model(Xtr[ix]), Ytr[ix])

        model.zero_grad()
        loss.backward()

        lr = 0.1 if k < steps * 3 // 4 else 0.01
        for p in params:
            p.data -= lr * p.grad

        if k % log_every == 0:
            ratios.append([np.log10((lr * w.grad).std() / w.data.std()) for w in weight_matrices(model)])
        if k % (steps // 5) == 0:
            print(f"  [{name}] step {k:>6} loss {loss.data:.4f}")
    elapsed = time.perf_counter() - t0

    result = dict(
        init_loss=init_loss,
        saturated=saturated,
        acts=acts,
        grads=grads,
        ratios=np.array(ratios),
        train=split_loss(model, Xtr, Ytr),
        val=split_loss(model, Xva, Yva),
        ms_per_step=1e3 * elapsed / steps,
        params=sum(p.data.size for p in params),
    )
    print(f"  [{name}] init loss {init_loss:.3f}, train {result['train']:.4f}, val {result['val']:.4f}, "
          f"{result['ms_per_step']:.2f} ms/step")
    return result


def plot_histograms(results, key, path, title, xlim=None):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(results), figsize=(4.2 * len(results), 3.4), sharey=False)
    for ax, (name, r) in zip(axes, results.items()):
        for i, a in enumerate(r[key]):
            hy, hx = np.histogram(a, bins=50, range=xlim, density=True)
            ax.plot(0.5 * (hx[1:] + hx[:-1]), hy, label=f"layer {i + 1}")
        ax.set_title(name, fontsize=10)
        if xlim:
            ax.set_xlim(*xlim)
    axes[0].legend(fontsize=8)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=110, bbox_inches='tight')
    plt.close(fig)
    print(f"saved {path}")


def plot_ratios(results, path, log_every):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(results), figsize=(4.2 * len(results), 3.4), sharey=True)
    for ax, (name, r) in zip(axes, results.items()):
        steps = np.arange(len(r['ratios'])) * log_every
        for i in range(r['ratios'].shape[1]):
            label = f"layer {i + 1}" if i < r['ratios'].shape[1] - 1 else "output"
            ax.plot(steps, r['ratios'][:, i], label=label, lw=0.8)
        ax.axhline(-3, color='k', ls='--', lw=0.8)  # rough rule of thumb: updates ~1/1000 of the weights
        ax.set_title(name, fontsize=10)
        ax.set_xlabel('step')
    axes[0].set_ylabel('log10(update std / weight std)')
    axes[0].legend(fontsize=8)
    fig.suptitle('update to data ratio during training')
    fig.tight_layout()
    fig.savefig(path, dpi=110, bbox_inches='tight')
    plt.close(fig)
    print(f"saved {path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--steps', type=int, default=50_000)
    args = parser.parse_args()

    words = load_words()
    stoi, itos = build_vocab(words)
    train_w, val_w, _ = split_words(words)
    Xtr, Ytr = build_dataset(train_w, stoi, BLOCK_SIZE)
    Xva, Yva = build_dataset(val_w, stoi, BLOCK_SIZE)
    data = (Xtr, Ytr, Xva, Yva)

    results = {}
    for name, cfg in CONFIGS.items():
        print(f"{name}:")
        results[name] = train(name, cfg, data, args.steps)

    print(f"\n{N_LAYERS} hidden layers of {N_HIDDEN}, {args.steps} steps, batch {BATCH_SIZE}\n")
    print("| setup | params | loss at init | tanh saturated at init (layer 1 / 5) | train | val | ms/step |")
    print("|---|---:|---:|---:|---:|---:|---:|")
    for name, r in results.items():
        s = r['saturated']
        print(f"| {name} | {r['params']:,} | {r['init_loss']:.2f} | {100 * s[0]:.0f}% / {100 * s[-1]:.0f}% | "
              f"{r['train']:.4f} | {r['val']:.4f} | {r['ms_per_step']:.2f} |")

    plot_histograms(results, 'acts', 'deep_mlp_activations.png', 'tanh activations at init', xlim=(-1, 1))
    plot_histograms(results, 'grads', 'deep_mlp_gradients.png', 'gradients of tanh outputs at init')
    plot_ratios(results, 'deep_mlp_update_ratio.png', log_every=100)
