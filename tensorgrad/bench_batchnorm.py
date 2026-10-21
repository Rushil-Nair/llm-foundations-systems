# fused batchnorm (one op, hand-written backward) vs batchnorm built from primitive ops.
# 1) just the op: forward + backward at a few batch sizes
# 2) a full training step of makemore's 5-layer deep MLP with batchnorm
# runs are interleaved and we keep the best time, so background noise on a laptop
# hurts both sides equally
# usage: python bench_batchnorm.py

import os
import sys
import time

os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')

import numpy as np

from tensor import Tensor, cross_entropy
from layers import BatchNorm1d

HERE = os.path.dirname(os.path.abspath(__file__))


def best_time(fn, reps, rounds=5):
    best = float('inf')
    for _ in range(rounds):
        t0 = time.perf_counter()
        for _ in range(reps):
            fn()
        best = min(best, (time.perf_counter() - t0) / reps)
    return best


def op_bench():
    rng = np.random.default_rng(0)
    rows = []
    for B, D in [(32, 100), (256, 100), (4096, 100), (32, 1024)]:
        x0 = rng.standard_normal((B, D)).astype(np.float32)
        g = rng.standard_normal((B, D)).astype(np.float32)
        layers = {fused: BatchNorm1d(D, fused=fused) for fused in (True, False)}

        def run(fused):
            x = Tensor(x0)
            out = layers[fused](x)
            out.backward(g)

        reps = 2000 if B * D <= 30_000 else 200
        times = {}
        for fused in (True, False, True, False):
            t = best_time(lambda: run(fused), reps)
            times[fused] = min(t, times.get(fused, float('inf')))
        rows.append((f"({B}, {D})", 1e6 * times[False], 1e6 * times[True]))
    return rows


def model_bench(steps=1000):
    sys.path.insert(1, os.path.join(HERE, '..', 'makemore'))
    import deep_mlp
    from data import load_words, build_vocab, split_words, build_dataset

    words = load_words()
    stoi, _ = build_vocab(words)
    train_w, _, _ = split_words(words)
    Xtr, Ytr = build_dataset(train_w, stoi, deep_mlp.BLOCK_SIZE)

    def make(fused):
        model = deep_mlp.build_model(27, gain=5 / 3, batchnorm=True)
        for layer in model.layers:
            if isinstance(layer, BatchNorm1d):
                layer.fused = fused
        return model, model.parameters()

    models = {fused: make(fused) for fused in (True, False)}
    rng = np.random.default_rng(0)
    batches = [rng.integers(0, len(Xtr), 32) for _ in range(steps)]

    def train(fused):
        model, params = models[fused]
        for ix in batches:
            loss = cross_entropy(model(Xtr[ix]), Ytr[ix])
            model.zero_grad()
            loss.backward()
            for p in params:
                p.data -= 0.01 * p.grad

    times = {}
    for fused in (True, False, True, False):
        t = best_time(lambda: train(fused), reps=1, rounds=2) / steps
        times[fused] = min(t, times.get(fused, float('inf')))
    return 1e3 * times[False], 1e3 * times[True]


if __name__ == '__main__':
    print("| batchnorm forward + backward (B, D) | composite (us) | fused (us) | speedup |")
    print("|---|---:|---:|---:|")
    for shape, comp, fused in op_bench():
        print(f"| {shape} | {comp:.1f} | {fused:.1f} | {comp / fused:.1f}x |")

    comp, fused = model_bench()
    print()
    print("| deep_mlp training step (5 bn layers, batch 32) | composite (ms) | fused (ms) | speedup |")
    print("|---|---:|---:|---:|")
    print(f"| scaled init + bn | {comp:.2f} | {fused:.2f} | {comp / fused:.1f}x |")
