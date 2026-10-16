# time one training step (forward + backward + sgd update) of the same
# 2-16-16-1 MLP on two moons, for: scalar micrograd, tensorgrad (numpy), pytorch (cpu)
# usage: python bench_vs_scalar.py

import os
import platform
import statistics
import sys
import time

import numpy as np

# tensorgrad's own folder stays first on the path, micrograd goes after it
sys.path.insert(1, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'micrograd'))

from engine import Value                # micrograd
from nn import MLP as ScalarMLP        # micrograd
from layers import MLP as TensorMLP    # tensorgrad
from train import make_moons, loss_fn  # tensorgrad

SCALAR_SIZES = [16, 32, 64, 128, 256]
TENSOR_SIZES = [16, 32, 64, 128, 256, 1024, 4096, 16384, 65536]
LR = 0.1


def time_step(step_fn, repeats, warmup=3):
    for _ in range(warmup):
        step_fn()
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        step_fn()
        times.append(time.perf_counter() - t0)
    return statistics.median(times) * 1000  # ms


def scalar_bench(n):
    X, y = make_moons(n=n)
    X, y = X.tolist(), y.tolist()
    model = ScalarMLP(2, [16, 16, 1])

    def step():
        scores = [model([Value(a), Value(b)]) for a, b in X]
        losses = [(1 + -yi * si).relu() for yi, si in zip(y, scores)]
        loss = sum(losses) * (1.0 / len(losses)) + 1e-4 * sum(p * p for p in model.parameters())
        model.zero_grad()
        loss.backward()
        for p in model.parameters():
            p.data -= LR * p.grad

    return time_step(step, repeats=3 if n >= 128 else 5, warmup=1)


def tensor_bench(n):
    X, y = make_moons(n=n)
    model = TensorMLP(2, [16, 16, 1])
    params = model.parameters()

    def step():
        loss, _ = loss_fn(model, X, y)
        model.zero_grad()
        loss.backward()
        for p in params:
            p.data -= LR * p.grad

    return time_step(step, repeats=20)


def torch_bench(n):
    import torch

    X, y = make_moons(n=n)
    X, y = torch.from_numpy(X), torch.from_numpy(y)
    model = torch.nn.Sequential(
        torch.nn.Linear(2, 16), torch.nn.ReLU(),
        torch.nn.Linear(16, 16), torch.nn.ReLU(),
        torch.nn.Linear(16, 1),
    )
    opt = torch.optim.SGD(model.parameters(), lr=LR)

    def step():
        scores = model(X).reshape(-1)
        loss = (1 - y * scores).relu().mean() + 1e-4 * sum((p * p).sum() for p in model.parameters())
        opt.zero_grad()
        loss.backward()
        opt.step()

    return time_step(step, repeats=20)


def fmt(ms):
    return "-" if ms is None else (f"{ms:.3f}" if ms < 1 else f"{ms:.1f}")


if __name__ == '__main__':
    try:
        import torch
        have_torch = True
    except ImportError:
        have_torch = False

    rows = []
    for n in TENSOR_SIZES:
        s = scalar_bench(n) if n in SCALAR_SIZES else None
        t = tensor_bench(n)
        p = torch_bench(n) if have_torch else None
        rows.append((n, s, t, p))
        print(f"n={n:>6}  scalar {fmt(s):>8} ms   tensorgrad {fmt(t):>8} ms   torch {fmt(p):>8} ms", flush=True)

    print()
    print(f"cpu: {platform.processor()}, numpy {np.__version__}" + (f", torch {torch.__version__}" if have_torch else ""))
    print()
    print("| batch size | micrograd (ms/step) | tensorgrad (ms/step) | pytorch cpu (ms/step) | tensorgrad vs micrograd |")
    print("|---:|---:|---:|---:|---:|")
    for n, s, t, p in rows:
        speedup = f"{s / t:,.0f}x" if s else "-"
        print(f"| {n:,} | {fmt(s)} | {fmt(t)} | {fmt(p)} | {speedup} |")

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        sys.exit(0)

    plt.figure(figsize=(7, 4.5))
    ns = [r[0] for r in rows]
    plt.plot([r[0] for r in rows if r[1]], [r[1] for r in rows if r[1]], 'o-', label='micrograd (scalar, python)')
    plt.plot(ns, [r[2] for r in rows], 's-', label='tensorgrad (numpy)')
    if have_torch:
        plt.plot(ns, [r[3] for r in rows], '^-', label='pytorch (cpu)')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('batch size (points)')
    plt.ylabel('ms per training step')
    plt.title('2-16-16-1 MLP, one forward + backward + sgd step')
    plt.grid(True, which='both', alpha=0.3)
    plt.legend()
    plt.savefig('benchmark.png', dpi=120, bbox_inches='tight')
    print("\nsaved benchmark.png")
