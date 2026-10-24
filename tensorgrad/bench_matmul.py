# (B, T, k) @ (k, m): numpy's broadcasting matmul vs flattening to one 2d matmul.
# these are the shapes the wavenet model in makemore/wavenet.py uses.
# usage: python bench_matmul.py

import os

os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')

import time

import numpy as np

# (leading dims, k, m)
SHAPES = [
    ((32, 4), 48, 128),     # wavenet layer 1: 4 pairs of 24-d char embeddings
    ((32, 2), 256, 128),    # wavenet layer 2
    ((32, 4), 20, 68),      # small wavenet layer 1
    ((256, 4), 48, 128),    # bigger batch
    ((32, 64), 64, 64),     # longer sequence, like a small transformer block
]


def best(fn, reps, rounds=5):
    b = float('inf')
    for _ in range(rounds):
        t0 = time.perf_counter()
        for _ in range(reps):
            fn()
        b = min(b, (time.perf_counter() - t0) / reps)
    return 1e6 * b


if __name__ == '__main__':
    rng = np.random.default_rng(0)
    print(f"numpy {np.__version__}, OPENBLAS_NUM_THREADS={os.environ['OPENBLAS_NUM_THREADS']}\n")
    print("| A @ W | broadcasting matmul (us) | reshape to 2d (us) | speedup |")
    print("|---|---:|---:|---:|")
    for lead, k, m in SHAPES:
        a = rng.standard_normal((*lead, k)).astype(np.float32)
        w = rng.standard_normal((k, m)).astype(np.float32)

        def broadcast():
            return a @ w

        def flat():
            return (a.reshape(-1, k) @ w).reshape(*lead, m)

        assert np.allclose(broadcast(), flat(), atol=1e-3)
        reps = 2000 if np.prod(lead) * k * m < 5e6 else 200
        # interleaved so background noise hits both
        t = {'b': float('inf'), 'f': float('inf')}
        for _ in range(2):
            t['b'] = min(t['b'], best(broadcast, reps))
            t['f'] = min(t['f'], best(flat, reps))
        shape = f"({', '.join(map(str, lead))}, {k}) @ ({k}, {m})"
        print(f"| {shape} | {t['b']:.1f} | {t['f']:.1f} | {t['b'] / t['f']:.1f}x |")
