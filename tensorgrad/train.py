# same two moons task as micrograd/train.py, but every op runs on whole arrays

import time

import numpy as np

from tensor import Tensor
from layers import MLP, SGD


def make_moons(n=100, noise=0.1, seed=1337):
    rng = np.random.default_rng(seed)
    t = rng.uniform(0, np.pi, n)
    upper = np.arange(n) % 2 == 0
    x1 = np.where(upper, np.cos(t), 1 - np.cos(t))
    x2 = np.where(upper, np.sin(t), 0.5 - np.sin(t))
    X = np.stack([x1, x2], axis=1) + rng.normal(0, noise, (n, 2))
    y = np.where(upper, -1.0, 1.0)
    return X.astype(np.float32), y.astype(np.float32)


def loss_fn(model, X, y, alpha=1e-4):
    scores = model(Tensor(X)).reshape(-1)

    # svm max-margin loss, the whole batch in one expression
    data_loss = (1 - Tensor(y) * scores).relu().mean()

    # L2 regularization
    reg_loss = alpha * sum((p * p).sum() for p in model.parameters())

    acc = ((y > 0) == (scores.data > 0)).mean()
    return data_loss + reg_loss, acc


def train(n=100, steps=100, hidden=16, seed=1337, verbose=True):
    np.random.seed(seed)
    X, y = make_moons(n=n, seed=seed)
    model = MLP(2, [hidden, hidden, 1])
    opt = SGD(model.parameters(), lr=1.0)

    if verbose:
        print(model)
        print("number of parameters", sum(p.data.size for p in model.parameters()))

    for k in range(steps):
        total_loss, acc = loss_fn(model, X, y)

        opt.zero_grad()
        total_loss.backward()

        opt.lr = 1.0 - 0.9 * k / steps
        opt.step()

        if verbose and (k % 10 == 0 or k == steps - 1):
            print(f"step {k} loss {total_loss.data:.4f}, accuracy {acc * 100:.1f}%")

    return model, acc


if __name__ == '__main__':
    t0 = time.perf_counter()
    train()
    print(f"100 points, 100 steps: {time.perf_counter() - t0:.2f}s")

    # the scalar engine can't realistically do this one
    print()
    t0 = time.perf_counter()
    _, acc = train(n=10_000, steps=100, verbose=False)
    print(f"10,000 points, 100 steps: {time.perf_counter() - t0:.2f}s, final accuracy {acc * 100:.1f}%")
