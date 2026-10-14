# train a small MLP on the two moons dataset using only the Value engine

import math
import random

from engine import Value
from nn import MLP


def make_moons(n=100, noise=0.1):
    # two interleaving half circles, labels are -1 / +1
    X, y = [], []
    for i in range(n):
        t = random.uniform(0, math.pi)
        if i % 2 == 0:
            x1, x2, label = math.cos(t), math.sin(t), -1
        else:
            x1, x2, label = 1 - math.cos(t), 0.5 - math.sin(t), 1
        X.append([x1 + random.gauss(0, noise), x2 + random.gauss(0, noise)])
        y.append(label)
    return X, y


def loss_fn(model, X, y, alpha=1e-4):
    scores = [model([Value(a), Value(b)]) for a, b in X]

    # svm max-margin loss
    losses = [(1 + -yi * si).relu() for yi, si in zip(y, scores)]
    data_loss = sum(losses) * (1.0 / len(losses))

    # L2 regularization
    reg_loss = alpha * sum(p * p for p in model.parameters())

    acc = sum((yi > 0) == (si.data > 0) for yi, si in zip(y, scores)) / len(y)
    return data_loss + reg_loss, acc


def plot_decision_boundary(model, X, y, path='decision_boundary.png'):
    import numpy as np
    import matplotlib.pyplot as plt

    X = np.array(X)
    h = 0.1
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    grid = np.c_[xx.ravel(), yy.ravel()]
    Z = np.array([model([Value(a), Value(b)]).data > 0 for a, b in grid]).reshape(xx.shape)

    plt.figure(figsize=(6, 5))
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral, alpha=0.8)
    plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.Spectral, edgecolors='k')
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.title('micrograd MLP on two moons')
    plt.savefig(path, dpi=120, bbox_inches='tight')
    print(f"saved {path}")


if __name__ == '__main__':
    random.seed(1337)
    X, y = make_moons(n=100, noise=0.1)

    model = MLP(2, [16, 16, 1])
    print(model)
    print("number of parameters", len(model.parameters()))

    steps = 100
    for k in range(steps):
        total_loss, acc = loss_fn(model, X, y)

        model.zero_grad()
        total_loss.backward()

        # sgd with a linearly decaying learning rate
        lr = 1.0 - 0.9 * k / steps
        for p in model.parameters():
            p.data -= lr * p.grad

        if k % 10 == 0 or k == steps - 1:
            print(f"step {k} loss {total_loss.data:.4f}, accuracy {acc * 100:.1f}%")

    plot_decision_boundary(model, X, y)
