# check the engine's gradients against finite differences and against pytorch
# run with: python test_engine.py  (or pytest)

import math

from engine import Value


def numerical_grad(f, xs, h=1e-6):
    # central difference for each input, same idea as derivatives.py
    grads = []
    for i in range(len(xs)):
        up = list(xs)
        down = list(xs)
        up[i] += h
        down[i] -= h
        grads.append((f(*up) - f(*down)) / (2 * h))
    return grads


def expr(a, b, c):
    # works on both floats and Values, so we can reuse it for the numerical check
    d = a * b + c
    e = (d ** 2) / (a + 4.0)
    f = e - 3.0 * b
    g = (f * 0.1).tanh() if isinstance(f, Value) else math.tanh(f * 0.1)
    return g * a + (b ** 3) / 2.0


def test_against_finite_differences():
    xs = [2.0, -3.0, 10.0]
    vals = [Value(x) for x in xs]
    out = expr(*vals)
    out.backward()

    num = numerical_grad(expr, xs)
    for v, n in zip(vals, num):
        assert abs(v.grad - n) < 1e-4, (v.grad, n)


def test_against_torch():
    try:
        import torch
    except ImportError:
        print("torch not installed, skipping")
        return

    a = Value(-4.0)
    b = Value(2.0)
    c = a + b
    d = a * b + b ** 3
    c = c + c + 1
    c = c + 1 + c + (-a)
    d = d + d * 2 + (b + a).relu()
    d = d + 3 * d + (b - a).relu()
    e = c - d
    f = e ** 2
    g = f / 2.0
    g = g + 10.0 / f
    h = (g * 0.01).exp().log() + (a * 0.5).tanh()
    h.backward()

    at = torch.tensor([-4.0], dtype=torch.double, requires_grad=True)
    bt = torch.tensor([2.0], dtype=torch.double, requires_grad=True)
    ct = at + bt
    dt = at * bt + bt ** 3
    ct = ct + ct + 1
    ct = ct + 1 + ct + (-at)
    dt = dt + dt * 2 + (bt + at).relu()
    dt = dt + 3 * dt + (bt - at).relu()
    et = ct - dt
    ft = et ** 2
    gt = ft / 2.0
    gt = gt + 10.0 / ft
    ht = (gt * 0.01).exp().log() + (at * 0.5).tanh()
    ht.backward()

    tol = 1e-6
    assert abs(h.data - ht.data.item()) < tol
    assert abs(a.grad - at.grad.item()) < tol
    assert abs(b.grad - bt.grad.item()) < tol


def test_reused_node():
    # a node used twice must accumulate both gradient contributions
    a = Value(3.0)
    b = a + a
    b.backward()
    assert a.grad == 2.0


if __name__ == '__main__':
    test_against_finite_differences()
    test_against_torch()
    test_reused_node()
    print("all tests passed")
