# every op checked against pytorch (float64), plus a finite difference check
# run with: python test_tensor.py  (or pytest)

import numpy as np

from tensor import Tensor, softmax, cross_entropy

rng = np.random.default_rng(0)
TOL = 1e-8


def compare(f_ours, f_torch, *shapes, positive=False):
    import torch

    arrays = [rng.standard_normal(s) for s in shapes]
    if positive:
        arrays = [np.abs(a) + 0.5 for a in arrays]

    ours = [Tensor(a.copy()) for a in arrays]
    out = f_ours(*ours)
    # random weights before summing so every output element gets a different upstream grad
    w = rng.standard_normal(out.shape)
    (out * w).sum().backward()

    theirs = [torch.tensor(a, requires_grad=True) for a in arrays]
    tout = f_torch(*theirs)
    (tout * torch.tensor(w)).sum().backward()

    assert out.shape == tuple(tout.shape), (out.shape, tout.shape)
    assert np.allclose(out.data, tout.detach().numpy(), atol=TOL)
    for o, t in zip(ours, theirs):
        assert np.allclose(o.grad, t.grad.numpy(), atol=TOL)


def same(f):
    return f, f


def test_elementwise():
    compare(*same(lambda a, b: a + b), (4, 3), (4, 3))
    compare(*same(lambda a, b: a * b - a / b), (4, 3), (4, 3), positive=True)
    compare(*same(lambda a: a ** 3 + 2 * a - 1), (5,))
    compare(*same(lambda a: a.exp() + a.tanh() + a.relu()), (3, 4))
    compare(*same(lambda a: a.log()), (3, 4), positive=True)
    compare(*same(lambda a: 1 - a + 2 / a), (6,), positive=True)


def test_broadcasting():
    compare(*same(lambda a, b: a + b), (4, 3), (3,))       # bias add
    compare(*same(lambda a, b: a * b), (4, 1), (1, 3))     # outer product style
    compare(*same(lambda a, b: a - b), (2, 4, 3), (4, 1))
    compare(*same(lambda a, b: a / b), (5, 3), (1,), positive=True)


def test_matmul():
    compare(*same(lambda a, b: a @ b), (5, 3), (3, 4))
    compare(*same(lambda a, b, c: (a @ b + c).relu()), (8, 3), (3, 6), (6,))
    # extra leading dims on the left: same weights applied at every position
    compare(*same(lambda a, b: a @ b), (4, 3, 5), (5, 2))
    compare(*same(lambda a, b, c: (a @ b + c).tanh()), (2, 3, 4, 5), (5, 6), (6,))


def test_reductions():
    compare(*same(lambda a: a.sum()), (4, 3))
    compare(lambda a: a.sum(axis=1), lambda a: a.sum(dim=1), (4, 3))
    compare(lambda a: a.sum(axis=0, keepdims=True), lambda a: a.sum(dim=0, keepdim=True), (4, 3))
    compare(lambda a: a.mean(axis=1), lambda a: a.mean(dim=1), (4, 3))
    compare(*same(lambda a: a.mean()), (2, 3, 4))
    compare(lambda a: a.max(axis=1), lambda a: a.max(dim=1).values, (4, 5))
    compare(lambda a: a.max(axis=0, keepdims=True), lambda a: a.max(dim=0, keepdim=True).values, (4, 5))


def test_shape_ops():
    compare(*same(lambda a: a.reshape(6, 2)), (3, 4))
    compare(*same(lambda a: a.T), (3, 4))
    compare(lambda a: a.transpose(2, 0, 1), lambda a: a.permute(2, 0, 1), (2, 3, 4))
    compare(*same(lambda a: a[1:, :2]), (4, 3))


def test_indexing_repeated_rows():
    # embedding lookup: the same row picked several times must accumulate grad
    idx = np.array([0, 2, 2, 1, 2])
    compare(*same(lambda emb: emb[idx]), (4, 3))


def test_indexing_2d_and_negative():
    # (batch, block_size) context -> (batch, block_size, n_embd), like makemore's C[X]
    idx = rng.integers(0, 27, size=(32, 3))
    compare(*same(lambda emb: emb[idx]), (27, 10))
    neg = np.array([-1, 3, 26, -27, 0, -1])
    compare(*same(lambda emb: emb[neg]), (27, 4))
    compare(*same(lambda v: v[np.array(5)]), (8,))


def test_softmax_and_cross_entropy():
    import torch
    import torch.nn.functional as F

    compare(lambda a: softmax(a, axis=1), lambda a: F.softmax(a, dim=1), (6, 5))

    targets = rng.integers(0, 5, size=6)
    compare(lambda a: cross_entropy(a, targets),
            lambda a: F.cross_entropy(a, torch.tensor(targets)), (6, 5))

    # fused version should match the one built from primitive ops
    logits = rng.standard_normal((6, 5))
    a = Tensor(logits.copy())
    cross_entropy(a, targets).backward()
    b = Tensor(logits.copy())
    p = softmax(b, axis=1)
    (-(p[np.arange(6), targets].log()).mean()).backward()
    assert np.allclose(a.grad, b.grad, atol=TOL)


def test_finite_differences():
    # no torch needed: compare backward against (f(x+h) - f(x-h)) / 2h
    x0 = rng.standard_normal((3, 4))
    w0 = rng.standard_normal((4, 2))

    def f(x, w):
        return ((x @ w).tanh() ** 2).sum(axis=0).mean()

    x, w = Tensor(x0.copy()), Tensor(w0.copy())
    f(x, w).backward()

    h = 1e-6
    for param, base in [(x, x0), (w, w0)]:
        num = np.zeros_like(base)
        for i in np.ndindex(base.shape):
            up, down = base.copy(), base.copy()
            up[i] += h
            down[i] -= h
            if param is x:
                num[i] = (f(Tensor(up), Tensor(w0)).data - f(Tensor(down), Tensor(w0)).data) / (2 * h)
            else:
                num[i] = (f(Tensor(x0), Tensor(up)).data - f(Tensor(x0), Tensor(down)).data) / (2 * h)
        assert np.allclose(param.grad, num, atol=1e-6)


def test_reused_tensor():
    a = Tensor(np.array([1.0, 2.0]))
    (a * a + a).sum().backward()
    assert np.allclose(a.grad, 2 * a.data + 1)


def test_numpy_on_left():
    a = Tensor(np.array([1.0, 2.0]))
    out = np.array([3.0, 4.0]) * a
    assert isinstance(out, Tensor)
    out.sum().backward()
    assert np.allclose(a.grad, [3.0, 4.0])


if __name__ == '__main__':
    tests = [v for k, v in list(globals().items()) if k.startswith('test_')]
    for t in tests:
        t()
        print(f"ok  {t.__name__}")
    print(f"all {len(tests)} tests passed")
