# layers checked against their pytorch equivalents (float64)
# run with: python test_layers.py  (or pytest)

import numpy as np

from tensor import Tensor
from layers import Linear, Tanh, Embedding, Flatten, BatchNorm1d, Sequential

TOL = 1e-8


def test_batchnorm_train_and_eval():
    _check_batchnorm_vs_torch(fused=True)
    _check_batchnorm_vs_torch(fused=False)


def test_fused_batchnorm_matches_composite():
    # the hand-derived backward has to agree with autograd through the primitive ops
    rng = np.random.default_rng(3)
    x0 = rng.standard_normal((64, 10)) * 2 - 1
    w = rng.standard_normal((64, 10))
    outs = []
    for fused in (True, False):
        bn = BatchNorm1d(10, fused=fused, dtype=np.float64)
        bn.gamma.data[:] = np.linspace(0.5, 2, 10)
        x = Tensor(x0.copy())
        (bn(x).tanh() * w).sum().backward()
        outs.append((x.grad, bn.gamma.grad, bn.beta.grad, bn.running_mean, bn.running_var))
    for a, b in zip(*outs):
        assert np.allclose(a, b, atol=TOL)


def _check_batchnorm_vs_torch(fused):
    import torch

    rng = np.random.default_rng(0)
    B, D = 32, 6
    gamma, beta = rng.standard_normal(D), rng.standard_normal(D)

    bn = BatchNorm1d(D, fused=fused, dtype=np.float64)
    bn.gamma.data[:], bn.beta.data[:] = gamma, beta
    tbn = torch.nn.BatchNorm1d(D, dtype=torch.float64)
    with torch.no_grad():
        tbn.weight[:] = torch.tensor(gamma)
        tbn.bias[:] = torch.tensor(beta)

    # a few training steps so the running stats move
    for _ in range(3):
        x0 = rng.standard_normal((B, D)) * 3 + 1
        w = rng.standard_normal((B, D))

        x = Tensor(x0.copy())
        out = bn(x)
        (out * w).sum().backward()

        tx = torch.tensor(x0, requires_grad=True)
        tout = tbn(tx)
        (tout * torch.tensor(w)).sum().backward()

        assert np.allclose(out.data, tout.detach().numpy(), atol=TOL)
        assert np.allclose(x.grad, tx.grad.numpy(), atol=TOL)
        assert np.allclose(bn.gamma.grad, tbn.weight.grad.numpy(), atol=TOL)
        assert np.allclose(bn.beta.grad, tbn.bias.grad.numpy(), atol=TOL)
        bn.zero_grad()
        tbn.zero_grad()

    assert np.allclose(bn.running_mean, tbn.running_mean.numpy(), atol=TOL)
    assert np.allclose(bn.running_var, tbn.running_var.numpy(), atol=TOL)

    # eval mode uses the running stats, works on a single example
    bn.eval()
    tbn.eval()
    x0 = rng.standard_normal((1, D))
    with torch.no_grad():
        assert np.allclose(bn(Tensor(x0)).data, tbn(torch.tensor(x0)).numpy(), atol=TOL)


def test_train_eval_propagates():
    model = Sequential(Linear(4, 3), BatchNorm1d(3), Tanh())
    model.eval()
    assert all(not layer.training for layer in model.layers)
    model.train()
    assert all(layer.training for layer in model.layers)


def test_embedding_flatten_tanh():
    import torch

    rng = np.random.default_rng(1)
    W = rng.standard_normal((27, 10))
    idx = rng.integers(0, 27, size=(8, 3))
    g = rng.standard_normal((8, 30))

    emb = Embedding(27, 10, dtype=np.float64)
    emb.weight.data[:] = W
    out = Tanh()(Flatten()(emb(idx)))
    (out * g).sum().backward()

    tW = torch.tensor(W, requires_grad=True)
    tout = torch.nn.functional.embedding(torch.tensor(idx), tW).view(8, -1).tanh()
    (tout * torch.tensor(g)).sum().backward()

    assert np.allclose(out.data, tout.detach().numpy(), atol=TOL)
    assert np.allclose(emb.weight.grad, tW.grad.numpy(), atol=TOL)


def test_linear_gain():
    np.random.seed(0)
    lin = Linear(1000, 500, gain=5 / 3, dtype=np.float64)
    assert abs(lin.w.data.std() - (5 / 3) / np.sqrt(1000)) < 1e-3


if __name__ == '__main__':
    tests = [v for k, v in list(globals().items()) if k.startswith('test_')]
    for t in tests:
        t()
        print(f"ok  {t.__name__}")
    print(f"all {len(tests)} tests passed")
