# makemore part 4: backprop by hand.
# the MLP from mlp.py plus a batchnorm layer, forward pass broken into tiny steps.
# every gradient is derived manually in numpy and checked against tensorgrad's autograd,
# then the net is trained using only the manual backward pass.
# usage: python backprop_ninja.py [--steps 100000]

import argparse
import os
import sys
import time

os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')  # small matmuls, see deep_mlp.py

import numpy as np

sys.path.insert(1, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'tensorgrad'))

from tensor import Tensor
from data import load_words, build_vocab, split_words, build_dataset

BLOCK_SIZE = 3
N_EMBD = 10
N_HIDDEN = 64
BATCH_SIZE = 32
EPS = 1e-5
SEED = 2147483647


def init_params(vocab_size, dtype, seed=SEED):
    # small random values everywhere (even biases), so a wrong gradient
    # can't hide behind something that happens to be zero
    rng = np.random.default_rng(seed)
    fan_in = BLOCK_SIZE * N_EMBD
    p = dict(
        C=rng.standard_normal((vocab_size, N_EMBD)),
        W1=rng.standard_normal((fan_in, N_HIDDEN)) * (5 / 3) / fan_in ** 0.5,
        b1=rng.standard_normal(N_HIDDEN) * 0.1,
        W2=rng.standard_normal((N_HIDDEN, vocab_size)) * 0.1,
        b2=rng.standard_normal(vocab_size) * 0.1,
        bngain=1 + rng.standard_normal(N_HIDDEN) * 0.1,
        bnbias=rng.standard_normal(N_HIDDEN) * 0.1,
    )
    return {k: v.astype(dtype) for k, v in p.items()}


def forward_tensorgrad(P, Xb, Yb):
    # every intermediate is kept so we can compare its grad with the manual one
    n = len(Xb)
    T = {k: Tensor(v) for k, v in P.items()}
    s = {}
    s['emb'] = T['C'][Xb]                                       # (n, 3, 10)
    s['embcat'] = s['emb'].reshape(n, -1)                       # (n, 30)
    s['hprebn'] = s['embcat'] @ T['W1'] + T['b1']               # (n, 64)
    # batchnorm, one op at a time
    s['bnmeani'] = s['hprebn'].mean(axis=0, keepdims=True)
    s['bndiff'] = s['hprebn'] - s['bnmeani']
    s['bndiff2'] = s['bndiff'] ** 2
    s['bnvar'] = s['bndiff2'].mean(axis=0, keepdims=True)       # biased, like pytorch
    s['bnvar_inv'] = (s['bnvar'] + EPS) ** -0.5
    s['bnraw'] = s['bndiff'] * s['bnvar_inv']
    s['hpreact'] = T['bngain'] * s['bnraw'] + T['bnbias']
    s['h'] = s['hpreact'].tanh()
    s['logits'] = s['h'] @ T['W2'] + T['b2']                    # (n, 27)
    # cross entropy, one op at a time
    s['logit_maxes'] = s['logits'].max(axis=1, keepdims=True)
    s['norm_logits'] = s['logits'] - s['logit_maxes']
    s['counts'] = s['norm_logits'].exp()
    s['counts_sum'] = s['counts'].sum(axis=1, keepdims=True)
    s['counts_sum_inv'] = s['counts_sum'] ** -1
    s['probs'] = s['counts'] * s['counts_sum_inv']
    s['logprobs'] = s['probs'].log()
    loss = -s['logprobs'][np.arange(n), Yb].mean()
    return loss, T, s


def manual_backward(P, s, Xb, Yb):
    # s holds the forward values (numpy arrays). walk backwards one line at a time
    n = len(Xb)
    d = {}
    d['logprobs'] = np.zeros_like(s['logprobs'])
    d['logprobs'][np.arange(n), Yb] = -1.0 / n                  # loss = -mean of the picked entries
    d['probs'] = d['logprobs'] / s['probs']                      # d log(p) = 1/p
    d['counts_sum_inv'] = (d['probs'] * s['counts']).sum(axis=1, keepdims=True)  # broadcast in fwd -> sum in bwd
    d['counts'] = d['probs'] * s['counts_sum_inv']
    d['counts_sum'] = -s['counts_sum'] ** -2 * d['counts_sum_inv']
    d['counts'] += d['counts_sum']                               # sum in fwd -> broadcast in bwd
    d['norm_logits'] = s['counts'] * d['counts']                 # d exp(x) = exp(x)
    d['logits'] = d['norm_logits'].copy()
    d['logit_maxes'] = -d['norm_logits'].sum(axis=1, keepdims=True)
    # the max only shifts every logit in a row, which doesn't change softmax,
    # so this grad should be ~0. it still has to be routed to the argmax position
    onehot = np.zeros_like(s['logits'])
    onehot[np.arange(n), s['logits'].argmax(axis=1)] = 1
    d['logits'] += onehot * d['logit_maxes']
    d['h'] = d['logits'] @ P['W2'].T
    d['W2'] = s['h'].T @ d['logits']
    d['b2'] = d['logits'].sum(axis=0)
    d['hpreact'] = (1 - s['h'] ** 2) * d['h']
    d['bngain'] = (s['bnraw'] * d['hpreact']).sum(axis=0)
    d['bnbias'] = d['hpreact'].sum(axis=0)
    d['bnraw'] = P['bngain'] * d['hpreact']
    d['bnvar_inv'] = (s['bndiff'] * d['bnraw']).sum(axis=0, keepdims=True)
    d['bndiff'] = s['bnvar_inv'] * d['bnraw']
    d['bnvar'] = -0.5 * (s['bnvar'] + EPS) ** -1.5 * d['bnvar_inv']
    d['bndiff2'] = np.ones_like(s['bndiff2']) / n * d['bnvar']
    d['bndiff'] += 2 * s['bndiff'] * d['bndiff2']
    d['hprebn'] = d['bndiff'].copy()
    d['bnmeani'] = -d['bndiff'].sum(axis=0, keepdims=True)
    d['hprebn'] += np.ones_like(s['hprebn']) / n * d['bnmeani']
    d['embcat'] = d['hprebn'] @ P['W1'].T
    d['W1'] = s['embcat'].T @ d['hprebn']
    d['b1'] = d['hprebn'].sum(axis=0)
    d['emb'] = d['embcat'].reshape(s['emb'].shape)
    d['C'] = np.zeros_like(P['C'])
    np.add.at(d['C'], Xb, d['emb'])                              # rows picked several times accumulate
    return d


def dlogits_one_shot(logits, Yb):
    # whole cross entropy backward in one line: (softmax - onehot) / n
    n = len(Yb)
    e = np.exp(logits - logits.max(axis=1, keepdims=True))
    dlogits = e / e.sum(axis=1, keepdims=True)
    dlogits[np.arange(n), Yb] -= 1
    return dlogits / n


def dhprebn_one_shot(hprebn, bngain, dhpreact):
    # whole batchnorm backward in one line. the 3 terms: the direct path through
    # (x - mean), the path through the mean, and the path through the variance
    n = hprebn.shape[0]
    xhat = (hprebn - hprebn.mean(axis=0)) / np.sqrt(hprebn.var(axis=0) + EPS)
    invstd = 1.0 / np.sqrt(hprebn.var(axis=0) + EPS)
    dxhat = bngain * dhpreact
    return invstd / n * (n * dxhat - dxhat.sum(axis=0) - xhat * (dxhat * xhat).sum(axis=0))


def cmp(name, manual, auto):
    exact = np.array_equal(manual, auto)
    approx = np.allclose(manual, auto, atol=1e-10, rtol=0)
    maxdiff = np.abs(manual - auto).max()
    print(f"{name:<15} | exact: {str(exact):<5} | approx: {str(approx):<5} | maxdiff: {maxdiff:.2e}")
    return approx


def check_gradients(Xtr, Ytr, vocab_size):
    # float64 so the comparison can be tight
    P = init_params(vocab_size, np.float64)
    rng = np.random.default_rng(SEED)
    ix = rng.integers(0, len(Xtr), BATCH_SIZE)
    Xb, Yb = Xtr[ix], Ytr[ix]

    loss, T, s = forward_tensorgrad(P, Xb, Yb)
    loss.backward()
    vals = {k: v.data for k, v in s.items()}
    d = manual_backward(P, vals, Xb, Yb)

    print(f"loss {loss.data:.6f}\n")
    print("exercise 1: every intermediate, step by step")
    ok = True
    for name in ['logprobs', 'probs', 'counts_sum_inv', 'counts_sum', 'counts', 'norm_logits',
                 'logit_maxes', 'logits', 'h', 'W2', 'b2', 'hpreact', 'bngain', 'bnbias', 'bnraw',
                 'bnvar_inv', 'bnvar', 'bndiff2', 'bndiff', 'bnmeani', 'hprebn', 'embcat', 'W1',
                 'b1', 'emb', 'C']:
        auto = T[name].grad if name in T else s[name].grad
        ok &= cmp(name, d[name], auto)

    print("\nexercise 2: cross entropy backward in one step")
    ok &= cmp('logits', dlogits_one_shot(vals['logits'], Yb), s['logits'].grad)

    print("\nexercise 3: batchnorm backward in one step")
    ok &= cmp('hprebn', dhprebn_one_shot(vals['hprebn'], P['bngain'], s['hpreact'].grad), s['hprebn'].grad)
    print(f"\nlogit_maxes grad is ~0 as expected: max |d| = {np.abs(d['logit_maxes']).max():.1e}")
    assert ok, "some manual gradient doesn't match"
    return ok


def forward_numpy(P, Xb, bn_mean=None, bn_var=None):
    n = len(Xb)
    embcat = P['C'][Xb].reshape(n, -1)
    hprebn = embcat @ P['W1'] + P['b1']
    if bn_mean is None:
        bn_mean, bn_var = hprebn.mean(axis=0), hprebn.var(axis=0)
    hpreact = P['bngain'] * (hprebn - bn_mean) / np.sqrt(bn_var + EPS) + P['bnbias']
    h = np.tanh(hpreact)
    logits = h @ P['W2'] + P['b2']
    return embcat, hprebn, hpreact, h, logits


def loss_np(logits, Y):
    logits = logits.astype(np.float64)
    logits = logits - logits.max(axis=1, keepdims=True)
    logp = logits - np.log(np.exp(logits).sum(axis=1, keepdims=True))
    return -logp[np.arange(len(Y)), Y].mean()


def train_manual(Xtr, Ytr, Xva, Yva, vocab_size, steps):
    # exercise 4: no autograd at all, only the one-shot backward formulas
    P = init_params(vocab_size, np.float32)
    rng = np.random.default_rng(SEED)
    run_mean = np.zeros(N_HIDDEN, np.float32)
    run_var = np.ones(N_HIDDEN, np.float32)
    t0 = time.perf_counter()
    for k in range(steps):
        ix = rng.integers(0, len(Xtr), BATCH_SIZE)
        Xb, Yb = Xtr[ix], Ytr[ix]
        n = len(Xb)

        embcat, hprebn, hpreact, h, logits = forward_numpy(P, Xb)
        m, v = hprebn.mean(axis=0), hprebn.var(axis=0)
        run_mean = 0.999 * run_mean + 0.001 * m
        run_var = 0.999 * run_var + 0.001 * v * n / (n - 1)

        dlogits = dlogits_one_shot(logits, Yb)
        dh = dlogits @ P['W2'].T
        dhpreact = (1 - h ** 2) * dh
        dhprebn = dhprebn_one_shot(hprebn, P['bngain'], dhpreact)
        dembcat = dhprebn @ P['W1'].T
        G = dict(
            W2=h.T @ dlogits, b2=dlogits.sum(axis=0),
            bngain=(((hprebn - m) / np.sqrt(v + EPS)) * dhpreact).sum(axis=0), bnbias=dhpreact.sum(axis=0),
            W1=embcat.T @ dhprebn, b1=dhprebn.sum(axis=0),
            C=np.zeros_like(P['C']),
        )
        np.add.at(G['C'], Xb, dembcat.reshape(n, BLOCK_SIZE, N_EMBD))

        lr = 0.1 if k < steps // 2 else 0.01
        for name in P:
            P[name] -= lr * G[name]
        if k % (steps // 5) == 0:
            print(f"  step {k:>6} loss {loss_np(logits, Yb):.4f}")
    elapsed = time.perf_counter() - t0

    def split(X, Y):
        return loss_np(forward_numpy(P, X, run_mean, run_var)[-1], Y)
    return split(Xtr, Ytr), split(Xva, Yva), 1e3 * elapsed / steps


def train_autograd(Xtr, Ytr, Xva, Yva, vocab_size, steps):
    # the same model and init through tensorgrad + its fused batchnorm, for a speed comparison
    from layers import Linear, Embedding, Flatten, BatchNorm1d, Tanh, Sequential
    from tensor import cross_entropy

    P = init_params(vocab_size, np.float32)
    model = Sequential(Embedding(vocab_size, N_EMBD), Flatten(), Linear(BLOCK_SIZE * N_EMBD, N_HIDDEN),
                       BatchNorm1d(N_HIDDEN, momentum=0.001), Tanh(), Linear(N_HIDDEN, vocab_size))
    emb, _, l1, bn, _, l2 = model.layers
    emb.weight.data, l1.w.data, l1.b.data = P['C'], P['W1'], P['b1']
    bn.gamma.data, bn.beta.data = P['bngain'], P['bnbias']
    l2.w.data, l2.b.data = P['W2'], P['b2']
    params = model.parameters()

    rng = np.random.default_rng(SEED)
    t0 = time.perf_counter()
    for k in range(steps):
        ix = rng.integers(0, len(Xtr), BATCH_SIZE)
        loss = cross_entropy(model(Xtr[ix]), Ytr[ix])
        model.zero_grad()
        loss.backward()
        lr = 0.1 if k < steps // 2 else 0.01
        for p in params:
            p.data -= lr * p.grad
    elapsed = time.perf_counter() - t0

    model.eval()
    def split(X, Y):
        return float(cross_entropy(model(X), Y).data)
    return split(Xtr, Ytr), split(Xva, Yva), 1e3 * elapsed / steps


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--steps', type=int, default=100_000)
    args = parser.parse_args()

    words = load_words()
    stoi, itos = build_vocab(words)
    train_w, val_w, _ = split_words(words)
    Xtr, Ytr = build_dataset(train_w, stoi, BLOCK_SIZE)
    Xva, Yva = build_dataset(val_w, stoi, BLOCK_SIZE)
    V = len(stoi)

    check_gradients(Xtr, Ytr, V)

    print(f"\nexercise 4: train {args.steps} steps with the manual backward only")
    m_train, m_val, m_ms = train_manual(Xtr, Ytr, Xva, Yva, V, args.steps)
    print(f"  manual:   train {m_train:.4f}  val {m_val:.4f}  {m_ms:.3f} ms/step")

    print(f"\nsame model through tensorgrad autograd (fused batchnorm)")
    a_train, a_val, a_ms = train_autograd(Xtr, Ytr, Xva, Yva, V, args.steps)
    print(f"  autograd: train {a_train:.4f}  val {a_val:.4f}  {a_ms:.3f} ms/step")
    print(f"\nmanual backward is {a_ms / m_ms:.1f}x faster per step")
