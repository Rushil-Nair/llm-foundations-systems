import numpy as np


def _unbroadcast(grad, shape):
    # numpy broadcasting stretches inputs to match, so in the backward pass
    # the gradient has to be summed back down to the input's original shape.
    # e.g. bias (16,) added to (100, 16) -> bias grad is the sum over the 100 rows
    while grad.ndim > len(shape):
        grad = grad.sum(axis=0)
    for i, dim in enumerate(shape):
        if dim == 1 and grad.shape[i] != 1:
            grad = grad.sum(axis=i, keepdims=True)
    return grad


def _restore_reduced_dims(grad, axis, ndim, keepdims):
    # put back the axes a reduction removed so the grad broadcasts against the input
    if keepdims:
        return grad
    if axis is None:
        return np.reshape(grad, (1,) * ndim)
    return np.expand_dims(grad, axis)


class Tensor:
    # same idea as micrograd's Value, but data and grad are numpy arrays

    __array_ufunc__ = None  # makes numpy_array + Tensor call Tensor.__radd__ instead of numpy's add

    def __init__(self, data, _children=(), _op='', dtype=None):
        data = np.asarray(data, dtype=dtype)
        if not np.issubdtype(data.dtype, np.floating):
            data = data.astype(np.float32)
        self.data = data
        self.grad = np.zeros_like(data)
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op

    @property
    def shape(self):
        return self.data.shape

    @property
    def ndim(self):
        return self.data.ndim

    def __repr__(self):
        return f"Tensor(shape={self.shape}, dtype={self.data.dtype}, op='{self._op}')"

    def _wrap(self, other):
        return other if isinstance(other, Tensor) else Tensor(np.asarray(other, dtype=self.data.dtype))

    # ---- elementwise ----

    def __add__(self, other):
        other = self._wrap(other)
        out = Tensor(self.data + other.data, (self, other), '+')

        def _backward():
            self.grad += _unbroadcast(out.grad, self.shape)
            other.grad += _unbroadcast(out.grad, other.shape)
        out._backward = _backward
        return out

    def __mul__(self, other):
        other = self._wrap(other)
        out = Tensor(self.data * other.data, (self, other), '*')

        def _backward():
            self.grad += _unbroadcast(other.data * out.grad, self.shape)
            other.grad += _unbroadcast(self.data * out.grad, other.shape)
        out._backward = _backward
        return out

    def __pow__(self, k):
        assert isinstance(k, (int, float)), "only int/float powers"
        out = Tensor(self.data ** k, (self,), f'**{k}')

        def _backward():
            self.grad += k * self.data ** (k - 1) * out.grad
        out._backward = _backward
        return out

    def exp(self):
        out = Tensor(np.exp(self.data), (self,), 'exp')

        def _backward():
            self.grad += out.data * out.grad
        out._backward = _backward
        return out

    def log(self):
        out = Tensor(np.log(self.data), (self,), 'log')

        def _backward():
            self.grad += out.grad / self.data
        out._backward = _backward
        return out

    def tanh(self):
        t = np.tanh(self.data)
        out = Tensor(t, (self,), 'tanh')

        def _backward():
            self.grad += (1 - t ** 2) * out.grad
        out._backward = _backward
        return out

    def relu(self):
        out = Tensor(np.maximum(self.data, 0), (self,), 'relu')

        def _backward():
            self.grad += (self.data > 0) * out.grad
        out._backward = _backward
        return out

    # ---- matmul ----

    def __matmul__(self, other):
        other = self._wrap(other)
        assert self.ndim == 2 and other.ndim == 2, "2D matmul only for now"
        out = Tensor(self.data @ other.data, (self, other), '@')

        def _backward():
            # (n,k) @ (k,m) = (n,m)  ->  dA = dOut @ B.T,  dB = A.T @ dOut
            self.grad += out.grad @ other.data.T
            other.grad += self.data.T @ out.grad
        out._backward = _backward
        return out

    # ---- reductions ----

    def sum(self, axis=None, keepdims=False):
        out = Tensor(self.data.sum(axis=axis, keepdims=keepdims), (self,), 'sum')

        def _backward():
            g = _restore_reduced_dims(out.grad, axis, self.ndim, keepdims)
            self.grad += np.broadcast_to(g, self.shape)
        out._backward = _backward
        return out

    def mean(self, axis=None, keepdims=False):
        out = self.sum(axis=axis, keepdims=keepdims)
        return out * (out.data.size / self.data.size)

    def max(self, axis=None, keepdims=False):
        m = self.data.max(axis=axis, keepdims=True)
        out = Tensor(m if keepdims else self.data.max(axis=axis), (self,), 'max')

        def _backward():
            # gradient only goes to the max element(s), split evenly on ties
            mask = (self.data == m).astype(self.data.dtype)
            mask /= mask.sum(axis=axis, keepdims=True)
            g = _restore_reduced_dims(out.grad, axis, self.ndim, keepdims)
            self.grad += mask * g
        out._backward = _backward
        return out

    # ---- shape ops ----

    def reshape(self, *shape):
        out = Tensor(self.data.reshape(*shape), (self,), 'reshape')

        def _backward():
            self.grad += out.grad.reshape(self.shape)
        out._backward = _backward
        return out

    def transpose(self, *axes):
        axes = axes or None
        out = Tensor(np.transpose(self.data, axes), (self,), 'transpose')

        def _backward():
            inv = None if axes is None else np.argsort(axes)
            self.grad += np.transpose(out.grad, inv)
        out._backward = _backward
        return out

    @property
    def T(self):
        return self.transpose()

    def __getitem__(self, idx):
        if isinstance(idx, Tensor):
            idx = idx.data.astype(np.int64)
        out = Tensor(self.data[idx], (self,), 'getitem')

        def _backward():
            # add.at so repeated indices (same embedding row picked twice) accumulate.
            # tried a sort + np.add.reduceat version, it was slower than add.at on numpy 1.26
            np.add.at(self.grad, idx, out.grad)
        out._backward = _backward
        return out

    # ---- backprop ----

    def backward(self, grad=None):
        # same iterative topological sort as micrograd
        topo = []
        visited = set()
        stack = [(self, False)]
        while stack:
            v, children_done = stack.pop()
            if children_done:
                topo.append(v)
                continue
            if v in visited:
                continue
            visited.add(v)
            stack.append((v, True))
            for child in v._prev:
                if child not in visited:
                    stack.append((child, False))

        self.grad = np.ones_like(self.data) if grad is None else np.asarray(grad, dtype=self.data.dtype)
        for v in reversed(topo):
            v._backward()

    # ---- built from the ops above ----

    def __neg__(self):
        return self * -1

    def __sub__(self, other):
        return self + (-self._wrap(other))

    def __truediv__(self, other):
        return self * self._wrap(other) ** -1

    def __radd__(self, other):
        return self + other

    def __rmul__(self, other):
        return self * other

    def __rsub__(self, other):
        return self._wrap(other) + (-self)

    def __rtruediv__(self, other):
        return self._wrap(other) * self ** -1


def softmax(x, axis=-1):
    # composed from primitive ops, subtracting the max for numerical stability
    e = (x - x.max(axis=axis, keepdims=True)).exp()
    return e / e.sum(axis=axis, keepdims=True)


def cross_entropy(logits, targets):
    # fused log-softmax + negative log likelihood as a single op.
    # doing it in one node means we never build the softmax graph, and the
    # backward collapses to (softmax - onehot) / n
    x = logits.data
    t = np.asarray(targets.data if isinstance(targets, Tensor) else targets).astype(np.int64)
    n = x.shape[0]
    rows = np.arange(n)

    shifted = x - x.max(axis=1, keepdims=True)
    logp = shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))
    out = Tensor(np.asarray(-logp[rows, t].mean(), dtype=x.dtype), (logits,), 'cross_entropy')

    def _backward():
        # recomputing exp here measured the same or faster than caching the
        # softmax from the forward pass (caching needs a full copy anyway)
        dlogits = np.exp(logp)
        dlogits[rows, t] -= 1
        logits.grad += dlogits / n * out.grad
    out._backward = _backward
    return out
