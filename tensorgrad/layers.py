import numpy as np

from tensor import Tensor, batchnorm


class Module:

    training = True

    def parameters(self):
        return []

    def children(self):
        return []

    def zero_grad(self):
        for p in self.parameters():
            p.grad = np.zeros_like(p.data)

    def train(self, mode=True):
        # only matters for layers that act differently at eval time (batchnorm)
        self.training = mode
        for c in self.children():
            c.train(mode)
        return self

    def eval(self):
        return self.train(False)


class Linear(Module):

    def __init__(self, nin, nout, bias=True, gain=None, dtype=np.float32):
        # default is kaiming init for relu, std = sqrt(2 / fan_in).
        # for tanh pass gain=5/3 -> std = (5/3) / sqrt(fan_in)
        scale = np.sqrt(2.0 / nin) if gain is None else gain / np.sqrt(nin)
        self.w = Tensor(np.random.randn(nin, nout) * scale, dtype=dtype)
        self.b = Tensor(np.zeros(nout), dtype=dtype) if bias else None

    def __call__(self, x):
        out = x @ self.w
        return out + self.b if self.b is not None else out

    def parameters(self):
        return [self.w] + ([self.b] if self.b is not None else [])

    def __repr__(self):
        return f"Linear({self.w.shape[0]}, {self.w.shape[1]})"


class ReLU(Module):

    def __call__(self, x):
        return x.relu()

    def __repr__(self):
        return "ReLU()"


class Tanh(Module):

    def __call__(self, x):
        self.out = x.tanh()  # kept so we can look at saturation after a forward pass
        return self.out

    def __repr__(self):
        return "Tanh()"


class Embedding(Module):

    def __init__(self, num_embeddings, dim, dtype=np.float32):
        self.weight = Tensor(np.random.randn(num_embeddings, dim), dtype=dtype)

    def __call__(self, idx):
        return self.weight[idx]

    def parameters(self):
        return [self.weight]

    def __repr__(self):
        return f"Embedding({self.weight.shape[0]}, {self.weight.shape[1]})"


class Flatten(Module):

    def __call__(self, x):
        return x.reshape(x.shape[0], -1)

    def __repr__(self):
        return "Flatten()"


class BatchNorm1d(Module):
    # normalize each feature to zero mean / unit variance over the batch, then
    # let the network rescale it with gamma and beta.
    # at eval time there might be a single example, so use running averages instead

    def __init__(self, dim, eps=1e-5, momentum=0.1, fused=True, dtype=np.float32):
        self.eps = eps
        self.momentum = momentum
        self.fused = fused  # False -> build it from primitive ops (slower, same math)
        self.gamma = Tensor(np.ones(dim), dtype=dtype)
        self.beta = Tensor(np.zeros(dim), dtype=dtype)
        # buffers, not parameters: updated by the forward pass, not by gradients
        self.running_mean = np.zeros(dim, dtype=dtype)
        self.running_var = np.ones(dim, dtype=dtype)

    def _update_running(self, mean, var, n):
        m = self.momentum
        self.running_mean = (1 - m) * self.running_mean + m * mean
        # pytorch keeps the unbiased variance in the running estimate
        self.running_var = (1 - m) * self.running_var + m * var * n / (n - 1)

    def __call__(self, x):
        if self.training and self.fused:
            out, mean, var = batchnorm(x, self.gamma, self.beta, self.eps)
            self._update_running(mean, var, x.shape[0])
            return out
        if self.training:
            mean = x.mean(axis=0, keepdims=True)
            var = ((x - mean) ** 2).mean(axis=0, keepdims=True)  # biased, same as pytorch uses to normalize
            self._update_running(mean.data[0], var.data[0], x.shape[0])
        else:
            mean = Tensor(self.running_mean)
            var = Tensor(self.running_var)
        xhat = (x - mean) * (var + self.eps) ** -0.5
        return xhat * self.gamma + self.beta

    def parameters(self):
        return [self.gamma, self.beta]

    def __repr__(self):
        return f"BatchNorm1d({self.gamma.shape[0]})"


class Sequential(Module):

    def __init__(self, *layers):
        self.layers = list(layers)

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]

    def children(self):
        return self.layers

    def __repr__(self):
        return "Sequential(" + ", ".join(str(layer) for layer in self.layers) + ")"


def MLP(nin, nouts):
    # same shape as micrograd's MLP: relu everywhere except the last layer
    sizes = [nin] + nouts
    layers = []
    for i in range(len(nouts)):
        layers.append(Linear(sizes[i], sizes[i + 1]))
        if i != len(nouts) - 1:
            layers.append(ReLU())
    return Sequential(*layers)


class SGD:

    def __init__(self, params, lr=0.1):
        self.params = list(params)
        self.lr = lr

    def zero_grad(self):
        for p in self.params:
            p.grad = np.zeros_like(p.data)

    def step(self):
        for p in self.params:
            p.data -= self.lr * p.grad
