import numpy as np

from tensor import Tensor


class Module:

    def parameters(self):
        return []

    def zero_grad(self):
        for p in self.parameters():
            p.grad = np.zeros_like(p.data)


class Linear(Module):

    def __init__(self, nin, nout, bias=True, dtype=np.float32):
        # kaiming init: keeps activation variance roughly constant through relu layers
        self.w = Tensor(np.random.randn(nin, nout) * np.sqrt(2.0 / nin), dtype=dtype)
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


class Sequential(Module):

    def __init__(self, *layers):
        self.layers = list(layers)

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]

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
