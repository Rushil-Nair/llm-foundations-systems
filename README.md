# llm-foundations-systems
Implementation of a GPT-style Large Language Model from scratch, focusing on the intersection of Transformer architecture and training efficiency. Built to understand the lower-level mechanics of autograd engines and tensor operations.

## Progress

- [x] **micrograd** - scalar autograd engine + a small MLP trained on two moons (`micrograd/`)
- [x] **tensorgrad** - the same engine rebuilt on numpy arrays, benchmarked against micrograd and pytorch (`tensorgrad/`)
- [ ] triton kernels for the hot ops (matmul, softmax, fused cross-entropy)

## micrograd

```
cd micrograd
python test_engine.py   # gradients vs finite differences and pytorch
python train.py         # trains a 2-16-16-1 MLP, saves decision_boundary.png
```

![decision boundary](micrograd/decision_boundary.png)

## tensorgrad

micrograd tracks every single number as its own python object, so one training step on 256 points builds a graph of ~160k nodes. tensorgrad keeps the same design (each op stores a `_backward`, topological sort, chain rule) but every node is a whole numpy array.

What's in it:
- ops: `+ - * / **`, `exp log tanh relu`, `@`, `sum mean max` (with `axis`/`keepdims`), `reshape transpose`, indexing (for embedding lookups later)
- broadcasting-aware backward: gradients get summed back down to the input's shape (`_unbroadcast`)
- `cross_entropy` as a single fused op: backward is just `(softmax - onehot) / n` instead of going through the softmax graph
- `Linear`, `ReLU`, `Sequential`, `MLP`, `SGD`
- every op tested against pytorch in float64 plus a finite difference check

```
cd tensorgrad
python test_tensor.py       # all ops vs pytorch
python train.py             # two moons, same setup as micrograd
python bench_vs_scalar.py   # micrograd vs tensorgrad vs pytorch
```

### Benchmark

Same 2-16-16-1 MLP, one training step (forward + backward + SGD update), median of several runs. Laptop CPU (i7-9750H), numpy 1.26, torch 2.6.

| batch size | micrograd (ms/step) | tensorgrad (ms/step) | pytorch cpu (ms/step) | tensorgrad vs micrograd |
|---:|---:|---:|---:|---:|
| 16 | 175.7 | 0.689 | 1.1 | 255x |
| 32 | 352.3 | 0.590 | 0.911 | 598x |
| 64 | 832.5 | 0.900 | 1.4 | 925x |
| 128 | 1343.7 | 0.792 | 1.7 | 1,696x |
| 256 | 3178.5 | 1.3 | 2.4 | 2,422x |
| 1,024 | - | 3.9 | 2.7 | - |
| 4,096 | - | 8.7 | 4.1 | - |
| 16,384 | - | 25.2 | 5.8 | - |
| 65,536 | - | 70.7 | 15.6 | - |

![benchmark](tensorgrad/benchmark.png)

Takeaways:
- micrograd's cost grows linearly with batch size since every element is a python object. tensorgrad's time barely moves until ~256 points because at that size it's dominated by fixed per-op overhead, not math.
- tensorgrad is actually *faster* than pytorch at small batches. pytorch pays more per op (dispatcher, autograd bookkeeping), and at tiny sizes that overhead is the whole cost.
- past ~1k points pytorch pulls ahead (~4.5x at 65k). numpy runs each elementwise op single-threaded and writes a full intermediate array to memory for every op (`1 - y*s`, `relu`, `mean` are 3 separate passes), while pytorch uses multiple threads and better kernels. that memory traffic between ops is exactly what kernel fusion removes, which is the next step with triton.
