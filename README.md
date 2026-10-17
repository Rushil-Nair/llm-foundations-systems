# llm-foundations-systems
Implementation of a GPT-style Large Language Model from scratch, focusing on the intersection of Transformer architecture and training efficiency. Built to understand the lower-level mechanics of autograd engines and tensor operations.

## Progress

- [x] **micrograd** - scalar autograd engine + a small MLP trained on two moons (`micrograd/`)
- [x] **tensorgrad** - the same engine rebuilt on numpy arrays, benchmarked against micrograd and pytorch (`tensorgrad/`)
- [x] **makemore** - bigram (counting + neural) and Bengio-style MLP character models, trained on tensorgrad, no pytorch (`makemore/`)
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

## makemore

Character-level language models on the names dataset (32k names, `makemore/names.txt` from karpathy/makemore). Everything trains on tensorgrad. Split is 80/10/10 on whole names.

```
cd makemore
python bigram_counts.py   # count bigrams, sample, saves bigram_counts.png
python bigram_nn.py       # same model learned by gradient descent
python mlp.py             # 3-char context MLP, ~100s on a laptop CPU
```

| model | params | train loss | val loss |
|---|---:|---:|---:|
| uniform guess | 0 | 3.296 | 3.296 |
| bigram, counting (add-one smoothing) | 729 | 2.455 | 2.453 |
| bigram, gradient descent | 729 | 2.463 | 2.461 |
| MLP (context 3, embd 10, hidden 200) | 11,897 | 2.039 | **2.105** |

MLP test loss: 2.105. 200k steps, batch 32, lr 0.1 then 0.01, ~0.5 ms/step on tensorgrad (100-110s total on an i7-9750H).

Samples from the MLP:

```
kaiyan  calani  dime  victan  asaia  samari  navyn  keyla  desia  keilyanleigh
```

![mlp loss](makemore/mlp_loss.png)

![embeddings](makemore/mlp_embeddings.png)

Notes:
- The neural bigram is just a 27x27 table of logits, so it can't do better than counting. Gradient descent lands almost on the counted probabilities (max difference 0.048); the small gap comes from the L2 term, which plays the role of smoothing.
- `onehot(x) @ W` is the same as picking row `x` of `W`, so the bigram and the MLP's embedding table both use indexing instead of a one-hot matmul. The backward for indexing is a scatter-add (`np.add.at`) so repeated characters accumulate their gradients.
- Init matters: tanh layer scaled by `(5/3) / sqrt(fan_in)` and a near-zero output layer. That way the loss at init is 3.30, the same as uniform guessing, instead of starting around 20+, and val loss ends at 2.105 compared to ~2.17 with plain random init in the lecture.
- The embeddings learn structure nobody told them about: in the PCA plot the vowels `a e i o y` cluster together, and rare letters like `q` sit off on their own.
- Two tensorgrad optimizations that didn't work. Profiling the neural bigram (full batch, 182k rows) showed most of each step going to the cross-entropy and the scatter-add in backward.
  - Replacing `np.add.at` with a sort + `np.add.reduceat` scatter was *slower* (77ms vs 54ms per call). `np.add.at` got a fast path in numpy 1.25.
  - Caching the softmax from the forward pass so `exp` only runs once looked like a 24% win on whole training runs, but that was run-to-run noise (the same code ranged from 70s to 112s on this laptop). Timed in isolation with interleaved runs it was 12% *slower*, because the backward needs a copy of the cached array anyway and with only 27 columns `exp` is cheap. Lesson: microbenchmark the op itself, not the whole training loop.
