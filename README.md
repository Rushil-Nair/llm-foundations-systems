# llm-foundations-systems
Implementation of a GPT-style Large Language Model from scratch, focusing on the intersection of Transformer architecture and training efficiency. Built to understand the lower-level mechanics of autograd engines and tensor operations.

## Progress

- [x] **micrograd** - scalar autograd engine + a small MLP trained on two moons (`micrograd/`)

## micrograd

```
cd micrograd
python test_engine.py   # gradients vs finite differences and pytorch
python train.py         # trains a 2-16-16-1 MLP, saves decision_boundary.png
```

![decision boundary](micrograd/decision_boundary.png)
