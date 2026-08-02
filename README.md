# Needle — Autograd Framework

CMU 10-714 / 15-714 Deep Learning Systems, Homeworks 1–2.

A minimal automatic differentiation framework built from scratch in NumPy. Needle constructs a **computational graph** of tensor operations, uses **reverse-mode automatic differentiation** to compute gradients, and layers a small deep-learning library (nn modules, data loading, optimizers, parameter initializers) on top.

## Architecture

- **`Tensor`** — central data structure wrapping an NDArray, tracking its position in the computation graph via `op` and `inputs` references
- **`TensorTuple`** — a tuple of tensors as a first-class graph value, used by tuple-producing ops
- **`Op` / `TensorOp`** — defines a computation node with:
  - `compute()` — forward pass (NumPy operations on raw arrays)
  - `gradient()` — backward pass returning gradient w.r.t. each input given the output gradient
- **Autograd engine** — traverses the graph in reverse topological order, sums gradient contributions at each node, and stores the result in each `Tensor.grad`

### Example
```python
import needle as ndl

a = ndl.Tensor([[1., 2.], [3., 4.]])
b = ndl.Tensor([[5., 6.], [7., 8.]])
c = a @ b + a
c.sum().backward()

print(a.grad)  # gradient of loss w.r.t. a
```

## Implemented Ops

| Category | Ops |
|---|---|
| Arithmetic | EWiseAdd, AddScalar, EWiseMul, MulScalar, EWiseDiv, DivScalar |
| Power | EWisePow, PowerScalar |
| Linear algebra | MatMul, Transpose |
| Shape ops | Reshape, BroadcastTo, Summation |
| Activations | ReLU |
| Math | Log, Exp, Negate |
| Log-domain | LogSoftmax, LogSumExp |
| Tensor tuples | MakeTensorTuple, TupleGetItem, FusedAddScalars |

## nn Module

`needle.nn` provides a PyTorch-style `Module` API: `parameters()` collects the module tree's parameters, `_children()` its submodules, and `eval()` / `train()` propagate `training` to the whole subtree.

| Module | Description |
|---|---|
| `Linear` | Fully-connected layer `X @ W + b`; weight and bias Kaiming-uniform initialized |
| `Flatten` | Collapses trailing dimensions to `(batch, -1)` |
| `ReLU` | ReLU activation |
| `Sequential` | Applies modules in order |
| `SoftmaxLoss` | Softmax cross-entropy loss (log-sum-exp formulation) |
| `LayerNorm1d` | Layer normalization over the feature dimension |

In progress: `BatchNorm1d`, `Dropout`, `Residual`.

## Initializers

| Function | Distribution |
|---|---|
| `xavier_uniform` / `xavier_normal` | Uniform / normal with bound or std `gain · √(2/(fan_in + fan_out))` |
| `kaiming_uniform` / `kaiming_normal` | Uniform / normal with bound or std `√(2/fan_in)` (ReLU gain) |

All accept `shape=`, `device=`, `dtype=` kwargs and default to `(fan_in, fan_out)`.

## Data Loading

`needle.data` provides `Dataset` / `DataLoader` (batching, shuffling, transforms) with `MNISTDataset` and `NDArrayDataset` implementations, plus image transforms `RandomFlipHorizontal` and `RandomCrop`.

In progress: `DataLoader.__iter__/__next__`, `RandomFlipHorizontal`, `RandomCrop`, `MNISTDataset`.

## Optimizers

`needle.optim` provides `Optimizer` base with `SGD` and `Adam` classes.

In progress: `SGD.step`, `Adam.step`, `clip_grad_norm`.

## Training Example

```python
import needle as ndl
import needle.nn as nn

model = nn.Sequential(nn.Linear(784, 100), nn.ReLU(), nn.Linear(100, 10))
logits = model(X)                      # X: (batch, 784)
loss = nn.SoftmaxLoss()(logits, y_one_hot)
loss.backward()

# SGD update
for p in model.parameters():
    p.data = (p - lr * p.grad).data
```
