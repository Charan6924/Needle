# Needle — Autograd Framework

CMU 10-714 / 15-714 Deep Learning Systems, Homework 1.

A minimal automatic differentiation framework built from scratch in NumPy. Needle constructs a **computational graph** of tensor operations and uses **reverse-mode automatic differentiation** to compute gradients.

## Architecture

- **`Tensor`** — central data structure wrapping an NDArray, tracking its position in the computation graph via `op` and `inputs` references
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

## Training Example

```python
# Two-layer neural network with ReLU activation
logits = ndl.relu(X @ W1) @ W2
# Softmax cross-entropy loss
loss = softmax_loss(logits, y_one_hot)
loss.backward()

# SGD update
W1.data = (W1 - lr * W1.grad).data
W2.data = (W2 - lr * W2.grad).data
```
