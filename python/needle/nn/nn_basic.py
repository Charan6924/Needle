"""The module.
"""
from numpy import array_api
from typing import Any
from needle.autograd import Tensor
from needle import ops
import needle.init as init
import numpy as np
import needle as ndl


class Parameter(Tensor):
    """A special kind of tensor that represents parameters."""


def _unpack_params(value: object) -> list[Tensor]:
    if isinstance(value, Parameter):
        return [value]
    elif isinstance(value, Module):
        return value.parameters()
    elif isinstance(value, dict):
        params = []
        for k, v in value.items():
            params += _unpack_params(v)
        return params
    elif isinstance(value, (list, tuple)):
        params = []
        for v in value:
            params += _unpack_params(v)
        return params
    else:
        return []


def _child_modules(value: object) -> list["Module"]:
    if isinstance(value, Module):
        modules = [value]
        modules.extend(_child_modules(value.__dict__))
        return modules
    if isinstance(value, dict):
        modules = []
        for k, v in value.items():
            modules += _child_modules(v)
        return modules
    elif isinstance(value, (list, tuple)):
        modules = []
        for v in value:
            modules += _child_modules(v)
        return modules
    else:
        return []


class Module:
    def __init__(self) -> None:
        self.training = True

    def parameters(self) -> list[Tensor]:
        """Return the list of parameters in the module."""
        return _unpack_params(self.__dict__)

    def _children(self) -> list["Module"]:
        return _child_modules(self.__dict__)

    def eval(self) -> None:
        self.training = False
        for m in self._children():
            m.training = False

    def train(self) -> None:
        self.training = True
        for m in self._children():
            m.training = True

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class Identity(Module):
    def forward(self, x: Tensor) -> Tensor:
        return x


class Linear(Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True, device: Any | None = None, dtype: str = "float32") -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(init.kaiming_uniform(fan_in=in_features,fan_out=out_features,shape=(in_features,out_features),device=device,dtype=dtype))
        if bias:
            self.bias = Parameter(init.kaiming_uniform(fan_in=out_features,fan_out=1,shape = (1,out_features,), device=device,dtype=dtype))
        else:
            self.bias = None

    def forward(self, X: Tensor) -> Tensor:
        out = ndl.matmul(X,self.weight)
        if self.bias is not None:
            out = out + self.bias.broadcast_to(out.shape)
        return out


class Flatten(Module):
    def forward(self, X: Tensor) -> Tensor:
        # multiplies all non batch dims together and returns (B,X1*X2*X3....)
        b = X.shape[0]
        flat_dim = 1
        for dim in X.shape[1:]:
            flat_dim *= dim
        return ndl.reshape(X,(b,flat_dim))
        


class ReLU(Module):
    def forward(self, x: Tensor) -> Tensor:
        return ops.relu(x)

class Sequential(Module):
    def __init__(self, *modules: Module) -> None:
        super().__init__()
        self.modules = modules

    def forward(self, x: Tensor) -> Tensor:
        for module in self.modules:
            x = module(x)
        return x


class SoftmaxLoss(Module):
    def forward(self, logits: Tensor, y: Tensor) -> Tensor:
        ### uses init.one_hot and logsumexp implementation
        y_one_hot = init.one_hot(n=logits.shape[-1],i=y,device=logits.device,dtype=logits.dtype)
        z_y = ndl.summation(logits * y_one_hot,axes=(-1,))
        lse = ndl.logsumexp(logits,axes=(-1,))
        losses = lse - z_y
        return ndl.summation(losses)/logits.shape[0]



class BatchNorm1d(Module):
    def __init__(self, dim: int, eps: float = 1e-5, momentum: float = 0.1, device: Any | None = None, dtype: str = "float32") -> None:
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.momentum = momentum
        self.weight = Parameter(init.ones(self.dim, device=device, dtype=dtype))
        self.bias = Parameter(init.zeros(self.dim, device=device, dtype=dtype))
        self.running_mean = init.zeros(self.dim,device=device,dtype=dtype)
        self.running_var = init.ones(self.dim,device=device,dtype=dtype)

    def forward(self, x: Tensor) -> Tensor:
        if self.training:
            mean = ndl.summation(x,axes=(0,))/x.shape[0]
            mean_broadcast = ndl.broadcast_to(ndl.reshape(mean, (1,self.dim)), x.shape)
            var = ndl.summation((x - mean_broadcast) ** 2, axes=(0,)) / x.shape[0]
            var_broadcast = ndl.broadcast_to(ndl.reshape(var, (1,self.dim)), x.shape)

            self.running_mean = ((1 - self.momentum) * self.running_mean + self.momentum * mean).detach()
            self.running_var = ((1 - self.momentum) * self.running_var + self.momentum * var).detach()

            y = (x - mean_broadcast)/ndl.power_scalar(var_broadcast+self.eps,0.5)
            w_broadcast = ndl.broadcast_to(ndl.reshape(self.weight, (1, self.dim)), x.shape)
            b_broadcast = ndl.broadcast_to(ndl.reshape(self.bias, (1, self.dim)), x.shape)
            return (w_broadcast * y) + b_broadcast
        else:
            mean_broadcast = ndl.broadcast_to(ndl.reshape(self.running_mean, (1,self.dim)), x.shape)
            var_broadcast = ndl.broadcast_to(ndl.reshape(self.running_var, (1,self.dim)), x.shape)

            y = (x - mean_broadcast)/ndl.power_scalar(var_broadcast+self.eps,0.5)
            w_broadcast = ndl.broadcast_to(ndl.reshape(self.weight, (1, self.dim)), x.shape)
            b_broadcast = ndl.broadcast_to(ndl.reshape(self.bias, (1, self.dim)), x.shape)
            return (w_broadcast * y) + b_broadcast

class LayerNorm1d(Module):
    def __init__(self, dim: int, eps: float = 1e-5, device: Any | None = None, dtype: str = "float32") -> None:
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = Parameter(init.ones(self.dim, device=device, dtype=dtype))
        self.bias = Parameter(init.zeros(self.dim, device=device, dtype=dtype))

    def forward(self, x: Tensor) -> Tensor:
        mu = ndl.summation(x,axes=(-1,))/self.dim
        mu_broadcast = ndl.broadcast_to(ndl.reshape(mu, (x.shape[0], 1)), x.shape)
        var = ndl.summation((x - mu_broadcast) ** 2, axes=(1,)) / self.dim
        var_broadcast = ndl.broadcast_to(ndl.reshape(var, (x.shape[0], 1)), x.shape)
        x_norm = (x - mu_broadcast)/ ndl.power_scalar(var_broadcast + self.eps,0.5)
        w_broadcast = ndl.broadcast_to(ndl.reshape(self.weight, (1, self.dim)), x.shape)
        b_broadcast = ndl.broadcast_to(ndl.reshape(self.bias, (1, self.dim)), x.shape)
        return w_broadcast * x_norm + b_broadcast

class Dropout(Module):
    def __init__(self, p: float = 0.5) -> None:
        super().__init__()
        self.p = p

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION


class Residual(Module):
    def __init__(self, fn: Module) -> None:
        super().__init__()
        self.fn = fn

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION
