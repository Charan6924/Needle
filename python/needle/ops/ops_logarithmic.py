from typing import Optional, Any, Union
from ..autograd import NDArray
from ..autograd import Op, Tensor, Value, TensorOp
from ..autograd import TensorTuple, TensorTupleOp
import needle as ndl

from .ops_mathematic import *

import numpy as array_api

class LogSoftmax(TensorOp):
    def compute(self, Z: NDArray) -> NDArray:
        m = array_api.max(Z,axis=-1,keepdims=True)
        m_broadcast = array_api.broadcast_to(m,Z.shape)
        exp_z = array_api.exp(Z - m_broadcast)
        exp_sum = array_api.sum(exp_z, axis=-1,keepdims=True)
        log_sum_exp = array_api.log(exp_sum) + m
        return Z - array_api.broadcast_to(log_sum_exp,Z.shape)

    def gradient(self, out_grad: Tensor, node: Tensor):
        softmax = ndl.exp(node)
        sum_out_grad = ndl.summation(out_grad,axes=(-1,))
        new_shape = list(node.shape)
        new_shape[-1] = 1
        sum_out_grad_broadcast = ndl.broadcast_to(ndl.reshape(sum_out_grad, tuple(new_shape)), node.shape)
        return out_grad - (softmax *sum_out_grad_broadcast) 

def logsoftmax(a: Tensor) -> Tensor:
    return LogSoftmax()(a)


class LogSumExp(TensorOp):
    def __init__(self, axes: Optional[tuple] = None) -> None:
        self.axes = axes

    def compute(self, Z: NDArray) -> NDArray:
        m = array_api.max(Z,axis=self.axes,keepdims=True)
        m_broadcast = array_api.broadcast_to(m,Z.shape)
        exp_z = array_api.exp(Z - m_broadcast)
        exp_sum = array_api.sum(exp_z, axis=self.axes)
        m_reduced = Z.max(axis=self.axes)
        return array_api.log(exp_sum) + m_reduced

    def gradient(self, out_grad: Tensor, node: Tensor):
        Z = node.inputs[0]
        m = Tensor(array_api.max(Z.cached_data, axis=self.axes, keepdims=True),
               device=Z.device, dtype=Z.dtype, requires_grad=False)
        exp_z = ndl.exp(Z - m.broadcast_to(Z.shape))
        sum_exp = ndl.summation(exp_z,axes=self.axes)

        new_shape = list(Z.shape)
        axes = range(len(new_shape)) if self.axes is None else self.axes
        for ax in axes:
            new_shape[ax] = 1
        sum_exp_broadcast = ndl.broadcast_to(ndl.reshape(sum_exp, tuple(new_shape)), Z.shape)
        softmax = exp_z / sum_exp_broadcast

        out_grad_broadcast = ndl.broadcast_to(ndl.reshape(out_grad, tuple(new_shape)), Z.shape)
        return out_grad_broadcast * softmax
        

def logsumexp(a: Tensor, axes: Optional[tuple] = None) -> Tensor:
    return LogSumExp(axes=axes)(a)
