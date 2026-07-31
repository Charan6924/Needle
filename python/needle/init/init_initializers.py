import math
from .init_basic import *
from typing import Any
import needle as ndl


def xavier_uniform(fan_in: int, fan_out: int, gain: float = 1.0, **kwargs: Any) -> "Tensor":
    a = gain * math.sqrt(6 / (fan_in + fan_out))
    shape = kwargs.pop("shape", (fan_in, fan_out))
    return rand(*shape, low=-a, high=a, **kwargs)


def xavier_normal(fan_in: int, fan_out: int, gain: float = 1.0, **kwargs: Any) -> "Tensor":
    a =  gain * math.sqrt(2/ (fan_in + fan_out))
    shape = kwargs.pop("shape", (fan_in, fan_out))
    return randn(*shape,std=a,**kwargs)


def kaiming_uniform(fan_in: int, fan_out: int, nonlinearity: str = "relu", **kwargs: Any) -> "Tensor":
    assert nonlinearity == "relu", "Only relu supported currently"
    gain = math.sqrt(2)
    bound = gain * math.sqrt(3/fan_in)
    shape = kwargs.pop("shape", (fan_in, fan_out))
    return rand(*shape, low=-bound, high=bound, **kwargs)



def kaiming_normal(fan_in: int, fan_out: int, nonlinearity: str = "relu", **kwargs: Any) -> "Tensor":
    assert nonlinearity == "relu", "Only relu supported currently"
    gain = math.sqrt(2)
    std = gain/math.sqrt(fan_in)
    shape = kwargs.pop("shape", (fan_in, fan_out))
    return randn(*shape, std=std, **kwargs)
