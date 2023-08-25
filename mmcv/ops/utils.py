from functools import wraps
from inspect import signature
from typing import Callable, Tuple

import torch
from mmengine import apply_to


def bf16_compatible(*target_args: Tuple[str]) -> Callable:

    def wrapper(func):
        ori_arg_names = list(signature(func).parameters)
        if not all([arg_name in ori_arg_names for arg_name in target_args]):
            raise ValueError(
                f'`{target_args}` should be subset of the function arguments!')

        @wraps(func)
        def new_func(*args, **kwargs):
            args = list(args)
            for arg_name in target_args:
                arg_index = list(signature(func).parameters).index(arg_name)
                args[arg_index] = apply_to(
                    args[arg_index], lambda x: isinstance(x, torch.Tensor) and
                    x.dtype == torch.bfloat16, lambda x: x.half())
            result = func(*args, **kwargs)
            return apply_to(
                result,
                lambda x: isinstance(x, torch.Tensor) and x.is_floating_point(
                ), lambda x: x.to(dtype=torch.bfloat16))

        return new_func

    return wrapper
