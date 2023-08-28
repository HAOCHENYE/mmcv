from functools import wraps
from inspect import signature
from typing import Callable, Tuple

import torch
from mmengine import apply_to


def bf16_compatible(*target_args: Tuple[str]) -> bool:

    def _has_bfloat16(data):
        if isinstance(data, (list, tuple)):
            for i in data:
                if _has_bfloat16(i):
                    return True
            return False
        elif isinstance(data, dict):
            for i in data.values():
                if _has_bfloat16(i):
                    return True
            return False
        elif isinstance(data, torch.Tensor):
            return data.dtype is torch.bfloat16
        else:
            return False

    def wrapper(func):
        ori_arg_names = list(signature(func).parameters)
        if not all([arg_name in ori_arg_names for arg_name in target_args]):
            raise ValueError(
                f'`{target_args}` should be subset of the function arguments!')

        @wraps(func)
        def new_func(*args, **kwargs):
            if not _has_bfloat16(args) and not _has_bfloat16(kwargs):
                return func(*args, **kwargs)
            args = list(args)
            for arg_name in target_args:
                arg_index = list(signature(func).parameters).index(arg_name)
                if arg_name in kwargs:
                    kwargs[arg_name] = apply_to(
                        kwargs[arg_name], lambda x: isinstance(
                            x, torch.Tensor) and x.dtype == torch.bfloat16,
                        lambda x: x.float())
                else:
                    args[arg_index] = apply_to(
                        args[arg_index], lambda x: isinstance(x, torch.Tensor)
                        and x.dtype == torch.bfloat16, lambda x: x.half())
            result = func(*args, **kwargs)
            return apply_to(
                result,
                lambda x: isinstance(x, torch.Tensor) and x.is_floating_point(
                ), lambda x: x.to(dtype=torch.bfloat16))

        return new_func

    return wrapper
