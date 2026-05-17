from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Callable, Optional

    import symdiff.types as sdt

import symdiff as sd


def register_unary_func(
    eval_func: sdt.UnaryFunction,
    diff_x: Callable[[sd.Expression], sd.Expression],
    repr_func: Optional[sdt.UnaryFunction] = None,
    func_name: Optional[str] = None,
) -> sdt.UnaryFunction:
    if func_name is None:
        func_name = "sd_anonymous_unary"

    if repr_func is None:
        repr_func = lambda x: f"{func_name}({x})"

    return sd.SymFunc(
        eval_func=eval_func,
        diff_funcs=[diff_x],
        repr_func=repr_func,
        func_name=func_name,
    )


def register_binary_func(
    eval_func: sdt.BinaryFunction,
    diff_x: Callable[[sd.Expression, sd.Expression], sd.Expression],
    diff_y: Callable[[sd.Expression, sd.Expression], sd.Expression],
    repr_func: Optional[sdt.BinaryFunction] = None,
    func_name: Optional[str] = None,
) -> sdt.BinaryFunction:
    if func_name is None:
        func_name = "sd_anonymous_binary"

    if repr_func is None:
        repr_func = lambda x, y: f"{func_name}({x}, {y})"

    return sd.SymFunc(
        eval_func=eval_func,
        diff_funcs=[diff_x, diff_y],
        repr_func=repr_func,
        func_name=func_name,
    )
