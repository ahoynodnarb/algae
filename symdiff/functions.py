from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Callable, Optional

    import symdiff.types as sdt

import numpy as np

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


add = register_binary_func(
    eval_func=np.add,
    diff_x=lambda x, y: sd.Constant(1),
    diff_y=lambda x, y: sd.Constant(1),
    repr_func=lambda x, y: f"({x} + {y})",
    func_name="add",
)
subtract = register_binary_func(
    eval_func=np.subtract,
    diff_x=lambda x, y: sd.Constant(1),
    diff_y=lambda x, y: sd.Constant(-1),
    repr_func=lambda x, y: f"({x} - {y})",
    func_name="subtract",
)
multiply = register_binary_func(
    eval_func=np.multiply,
    diff_x=lambda x, y: y,
    diff_y=lambda x, y: x,
    repr_func=lambda x, y: f"{x} * {y}",
    func_name="multiply",
)
divide = register_binary_func(
    eval_func=np.true_divide,
    diff_x=lambda x, y: 1 / y,
    diff_y=lambda x, y: -x / (y**2),
    repr_func=lambda x, y: f"{x} / {y}",
    func_name="divide",
)
pow = register_binary_func(
    eval_func=np.pow,
    diff_x=lambda x, y: y * x ** (y - 1),
    diff_y=lambda x, y: x**y * log(x),
    repr_func=lambda x, y: f"{x}**{y}",
    func_name="pow",
)
sqrt = register_unary_func(
    eval_func=np.sqrt,
    diff_x=lambda x: 1 / (2 * sqrt(x)),
    func_name="sqrt",
)
log = register_unary_func(
    eval_func=np.log,
    diff_x=lambda x: 1 / x,
    func_name="log",
)
exp = register_unary_func(
    eval_func=np.exp,
    diff_x=lambda x: exp(x),
    func_name="exp",
)
sin = register_unary_func(
    eval_func=np.sin,
    diff_x=lambda x: cos(x),
    func_name="sin",
)
cos = register_unary_func(
    eval_func=np.cos,
    diff_x=lambda x: -sin(x),
    func_name="cos",
)
tan = register_unary_func(
    eval_func=np.tan,
    diff_x=lambda x: sec(x) ** 2,
    func_name="tan",
)
csc = register_unary_func(
    eval_func=lambda x: 1 / np.sin(x),
    diff_x=lambda x: -csc(x) * cot(x),
    func_name="csc",
)
sec = register_unary_func(
    eval_func=lambda x: 1 / np.cos(x),
    diff_x=lambda x: sec(x) * tan(x),
    func_name="sec",
)
cot = register_unary_func(
    eval_func=lambda x: 1 / np.tan(x),
    diff_x=lambda x: -(csc(x) ** 2),
    func_name="cot",
)
asin = register_unary_func(
    eval_func=np.asin,
    diff_x=lambda x: 1 / sqrt(1 - x**2),
    func_name="asin",
)
acos = register_unary_func(
    eval_func=np.acos,
    diff_x=lambda x: -1 / sqrt(1 - x**2),
    func_name="acos",
)
atan = register_unary_func(
    eval_func=np.atan,
    diff_x=lambda x: 1 / (1 + x**2),
    func_name="atan",
)
