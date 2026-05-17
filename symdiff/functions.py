from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Optional

    import symdiff.types as sdt

import numpy as np

import symdiff as A


def create_unary_func(
    eval_func: sdt.UnaryFunction,
    diff_x: sdt.UnaryFunction,
    repr_func: Optional[sdt.UnaryFunction] = None,
    func_name: Optional[str] = None,
) -> sdt.UnaryFunction:
    if func_name is None:
        func_name = "sd_anonymous"

    if repr_func is None:
        repr_func = lambda x: f"{func_name}({x})"

    def generated(x: A.Expression) -> A.Expression:
        return A.Expression(
            x,
            eval_func=eval_func,
            repr_func=repr_func,
            expression_name=func_name,
        )

    return A.SymFunc(generated, [diff_x])


def create_binary_func(
    eval_func: sdt.BinaryFunction,
    diff_x: sdt.BinaryFunction,
    diff_y: sdt.BinaryFunction,
    repr_func: Optional[sdt.BinaryFunction] = None,
    func_name: Optional[str] = None,
) -> sdt.BinaryFunction:
    if func_name is None:
        func_name = "symdiff_anonymous_binary_function"

    if repr_func is None:
        repr_func = lambda x, y: f"{func_name}({x, y})"

    def generated(x: A.Expression, y: A.Expression) -> A.Expression:
        return A.Expression(
            x,
            y,
            eval_func=eval_func,
            repr_func=repr_func,
            expression_name=func_name,
        )

    return A.SymFunc(generated, [diff_x, diff_y])


add = create_binary_func(
    eval_func=np.add,
    repr_func=lambda x, y: f"({x} + {y})",
    func_name="add",
)
subtract = create_binary_func(
    eval_func=np.subtract,
    repr_func=lambda x, y: f"({x} - {y})",
    func_name="add",
)
multiply = create_binary_func(
    eval_func=np.multiply,
    repr_func=lambda x, y: f"{x} * {y}",
    func_name="add",
)
divide = create_binary_func(
    eval_func=np.true_divide,
    repr_func=lambda x, y: f"{x} / {y}",
    func_name="add",
)
pow = create_binary_func(
    eval_func=np.pow,
    repr_func=lambda x, y: f"{x}**{y}",
    func_name="pow",
)
sqrt = create_unary_func(
    eval_func=np.sqrt,
    func_name="sqrt",
)
cbrt = create_unary_func(
    eval_func=np.cbrt,
    func_name="cbrt",
)
log = create_unary_func(
    eval_func=np.log,
    func_name="log",
)
exp = create_unary_func(
    eval_func=np.exp,
    func_name="exp",
)
sin = create_unary_func(
    eval_func=np.sin,
    func_name="sin",
)
cos = create_unary_func(
    eval_func=np.cos,
    func_name="cos",
)
tan = create_unary_func(
    eval_func=np.tan,
    func_name="tan",
)
csc = create_unary_func(
    eval_func=lambda x: 1 / np.sin(x),
    func_name="csc",
)
sec = create_unary_func(
    eval_func=lambda x: 1 / np.cos(x),
    func_name="sec",
)
cot = create_unary_func(
    eval_func=lambda x: 1 / np.tan(x),
    func_name="cot",
)
asin = create_unary_func(
    eval_func=np.asin,
    func_name="asin",
)
acos = create_unary_func(
    eval_func=np.acos,
    func_name="acos",
)
atan = create_unary_func(
    eval_func=np.atan,
    func_name="atan",
)
