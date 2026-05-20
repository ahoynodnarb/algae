from __future__ import annotations

import math

import algae.algebra as A
from algae.registry import register_binary_func, register_unary_func

add = register_binary_func(
    eval_func=lambda x, y: x + y,
    diff_x=lambda x, y: A.Constant(1),
    diff_y=lambda x, y: A.Constant(1),
    repr_func=lambda x, y: f"({x} + {y})",
    func_name="add",
    commutes=True,
    associates=True,
)
subtract = register_binary_func(
    eval_func=lambda x, y: x - y,
    diff_x=lambda x, y: A.Constant(1),
    diff_y=lambda x, y: A.Constant(-1),
    repr_func=lambda x, y: f"({x} - {y})",
    func_name="subtract",
    associates=True,
)
multiply = register_binary_func(
    eval_func=lambda x, y: x * y,
    diff_x=lambda x, y: y,
    diff_y=lambda x, y: x,
    repr_func=lambda x, y: f"{x} * {y}",
    func_name="multiply",
    commutes=True,
    associates=True,
)
divide = register_binary_func(
    eval_func=lambda x, y: x / y,
    diff_x=lambda x, y: 1 / y,
    diff_y=lambda x, y: -x / (y**2),
    repr_func=lambda x, y: f"{x} / {y}",
    func_name="divide",
    associates=True,
)
pow = register_binary_func(
    eval_func=lambda x, y: x**y,
    diff_x=lambda x, y: y * x ** (y - 1),
    diff_y=lambda x, y: x**y * log(x),
    repr_func=lambda x, y: f"{x}**{y}",
    func_name="pow",
)
sqrt = register_unary_func(
    eval_func=math.sqrt,
    diff_x=lambda x: 1 / (2 * sqrt(x)),
    func_name="sqrt",
)
log = register_unary_func(
    eval_func=math.log,
    diff_x=lambda x: 1 / x,
    func_name="log",
)
exp = register_unary_func(
    eval_func=math.exp,
    diff_x=lambda x: exp(x),
    func_name="exp",
)
sin = register_unary_func(
    eval_func=math.sin,
    diff_x=lambda x: cos(x),
    func_name="sin",
)
cos = register_unary_func(
    eval_func=math.cos,
    diff_x=lambda x: -sin(x),
    func_name="cos",
)
tan = register_unary_func(
    eval_func=math.tan,
    diff_x=lambda x: sec(x) ** 2,
    func_name="tan",
)
csc = register_unary_func(
    eval_func=lambda x: 1 / math.sin(x),
    diff_x=lambda x: -csc(x) * cot(x),
    func_name="csc",
)
sec = register_unary_func(
    eval_func=lambda x: 1 / math.cos(x),
    diff_x=lambda x: sec(x) * tan(x),
    func_name="sec",
)
cot = register_unary_func(
    eval_func=lambda x: 1 / math.tan(x),
    diff_x=lambda x: -(csc(x) ** 2),
    func_name="cot",
)
asin = register_unary_func(
    eval_func=math.asin,
    diff_x=lambda x: 1 / sqrt(1 - x**2),
    func_name="asin",
)
acos = register_unary_func(
    eval_func=math.acos,
    diff_x=lambda x: -1 / sqrt(1 - x**2),
    func_name="acos",
)
atan = register_unary_func(
    eval_func=math.atan,
    diff_x=lambda x: 1 / (1 + x**2),
    func_name="atan",
)
