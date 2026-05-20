from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Any, Callable, Optional

    import algae.types as algt

import algae as alg


def register_unary_func(
    eval_func: Callable[[Any], Any],
    diff_x: algt.UnaryFunction,
    repr_func: Optional[Callable[[alg.Expression], str]] = None,
    func_name: Optional[str] = None,
) -> algt.UnaryFunction:
    if func_name is None:
        func_name = "alg_anonymous_unary"

    if repr_func is None:
        repr_func = lambda x: f"{func_name}({x})"

    return alg.UnarySymFunc(
        eval_func=eval_func,
        diff_x=diff_x,
        repr_func=repr_func,
        func_name=func_name,
    )


def register_binary_func(
    eval_func: Callable[[Any, Any], Any],
    diff_x: algt.BinaryFunction,
    diff_y: algt.BinaryFunction,
    repr_func: Optional[algt.BinaryFunction] = None,
    func_name: Optional[str] = None,
    associates: bool = False,
    commutes: bool = False,
) -> algt.BinaryFunction:
    if func_name is None:
        func_name = "alg_anonymous_binary"

    if repr_func is None:
        repr_func = lambda x, y: f"{func_name}({x}, {y})"

    return alg.BinarySymFunc(
        eval_func=eval_func,
        diff_x=diff_x,
        diff_y=diff_y,
        repr_func=repr_func,
        func_name=func_name,
        associates=associates,
        commutes=commutes,
    )
