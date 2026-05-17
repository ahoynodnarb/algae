from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Any, Callable, List, Optional

import symdiff as sd
from symdiff.utils import remove_redudant_parens


class ExpressionNode:
    def __init__(
        self,
        *args: Expression,
        eval_func: Optional[SymFunc] = None,
        repr_func: Optional[Callable] = None,
        expression_name: str = "",
    ):
        if eval_func is None:
            diff_funcs = {}

        self.args = args
        self.eval_func = eval_func
        self.diff_funcs = diff_funcs
        self.repr_func = repr_func
        self.name = expression_name

    def diff(self, wrt: Variable) -> Expression:
        diff_func = self.diff_funcs.get(wrt)
        if diff_func is None:
            return Constant(0)
        return diff_func(*self.args)

    def eval(self) -> Any:
        if self.eval_func is None:
            raise ValueError("Expression has no evaluation function")

        return self.eval_func(*[arg.eval() for arg in self.args])

    def __repr__(self) -> str:
        if self.repr_func is None:
            return super().__repr__()

        return remove_redudant_parens(self.repr_func(*self.args))


class AlgebraObj:
    def __repr__(self) -> str:
        raise NotImplementedError()


class Expression(AlgebraObj):
    def __init__(
        self,
        *args: Expression,
        eval_func: Callable = None,
        repr_func: Callable = None,
        expression_name: str = "",
    ):
        self.node = ExpressionNode(*args, eval_func, repr_func, expression_name)

    def __add__(self, other: Expression) -> Expression:
        return sd.add(self, other)

    def __radd__(self, other: Expression) -> Expression:
        return sd.add(other, self)

    def __sub__(self, other: Expression) -> Expression:
        return sd.subtract(self, other)

    def __rsub__(self, other: Expression) -> Expression:
        return sd.subtract(other, self)

    def __mul__(self, other: Expression) -> Expression:
        return sd.multiply(self, other)

    def __rmul__(self, other: Expression) -> Expression:
        return sd.multiply(other, self)

    def __truediv__(self, other: Expression) -> Expression:
        return sd.divide(self, other)

    def __rtruediv__(self, other: Expression) -> Expression:
        return sd.divide(other, self)

    def eval(self) -> Any:
        return self.node.eval()

    def __repr__(self) -> str:
        return self.node.__repr__()

    def diff(self, wrt: Variable) -> Expression:
        return self.node.diff(wrt)


class Constant(Expression):
    def __init__(self, val: Any):
        self.val = val

    def eval(self) -> Any:
        return self.val

    def __repr__(self) -> str:
        return self.val.__repr__()


class Variable(Expression):
    def __init__(self, name: str):
        self.name = name
        self.val = None

    def eval(self, val: Optional[Any] = None) -> Any:
        if val is not None:
            self.val = val

        if self.val is None:
            raise ValueError(
                f"Variable {self.name} uninitialized while attempting to evaluate expression"
            )

        return self.val

    def __eq__(self, value: Any) -> bool:
        if not isinstance(value, Variable):
            return False
        return self.name == value.name

    def __hash__(self):
        return self.name.__hash__()

    def __repr__(self) -> str:
        return self.name


class SymFunc(AlgebraObj):
    eval_func: Callable
    diff_funcs: List[Callable]

    def __init__(
        self, eval_func: Callable, diff_funcs: Optional[List[Callable]] = None
    ):
        if diff_funcs is None:
            diff_funcs = []

        self.eval_func = eval_func
        self.diff_funcs = diff_funcs

    def __call__(self, *args):
        args = [x if isinstance(x, AlgebraObj) else Constant(x) for x in args]
        return self.eval_func(*args)
