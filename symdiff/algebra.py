from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Any, Callable, List, Optional, Set

import symdiff as sd
from symdiff.utils import remove_redudant_parens


class ExpressionNode:
    variables: Set[Variable]

    def __init__(
        self,
        *args: Expression,
        source: Optional[SymFunc] = None,
        repr_func: Optional[Callable] = None,
        expression_name: str = "",
    ):
        self.args = args
        self.source = source
        self.diff_funcs = None if source is None else source.diff_funcs
        self.repr_func = repr_func
        self.name = expression_name

        self.variables = set.union(*[arg.variables for arg in self.args])

    def diff(self, wrt: Variable) -> Expression:
        if self.diff_funcs is None or wrt not in self.variables:
            return Constant(0)
        return sum(
            [
                diff_func(*self.args) * arg.diff(wrt)
                for diff_func, arg in zip(self.diff_funcs, self.args)
            ]
        )

    def eval(self) -> Any:
        if self.source is None:
            raise ValueError("Expression has no evaluation function")

        return self.source.eval(*[arg.eval() for arg in self.args])

    def __repr__(self) -> str:
        if self.repr_func is None:
            return super().__repr__()

        return remove_redudant_parens(self.repr_func(*self.args))


class AlgebraObj:
    def eval(self) -> Any:
        raise NotImplementedError

    def __repr__(self) -> str:
        raise NotImplementedError()


class Expression(AlgebraObj):
    variables: Set[Variable]

    def __init__(self, node: ExpressionNode):
        self.node = node
        self.variables = node.variables

    def __neg__(self) -> Expression:
        return Constant(-1) * self

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

    def __pow__(self, other: Expression) -> Expression:
        return sd.pow(self, other)

    def __rpow__(self, other: Expression) -> Expression:
        return sd.pow(other, self)

    def eval(self) -> Any:
        return self.node.eval()

    def diff(self, wrt: Variable) -> Expression:
        return self.node.diff(wrt)

    def __repr__(self) -> str:
        return self.node.__repr__()


class Constant(Expression):
    def __init__(self, val: Any):
        self.val = val
        self.variables = set()

    def eval(self) -> Any:
        return self.val

    def diff(self, wrt: Variable) -> Expression:
        return Constant(0)

    def __repr__(self) -> str:
        return self.val.__repr__()


class Variable(Expression):
    def __init__(self, name: str):
        self.name = name
        self.val = None
        self.variables = set([self])

    def eval(self, val: Optional[Any] = None) -> Any:
        if val is not None:
            self.val = val

        if self.val is None:
            raise ValueError(
                f"Variable {self.name} uninitialized while attempting to evaluate expression"
            )

        return self.val

    def diff(self, wrt: Variable) -> Expression:
        return Constant(1) if wrt == self else Constant(0)

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
        self,
        eval_func: Callable,
        diff_funcs: List[Callable],
        repr_func: Callable,
        func_name: str,
    ):
        self.eval_func = eval_func
        self.diff_funcs = diff_funcs
        self.repr_func = repr_func
        self.func_name = func_name

    def __call__(self, *args: Any) -> Expression:
        args = [x if isinstance(x, AlgebraObj) else Constant(x) for x in args]
        node = ExpressionNode(
            *args,
            source=self,
            repr_func=self.repr_func,
            expression_name=self.func_name,
        )
        return Expression(node)

    def eval(self, *args: Any) -> Any:
        return self.eval_func(*args)
