from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Any, Callable, List, Optional, Set

    import algae.types as sdt

import algae as sd
from algae.utils import remove_redudant_parens


class ExpressionNode:
    args: List[Expression]
    source: Optional[SymFunc]
    repr_func: Optional[Callable]
    name: str
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
        self.repr_func = repr_func
        self.name = expression_name

        self.diff_funcs = None if source is None else source.diff_funcs
        self.variables = (
            set() if not args else set.union(*[arg.node.variables for arg in self.args])
        )

    def diff(self, wrt: Variable) -> Expression:
        if self.diff_funcs is None or wrt not in self.variables:
            return Constant(0)

        return sum(
            [
                diff_func(*self.args) * arg.diff(wrt)
                for diff_func, arg in zip(self.diff_funcs, self.args)
            ]
        )

    def eval(self, *vars: FrozenVariable) -> Any:
        if self.source is None:
            raise ValueError("Expression has no evaluation function")

        if any([var not in vars for var in self.variables]):
            raise ValueError(
                f"Variable {self.name} uninitialized while attempting to evaluate expression"
            )

        return self.source.eval(*[arg.eval(*vars) for arg in self.args])

    def is_constant(self) -> bool:
        return len(self.variables) == 0

    def __repr__(self) -> str:
        if self.repr_func is None:
            return super().__repr__()

        return remove_redudant_parens(self.repr_func(*self.args))


class ConstantNode(ExpressionNode):
    def __init__(self, const: Constant):
        super().__init__(expression_name=str(const.val))
        self.val = const.val

    def diff(self, wrt: Variable) -> Expression:
        return Constant(0)

    def eval(self, *vars: FrozenVariable) -> Any:
        return self.val

    def __repr__(self) -> str:
        return str(self.val)


class VariableNode(ExpressionNode):
    def __init__(self, var: Variable):
        super().__init__(expression_name=var.name)
        self.var = var
        self.variables = set([self.var])

    def diff(self, wrt: Variable) -> Expression:
        return Constant(1) if wrt == self.var else Constant(0)

    def eval(self, *vars: FrozenVariable) -> Any:
        frozen = vars[vars.index(self.var)]
        return frozen.val


class FrozenVariableNode(VariableNode):
    def __init__(self, var: FrozenVariable):
        super().__init__(var)
        self.val = var.val

    def eval(self, *vars: FrozenVariable) -> Any:
        return self.val


class AlgebraObj:
    def eval(self) -> Any:
        raise NotImplementedError

    def __repr__(self) -> str:
        raise NotImplementedError()


class Expression(AlgebraObj):
    node: ExpressionNode

    def __init__(self, node: ExpressionNode):
        self.node = node

    def __neg__(self) -> Expression:
        return -1 * self

    def __add__(self, other: Expression | Any) -> Expression:
        return sd.add(self, other)

    def __radd__(self, other: Expression | Any) -> Expression:
        return sd.add(other, self)

    def __sub__(self, other: Expression | Any) -> Expression:
        return sd.subtract(self, other)

    def __rsub__(self, other: Expression | Any) -> Expression:
        return sd.subtract(other, self)

    def __mul__(self, other: Expression | Any) -> Expression:
        return sd.multiply(self, other)

    def __rmul__(self, other: Expression | Any) -> Expression:
        return sd.multiply(other, self)

    def __truediv__(self, other: Expression | Any) -> Expression:
        return sd.divide(self, other)

    def __rtruediv__(self, other: Expression | Any) -> Expression:
        return sd.divide(other, self)

    def __pow__(self, other: Expression | Any) -> Expression:
        return sd.pow(self, other)

    def __rpow__(self, other: Expression | Any) -> Expression:
        return sd.pow(other, self)

    def eval(self, *vars: FrozenVariable) -> Any:
        return self.node.eval(*vars)

    def diff(self, wrt: Variable) -> Expression:
        return self.node.diff(wrt)

    def is_constant(self) -> bool:
        return self.node.is_constant()

    def __repr__(self) -> str:
        return self.node.__repr__()


class Constant(Expression):
    def __init__(self, val: Any):
        self.val = val
        self.node = ConstantNode(self)

    def __neg__(self) -> Constant:
        return Constant(-self.val)

    def __hash__(self) -> int:
        return self.val.__hash__()

    def __eq__(self, other: Constant | FrozenVariable | Any) -> bool:
        if isinstance(other, (Constant, FrozenVariable)):
            return self.val == other.val

        return self.val == other

    def __lt__(self, other: Constant | FrozenVariable | Any) -> bool:
        if isinstance(other, (Constant, FrozenVariable)):
            return self.val < other.val

        return self.val < other

    def __le__(self, other: Constant | FrozenVariable | Any) -> bool:
        if isinstance(other, (Constant, FrozenVariable)):
            return self.val <= other.val

        return self.val <= other

    def __gt__(self, other: Constant | FrozenVariable | Any) -> bool:
        if isinstance(other, (Constant, FrozenVariable)):
            return self.val > other.val

        return self.val > other

    def __ge__(self, other: Constant | FrozenVariable | Any) -> bool:
        if isinstance(other, (Constant, FrozenVariable)):
            return self.val >= other.val

        return self.val >= other


class Variable(Expression):
    def __init__(self, name: str):
        self.name = name
        self.node = VariableNode(self)

    def set(self, val: Any) -> Any:
        return FrozenVariable(self, val)

    def copy(self) -> Variable:
        return Variable(self.name)

    def __hash__(self) -> int:
        return self.name.__hash__()

    def __eq__(self, value: Any) -> bool:
        if not isinstance(value, Variable):
            return False
        return self.name == value.name

    def __repr__(self) -> str:
        return self.name


class FrozenVariable(Variable, Constant):
    def __init__(self, var: Variable, val: Any):
        self.name = var.name
        self.val = val
        self.node = FrozenVariableNode(self)


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
        import algae.engine as engine

        args = [x if isinstance(x, AlgebraObj) else Constant(x) for x in args]
        expr = Expression(
            ExpressionNode(
                *args,
                source=self,
                repr_func=self.repr_func,
                expression_name=self.func_name,
            )
        )
        if engine.engine_enabled():
            expr = engine.simplify(expr)
        return expr

    def eval(self, *args: Any) -> Any:
        return self.eval_func(*args)

    def __repr__(self) -> str:
        return self.func_name


class UnarySymFunc(SymFunc):
    def __init__(
        self,
        eval_func: Callable[[Any], Any],
        diff_x: sdt.UnaryFunction,
        repr_func: Callable[[Expression], str],
        func_name: str,
    ):
        return super().__init__(
            eval_func=eval_func,
            diff_funcs=[diff_x],
            repr_func=repr_func,
            func_name=func_name,
        )

    def __call__(self, arg: Any) -> Expression:
        return super().__call__(arg)

    def eval(self, arg: Any) -> Any:
        return super().eval(arg)


class BinarySymFunc(SymFunc):
    associates: bool
    commutes: bool

    def __init__(
        self,
        eval_func: Callable[[Any, Any], Any],
        diff_x: sdt.BinaryFunction,
        diff_y: sdt.BinaryFunction,
        repr_func: Callable[[Expression, Expression], str],
        func_name: str,
        associates: bool = False,
        commutes: bool = False,
    ):
        super().__init__(
            eval_func=eval_func,
            diff_funcs=[diff_x, diff_y],
            repr_func=repr_func,
            func_name=func_name,
        )
        self.associates = associates
        self.commutes = commutes

    def __call__(self, arg1: Any, arg2: Any) -> Expression:
        return super().__call__(arg1, arg2)

    def eval(self, arg1: Any, arg2: Any) -> Any:
        return super().eval(arg1, arg2)
