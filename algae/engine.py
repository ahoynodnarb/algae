from __future__ import annotations

from contextvars import ContextVar
from typing import TYPE_CHECKING

import algae.algebra as alg
import algae.functions as F

if TYPE_CHECKING:
    from typing import Any, List

    import algae.types as algt


_engine_enabled = ContextVar("engine_enabled", default=True)


def engine_enabled() -> bool:
    return _engine_enabled.get()


class no_engine:
    def __enter__(self):
        self.prev = _engine_enabled.get()
        _engine_enabled.set(False)

    def __exit__(self, type, value, traceback):
        _engine_enabled.set(self.prev)


class Tag:
    identifier: Any
    args: List[Tag]
    expr: alg.Expression

    def __init__(self, identifier: Any):
        self.identifier = identifier
        self.args = []
        self.expr = None

    def matches(self, other: Tag, ctx: algt.RuleContext) -> bool:
        if isinstance(other, GenericTag):
            return self == other or other.matches(self, ctx)

        return self == other

    def __hash__(self) -> int:
        return hash((type(self), self.identifier))

    def __eq__(self, other) -> bool:
        return type(self) is type(other) and self.identifier == other.identifier


# matches with any expression subclassing generic
class GenericTag(Tag):
    generic: Tag

    def __init__(self, generic: Tag, identifier: Any):
        super().__init__(identifier=identifier)
        self.generic = generic

    def matches(self, other: Tag, ctx: algt.RuleContext) -> bool:
        if not isinstance(other, self.generic):
            return False
        if self.identifier in ctx and ctx[self.identifier] != other:
            return False
        return True


# matches exactly if a tag does not subclass a generic tag
class GenericAntiTag(GenericTag):
    def matches(self, other: Tag, ctx: algt.RuleContext) -> bool:
        if isinstance(other, self.generic):
            return False
        if self.identifier in ctx and ctx[self.identifier] != other:
            return False
        return True


# matches exactly if two expressions of any subclass of the generic type are exactly the same
class ExactGenericTag(GenericTag):
    def matches(self, other: Tag, ctx: algt.RuleContext) -> bool:
        if not isinstance(other, self.generic):
            return False
        if self.identifier in ctx and ctx[self.identifier].expr != other.expr:
            return False
        return True


# matches exactly if two expressions of have the same type and are exactly the same
class ExactTag(GenericTag):
    generic: type

    def matches(self, other: Tag, ctx: algt.RuleContext) -> bool:
        if type(other) != self.generic:
            return False
        if self.identifier in ctx and ctx[self.identifier].expr != other.expr:
            return False
        return True


# matches exactly if two expressions are different
class ExactAntiTag(GenericTag):
    def matches(self, other: Tag, ctx: algt.RuleContext) -> bool:
        if self.generic == other:
            return False
        if self.identifier in ctx and ctx[self.identifier].expr != other.expr:
            return False
        return True


class ConstantTag(Tag):
    identifier: alg.Constant


class VariableTag(Tag):
    identifier: alg.Variable


class FuncTag(Tag):
    identifier: alg.SymFunc

    def __init__(self, identifier: alg.SymFunc):
        super().__init__(identifier)

    def __call__(self, *args: List[Tag]) -> Tag:
        ret = type(self)(self.identifier)
        ret.args = args
        return ret


class UnaryFuncAntiTag(ExactAntiTag, FuncTag):
    generic: FuncTag
    identifier: Any

    def __init__(self, generic: UnaryFuncTag, identifier: Any):
        super().__init__(generic, identifier)

    def matches(self, other: Tag, ctx: algt.RuleContext) -> bool:
        return super().matches(other, ctx) and isinstance(other, UnaryFuncTag)

    def __call__(self, *args: List[Tag]) -> Tag:
        ret = type(self)(self.generic, self.identifier)
        ret.args = args
        return ret


class BinaryFuncAntiTag(ExactAntiTag, FuncTag):
    generic: FuncTag
    identifier: Any

    def __init__(self, generic: BinaryFuncTag, identifier: Any):
        super().__init__(generic, identifier)

    def matches(self, other: Tag, ctx: algt.RuleContext) -> bool:
        return super().matches(other, ctx) and isinstance(other, BinaryFuncTag)

    def __call__(self, *args: List[Tag]) -> Tag:
        ret = type(self)(self.generic, self.identifier)
        ret.args = args
        return ret


class GenericFuncTag(GenericTag, FuncTag):
    generic: alg.SymFunc

    def __init__(self, generic: alg.SymFunc, identifier: Any):
        super().__init__(generic, identifier)

    def __call__(self, *args: List[Tag]) -> Tag:
        ret = type(self)(self.generic, self.identifier)
        ret.args = args
        return ret


class UnaryFuncTag(FuncTag):
    identifier: alg.UnarySymFunc

    def __call__(self, arg: Tag):
        return super().__call__(arg)


class BinaryFuncTag(FuncTag):
    identifier: alg.BinarySymFunc

    def __call__(self, arg1: Tag, arg2: Tag):
        return super().__call__(arg1, arg2)


class CommutativeTag(GenericTag, BinaryFuncTag):
    def __init__(self, identifier: alg.BinarySymFunc):
        super().__init__(BinaryFuncTag, identifier)

    def matches(self, other: Tag, ctx: algt.RuleContext) -> bool:
        return super().matches(other, ctx) and other.identifier.commutes

    def __call__(self, *args: List[Tag]) -> Tag:
        ret = type(self)(self.identifier)
        ret.args = args
        return ret


class AssociativeTag(GenericTag, BinaryFuncTag):
    def __init__(self, identifier: alg.BinarySymFunc):
        super().__init__(BinaryFuncTag, identifier)

    def matches(self, other: Tag, ctx: algt.RuleContext) -> bool:
        return super().matches(other, ctx) and other.identifier.associates

    def __call__(self, *args: List[Tag]) -> Tag:
        ret = type(self)(self.identifier)
        ret.args = args
        return ret


class PositiveConstantTag(GenericTag):
    def __init__(self, identifier: Any):
        super().__init__(ConstantTag, identifier)

    def matches(self, other: Tag, ctx: algt.RuleContext) -> bool:
        return super().matches(other, ctx) and other.identifier > 0


class NegativeConstantTag(GenericTag):
    def __init__(self, identifier: Any):
        super().__init__(ConstantTag, identifier)

    def matches(self, other: Tag, ctx: algt.RuleContext) -> bool:
        return super().matches(other, ctx) and other.identifier < 0


AddTag = BinaryFuncTag(F.add)
SubTag = BinaryFuncTag(F.subtract)
MulTag = BinaryFuncTag(F.multiply)
DivTag = BinaryFuncTag(F.divide)
PowTag = BinaryFuncTag(F.pow)
SqrtTag = UnaryFuncTag(F.sqrt)
LogTag = UnaryFuncTag(F.log)
ExpTag = UnaryFuncTag(F.exp)
SinTag = UnaryFuncTag(F.sin)
CosTag = UnaryFuncTag(F.cos)
TanTag = UnaryFuncTag(F.tan)
CscTag = UnaryFuncTag(F.csc)
SecTag = UnaryFuncTag(F.sec)
CotTag = UnaryFuncTag(F.cot)
AsinTag = UnaryFuncTag(F.asin)
AcosTag = UnaryFuncTag(F.acos)
AtanTag = UnaryFuncTag(F.atan)


def fold_constants_binary(ctx: algt.RuleContext) -> alg.Constant:
    f = ctx["f"].expr
    a = ctx["a"].expr
    b = ctx["b"].expr
    return alg.Constant(f(a, b).eval())


def fold_constants_unary(ctx: algt.RuleContext) -> alg.Constant:
    f = ctx["f"].expr
    a = ctx["a"].expr
    return alg.Constant(f(a).eval())


def group_commutative(ctx: algt.RuleContext) -> alg.Expression:
    f = ctx["f"].expr
    e = ctx["e"].expr
    a = ctx["a"].expr
    return f(e, a)


def group_associative(ctx: algt.RuleContext) -> alg.Expression:
    f = ctx["f"].expr
    a = ctx["a"].expr
    b = ctx["b"].expr
    c = ctx["c"].expr
    return f(b, f(a, c))


def transpose_add(ctx: algt.RuleContext) -> alg.Expression:
    a = ctx["a"].expr
    e = ctx["e"].expr
    return e + a


def transpose_mul(ctx: algt.RuleContext) -> alg.Expression:
    a = ctx["a"].expr
    e = ctx["e"].expr
    return a * e


def add_like_no_coeff(ctx: algt.RuleContext) -> alg.Expression:
    e = ctx["e"].expr
    return 2 * e


def add_like_one_coeff(ctx: algt.RuleContext) -> alg.Expression:
    a = ctx["a"].expr
    e = ctx["e"].expr
    return (a + 1) * e


def add_like_two_coeff(ctx: algt.RuleContext) -> alg.Expression:
    a = ctx["a"].expr
    b = ctx["b"].expr
    e = ctx["e"].expr
    return (a + b) * e


def mul_var_no_coeff(ctx: algt.RuleContext) -> alg.Expression:
    x = ctx["x"].expr
    a = ctx["a"].expr
    b = ctx["b"].expr
    return x ** (a + b)


def mul_var_one_coeff(ctx: algt.RuleContext) -> alg.Expression:
    x = ctx["x"].expr
    a = ctx["a"].expr
    b = ctx["b"].expr
    c = ctx["c"].expr
    return c * x ** (a + b)


def distribute(ctx: algt.RuleContext) -> alg.Expression:
    e1 = ctx["e1"].expr
    e2 = ctx["e2"].expr
    e3 = ctx["e3"].expr
    return e1 * e2 + e1 * e3


def format_sub(ctx: algt.RuleContext) -> alg.Expression:
    a = ctx["a"].expr
    b = ctx["b"].expr
    return a + (-b)


def reformat_sub(ctx: algt.RuleContext) -> alg.Expression:
    e = ctx["e"].expr
    a = ctx["a"].expr
    return e - (-a)


def format_unary_linear(ctx: algt.RuleContext) -> alg.Expression:
    f = ctx["f"].expr
    x = ctx["x"].expr
    return f(x**1)


def format_binary_linear_first(ctx: algt.RuleContext) -> alg.Expression:
    f = ctx["f"].expr
    x = ctx["x"].expr
    e = ctx["e"].expr
    ret = f(x**1, e)
    return ret


def format_binary_linear_second(ctx: algt.RuleContext) -> alg.Expression:
    f = ctx["f"].expr
    e = ctx["e"].expr
    x = ctx["x"].expr
    return f(e, x**1)


def reformat_pow(ctx: algt.RuleContext) -> alg.Expression:
    e = ctx["e"].expr
    return e


def reformat_mul(ctx: algt.RuleContext) -> alg.Expression:
    a = ctx["a"].expr
    b = ctx["b"].expr
    c = ctx["c"].expr
    return a - (-b * c)


def inverse_short_circuit(ctx: algt.RuleContext) -> alg.Expression:
    e = ctx["e"].expr
    return e


def zero_short_circuit(ctx: algt.RuleContext) -> alg.Expression:
    return alg.Constant(0)


def one_short_circuit(ctx: algt.RuleContext) -> alg.Expression:
    return alg.Constant(1)


folding_rules = (
    (
        GenericFuncTag(UnaryFuncTag, "f")(
            GenericTag(ConstantTag, "a"),
        ),
        fold_constants_unary,
    ),
    (
        GenericFuncTag(BinaryFuncTag, "f")(
            GenericTag(ConstantTag, "a"),
            GenericTag(ConstantTag, "b"),
        ),
        fold_constants_binary,
    ),
)
combining_rules = (
    (
        AddTag(
            ExactAntiTag(AddTag, "e"),
            ExactAntiTag(AddTag, "e"),
        ),
        add_like_no_coeff,
    ),
    (
        AddTag(
            MulTag(
                GenericTag(ConstantTag, "a"),
                ExactAntiTag(AddTag, "e"),
            ),
            ExactAntiTag(AddTag, "e"),
        ),
        add_like_one_coeff,
    ),
    (
        AddTag(
            MulTag(
                GenericTag(ConstantTag, "a"),
                ExactAntiTag(AddTag, "e"),
            ),
            MulTag(
                GenericTag(ConstantTag, "b"),
                ExactAntiTag(AddTag, "e"),
            ),
        ),
        add_like_two_coeff,
    ),
    (
        MulTag(
            PowTag(
                GenericTag(VariableTag, "x"),
                GenericTag(ConstantTag, "a"),
            ),
            PowTag(
                GenericTag(VariableTag, "x"),
                GenericTag(ConstantTag, "b"),
            ),
        ),
        mul_var_no_coeff,
    ),
    (
        MulTag(
            MulTag(
                GenericTag(ConstantTag, "c"),
                PowTag(
                    GenericTag(VariableTag, "x"),
                    GenericTag(ConstantTag, "a"),
                ),
            ),
            PowTag(
                GenericTag(VariableTag, "x"),
                GenericTag(ConstantTag, "b"),
            ),
        ),
        mul_var_one_coeff,
    ),
)
distributing_rules = (
    (
        MulTag(
            GenericTag(Tag, "e1"),
            AddTag(
                GenericAntiTag(ConstantTag, "e2"),
                GenericTag(Tag, "e3"),
            ),
        ),
        distribute,
    ),
)
formatting_rules = (
    (
        AddTag(
            GenericTag(ConstantTag, "a"),
            GenericAntiTag(ConstantTag, "e"),
        ),
        transpose_add,
    ),
    (
        MulTag(
            GenericAntiTag(ConstantTag, "e"),
            GenericTag(ConstantTag, "a"),
        ),
        transpose_mul,
    ),
    (
        AssociativeTag("f")(
            AssociativeTag("f")(
                GenericTag(Tag, "a"),
                GenericTag(ConstantTag, "b"),
            ),
            GenericTag(Tag, "c"),
        ),
        group_associative,
    ),
    (
        SubTag(
            GenericTag(Tag, "a"),
            GenericTag(Tag, "b"),
        ),
        format_sub,
    ),
    (
        UnaryFuncAntiTag(PowTag, "f")(
            GenericTag(VariableTag, "x"),
        ),
        format_unary_linear,
    ),
    (
        BinaryFuncAntiTag(PowTag, "f")(
            GenericTag(VariableTag, "x"),
            GenericTag(Tag, "e"),
        ),
        format_binary_linear_first,
    ),
    (
        BinaryFuncAntiTag(PowTag, "f")(
            GenericTag(Tag, "e"),
            GenericTag(VariableTag, "x"),
        ),
        format_binary_linear_second,
    ),
)
grouping_rules = (
    (
        AssociativeTag("f")(
            GenericTag(ConstantTag, "a"),
            AssociativeTag("f")(
                GenericAntiTag(ConstantTag, "b"),
                GenericTag(ConstantTag, "c"),
            ),
        ),
        group_associative,
    ),
    (
        AssociativeTag("f")(
            AssociativeTag("f")(
                GenericTag(Tag, "a"),
                GenericTag(ConstantTag, "b"),
            ),
            GenericTag(ConstantTag, "c"),
        ),
        group_associative,
    ),
)
identity_rules = (
    (
        MulTag(
            ConstantTag(0),
            GenericTag(Tag, "e"),
        ),
        zero_short_circuit,
    ),
    (
        MulTag(
            GenericTag(Tag, "e"),
            ConstantTag(0),
        ),
        zero_short_circuit,
    ),
    (
        PowTag(
            ExactAntiTag(ConstantTag(0), "0"),
            ConstantTag(0),
        ),
        one_short_circuit,
    ),
    (
        MulTag(
            ConstantTag(1),
            GenericTag(Tag, "e"),
        ),
        inverse_short_circuit,
    ),
    (
        MulTag(
            GenericTag(Tag, "e"),
            ConstantTag(1),
        ),
        inverse_short_circuit,
    ),
    (
        AddTag(
            ConstantTag(0),
            GenericTag(Tag, "e"),
        ),
        inverse_short_circuit,
    ),
    (
        AddTag(
            GenericTag(Tag, "e"),
            ConstantTag(0),
        ),
        inverse_short_circuit,
    ),
    (
        ExpTag(LogTag(GenericAntiTag(NegativeConstantTag, "e"))),
        inverse_short_circuit,
    ),
    (
        LogTag(ExpTag(GenericAntiTag(NegativeConstantTag, "e"))),
        inverse_short_circuit,
    ),
)
normalization_rules = (
    (
        AddTag(
            GenericTag(Tag, "e"),
            NegativeConstantTag("a"),
        ),
        reformat_sub,
    ),
    (
        AddTag(
            GenericTag(Tag, "a"),
            MulTag(
                NegativeConstantTag("b"),
                GenericTag(Tag, "c"),
            ),
        ),
        reformat_mul,
    ),
    (
        PowTag(
            GenericTag(Tag, "e"),
            ConstantTag(1),
        ),
        reformat_pow,
    ),
)

EARLY_DISPATCH_RULES = (
    *folding_rules,
    *identity_rules,
    *combining_rules,
)
REGULAR_DISPATCH_RULES = (
    *folding_rules,
    *identity_rules,
    *grouping_rules,
    *formatting_rules,
    *combining_rules,
    *distributing_rules,
)
FINAL_DISPATCH_RULES = (
    *folding_rules,
    *identity_rules,
    *normalization_rules,
)


def get_tag(expr: alg.Expression) -> Tag:
    if isinstance(expr, alg.Variable):
        return VariableTag(expr.name)
    if isinstance(expr, alg.Constant):
        return ConstantTag(expr.val)

    source = expr.node.source
    if source is None:
        return None

    if isinstance(source, alg.UnarySymFunc):
        return UnaryFuncTag(source)
    if isinstance(source, alg.BinarySymFunc):
        return BinaryFuncTag(source)

    return None


def update_pattern_ctx(
    expr: alg.Expression, pattern: Tag, ctx: algt.RuleContext
) -> bool:
    expr_tag = get_tag(expr)

    if expr_tag is None:
        return False

    if isinstance(pattern, FuncTag):
        expr_tag.expr = expr.node.source
    else:
        expr_tag.expr = expr

    if not pattern.matches(expr_tag, ctx):
        return False

    ctx[pattern.identifier] = expr_tag

    for subexpr, subpattern in zip(expr.node.args, pattern.args):
        if not update_pattern_ctx(subexpr, subpattern, ctx=ctx):
            return False

    return True


def post_order_apply(expr: alg.Expression, ruleset: tuple[algt.Rule]) -> alg.Expression:
    expr.node.args = [post_order_apply(arg, ruleset) for arg in expr.node.args]
    for pattern, applier in ruleset:
        ctx = {}
        matched = update_pattern_ctx(expr, pattern, ctx)

        if not matched:
            continue

        expr = applier(ctx)
        expr = post_order_apply(expr, ruleset)
        return expr

    return expr


def pre_order_apply(expr: alg.Expression, ruleset: tuple[algt.Rule]) -> alg.Expression:
    for pattern, applier in ruleset:
        ctx = {}
        matched = update_pattern_ctx(expr, pattern, ctx)

        if not matched:
            continue

        return pre_order_apply(applier(ctx), ruleset)

    expr.node.args = [pre_order_apply(arg, ruleset) for arg in expr.node.args]

    return expr


def simplify(expr: alg.Expression) -> alg.Expression:
    try:
        with no_engine():
            expr = pre_order_apply(expr, EARLY_DISPATCH_RULES)
            expr = post_order_apply(expr, REGULAR_DISPATCH_RULES)
            expr = post_order_apply(expr, FINAL_DISPATCH_RULES)
    except RecursionError:
        pass
    return expr
