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

    def __init__(self, identifier: Any):
        self.identifier = identifier
        self.args = []

    def matches(self, other: Tag, ctx: dict[Tag, Tag]) -> bool:
        if isinstance(other, GenericTag):
            return self == other or other.matches(self, ctx)

        return self == other

    def __hash__(self) -> int:
        return hash((type(self), self.identifier))

    def __eq__(self, other) -> bool:
        return type(self) is type(other) and self.identifier == other.identifier


# matches with any expression subclassing generic
class GenericTag(Tag):
    def __init__(self, generic: Tag, identifier: Any):
        super().__init__(identifier=identifier)
        self.generic = generic

    def matches(self, other: Tag, ctx: dict[Tag, Tag]) -> bool:
        if not isinstance(other, self.generic):
            return False
        if self.identifier in ctx and ctx[self.identifier][0] != other:
            return False
        return True


class ExactGenericTag(GenericTag):
    def matches(self, other: Tag, ctx: dict[Tag, Tag]) -> bool:
        if not isinstance(other, self.generic):
            return False
        if self.identifier in ctx and ctx[self.identifier][1] != other:
            return False
        return True


class ConstantTag(Tag):
    identifier: alg.Constant


class VariableTag(GenericTag):
    identifier: alg.Variable

    def __init__(self, identifier):
        super().__init__(type(self), identifier)


class FuncTag(Tag):
    identifier: alg.SymFunc

    def __init__(self, identifier: alg.SymFunc):
        super().__init__(identifier)

    def __call__(self, *args: List[Tag]) -> Tag:
        ret = type(self)(self.identifier)
        ret.args = args
        return ret


class GenericFuncTag(GenericTag, FuncTag):
    def __init__(self, generic, identifier):
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
    def __init__(self, identifier):
        super().__init__(BinaryFuncTag, identifier)

    def matches(self, other: Tag, ctx: dict[Tag, Tag]) -> bool:
        return super().matches(other, ctx) and other.identifier.commutes

    def __call__(self, *args: List[Tag]) -> Tag:
        ret = type(self)(self.identifier)
        ret.args = args
        return ret


class AssociativeTag(GenericTag, BinaryFuncTag):
    def __init__(self, identifier):
        super().__init__(BinaryFuncTag, identifier)

    def matches(self, other: Tag, ctx: dict[Tag, Tag]) -> bool:
        return super().matches(other, ctx) and other.identifier.associates

    def __call__(self, *args: List[Tag]) -> Tag:
        ret = type(self)(self.identifier)
        ret.args = args
        return ret


class PositiveConstantTag(GenericTag):
    def __init__(self, identifier):
        super().__init__(ConstantTag, identifier)

    def matches(self, other, ctx):
        return super().matches(other, ctx) and other.identifier > 0


class NegativeConstantTag(GenericTag):
    def __init__(self, identifier):
        super().__init__(ConstantTag, identifier)

    def matches(self, other, ctx):
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
    _, f = ctx["f"]
    _, a = ctx["a"]
    _, b = ctx["b"]
    return alg.Constant(f(a, b).eval())


def fold_constants_unary(ctx: algt.RuleContext) -> alg.Constant:
    _, f = ctx["f"]
    _, a = ctx["a"]
    return alg.Constant(f(a).eval())


def group_commutative(ctx: algt.RuleContext) -> alg.Expression:
    _, f = ctx["f"]
    _, e = ctx["e"]
    _, a = ctx["a"]
    return f(e, a)


def group_associative(ctx: algt.RuleContext) -> alg.Expression:
    _, f = ctx["f"]
    _, a = ctx["a"]
    _, b = ctx["b"]
    _, c = ctx["c"]
    return f(b, f(a, c))


def transpose_add(ctx: algt.RuleContext) -> alg.Expression:
    _, a = ctx["a"]
    _, e = ctx["e"]
    return e + a


def transpose_mul(ctx: algt.RuleContext) -> alg.Expression:
    _, a = ctx["a"]
    _, e = ctx["e"]
    return a * e


def combine_no_coeff(ctx: algt.RuleContext) -> alg.Expression:
    _, e = ctx["e"]
    return 2 * e


def combine_one_coeff(ctx: algt.RuleContext) -> alg.Expression:
    _, a = ctx["a"]
    _, e = ctx["e"]
    return (a + 1) * e


def combine_two_coeff(ctx: algt.RuleContext) -> alg.Expression:
    _, a = ctx["a"]
    _, b = ctx["b"]
    _, e = ctx["e"]
    return (a + b) * e


def format_sub(ctx: algt.RuleContext) -> alg.Expression:
    _, a = ctx["a"]
    _, b = ctx["b"]
    return a + (-b)


def reformat_sub(ctx: algt.RuleContext) -> alg.Expression:
    _, e = ctx["e"]
    _, a = ctx["a"]
    return e - (-a)


def reformat_mul(ctx: algt.RuleContext) -> alg.Expression:
    _, a = ctx["a"]
    _, b = ctx["b"]
    _, c = ctx["c"]
    return a - (-b * c)


def expr_short_circuit(ctx: algt.RuleContext) -> alg.Expression:
    _, e = ctx["e"]
    return e


def zero_short_circuit(ctx: algt.RuleContext) -> alg.Expression:
    return alg.Constant(0)


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
            ExactGenericTag(Tag, "e"),
            ExactGenericTag(Tag, "e"),
        ),
        combine_no_coeff,
    ),
    (
        AddTag(
            MulTag(GenericTag(ConstantTag, "a"), ExactGenericTag(Tag, "e")),
            ExactGenericTag(Tag, "e"),
        ),
        combine_one_coeff,
    ),
    (
        AddTag(
            MulTag(GenericTag(ConstantTag, "a"), ExactGenericTag(Tag, "e")),
            MulTag(GenericTag(ConstantTag, "b"), ExactGenericTag(Tag, "e")),
        ),
        combine_two_coeff,
    ),
)
formatting_rules = (
    (
        AddTag(
            GenericTag(ConstantTag, "a"),
            VariableTag("e"),
        ),
        transpose_add,
    ),
    (
        AddTag(
            GenericTag(ConstantTag, "a"),
            GenericTag(FuncTag, "e"),
        ),
        transpose_add,
    ),
    (
        MulTag(
            GenericTag(VariableTag, "e"),
            GenericTag(ConstantTag, "a"),
        ),
        transpose_mul,
    ),
    (
        MulTag(
            GenericTag(FuncTag, "e"),
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
)


grouping_rules = (
    (
        AssociativeTag("f")(
            GenericTag(ConstantTag, "a"),
            AssociativeTag("f")(
                GenericTag(FuncTag, "b"),
                GenericTag(ConstantTag, "c"),
            ),
        ),
        group_associative,
    ),
    (
        AssociativeTag("f")(
            GenericTag(ConstantTag, "a"),
            AssociativeTag("f")(
                VariableTag("b"),
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
        MulTag(
            ConstantTag(1),
            GenericTag(Tag, "e"),
        ),
        expr_short_circuit,
    ),
    (
        MulTag(
            GenericTag(Tag, "e"),
            ConstantTag(1),
        ),
        expr_short_circuit,
    ),
    (
        AddTag(
            ConstantTag(0),
            GenericTag(Tag, "e"),
        ),
        expr_short_circuit,
    ),
    (
        AddTag(
            GenericTag(Tag, "e"),
            ConstantTag(0),
        ),
        expr_short_circuit,
    ),
    (
        ExpTag(LogTag(PositiveConstantTag("e"))),
        expr_short_circuit,
    ),
    (
        ExpTag(LogTag(GenericTag(FuncTag, "e"))),
        expr_short_circuit,
    ),
    (
        ExpTag(LogTag(GenericTag(VariableTag, "e"))),
        expr_short_circuit,
    ),
    (
        LogTag(ExpTag(PositiveConstantTag("e"))),
        expr_short_circuit,
    ),
    (
        LogTag(ExpTag(GenericTag(FuncTag, "e"))),
        expr_short_circuit,
    ),
    (
        LogTag(ExpTag(GenericTag(VariableTag, "e"))),
        expr_short_circuit,
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
)

EARLY_DISPATCH_RULES = (
    *folding_rules,
    *identity_rules,
)
REGULAR_DISPATCH_RULES = (
    *folding_rules,
    *grouping_rules,
    *formatting_rules,
    *combining_rules,
)
FINAL_DISPATCH_RULES = (
    *folding_rules,
    *normalization_rules,
    *identity_rules,
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

    if not pattern.matches(expr_tag, ctx):
        return False

    if isinstance(pattern, FuncTag):
        ctx[pattern.identifier] = (expr_tag, expr.node.source)
    else:
        ctx[pattern.identifier] = (expr_tag, expr)

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
    with no_engine():
        expr = pre_order_apply(expr, EARLY_DISPATCH_RULES)
        expr = post_order_apply(expr, REGULAR_DISPATCH_RULES)
        expr = post_order_apply(expr, FINAL_DISPATCH_RULES)
    return expr
