from __future__ import annotations

from contextvars import ContextVar
from typing import TYPE_CHECKING

import algae.algebra as sd
import algae.functions as F

if TYPE_CHECKING:
    from typing import Any, List

    import algae.types as sdt


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


class ConstantTag(Tag):
    identifier: sd.Constant


class VariableTag(GenericTag):
    identifier: sd.Variable

    def __init__(self, identifier):
        super().__init__(type(self), identifier)


class FuncTag(Tag):
    identifier: sd.SymFunc

    def __init__(self, identifier: sd.SymFunc):
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
    identifier: sd.UnarySymFunc

    def __call__(self, arg: Tag):
        return super().__call__(arg)


class BinaryFuncTag(FuncTag):
    identifier: sd.BinarySymFunc

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


def fold_constants(ctx: sdt.RuleContext) -> sd.Constant:
    _, f = ctx["f"]
    _, a = ctx["a"]
    _, b = ctx["b"]
    return sd.Constant(f(a, b).eval())


def group_commutative(ctx: sdt.RuleContext) -> sd.Expression:
    _, f = ctx["f"]
    _, e = ctx["e"]
    _, a = ctx["a"]
    return f(e, a)


def group_associative(ctx: sdt.RuleContext) -> sd.Expression:
    _, f = ctx["f"]
    _, a = ctx["a"]
    _, b = ctx["b"]
    _, c = ctx["c"]
    return f(b, f(a, c))


def transpose_add(ctx: sdt.RuleContext) -> sd.Expression:
    _, a = ctx["a"]
    _, e = ctx["e"]
    return e + a


def transpose_mul(ctx: sdt.RuleContext) -> sd.Expression:
    _, a = ctx["a"]
    _, e = ctx["e"]
    return a * e


def combine_add(ctx: sdt.RuleContext) -> sd.Expression:
    _, x = ctx["x"]
    return 2 * x


def combine_mul(ctx: sdt.RuleContext) -> sd.Expression:
    _, a = ctx["a"]
    _, x = ctx["x"]
    return (a + 1) * x


folding_rules = (
    (
        GenericFuncTag(BinaryFuncTag, "f")(
            GenericTag(ConstantTag, "a"),
            GenericTag(ConstantTag, "b"),
        ),
        fold_constants,
    ),
)
combining_rules = (
    (
        AddTag(
            VariableTag("x"),
            VariableTag("x"),
        ),
        combine_add,
    ),
    (
        AddTag(
            MulTag(GenericTag(ConstantTag, "a"), VariableTag("x")),
            VariableTag("x"),
        ),
        combine_mul,
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
            GenericTag(Tag, "e"),
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
        lambda ctx: ctx["a"][1] + (-ctx["b"][1]),
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
        lambda ctx: sd.Constant(0),
    ),
    (
        MulTag(
            GenericTag(Tag, "e"),
            ConstantTag(0),
        ),
        lambda ctx: sd.Constant(0),
    ),
    (
        MulTag(
            ConstantTag(1),
            GenericTag(Tag, "e"),
        ),
        lambda ctx: ctx["e"][1],
    ),
    (
        MulTag(
            GenericTag(Tag, "e"),
            ConstantTag(1),
        ),
        lambda ctx: ctx["e"][1],
    ),
    (
        AddTag(
            ConstantTag(0),
            GenericTag(Tag, "e"),
        ),
        lambda ctx: ctx["e"][1],
    ),
    (
        AddTag(
            GenericTag(Tag, "e"),
            ConstantTag(0),
        ),
        lambda ctx: ctx["e"][1],
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
    (
        AddTag(
            GenericTag(Tag, "e"),
            NegativeConstantTag("a"),
        ),
        lambda ctx: ctx["e"][1] - (-ctx["a"][1]),
    ),
)


def get_tag(expr: sd.Expression) -> Tag:
    if isinstance(expr, sd.Variable):
        return VariableTag(expr.name)
    if isinstance(expr, sd.Constant):
        return ConstantTag(expr.val)

    source = expr.node.source
    if source is None:
        return None

    if isinstance(source, sd.UnarySymFunc):
        return UnaryFuncTag(source)
    if isinstance(source, sd.BinarySymFunc):
        return BinaryFuncTag(source)

    return None


def update_pattern_ctx(expr: sd.Expression, pattern: Tag, ctx: sdt.RuleContext) -> bool:
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


def post_order_apply(expr: sd.Expression, ruleset: tuple[sdt.Rule]) -> sd.Expression:
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


def pre_order_apply(expr: sd.Expression, ruleset: tuple[sdt.Rule]) -> sd.Expression:
    for pattern, applier in ruleset:
        ctx = {}
        matched = update_pattern_ctx(expr, pattern, ctx)

        if not matched:
            continue

        return pre_order_apply(applier(ctx), ruleset)

    expr.node.args = [pre_order_apply(arg, ruleset) for arg in expr.node.args]

    return expr


def simplify(expr: sd.Expression) -> sd.Expression:
    with no_engine():
        expr = pre_order_apply(expr, EARLY_DISPATCH_RULES)
        expr = post_order_apply(expr, REGULAR_DISPATCH_RULES)
        expr = post_order_apply(expr, FINAL_DISPATCH_RULES)
    return expr
