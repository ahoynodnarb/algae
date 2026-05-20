from __future__ import annotations

from typing import TYPE_CHECKING

import algae as A
import algae.engine as engine

if TYPE_CHECKING:
    from typing import Callable, Dict, Tuple

    UnaryFunction = Callable[[A.Expression], A.Expression]
    BinaryFunction = Callable[[A.Expression, A.Expression], A.Expression]

    RuleContext = Dict[engine.Tag, Tuple[engine.Tag, A.Expression]]
    Applier = Callable[[RuleContext], A.Expression]
    Rule = Tuple[engine.Tag, Applier]
