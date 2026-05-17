from __future__ import annotations

from typing import TYPE_CHECKING

import symdiff as A

if TYPE_CHECKING:
    from typing import Callable

    UnaryFunction = Callable[[A.Expression], A.Expression]
    BinaryFunction = Callable[[A.Expression, A.Expression], A.Expression]
