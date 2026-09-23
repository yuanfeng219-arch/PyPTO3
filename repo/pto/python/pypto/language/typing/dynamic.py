# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Dynamic shape variables for use in type annotations."""

from typing import Any

from pypto.pypto_core import DataType
from pypto.pypto_core.ir import Expr, ScalarType, Span, Var

from .scalar import Scalar


class DynVar(Scalar):
    """Dynamic shape variable for use in type annotations.

    Creates a symbolic dimension that becomes an ir.Var node in the IR shape.
    Inherits from Scalar so that DynVar is accepted wherever Scalar/IntLike
    is expected (e.g. shape parameters, TensorView valid_shape).

    Example:
        M = pl.dynamic("M")
        N = pl.dynamic("N")

        @pl.function
        def func(a: pl.Tensor[[M, N], pl.FP32]) -> ...:
            ...
    """

    def __init__(self, name: str) -> None:
        if not name.isidentifier():
            raise ValueError(f"DynVar name must be a valid identifier, got {name!r}")
        self.name = name
        # Lazily populated on first use by TypeResolver.  Once set, all parsers
        # that encounter this DynVar object share the same ir.Var instance,
        # ensuring structural equality across @pl.function boundaries.
        self._ir_var: Any = None
        # Bypass Scalar.__init__ (which requires dtype or expr) and set
        # its fields directly.  The actual expr is provided lazily via unwrap().
        self.dtype = DataType.INDEX
        self.expr = None
        self._annotation_only = False

    def unwrap(self) -> Expr:
        """Return the underlying ir.Var, creating it eagerly if needed.

        This allows DynVar to participate in _normalize_intlike() and other
        Scalar-consuming paths without requiring prior TypeResolver resolution.
        """
        if self._ir_var is None:
            self._ir_var = Var(self.name, ScalarType(DataType.INDEX), Span.unknown())
        # Keep Scalar.expr in sync so direct .expr access returns a valid value.
        self.expr = self._ir_var
        return self._ir_var

    def __repr__(self) -> str:
        return f"DynVar({self.name!r})"


def dynamic(name: str) -> DynVar:
    """Create a dynamic shape variable for type annotations.

    Args:
        name: Variable name for the dynamic dimension

    Returns:
        DynVar that can be used in shape annotations
    """
    return DynVar(name)  # type: ignore[return-value]  # metaclass __call__ typed as -> Scalar


__all__ = ["DynVar", "dynamic"]
