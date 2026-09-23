# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# ruff: noqa: F722, F821

"""Parser-level polymorphism for ``pld.DistributedTensor`` arguments.

When ``pl.store`` / ``pl.tile.mscatter`` / ``pl.tensor.slice`` /
``pl.tensor.assemble`` receive a ``pld.DistributedTensor`` argument, the
resulting ``Call``'s type must remain ``ir.DistributedTensorType`` (not be
silently downgraded to plain ``ir.TensorType``). This lets the SSA-capture
form ``data = pl.store(local, [0,0], data)`` keep the rebound ``data`` var
typed as a window-bound distributed tensor — which downstream passes
(``CollectCommGroups`` etc.) rely on for ``window_buffer_`` threading.

The fix has two coupled parts both exercised here:

1. ``parser/_dsl_invoker.py::_wrap_arg`` must dispatch
   ``DistributedTensorType`` *before* ``TensorType`` (the former is a
   pybind subclass of the latter; the naive ordering would always wrap
   distributed tensors as plain ``Tensor``).
2. The DSL wrappers must return ``input.__class__(expr=call_expr)`` so the
   wrapper class matches the input's runtime class.

The InCore function shape mirrors ``tests/st/distributed/test_l3_allreduce.py``:
the body returns by writing the final tile into a ``pl.Out[pl.Tensor]``
output parameter via ``pl.store(tile, ..., out)`` — that satisfies the
``-> pl.Tensor[...]`` signature (``pl.store`` returns the destination tensor
for chaining).
"""

import pypto.language as pl
import pypto.language.distributed as pld
import pytest
from pypto.pypto_core import ir


def _collect_calls(stmt: ir.Stmt, op_name: str) -> list[ir.Call]:
    """Walk a statement tree and collect every Call whose op matches ``op_name``."""
    found: list[ir.Call] = []

    def visit(value: object) -> None:
        if isinstance(value, ir.Call) and value.op.name == op_name:
            found.append(value)

    def walk(s: ir.Stmt) -> None:
        if isinstance(s, ir.AssignStmt):
            visit(s.value)
        if isinstance(s, ir.EvalStmt):
            visit(s.expr)
        if isinstance(s, ir.ReturnStmt):
            for e in s.value:
                visit(e)
        if isinstance(s, ir.SeqStmts):
            for sub in s.stmts:
                walk(sub)
        if isinstance(s, ir.ForStmt):
            walk(s.body)
        if isinstance(s, ir.IfStmt):
            walk(s.then_body)
            if s.else_body is not None:
                walk(s.else_body)

    walk(stmt)
    return found


def _get_func(program: ir.Program, name: str) -> ir.Function:
    gvar = program.get_global_var(name)
    assert gvar is not None
    return program.functions[gvar]


# ---------------------------------------------------------------------------
# pl.store on DistributedTensor — discard form (the user-reported case)
# ---------------------------------------------------------------------------


def test_store_discard_form_preserves_distributed_tensor_type():
    """``_ = pl.store(local, [0,0], data)`` keeps the Call typed as
    ``DistributedTensorType`` so SSA-versioned uses of the destination
    retain their comm-group binding."""

    @pl.program
    class P:
        @pl.function(type=pl.FunctionType.InCore)
        def kernel(
            self,
            inp: pl.Tensor[[1, 64], pl.FP32],
            out: pl.Out[pl.Tensor[[1, 64], pl.FP32]],
            data: pl.InOut[pld.DistributedTensor[[1, 64], pl.FP32]],
        ) -> pl.Tensor[[1, 64], pl.FP32]:
            local = pl.load(inp, [0, 0], [1, 64])
            _ = pl.store(local, [0, 0], data)
            return pl.store(local, [0, 0], out)

    func = _get_func(P, "kernel")
    stores = _collect_calls(func.body, "tile.store")
    # Two tile.store calls: one into ``data`` (the polymorphism case under
    # test), one into ``out`` (satisfies the function's return signature).
    assert len(stores) == 2, f"expected two tile.store calls, got {len(stores)}"
    # The store-into-DistributedTensor is the first one in source order.
    data_store = stores[0]
    assert isinstance(data_store.type, ir.DistributedTensorType), (
        f"tile.store on a DistributedTensor must return DistributedTensorType, "
        f"got {type(data_store.type).__name__}"
    )


# ---------------------------------------------------------------------------
# pl.store on DistributedTensor — capture form (functional SSA)
# ---------------------------------------------------------------------------


def test_store_capture_form_rebinds_distributed_tensor_var():
    """``data = pl.store(local, [0,0], data)`` lets the parser rebind ``data``
    because the wrapper's polymorphic return type matches the parameter type
    via ``_types_match`` — the rebound var stays a DistributedTensorType."""

    @pl.program
    class P:
        @pl.function(type=pl.FunctionType.InCore)
        def kernel(
            self,
            inp: pl.Tensor[[1, 64], pl.FP32],
            out: pl.Out[pl.Tensor[[1, 64], pl.FP32]],
            data: pl.InOut[pld.DistributedTensor[[1, 64], pl.FP32]],
        ) -> pl.Tensor[[1, 64], pl.FP32]:
            local = pl.load(inp, [0, 0], [1, 64])
            data = pl.store(local, [0, 0], data)
            return pl.store(local, [0, 0], out)

    func = _get_func(P, "kernel")
    stores = _collect_calls(func.body, "tile.store")
    assert len(stores) == 2
    # The capture-form store into ``data`` is first in source order.
    assert isinstance(stores[0].type, ir.DistributedTensorType)


# ---------------------------------------------------------------------------
# Non-distributed regression — plain Tensor must stay a plain Tensor
# ---------------------------------------------------------------------------


def test_store_plain_tensor_stays_tensor_type():
    """When the destination is a plain ``Tensor``, the Call's type must
    remain ``TensorType`` (not accidentally become DistributedTensorType)."""

    @pl.program
    class P:
        @pl.function(type=pl.FunctionType.InCore)
        def kernel(
            self,
            inp: pl.Tensor[[1, 64], pl.FP32],
            out: pl.Out[pl.Tensor[[1, 64], pl.FP32]],
        ) -> pl.Tensor[[1, 64], pl.FP32]:
            local = pl.load(inp, [0, 0], [1, 64])
            return pl.store(local, [0, 0], out)

    func = _get_func(P, "kernel")
    stores = _collect_calls(func.body, "tile.store")
    assert len(stores) == 1
    t = stores[0].type
    assert isinstance(t, ir.TensorType)
    assert not isinstance(t, ir.DistributedTensorType), (
        f"plain Tensor destination must not promote to DistributedTensorType, got {type(t).__name__}"
    )


# ---------------------------------------------------------------------------
# tensor.slice — β strategy: result keeps DistributedTensorType + window_buffer_
# ---------------------------------------------------------------------------


def test_tensor_slice_on_distributed_tensor_preserves_kind():
    """``pl.tensor.slice`` on a DistributedTensor returns a slice view that
    is still a DistributedTensorType — a slice is a view into the same
    comm-group window allocation."""

    @pl.program
    class P:
        @pl.function(type=pl.FunctionType.InCore)
        def kernel(
            self,
            out: pl.Out[pl.Tensor[[1, 32], pl.FP32]],
            data: pl.InOut[pld.DistributedTensor[[1, 64], pl.FP32]],
        ) -> pl.Tensor[[1, 32], pl.FP32]:
            sub = pl.tensor.slice(data, [1, 32], [0, 0])
            tile = pl.load(sub, [0, 0], [1, 32])
            return pl.store(tile, [0, 0], out)

    func = _get_func(P, "kernel")
    slices = _collect_calls(func.body, "tensor.slice")
    assert len(slices) == 1
    assert isinstance(slices[0].type, ir.DistributedTensorType)


# ---------------------------------------------------------------------------
# tensor.assemble — β strategy: target's DistributedTensorType propagates
# ---------------------------------------------------------------------------


def test_tensor_assemble_target_distributed_tensor_keeps_kind():
    """When the assemble target is a DistributedTensor, the result keeps
    the DistributedTensorType (and the target's window_buffer_)."""

    @pl.program
    class P:
        @pl.function(type=pl.FunctionType.InCore)
        def kernel(
            self,
            src: pl.Tensor[[1, 64], pl.FP32],
            out: pl.Out[pl.Tensor[[1, 64], pl.FP32]],
            data: pl.InOut[pld.DistributedTensor[[1, 64], pl.FP32]],
        ) -> pl.Tensor[[1, 64], pl.FP32]:
            data = pl.tensor.assemble(data, src, [0, 0])
            tile = pl.load(data, [0, 0], [1, 64])
            return pl.store(tile, [0, 0], out)

    func = _get_func(P, "kernel")
    assembles = _collect_calls(func.body, "tensor.assemble")
    assert len(assembles) == 1
    assert isinstance(assembles[0].type, ir.DistributedTensorType)


# ---------------------------------------------------------------------------
# Subscript-write sugar (``dst[i:j, :] = src``) on a DistributedTensor target.
# Lowers to ``tensor.assemble`` — accepts a plain Tensor source for local
# stage-in into a window slice (issue #1672).
# ---------------------------------------------------------------------------


def test_subscript_write_distributed_target_plain_source():
    """``dist_win[i:j, :] = plain_tensor_src`` lowers to ``tensor.assemble``
    with a DistributedTensor target and a plain Tensor source. The parser's
    kind check must accept the mixed kinds (both are tensor-shaped)."""

    @pl.program
    class P:
        @pl.function(type=pl.FunctionType.InCore)
        def kernel(
            self,
            src: pl.Tensor[[1, 32], pl.FP32],
            out: pl.Out[pl.Tensor[[1, 64], pl.FP32]],
            data: pl.InOut[pld.DistributedTensor[[1, 64], pl.FP32]],
        ) -> pl.Tensor[[1, 64], pl.FP32]:
            data[0:1, 0:32] = src
            tile = pl.load(data, [0, 0], [1, 64])
            return pl.store(tile, [0, 0], out)

    func = _get_func(P, "kernel")
    assembles = _collect_calls(func.body, "tensor.assemble")
    assert len(assembles) == 1
    # Target's DistributedTensorType propagates onto the assemble result.
    assert isinstance(assembles[0].type, ir.DistributedTensorType)


def test_subscript_write_plain_target_distributed_source_slice():
    """Reverse direction — plain Tensor target, DistributedTensor source slice."""

    @pl.program
    class P:
        @pl.function(type=pl.FunctionType.InCore)
        def kernel(
            self,
            src_data: pl.InOut[pld.DistributedTensor[[1, 64], pl.FP32]],
            out: pl.Out[pl.Tensor[[1, 64], pl.FP32]],
            plain: pl.InOut[pl.Tensor[[1, 64], pl.FP32]],
        ) -> pl.Tensor[[1, 64], pl.FP32]:
            plain[0:1, 0:32] = src_data[0:1, 0:32]
            tile = pl.load(plain, [0, 0], [1, 64])
            return pl.store(tile, [0, 0], out)

    func = _get_func(P, "kernel")
    assembles = _collect_calls(func.body, "tensor.assemble")
    assert len(assembles) == 1
    # Plain target -> plain TensorType result, despite the DistributedTensor source.
    t = assembles[0].type
    assert isinstance(t, ir.TensorType)
    assert not isinstance(t, ir.DistributedTensorType)


# ---------------------------------------------------------------------------
# PR #1528 narrow-write semantics compose with DistributedTensor (issue #1672).
# A source with ``static_shape`` padded for ISA alignment is accepted when
# its ``valid_shape`` matches the destination slot. The check (via
# ``_get_source_valid_shape``) uses ``isinstance(source_type, ir.TensorType)``
# which matches ``DistributedTensorType`` via inheritance, and the target
# side reads ``base_type.shape`` (inherited) — so the path works both ways.
# ---------------------------------------------------------------------------


def test_narrow_plain_source_into_distributed_target():
    """Narrow plain Tensor source (static [1,32] / valid [1,16]) lands in a
    16-wide slice of a DistributedTensor target (mirrors PR #1528 pattern
    with the destination promoted to a window-bound view)."""

    @pl.program
    class P:
        @pl.function(type=pl.FunctionType.InCore)
        def kernel(
            self,
            out: pl.Out[pl.Tensor[[1, 64], pl.FP32]],
            data: pl.InOut[pld.DistributedTensor[[1, 64], pl.FP32]],
        ) -> pl.Tensor[[1, 64], pl.FP32]:
            local_buf = pl.create_tensor([1, 32], dtype=pl.FP32)
            narrowed = pl.tensor.slice(local_buf, [1, 32], [0, 0], valid_shape=[1, 16])
            data[0:1, 0:16] = narrowed
            tile = pl.load(data, [0, 0], [1, 64])
            return pl.store(tile, [0, 0], out)

    func = _get_func(P, "kernel")
    assembles = _collect_calls(func.body, "tensor.assemble")
    assert len(assembles) == 1
    assert isinstance(assembles[0].type, ir.DistributedTensorType)


def test_narrow_distributed_source_into_plain_target():
    """A DistributedTensor sliced with ``valid_shape=[1, 16]`` is itself
    a narrow source, and writing it into a 16-wide slice of a plain Tensor
    target also lands cleanly — exercises ``_get_source_valid_shape`` on
    the DistributedTensorType path."""

    @pl.program
    class P:
        @pl.function(type=pl.FunctionType.InCore)
        def kernel(
            self,
            dist_data: pl.InOut[pld.DistributedTensor[[1, 64], pl.FP32]],
            out: pl.Out[pl.Tensor[[1, 64], pl.FP32]],
            plain: pl.InOut[pl.Tensor[[1, 64], pl.FP32]],
        ) -> pl.Tensor[[1, 64], pl.FP32]:
            narrow_dist = pl.tensor.slice(dist_data, [1, 32], [0, 0], valid_shape=[1, 16])
            plain[0:1, 0:16] = narrow_dist
            tile = pl.load(plain, [0, 0], [1, 64])
            return pl.store(tile, [0, 0], out)

    func = _get_func(P, "kernel")
    assembles = _collect_calls(func.body, "tensor.assemble")
    assert len(assembles) == 1
    # Plain target wins: result is plain TensorType, not DistributedTensorType.
    t = assembles[0].type
    assert isinstance(t, ir.TensorType)
    assert not isinstance(t, ir.DistributedTensorType)


# ---------------------------------------------------------------------------
# Unified facade (``pl.slice`` / ``pl.fillpad`` / ``pl.reshape`` / ...) routes
# DistributedTensor inputs to the tensor-side wrapper via ``isinstance``
# dispatch; the IR Call's kind must survive that detour.
# ---------------------------------------------------------------------------


def test_unified_pl_slice_dispatches_to_distributed_tensor_kind():
    """``pl.slice(dist_tensor, ...)`` (the unified facade in ``unified_ops``)
    routes through ``isinstance(input, Tensor)`` → ``_tensor.slice`` and must
    end up with a ``DistributedTensorType`` Call — equivalent to calling
    ``pl.tensor.slice`` directly."""

    @pl.program
    class P:
        @pl.function(type=pl.FunctionType.InCore)
        def kernel(
            self,
            out: pl.Out[pl.Tensor[[1, 32], pl.FP32]],
            data: pl.InOut[pld.DistributedTensor[[1, 64], pl.FP32]],
        ) -> pl.Tensor[[1, 32], pl.FP32]:
            sub = pl.slice(data, [1, 32], [0, 0])
            tile = pl.load(sub, [0, 0], [1, 32])
            return pl.store(tile, [0, 0], out)

    func = _get_func(P, "kernel")
    slices = _collect_calls(func.body, "tensor.slice")
    assert len(slices) == 1
    assert isinstance(slices[0].type, ir.DistributedTensorType)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
