# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import pytest
from pypto import ir


def test_inline_stmt_creation():
    span = ir.Span("test.py", 1, 1)
    stmt = ir.InlineStmt("z = x + y\nreturn z", ir.InlineLanguage.Python, span)

    assert isinstance(stmt, ir.InlineStmt)
    assert isinstance(stmt, ir.Stmt)
    assert stmt.body == "z = x + y\nreturn z"
    assert stmt.language == ir.InlineLanguage.Python


def test_inline_stmt_structural_equal():
    span_a = ir.Span("a.py", 1, 1)
    span_b = ir.Span("b.py", 2, 2)
    a = ir.InlineStmt("pass", ir.InlineLanguage.Python, span_a)
    b = ir.InlineStmt("pass", ir.InlineLanguage.Python, span_b)
    ir.assert_structural_equal(a, b)


def test_inline_stmt_not_equal_when_body_differs():
    span = ir.Span("test.py", 1, 1)
    a = ir.InlineStmt("foo()", ir.InlineLanguage.Python, span)
    b = ir.InlineStmt("bar()", ir.InlineLanguage.Python, span)
    assert not ir.structural_equal(a, b)


def test_inline_stmt_structural_hash():
    span_a = ir.Span("a.py", 1, 1)
    span_b = ir.Span("b.py", 2, 2)
    a = ir.InlineStmt("pass", ir.InlineLanguage.Python, span_a)
    b = ir.InlineStmt("pass", ir.InlineLanguage.Python, span_b)
    assert ir.structural_hash(a) == ir.structural_hash(b)


def test_inline_stmt_serialization_roundtrip():
    span = ir.Span("test.py", 1, 1)
    stmt = ir.InlineStmt("z = x + y\nreturn z", ir.InlineLanguage.Python, span)

    data = ir.serialize(stmt)
    restored = ir.deserialize(data)

    assert isinstance(restored, ir.InlineStmt)
    assert restored.body == stmt.body
    assert restored.language == stmt.language
    ir.assert_structural_equal(stmt, restored)


def test_inline_stmt_immutability():
    span = ir.Span("test.py", 1, 1)
    stmt = ir.InlineStmt("pass", ir.InlineLanguage.Python, span)
    with pytest.raises(AttributeError):
        stmt.body = "other"  # type: ignore[misc]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
