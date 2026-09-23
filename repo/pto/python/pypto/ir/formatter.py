# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Code formatter for IR printer output using ruff."""

import subprocess


def ruff_format(code: str, line_length: int = 200) -> str:
    """Format a Python code string using ruff.

    Returns the original code unchanged on any failure (ruff not found, timeout,
    parse error), so it is safe to use as a best-effort post-processor.

    Args:
        code: Python source code to format
        line_length: Maximum line length (default 200)

    Returns:
        Formatted code, or original code on failure
    """
    try:
        result = subprocess.run(
            [
                "ruff",
                "format",
                "--line-length",
                str(line_length),
                "--stdin-filename",
                "ir_output.py",
                "-",
            ],
            check=False,
            input=code,
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            # ruff always appends a trailing newline; strip it to match the
            # raw printer output which may or may not end with one.
            formatted = result.stdout
            if formatted.endswith("\n") and not code.endswith("\n"):
                formatted = formatted[:-1]
            return formatted
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return code
