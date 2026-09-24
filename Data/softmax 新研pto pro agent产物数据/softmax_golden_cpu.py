#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2024-2026. All rights reserved.

"""PyPTO-Pro softmax CPU golden reference (higher precision, FP32).

Based on the NPU golden (softmax_golden.py) but executed on CPU in FP32.
Used by test_softmax.py for precision comparison (Stage 4 精度校验基准).

Differences vs NPU golden (softmax_golden.py):
  - No torch_npu import; pure CPU computation.
  - No device selection / .to(device).
  - FP16/BF16 input promoted to FP32, computed in FP32, **returned as FP32**
    (not downcast back to input dtype) for higher precision reference.

Golden operation inventory (identical to NPU golden, per SPEC formula):
  1. row max:        x[B,N] (fp16 -> fp32) -> m[B,1] (fp32)        # max_j(x[b,j])
  2. shift & exp:    (x_fp32 - m)            -> e[B,N] (fp32)      # exp(x[b,i]-m[b])
  3. row sum:        sum(e, dim=-1)          -> s[B,1] (fp32)      # sum_j(e[b,j])
  4. normalize:      e / s                   -> y[B,N] (fp32)       # e[b,i]/s[b]

No final cast back to fp16: the CPU golden returns FP32 so Stage 4's
precision comparison can compare the NPU fp16 output against a higher-precision
reference. max-subtract guarantees exp input <= 0 so no FP32 overflow on exp.
"""

import torch

# Canonical mathematical semantics copied verbatim from the SPEC machine contract.
_SPEC_FORMULA = 'y[b,i] = exp(x[b,i] - m[b]) / s[b]; m[b] = max_j(x[b,j]); s[b] = sum_j(exp(x[b,j] - m[b]))'


# ─────────────────────────────────────────────
# CPU Golden 参考实现（更高精度）
# ─────────────────────────────────────────────

def softmax_golden_cpu(x: torch.Tensor) -> torch.Tensor:
    """Numerically stable row-wise softmax over the last axis (axis=-1), FP32.

    Implements the SPEC machine-contract formula:
        m[b] = max_j(x[b, j])
        s[b] = sum_j(exp(x[b, j] - m[b]))
        y[b, i] = exp(x[b, i] - m[b]) / s[b]

    Pure torch on CPU. FP16/BF16 input is promoted to FP32, computation stays
    in FP32, and the output is returned as FP32 (NOT downcast) — this is the
    higher-precision reference for Stage 4 precision comparison.

    Args:
        x: input tensor of shape [B, N], any dtype (fp16 promoted to fp32).

    Returns:
        Output tensor of shape [B, N], dtype FP32. Each row sums to 1
        (within fp32 precision); each element lies in [0, 1].
    """
    # Step 0: promote fp16/bf16 input to fp32; do NOT downcast at the end.
    if x.dtype in (torch.float16, torch.bfloat16):
        x_fp32 = x.to(torch.float32)
    else:
        x_fp32 = x
    # Step 1: per-row max along axis=-1, keepdim for broadcast.
    m = x_fp32.max(dim=-1, keepdim=True).values
    # Step 2: shift & exp. (x - m) <= 0 so exp never overflows.
    e = torch.exp(x_fp32 - m)
    # Step 3: per-row sum of exp. s[b] >= 1 always (exp(0)=1 for the row max).
    s = e.sum(dim=-1, keepdim=True)
    # Step 4: normalize. Returns FP32 (no downcast).
    return e / s


# ==========================================
# 验证
# ==========================================

def _validate():
    """Validate softmax_golden_cpu: runability, FP32 output, shape, finite,
    value range, row-sum, oracle cross-check, boundary cases, and dynamic
    axis generalization. Mirrors the NPU golden's coverage."""
    print('=' * 60)
    print('softmax_golden_cpu validation report')
    print('=' * 60)

    ATOL = 1e-3
    RTOL = 1e-3
    all_pass = True

    def _check(label, cond, detail=""):
        nonlocal all_pass
        status = "PASS" if cond else "FAIL"
        print(f'  [{label}] {detail} ... {status}')
        if not cond:
            all_pass = False

    # -- Basic: FP32 input -> FP32 output, shape preserved, finite --
    print('\n--- 1. Basic runability (FP32 / FP16 / BF16 inputs) ---')
    x_f32 = torch.randn(4, 128, dtype=torch.float32)
    y_f32 = softmax_golden_cpu(x_f32)
    _check('fp32_in',
           y_f32.shape == x_f32.shape and y_f32.dtype == torch.float32
           and torch.isfinite(y_f32).all().item(),
           f'shape={tuple(y_f32.shape)} dtype={y_f32.dtype}')

    x_f16 = torch.randn(4, 128, dtype=torch.float16)
    y_f16 = softmax_golden_cpu(x_f16)
    _check('fp16_in',
           y_f16.shape == x_f16.shape and y_f16.dtype == torch.float32
           and torch.isfinite(y_f16).all().item(),
           f'shape={tuple(y_f16.shape)} dtype={y_f16.dtype} (promoted)')

    if hasattr(torch, 'bfloat16'):
        x_bf16 = torch.randn(4, 128, dtype=torch.bfloat16)
        y_bf16 = softmax_golden_cpu(x_bf16)
        _check('bf16_in',
               y_bf16.shape == x_bf16.shape and y_bf16.dtype == torch.float32
               and torch.isfinite(y_bf16).all().item(),
               f'shape={tuple(y_bf16.shape)} dtype={y_bf16.dtype} (promoted)')

    # -- 2. SPEC P0 cases (FP16 input, like the NPU golden) --
    print('\n--- 2. SPEC P0 cases (FP16 input -> FP32 output) ---')
    p0_shapes = [(1, 128), (4, 2048), (32, 4096)]
    p0_names = ['p0_batch1_n128', 'p0_batch4_n2048', 'p0_batch32_n4096']
    for name, shape in zip(p0_names, p0_shapes):
        x = torch.randn(shape, dtype=torch.float16)
        y = softmax_golden_cpu(x)
        shape_ok = y.shape == shape
        dtype_ok = y.dtype == torch.float32
        finite_ok = torch.isfinite(y).all().item()
        # Value range [0, 1].
        y_min, y_max = y.min().item(), y.max().item()
        range_ok = (y_min >= 0.0) and (y_max <= 1.0)
        # Row-sum == 1 (fp32 precision, tight bound).
        row_dev = (y.sum(dim=-1) - 1.0).abs().max().item()
        sum_ok = row_dev <= 1e-5
        # Oracle cross-check: torch.softmax in fp32.
        oracle = torch.softmax(x.to(torch.float32), dim=-1)
        diff = (y - oracle).abs().max().item()
        oracle_ok = diff <= ATOL + RTOL * oracle.abs().max().item()
        _check(name,
               shape_ok and dtype_ok and finite_ok and range_ok and sum_ok and oracle_ok,
               f'shape={tuple(y.shape)} dtype={y.dtype} range=[{y_min:.6f},{y_max:.6f}] '
               f'row_dev={row_dev:.2e} oracle_diff={diff:.2e}')

    # -- 3. Boundary / special cases (from SPEC §9) ---
    print('\n--- 3. Boundary / special cases ---')

    # N=1: single-element row -> y = 1.0.
    x_n1 = torch.randn(4, 1, dtype=torch.float16)
    y_n1 = softmax_golden_cpu(x_n1)
    _check('N=1', torch.allclose(y_n1, torch.ones_like(y_n1), atol=ATOL, rtol=RTOL),
           f'y={y_n1.tolist()}')

    # All-same row -> y = 1/N.
    n_u = 8
    x_u = torch.full((2, n_u), 0.5, dtype=torch.float16)
    y_u = softmax_golden_cpu(x_u)
    exp_u = torch.full((2, n_u), 1.0 / n_u, dtype=torch.float32)
    _check('all_same_row', torch.allclose(y_u, exp_u, atol=ATOL, rtol=RTOL),
           f'1/N={1.0/n_u:.6f}')

    # Large positive (65504, no overflow).
    x_p = torch.full((2, 64), 65504.0, dtype=torch.float16)
    y_p = softmax_golden_cpu(x_p)
    _check('large_pos',
           torch.isfinite(y_p).all().item() and torch.allclose(y_p, torch.full((2, 64), 1.0/64), atol=ATOL, rtol=RTOL),
           f'finite={torch.isfinite(y_p).all().item()}')

    # Large negative (-65504, exp underflow to 0, no NaN).
    x_n = torch.full((2, 64), -65504.0, dtype=torch.float16)
    y_n = softmax_golden_cpu(x_n)
    _check('large_neg',
           torch.isfinite(y_n).all().item() and torch.allclose(y_n, torch.full((2, 64), 1.0/64), atol=ATOL, rtol=RTOL),
           f'finite={torch.isfinite(y_n).all().item()}')

    # One-hot mix: one element = 65504, rest = -65504 -> ~one-hot.
    B_oh, N_oh = 3, 16
    x_oh = torch.full((B_oh, N_oh), -65504.0, dtype=torch.float16)
    for b in range(B_oh):
        x_oh[b, b % N_oh] = 65504.0
    y_oh = softmax_golden_cpu(x_oh)
    oh_finite = torch.isfinite(y_oh).all().item()
    argmax_ok = all((y_oh[b].argmax().item() == b % N_oh) for b in range(B_oh))
    near_one = torch.allclose(y_oh.max(dim=-1).values, torch.ones(B_oh), atol=ATOL, rtol=RTOL)
    _check('one_hot_mix', oh_finite and argmax_ok and near_one,
           f'finite={oh_finite} argmax_ok={argmax_ok} max~1={near_one}')

    # -- 4. Dynamic axis generalization (B in [1,1024], N in [1,8192]) --
    print('\n--- 4. Dynamic axis generalization ---')
    for B in (1, 16, 1024):
        x = torch.randn((B, 128), dtype=torch.float16)
        y = softmax_golden_cpu(x)
        ok = (y.shape == (B, 128) and y.dtype == torch.float32
              and torch.isfinite(y).all().item()
              and (y.sum(dim=-1) - 1.0).abs().max().item() <= 1e-5)
        _check(f'B={B},N=128', ok, f'shape={tuple(y.shape)} dtype={y.dtype}')
    for N in (1, 256, 8192):
        x = torch.randn((4, N), dtype=torch.float16)
        y = softmax_golden_cpu(x)
        ok = (y.shape == (4, N) and y.dtype == torch.float32
              and torch.isfinite(y).all().item()
              and (y.sum(dim=-1) - 1.0).abs().max().item() <= 1e-5)
        _check(f'B=4,N={N}', ok, f'shape={tuple(y.shape)} dtype={y.dtype}')
    # All-low (1,1).
    x_ll = torch.randn((1, 1), dtype=torch.float16)
    y_ll = softmax_golden_cpu(x_ll)
    _check('all_low(1,1)', torch.allclose(y_ll, torch.ones(1, 1), atol=ATOL, rtol=RTOL),
           f'y={y_ll.tolist()}')
    # All-high (1024,8192).
    x_hh = torch.randn((1024, 8192), dtype=torch.float16)
    y_hh = softmax_golden_cpu(x_hh)
    ok = (y_hh.shape == (1024, 8192) and y_hh.dtype == torch.float32
          and torch.isfinite(y_hh).all().item()
          and (y_hh.sum(dim=-1) - 1.0).abs().max().item() <= 1e-5)
    _check('all_high(1024,8192)', ok, f'shape={tuple(y_hh.shape)} dtype={y_hh.dtype}')

    print('\n' + '=' * 60)
    if all_pass:
        print('所有验证通过')
    else:
        print('存在验证失败项')
    print('=' * 60)
    return all_pass


if __name__ == "__main__":
    ok = _validate()
    if not ok:
        exit(1)
