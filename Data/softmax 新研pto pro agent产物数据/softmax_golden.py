#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2024-2026. All rights reserved.

"""PyPTO-Pro softmax golden reference implementation.

基于 torch + torch_npu 的 NPU 参考实现，计算在 NPU 上执行。
torch_npu 未安装时直接报错引导安装；仅无 NPU 硬件时回退 CPU。
导出函数 softmax_golden() 供 test_softmax.py 调用。

Golden operation inventory (per SPEC machine contract formula):
  1. row max:        x[B,N] (fp16 -> fp32) -> m[B,1] (fp32)        # max_j(x[b,j])
  2. shift & exp:    (x_fp32 - m)            -> e[B,N] (fp32)      # exp(x[b,i]-m[b])
  3. row sum:        sum(e, dim=-1)          -> s[B,1] (fp32)      # sum_j(e[b,j])
  4. normalize:      e / s                   -> y_fp32[B,N] (fp32) # e[b,i]/s[b]
  5. cast output:    y_fp32 -> fp16          -> y[B,N] (fp16)

Cast chain follows SPEC kernel-contract supplement: fp16(in) -> fp32(acc) ->
fp16(out). fp32 accumulation is a strong recommendation for fp16 tolerance;
max-subtract guarantees exp input <= 0 so no fp16/fp32 overflow on exp.
"""

import os
import torch

_DEVICE = None
_HAS_NPU = False

try:
    import torch_npu
    _HAS_NPU = torch.npu.is_available() and torch.npu.device_count() > 0
except ImportError:
    raise ImportError(
        "torch_npu is not installed. Please install it following the torch_npu official guide:\n"
        "  pip install torch_npu\n"
        "Refer to the CANN and PyPTO/PyPTO-Pro installation docs for the full NPU environment setup."
    )

def _get_device() -> torch.device:
    global _DEVICE
    if _DEVICE is None:
        if not _HAS_NPU:
            _DEVICE = torch.device("cpu")
        else:
            device_id = int(os.environ.get("TILE_FWK_DEVICE_ID", "0"))
            torch.npu.set_device(device_id)
            _DEVICE = torch.device(f"npu:{device_id}")
    return _DEVICE


# ─────────────────────────────────────────────
# Golden 参考实现（NPU torch）
# ─────────────────────────────────────────────

# Canonical mathematical semantics copied verbatim from the SPEC machine contract.
_SPEC_FORMULA = 'y[b,i] = exp(x[b,i] - m[b]) / s[b]; m[b] = max_j(x[b,j]); s[b] = sum_j(exp(x[b,j] - m[b]))'

def softmax_golden(
    x: torch.Tensor,
) -> torch.Tensor:
    """Numerically stable row-wise softmax over the last axis (axis=-1).

    Implements the SPEC machine-contract formula:
        m[b] = max_j(x[b, j])
        s[b] = sum_j(exp(x[b, j] - m[b]))
        y[b, i] = exp(x[b, i] - m[b]) / s[b]

    Only torch + torch_npu are used. dtype follows the input (fp16 -> fp16);
    accumulation happens in fp32 for numerical stability (max-subtract,
    exp, sum, div), matching the SPEC kernel-contract cast chain
    fp16(in) -> fp32(acc) -> fp16(out).

    Args:
        x: input tensor of shape [B, N], dtype float16.

    Returns:
        Output tensor of shape [B, N], dtype float16. Each row sums to 1
        (within fp16 precision); each element lies in [0, 1].
    """
    device = _get_device()
    x = x.to(device)
    # Step 0: promote to fp32 for stable accumulation (SPEC strong recommendation).
    x_fp32 = x.to(torch.float32)
    # Step 1: per-row max along axis=-1, keepdim for broadcast.
    m = x_fp32.max(dim=-1, keepdim=True).values
    # Step 2: shift & exp. m - (x - m) <= 0 so exp never overflows.
    e = torch.exp(x_fp32 - m)
    # Step 3: per-row sum of exp. s[b] >= 1 always (exp(0)=1 for the row max).
    s = e.sum(dim=-1, keepdim=True)
    # Step 4: normalize.
    y_fp32 = e / s
    # Step 5: cast back to SPEC output dtype (fp16).
    return y_fp32.to(torch.float16)


# ==========================================
# 输入构造（供验证使用；启用性能采集时供 profiling --factory 复用）
# ==========================================

def _make_inputs(device):
    """Construct every P0 case from the validated SPEC machine contract.

    Each case builds the single input tensor x of shape [B, N] dtype fp16.
    randn on fp16 stays well within the SPEC value range [-65504, 65504]
    (statistical range ~[-6, 6]); any finite fp16 logit is legal for softmax.
    All tensors are created with device=device so they live on the target device.
    """
    cases = []
    # P0 case: p0_batch1_n128
    x = torch.randn((1, 128,), dtype=torch.float16, device=device)
    args = [x]
    kwargs = {}
    cases.append(("p0_batch1_n128", args, kwargs))
    # P0 case: p0_batch4_n2048
    x = torch.randn((4, 2048,), dtype=torch.float16, device=device)
    args = [x]
    kwargs = {}
    cases.append(("p0_batch4_n2048", args, kwargs))
    # P0 case: p0_batch32_n4096
    x = torch.randn((32, 4096,), dtype=torch.float16, device=device)
    args = [x]
    kwargs = {}
    cases.append(("p0_batch32_n4096", args, kwargs))
    return cases

# ==========================================
# 验证
# ==========================================

def _validate():
    """Validate softmax_golden against the SPEC machine contract.

    Coverage (per pypto-pro-golden-generate skill §4):
      1. All contract P0 cases (shape, dtype, finite, value range [0,1],
         row-sum == 1, cross-check vs torch.softmax oracle in fp32).
      2. Boundary / special cases derivable from _SPEC_FORMULA:
         N=1 (single-element row -> output 1.0), all-same row (output 1/N),
         large positive (65504, no overflow), large negative (-65504,
         exp underflow to 0, no NaN), one-hot mix (large positive +
         large negative -> one-hot row).
      3. Dynamic axis generalization (B in [1,1024], N in [1,8192]):
         low/mid/high for each axis, all-low, all-high, and the SPEC P0
         shapes sit inside this envelope.
    """
    device = _get_device()
    print('=' * 60)
    print('softmax_golden validation report')
    print('=' * 60)
    print(f'Device: {device}')

    ATOL = 1e-3
    RTOL = 1e-3
    expected_case_names = ('p0_batch1_n128', 'p0_batch4_n2048', 'p0_batch32_n4096')
    # SPEC P0 output shapes, in the same order as expected_case_names.
    expected_output_shapes = {(1, 128), (4, 2048), (32, 4096)}

    raw_cases = _make_inputs(device)
    if (isinstance(raw_cases, tuple) and len(raw_cases) == 2
            and isinstance(raw_cases[0], list) and isinstance(raw_cases[1], dict)):
        cases = [(expected_case_names[0], raw_cases[0], raw_cases[1])]
    else:
        cases = raw_cases
    assert isinstance(cases, list) and cases, '_make_inputs returned no cases'
    observed_names = [case[0] for case in cases]
    assert len(observed_names) == len(set(observed_names)), 'duplicate case names'
    missing = [name for name in expected_case_names if name not in observed_names]
    assert not missing, f'missing contract P0 cases: {missing}'
    unexpected = [name for name in observed_names if name not in expected_case_names]
    assert not unexpected, f'unexpected P0 cases: {unexpected}'

    def _check_p0_case(case_name, x):
        """Run golden + oracle on a P0 case and assert all properties."""
        print(f'\n[case: {case_name}]')
        y = softmax_golden(x)
        outs = (y,)
        # Finiteness.
        print('[finiteness]')
        for i, o in enumerate(outs):
            ok = torch.isfinite(o).all().item()
            print(f'  out[{i}] shape={tuple(o.shape)} finite={ok} '
                  f'... {"PASS" if ok else "FAIL"}')
            assert ok, f'{case_name} out[{i}] contains NaN/Inf'
        # Output shape == input shape (SPEC: y same shape as x).
        assert y.shape == x.shape, (
            f'{case_name} shape mismatch: y={tuple(y.shape)} vs x={tuple(x.shape)}')
        # Output dtype == fp16 (SPEC machine contract).
        assert y.dtype == torch.float16, (
            f'{case_name} dtype mismatch: y={y.dtype} vs expected fp16')
        print(f'[shape/dtype] y shape={tuple(y.shape)} dtype={y.dtype} ... PASS')
        # Value range [0, 1] (softmax probability semantics).
        y_min = y.to(torch.float32).min().item()
        y_max = y.to(torch.float32).max().item()
        range_ok = (y_min >= 0.0) and (y_max <= 1.0)
        print(f'[value_range] min={y_min:.6f} max={y_max:.6f} '
              f'... {"PASS" if range_ok else "FAIL"}')
        assert range_ok, f'{case_name} value range violated: [{y_min}, {y_max}]'
        # Row-sum == 1 (within fp16 precision; allow 2*fp16_eps slack).
        row_sums = y.to(torch.float32).sum(dim=-1)
        max_dev = (row_sums - 1.0).abs().max().item()
        sum_ok = max_dev <= 1e-2  # generous; fp16 row-sum drift is bounded
        print(f'[row_sum] max |sum-1|={max_dev:.3e} '
              f'... {"PASS" if sum_ok else "FAIL"}')
        assert sum_ok, f'{case_name} row sum deviates from 1: {max_dev}'
        # Cross-check vs torch.softmax oracle in fp32 on CPU (the official
        # sample's golden recipe; see pro_ops/vf_api/test_softmax_tile_group_vf.py
        # line 168: torch.softmax(x.cpu().float(), dim=-1)). Computing the
        # oracle on CPU makes it independent of any NPU softmax op precision.
        oracle = torch.softmax(x.to(torch.float32).cpu(), dim=-1).to(torch.float16)
        torch.testing.assert_close(
            y.to(torch.float32).cpu(), oracle.to(torch.float32),
            rtol=RTOL, atol=ATOL,
            msg=f'{case_name} mismatch vs torch.softmax oracle')
        diff = (y.to(torch.float32).cpu() - oracle.to(torch.float32)).abs().max().item()
        print(f'[oracle] max |y - torch.softmax|={diff:.3e} '
              f'(rtol={RTOL}, atol={ATOL}) ... PASS')
        return y

    # -- 1. All contract P0 cases --
    print('\n--- 1. Contract P0 cases ---')
    for case_name, args, kwargs in cases:
        assert isinstance(case_name, str) and case_name
        assert isinstance(args, list) and isinstance(kwargs, dict)
        assert len(args) == 1, f'{case_name}: expected 1 tensor arg, got {len(args)}'
        _check_p0_case(case_name, args[0])

    # -- 2. Boundary / special cases (from SPEC §9 边界条件) --
    print('\n--- 2. Boundary / special cases ---')

    # N=1: single-element row -> m=x, exp(0)=1, s=1, y=1.
    x_n1 = torch.randn((4, 1), dtype=torch.float16, device=device)
    y_n1 = softmax_golden(x_n1)
    assert y_n1.shape == (4, 1) and y_n1.dtype == torch.float16
    ones_ok = torch.allclose(y_n1.to(torch.float32), torch.ones_like(y_n1, dtype=torch.float32), atol=ATOL, rtol=RTOL)
    print(f'[N=1] y shape={tuple(y_n1.shape)} all-ones={ones_ok} ... '
          f'{"PASS" if ones_ok else "FAIL"}')
    assert ones_ok, f'N=1 case: expected all ones, got {y_n1}'

    # All-same row -> y = 1/N for every element.
    n_uniform = 8
    x_uniform = torch.full((2, n_uniform), 0.5, dtype=torch.float16, device=device)
    y_uniform = softmax_golden(x_uniform)
    expected_uniform = torch.full((2, n_uniform), 1.0 / n_uniform, dtype=torch.float32, device=device)
    uniform_ok = torch.allclose(y_uniform.to(torch.float32), expected_uniform, atol=ATOL, rtol=RTOL)
    print(f'[all_same_row] y shape={tuple(y_uniform.shape)} '
          f'uniform={uniform_ok} (1/N={1.0/n_uniform:.6f}) ... '
          f'{"PASS" if uniform_ok else "FAIL"}')
    assert uniform_ok, f'all-same row case: expected uniform 1/N, got {y_uniform}'

    # Large positive values (max-subtract -> exp(0)=1, no overflow).
    x_pos = torch.full((2, 64), 65504.0, dtype=torch.float16, device=device)
    y_pos = softmax_golden(x_pos)
    pos_ok = torch.isfinite(y_pos).all().item() and torch.allclose(
        y_pos.to(torch.float32), torch.full((2, 64), 1.0/64, dtype=torch.float32, device=device), atol=ATOL, rtol=RTOL)
    print(f'[large_pos] all 65504, finite={torch.isfinite(y_pos).all().item()} '
          f'uniform={pos_ok} ... {"PASS" if pos_ok else "FAIL"}')
    assert pos_ok, f'large positive case failed: {y_pos}'

    # Large negative values (exp underflows to 0, no NaN; all-same -> 1/N).
    x_neg = torch.full((2, 64), -65504.0, dtype=torch.float16, device=device)
    y_neg = softmax_golden(x_neg)
    neg_ok = torch.isfinite(y_neg).all().item() and torch.allclose(
        y_neg.to(torch.float32), torch.full((2, 64), 1.0/64, dtype=torch.float32, device=device), atol=ATOL, rtol=RTOL)
    print(f'[large_neg] all -65504, finite={torch.isfinite(y_neg).all().item()} '
          f'uniform={neg_ok} ... {"PASS" if neg_ok else "FAIL"}')
    assert neg_ok, f'large negative case failed: {y_neg}'

    # One-hot mix: one element = large positive, rest = large negative.
    # exp(0)=1 for the max element, exp(very_negative) -> 0 -> y ~ one-hot.
    B_oh, N_oh = 3, 16
    x_oh = torch.full((B_oh, N_oh), -65504.0, dtype=torch.float16, device=device)
    for b in range(B_oh):
        x_oh[b, b % N_oh] = 65504.0
    y_oh = softmax_golden(x_oh)
    oh_finite = torch.isfinite(y_oh).all().item()
    # Each row's argmax should be the one we set.
    argmax_ok = all((y_oh[b].to(torch.float32).argmax().item() == b % N_oh) for b in range(B_oh))
    # Each row's max should be ~1 (one-hot-ish).
    row_maxes = y_oh.to(torch.float32).max(dim=-1).values
    near_one_ok = torch.allclose(row_maxes, torch.ones_like(row_maxes), atol=ATOL, rtol=RTOL)
    print(f'[one_hot_mix] finite={oh_finite} argmax_correct={argmax_ok} '
          f'max~1={near_one_ok} ... '
          f'{"PASS" if (oh_finite and argmax_ok and near_one_ok) else "FAIL"}')
    assert oh_finite and argmax_ok and near_one_ok, f'one-hot mix case failed: {y_oh}'

    # -- 3. Dynamic axis generalization (B in [1,1024], N in [1,8192]) --
    print('\n--- 3. Dynamic axis generalization ---')
    # B low/mid/high (N fixed at 128, a mid representative value).
    for B in (1, 16, 1024):
        xg = torch.randn((B, 128), dtype=torch.float16, device=device)
        yg = softmax_golden(xg)
        assert yg.shape == (B, 128) and yg.dtype == torch.float16
        assert torch.isfinite(yg).all().item()
        rs_ok = (yg.to(torch.float32).sum(dim=-1) - 1.0).abs().max().item() <= 1e-2
        print(f'[B={B}, N=128] shape={tuple(yg.shape)} finite=True '
              f'row_sum_ok={rs_ok} ... {"PASS" if rs_ok else "FAIL"}')
        assert rs_ok, f'B={B} row sum failed'
    # N low/mid/high (B fixed at 4).
    for N in (1, 256, 8192):
        xg = torch.randn((4, N), dtype=torch.float16, device=device)
        yg = softmax_golden(xg)
        assert yg.shape == (4, N) and yg.dtype == torch.float16
        assert torch.isfinite(yg).all().item()
        rs_ok = (yg.to(torch.float32).sum(dim=-1) - 1.0).abs().max().item() <= 1e-2
        print(f'[B=4, N={N}] shape={tuple(yg.shape)} finite=True '
              f'row_sum_ok={rs_ok} ... {"PASS" if rs_ok else "FAIL"}')
        assert rs_ok, f'N={N} row sum failed'
    # All-low corner (1, 1): single row, single element -> output [[1.0]].
    x_ll = torch.randn((1, 1), dtype=torch.float16, device=device)
    y_ll = softmax_golden(x_ll)
    ll_ok = torch.allclose(y_ll.to(torch.float32), torch.ones_like(y_ll, dtype=torch.float32), atol=ATOL, rtol=RTOL)
    print(f'[all_low (1,1)] y={y_ll.to(torch.float32).tolist()} '
          f'ones={ll_ok} ... {"PASS" if ll_ok else "FAIL"}')
    assert ll_ok, f'all-low (1,1) failed: {y_ll}'
    # All-high corner (1024, 8192): upper bound of both dynamic axes.
    x_hh = torch.randn((1024, 8192), dtype=torch.float16, device=device)
    y_hh = softmax_golden(x_hh)
    hh_finite = torch.isfinite(y_hh).all().item()
    hh_shape = y_hh.shape == (1024, 8192)
    hh_sum = (y_hh.to(torch.float32).sum(dim=-1) - 1.0).abs().max().item() <= 1e-2
    print(f'[all_high (1024,8192)] shape={tuple(y_hh.shape)} '
          f'finite={hh_finite} row_sum_ok={hh_sum} ... '
          f'{"PASS" if (hh_shape and hh_finite and hh_sum) else "FAIL"}')
    assert hh_shape and hh_finite and hh_sum, f'all-high (1024,8192) failed'

    print('\n' + '=' * 60)
    print('validation complete')
    print('=' * 60)


if __name__ == "__main__":
    _validate()