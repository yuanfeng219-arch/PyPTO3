#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2024-2026. All rights reserved.

"""PyPTO-Pro softmax kernel — fp16 input, fp32 accumulation, fp16 output.

Implements numerical-stability row-wise softmax over the last axis (axis=-1)
for 2D input x[B, N] (float16) → y[B, N] (float16).

Design contract: custom/softmax/DESIGN.md
  - Pure Vector (VF) implementation, single @pl.jit kernel + single @pl.vector_function
  - 3-pass per row: pass1 row max (fp16 exact) → pass2 exp+sum (fp16→fp32 widen,
    fp32 reduce_sum) → pass3 div+astype+store (fp32 div → fp32→fp16 narrow → store)
  - fp16→fp32 widen via vf.exp_sub(dtype=DT_FP32, layout=ZERO/ONE): each call
    processes 64 of 128 fp16 elements (even/odd lanes); ZERO+ONE cover the row
  - fp32→fp16 narrow via vf.astype(dtype=DT_FP16, layout=ZERO/ONE): even/odd
    fp16 slots reassembled by vf.add
  - Double-buffered in/out tile groups (make_tile_group + auto_mutex)
  - Multicore strided row-tile loop; row-tail via set_validshape;
    column-tail via update_mask (mask-before-max/reduce)
  - m (row max) kept fp16 (exact); s (row sum) accumulated fp32

Cast chain (SPEC kernel-contract): fp16(in) → fp32(acc) → fp16(out).
max-subtract guarantees exp input ≤ 0 (no overflow); s ≥ 1 (no div-by-zero).
"""

import logging
import os

import torch
import torch_npu  # noqa: F401  (required for torch.npu device)

import pypto_pro.language as pl
from pypto_pro.language import Vf as vf  # noqa: N813
from pypto_pro.runtime.platform import get_platform_info

logging.basicConfig(level=logging.INFO, format="%(message)s")


# ================================================================
# Constants (DESIGN.md §2.1 — coder must use verbatim)
# ================================================================
LANES = 128          # fp16 VF register width (elements per register)
MAX_N = 8192         # UB tile column physical size (multiple of LANES, covers N∈[1,8192])
TILE_ROWS = 3        # UB tile row physical size (double-buffer in+out ≤ 248KB UB)
NEG_INF = -1e30      # reduce_max finite negative sentinel (float('inf') does not compile)
SLOT_BYTES = TILE_ROWS * MAX_N * 2   # fp16 single slot = 3*8192*2 = 49152 B (48KB)

# UB addresses (32-byte aligned; 49152 = 1536*32)
VA_IN0 = 0x00000
VA_IN1 = VA_IN0 + SLOT_BYTES    # 0x0C000 (49152)
VA_OUT0 = VA_IN1 + SLOT_BYTES   # 0x18000 (98304)
VA_OUT1 = VA_OUT0 + SLOT_BYTES  # 0x24000 (147456)
# END = VA_OUT1 + SLOT_BYTES    # 0x30000 (196608) = 192KB ≤ 248KB UB


# ================================================================
# Precision comparison helper (dev-only imports inside function body)
# ================================================================
def _assert_precision(actual, *inputs, label="", **kwargs):
    """Compare NPU output against CPU FP32 golden via precision_compare.

    Imports precision_compare and softmax_golden_cpu inside the function body
    so the delivery module (test_softmax.py + softmax_golden.py) can be safely
    imported without the dev-only tools.
    """
    from precision_compare import check_precision
    from softmax_golden_cpu import softmax_golden_cpu
    inputs_cpu = [i.cpu() if hasattr(i, "cpu") else i for i in inputs]
    golden = softmax_golden_cpu(*inputs_cpu, **kwargs)
    actual_cpu = actual.cpu() if hasattr(actual, "cpu") else actual
    passed, summary = check_precision(actual_cpu, golden)
    if not passed:
        raise AssertionError(f"精度不达标: {summary}")
    if label:
        logging.info("[{}] PASS ({})".format(label, summary))
    return summary


# ================================================================
# VF function — 3-pass row softmax (single @pl.vector_function)
# ================================================================
@pl.vector_function
def softmax_rows_vf(in_tile, out_tile, n_rows: pl.DT_INT64, n_cols: pl.DT_INT64):
    """Row-wise softmax over n_cols columns of each of n_rows rows.

    Each row spans n_regs = ceil(n_cols / 128) fp16 registers inside the
    [TILE_ROWS, MAX_N] UB tile. Row m starts at element offset m * MAX_N.

    Three passes over the same in_tile registers (m/s are VF-local register
    scalars, broadcast via vf.full; no UB scratch round-trip):

      pass1: row max  — fp16 exact reduce_max across n_regs, vf.max combine
      pass2: row sum   — exp_sub fp16→fp32 (ZERO/ONE) + reduce_sum fp32 + vf.add
      pass3: div+store — exp_sub fp16→fp32 + div fp32 + astype fp32→fp16 + add + store

    Stage 5 optimization: full-register masks (128 lanes) are lifted out of the
    register loop and computed once per VF call.  The tail register (when
    n_cols % LANES != 0) is handled by a separate single-iteration loop whose
    range is empty when n_cols is an exact multiple of LANES — no runtime ``if``
    is needed.  This eliminates per-register update_mask scalar overhead for
    the common case where N is a multiple of 128 (all P0 cases).
    """
    # Two mask families: fp16 (128 elem × 2bit) and fp32 (64 elem × 4bit)
    preg_fp16 = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP16)
    preg_fp32 = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)

    n_full = n_cols // LANES          # count of fully-populated 128-lane registers
    n_tail = n_cols - n_full * LANES  # live lanes in the (optional) tail register
    # 0 iterations when n_tail==0 (exact multiple), 1 iteration otherwise
    n_tail_iters = (n_tail + LANES - 1) // LANES

    # Lifted full-register masks: identical for every full register, computed once
    full_mreg_fp16 = vf.update_mask(LANES, dtype=pl.DT_FP16)
    full_mreg_fp32_z = vf.update_mask((LANES + 1) // 2, dtype=pl.DT_FP32)   # = 64
    full_mreg_fp32_o = vf.update_mask(LANES // 2, dtype=pl.DT_FP32)         # = 64

    for m in pl.range(0, n_rows):
        base = m * MAX_N

        # ---- pass1: row max (fp16 exact; max of fp16 is one of the inputs) ----
        row_max = vf.full(NEG_INF, preg_fp16, dtype=pl.DT_FP16)
        for r in pl.range(0, n_full):
            reg = vf.load_align(in_tile, base + r * LANES)
            part = vf.reduce_max(reg, full_mreg_fp16)
            row_max = vf.max(row_max, part, preg_fp16)
        for r in pl.range(n_full, n_full + n_tail_iters):
            tail_valid = n_cols - r * LANES
            tail_mreg_fp16 = vf.update_mask(tail_valid, dtype=pl.DT_FP16)
            reg = vf.load_align(in_tile, base + r * LANES)
            part = vf.reduce_max(reg, tail_mreg_fp16)
            row_max = vf.max(row_max, part, preg_fp16)
        row_max_b = vf.full(row_max, preg_fp16)

        # ---- pass2: row sum-of-exp (fp16→fp32 widen via exp_sub, fp32 reduce_sum) ----
        row_sum = vf.full(0.0, preg_fp32, dtype=pl.DT_FP32)
        for r in pl.range(0, n_full):
            reg = vf.load_align(in_tile, base + r * LANES)
            e_even = vf.exp_sub(reg, row_max_b, full_mreg_fp16, dtype=pl.DT_FP32, layout=pl.CastLayout.ZERO)
            e_odd = vf.exp_sub(reg, row_max_b, full_mreg_fp16, dtype=pl.DT_FP32, layout=pl.CastLayout.ONE)
            part_even = vf.reduce_sum(e_even, full_mreg_fp32_z)
            row_sum = vf.add(row_sum, part_even, preg_fp32)
            part_odd = vf.reduce_sum(e_odd, full_mreg_fp32_o)
            row_sum = vf.add(row_sum, part_odd, preg_fp32)
        for r in pl.range(n_full, n_full + n_tail_iters):
            tail_valid = n_cols - r * LANES
            tail_mreg_fp16 = vf.update_mask(tail_valid, dtype=pl.DT_FP16)
            tail_mreg_fp32_z = vf.update_mask((tail_valid + 1) // 2, dtype=pl.DT_FP32)
            tail_mreg_fp32_o = vf.update_mask(tail_valid // 2, dtype=pl.DT_FP32)
            reg = vf.load_align(in_tile, base + r * LANES)
            e_even = vf.exp_sub(reg, row_max_b, tail_mreg_fp16, dtype=pl.DT_FP32, layout=pl.CastLayout.ZERO)
            e_odd = vf.exp_sub(reg, row_max_b, tail_mreg_fp16, dtype=pl.DT_FP32, layout=pl.CastLayout.ONE)
            part_even = vf.reduce_sum(e_even, tail_mreg_fp32_z)
            row_sum = vf.add(row_sum, part_even, preg_fp32)
            part_odd = vf.reduce_sum(e_odd, tail_mreg_fp32_o)
            row_sum = vf.add(row_sum, part_odd, preg_fp32)
        row_sum_b = vf.full(row_sum, preg_fp32)

        # ---- pass3: div + astype fp32→fp16 + store ----
        for r in pl.range(0, n_full):
            reg = vf.load_align(in_tile, base + r * LANES)
            e_even = vf.exp_sub(reg, row_max_b, full_mreg_fp16, dtype=pl.DT_FP32, layout=pl.CastLayout.ZERO)
            e_odd = vf.exp_sub(reg, row_max_b, full_mreg_fp16, dtype=pl.DT_FP32, layout=pl.CastLayout.ONE)
            d_even = vf.div(e_even, row_sum_b, full_mreg_fp32_z)
            d_odd = vf.div(e_odd, row_sum_b, full_mreg_fp32_o)
            out16_even = vf.astype(d_even, full_mreg_fp32_z, dtype=pl.DT_FP16, layout=pl.CastLayout.ZERO)
            out16_odd = vf.astype(d_odd, full_mreg_fp32_o, dtype=pl.DT_FP16, layout=pl.CastLayout.ONE)
            out16 = vf.add(out16_even, out16_odd, full_mreg_fp16)
            vf.store_align(out_tile + (base + r * LANES), out16, full_mreg_fp16)
        for r in pl.range(n_full, n_full + n_tail_iters):
            tail_valid = n_cols - r * LANES
            tail_mreg_fp16 = vf.update_mask(tail_valid, dtype=pl.DT_FP16)
            tail_mreg_fp32_z = vf.update_mask((tail_valid + 1) // 2, dtype=pl.DT_FP32)
            tail_mreg_fp32_o = vf.update_mask(tail_valid // 2, dtype=pl.DT_FP32)
            reg = vf.load_align(in_tile, base + r * LANES)
            e_even = vf.exp_sub(reg, row_max_b, tail_mreg_fp16, dtype=pl.DT_FP32, layout=pl.CastLayout.ZERO)
            e_odd = vf.exp_sub(reg, row_max_b, tail_mreg_fp16, dtype=pl.DT_FP32, layout=pl.CastLayout.ONE)
            d_even = vf.div(e_even, row_sum_b, tail_mreg_fp32_z)
            d_odd = vf.div(e_odd, row_sum_b, tail_mreg_fp32_o)
            out16_even = vf.astype(d_even, tail_mreg_fp32_z, dtype=pl.DT_FP16, layout=pl.CastLayout.ZERO)
            out16_odd = vf.astype(d_odd, tail_mreg_fp32_o, dtype=pl.DT_FP16, layout=pl.CastLayout.ONE)
            out16 = vf.add(out16_even, out16_odd, tail_mreg_fp16)
            vf.store_align(out_tile + (base + r * LANES), out16, tail_mreg_fp16)


# ================================================================
# Kernel — single @pl.jit, double-buffer in/out, multicore strided
# ================================================================
@pl.jit(auto_mutex=True)
def softmax_tile_group_kernel(
    x: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
    y: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
):
    # valid_shape=[-1,-1] makes per-tile valid window dynamic (set at runtime)
    fp16_tt = pl.TileType(
        shape=[TILE_ROWS, MAX_N], dtype=pl.DT_FP16,
        target_memory=pl.MemorySpace.Vec, valid_shape=[-1, -1],
    )
    in_group = pl.make_tile_group(type=fp16_tt, addrs=[VA_IN0, VA_IN1], mutex_ids=[0, 1])
    out_group = pl.make_tile_group(type=fp16_tt, addrs=[VA_OUT0, VA_OUT1], mutex_ids=[2, 3])

    with pl.section_vector():
        rows = x.shape[0]
        cols = x.shape[1]
        num_cores = pl.get_block_num()
        core_id = pl.get_block_idx()

        num_tiles = (rows + TILE_ROWS - 1) // TILE_ROWS

        for tile_id in pl.range(core_id, num_tiles, num_cores):
            row_off = tile_id * TILE_ROWS
            valid_rows = pl.min(TILE_ROWS, rows - row_off)

            in_slot = in_group.next()
            pl.set_validshape(in_slot, [valid_rows, cols])
            pl.load(in_slot, x, [row_off, 0])

            out_slot = out_group.next()
            pl.set_validshape(out_slot, [valid_rows, cols])
            softmax_rows_vf(in_slot, out_slot, valid_rows, cols)

            pl.store(y, out_slot, [row_off, 0])


# ================================================================
# Wrapper — single kernel launch, no host tensor ops
# ================================================================
def softmax_wrapper(x: torch.Tensor) -> torch.Tensor:
    """Entry point: validate args, allocate output, launch kernel once.

    Per DESIGN §0 Wrapper boundary = 空 and wrapper-boundary.md R01:
    only dtype/shape/contiguity validation + torch.empty_like + one kernel launch.
    No .to()/.contiguous()/.reshape()/.movedim()/.t()/torch.zeros/torch.npu.synchronize.
    """
    assert x.dtype == torch.float16, f"x dtype must be float16, got {x.dtype}"
    assert x.ndim == 2, f"x must be 2D [B, N], got ndim={x.ndim}"
    assert x.is_contiguous(), "x must be contiguous (RowMajor 2D)"
    y = torch.empty_like(x)
    num_tiles = (x.shape[0] + TILE_ROWS - 1) // TILE_ROWS
    block_dim = min(get_platform_info().vector_core_num, num_tiles)
    softmax_tile_group_kernel[None, block_dim](x, y)
    return y


# ================================================================
# Test cases (DESIGN.md §8 — 8 cases)
# ================================================================

def test_softmax_aligned():
    """[3, 128]: fully aligned (B%3=0, N%128=0)."""
    from softmax_golden import _get_device
    device = _get_device()
    torch.manual_seed(42)
    x = torch.randn([3, 128], device=device, dtype=torch.float16)
    y = softmax_wrapper(x)
    _assert_precision(y, x, label="aligned [3,128]")


def test_softmax_row_tail():
    """[4, 128]: row-tail only (B%3=1 → 1 partial row; N%128=0)."""
    from softmax_golden import _get_device
    device = _get_device()
    torch.manual_seed(42)
    x = torch.randn([4, 128], device=device, dtype=torch.float16)
    y = softmax_wrapper(x)
    _assert_precision(y, x, label="row_tail [4,128]")


def test_softmax_col_tail():
    """[3, 200]: column-tail only (N%128=72 → partial register; B%3=0)."""
    from softmax_golden import _get_device
    device = _get_device()
    torch.manual_seed(42)
    x = torch.randn([3, 200], device=device, dtype=torch.float16)
    y = softmax_wrapper(x)
    _assert_precision(y, x, label="col_tail [3,200]")


def test_softmax_tail2d():
    """[4, 200]: 2D tail (row-tail 1 row + column-tail 72 lanes)."""
    from softmax_golden import _get_device
    device = _get_device()
    torch.manual_seed(42)
    x = torch.randn([4, 200], device=device, dtype=torch.float16)
    y = softmax_wrapper(x)
    _assert_precision(y, x, label="tail2d [4,200]")


def test_softmax_multitile():
    """[7, 200]: multi-tile + tail (3 row-tiles: 2 full + 1 tail-1; N=200 col-tail)."""
    from softmax_golden import _get_device
    device = _get_device()
    torch.manual_seed(42)
    x = torch.randn([7, 200], device=device, dtype=torch.float16)
    y = softmax_wrapper(x)
    _assert_precision(y, x, label="multitile [7,200]")


def test_softmax_p0_batch1_n128():
    """P0 [1, 128]: single row, single tile, single register."""
    from softmax_golden import _get_device
    device = _get_device()
    torch.manual_seed(42)
    x = torch.randn([1, 128], device=device, dtype=torch.float16)
    y = softmax_wrapper(x)
    _assert_precision(y, x, label="p0_batch1_n128")


def test_softmax_p0_batch4_n2048():
    """P0 [4, 2048]: mid batch, mid-large N (row-tail + 16 registers)."""
    from softmax_golden import _get_device
    device = _get_device()
    torch.manual_seed(42)
    x = torch.randn([4, 2048], device=device, dtype=torch.float16)
    y = softmax_wrapper(x)
    _assert_precision(y, x, label="p0_batch4_n2048")


def test_softmax_p0_batch32_n4096():
    """P0 [32, 4096]: large batch, large N (multi-core + 32 registers, fp32 acc stress)."""
    from softmax_golden import _get_device
    device = _get_device()
    torch.manual_seed(42)
    x = torch.randn([32, 4096], device=device, dtype=torch.float16)
    y = softmax_wrapper(x)
    _assert_precision(y, x, label="p0_batch32_n4096")


# ================================================================
# Main
# ================================================================
if __name__ == "__main__":
    logging.info("softmax — Pure Vec (VF) fp16, make_tile_group + auto_mutex, multicore")
    logging.info("=" * 60)
    test_softmax_aligned()
    test_softmax_row_tail()
    test_softmax_col_tail()
    test_softmax_tail2d()
    test_softmax_multitile()
    test_softmax_p0_batch1_n128()
    test_softmax_p0_batch4_n2048()
    test_softmax_p0_batch32_n4096()
    logging.info("\nAll tests PASS!")
