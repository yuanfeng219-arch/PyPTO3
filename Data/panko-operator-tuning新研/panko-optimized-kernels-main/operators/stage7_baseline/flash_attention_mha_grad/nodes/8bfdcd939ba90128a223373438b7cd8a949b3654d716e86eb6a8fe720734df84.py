# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2024-2026. All rights reserved.
# =============================================================================
# flash_attention_mha_grad_impl.py — L0 single-module fusion kernel
#
# Operator: flash_attention_mha_grad — varlen MHA attention backward gradient
#           (dQ/dK/dV) with single-pass fusion (SK-16).
#
# Module:  M1 — fused_attention_backward_kernel
# Algo:    varlen batch→head→s1→s2 nested loops, each (s1,s2) pair computes
#          C1 (S=dP), V1 (D/P/dS/BF16 cast), C2 (dQ/dK/dV), V2 (scale+atomic_add).
#
# Inputs (3D, reshaped to 2D by wrapper):
#   q/k/v/o/do: [total_seq, num_heads, head_dim] BF16
#   l_input/m_input: [total_seq, num_heads, 1] FP32
#   actual_q/actual_kv: [batch+1] INT32 cumsum
#
# Outputs:
#   dq [total_q, hidden_dim] FP32, dk [total_kv, hidden_dim] FP32,
#   dv [total_kv, hidden_dim] FP32
#
# Tile config (from DESIGN.md, "Patterns and Design Decisions"):
#   S1_TILE = S2_TILE = 128, HEAD_DIM = 128, HIDDEN_DIM = 1024
#   Cube TileShape: [128,128] for M/K/N axes (shared by all 5 matmuls)
#   Vec TileShape: [128,128] for V1 and V2 elementwise chains
#   div precision: pypto.PrecisionType.INTRINSIC
#
# Lint compliance:
#   OL01: exactly one @pypto.frontend.jit
#   OL02: output via atomic_add (no out=expr)
#   OL03: no return in JIT body
#   OL04: set_cube_tile_shapes + set_vec_tile_shapes called
#   OL05: all tensor params annotated
#   OL07: import pypto (not from-import)
#   OL08: wrapper named flash_attention_mha_grad_wrapper
#   OL45: no Python for...range in wrapper
#   OL46: no pypto.loop(1) wrapper with inner loop(N) — reshape done in wrapper
#   OL48: all tile params compile-time literals
#   OL57: only pypto.loop inside JIT body
#   OL58: output buffers allocated via torch.zeros in wrapper
#   OL62: torch only for layout/alloc/cast/reshape in wrapper
#
# Snapshot marker pairs (empty — ready for snapshot generator):
# SIG_IMPL, SIG_JIT, CALL_IMPL, HOST_WRAPPER_INSPECT_ALLOC,
# HOST_WRAPPER_INSPECT_PASS, before_LOOP_b, inside_LOOP_s1, after_LOOP_s1
# =============================================================================

import math
from typing import Tuple

import pypto
import torch
import torch_npu  # noqa: F401  required for NPU device init

# =============================================================================
# Layer B — Compile-time constants (OL48: must be Python int literals)
# =============================================================================

S1_TILE = 512       # Q tile size (tunable, design baseline)
S2_TILE = 512       # KV tile size (tunable, design baseline)
HEAD_DIM = 128      # head dimension
NUM_HEADS = 8       # number of attention heads
HIDDEN_DIM = 1024   # num_heads * head_dim = 8 * 128


# =============================================================================
# Layer J — JIT entry (entire kernel body inlined; no is_loop_begin/end needed
#            since atomic_add handles cross-iteration accumulation).
#
# NOTE: The decorator MUST be written literally as @pypto.frontend.jit
# (OL01). No alias, no from-import.
# =============================================================================

@pypto.frontend.jit(
    runtime_options={
        "run_mode": pypto.RunMode.NPU,
        "stitch_function_max_num": 1024,
        "device_sched_mode": 1,
        "ready_on_host_tensors": ["actual_q", "actual_kv"],
    },
    host_options={"compile_monitor_enable": 0},
    pass_options={"cube_nbuffer_setting": {-1: 2}, "vec_nbuffer_setting": {-1: 16}, "cube_l1_reuse_setting": {-1: 2}},
    debug_options={"runtime_debug_mode": 1},
)
def flash_attention_mha_grad_kernel_npu(
    q_2d: pypto.Tensor([pypto.DYNAMIC, 1024], pypto.DT_BF16),      # [total_q, hidden_dim] BF16
    k_2d: pypto.Tensor([pypto.DYNAMIC, 1024], pypto.DT_BF16),      # [total_kv, hidden_dim] BF16
    v_2d: pypto.Tensor([pypto.DYNAMIC, 1024], pypto.DT_BF16),      # [total_kv, hidden_dim] BF16
    o_2d: pypto.Tensor([pypto.DYNAMIC, 1024], pypto.DT_BF16),      # [total_q, hidden_dim] BF16
    do_2d: pypto.Tensor([pypto.DYNAMIC, 1024], pypto.DT_BF16),     # [total_q, hidden_dim] BF16
    l_2d: pypto.Tensor([pypto.DYNAMIC, 8], pypto.DT_FP32),         # [total_q, num_heads] FP32
    m_2d: pypto.Tensor([pypto.DYNAMIC, 8], pypto.DT_FP32),         # [total_q, num_heads] FP32
    actual_q: pypto.Tensor([pypto.DYNAMIC], pypto.DT_INT32),       # [batch+1] INT32 cumsum
    actual_kv: pypto.Tensor([pypto.DYNAMIC], pypto.DT_INT32),      # [batch+1] INT32 cumsum
    dq_out: pypto.Tensor([pypto.DYNAMIC, 1024], pypto.DT_FP32),    # [total_q, hidden_dim] FP32 output
    dk_out: pypto.Tensor([pypto.DYNAMIC, 1024], pypto.DT_FP32),    # [total_kv, hidden_dim] FP32 output
    dv_out: pypto.Tensor([pypto.DYNAMIC, 1024], pypto.DT_FP32),    # [total_kv, hidden_dim] FP32 output
    scale: float,                                                  # 1/sqrt(head_dim) ~ 0.08839
):
    """Single-pass fusion backward kernel for varlen MHA gradient.

    All tensor params first (OL26). No return (OL03). All outputs written
    via pypto.atomic_add (OL02). The 4-level loop nest (batch → head →
    s1_tile → s2_tile) is driven by pypto.loop (OL57) with trip counts
    derived from actual_q/actual_kv cumsum arrays and total_q/total_kv.
    """
    # ── Global config ──
    pypto.experimental.set_operation_options(combine_axis=True)

    # ── Dynamic dimensions from tensors (SymbolicScalar) ──
    total_q = q_2d.shape[0]        # SymbolicScalar — total Q sequence length
    total_kv = k_2d.shape[0]       # SymbolicScalar — total KV sequence length
    batch = actual_q.shape[0] - 1   # SymbolicScalar — number of batches

    # ==========================================================================
    # LOOP_b: varlen batch iteration (dynamic trip count from actual_q cumsum)
    # ==========================================================================
    for b_idx in pypto.loop(batch, name="LOOP_b"):
        # Per-batch offsets and sequence lengths from cumsum arrays
        # (actual_q/actual_kv marked ready_on_host_tensors for host-side index)
        q_start = actual_q[b_idx]                                  # batch b Q offset
        s1_len = actual_q[b_idx + 1] - q_start                     # batch b Q seq len
        kv_start = actual_kv[b_idx]                                # batch b KV offset
        s2_len = actual_kv[b_idx + 1] - kv_start                   # batch b KV seq len

        # Number of tiles for this batch (SymbolicScalar ceildiv)
        s1_loop = (s1_len + S1_TILE - 1) // S1_TILE               # ceildiv(s1_len, S1_TILE)
        s2_loop = (s2_len + S2_TILE - 1) // S2_TILE               # ceildiv(s2_len, S2_TILE)

        # ======================================================================
        # LOOP_n: per-head iteration (static count = NUM_HEADS)
        # ======================================================================
        for n_idx in pypto.loop(NUM_HEADS, name="LOOP_n"):
            h_ofs = n_idx * HEAD_DIM                               # column offset in [hidden_dim]

            # ==================================================================
            # LOOP_s1: Q tile outer loop
            # ==================================================================
            for s1_idx in pypto.loop(s1_loop, name="LOOP_s1"):
                s1_off = q_start + s1_idx * S1_TILE                # row offset in total_q
                actual_s1 = (s1_len - s1_idx * S1_TILE).min(S1_TILE)  # valid rows in this tile

                # Q/O/dO tile views  [S1_TILE, HEAD_DIM] BF16
                q_i = pypto.view(q_2d, [S1_TILE, HEAD_DIM], [s1_off, h_ofs],
                                 valid_shape=[actual_s1, HEAD_DIM])   # [s1_tile, D] BF16
                o_i = pypto.view(o_2d, [S1_TILE, HEAD_DIM], [s1_off, h_ofs],
                                 valid_shape=[actual_s1, HEAD_DIM])   # [s1_tile, D] BF16
                do_i = pypto.view(do_2d, [S1_TILE, HEAD_DIM], [s1_off, h_ofs],
                                  valid_shape=[actual_s1, HEAD_DIM])  # [s1_tile, D] BF16

                # l/m tile views ([total_q, NUM_HEADS] → [S1_TILE, 1] FP32)
                m_i = pypto.view(m_2d, [S1_TILE, 1], [s1_off, n_idx],
                                 valid_shape=[actual_s1, 1])         # [s1_tile, 1] FP32
                l_i = pypto.view(l_2d, [S1_TILE, 1], [s1_off, n_idx],
                                 valid_shape=[actual_s1, 1])         # [s1_tile, 1] FP32

                # ==============================================================
                # Hoist the D reduction: D = sum(O * dO, -1, keepdim=True) depends only on s1.
                #   (Hoist the loop invariant to avoid computing it eight times in LOOP_s2.)
                # ==============================================================
                pypto.set_vec_tile_shapes(128, 128)
                o_fp32 = pypto.cast(o_i, pypto.DT_FP32)             # [s1_tile, D] FP32
                do_fp32 = pypto.cast(do_i, pypto.DT_FP32)           # [s1_tile, D] FP32
                # I-4: remove keepdim to avoid tail axis = 1; reshape back for broadcast
                d_1d = pypto.sum(pypto.mul(o_fp32, do_fp32),
                                 -1, keepdim=False)                  # [s1_tile] FP32
                d_i = pypto.reshape(d_1d, [S1_TILE, 1])              # [s1_tile, 1] FP32

                # ==============================================================
                # LOOP_s2: KV tile inner loop (dQ accumulates across s2 via atomic_add)
                # ==============================================================
                for s2_idx in pypto.loop(s2_loop, unroll_list=[8, 4, 2, 1], name="LOOP_s2"):
                    s2_off = kv_start + s2_idx * S2_TILE            # row offset in total_kv
                    actual_s2 = (s2_len - s2_idx * S2_TILE).min(S2_TILE)  # valid rows

                    # K/V tile views  [S2_TILE, HEAD_DIM] BF16
                    k_j = pypto.view(k_2d, [S2_TILE, HEAD_DIM], [s2_off, h_ofs],
                                     valid_shape=[actual_s2, HEAD_DIM])  # [s2_tile, D] BF16
                    v_j = pypto.view(v_2d, [S2_TILE, HEAD_DIM], [s2_off, h_ofs],
                                     valid_shape=[actual_s2, HEAD_DIM])  # [s2_tile, D] BF16

                    # ==========================================================
                    # C1: S = Q @ K^T, dP = dO @ V^T (BF16 x BF16 -> FP32, outside the scope)
                    # ==========================================================
                    pypto.set_cube_tile_shapes([128, 128], [128, 128], [128, 128])
                    s_ij = pypto.matmul(q_i, k_j, pypto.DT_FP32,
                                        b_trans=True)                # [s1_tile, s2_tile] FP32
                    dp_ij = pypto.matmul(do_i, v_j, pypto.DT_FP32,
                                         b_trans=True)               # [s1_tile, s2_tile] FP32

                    # ==========================================================
                    # V1: Reconstruct P directly from m/l, compute dS, and truncate to BF16 (sg_set_scope graph fusion)
                    # ==========================================================
                    # I-4 tuning: V1 vec_tile (128,128) -> (32,512), covering the last axis in one pass
                    pypto.set_vec_tile_shapes(32, 512)               # 2D vec tile
                    pypto.set_pass_options(sg_set_scope=1)

                    # S *= scale
                    s_ij = pypto.mul(s_ij, scale)                   # [s1_tile, s2_tile] FP32

                    # P = exp(S - m) / l: use forward statistics directly without recomputing softmax.
                    p_ij = pypto.div(
                        pypto.exp(pypto.sub(s_ij, m_i)), l_i,
                        precision_type=pypto.PrecisionType.INTRINSIC,
                    )                                                # [s1_tile, s2_tile] FP32

                    # dS = P * (dP - D)  [s1_tile, s2_tile] FP32
                    ds_ij = pypto.mul(p_ij, pypto.sub(dp_ij, d_i))  # [s1_tile, s2_tile] FP32

                    # BF16 truncation: cast FP32 to BF16 (golden to(bfloat16) semantics)
                    ds_bf16 = pypto.cast(ds_ij, pypto.DT_BF16)      # [s1_tile, s2_tile] BF16
                    p_bf16 = pypto.cast(p_ij, pypto.DT_BF16)        # [s1_tile, s2_tile] BF16
                    pypto.set_pass_options(sg_set_scope=-1)

                    # ==========================================================
                    # C2: dQ/dK/dV (BF16 x BF16 -> FP32, outside the scope)
                    # ==========================================================
                    pypto.set_cube_tile_shapes([128, 128], [128, 128], [128, 128])

                    # dQ_tile = dS_bf16 @ k_j  [s1_tile, D] FP32
                    dq_tile = pypto.matmul(ds_bf16, k_j, pypto.DT_FP32)

                    # dK_tile = dS_bf16^T @ q_i  [s2_tile, D] FP32
                    dk_tile = pypto.matmul(ds_bf16, q_i, pypto.DT_FP32,
                                           a_trans=True)

                    # dV_tile = P_bf16^T @ do_i  [s2_tile, D] FP32
                    dv_tile = pypto.matmul(p_bf16, do_i, pypto.DT_FP32,
                                           a_trans=True)

                    # ==========================================================
                    # V2: scale + atomic_add three-way writeback (zero-initialized on the host)
                    # ==========================================================
                    pypto.set_vec_tile_shapes(128, 128)

                    # atomic_add(dQ*scale, [s1_off, h_ofs], dq_out)
                    pypto.atomic_add(pypto.mul(dq_tile, scale),
                                     [s1_off, h_ofs], dq_out)

                    # atomic_add(dK*scale, [s2_off, h_ofs], dk_out)
                    pypto.atomic_add(pypto.mul(dk_tile, scale),
                                     [s2_off, h_ofs], dk_out)

                    # atomic_add(dV, [s2_off, h_ofs], dv_out): dV is not multiplied by scale.
                    pypto.atomic_add(dv_tile, [s2_off, h_ofs], dv_out)

    # NOTE: No return statement (OL03). All outputs written via atomic_add.


# =============================================================================
# Layer K — Host wrapper (public entry)
# RESPONSIBILITIES (OL62-compliant):
#   1. torch reshape 3D→2D (no-copy view for contiguous tensors)
#   2. Allocate dq/dk/dv output buffers (torch.zeros)
#   3. Invoke JIT entry ONCE (no Python for...range — OL45)
#   4. Return (dq, dk, dv) tuple
# =============================================================================

def flash_attention_mha_grad_wrapper(
    q: torch.Tensor,           # [total_q, num_heads, head_dim] BF16
    k: torch.Tensor,           # [total_kv, num_heads, head_dim] BF16
    v: torch.Tensor,           # [total_kv, num_heads, head_dim] BF16
    o: torch.Tensor,           # [total_q, num_heads, head_dim] BF16
    do: torch.Tensor,          # [total_q, num_heads, head_dim] BF16
    l_input: torch.Tensor,     # [total_q, num_heads, 1] FP32
    m_input: torch.Tensor,     # [total_q, num_heads, 1] FP32
    actual_q: torch.Tensor,    # [batch+1] INT32 cumsum
    actual_kv: torch.Tensor,   # [batch+1] INT32 cumsum
    num_heads: int = 8,
    head_dim: int = 128,
    scale: float = 0.08838834764831845,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Varlen flash attention backward wrapper.

    Reshapes 3D inputs to 2D on host (no-copy contiguous view), allocates
    output buffers, dispatches single fused JIT kernel.

    Args:
        q: [total_q, num_heads, head_dim] BF16
        k: [total_kv, num_heads, head_dim] BF16
        v: [total_kv, num_heads, head_dim] BF16
        o: [total_q, num_heads, head_dim] BF16 — forward output
        do: [total_q, num_heads, head_dim] BF16 — output gradient
        l_input: [total_q, num_heads, 1] FP32 — softmax sum
        m_input: [total_q, num_heads, 1] FP32 — softmax max
        actual_q: [batch+1] INT32 — Q cumsum prefix
        actual_kv: [batch+1] INT32 — KV cumsum prefix
        num_heads: attention heads (default 8)
        head_dim: head dimension (default 128)
        scale: 1/sqrt(head_dim) (default ~0.08839)

    Returns:
        (dq, dk, dv):
          dq: [total_q, hidden_dim] FP32
          dk: [total_kv, hidden_dim] FP32
          dv: [total_kv, hidden_dim] FP32
    """
    # Derive layout dimensions
    total_q = q.shape[0]
    total_kv = k.shape[0]
    hidden_dim = num_heads * head_dim

    # ── 1. 3D → 2D reshape on host (no-copy view for contiguous tensors) ──
    #    [total_seq, num_heads, head_dim] → [total_seq, hidden_dim]
    #    Cast： q/k/v/o/do remain BF16; l/m_input remain FP32
    q_2d = q.reshape(total_q, hidden_dim)          # [total_q, 1024] BF16
    k_2d = k.reshape(total_kv, hidden_dim)         # [total_kv, 1024] BF16
    v_2d = v.reshape(total_kv, hidden_dim)         # [total_kv, 1024] BF16
    o_2d = o.reshape(total_q, hidden_dim)          # [total_q, 1024] BF16
    do_2d = do.reshape(total_q, hidden_dim)        # [total_q, 1024] BF16
    # l/m: [total_q, num_heads, 1] → [total_q, num_heads]
    l_2d = l_input.reshape(total_q, num_heads)     # [total_q, 8] FP32
    m_2d = m_input.reshape(total_q, num_heads)     # [total_q, 8] FP32

    # ── 2. Allocate output buffers (torch.zeros — OL58) ──
    dq_out = torch.zeros(total_q, hidden_dim,
                         dtype=torch.float32, device=q.device)
    dk_out = torch.zeros(total_kv, hidden_dim,
                         dtype=torch.float32, device=k.device)
    dv_out = torch.zeros(total_kv, hidden_dim,
                         dtype=torch.float32, device=v.device)

    # ── 3. Single JIT call (no Python for...range — OL45) ──
    flash_attention_mha_grad_kernel_npu(
        q_2d, k_2d, v_2d, o_2d, do_2d, l_2d, m_2d,
        actual_q, actual_kv,
        dq_out, dk_out, dv_out,
        scale,
    )

    return dq_out, dk_out, dv_out