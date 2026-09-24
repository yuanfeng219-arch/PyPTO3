# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2024-2026. All rights reserved.
# =============================================================================
# flash_attention_mha_impl.py
# Varlen Flash Attention MHA Forward — PyPTO kernel (online softmax, uniform update).
#
# L0 (module_count=1) single-shot implementation.
# =============================================================================
"""
Varlen Flash Attention MHA Forward (online softmax, xi-block tiling).

Algorithm:
    for batch (cu_seqlens slicing):
      for head:
        for q_tile:
          mi, li, oi = full(-inf), full(0), full(0)     # online softmax accumulators
          for k_tile:
            C1:  S = matmul(Q_tile, K_tile, b_trans=True, DT_FP32)   # [QT, KT] FP32
            V1:  P_ij = exp(S*scale - rowmax(S*scale))               # online softmax per tile
                 l_ij = sum(P_ij),  P_bf16 = cast(P_ij, BF16)
            C2:  O_ij = matmul(P_bf16, V_tile, DT_FP32)              # [QT, D] FP32
            V2:  mi' = max(mi, m_ij)                                   # uniform update (no branch)
                 t1 = expand_exp_dif(mi, mi'),  t2 = expand_exp_dif(m_ij, mi')
                 li' = t1*li + t2*l_ij,  O_tmp = t1*oi + t2*o_ij
          final: o_norm = div(oi, li, INTRINSIC) -> cast BF16 -> assemble O
                 l_final -> assemble L,  m_final -> assemble M

Reference: DESIGN.md pseudocode + SK-01 online flash attention skeleton.
"""

import pypto
import torch
import torch_npu  # noqa: F401  required for NPU device init

# =============================================================================
# Compile-time constants (Python int literals per OL48)
# =============================================================================

Q_TILE = 256   # Unchanged: Q-tile granularity does not affect numerical results (bitwise identical on CPU).
K_TILE = 320   # Align with the golden KV tile boundaries (flash_attention_mha_golden.py L81-82).

# IEEE -inf for mi accumulator init (uniform formula: first tile auto-degenerates)
_NEG_INF = float("-inf")


# =============================================================================
# Layer J — JIT entry (full algorithm inlined; no is_loop_begin/is_loop_end)
# =============================================================================

@pypto.frontend.jit(
    pass_options={
        "cube_l1_reuse_setting": {-1: 2},
        "vec_nbuffer_setting": {-1: 4},
        "cube_nbuffer_setting": {-1: 4},
    },
    runtime_options={
        "stitch_function_max_num": 128,
        "device_sched_mode": 1,
        "max_workspace_kb": 1068440,
    },
    host_options={"compile_monitor_enable": 0},
    debug_options={"runtime_debug_mode": 1},
)
def flash_attention_mha_kernel(
    q: pypto.Tensor([pypto.DYNAMIC, ...], pypto.DT_BF16),               # [total_q, hidden]  BF16
    k: pypto.Tensor([pypto.DYNAMIC, ...], pypto.DT_BF16),               # [total_kv, hidden] BF16
    v: pypto.Tensor([pypto.DYNAMIC, ...], pypto.DT_BF16),               # [total_kv, hidden] BF16
    o: pypto.Tensor([pypto.DYNAMIC, ...], pypto.DT_BF16),               # [total_q, hidden]  BF16 (output)
    l_output: pypto.Tensor([pypto.DYNAMIC, ...], pypto.DT_FP32),        # [total_q, num_heads] FP32 (output)
    m_output: pypto.Tensor([pypto.DYNAMIC, ...], pypto.DT_FP32),        # [total_q, num_heads] FP32 (output)
    cu_seqlens_q: pypto.Tensor([pypto.DYNAMIC], pypto.DT_INT32),        # [batch+1] INT32
    cu_seqlens_k: pypto.Tensor([pypto.DYNAMIC], pypto.DT_INT32),        # [batch+1] INT32
    num_heads: int,        # compile-time constant
    head_dim: int,         # compile-time constant
    scale: float,          # compile-time constant = 1/sqrt(head_dim)
):
    """Varlen Flash Attention MHA forward: 4-level loops, online softmax, uniform update."""
    pypto.experimental.set_operation_options(combine_axis=True)

    # derive batch count from cu_seqlens length
    batch = cu_seqlens_q.shape[0] - 1

    # batch loop
    for b_idx in pypto.loop(batch, name="batch_loop"):
        q_start = cu_seqlens_q[b_idx]
        q_end = cu_seqlens_q[b_idx + 1]
        seq_len_q = q_end - q_start
        seq_len_q.as_variable()

        k_start = cu_seqlens_k[b_idx]
        k_end = cu_seqlens_k[b_idx + 1]
        seq_len_k = k_end - k_start
        seq_len_k.as_variable()

        q_tile_count = (seq_len_q + Q_TILE - 1) // Q_TILE
        k_tile_count = (seq_len_k + K_TILE - 1) // K_TILE

        # head loop
        for h_idx in pypto.loop(num_heads, name="head_loop"):
            h_offset = h_idx * head_dim

            # q_tile loop
            for qi_idx in pypto.loop(q_tile_count, name="q_tile_loop"):
                q_tile_start = qi_idx * Q_TILE
                q_tile_len = (seq_len_q - q_tile_start).min(Q_TILE)

                # online softmax accumulators (per q_tile, outside k_tile loop)
                pypto.set_vec_tile_shapes(64, 512)
                mi = pypto.full([Q_TILE, 1], _NEG_INF, pypto.DT_FP32, valid_shape=[q_tile_len, 1])      # [QT,1] FP32, running max
                li = pypto.full([Q_TILE, 1], 0.0, pypto.DT_FP32, valid_shape=[q_tile_len, 1])          # [QT,1] FP32, running sum
                oi = pypto.full([Q_TILE, head_dim], 0.0, pypto.DT_FP32, valid_shape=[q_tile_len, head_dim])  # [QT,D] FP32

                # Q tile view
                q_tile_view = pypto.view(
                    q, [Q_TILE, head_dim],
                    [q_start + q_tile_start, h_offset],
                    valid_shape=[q_tile_len, head_dim],
                )                                                           # [QT,D] BF16

                # k_tile loop (carries mi/li/oi across iterations)
                for ki_idx in pypto.loop(k_tile_count, unroll_list=[13, 8, 4, 2, 1], name="k_tile_loop"):
                    k_tile_start = ki_idx * K_TILE
                    k_tile_len = (seq_len_k - k_tile_start).min(K_TILE)

                    # K/V tile views
                    k_tile_view = pypto.view(
                        k, [K_TILE, head_dim],
                        [k_start + k_tile_start, h_offset],
                        valid_shape=[k_tile_len, head_dim],
                    )                                                       # [KT,D] BF16
                    v_tile_view = pypto.view(
                        v, [K_TILE, head_dim],
                        [k_start + k_tile_start, h_offset],
                        valid_shape=[k_tile_len, head_dim],
                    )                                                       # [KT,D] BF16

                    # C1: Q @ K^T (BF16 BF16 -> FP32)
                    pypto.set_cube_tile_shapes([256, 256], [128, 256], [128, 128])
                    scores = pypto.matmul(q_tile_view, k_tile_view, pypto.DT_FP32, b_trans=True)  # [QT,KT] FP32

                    # V1: per-tile softmax statistics
                    pypto.set_vec_tile_shapes(64, 512)
                    scores_scaled = pypto.mul(scores, scale)                 # [QT,KT] FP32
                    mij = pypto.amax(scores_scaled, dim=-1, keepdim=True)    # [QT,1] FP32
                    s_shifted = pypto.sub(scores_scaled, mij)                # [QT,KT] FP32
                    pij = pypto.exp(s_shifted)                               # [QT,KT] FP32
                    lij = pypto.sum(pij, dim=-1, keepdim=True)               # [QT,1] FP32
                    p_bf16 = pypto.cast(pij, pypto.DT_BF16)                  # [QT,KT] BF16
                    if pypto.platform.npuarch == 'DAV_2201':
                        pypto.set_pass_options(sg_set_scope=-1)

                    # C2: P @ V (BF16 BF16 -> FP32)
                    pypto.set_cube_tile_shapes([256, 256], [128, 256], [160, 320])
                    oij = pypto.matmul(p_bf16, v_tile_view, pypto.DT_FP32)  # [QT,D] FP32

                    # V2: online softmax state update (branch: skip formula on first tile)
                    if pypto.platform.npuarch == 'DAV_2201':
                        pypto.set_pass_options(sg_set_scope=1)
                    pypto.set_vec_tile_shapes(128, 128)

                    if pypto.is_loop_begin(ki_idx):
                        mi[:] = mij
                        li[:] = lij
                        oi[:] = oij
                    else:
                        mi_cur = pypto.view(mi, [Q_TILE, 1], [0, 0], valid_shape=[q_tile_len, 1])
                        li_cur = pypto.view(li, [Q_TILE, 1], [0, 0], valid_shape=[q_tile_len, 1])
                        oi_cur = pypto.view(oi, [Q_TILE, head_dim], [0, 0], valid_shape=[q_tile_len, head_dim])

                        mi_new = pypto.maximum(mi_cur, mij)
                        t1 = pypto.expand_exp_dif(mi_cur, mi_new)
                        t2 = pypto.expand_exp_dif(mij, mi_new)
                        li_new = pypto.add(pypto.mul(t1, li_cur), pypto.mul(t2, lij))
                        oi_tmp = pypto.add(pypto.mul(oi_cur, t1), pypto.mul(oij, t2))

                        mi[:] = mi_new
                        li[:] = li_new
                        oi[:] = oi_tmp
                    if pypto.platform.npuarch == 'DAV_2201':
                        pypto.set_pass_options(sg_set_scope=-1)

                # final normalize + writeback (after last KV tile)
                pypto.set_vec_tile_shapes(128, 128)
                mi_final = pypto.view(mi, [Q_TILE, 1], [0, 0], valid_shape=[q_tile_len, 1])            # [QT,1] FP32
                li_final = pypto.view(li, [Q_TILE, 1], [0, 0], valid_shape=[q_tile_len, 1])            # [QT,1] FP32
                oi_final = pypto.view(oi, [Q_TILE, head_dim], [0, 0], valid_shape=[q_tile_len, head_dim])  # [QT,D] FP32

                o_norm = pypto.div(oi_final, li_final, precision_type=pypto.PrecisionType.INTRINSIC)   # [QT,D] FP32
                o_bf16 = pypto.cast(o_norm, pypto.DT_BF16)                                             # [QT,D] BF16

                pypto.assemble(o_bf16, [q_start + q_tile_start, h_offset], o)          # -> o[total_q, hidden] BF16
                pypto.assemble(li_final, [q_start + q_tile_start, h_idx], l_output)    # -> l[total_q, num_heads] FP32
                pypto.assemble(mi_final, [q_start + q_tile_start, h_idx], m_output)    # -> m[total_q, num_heads] FP32


# =============================================================================
# Layer K — Host wrapper
# =============================================================================

def flash_attention_mha_wrapper(q, k, v, cu_seqlens_q, cu_seqlens_k,
                                num_heads, head_dim, scale):
    """Varlen Flash Attention MHA forward host wrapper.

    Args:
        q:  [total_q, num_heads*head_dim] BF16 -- query (device tensor)
        k:  [total_kv, num_heads*head_dim] BF16 -- key (device tensor)
        v:  [total_kv, num_heads*head_dim] BF16 -- value (device tensor)
        cu_seqlens_q: [batch+1] INT32 -- cumulative Q sequence lengths
        cu_seqlens_k: [batch+1] INT32 -- cumulative KV sequence lengths
        num_heads: number of attention heads
        head_dim: dimension per head
        scale: 1/sqrt(head_dim)

    Returns:
        o: [total_q, num_heads*head_dim] BF16 -- attention output
        l: [total_q, num_heads] FP32 -- softmax log-sum (rescaled to running max)
        m: [total_q, num_heads] FP32 -- softmax running max
    """
    device = q.device
    total_q = q.shape[0]
    hidden = num_heads * head_dim

    o = torch.empty(total_q, hidden, dtype=torch.bfloat16, device=device)
    l_out = torch.empty(total_q, num_heads, dtype=torch.float32, device=device)
    m_out = torch.empty(total_q, num_heads, dtype=torch.float32, device=device)

    flash_attention_mha_kernel(
        q, k, v, o, l_out, m_out,
        cu_seqlens_q, cu_seqlens_k,
        num_heads, head_dim, scale,
    )

    return o, l_out, m_out


if __name__ == "__main__":
    print("flash_attention_mha_impl.py loaded.")
    print(f"  Q_TILE={Q_TILE}, K_TILE={K_TILE}")
    print("  Run smoke_check_impl.py for host-side codegen check.")