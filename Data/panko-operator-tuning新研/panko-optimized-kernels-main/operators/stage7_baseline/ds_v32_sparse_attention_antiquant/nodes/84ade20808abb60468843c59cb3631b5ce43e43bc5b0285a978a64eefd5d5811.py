# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2024-2026. All rights reserved.
# =============================================================================
# ds_v32_sparse_attention_antiquant_impl.py — L0 single-shot kernel
#
# Fused kernel: sparse gather -> int8 dequant -> rope view -> C1(QK^T) ->
# V1(online softmax) → C2(PV) → running-max state update → final normalize.
#
# Both the outer b/s1 loops and the inner s2 tile loop use pypto.loop inside JIT.
# =============================================================================

import pypto
import torch
import torch_npu  # noqa: F401

# Layer B: Constants

S2_TILE = 512
SCALE_GROUP = 128

# C1: Q[128,576] @ kv_up^T[576, tile_len] -> [128, tile_len]
C1_KL0 = 64
C1_KL1 = 576
C1_NL0 = 128
C1_NL1 = 512

# V1: online softmax [128, S2_TILE] fp32
V1_VT0 = 128
V1_VT1 = 64

# C2: p_bf16[128, tile_len] @ vj[tile_len, 512] -> [128, 512]
C2_KL0 = 128
C2_KL1 = 512
C2_NL0 = 128
C2_NL1 = 512


# =============================================================================
# Layer H — PyPTO sub-kernels
# =============================================================================

def pypto_stage_dequant(kn_i8, sc_vint8, tile_len):
    kn_r = pypto.reshape(kn_i8, [-1, SCALE_GROUP])

    pypto.set_vec_tile_shapes(128, 128)
    kn_f16 = pypto.cast(kn_r, pypto.DT_FP16)
    kn_f32 = pypto.cast(kn_f16, pypto.DT_FP32)

    sc_f32_raw = pypto.view(sc_vint8, dtype=pypto.DT_FP32)
    sc_f32 = pypto.reshape(sc_f32_raw, [-1, 1])

    pypto.set_vec_tile_shapes(128, 128)
    kn_deq_f32 = pypto.mul(kn_f32, sc_f32)

    kn_deq = pypto.cast(kn_deq_f32, pypto.DT_BF16)
    kn_deq = pypto.reshape(kn_deq, [S2_TILE, 512])
    return kn_deq


# =============================================================================
# Layer I + J: JIT entry (including all pypto.loop calls)
# =============================================================================

@pypto.frontend.jit(
    pass_options={
        "cube_l1_reuse_setting": {-1: 2},
        "vec_nbuffer_setting": {-2: 1, -1: 8},
        "cube_nbuffer_setting": {-1: 2},
    },
    runtime_options={
"stitch_function_max_num": 128,
"device_sched_mode": 0,
    }, host_options={"compile_monitor_enable": 0}, debug_options={"runtime_debug_mode": 1})
def ds_v32_sparse_attention_antiquant_kernel_npu(
    q: pypto.Tensor([pypto.DYNAMIC, 576], pypto.DT_BF16),
    nope_cache: pypto.Tensor([262144, 656], pypto.DT_INT8),
    topk_idx: pypto.Tensor([8, 2048], pypto.DT_INT32),
    block_table: pypto.Tensor([4, 512], pypto.DT_INT32),
    act_seq: pypto.Tensor([4], pypto.DT_INT32),
    y0: pypto.Tensor([pypto.DYNAMIC, 512], pypto.DT_BF16),
    nq: int = 128,
    kv_lora_rank: int = 512,
    qk_rope_dim: int = 64,
    softmax_scale: float = 0.041666666666666664,
    topk: int = 2048,
    block_size: int = 128,
    b: int = 4,
    s1_count: int = 2,
):
    """Single fused JIT kernel: all three b/s1/s2 loops execute inside the kernel.

    Tensor shape annotations: the M axis of q/y0 is pypto.DYNAMIC; all other dimensions are static.
    The outer b/s1 loops use pypto.loop(N); the inner s2 loop uses pypto.loop(start, end, step).
    Allocate the mi/li/oi FP32 accumulators inside the s1 loop, with a separate scope per (b,s1).
    Initialize each (b,s1) pair from its first tile using `is_loop_begin(s2_idx)` and `[:]` writes.
    """
    pypto.experimental.set_operation_options(combine_axis=True)

    # Outermost loop: batch
    for b_idx in pypto.loop(b, name="batch"):
        cur_k_seq = act_seq[b_idx]

        # Second-level loop: s1
        for s1_idx in pypto.loop(s1_count, name="s1"):
            cur_seq_raw = cur_k_seq - s1_count + 1 + s1_idx
            cur_seq = cur_seq_raw.max(0).min(topk)

            # qi = q[b_idx, s1_idx, :, :]
            q_off = b_idx * s1_count * nq + s1_idx * nq
            qi = pypto.view(q, [nq, kv_lora_rank + qk_rope_dim],
                            [q_off, 0])  # [128, 576] bf16

            # Allocate accumulators in a separate scope per (b,s1), avoiding static persistence across JIT calls.
            oi_buf = pypto.tensor([128, 512], pypto.DT_FP32)
            mi_buf = pypto.tensor([128, 1], pypto.DT_FP32)
            li_buf = pypto.tensor([128, 1], pypto.DT_FP32)

            # Innermost loop: s2 KV tile
            for s2_idx in pypto.loop(0, cur_seq, S2_TILE, name="s2",
                                     unroll_list=[2, 1]):
                s2_off = s2_idx
                tile_len = (cur_seq - s2_off).min(S2_TILE)

                # ── [a] sparse gather: topk_idx → offset → cache row ──
                pypto.set_vec_tile_shapes(1, 512)
                row_idx = b_idx * s1_count + s1_idx
                topk_vals_v = pypto.view(topk_idx, [1, S2_TILE],
                                         [row_idx, s2_off],
                                         valid_shape=[1, tile_len])
                blk_ids_v = pypto.floor_div(topk_vals_v, block_size)

                if pypto.is_loop_end(s2_idx):
                    pos_idx = pypto.reshape(pypto.arange(S2_TILE),
                                            [1, S2_TILE])
                    pos_f = pypto.cast(pos_idx, pypto.DT_FP32)
                    tile_f = pypto.full([1, S2_TILE], tile_len,
                                        pypto.DT_FP32)
                    pos_mask = pypto.lt(pos_f, tile_f)
                    zero_i32 = pypto.zeros([1, S2_TILE], dtype=pypto.DT_INT32)
                    blk_ids_c = pypto.where(pos_mask, blk_ids_v,
                                            zero_i32)
                    b_tbl_row = pypto.view(block_table, [1, 512],
                                           [b_idx, 0])
                    blk_id_v = pypto.gather(b_tbl_row, 1, blk_ids_c)
                    mod_v = pypto.sub(topk_vals_v,
                                      pypto.mul(blk_ids_v, block_size))
                    offsets_v = pypto.add(pypto.mul(blk_id_v, block_size),
                                          mod_v)
                    offsets_c = pypto.where(pos_mask, offsets_v,
                                            zero_i32)
                    pypto.set_vec_tile_shapes(512)
                    offsets_flat = pypto.reshape(offsets_c,
                                                 [S2_TILE])
                else:
                    b_tbl_row = pypto.view(block_table, [1, 512],
                                           [b_idx, 0])
                    blk_id_v = pypto.gather(b_tbl_row, 1, blk_ids_v)
                    mod_v = pypto.sub(topk_vals_v,
                                      pypto.mul(blk_ids_v, block_size))
                    offsets_v = pypto.add(pypto.mul(blk_id_v, block_size),
                                          mod_v)
                    pypto.set_vec_tile_shapes(512)
                    offsets_flat = pypto.reshape(offsets_v,
                                                 [S2_TILE])
                pypto.set_vec_tile_shapes(128, 128)
                rows = pypto.index_select(nope_cache, 0,
                                          offsets_flat)               # [512, 656] int8

                # [b] Unpack and slice the cache
                kn_i8 = pypto.view(rows, [S2_TILE, 512], [0, 0],
                                   valid_shape=[tile_len, 512])       # [tile_len, 512] int8
                kr_vint8 = pypto.view(rows, [S2_TILE, 128], [0, 512],
                                      valid_shape=[tile_len, 128])    # [tile_len, 128] int8
                sc_vint8 = pypto.view(rows, [S2_TILE, 16], [0, 640],
                                      valid_shape=[tile_len, 16])     # [tile_len, 16] int8

                # [c] Dequantize nope keys: int8 -> fp16 -> fp32 x scale -> bf16
                kn_deq = pypto_stage_dequant(kn_i8, sc_vint8,
                                             tile_len)                # [tile_len, 512] bf16

                # [d] Reinterpret rope key bits
                kr_fp = pypto.view(kr_vint8,
                                   dtype=pypto.DT_BF16)               # [tile_len, 64] bf16

                # ── [e] concat kv_up, vj = kn_deq ────────────────────
                pypto.set_vec_tile_shapes(1, 512)
                kv_up = pypto.concat([kn_deq, kr_fp], dim=-1)         # [tile_len, 576] bf16
                vj = kn_deq                                            # alias: [tile_len, 512] bf16

                # ── [f] C1: QK^T MatMul (Cube) ───────────────────────
                pypto.set_cube_tile_shapes([128, 128], [64, 576], [128, 512])
                sij = pypto.matmul(qi, kv_up, pypto.DT_FP32,
                                   b_trans=True)                      # [128, tile_len] fp32

                # ── [g] V1: Online Softmax (Vector) ──────────────────
                pypto.set_pass_options(sg_set_scope=1)
                pypto.set_vec_tile_shapes(128, 128)
                sij_s = pypto.mul(sij, softmax_scale)                 # [128, tile_len] fp32
                m_tile = pypto.amax(sij_s, dim=-1, keepdim=True)      # [128, 1] fp32
                p = pypto.exp(pypto.sub(sij_s, m_tile), pypto.PrecisionType.HIGH_PRECISION)  # [128, tile_len] fp32
                l_tile = pypto.sum(p, dim=-1, keepdim=True)           # [128, 1] fp32
                p_bf16 = pypto.cast(p, pypto.DT_BF16)                 # [128, tile_len] bf16
                pypto.set_pass_options(sg_set_scope=-1)

                # ── [h] C2: P@V MatMul (Cube) ────────────────────────
                pypto.set_cube_tile_shapes([128, 128], [128, 512], [128, 512])
                q1 = pypto.matmul(p_bf16, vj, pypto.DT_FP32)          # [128, 512] fp32

                # [i] Online state update (three branches)
                if pypto.is_loop_begin(s2_idx):
                    oi_buf[:] = q1                                    # [128, 512] fp32
                    mi_buf[:] = m_tile                                 # [128, 1] fp32
                    li_buf[:] = l_tile                                 # [128, 1] fp32
                else:
                    mi_prev = pypto.view(mi_buf, [128, 1], [0, 0])      # [128, 1] fp32 snapshot
                    mi_new = pypto.maximum(mi_prev, m_tile)             # [128, 1] fp32
                    alpha = pypto.exp(pypto.sub(mi_prev, mi_new), pypto.PrecisionType.HIGH_PRECISION)  # [128, 1] fp32
                    beta = pypto.exp(pypto.sub(m_tile, mi_new), pypto.PrecisionType.HIGH_PRECISION)    # [128, 1] fp32
                    li_buf[:] = pypto.add(
                        pypto.mul(alpha, pypto.view(li_buf, [128, 1], [0, 0])),
                        pypto.mul(beta, l_tile),
                    )                                                 # [128, 1] fp32
                    oi_buf[:] = pypto.add(
                        pypto.mul(alpha, pypto.view(oi_buf, [128, 512], [0, 0])),
                        pypto.mul(beta, q1),
                    )                                                 # [128, 512] fp32
                    mi_buf[:] = mi_new                                 # [128, 1] fp32

                # [j] Last tile: normalize and write back
                if pypto.is_loop_end(s2_idx):
                    oi_final = pypto.view(oi_buf, [128, 512], [0, 0])   # [128, 512] fp32
                    li_final = pypto.view(li_buf, [128, 1], [0, 0])     # [128, 1] fp32
                    oi_norm = pypto.cast(
                        pypto.div(oi_final, li_final),
                        pypto.DT_BF16,
                    )                                                 # [128, 512] bf16
                    pypto.assemble(oi_norm, [q_off, 0], y0)           # → y0[M, 512]


# =============================================================================
# Layer K — Host wrapper
# =============================================================================

def ds_v32_sparse_attention_antiquant_wrapper(
    q_nope, q_rope, nope_cache, topk_idx, block_table, act_seq,
    **kwargs,
):
    """Host wrapper: concat, allocate, call JIT once, and restore the shape."""
    nq = kwargs.get("nq", 128)
    kv_lora_rank = kwargs.get("kv_lora_rank", 512)
    qk_rope_dim = kwargs.get("qk_rope_dim", 64)
    softmax_scale = kwargs.get("softmax_scale", 0.041666666666666664)
    topk = kwargs.get("topk", 2048)
    block_size = kwargs.get("block_size", 128)
    device = q_nope.device
    M = q_nope.shape[0]
    b = act_seq.shape[0]
    s1_count = M // (nq * b)

    q = torch.cat([q_nope, q_rope], dim=-1)  # [M, 576] bf16

    y0 = torch.zeros(M, kv_lora_rank, dtype=torch.bfloat16, device=device)

    ds_v32_sparse_attention_antiquant_kernel_npu(
        q, nope_cache, topk_idx, block_table, act_seq, y0,
        nq, kv_lora_rank, qk_rope_dim, softmax_scale, topk, block_size,
        b, s1_count,
    )

    return y0


# =============================================================================
# Layer L: ModelNew bridge (KernelBench interface specification)
# =============================================================================

class ModelNew(torch.nn.Module):
    """KernelBench-compatible entry point forwarding to the wrapper."""
    def __init__(self, nq=128, n_kv=1, kv_lora_rank=512, qk_rope_dim=64,
                 softmax_scale=0.041666666666666664, topk=2048, block_size=128):
        super().__init__()
        self.nq = nq
        self.n_kv = n_kv
        self.kv_lora_rank = kv_lora_rank
        self.qk_rope_dim = qk_rope_dim
        self.softmax_scale = softmax_scale
        self.topk = topk
        self.block_size = block_size

    def forward(self, q_nope, q_rope, nope_cache, topk_idx, block_table, act_seq):
        return ds_v32_sparse_attention_antiquant_wrapper(
            q_nope, q_rope, nope_cache, topk_idx, block_table, act_seq,
            nq=self.nq, n_kv=self.n_kv, kv_lora_rank=self.kv_lora_rank,
            qk_rope_dim=self.qk_rope_dim, softmax_scale=self.softmax_scale,
            topk=self.topk, block_size=self.block_size,
        )


if __name__ == "__main__":
    print("ds_v32_sparse_attention_antiquant_impl.py — run via verifier's test")