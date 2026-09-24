# Integrated production kernel for mla_prolog (consolidated from modules/mla_prolog_module1234_impl.py).
# Exports mla_prolog_wrapper. Same kernel logic as the final staged module.
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2024-2026. All rights reserved.
# =============================================================================
# mla_prolog_module1234_impl.py — Phase M1+M2+M3+M4 cumulative (full kernel)
#
# M1 (Q down-proj + RMSNorm + up-proj):
#   1. hidden_states @ w_dq           → q_a     [M, 1536] BF16
#   2. RMSNorm(q_a, gamma_cq)         → q_a_norm [M, 1536] BF16  (FINAL output)
#   3. q_a_norm @ w_uqqr              → q_b     [M, 24576] BF16 (consumed by M2)
#
# M2 (Q split nope/rope → nope@w_uk + RoPE):
#   4. reshape(q_b, [M, 128, 192])    → q_reshape [M, 128, 192]
#   5. view(nope part)                → q_nope_raw [M, 128, 128]
#   6. permute+matmul(+set_matrix_size)+permute → q_nope [M, 128, 512] (FINAL)
#   7. view(rope part)                → q_rope_raw [M, 128, 64]
#   8. interleaved RoPE(cos, sin)     → q_rope [M, 128, 64]   (FINAL output)
#
# M3 (KV projection + RMSNorm + RoPE):
#   9.  hidden_states @ w_dkvkr       → kv      [M, 576] BF16
#   10. view(kv[:,:512])              → compressed_kv [M, 512]
#   11. RMSNorm(compressed_kv, gamma_ckv) → k_nope [M, 512] (FINAL)
#   12. view(kv[:,512:])              → k_rope_raw [M, 64]
#   13. interleaved RoPE(cos, sin)    → k_rope [M, 64]    (FINAL output)
#
# M4 (PageAttention Scatter Update) — in HOST WRAPPER (torch.index_copy_):
#   pypto.scatter_update does NOT respect valid_shape (iterates full index
#   storage shape) → OOB GM read on unpadded cache_index → MTE 507015.
#   Moved out of the kernel: the wrapper scatters k_nope/k_rope rows into
#   kv_cache/kr_cache via torch.index_copy_ (bit-exact with golden) and
#   returns them as kv_cache_out / kr_cache_out (FINAL).
#
# Dynamic axis: M (b*s1). Loop-unrolled along M with tile_bs=8.
# Weights w_dq, w_uqqr, w_dkvkr are in TILEOP_NZ format; w_uk is ND (3D batch weight, no NZ needed).
#
# Kernel FINAL outputs (5): q_nope, q_rope, k_nope, k_rope, q_a_norm.
# Wrapper returns 7 (adds kv_cache_out, kr_cache_out via torch scatter).
# =============================================================================

import pypto
import torch
import torch_npu  # noqa: F401  required for NPU device init


# =============================================================================
# Layer H — PyPTO sub-kernels (one logical step each)
# Each sub-kernel sets its own tile shapes for optimal per-stage tiling.
# =============================================================================


def pypto_stage_down_proj(x_tile, w_dq):
    """Stage C1 (M1): x_tile @ w_dq → q_a  [8, 7168] @ [7168, 1536] → [8, 1536] BF16

    SWIMLANE S-11: Split K across cores (enable_split_k=True); M1_C1 has small M=8, N=1536,
    but a very large K=7168 (TileShape decision tree: "small M,N and K>4096 -> enable_split_k=True").
    The original implementation serializes 28 K chunks on one AIC core (~93 us, 5% core utilization).
    split-k requires FP32/INT32 out_dtype, so matmul outputs FP32 before casting back to BF16.
    (FP32 accumulation is more precise than BF16 accumulation.)
    """
    pypto.set_semantic_label("M1_C1")
    pypto.set_cube_tile_shapes([16, 16], [128, 1024], [64, 256], enable_split_k=True)
    q_a_f32 = pypto.matmul(x_tile, w_dq, pypto.DT_FP32)  # [8, 1536] FP32 (split-K partial sums)
    # Matmul out_dtype FP32 need cast back to BF16 for RMSNorm
    pypto.set_vec_tile_shapes(8, 1536)
    return pypto.cast(q_a_f32, pypto.DT_BF16)            # [8, 1536] BF16


def pypto_stage_rms_norm(q_a, gamma_cq):
    """Stage V (M1): RMSNorm(q_a, gamma_cq) → q_a_norm  [8, 1536] BF16 (epsilon=0.0)"""
    pypto.set_semantic_label("M1_V")
    pypto.set_vec_tile_shapes(8, 1536)
    return pypto.rms_norm(q_a, gamma_cq, epsilon=0.0)  # [8, 1536] BF16


def pypto_stage_up_proj(q_a_norm_tile, w_uqqr):
    """Stage C2 (M1): q_a_norm_tile @ w_uqqr → q_b_tile  [8,1536] @ [1536,24576] → [8,24576] BF16"""
    pypto.set_semantic_label("M1_C2")
    pypto.set_cube_tile_shapes([16, 16], [128, 512], [64, 1024])
    return pypto.matmul(q_a_norm_tile, w_uqqr, pypto.DT_BF16)  # [8, 24576] BF16


def pypto_stage_q_nope(q_reshape, w_uk_0, w_uk_1, w_uk_2, w_uk_3, q_nope_out, bs_offset):
    """Stage C (M2): per-32-head-group transposed_batchmatmul

    F-22: nope-weight slicing moved OUT of the kernel into the host wrapper.
    The kernel consumes 4 pre-sliced whole tensors w_uk_g [32,128,512] instead
    of a 4096KB pypto.view on the full w_uk [128,128,512]. The 128-head axis is
    split into 4 groups of 32 heads; group g computes:
        q_g [8,32,128] (M,B,K) → transposed_batchmatmul(w_uk_g (B,K,N))
        → [8,32,512] (M,B,N) → assemble into q_nope_out at [bs_offset, g*32, 0]
    """
    pypto.set_semantic_label("M2_C")
    pypto.set_cube_tile_shapes([16, 16], [128, 128], [64, 128])
    w_uk_gs = (w_uk_0, w_uk_1, w_uk_2, w_uk_3)
    # g is a plain Python int (OL57: for...in range allowed in JIT graph).
    for g in range(4):
        # Nope part of q_reshape [8,128,192]: heads g*32:(g+1)*32, features :128
        q_g = pypto.view(q_reshape, [8, 32, 128], [0, g * 32, 0])  # [8, 32, 128] BF16 = (M, B, K)
        # transposed_batchmatmul folds the permute(1,0,2)→reshape(contig)→matmul→
        # permute(1,0,2) chain into one op, eliminating 2 permute copies + 1
        # force-contiguous reshape: (M,B,K)=(8,32,128) → transpose→(B,M,K) →
        # @w_uk_g (B,K,N)=(32,128,512) → (B,M,N) → transpose back → (M,B,N)=(8,32,512).
        q_g_out = pypto.experimental.transposed_batchmatmul(q_g, w_uk_gs[g], pypto.DT_BF16)  # [8, 32, 512] BF16
        pypto.set_vec_tile_shapes(1, 8, 512)
        # Assemble this 32-head group into q_nope_out at [bs_offset, g*32, 0]
        pypto.assemble(q_g_out, [bs_offset, g * 32, 0], q_nope_out)


def pypto_stage_rope_3d(q_rope_raw, cos_slice, sin_slice, q_rope_out, bs_offset):
    """Stage V (M2): interleaved RoPE for 3D input [8, n, d] in FP32.

    Input:  q_rope_raw  [8, 128, 64] BF16 — interleaved layout
    Output: written into q_rope_out [M, 128, 64] BF16 at [bs_offset, :, :]
            (deinterleaved + rotated)

    Steps (all in FP32):
        cast → deinterleave (reshape→permute→reshape)
        → rotate_half (neg+concat)
        → out = x*cos + rotate_half(x)*sin
        → cast back to BF16

    UB budget: the full [8,128,64] FP32 intermediate (256KB) exceeds
    MEM_UB (192KB). Even 64-head groups push the ADD op past UB (mul
    intermediates on both sides of the + are simultaneously live).
    The 128-head axis is split into 8 groups of 16 heads, so each FP32
    tensor is [8,16,64] = 32KB << 192KB (DESIGN.md contingency: "Split the 128 heads
    along the n axis"; refined to groups of 16 heads).
    """
    pypto.set_semantic_label("M2_V")

    d = 64
    half = 32
    num_groups = 32
    heads_per_group = 128 // num_groups  # 4

    # Broadcast cos/sin once: [8, 64] → [8, 1, 64] FP32
    cos_f32 = pypto.cast(cos_slice, pypto.DT_FP32)   # [8, 64] FP32
    sin_f32 = pypto.cast(sin_slice, pypto.DT_FP32)   # [8, 64] FP32
    pypto.set_vec_tile_shapes(8, 1, d)
    cos_3d = pypto.reshape(cos_f32, [8, 1, d])       # [8, 1, 64] FP32
    sin_3d = pypto.reshape(sin_f32, [8, 1, d])       # [8, 1, 64] FP32

    # Split the 128-head axis into 8 groups of 16 heads (UB budget).
    # g is a plain Python int (OL57: for...in range allowed in JIT graph).
    for g in range(num_groups):
        head_off = g * heads_per_group

        pypto.set_vec_tile_shapes(8, heads_per_group, d)
        x_g = pypto.view(q_rope_raw, [8, heads_per_group, d],
                         [0, head_off, 0])            # [8, 16, 64] BF16

        # FP32 computation (SPEC §7: q_rope atol=0.005, rtol=0.0078125)
        x_f32 = pypto.cast(x_g, pypto.DT_FP32)       # [8, 16, 64] FP32

        # Deinterleave: [8,16,32,2] → permute(0,1,3,2) → [8,16,2,32] → reshape [8,16,64]
        pypto.set_vec_tile_shapes(8, heads_per_group, 2, half)
        x_d2 = pypto.reshape(x_f32, [8, heads_per_group, half, 2])  # [8, 16, 32, 2] FP32
        x_deint = pypto.permute(x_d2, [0, 1, 3, 2])                 # [8, 16, 2, 32] FP32
        pypto.set_vec_tile_shapes(8, heads_per_group, d)
        x_deint = pypto.reshape(x_deint, [8, heads_per_group, d])    # [8, 16, 64] FP32

        # Rotate half: split → neg(x2) concat x1
        pypto.set_vec_tile_shapes(8, heads_per_group, half)
        x1 = pypto.view(x_deint, [8, heads_per_group, half],
                        [0, 0, 0])                                   # [8, 16, 32] FP32
        x2 = pypto.view(x_deint, [8, heads_per_group, half],
                        [0, 0, half])                                # [8, 16, 32] FP32
        neg_x2 = pypto.neg(x2)                                       # [8, 16, 32] FP32
        pypto.set_vec_tile_shapes(8, heads_per_group, d)
        rotated = pypto.concat([neg_x2, x1], dim=-1)                 # [8, 16, 64] FP32

        # out = x_deint * cos + rotated * sin  (each intermediate ≪ 192KB UB)
        out = pypto.add(
            pypto.mul(x_deint, cos_3d), pypto.mul(rotated, sin_3d)
        )                                                            # [8, 16, 64] FP32

        out_bf16 = pypto.cast(out, pypto.DT_BF16) + 0.0              # [8, 16, 64] BF16

        # Assemble this 16-head group into q_rope_out at [bs_offset, head_off, 0]
        pypto.assemble(out_bf16, [bs_offset, head_off, 0], q_rope_out)


def pypto_stage_kv_proj(x_tile, w_dkvkr):
    """Stage C (M3): x_tile @ w_dkvkr → kv_tile  [8, 7168] @ [7168, 576] → [8, 576] BF16

    w_dkvkr [7168, 576] NZ. N=576 = 3×192, nL1=192 divides N exactly
    (192%64=0, L0B=48KB ≤ 64KB ✓).

    set_matrix_size([8, 7168, 576]) is REQUIRED: pypto.set_matrix_size is
    stage-global state and M2_C (pypto_stage_q_nope) set it to [8,128,512].
    Without overriding it here, the M3 cube engine reads w_dkvkr NZ with the
    stale (128,512) logical shape → wrong fractal blocks → sign-flipped /
    garbage kv_tile → k_nope/k_rope 97-98% OOT. Args are [m, k, n]: K=7168
    (inner dim), N=576 (output dim), matching w_dkvkr's logical shape.
    """
    pypto.set_semantic_label("M3_C")
    pypto.set_cube_tile_shapes([16, 16], [128, 1024], [64, 192])
    pypto.set_matrix_size([8, 7168, 576])  # override stale [8,128,512] from M2_C
    return pypto.matmul(x_tile, w_dkvkr, pypto.DT_BF16)  # [8, 576] BF16


def pypto_stage_kv_rms_norm(compressed_kv, gamma_ckv):
    """Stage V (M3): RMSNorm(compressed_kv, gamma_ckv) → k_nope  [8, 512] BF16 (epsilon=0.0)"""
    pypto.set_semantic_label("M3_V")
    pypto.set_vec_tile_shapes(8, 512)
    return pypto.rms_norm(compressed_kv, gamma_ckv, epsilon=0.0)  # [8, 512] BF16


def pypto_stage_rope_2d(k_rope_raw, cos_slice, sin_slice):
    """Stage V (M3): interleaved RoPE for 2D input [8, 64] in FP32 (one-shot).

    Input:  k_rope_raw  [8, 64] BF16 — interleaved layout (from kv[:,512:])
    Output: k_rope_tile [8, 64] BF16 — deinterleaved + rotated

    ⚠️ Returns the tile (no internal assemble). The caller must scatter + assemble
        separately. This avoids same-iteration assemble→view RAW that would place
        the loop index in GM address space (Coord2Dim codegen issue).

    UB: full [8,64] FP32 = 2KB ≪ 192KB.
    """
    pypto.set_semantic_label("M3_V")

    x_f32 = pypto.cast(k_rope_raw, pypto.DT_FP32)                    # [8, 64] FP32

    # Deinterleave: reshape[8,32,2] → permute(0,2,1) → reshape[8,64]
    pypto.set_vec_tile_shapes(8, 32, 2)
    x_d2 = pypto.reshape(x_f32, [8, 32, 2])                          # [8, 32, 2] FP32
    pypto.set_vec_tile_shapes(8, 2, 32)
    x_deint = pypto.permute(x_d2, [0, 2, 1])                         # [8, 2, 32] FP32
    pypto.set_vec_tile_shapes(8, 64)
    x_deint = pypto.reshape(x_deint, [8, 64])                         # [8, 64] FP32

    # Rotate half: split → neg(x2) concat x1
    pypto.set_vec_tile_shapes(8, 32)
    x1 = pypto.view(x_deint, [8, 32], [0, 0])                        # [8, 32] FP32 — first half
    x2 = pypto.view(x_deint, [8, 32], [0, 32])                       # [8, 32] FP32 — second half
    neg_x2 = pypto.neg(x2)                                            # [8, 32] FP32
    pypto.set_vec_tile_shapes(8, 64)
    x_rot = pypto.concat([neg_x2, x1], dim=-1)                       # [8, 64] FP32

    # Apply RoPE: out = x * cos + rotate_half(x) * sin
    cos_f32 = pypto.cast(cos_slice, pypto.DT_FP32)                   # [8, 64] FP32
    sin_f32 = pypto.cast(sin_slice, pypto.DT_FP32)                   # [8, 64] FP32
    out = pypto.add(
        pypto.mul(x_deint, cos_f32), pypto.mul(x_rot, sin_f32)
    )                                                                  # [8, 64] FP32

    return pypto.cast(out, pypto.DT_BF16) + 0.0                      # [8, 64] BF16


# =============================================================================
# Layer I — Kernel implementation (no @pypto.frontend.jit here)
# Owns all pypto.loop calls. Reads as a high-level recipe.
# =============================================================================

def _mla_prolog_module1234_kernel_impl(
    hidden_states, cos, sin, w_dq, w_uqqr, w_uk_0, w_uk_1, w_uk_2, w_uk_3, w_dkvkr, gamma_cq, gamma_ckv,
    q_a_norm_out, q_nope_out, k_nope_out, k_rope_out, q_rope_out,
    # <<< SNAPSHOT:SIG_IMPL
    # >>> SNAPSHOT:SIG_IMPL
):
    """M1+M2+M3+M4 kernel impl: TWO sequential M-axis loops; M4 scatter in
    host wrapper.

    INCORE I-2: Batch NONE_CACHEABLE on all 4 large weights (~123MB total,
    read-once per M-tile). Avoid L2 pollution so activations and intermediate
    data benefit from L2 cache capacity. Pangu 7B case: -19.1% with same
    pattern (ref: pypto-op-perf-tune tune-incore cases/weight-none-l2-cacheable).

    Loop 1 (MLA_M12_LOOP): M1 produces q_a_norm (FINAL) and q_b (intermediate
        for M2); M2 consumes q_b → reshape → split nope/rope → q_nope/q_rope
        (both FINAL). q_rope_out writeback completes inside loop 1.
    Loop 2 (MLA_M3_LOOP): M3 re-views hidden_states/cos/sin from GM (no
        dependency on loop 1) → kv proj → split → k_nope/k_rope (both FINAL).
        M4 PageAttention scatter is NOT in this loop — it runs in the host
        wrapper via torch.index_copy_ (bit-exact with golden).

    M4 scatter (host wrapper): pypto.scatter_update iterates the FULL index
        storage shape and ignores valid_shape — for wrapper-padded cases
        (M=1→8, M=4→8) the [8,1] cache_index view reads OOB GM → MTE 507015
        (aicore 507015). The wrapper scatters the real k_nope[:M]/k_rope[:M]
        rows into cloned flattened 2D cache buffers with torch.index_copy_,
        then reshapes back to 4D — bit-exact with golden's _scatter_update_4d.
        The kernel takes no cache_index / cache / real_M params.

    Structural split rationale (debugger cycle 4): M3's large workspace
        previously shared one fused pypto.loop_unroll with M1+M2; at M≥16 it
        re-contaminated q_rope_out. Two sequential loops create a hard liveness
        boundary; keeping M4 out of the kernel preserves that boundary and
        avoids the cross-loop assemble→view unreliability (machine.md §10).

    Args:
        hidden_states:  [M, 7168] BF16 — full input
        cos:            [M, 64] BF16 — RoPE cosine
        sin:            [M, 64] BF16 — RoPE sine
        w_dq:           [7168, 1536] BF16 NZ
        w_uqqr:         [1536, 24576] BF16 NZ
        w_uk_0..w_uk_3: [32, 128, 512] BF16 ND — 4 pre-sliced batch weights
                       (host splits w_uk [128,128,512] along dim0 into groups
                       g*32:(g+1)*32; kernel consumes whole tensors, no view)
        w_dkvkr:        [7168, 576] BF16 NZ — KV projection weight
        gamma_cq:       [1536] BF16 — RMSNorm scale for Q
        gamma_ckv:      [512] BF16 — RMSNorm scale for KV
        q_a_norm_out:   [M, 1536] BF16 — output buffer (M1 FINAL)
        q_nope_out:     [M, 128, 512] BF16 — output buffer (M2 FINAL)
        k_nope_out:     [M, 512] BF16 — output buffer (M3 FINAL)
        k_rope_out:     [M, 64] BF16 — output buffer (M3 FINAL)
        q_rope_out:     [M, 128, 64] BF16 — output buffer (M2 FINAL, last param)
    """
    # INCORE I-2: Batch NONE_CACHEABLE on all 4 large weights (~123MB total,
    # read-once per M-tile). Avoid L2 pollution for activations.
    w_dq.set_cache_policy(pypto.CachePolicy.NONE_CACHEABLE, True)
    w_uqqr.set_cache_policy(pypto.CachePolicy.NONE_CACHEABLE, True)
    w_uk_0.set_cache_policy(pypto.CachePolicy.NONE_CACHEABLE, True)
    w_uk_1.set_cache_policy(pypto.CachePolicy.NONE_CACHEABLE, True)
    w_uk_2.set_cache_policy(pypto.CachePolicy.NONE_CACHEABLE, True)
    w_uk_3.set_cache_policy(pypto.CachePolicy.NONE_CACHEABLE, True)
    w_dkvkr.set_cache_policy(pypto.CachePolicy.NONE_CACHEABLE, True)

    token_count = hidden_states.shape[0]  # M (SymbolicScalar)

    # F-15: combine_axis — [8,16,64] × [8,1,64] broadcasts in RoPE 3D/2D
    # use brcb inline instead of full broadcast expansion.
    pypto.experimental.set_operation_options(combine_axis=True)

    # ════════════════════════════════════════════════════════════
    # Loop 1: M1+M2 (Q path) — q_rope writeback completes here
    # ════════════════════════════════════════════════════════════

    for bs_offset, _ in pypto.loop_unroll(
        0, token_count, 1, name="MLA_M12_LOOP", idx_name="bs_offset",
        unroll_list=[8], submit_before_loop=True,
    ):
        # Tail block: valid_len = min(remaining, tile_bs=8)
        valid_len = (token_count - bs_offset).min(8)

        # Slice input tiles from the M axis
        x_tile = pypto.view(hidden_states, [8, 7168], [bs_offset, 0],
                            valid_shape=[valid_len, 7168])  # [8, 7168] BF16
        cos_slice = pypto.view(cos, [8, 64], [bs_offset, 0])   # [8, 64] BF16
        sin_slice = pypto.view(sin, [8, 64], [bs_offset, 0])   # [8, 64] BF16

        # <<< SNAPSHOT:before_nt_loop
        # >>> SNAPSHOT:before_nt_loop

        # ════════════════════════════════════════════════════════════
        # Module M1: Q down-proj + RMSNorm + up-proj
        # ════════════════════════════════════════════════════════════

        # Stage 1: down projection  [8, 7168] @ [7168, 1536] → [8, 1536]
        q_a = pypto_stage_down_proj(x_tile, w_dq)  # [8, 1536] BF16

        # Stage 2: RMSNorm  [8, 1536] → [8, 1536]
        q_a_norm_tile = pypto_stage_rms_norm(q_a, gamma_cq)  # [8, 1536] BF16

        # Stage 3: up projection  [8, 1536] @ [1536, 24576] → [8, 24576]
        q_b_tile = pypto_stage_up_proj(q_a_norm_tile, w_uqqr)  # [8, 24576] BF16

        # ════════════════════════════════════════════════════════════
        # Module M2: Q split nope/rope → nope@w_uk + RoPE
        # ════════════════════════════════════════════════════════════

        # Step 4: reshape q_b → [8, 128, 192] (inplace: q_b_tile no longer needed)
        q_reshape = pypto.reshape(q_b_tile, [8, 128, 192], inplace=True)  # [8, 128, 192]

        # F-22: nope-part view on q_reshape removed — pypto_stage_q_nope slices
        # 4×[8,32,128] head groups directly and assembles into q_nope_out.

        # Step 7: view rope part [8, 128, 192] → [8, 128, 64]
        q_rope_raw = pypto.view(q_reshape, [8, 128, 64], [0, 0, 128])  # [8, 128, 64] BF16

        # INCORE I-8/I-3: Move M2_V (RoPE) before M2_C (nope matmul).
        # M2_V uses AIV and M2_C uses AIC; they are independent and consume
        # different views of q_reshape. Previously M2_V followed M2_C, waiting about 17 us for Cube completion.
        # Moving M2_V earlier overlaps AIV and AIC execution and removes its wait for M2_C.
        # Step 8: interleaved RoPE (3D)  [8, 128, 64] BF16 → FP32 → BF16
        # Head axis split into 8×16-head groups inside; each group assembled
        # directly into q_rope_out at [bs_offset, g*16, 0].
        pypto_stage_rope_3d(q_rope_raw, cos_slice, sin_slice, q_rope_out, bs_offset)

        # Step 6: per-32-head-group permute+matmul+permute with pre-sliced w_uk
        #   [8,32,128] @ [32,128,512] → [8,32,512] ×4 → assembled [8,128,512]
        pypto_stage_q_nope(q_reshape, w_uk_0, w_uk_1, w_uk_2, w_uk_3, q_nope_out, bs_offset)

        # Write Q-path output tiles (assemble to full output buffers)
        pypto.assemble(q_a_norm_tile, [bs_offset, 0], q_a_norm_out)
        # q_nope assembled inside pypto_stage_q_nope; q_rope assembled inside
        # pypto_stage_rope_3d (loop 1 completes Q path)

    # ════════════════════════════════════════════════════════════
    # Loop 2: M3 (KV path) — re-views inputs from GM, no dep on loop 1
    # ════════════════════════════════════════════════════════════

    for bs_offset, _ in pypto.loop_unroll(
        0, token_count, 1, name="MLA_M3_LOOP", idx_name="bs_offset",
        unroll_list=[8], submit_before_loop=True,
    ):
        # Tail block: valid_len = min(remaining, tile_bs=8)
        valid_len = (token_count - bs_offset).min(8)

        # Re-view input tiles from GM (hard liveness boundary: no dependency
        # on loop 1 — M3 does not consume any M1/M2 intermediate)
        x_tile = pypto.view(hidden_states, [8, 7168], [bs_offset, 0],
                            valid_shape=[valid_len, 7168])  # [8, 7168] BF16
        cos_slice = pypto.view(cos, [8, 64], [bs_offset, 0])   # [8, 64] BF16
        sin_slice = pypto.view(sin, [8, 64], [bs_offset, 0])   # [8, 64] BF16

        # ════════════════════════════════════════════════════════════
        # Module M3: KV projection + RMSNorm + RoPE
        # ════════════════════════════════════════════════════════════

        # Step 9: matmul(x_tile, w_dkvkr) → kv_tile  [8, 7168] @ [7168, 576] → [8, 576]
        # Tile shapes re-set inside pypto_stage_kv_proj (loop boundary does not
        # preserve tile state): cube [8,8]/[128,1024]/[64,192] + set_matrix_size
        # [8,7168,576] override of M2_C's stale [8,128,512].
        kv_tile = pypto_stage_kv_proj(x_tile, w_dkvkr)  # [8, 576] BF16

        # Step 10: view compressed_kv  [8, 576] → [8, 512]
        compressed_kv = pypto.view(kv_tile, [8, 512], [0, 0])

        # S-4: Standard graph fusion around consecutive M3 vector stages (RMSNorm + RoPE, no Cube ops)
        if pypto.platform.npuarch == 'DAV_2201':
            pypto.set_pass_options(sg_set_scope=1)

        # Step 11: RMSNorm(compressed_kv, gamma_ckv) → k_nope  [8, 512] BF16 (FINAL)
        k_nope_tile = pypto_stage_kv_rms_norm(compressed_kv, gamma_ckv)  # [8, 512] BF16

        # Step 12: view k_rope_raw  [8, 576] → [8, 64]
        k_rope_raw = pypto.view(kv_tile, [8, 64], [0, 512])

        # Step 13: interleaved RoPE (2D) — returns live k_rope_tile
        k_rope_tile = pypto_stage_rope_2d(k_rope_raw, cos_slice, sin_slice)  # [8, 64] BF16

        if pypto.platform.npuarch == 'DAV_2201':
            pypto.set_pass_options(sg_set_scope=-1)

        # Write KV-path output tiles (assemble to full output buffers)
        pypto.assemble(k_nope_tile, [bs_offset, 0], k_nope_out)
        pypto.assemble(k_rope_tile, [bs_offset, 0], k_rope_out)


# =============================================================================
# Layer J — JIT entry
# @pypto.frontend.jit with static signatures, runtime_options.
# =============================================================================

@pypto.frontend.jit(
    runtime_options={
        "run_mode": pypto.RunMode.NPU,
        "max_workspace_kb": 249184,
        "device_sched_mode": 3,
    },
    pass_options={
        "vec_nbuffer_setting": {"func8_7": 8},
    },
    host_options={"compile_monitor_enable": 0},
    debug_options={"runtime_debug_mode": 1},
    )
def mla_prolog_module1234_kernel_npu(
    hidden_states: pypto.Tensor([pypto.DYNAMIC, 7168], pypto.DT_BF16),
    cos: pypto.Tensor([pypto.DYNAMIC, 64], pypto.DT_BF16),
    sin: pypto.Tensor([pypto.DYNAMIC, 64], pypto.DT_BF16),
    w_dq: pypto.Tensor([7168, 1536], pypto.DT_BF16, format=pypto.TileOpFormat.TILEOP_NZ),
    w_uqqr: pypto.Tensor([1536, 24576], pypto.DT_BF16, format=pypto.TileOpFormat.TILEOP_NZ),
    w_uk_0: pypto.Tensor([32, 128, 512], pypto.DT_BF16),
    w_uk_1: pypto.Tensor([32, 128, 512], pypto.DT_BF16),
    w_uk_2: pypto.Tensor([32, 128, 512], pypto.DT_BF16),
    w_uk_3: pypto.Tensor([32, 128, 512], pypto.DT_BF16),
    w_dkvkr: pypto.Tensor([7168, 576], pypto.DT_BF16, format=pypto.TileOpFormat.TILEOP_NZ),
    gamma_cq: pypto.Tensor([1536], pypto.DT_BF16),
    gamma_ckv: pypto.Tensor([512], pypto.DT_BF16),
    q_a_norm_out: pypto.Tensor([pypto.DYNAMIC, 1536], pypto.DT_BF16),
    q_nope_out: pypto.Tensor([pypto.DYNAMIC, 128, 512], pypto.DT_BF16),
    k_nope_out: pypto.Tensor([pypto.DYNAMIC, 512], pypto.DT_BF16),
    k_rope_out: pypto.Tensor([pypto.DYNAMIC, 64], pypto.DT_BF16),
    q_rope_out: pypto.Tensor([pypto.DYNAMIC, 128, 64], pypto.DT_BF16),
    # <<< SNAPSHOT:SIG_JIT
    # >>> SNAPSHOT:SIG_JIT
):
    """JIT entry for M1+M2+M3+M4: full MLA Prolog kernel (5 outputs).

    Tensor args only (OL26). Output buffers written in-place via
    pypto.assemble (two sequential loops). M4 scatter is handled in the host
    wrapper (torch.index_copy_), NOT here — pypto.scatter_update ignores
    valid_shape and reads the full index storage shape (OOB → MTE 507015).
    Dynamic axis M marked with pypto.DYNAMIC. Weights declared with TILEOP_NZ.
    """
    _mla_prolog_module1234_kernel_impl(
        hidden_states, cos, sin, w_dq, w_uqqr, w_uk_0, w_uk_1, w_uk_2, w_uk_3, w_dkvkr, gamma_cq, gamma_ckv,
        q_a_norm_out, q_nope_out, k_nope_out, k_rope_out, q_rope_out,
        # <<< SNAPSHOT:CALL_IMPL
        # >>> SNAPSHOT:CALL_IMPL
    )


# =============================================================================
# Layer K — Host wrapper (public entry of this module file)
# Responsibilities (only 4):
#   1. Move tensors to device / set layout (pure torch).
#   2. Allocate output buffers via torch.empty(...).
#   3. Invoke the JIT entry ONCE.
#   4. Reshape outputs back.
# =============================================================================

def mla_prolog_wrapper(
    hidden_states,
    cos,
    sin,
    kv_cache,
    kr_cache,
    cache_index,
    w_dq,
    w_uqqr,
    w_uk,
    w_dkvkr,
    gamma_cq,
    gamma_ckv,
    **kwargs,
):
    """M1+M2+M3+M4 host wrapper: full MLA Prolog kernel (7 outputs).

    Signature matches the runner's module-aware filtered PRIMARY_INPUT_ORDER
    for modules[0..3] (M1+M2+M3+M4): all 12 primary inputs in declared order.
    Accepts **kwargs for snapshot generator compatibility.

    Kernel produces 5 outputs (q_nope, q_rope, k_nope, k_rope, q_a_norm).
    M4 scatter runs HERE in the wrapper via torch.index_copy_ (bit-exact with
    golden's _scatter_update_4d): pypto.scatter_update cannot be used safely
    because it ignores valid_shape and reads the full index storage shape —
    for M=1/M=4 (padded to 8) the [8,1] cache_index view reads OOB GM → MTE
    507015.

    Args:
        hidden_states: [M, 7168] BF16 torch tensor on NPU
        cos:           [M, 64] BF16 torch tensor on NPU
        sin:           [M, 64] BF16 torch tensor on NPU
        kv_cache:      [block_num, 128, 1, 512] BF16 torch tensor on NPU
        kr_cache:      [block_num, 128, 1, 64] BF16 torch tensor on NPU
        cache_index:   [b, s1] INT64 torch tensor on NPU
        w_dq:          [7168, 1536] BF16 NZ torch tensor on NPU
        w_uqqr:        [1536, 24576] BF16 NZ torch tensor on NPU
        w_uk:          [128, 128, 512] BF16 ND torch tensor on NPU
        w_dkvkr:       [7168, 576] BF16 NZ torch tensor on NPU
        gamma_cq:      [1536] BF16 torch tensor on NPU
        gamma_ckv:     [512] BF16 torch tensor on NPU

    Returns:
        q_nope:       [M, 128, 512] BF16 — M2 FINAL output (nope @ w_uk)
        q_rope:       [M, 128, 64] BF16 — M2 FINAL output (interleaved RoPE)
        k_nope:       [M, 512] BF16 — M3 FINAL output (KV RMSNorm)
        k_rope:       [M, 64] BF16 — M3 FINAL output (KV interleaved RoPE)
        q_a_norm:     [M, 1536] BF16 — M1 FINAL output (RMSNorm result)
        kv_cache_out: [block_num, 128, 1, 512] BF16 — M4 FINAL output
        kr_cache_out: [block_num, 128, 1, 64] BF16 — M4 FINAL output

    Note: Return tuple order matches the golden's output order for M1+M2+M3+M4
    (q_nope, q_rope, k_nope, k_rope, q_a_norm, kv_cache_out, kr_cache_out) so
    the adversarial runner's positional comparison succeeds.
    """
    M = hidden_states.shape[0]
    P = 8  # tile_bs

    # F-22: split w_uk [128,128,512] into 4 contiguous [32,128,512] slices on
    # the host. The kernel consumes whole pre-sliced tensors instead of a 4096KB
    # pypto.view on w_uk. Public wrapper signature (single w_uk) is unchanged.
    w_uk_0 = w_uk[0:32].contiguous()
    w_uk_1 = w_uk[32:64].contiguous()
    w_uk_2 = w_uk[64:96].contiguous()
    w_uk_3 = w_uk[96:128].contiguous()

    if M % P != 0:
        padded_M = (M + P - 1) // P * P
        # Pad input tensors (zeros beyond the valid rows to avoid OOB in assemble)
        hs_pad = torch.empty(padded_M, 7168, dtype=torch.bfloat16,
                             device=hidden_states.device)
        hs_pad[:M] = hidden_states
        hs_pad[M:] = 0
        cos_pad = torch.empty(padded_M, 64, dtype=torch.bfloat16,
                              device=hidden_states.device)
        cos_pad[:M] = cos
        cos_pad[M:] = 0
        sin_pad = torch.empty(padded_M, 64, dtype=torch.bfloat16,
                              device=hidden_states.device)
        sin_pad[:M] = sin
        sin_pad[M:] = 0
        # Allocate padded output buffers (OL58: use torch.*)
        # Fix 3: reorder k_nope/k_rope before q_rope, add guard padding between
        # k_rope and q_rope to physically isolate q_rope from M3 workspace aliasing.
        q_a_norm = torch.empty(padded_M, 1536, dtype=torch.bfloat16,
                               device=hidden_states.device)
        q_nope = torch.empty(padded_M, 128, 512, dtype=torch.bfloat16,
                             device=hidden_states.device)
        k_nope = torch.empty(padded_M, 512, dtype=torch.bfloat16,
                             device=hidden_states.device)
        k_rope = torch.empty(padded_M, 64, dtype=torch.bfloat16,
                             device=hidden_states.device)
        _guard = torch.empty(1024, dtype=torch.bfloat16,
                             device=hidden_states.device)
        q_rope = torch.empty(padded_M, 128, 64, dtype=torch.bfloat16,
                             device=hidden_states.device)  # ← last = farthest from M3 scratch
        # Single JIT call on padded tensors (5 outputs)
        mla_prolog_module1234_kernel_npu(
            hs_pad, cos_pad, sin_pad, w_dq, w_uqqr, w_uk_0, w_uk_1, w_uk_2, w_uk_3, w_dkvkr, gamma_cq, gamma_ckv,
            q_a_norm, q_nope, k_nope, k_rope, q_rope,
        )
        # M4 scatter: use torch.index_copy_ on the REAL rows (bit-exact with golden).
        # cache_index is [b, s1] INT64 — flatten to [M] row indices.
        idx_flat = cache_index.reshape(-1)                               # [M] INT64
        kv_cache_2d = kv_cache.clone().reshape(-1, kv_cache.shape[-1])   # [bk*bs, d] BF16
        kr_cache_2d = kr_cache.clone().reshape(-1, kr_cache.shape[-1])   # [bk*bs, d] BF16
        kv_cache_out = kv_cache_2d.index_copy_(0, idx_flat, k_nope[:M].contiguous())
        kr_cache_out = kr_cache_2d.index_copy_(0, idx_flat, k_rope[:M].contiguous())
        # Reshape cache outputs back to 4D
        kv_cache_out = kv_cache_out.reshape(kv_cache.shape)
        kr_cache_out = kr_cache_out.reshape(kr_cache.shape)
        # Slice back to real M
        # Return order matches golden for M1234: q_nope, q_rope, k_nope, k_rope, q_a_norm, kv_cache_out, kr_cache_out
        return (q_nope[:M], q_rope[:M], k_nope[:M], k_rope[:M], q_a_norm[:M],
                kv_cache_out, kr_cache_out)
    else:
        # M is multiple of 8 — no padding needed
        # Allocate output buffers (OL58: use torch.*, NOT pypto.zeros/pypto.empty)
        # Fix 3: reorder k_nope/k_rope before q_rope, add guard padding between
        # k_rope and q_rope to physically isolate q_rope from M3 workspace aliasing.
        q_a_norm = torch.empty(M, 1536, dtype=torch.bfloat16,
                               device=hidden_states.device)
        q_nope = torch.empty(M, 128, 512, dtype=torch.bfloat16,
                             device=hidden_states.device)
        k_nope = torch.empty(M, 512, dtype=torch.bfloat16,
                             device=hidden_states.device)
        k_rope = torch.empty(M, 64, dtype=torch.bfloat16,
                             device=hidden_states.device)
        _guard = torch.empty(1024, dtype=torch.bfloat16,
                             device=hidden_states.device)
        q_rope = torch.empty(M, 128, 64, dtype=torch.bfloat16,
                             device=hidden_states.device)  # ← last = farthest from M3 scratch

        # <<< SNAPSHOT:HOST_WRAPPER_INSPECT_ALLOC
        # >>> SNAPSHOT:HOST_WRAPPER_INSPECT_ALLOC

        # Single JIT call (OL45: no Python for-loop around kernel) — 5 outputs
        mla_prolog_module1234_kernel_npu(
            hidden_states, cos, sin, w_dq, w_uqqr, w_uk_0, w_uk_1, w_uk_2, w_uk_3, w_dkvkr, gamma_cq, gamma_ckv,
            q_a_norm, q_nope, k_nope, k_rope, q_rope,
            # <<< SNAPSHOT:HOST_WRAPPER_INSPECT_PASS
            # >>> SNAPSHOT:HOST_WRAPPER_INSPECT_PASS
        )

        # M4 scatter: use torch.index_copy_ on the REAL rows (bit-exact with golden)
        idx_flat = cache_index.reshape(-1)                               # [M] INT64
        kv_cache_2d = kv_cache.clone().reshape(-1, kv_cache.shape[-1])   # [bk*bs, d] BF16
        kr_cache_2d = kr_cache.clone().reshape(-1, kr_cache.shape[-1])   # [bk*bs, d] BF16
        kv_cache_out = kv_cache_2d.index_copy_(0, idx_flat, k_nope[:M].contiguous())
        kr_cache_out = kr_cache_2d.index_copy_(0, idx_flat, k_rope[:M].contiguous())
        # Reshape cache outputs back to 4D
        kv_cache_out = kv_cache_out.reshape(kv_cache.shape)
        kr_cache_out = kr_cache_out.reshape(kr_cache.shape)

        # Return order matches golden output order for M1+M2+M3+M4:
        # q_nope, q_rope, k_nope, k_rope, q_a_norm, kv_cache_out, kr_cache_out
        return (q_nope, q_rope, k_nope, k_rope, q_a_norm,
                kv_cache_out, kr_cache_out)