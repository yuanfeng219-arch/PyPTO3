#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""
"""
from dataclasses import dataclass
import pypto
import math
import torch
import torch_npu
from typing import List

SHAPE_DIM_2 = 2
SHAPE_DIM_3 = 3

NUM_0 = 0
NUM_1 = 1
NUM_2 = 2
NUM_3 = 3
NUM_7168 = 7168

TILE_CUBE_DIM = 6
CHUNK_SIZE = 2
Q_PARAM_DIM = 2
NZ_DIM = 4
COS_SIN_DIM = 2
L0M_INDEX = 0
L1M_INDEX = 1
L0K_INDEX = 2
L1K_INDEX = 3
L0N_INDEX = 4
L1N_INDEX = 5
SCATTER_DIM = -2
NZ_FIRST_DIM = 16
NZ_B8_C0 = 32
NZ_B16_C0 = 16

VEC_TILE_256 = 256
VEC_TILE_128 = 128
VEC_TILE_64 = 64
VEC_TILE_8 = 8
VEC_TILE_4 = 4


@dataclass
class IndexerPrologQuantInput:
    x: torch.tensor  # BF16, (t, h)
    q_norm: torch.tensor  # INT8, (t, qLoraRank)
    q_norm_scale: torch.tensor  # FP32, (t, 1)
    w_qb: torch.tensor  # INT8, (headNum * headDim // NZ_B8_C0, qLoraRank // NZ_FIRST_DIM, NZ_FIRST_DIM, NZ_B8_C0), NZ
    w_qb_scale: torch.tensor  # FP32, (headNum * headDim, 1)
    wk: torch.tensor  # BF16, (headDim // NZ_B16_C0, h // NZ_FIRST_DIM, NZ_FIRST_DIM, NZ_B16_C0), NZ
    w_proj: torch.tensor  # BF16, (headNum // NZ_B16_C0, h // NZ_FIRST_DIM, NZ_FIRST_DIM, NZ_B16_C0), NZ
    ln_gamma_k: torch.tensor  # BF16, (headDim,)
    ln_beta_k: torch.tensor  # BF16, (headDim,)
    cos_idx_rope: torch.tensor  # BF16, (t, ropeHeadDim)
    sin_idx_rope: torch.tensor  # BF16, (t, ropeHeadDim)
    hadamard_q: torch.tensor  # BF16, (headDim, headDim)
    hadamard_k: torch.tensor  # BF16, (headDim, headDim)
    k_cache: torch.tensor  # INT8, (blockNum, blockSize, nKv, headDim)
    k_cache_scale: torch.tensor  # FP16, (blockNum, blockSize, nKv, 1)
    k_cache_index: torch.tensor  # INT64, (t,)


@dataclass
class IndexerPrologQuantOutput:
    q_int8: torch.tensor
    q_scale: torch.tensor
    k_int8: torch.tensor
    k_scale: torch.tensor
    weights: torch.tensor


@dataclass
class IndexerPrologQuantAttr:
    eps: float
    layerout_query: str
    layerout_key: str


@dataclass
class IndexerPrologQuantConfigs:
    q_linear: List[int]
    q_hd: List[int]
    k_linear: List[int]
    w_linear: List[int]
    unroll_list: List[int]

    l1_reuse_param: dict[int, int]
    copy_in_threshold: int
    cycle_upper_bound: int
    block_size: int


def quant_layer_norm(x: pypto.tensor, gamma: pypto.tensor, beta: pypto.tensor, dim: int, epsilon: float):
    pypto.set_semantic_label("Key-LayerNorm")
    assert ((dim == len(x.shape) - 1) or (dim == -1))
    actual_dim = dim < 0 if dim + len(x.shape) else dim
    x_dtype = x.dtype

    x_fp32 = pypto.cast(x, pypto.DT_FP32)
    # do division first to avoid overflow
    x_scaled = x_fp32 * (1.0 / x.shape[actual_dim])
    mean = pypto.sum(x_scaled, actual_dim, keepdim=True)

    diff = x_fp32 - mean
    squared_diff = diff * diff
    squared_diff_scaled = squared_diff * (1.0 / x.shape[actual_dim])
    var = pypto.sum(squared_diff_scaled, actual_dim, keepdim=True)
    # add epsilon to avoid division by zero
    var_eps = var + epsilon
    std_var = pypto.sqrt(var_eps)
    res32 = diff / std_var

    gamma32 = pypto.cast(gamma, pypto.DT_FP32)
    beta32 = pypto.cast(beta, pypto.DT_FP32)
    return pypto.cast((res32 * gamma32) + beta32, x_dtype)


def quant_rope_2d(x: pypto.tensor, cos: pypto.tensor, sin: pypto.tensor):
    pypto.set_semantic_label("Key-Rope2D")
    key_rope_dim = 2
    x_dtype = x.dtype
    t_tile = x.shape[0]
    rope_dim = x.shape[1]
    assert (len(x.shape) == key_rope_dim and len(cos.shape) == COS_SIN_DIM and len(sin.shape) == COS_SIN_DIM)

    pypto.set_vec_tile_shapes(t_tile, rope_dim)
    cast_cos = pypto.cast(cos, pypto.DT_FP32)
    cast_sin = pypto.cast(sin, pypto.DT_FP32)
    x_view = pypto.cast(x, pypto.DT_FP32)

    pypto.set_vec_tile_shapes(t_tile, rope_dim)
    x_embed = (x_view * cast_cos) + ((rotate_half(x_view)) * cast_sin)
    res = pypto.cast(x_embed, x_dtype)
    return res


def prolog_quant(input: pypto.tensor):
    pypto.set_semantic_label("Prolog-Quant")
    s8_max_value = 127.0
    s8_one_value = 1.0
    input_fp32 = pypto.cast(input, pypto.DT_FP32)

    abs_res = pypto.abs(input_fp32)
    max_value = pypto.amax(abs_res, dim=-1, keepdim=True)
    temp127 = pypto.full(max_value.shape, s8_max_value, pypto.DT_FP32)

    scale_quant = temp127 / max_value
    out_fp32 = input_fp32 * scale_quant
    out_int32 = pypto.cast(out_fp32, pypto.DT_INT32)
    out_half = pypto.cast(out_int32, pypto.DT_FP16)
    out_int8 = pypto.cast(out_half, pypto.DT_INT8)
    temp1 = pypto.full(scale_quant.shape, s8_one_value, pypto.DT_FP32)
    scale_dequant = temp1 / scale_quant
    return (out_int8, scale_dequant)


def rotate_half(input_tensor: pypto.tensor) -> pypto.tensor:
    chunk_size = 2
    shape = input_tensor.shape
    shape_size = len(shape)
    assert shape_size >= 1
    assert shape[shape_size - 1] % chunk_size == 0
    shape[shape_size - 1] //= chunk_size
    offset1 = [0] * shape_size
    offset2 = [0] * shape_size
    offset2[shape_size - 1] = shape[shape_size - 1]
    x1 = pypto.view(input_tensor, shape, offset1)
    x2 = pypto.view(input_tensor, shape, offset2)
    return pypto.concat([x2 * (-1.0), x1 + 0.0], -1)


def rope_3d(x: pypto.tensor, cos: pypto.tensor, sin: pypto.tensor) -> pypto.tensor:
    head_num_axis = 1
    head_dim_axis = 2
    assert (len(x.shape) == SHAPE_DIM_3 and len(cos.shape) == SHAPE_DIM_2 and len(sin.shape) == SHAPE_DIM_2)

    x_dtype = x.dtype
    t_tile = x.shape[0]
    head_num = x.shape[head_num_axis]
    rope_dim = x.shape[head_dim_axis]

    pypto.set_vec_tile_shapes(1, rope_dim)
    cast_cos = pypto.cast(cos, pypto.DT_FP32)
    cast_sin = pypto.cast(sin, pypto.DT_FP32)

    pypto.set_vec_tile_shapes(1, head_num // CHUNK_SIZE, rope_dim)
    x_view = pypto.cast(x, pypto.DT_FP32)
    cast_cos = pypto.reshape(cast_cos, [t_tile, 1, rope_dim])
    cast_sin = pypto.reshape(cast_sin, [t_tile, 1, rope_dim])

    x_embed = (x_view * cast_cos) + ((rotate_half(x_view)) * cast_sin)
    res = pypto.cast(x_embed, x_dtype)
    return res


def lightning_indexer_prolog_quant_compute(inputs, outputs, attrs, configs):
    (x_in, q_norm_in, q_norm_scale_in, w_qb_in, w_qb_scale_in, wk_in, w_proj_in, ln_gamma_k_in, ln_beta_k_in,
     cos_idx_rope_in, sin_idx_rope_in, hadamard_q_in, hadamard_k_in, k_cache, k_cache_scale, k_cache_index_in) = inputs
    q_int8_out, q_scale_out, k_int8_out, k_scale_out, weights_out = outputs

    pypto.mark_dynamic(x_in, 0)
    pypto.mark_dynamic(q_norm_in, 0)
    pypto.mark_dynamic(q_norm_scale_in, 0)
    pypto.mark_dynamic(cos_idx_rope_in, 0)
    pypto.mark_dynamic(sin_idx_rope_in, 0)
    pypto.mark_dynamic(k_cache, 0)
    pypto.mark_dynamic(k_cache_scale, 0)
    pypto.mark_dynamic(k_cache_index_in, 0)

    pypto.set_host_options(only_codegen=True)

    pypto.set_pass_options(nbuffer_merge_mode=0)
    pypto.set_pass_options(l1_reuse_map=configs.l1_reuse_param)
    pypto.set_pass_options(copyin_threshold=configs.copy_in_threshold)
    pypto.set_pass_options(cycle_upper_bound=configs.cycle_upper_bound)

    x_dtype = x_in.dtype

    # 动态轴
    t = x_in.shape[0]

    h = x_in.shape[1]
    q_lora_rank = q_norm_in.shape[1]
    head_num = w_proj_in.shape[0] * NZ_B16_C0
    head_dim = hadamard_q_in.shape[0]
    rope_head_dim = cos_idx_rope_in.shape[1]

    for _ in pypto.loop(0, 1, 1, name="LOOP_RESHAPE", idx_name="dummy"): 
        #从第0个数据开始循环，步长1，循环一次，把每个数据都执行pypto.reshape算子操作，
        #这段代码的计算意图是对多个输入张量进行形状重塑，以适配后续的计算
        k_cache_index = pypto.reshape(k_cache_index_in, [t, 1], inplace=True)
        #将k_cache_index_in重塑为形状[t, 1]。这样做的目的可能是为了后续的广播操作或者满足某个算子的输入要求。
        w_qb_scale = pypto.reshape(w_qb_scale_in, [1, head_num * head_dim], inplace=True)
        gamma_2d = pypto.reshape(ln_gamma_k_in, [1, ln_gamma_k_in.shape[0]], inplace=True)
        #这是LayerNorm的缩放参数，重塑为2D可能是为了后续的广播操作。
        beta_2d = pypto.reshape(ln_beta_k_in, [1, ln_beta_k_in.shape[0]], inplace=True)
        # NZ pypto.reshape(
        w_qb = pypto.reshape(w_qb_in, [q_lora_rank, head_num * head_dim], inplace=True)
        wk = pypto.reshape(wk_in, [h, head_dim], inplace=True)
        w_proj = pypto.reshape(w_proj_in, [h, head_num], inplace=True)

    unroll_list = configs.unroll_list
    for tIdx, unrollLength in pypto.loop_unroll(0, t, 1, name="IndexerPrologQuantQuantLoop", idx_name="tIdx",
                                                unroll_list=unroll_list, ):
        def IndexerPrologQuantQuantLoopInner(params):
            (tIdx, unrollLength, x_in, x_dtype, h, head_num, head_dim, q_lora_rank,
             beta_2d, gamma_2d, hadamard_k_in, hadamard_q_in, k_cache, k_cache_index,
             k_cache_scale, k_int8_out, q_norm_in, q_norm_scale_in, k_scale_out,
             q_int8_out, q_scale_out, rope_head_dim, cos_idx_rope_in, sin_idx_rope_in,
             w_proj, w_qb, w_qb_scale, weights_out, wk, configs, attrs) = params

            t_tile = unrollLength
            # 获取query计算的各阶段Tile参数
            q_linear = configs.q_linear
            q_hd = configs.q_hd
            # 多分档内会将t_tile作为档位，offset无需乘t_tile
            q_norm = pypto.view(q_norm_in, [t_tile, q_lora_rank], [tIdx, 0], valid_shape=[t_tile, q_lora_rank])
            q_norm_scale = pypto.view(q_norm_scale_in, [t_tile, 1], [tIdx, 0], valid_shape=[t_tile, 1])
            pypto.set_semantic_label("Query-Linear")
            pypto.set_cube_tile_shapes([q_linear[L0M_INDEX], q_linear[L1M_INDEX]],
                                       [q_linear[L0K_INDEX], q_linear[L1K_INDEX]],
                                       [q_linear[L0N_INDEX], q_linear[L1N_INDEX]], True)
            q_s32 = pypto.matmul(q_norm, w_qb, pypto.DT_INT32)  # (t_tile, head_num * head_dim)

            pypto.set_semantic_label("Query-Dequant")
            pypto.set_vec_tile_shapes(1, head_num * head_dim // CHUNK_SIZE)  # (t_tile, head_num * head_dim), fp32
            q_f32 = pypto.cast(q_s32, pypto.DT_FP32)
            q_f32 = q_f32 * q_norm_scale  # (t_tile, head_num * head_dim), fp32
            q_f32 = q_f32 * w_qb_scale  # (t_tile, head_num * head_dim), fp32
            q_cast = pypto.cast(q_f32, x_dtype)

            q_bf16 = pypto.reshape(q_cast, [t_tile, head_num, head_dim], valid_shape=[t_tile, head_num, head_dim])
            # UB view
            q_rope = pypto.view(q_bf16, [t_tile, head_num, rope_head_dim], [0, 0, 0],
                                valid_shape=[t_tile, head_num, rope_head_dim])
            q_nope = pypto.view(q_bf16, [t_tile, head_num, head_dim - rope_head_dim], [0, 0, rope_head_dim],
                                valid_shape=[t_tile, head_num, head_dim - rope_head_dim])
            rope_cos = pypto.view(cos_idx_rope_in, [t_tile, rope_head_dim], [tIdx, 0],
                                  valid_shape=[t_tile, rope_head_dim])
            rope_sin = pypto.view(sin_idx_rope_in, [t_tile, rope_head_dim], [tIdx, 0],
                                  valid_shape=[t_tile, rope_head_dim])

            q_roped = rope_3d(q_rope, rope_cos, rope_sin)  # [t_tile, head_num, rope_head_dim]
            pypto.set_vec_tile_shapes(1, head_num // CHUNK_SIZE, head_dim)
            q_nope = pypto.cast(pypto.cast(q_nope, pypto.DT_FP32), q_bf16.dtype)
            q_concat = pypto.concat([q_roped, q_nope], -1)  # [t_tile, head_num, head_dim]
            hadamard_q = pypto.reshape(hadamard_q_in, [1, head_dim, head_dim], valid_shape=[1, head_dim, head_dim])

            pypto.set_semantic_label("Query-Hadamard")
            pypto.set_cube_tile_shapes([q_hd[L0M_INDEX], q_hd[L1M_INDEX]], [q_hd[L0K_INDEX], q_hd[L1K_INDEX]],
                                       [q_hd[L0N_INDEX], q_hd[L1N_INDEX]])
            q_hadamard = pypto.matmul(q_concat, hadamard_q, x_dtype)  # (t_tile, head_num, head_dim)

            pypto.set_semantic_label("Query-Quant")
            pypto.set_vec_tile_shapes(1, head_num // CHUNK_SIZE, head_dim)
            q_res = prolog_quant(q_hadamard)
            q_scale = pypto.cast(q_res[1], pypto.DT_FP16)

            pypto.assemble(q_res[0], [tIdx, 0, 0], q_int8_out)
            pypto.assemble(q_scale, [tIdx, 0, 0], q_scale_out)

            # 获取key计算的各阶段Tile参数
            k_linear = configs.k_linear
            pypto.set_semantic_label("Key-Linear")
            pypto.set_cube_tile_shapes([k_linear[L0M_INDEX], k_linear[L1M_INDEX]],
                                       [k_linear[L0K_INDEX], k_linear[L1K_INDEX]],
                                       [k_linear[L0N_INDEX], k_linear[L1N_INDEX]], True)
            x = pypto.view(x_in, [t_tile, h], [tIdx, 0], valid_shape=[t_tile, h])  # 这里将t_tile分档，offset不需要乘t_tile
            k = pypto.matmul(x, wk, pypto.DT_FP32)  # (t_tile, head_dim)

            pypto.set_vec_tile_shapes(min(t_tile, VEC_TILE_4), head_dim)
            k_bf16 = pypto.cast(quant_layer_norm(k, gamma_2d, beta_2d, -1, attrs.eps), x_dtype)

            k_rope = pypto.view(k_bf16, [t_tile, rope_head_dim], [0, 0], valid_shape=[t_tile, rope_head_dim])
            k_nope = pypto.view(k_bf16, [t_tile, head_dim - rope_head_dim], [0, rope_head_dim],
                                valid_shape=[t_tile, head_dim - rope_head_dim])
            k_roped = quant_rope_2d(k_rope, rope_cos, rope_sin)  # (t_tile, rope_head_dim)
            pypto.set_vec_tile_shapes(t_tile, head_dim)
            k_nope = pypto.cast(pypto.cast(k_nope, pypto.DT_FP32), k_bf16.dtype)
            k_concat = pypto.concat([k_roped, k_nope], -1)
            pypto.set_semantic_label("Key-Hadamard")
            hadamard_k = pypto.matmul(k_concat, hadamard_k_in, x_dtype)  # (t_tile, head_dim), bf16
            pypto.set_semantic_label("Key-Quant")
            k_res = prolog_quant(hadamard_k)
            k_cache_4D = pypto.reshape(k_res[0], [t_tile, 1, 1, head_dim], valid_shape=[t_tile, 1, 1, head_dim])
            k_scale_4D = pypto.reshape(pypto.cast(k_res[1], pypto.DT_FP16), [t_tile, 1, 1, 1],
                                       valid_shape=[t_tile, 1, 1, 1])

            index = pypto.view(k_cache_index, [t_tile, 1], [tIdx, 0], valid_shape=[t_tile, 1])
            pypto.set_vec_tile_shapes(t_tile, 1, 1, head_dim)
            k_int8_out.move(pypto.scatter_update(k_cache, SCATTER_DIM, index, k_cache_4D))
            k_scale_out.move(pypto.scatter_update(k_cache_scale, SCATTER_DIM, index, k_scale_4D))

            pypto.set_semantic_label("Weight-Linear")
            w_linear = configs.w_linear
            pypto.set_cube_tile_shapes([w_linear[L0M_INDEX], w_linear[L1M_INDEX]],
                                       [w_linear[L0K_INDEX], w_linear[L1K_INDEX]],
                                       [w_linear[L0N_INDEX], w_linear[L1N_INDEX]])
            pypto.set_vec_tile_shapes(t_tile, head_num)
            weights = pypto.cast(pypto.matmul(x, w_proj, x_dtype), pypto.DT_FP32)
            weights = pypto.mul(weights, 1.0 / (math.sqrt(head_num) * math.sqrt(head_dim)))
            weights_f16 = pypto.cast(weights, pypto.DT_FP16)
            pypto.assemble(weights_f16, [tIdx, 0], weights_out)

        params = (tIdx, unrollLength, x_in, x_dtype, h, head_num, head_dim, q_lora_rank,
                  beta_2d, gamma_2d, hadamard_k_in, hadamard_q_in, k_cache, k_cache_index,
                  k_cache_scale, k_int8_out, q_norm_in, q_norm_scale_in, k_scale_out,
                  q_int8_out, q_scale_out, rope_head_dim, cos_idx_rope_in, sin_idx_rope_in,
                  w_proj, w_qb, w_qb_scale, weights_out, wk, configs, attrs)
        IndexerPrologQuantQuantLoopInner(params)


@pypto.jit
def lightning_indexer_prolog_quant(input_tensors, output_tensors, attrs, configs):
    lightning_indexer_prolog_quant_compute(input_tensors, output_tensors, attrs, configs)

