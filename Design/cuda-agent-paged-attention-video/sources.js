// 视频中出现的源码与命令输出。
// agent_workdir 的文件结构、命令与日志格式取自 BytedTsinghua-SIA/CUDA-Agent 仓库（2026-11-14 读取）；
// PagedAttention 的 model.py / kernels/* 为本视频编写的示意实现，性能数字为示意值。
window.SRC = {};

SRC.HOST = 'gpu-h100-01';
SRC.WD = '/home/dev/CUDA-Agent/agent_workdir';

// 仓库自带的示例 model.py（axpby）
SRC.modelOld = `# This is an example model.py
# It implements a custom CUDA kernel for element-wise addition
# IMPORTANT: Do not change this file!
import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, alpha: float) -> None:
        super().__init__()
        self.alpha = alpha

    def forward(self, a, b):
        return self.alpha * a + b


def get_inputs():
    a = torch.randn(1, 128)
    b = torch.randn(1, 128)
    return [a, b]


def get_init_inputs():
    return [2.0]`;

// 仓库自带的示例 model_new.py
SRC.modelNewOld = `import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, alpha: float) -> None:
        super().__init__()
        self.alpha = alpha

    def forward(self, a, b):
        return cuda_extension.axpby_forward(a, b, self.alpha, 0)`;

// 开发者粘贴的 PagedAttention decode 参考实现（任务定义）
SRC.model = `# PagedAttention decode reference (vLLM-style paged KV cache, GQA)
# IMPORTANT: Do not change this file!
import math
import torch
import torch.nn as nn

NUM_SEQS = 32
NUM_HEADS = 32
NUM_KV_HEADS = 8
HEAD_DIM = 128
BLOCK_SIZE = 16
MAX_CONTEXT = 4096
NUM_BLOCKS = NUM_SEQS * MAX_CONTEXT // BLOCK_SIZE


class Model(nn.Module):
    def __init__(self, scale: float) -> None:
        super().__init__()
        self.scale = scale

    def forward(self, query, key_cache, value_cache, block_tables, context_lens):
        # query:        [num_seqs, num_heads, head_dim]
        # key_cache:    [num_blocks, num_kv_heads, block_size, head_dim]
        # value_cache:  [num_blocks, num_kv_heads, block_size, head_dim]
        # block_tables: [num_seqs, max_blocks_per_seq]   int32
        # context_lens: [num_seqs]                        int32
        num_seqs, num_heads, head_dim = query.shape
        num_kv_heads, block_size = key_cache.shape[1], key_cache.shape[2]
        group = num_heads // num_kv_heads
        max_len = block_tables.shape[1] * block_size

        k = key_cache[block_tables.long()]          # [S, B, Hkv, bs, D]
        v = value_cache[block_tables.long()]
        k = k.permute(0, 2, 1, 3, 4).reshape(num_seqs, num_kv_heads, max_len, head_dim)
        v = v.permute(0, 2, 1, 3, 4).reshape(num_seqs, num_kv_heads, max_len, head_dim)
        k = k.repeat_interleave(group, dim=1)        # GQA -> [S, H, L, D]
        v = v.repeat_interleave(group, dim=1)

        scores = torch.einsum("shd,shld->shl", query.float(), k.float()) * self.scale
        pos = torch.arange(max_len, device=query.device)
        mask = pos[None, :] >= context_lens[:, None]
        scores = scores.masked_fill(mask[:, None, :], float("-inf"))
        probs = torch.softmax(scores, dim=-1)
        out = torch.einsum("shl,shld->shd", probs, v.float())
        return out.to(query.dtype)


def get_inputs():
    query = torch.randn(NUM_SEQS, NUM_HEADS, HEAD_DIM, dtype=torch.float16)
    key_cache = torch.randn(NUM_BLOCKS, NUM_KV_HEADS, BLOCK_SIZE, HEAD_DIM, dtype=torch.float16)
    value_cache = torch.randn_like(key_cache)
    max_blocks = MAX_CONTEXT // BLOCK_SIZE
    block_tables = torch.randperm(NUM_BLOCKS)[: NUM_SEQS * max_blocks]
    block_tables = block_tables.view(NUM_SEQS, max_blocks).int()
    context_lens = torch.randint(1024, MAX_CONTEXT + 1, (NUM_SEQS,), dtype=torch.int32)
    return [query, key_cache, value_cache, block_tables, context_lens]


def get_init_inputs():
    return [1.0 / math.sqrt(HEAD_DIM)]`;

SRC.modelNew = `import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, scale: float) -> None:
        super().__init__()
        self.scale = scale

    def forward(self, query, key_cache, value_cache, block_tables, context_lens):
        return cuda_extension.paged_attention_forward(
            query, key_cache, value_cache, block_tables, context_lens, self.scale, 0)`;

// ---------- v1 kernel（含两处待修复的问题） ----------
SRC.kernelV1 = `// PagedAttention decode — v1
// one CTA per (head, seq): warp-per-token q.k, block softmax, thread-per-channel P.V
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cfloat>

#define WARP_SIZE 32

__device__ __forceinline__ float warp_reduce_sum(float v) {
#pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1)
        v += __shfl_xor(v, offset);
    return v;
}

__device__ __forceinline__ float warp_reduce_max(float v) {
#pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1)
        v = fmaxf(v, __shfl_xor(v, offset));
    return v;
}

template <int HEAD_DIM, int BLOCK_SIZE, int NUM_THREADS>
__global__ void paged_attention_v1_kernel(
    half* __restrict__ out,                 // [num_seqs, num_heads, HEAD_DIM]
    const half* __restrict__ q,             // [num_seqs, num_heads, HEAD_DIM]
    const half* __restrict__ k_cache,       // [num_blocks, num_kv_heads, BLOCK_SIZE, HEAD_DIM]
    const half* __restrict__ v_cache,       // [num_blocks, num_kv_heads, BLOCK_SIZE, HEAD_DIM]
    const int* __restrict__ block_tables,   // [num_seqs, max_blocks]
    const int* __restrict__ context_lens,   // [num_seqs]
    const float scale, const int num_kv_heads, const int max_blocks) {
    constexpr int NUM_WARPS = NUM_THREADS / WARP_SIZE;
    constexpr int ELEMS = HEAD_DIM / WARP_SIZE;
    const int head = blockIdx.x, seq = blockIdx.y, num_heads = gridDim.x;
    const int kv_head = head;
    const int ctx_len = context_lens[seq];
    const int warp = threadIdx.x / WARP_SIZE, lane = threadIdx.x % WARP_SIZE;
    const int* bt = block_tables + seq * max_blocks;

    extern __shared__ float logits[];       // [max_context]
    __shared__ float red[NUM_WARPS];

    // 1) q slice -> registers (ELEMS per lane)
    float qv[ELEMS];
    const half* q_ptr = q + (seq * num_heads + head) * HEAD_DIM;
#pragma unroll
    for (int i = 0; i < ELEMS; ++i) qv[i] = __half2float(q_ptr[lane + i * WARP_SIZE]);

    // 2) scores: one warp per token
    float local_max = -FLT_MAX;
    for (int t = warp; t < ctx_len; t += NUM_WARPS) {
        const int phys = bt[t / BLOCK_SIZE];
        const half* k_ptr = k_cache +
            ((size_t)(phys * num_kv_heads + kv_head) * BLOCK_SIZE + t % BLOCK_SIZE) * HEAD_DIM;
        float s = 0.f;
#pragma unroll
        for (int i = 0; i < ELEMS; ++i) s += qv[i] * __half2float(k_ptr[lane + i * WARP_SIZE]);
        s = warp_reduce_sum(s) * scale;
        if (lane == 0) logits[t] = s;
        local_max = fmaxf(local_max, s);
    }

    // 3) block-wide softmax
    local_max = warp_reduce_max(local_max);
    if (lane == 0) red[warp] = local_max;
    __syncthreads();
    float m = -FLT_MAX;
#pragma unroll
    for (int w = 0; w < NUM_WARPS; ++w) m = fmaxf(m, red[w]);
    __syncthreads();

    float local_sum = 0.f;
    for (int t = threadIdx.x; t < ctx_len; t += NUM_THREADS) {
        const float e = __expf(logits[t] - m);
        logits[t] = e;
        local_sum += e;
    }
    local_sum = warp_reduce_sum(local_sum);
    if (lane == 0) red[warp] = local_sum;
    __syncthreads();
    float sum = 0.f;
#pragma unroll
    for (int w = 0; w < NUM_WARPS; ++w) sum += red[w];
    const float inv_sum = 1.f / (sum + 1e-6f);

    // 4) out[d] = sum_t p_t * v_t[d]   (one thread per channel)
    for (int d = threadIdx.x; d < HEAD_DIM; d += NUM_THREADS) {
        float acc = 0.f;
        for (int t = 0; t < ctx_len; ++t) {
            const int phys = bt[t / BLOCK_SIZE];
            const half* v_ptr = v_cache +
                ((size_t)(phys * num_kv_heads + kv_head) * BLOCK_SIZE + t % BLOCK_SIZE) * HEAD_DIM;
            acc += logits[t] * __half2float(v_ptr[d]);
        }
        out[(seq * num_heads + head) * HEAD_DIM + d] = __float2half(acc * inv_sum);
    }
}

extern "C" void paged_attention_launcher(
    half* out, const half* q, const half* k_cache, const half* v_cache,
    const int* block_tables, const int* context_lens, float scale,
    int num_seqs, int num_heads, int num_kv_heads, int max_blocks,
    int config, cudaStream_t stream) {
    constexpr int THREADS = 128;
    const dim3 grid(num_heads, num_seqs);
    const size_t smem = (size_t)max_blocks * 16 * sizeof(float);
    paged_attention_v1_kernel<128, 16, THREADS><<<grid, THREADS, smem, stream>>>(
        out, q, k_cache, v_cache, block_tables, context_lens,
        scale, num_kv_heads, max_blocks);
}`;

SRC.bindingV1 = `#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

extern "C" void paged_attention_launcher(
    half* out, const half* q, const half* k_cache, const half* v_cache,
    const int* block_tables, const int* context_lens, float scale,
    int num_seqs, int num_heads, int num_kv_heads, int max_blocks,
    int config, cudaStream_t stream);

static torch::Tensor paged_attention_forward(
    torch::Tensor query, torch::Tensor key_cache, torch::Tensor value_cache,
    torch::Tensor block_tables, torch::Tensor context_lens,
    double scale, int config = 0) {
    TORCH_CHECK(query.is_cuda() && key_cache.is_cuda() && value_cache.is_cuda(),
                "query / key_cache / value_cache must be CUDA tensors");
    TORCH_CHECK(query.is_contiguous() && key_cache.is_contiguous() &&
                value_cache.is_contiguous(), "inputs must be contiguous");
    TORCH_CHECK(query.scalar_type() == torch::kHalf, "query must be float16");
    TORCH_CHECK(block_tables.scalar_type() == torch::kInt &&
                context_lens.scalar_type() == torch::kInt, "indices must be int32");
    TORCH_CHECK(query.size(2) == 128 && key_cache.size(2) == 16,
                "kernel is specialized for head_dim=128, block_size=16");

    auto out = torch::empty_like(query);
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    paged_attention_launcher(
        reinterpret_cast<half*>(out.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(query.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(key_cache.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(value_cache.data_ptr<at::Half>()),
        block_tables.data_ptr<int>(), context_lens.data_ptr<int>(),
        static_cast<float>(scale), query.size(0), query.size(1),
        key_cache.size(1), block_tables.size(1), config, stream);
    return out;
}

static void register_paged_attention(pybind11::module& m) {
    m.def("paged_attention_forward", &paged_attention_forward,
          py::arg("query"), py::arg("key_cache"), py::arg("value_cache"),
          py::arg("block_tables"), py::arg("context_lens"),
          py::arg("scale"), py::arg("config") = 0);
}

REGISTER_BINDING(paged_attention, register_paged_attention);`;

// ---------- v2 kernel：split-K + GQA 共享 K/V + 16B 向量化 ----------
SRC.kernelV2 = `// PagedAttention decode — v2
// split-K over context + GQA head grouping + 128-bit vectorized K/V loads
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cfloat>

#define WARP_SIZE 32

__device__ __forceinline__ float warp_reduce_sum(float v) {
#pragma unroll
    for (int o = WARP_SIZE / 2; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffff, v, o);
    return v;
}

__device__ __forceinline__ float warp_reduce_max(float v) {
#pragma unroll
    for (int o = WARP_SIZE / 2; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffff, v, o));
    return v;
}

// dot product of 8 halfs (one 16-byte vector) with 8 floats
__device__ __forceinline__ float dot8(const uint4& a, const float* q) {
    const half2* h = reinterpret_cast<const half2*>(&a);
    float s = 0.f;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const float2 f = __half22float2(h[i]);
        s += f.x * q[2 * i] + f.y * q[2 * i + 1];
    }
    return s;
}

template <int HEAD_DIM, int BLOCK_SIZE, int GROUP, int PARTITION, int NUM_THREADS>
__global__ void __launch_bounds__(NUM_THREADS) paged_attention_v2_kernel(
    float* __restrict__ part_out,           // [S, H, P, HEAD_DIM]
    float* __restrict__ part_max,           // [S, H, P]
    float* __restrict__ part_sum,           // [S, H, P]
    const half* __restrict__ q,
    const half* __restrict__ k_cache,
    const half* __restrict__ v_cache,
    const int* __restrict__ block_tables,
    const int* __restrict__ context_lens,
    const float scale, const int num_kv_heads, const int max_blocks) {
    constexpr int VEC = 8;                                // 16 B = 8 x half
    constexpr int TPT = HEAD_DIM / VEC;                   // threads per token
    constexpr int TOK_PER_ITER = NUM_THREADS / TPT;
    constexpr int NUM_WARPS = NUM_THREADS / WARP_SIZE;

    const int kv_head = blockIdx.x, seq = blockIdx.y, part = blockIdx.z;
    const int num_heads = num_kv_heads * GROUP, num_parts = gridDim.z;
    const int ctx_len = context_lens[seq];
    const int start = part * PARTITION;
    if (start >= ctx_len) return;                         // empty partition
    const int end = min(start + PARTITION, ctx_len);
    const int* bt = block_tables + seq * max_blocks;

    __shared__ float q_s[GROUP][HEAD_DIM];
    __shared__ float o_s[GROUP][HEAD_DIM];
    __shared__ float p_s[GROUP][PARTITION];

    // 1) stage the GROUP query heads that share this kv head (pre-scaled)
    for (int i = threadIdx.x; i < GROUP * HEAD_DIM; i += NUM_THREADS) {
        const int g = i / HEAD_DIM, d = i % HEAD_DIM;
        q_s[g][d] = __half2float(q[(seq * num_heads + kv_head * GROUP + g) * HEAD_DIM + d]) * scale;
        o_s[g][d] = 0.f;
    }
    __syncthreads();

    // 2) scores: every K row is loaded ONCE (16 B / thread) and reused by GROUP heads
    const int sub = threadIdx.x % TPT, row = threadIdx.x / TPT;
    for (int base = start; base < end; base += TOK_PER_ITER) {
        const int t = base + row;
        const bool valid = t < end;
        uint4 kv = make_uint4(0, 0, 0, 0);
        if (valid) {
            const int phys = __ldg(bt + t / BLOCK_SIZE);
            kv = *reinterpret_cast<const uint4*>(k_cache +
                ((size_t)(phys * num_kv_heads + kv_head) * BLOCK_SIZE + t % BLOCK_SIZE) * HEAD_DIM + sub * VEC);
        }
#pragma unroll
        for (int g = 0; g < GROUP; ++g) {
            float s = dot8(kv, &q_s[g][sub * VEC]);
#pragma unroll
            for (int o = TPT / 2; o > 0; o >>= 1) s += __shfl_xor_sync(0xffffffff, s, o);
            if (valid && sub == 0) p_s[g][t - start] = s;
        }
    }
    __syncthreads();

    // 3) partition-local softmax statistics (one warp per head)
    const int warp = threadIdx.x / WARP_SIZE, lane = threadIdx.x % WARP_SIZE;
    for (int g = warp; g < GROUP; g += NUM_WARPS) {
        float m = -FLT_MAX;
        for (int i = lane; i < end - start; i += WARP_SIZE) m = fmaxf(m, p_s[g][i]);
        m = warp_reduce_max(m);
        float l = 0.f;
        for (int i = lane; i < end - start; i += WARP_SIZE) {
            const float e = __expf(p_s[g][i] - m);
            p_s[g][i] = e;
            l += e;
        }
        l = warp_reduce_sum(l);
        if (lane == 0) {
            const int idx = (seq * num_heads + kv_head * GROUP + g) * num_parts + part;
            part_max[idx] = m;
            part_sum[idx] = l;
        }
    }
    __syncthreads();

    // 4) P.V: every V row loaded once, accumulated for all GROUP heads
    float acc[GROUP][VEC] = {};
    for (int t = start + row; t < end; t += TOK_PER_ITER) {
        const int phys = __ldg(bt + t / BLOCK_SIZE);
        const uint4 vv = *reinterpret_cast<const uint4*>(v_cache +
            ((size_t)(phys * num_kv_heads + kv_head) * BLOCK_SIZE + t % BLOCK_SIZE) * HEAD_DIM + sub * VEC);
        const half2* h = reinterpret_cast<const half2*>(&vv);
#pragma unroll
        for (int g = 0; g < GROUP; ++g) {
            const float p = p_s[g][t - start];
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                const float2 f = __half22float2(h[i]);
                acc[g][2 * i] += p * f.x;
                acc[g][2 * i + 1] += p * f.y;
            }
        }
    }
#pragma unroll
    for (int g = 0; g < GROUP; ++g)
#pragma unroll
        for (int i = 0; i < VEC; ++i) atomicAdd(&o_s[g][sub * VEC + i], acc[g][i]);
    __syncthreads();

    for (int i = threadIdx.x; i < GROUP * HEAD_DIM; i += NUM_THREADS) {
        const int g = i / HEAD_DIM, d = i % HEAD_DIM;
        const size_t idx = (size_t)(seq * num_heads + kv_head * GROUP + g) * num_parts + part;
        part_out[idx * HEAD_DIM + d] = o_s[g][d];
    }
}

// merge partitions with the log-sum-exp trick
template <int HEAD_DIM>
__global__ void paged_attention_reduce_kernel(
    half* __restrict__ out, const float* __restrict__ part_out,
    const float* __restrict__ part_max, const float* __restrict__ part_sum,
    const int* __restrict__ context_lens, const int partition, const int num_parts) {
    const int head = blockIdx.x, seq = blockIdx.y, num_heads = gridDim.x;
    const int n = (context_lens[seq] + partition - 1) / partition;
    const int base = (seq * num_heads + head) * num_parts;
    float m = -FLT_MAX;
    for (int p = 0; p < n; ++p) m = fmaxf(m, part_max[base + p]);
    float l = 0.f;
    for (int p = 0; p < n; ++p) l += part_sum[base + p] * __expf(part_max[base + p] - m);
    const float inv = 1.f / (l + 1e-6f);
    for (int d = threadIdx.x; d < HEAD_DIM; d += blockDim.x) {
        float o = 0.f;
        for (int p = 0; p < n; ++p)
            o += part_out[(size_t)(base + p) * HEAD_DIM + d] * __expf(part_max[base + p] - m);
        out[(seq * num_heads + head) * HEAD_DIM + d] = __float2half(o * inv);
    }
}

extern "C" void paged_attention_launcher(
    half* out, float* part_out, float* part_max, float* part_sum,
    const half* q, const half* k_cache, const half* v_cache,
    const int* block_tables, const int* context_lens, float scale,
    int num_seqs, int num_heads, int num_kv_heads, int max_blocks,
    int num_parts, int config, cudaStream_t stream) {
    constexpr int PARTITION = 512, THREADS = 256;
    const dim3 grid(num_kv_heads, num_seqs, num_parts);
    paged_attention_v2_kernel<128, 16, 4, PARTITION, THREADS><<<grid, THREADS, 0, stream>>>(
        part_out, part_max, part_sum, q, k_cache, v_cache, block_tables, context_lens,
        scale, num_kv_heads, max_blocks);
    paged_attention_reduce_kernel<128><<<dim3(num_heads, num_seqs), 128, 0, stream>>>(
        out, part_out, part_max, part_sum, context_lens, PARTITION, num_parts);
}`;

SRC.bindingV2 = `#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

extern "C" void paged_attention_launcher(
    half* out, float* part_out, float* part_max, float* part_sum,
    const half* q, const half* k_cache, const half* v_cache,
    const int* block_tables, const int* context_lens, float scale,
    int num_seqs, int num_heads, int num_kv_heads, int max_blocks,
    int num_parts, int config, cudaStream_t stream);

static torch::Tensor paged_attention_forward(
    torch::Tensor query, torch::Tensor key_cache, torch::Tensor value_cache,
    torch::Tensor block_tables, torch::Tensor context_lens,
    double scale, int config = 0) {
    TORCH_CHECK(query.is_cuda() && key_cache.is_cuda() && value_cache.is_cuda(),
                "query / key_cache / value_cache must be CUDA tensors");
    TORCH_CHECK(query.is_contiguous() && key_cache.is_contiguous() &&
                value_cache.is_contiguous(), "inputs must be contiguous");
    TORCH_CHECK(query.scalar_type() == torch::kHalf, "query must be float16");
    TORCH_CHECK(block_tables.scalar_type() == torch::kInt &&
                context_lens.scalar_type() == torch::kInt, "indices must be int32");
    TORCH_CHECK(query.size(2) == 128 && key_cache.size(2) == 16,
                "kernel is specialized for head_dim=128, block_size=16");
    TORCH_CHECK(query.size(1) == key_cache.size(1) * 4,
                "kernel is specialized for GQA group = 4");

    constexpr int PARTITION = 512;
    const int num_seqs = query.size(0), num_heads = query.size(1);
    const int max_blocks = block_tables.size(1);
    const int num_parts = (max_blocks * 16 + PARTITION - 1) / PARTITION;

    auto out = torch::empty_like(query);
    auto f32 = query.options().dtype(torch::kFloat);
    auto part_out = torch::empty({num_seqs, num_heads, num_parts, 128}, f32);
    auto part_max = torch::empty({num_seqs, num_heads, num_parts}, f32);
    auto part_sum = torch::empty({num_seqs, num_heads, num_parts}, f32);

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    paged_attention_launcher(
        reinterpret_cast<half*>(out.data_ptr<at::Half>()),
        part_out.data_ptr<float>(), part_max.data_ptr<float>(), part_sum.data_ptr<float>(),
        reinterpret_cast<const half*>(query.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(key_cache.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(value_cache.data_ptr<at::Half>()),
        block_tables.data_ptr<int>(), context_lens.data_ptr<int>(),
        static_cast<float>(scale), num_seqs, num_heads, key_cache.size(1),
        max_blocks, num_parts, config, stream);
    return out;
}

static void register_paged_attention(pybind11::module& m) {
    m.def("paged_attention_forward", &paged_attention_forward,
          py::arg("query"), py::arg("key_cache"), py::arg("value_cache"),
          py::arg("block_tables"), py::arg("context_lens"),
          py::arg("scale"), py::arg("config") = 0);
}

REGISTER_BINDING(paged_attention, register_paged_attention);`;

// 仓库 utils/compile.sh 原文
SRC.compileSh = `#!/bin/bash
START_TIME=$(date +%s.%N)
python3 -m utils.compile
EXIT_CODE=$?
END_TIME=$(date +%s.%N)
ELAPSED=$(awk "BEGIN {printf \\"%.2f\\", $END_TIME - $START_TIME}")
echo "[TIME] Compilation took \${ELAPSED}s"
exit $EXIT_CODE`;

SRC.lineOf = (text, needle, from = 0) => {
  const lines = text.split('\n');
  for (let i = from; i < lines.length; i++) if (lines[i].includes(needle)) return i + 1;
  return -1;
};
