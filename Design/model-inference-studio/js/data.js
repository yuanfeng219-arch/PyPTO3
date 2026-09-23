/* PyPTO Model Inference Studio — 模型接入与兼容性分析 数据模型
 * 覆盖状态: native | fused | library | fallback | unsupported
 * 语义色沿用 model-graphviz pattern 的 sem:* 契约
 */
window.MIS_DATA = (function () {
  'use strict';

  const SEM = {
    embedding: '#14B8A6',
    norm: '#06B6D4',
    attention: '#EC4899',
    rope: '#A855F7',
    linear: '#3B82F6',
    mlp: '#8B5CF6',
    gate: '#F59E0B',
    moe: '#F97316',
    comm: '#22D3EE'
  };

  const COVERAGE_META = {
    native: { label: 'PyPTO Native', short: 'Native', color: '#3B82F6', desc: 'PyPTO DSL 原生实现，参与全局调度与融合决策' },
    fused: { label: 'Fused Kernel', short: 'Fused', color: '#8B5CF6', desc: '已被融合进上游/下游 kernel，无独立 launch' },
    library: { label: 'Library Call', short: 'Library', color: '#14B8A6', desc: '转发到 CANN/ATB 库算子，接口稳定但不可调优' },
    fallback: { label: 'Host Fallback', short: 'Fallback', color: '#FF9D00', desc: '回落到 Host 或通用实现，存在同步与性能代价' },
    unsupported: { label: 'Unsupported', short: 'Unsupported', color: '#FF2D7A', desc: '当前平台无可用实现，阻塞整网推理' }
  };

  const PRECISION_ORDER = ['FP16', 'BF16', 'FP8', 'INT8'];

  /* ---------------- Qwen3-30B-A3B (MoE) ---------------- */
  const qwen3moe = {
    id: 'qwen3-moe',
    name: 'Qwen3-30B-A3B',
    family: 'Qwen3-MoE',
    params: '30.5B / 激活 3.3B',
    layers: 48,
    config: {
      hidden_size: 2048,
      num_attention_heads: 32,
      num_key_value_heads: 4,
      head_dim: 128,
      num_experts: 128,
      num_experts_per_tok: 8,
      moe_intermediate_size: 768,
      rms_norm_eps: 1e-6,
      rope_theta: 1000000,
      max_position_embeddings: 40960,
      tie_word_embeddings: false,
      torch_dtype: 'bfloat16'
    },
    artifacts: [
      { id: 'config', name: 'config.json', kind: '模型配置', size: '1.4 KB', detail: 'Qwen3MoeForCausalLM · 48 层 · 128 专家', status: 'ok' },
      { id: 'weights', name: 'model.safetensors.index.json', kind: '权重索引', size: '61.2 GB', detail: '16 分片 · 579 张量 · bfloat16', status: 'ok' },
      { id: 'tokenizer', name: 'tokenizer.json', kind: 'Tokenizer', size: '11.4 MB', detail: 'BPE · vocab 151936 · 特殊 token 26', status: 'ok' },
      { id: 'generation', name: 'generation_config.json', kind: '生成配置', size: '243 B', detail: 'temperature 0.6 · top_p 0.95 · top_k 20', status: 'ok' },
      { id: 'custom', name: 'modeling_qwen3_moe.py', kind: '自定义模型代码', size: '48.7 KB', detail: '含 trust_remote_code 路径 · 3 处自定义 forward', status: 'warn' }
    ],
    structures: [
      { id: 'attn', label: 'GQA Attention', value: '32Q / 4KV · head_dim 128', sem: 'attention', confidence: 'high', evidence: 'config.num_key_value_heads=4 < num_attention_heads=32' },
      { id: 'qknorm', label: 'QK-Norm', value: 'RMSNorm on q/k per-head', sem: 'norm', confidence: 'high', evidence: '权重索引命中 self_attn.q_norm.weight / k_norm.weight' },
      { id: 'rope', label: 'RoPE', value: 'theta 1e6 · 非线性缩放关闭', sem: 'rope', confidence: 'high', evidence: 'rope_theta=1000000 · rope_scaling=null' },
      { id: 'norm', label: 'RMSNorm', value: 'pre-norm · eps 1e-6', sem: 'norm', confidence: 'high', evidence: '无 bias 项 · input_layernorm/post_attention_layernorm' },
      { id: 'moe', label: 'MoE Sparse MLP', value: 'top-8 / 128 experts · SwiGLU', sem: 'moe', confidence: 'high', evidence: 'mlp.experts.{i}.{gate,up,down}_proj · 128 组' },
      { id: 'router', label: 'Router Gate', value: 'softmax · norm_topk_prob', sem: 'gate', confidence: 'medium', evidence: 'mlp.gate.weight [128, 2048] · 未见 aux loss 项' },
      { id: 'kvcache', label: 'Paged KV Cache', value: 'block 128 · 2 × 4 × 128', sem: 'attention', confidence: 'high', evidence: '由 GQA 形状与 max_position 推导' },
      { id: 'lmhead', label: 'LM Head', value: 'untied · [151936, 2048]', sem: 'linear', confidence: 'high', evidence: 'tie_word_embeddings=false · lm_head.weight 独立分片' }
    ],
    /* 图节点：手工布局坐标（遵循 model-graphviz 契约，无自动布局） */
    graph: {
      width: 980,
      height: 1180,
      clusters: [
        { id: 'c-layer', label: 'Qwen3MoeDecoderLayer × 48', x: 48, y: 214, w: 884, h: 700, repeats: 48 },
        { id: 'c-attn', label: 'Self-Attention (GQA)', x: 76, y: 268, w: 396, h: 476 },
        { id: 'c-moe', label: 'Sparse MoE Block', x: 512, y: 268, w: 392, h: 476 }
      ],
      nodes: [
        { id: 'input_ids', kind: 'io', label: 'input_ids', op: 'Placeholder', x: 392, y: 34, w: 196, h: 34, shape: '[32, 1]', dtype: 'int64', sem: 'embedding', cov: 'native' },
        { id: 'embed', kind: 'module', label: 'embed_tokens', op: 'Embedding', x: 380, y: 96, w: 220, h: 40, shape: '[32, 1, 2048]', dtype: 'bf16', sem: 'embedding', cov: 'native' },
        { id: 'in_norm', kind: 'op', label: 'input_layernorm', op: 'RMSNorm', x: 380, y: 162, w: 220, h: 38, sem: 'norm', cov: 'fused' },

        { id: 'qkv', kind: 'op', label: 'qkv_proj', op: 'Linear·merged', x: 116, y: 300, w: 200, h: 38, shape: '[32, 5120]', dtype: 'bf16', sem: 'linear', cov: 'native' },
        { id: 'q_norm', kind: 'op', label: 'q_norm', op: 'RMSNorm', x: 100, y: 362, w: 108, h: 38, sem: 'norm', cov: 'fused' },
        { id: 'k_norm', kind: 'op', label: 'k_norm', op: 'RMSNorm', x: 224, y: 362, w: 108, h: 38, sem: 'norm', cov: 'fused' },
        { id: 'rope_op', kind: 'op', label: 'apply_rotary_emb', op: 'RoPE', x: 116, y: 424, w: 200, h: 38, sem: 'rope', cov: 'native' },
        { id: 'kv_cache', kind: 'state', label: 'paged_kv_cache', op: 'ReshapeAndCache', x: 340, y: 424, w: 118, h: 38, shape: '[blk, 128, 4, 128]', dtype: 'bf16', sem: 'attention', cov: 'native' },
        { id: 'attn_core', kind: 'op', label: 'paged_attention', op: 'FlashAttention', x: 116, y: 492, w: 200, h: 40, sem: 'attention', cov: 'native' },
        { id: 'attn_mask', kind: 'op', label: 'build_attn_mask', op: 'MaskBuilder', x: 340, y: 492, w: 118, h: 40, sem: 'attention', cov: 'fallback' },
        { id: 'o_proj', kind: 'op', label: 'o_proj', op: 'Linear', x: 116, y: 566, w: 200, h: 38, shape: '[32, 2048]', dtype: 'bf16', sem: 'linear', cov: 'native' },
        { id: 'attn_add', kind: 'op', label: 'residual_add', op: 'Add', x: 116, y: 628, w: 200, h: 38, sem: 'norm', cov: 'fused' },
        { id: 'post_norm', kind: 'op', label: 'post_attention_layernorm', op: 'RMSNorm', x: 116, y: 682, w: 200, h: 38, sem: 'norm', cov: 'fused' },

        { id: 'router', kind: 'op', label: 'mlp.gate', op: 'Router·Softmax', x: 556, y: 300, w: 200, h: 38, shape: '[32, 128]', dtype: 'fp32', sem: 'gate', cov: 'native' },
        { id: 'topk', kind: 'op', label: 'topk_softmax', op: 'TopK', x: 556, y: 362, w: 200, h: 38, shape: '[32, 8]', dtype: 'fp32', sem: 'gate', cov: 'library' },
        { id: 'permute', kind: 'op', label: 'permute_tokens', op: 'Gather·Sort', x: 556, y: 424, w: 200, h: 38, sem: 'moe', cov: 'fallback' },
        { id: 'grouped_gemm', kind: 'op', label: 'experts.grouped_gemm', op: 'GroupedMatmul', x: 540, y: 492, w: 232, h: 40, shape: '[256, 768]', dtype: 'bf16', sem: 'moe', cov: 'native' },
        { id: 'act', kind: 'op', label: 'swiglu', op: 'SwiGLU', x: 796, y: 492, w: 92, h: 40, sem: 'mlp', cov: 'fused' },
        { id: 'unpermute', kind: 'op', label: 'unpermute_and_reduce', op: 'Scatter·Add', x: 556, y: 566, w: 200, h: 38, sem: 'moe', cov: 'fallback' },
        { id: 'ep_comm', kind: 'op', label: 'all_to_all_ep', op: 'AllToAll', x: 796, y: 566, w: 92, h: 38, sem: 'comm', cov: 'library' },
        { id: 'moe_add', kind: 'op', label: 'residual_add', op: 'Add', x: 556, y: 628, w: 200, h: 38, sem: 'norm', cov: 'fused' },
        { id: 'shared_exp', kind: 'op', label: 'shared_expert', op: 'MLP·SwiGLU', x: 556, y: 682, w: 200, h: 38, sem: 'mlp', cov: 'unsupported' },

        { id: 'final_norm', kind: 'op', label: 'model.norm', op: 'RMSNorm', x: 380, y: 950, w: 220, h: 38, sem: 'norm', cov: 'native' },
        { id: 'lm_head', kind: 'module', label: 'lm_head', op: 'Linear', x: 380, y: 1014, w: 220, h: 40, shape: '[32, 151936]', dtype: 'bf16', sem: 'linear', cov: 'native' },
        { id: 'logits', kind: 'io', label: 'logits', op: 'Output', x: 392, y: 1084, w: 196, h: 34, shape: '[32, 1, 151936]', dtype: 'fp32', sem: 'linear', cov: 'native' }
      ],
      edges: [
        ['input_ids', 'embed'], ['embed', 'in_norm'],
        ['in_norm', 'qkv'], ['qkv', 'q_norm'], ['qkv', 'k_norm'],
        ['q_norm', 'rope_op'], ['k_norm', 'rope_op'],
        ['rope_op', 'kv_cache'], ['rope_op', 'attn_core'], ['kv_cache', 'attn_core'],
        ['attn_mask', 'attn_core'], ['attn_core', 'o_proj'], ['o_proj', 'attn_add'],
        ['in_norm', 'attn_add'], ['attn_add', 'post_norm'],
        ['post_norm', 'router'], ['router', 'topk'], ['topk', 'permute'],
        ['permute', 'grouped_gemm'], ['grouped_gemm', 'act'], ['act', 'unpermute'],
        ['grouped_gemm', 'unpermute'], ['unpermute', 'ep_comm'], ['unpermute', 'moe_add'],
        ['attn_add', 'moe_add'], ['moe_add', 'shared_exp'],
        ['shared_exp', 'final_norm'], ['final_norm', 'lm_head'], ['lm_head', 'logits']
      ]
    }
  };

  /* ---------------- 算子覆盖报告（逐模型节点） ---------------- */
  qwen3moe.coverage = [
    { node: 'embed', op: 'Embedding', cov: 'native', impl: 'pypto.ops.embedding', latency: 0.04, share: 0.4, note: '词表并行已切分，无 gather 同步' },
    { node: 'in_norm', op: 'RMSNorm', cov: 'fused', impl: 'fused into qkv_proj prologue', latency: 0.00, share: 0.0, note: 'norm 与 qkv 前置融合，省一次 HBM 往返' },
    { node: 'qkv', op: 'Linear·merged', cov: 'native', impl: 'pypto.ops.matmul_nz', latency: 0.31, share: 3.1, note: 'q/k/v 合并为单次 GEMM' },
    { node: 'q_norm', op: 'RMSNorm', cov: 'fused', impl: 'fused into rope epilogue', latency: 0.00, share: 0.0, note: 'per-head norm 与 RoPE 融合' },
    { node: 'k_norm', op: 'RMSNorm', cov: 'fused', impl: 'fused into rope epilogue', latency: 0.00, share: 0.0, note: '同 q_norm' },
    { node: 'rope_op', op: 'RoPE', cov: 'native', impl: 'pypto.ops.rope_qk_norm', latency: 0.09, share: 0.9, note: 'cos/sin 表常驻 L1' },
    { node: 'kv_cache', op: 'ReshapeAndCache', cov: 'native', impl: 'pypto.ops.reshape_and_cache', latency: 0.07, share: 0.7, note: 'block_size 128 与 paged attention 对齐' },
    { node: 'attn_core', op: 'FlashAttention', cov: 'native', impl: 'pypto.ops.paged_attention_v2', latency: 1.12, share: 11.2, note: 'GQA 4KV 分组，decode 走 split-KV' },
    { node: 'attn_mask', op: 'MaskBuilder', cov: 'fallback', impl: 'aten::triu → Host 构造', latency: 1.86, share: 18.6, note: '每步在 Host 重建 mask 并 H2D 拷贝，引入流同步', p: 'P0' },
    { node: 'o_proj', op: 'Linear', cov: 'native', impl: 'pypto.ops.matmul_nz', latency: 0.28, share: 2.8, note: 'TP 行并行 + reduce-scatter 重叠' },
    { node: 'attn_add', op: 'Add', cov: 'fused', impl: 'fused into o_proj epilogue', latency: 0.00, share: 0.0, note: '残差加法融合进 GEMM epilogue' },
    { node: 'post_norm', op: 'RMSNorm', cov: 'fused', impl: 'fused into router prologue', latency: 0.00, share: 0.0, note: '与 router GEMM 前置融合' },
    { node: 'router', op: 'Router·Softmax', cov: 'native', impl: 'pypto.ops.moe_router', latency: 0.06, share: 0.6, note: 'fp32 累加保证 top-k 稳定性' },
    { node: 'topk', op: 'TopK', cov: 'library', impl: 'ATB::TopkSoftmax', latency: 0.14, share: 1.4, note: '库算子接口稳定，但无法与 permute 融合', p: 'P2' },
    { node: 'permute', op: 'Gather·Sort', cov: 'fallback', impl: 'aten::sort + index_select', latency: 1.42, share: 14.2, note: 'token 重排在 Host 侧计算 offset，device 空转', p: 'P0' },
    { node: 'grouped_gemm', op: 'GroupedMatmul', cov: 'native', impl: 'pypto.ops.grouped_matmul', latency: 2.31, share: 23.1, note: '128 专家动态 group_list，已开启 A2A 重叠' },
    { node: 'act', op: 'SwiGLU', cov: 'fused', impl: 'fused into grouped_matmul epilogue', latency: 0.00, share: 0.0, note: 'gate/up 双路激活融合' },
    { node: 'unpermute', op: 'Scatter·Add', cov: 'fallback', impl: 'aten::index_add_', latency: 1.28, share: 12.8, note: '与 permute 对称的回排，同样受 Host 依赖', p: 'P1' },
    { node: 'ep_comm', op: 'AllToAll', cov: 'library', impl: 'HCCL::AllToAllV', latency: 0.52, share: 5.2, note: 'EP=8 通信，已与 grouped_gemm 部分重叠' },
    { node: 'moe_add', op: 'Add', cov: 'fused', impl: 'fused into unpermute epilogue', latency: 0.00, share: 0.0, note: '' },
    { node: 'shared_exp', op: 'MLP·SwiGLU', cov: 'unsupported', impl: '—', latency: 0.00, share: 0.0, note: 'A3 平台缺少 shared-expert 与稀疏专家并行执行的调度原语，整网无法闭环', p: 'P0' },
    { node: 'final_norm', op: 'RMSNorm', cov: 'native', impl: 'pypto.ops.rms_norm', latency: 0.03, share: 0.3, note: '' },
    { node: 'lm_head', op: 'Linear', cov: 'native', impl: 'pypto.ops.matmul_nz', latency: 0.41, share: 4.1, note: '词表并行 + all-gather logits' }
  ];

  /* ---------------- 精度与量化画像 ---------------- */
  qwen3moe.precision = {
    baseline: 'BF16',
    modules: [
      { id: 'embed', name: 'embed_tokens', support: { FP16: 'ok', BF16: 'ok', FP8: 'na', INT8: 'ok' }, current: 'BF16', note: '量化收益低，建议保持 BF16' },
      { id: 'qkv', name: 'qkv_proj', support: { FP16: 'ok', BF16: 'ok', FP8: 'ok', INT8: 'calib' }, current: 'BF16', note: 'INT8 需 per-channel 校准' },
      { id: 'attn_core', name: 'paged_attention', support: { FP16: 'ok', BF16: 'ok', FP8: 'calib', INT8: 'no' }, current: 'BF16', note: 'FP8 仅支持 KV cache，QK 累加须 FP32' },
      { id: 'grouped_gemm', name: 'experts.grouped_gemm', support: { FP16: 'ok', BF16: 'ok', FP8: 'ok', INT8: 'calib' }, current: 'BF16', note: '专家权重 INT8 收益最高（-42% 权重带宽）' },
      { id: 'router', name: 'mlp.gate', support: { FP16: 'no', BF16: 'no', FP8: 'no', INT8: 'no' }, current: 'FP32', note: 'router 必须 FP32，否则 top-k 抖动导致输出漂移' },
      { id: 'lm_head', name: 'lm_head', support: { FP16: 'ok', BF16: 'ok', FP8: 'calib', INT8: 'calib' }, current: 'BF16', note: 'logits 输出需 FP32 upcast' }
    ],
    boundaries: [
      { from: 'router', to: 'topk', at: 'FP32 → FP32', reason: '路由分数全程 FP32，禁止降精度', level: 'block' },
      { from: 'attn_core', to: 'o_proj', at: 'FP32 累加 → BF16', reason: 'softmax 累加器 FP32，输出 cast 回 BF16', level: 'ok' },
      { from: 'grouped_gemm', to: 'unpermute', at: 'INT8 → BF16', reason: '开启专家量化后需在回排前反量化', level: 'warn' },
      { from: 'lm_head', to: 'logits', at: 'BF16 → FP32', reason: '采样前 upcast，避免 top_p 截断误差', level: 'ok' }
    ],
    quant: {
      scheme: 'W8A16 per-channel',
      calib: { dataset: 'C4 + 内部指令集', samples: 512, seqlen: 2048, status: 'ready' },
      metrics: [
        { name: '权重带宽', before: '61.2 GB', after: '32.4 GB', delta: '-47.1%', good: true },
        { name: 'PPL (wikitext2)', before: '6.84', after: '6.91', delta: '+0.07', good: true },
        { name: 'MMLU', before: '78.2', after: '77.6', delta: '-0.6', good: true },
        { name: 'Decode 吞吐', before: '1.00×', after: '1.38×', delta: '+38%', good: true }
      ],
      risks: [
        'router 与 shared_expert 不参与量化，需在 recipe 中显式排除',
        'FP8 路径依赖 A5/950B，A3 上会静默回落到 BF16'
      ]
    }
  };

  /* ---------------- Model Recipe（可组合） ---------------- */
  qwen3moe.recipes = [
    {
      id: 'r-prefill', name: 'Prefill · Chunked', stage: 'prefill', sem: 'attention', selected: true,
      summary: '长序列分块预填充，KV 边算边写入 paged cache',
      applies: { models: 'Qwen3 / Qwen2.5 全系', shapes: 'seq 512–40960 · bs 1–8', platform: 'A2 / A3 / A5' },
      refs: ['pypto/recipes/prefill_chunked.py', 'vllm_ascend/attention/prefill.py'],
      precision: { threshold: 'rel_err ≤ 2e-3 vs BF16 参考', accum: 'FP32 softmax 累加' },
      baseline: { metric: 'MFU', value: '48.6%', at: 'seq 4096 · A3' },
      limits: ['chunk 大小需为 block_size 整数倍', 'sliding-window 变体尚未验证'],
      tuning: ['chunk_size 1024 → 2048 可换取 6% MFU', 'double-buffer KV 写入以掩盖 HBM 延迟'],
      nodes: ['attn_core', 'kv_cache', 'attn_mask']
    },
    {
      id: 'r-decode', name: 'Decode · Split-KV', stage: 'decode', sem: 'attention', selected: true,
      summary: '增量解码沿 KV 长度切分，提升小 batch 下的 AIC 占用',
      applies: { models: 'GQA / MQA 架构', shapes: 'bs 1–256 · seq 1', platform: 'A2 / A3 / A5' },
      refs: ['pypto/recipes/decode_split_kv.py'],
      precision: { threshold: 'rel_err ≤ 1e-3', accum: 'FP32' },
      baseline: { metric: 'TPOT', value: '18.4 ms', at: 'bs 32 · A3 · TP8' },
      limits: ['split 数 > 8 时 reduce 开销反超收益'],
      tuning: ['按 seq_len 动态选择 split 数', 'KV cache 采用 FP8 可再降 30% 带宽'],
      nodes: ['attn_core', 'kv_cache']
    },
    {
      id: 'r-paged', name: 'Paged Attention', stage: 'both', sem: 'attention', selected: true,
      summary: '块表管理的非连续 KV 存储，支持前缀复用',
      applies: { models: '全部自回归模型', shapes: 'block 64 / 128', platform: 'A2 / A3 / A5' },
      refs: ['pypto/recipes/paged_attention.py', 'docs/kv_cache_layout.md'],
      precision: { threshold: '与连续 KV 逐元素一致', accum: 'FP32' },
      baseline: { metric: '显存利用率', value: '92.1%', at: 'block 128' },
      limits: ['block_size 128 在 A2 上有 8% 对齐浪费'],
      tuning: ['前缀缓存命中率 > 40% 时启用 copy-on-write'],
      nodes: ['kv_cache', 'attn_core']
    },
    {
      id: 'r-rmsnorm-rope', name: 'RMSNorm + RoPE', stage: 'both', sem: 'norm', selected: true,
      summary: 'norm 与旋转位置编码融合，含 QK-Norm 变体',
      applies: { models: 'Qwen3 (QK-Norm) / LLaMA', shapes: 'hidden 2048–8192', platform: 'A2 / A3 / A5' },
      refs: ['pypto/recipes/rmsnorm_rope_fused.py'],
      precision: { threshold: 'rel_err ≤ 5e-4', accum: 'FP32 平方和' },
      baseline: { metric: '带宽利用率', value: '86.3%', at: 'hidden 2048' },
      limits: ['eps < 1e-8 时 FP16 下溢', 'rope_scaling 动态变体需另配 recipe'],
      tuning: ['cos/sin 表常驻 L1 可省 12% 访存'],
      nodes: ['in_norm', 'q_norm', 'k_norm', 'rope_op', 'post_norm', 'final_norm']
    },
    {
      id: 'r-moe', name: 'MoE Expert · Grouped', stage: 'both', sem: 'moe', selected: true,
      summary: '动态 group_list 的分组 GEMM，含 permute/unpermute 与 EP 通信重叠',
      applies: { models: 'Qwen3-MoE / Mixtral / DeepSeek', shapes: 'experts 8–256 · top-k 2–8', platform: 'A3 / A5' },
      refs: ['pypto/recipes/moe_grouped_gemm.py', 'pypto/recipes/moe_dispatch.py'],
      precision: { threshold: 'router FP32 · 专家 rel_err ≤ 2e-3', accum: 'FP32' },
      baseline: { metric: 'AIC 利用率', value: '71.2%', at: 'top-8 / 128 · EP8' },
      limits: ['专家负载不均时长尾明显', 'shared-expert 并行调度在 A3 上不可用'],
      tuning: ['permute 下沉到 device 可消除 Host 同步', 'A2A 与 GEMM 分块重叠'],
      nodes: ['router', 'topk', 'permute', 'grouped_gemm', 'act', 'unpermute', 'ep_comm', 'shared_exp']
    },
    {
      id: 'r-lmhead', name: 'LM Head · Vocab Parallel', stage: 'both', sem: 'linear', selected: false,
      summary: '词表并行投影，采样前 all-gather logits',
      applies: { models: 'vocab ≥ 32000', shapes: 'vocab 151936 · TP 2–16', platform: 'A2 / A3 / A5' },
      refs: ['pypto/recipes/lm_head_vocab_parallel.py'],
      precision: { threshold: 'logits FP32 输出', accum: 'FP32' },
      baseline: { metric: '通信占比', value: '4.1%', at: 'TP8 · bs 32' },
      limits: ['bs > 128 时 all-gather 成为瓶颈，需切换到分布式采样'],
      tuning: ['只 gather top-k 候选可省 90% 通信'],
      nodes: ['lm_head', 'logits', 'final_norm']
    }
  ];

  /* ---------------- 子图 pattern matching → 实现骨架 ---------------- */
  qwen3moe.patterns = [
    {
      id: 'p-moe', name: 'MoE Sparse Block', recipe: 'r-moe', confidence: 0.94,
      matched: ['router', 'topk', 'permute', 'grouped_gemm', 'act', 'unpermute'],
      source: 'modeling_qwen3_moe.py:Qwen3MoeSparseMoeBlock.forward:412',
      target: 'pypto/models/qwen3_moe/moe_block.py',
      gaps: ['permute/unpermute 需下沉 device', 'shared_expert 无平台实现'],
      code: [
        { l: 1, t: '@pypto.recipe("moe_grouped_gemm")', kind: 'dec' },
        { l: 2, t: 'class Qwen3MoeSparseBlock(pypto.Module):', kind: 'def' },
        { l: 3, t: '    """由 modeling_qwen3_moe.py:412 pattern-match 生成。"""', kind: 'doc' },
        { l: 4, t: '', kind: 'blank' },
        { l: 5, t: '    def __init__(self, cfg: Qwen3MoeConfig):', kind: 'def' },
        { l: 6, t: '        self.top_k = cfg.num_experts_per_tok        # 8', kind: 'code', map: 'router' },
        { l: 7, t: '        self.n_experts = cfg.num_experts            # 128', kind: 'code', map: 'router' },
        { l: 8, t: '        self.gate = pypto.Linear(cfg.hidden_size, cfg.num_experts,', kind: 'code', map: 'router' },
        { l: 9, t: '                                 dtype=pypto.float32)  # 精度边界: 强制 FP32', kind: 'code', map: 'router' },
        { l: 10, t: '        self.experts = pypto.GroupedLinear(', kind: 'code', map: 'grouped_gemm' },
        { l: 11, t: '            cfg.num_experts, cfg.hidden_size, cfg.moe_intermediate_size)', kind: 'code', map: 'grouped_gemm' },
        { l: 12, t: '', kind: 'blank' },
        { l: 13, t: '    def forward(self, x: pypto.Tensor) -> pypto.Tensor:', kind: 'def' },
        { l: 14, t: '        scores = pypto.ops.moe_router(x, self.gate.weight)         # native', kind: 'code', map: 'router' },
        { l: 15, t: '        w, idx = pypto.ops.topk_softmax(scores, self.top_k)        # library', kind: 'code', map: 'topk' },
        { l: 16, t: '        # TODO(P0): 当前回落到 aten::sort，需实现 device 侧 permute', kind: 'todo', map: 'permute' },
        { l: 17, t: '        xp, group_list, inv = pypto.ops.permute_tokens(x, idx)     # fallback', kind: 'code', map: 'permute' },
        { l: 18, t: '        h = pypto.ops.grouped_matmul(xp, self.experts.w, group_list,', kind: 'code', map: 'grouped_gemm' },
        { l: 19, t: '                                     epilogue="swiglu")            # act 已融合', kind: 'code', map: 'act' },
        { l: 20, t: '        # TODO(P1): index_add_ 回落，与 permute 一并下沉', kind: 'todo', map: 'unpermute' },
        { l: 21, t: '        return pypto.ops.unpermute_reduce(h, inv, w)               # fallback', kind: 'code', map: 'unpermute' },
        { l: 22, t: '', kind: 'blank' },
        { l: 23, t: '    # UNSUPPORTED: shared_expert 需与稀疏专家并行调度，A3 缺少原语', kind: 'err', map: 'shared_exp' },
        { l: 24, t: '    #   → 建议切换 A5/950B 或串行执行（TPOT +2.1ms）', kind: 'err', map: 'shared_exp' }
      ]
    },
    {
      id: 'p-attn', name: 'GQA Attention + QK-Norm', recipe: 'r-decode', confidence: 0.97,
      matched: ['qkv', 'q_norm', 'k_norm', 'rope_op', 'kv_cache', 'attn_core', 'o_proj'],
      source: 'modeling_qwen3_moe.py:Qwen3MoeAttention.forward:238',
      target: 'pypto/models/qwen3_moe/attention.py',
      gaps: ['attn_mask 每步 Host 重建'],
      code: [
        { l: 1, t: '@pypto.recipe("decode_split_kv", "rmsnorm_rope_fused")', kind: 'dec' },
        { l: 2, t: 'class Qwen3MoeAttention(pypto.Module):', kind: 'def' },
        { l: 3, t: '    """由 modeling_qwen3_moe.py:238 pattern-match 生成。"""', kind: 'doc' },
        { l: 4, t: '', kind: 'blank' },
        { l: 5, t: '    def __init__(self, cfg: Qwen3MoeConfig):', kind: 'def' },
        { l: 6, t: '        self.n_q, self.n_kv = cfg.num_attention_heads, cfg.num_key_value_heads', kind: 'code', map: 'qkv' },
        { l: 7, t: '        self.qkv = pypto.MergedLinear(cfg.hidden_size,', kind: 'code', map: 'qkv' },
        { l: 8, t: '            [self.n_q * cfg.head_dim] + [self.n_kv * cfg.head_dim] * 2)', kind: 'code', map: 'qkv' },
        { l: 9, t: '        self.q_norm = pypto.RMSNorm(cfg.head_dim, eps=cfg.rms_norm_eps)', kind: 'code', map: 'q_norm' },
        { l: 10, t: '        self.k_norm = pypto.RMSNorm(cfg.head_dim, eps=cfg.rms_norm_eps)', kind: 'code', map: 'k_norm' },
        { l: 11, t: '        self.o_proj = pypto.Linear(self.n_q * cfg.head_dim, cfg.hidden_size)', kind: 'code', map: 'o_proj' },
        { l: 12, t: '', kind: 'blank' },
        { l: 13, t: '    def forward(self, x, pos, kv_cache, meta):', kind: 'def' },
        { l: 14, t: '        q, k, v = self.qkv(x).split()                              # native', kind: 'code', map: 'qkv' },
        { l: 15, t: '        # q_norm / k_norm 已融合进 rope epilogue', kind: 'note', map: 'q_norm' },
        { l: 16, t: '        q, k = pypto.ops.rope_qk_norm(q, k, pos,', kind: 'code', map: 'rope_op' },
        { l: 17, t: '            q_weight=self.q_norm.weight, k_weight=self.k_norm.weight)', kind: 'code', map: 'rope_op' },
        { l: 18, t: '        pypto.ops.reshape_and_cache(k, v, kv_cache, meta.slot_map)  # native', kind: 'code', map: 'kv_cache' },
        { l: 19, t: '        # TODO(P0): mask 由 Host 构造并 H2D，占 18.6% 端到端耗时', kind: 'todo', map: 'attn_mask' },
        { l: 20, t: '        mask = meta.attn_mask                                      # fallback', kind: 'code', map: 'attn_mask' },
        { l: 21, t: '        o = pypto.ops.paged_attention_v2(q, kv_cache, meta.block_table,', kind: 'code', map: 'attn_core' },
        { l: 22, t: '            mask=mask, n_kv=self.n_kv, accum=pypto.float32)        # native', kind: 'code', map: 'attn_core' },
        { l: 23, t: '        return self.o_proj(o, epilogue="residual_add")              # add 已融合', kind: 'code', map: 'o_proj' }
      ]
    }
  ];

  return { SEM, COVERAGE_META, PRECISION_ORDER, models: { 'qwen3-moe': qwen3moe } };
})();
