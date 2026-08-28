(function registerDeepSeekV4ModelViz() {
  'use strict';

  const MODEL_ID = 'deepseek-v4-flash';
  const stageId = 'qwen3ModelGraph';
  const CONFIG_URL = '../../Data/DeepSeek-V4-Flash-Official/inference-config.json';

  const graph = {
    width: 1240,
    height: 2140,
    clusters: [
      { id: 'embedding', label: 'Embedding · Input', x: 270, y: 50, width: 700, height: 180 },
      { id: 'decoder-stack', label: 'DeepSeek-V4 Flash Decoder × 43', x: 150, y: 280, width: 940, height: 1550, repeat: 43 },
      { id: 'hybrid-attention', label: 'Hybrid Attention · CSA / HCA', x: 200, y: 480, width: 840, height: 420, parent: 'decoder-stack' },
      { id: 'moe-block', label: 'DeepSeekMoE · 256 routed experts', x: 200, y: 1020, width: 840, height: 500, parent: 'decoder-stack' },
      { id: 'mhc-block', label: 'mHC Residual Connection', x: 200, y: 1590, width: 840, height: 180, parent: 'decoder-stack' },
      { id: 'output', label: 'Output · MTP / LM Head', x: 270, y: 1900, width: 700, height: 170 },
    ],
    nodes: [
      { id: 'tokens', label: 'input_ids', typeLabel: 'Input · [batch, seq] · INT64', kind: 'tensor', x: 620, y: 88, width: 260, height: 54, colorKey: 'io:input', phase: 'embedding', parent: 'embedding' },
      { id: 'embedding-op', label: 'token_embedding', typeLabel: 'Vocab 129,280 · Hidden 4,096', kind: 'op', x: 620, y: 178, width: 300, height: 58, colorKey: 'sem:linear', phase: 'embedding', parent: 'embedding' },
      { id: 'layer-input', label: 'hidden_states', typeLabel: 'BF16 · [batch, seq, 4096]', kind: 'tensor', x: 620, y: 340, width: 300, height: 54, colorKey: 'io:activation', phase: 'layer' },
      { id: 'pre-norm', label: 'input_rmsnorm', typeLabel: 'RMSNorm · eps 1e-6', kind: 'op', x: 620, y: 420, width: 270, height: 56, colorKey: 'sem:norm', phase: 'layer', parent: 'decoder-stack' },
      // Attention 拆成三个 L2 节点，边界对齐 torch_npu 的算子边界：
      // qkv-proj ↔ npu_mla_prolog_v3 · sparse-attn ↔ npu_sparse_flash_attention · out-proj ↔ 输出投影
      { id: 'qkv-proj', label: 'mla_prolog', typeLabel: 'Q/KV 投影 + RMSNorm + RoPE + Cache 写入', kind: 'op', x: 620, y: 540, width: 380, height: 62, colorKey: 'sem:linear', phase: 'attention', parent: 'hybrid-attention' },
      { id: 'indexer', label: 'indexer', typeLabel: '64 heads · top-k 512 · compress_ratio 4 的层', kind: 'op', x: 360, y: 640, width: 280, height: 58, colorKey: 'sem:attention', phase: 'attention', parent: 'hybrid-attention' },
      { id: 'compressor', label: 'compressor', typeLabel: 'Gated pooling 压缩 K/V · CSA/HCA', kind: 'op', x: 870, y: 640, width: 280, height: 58, colorKey: 'sem:attention', phase: 'attention', parent: 'hybrid-attention' },
      { id: 'sparse-attn', label: 'sparse_attn', typeLabel: '稀疏注意力 · top-k KV block', kind: 'op', x: 600, y: 745, width: 300, height: 58, colorKey: 'sem:attention', phase: 'attention', parent: 'hybrid-attention' },
      { id: 'rope-cache', label: 'KV cache', typeLabel: 'Window 128 · paged / ring buffer', kind: 'state', x: 895, y: 745, width: 230, height: 54, colorKey: 'io:state', phase: 'attention', parent: 'hybrid-attention' },
      { id: 'out-proj', label: 'out_projection', typeLabel: 'wo_a 分组 einsum → wo_b · 8 组 × rank 1,024', kind: 'op', x: 620, y: 845, width: 320, height: 58, colorKey: 'sem:linear', phase: 'attention', parent: 'hybrid-attention' },
      { id: 'attn-output', label: 'attention_output', typeLabel: 'BF16 · compressed attention output', kind: 'tensor', x: 620, y: 925, width: 330, height: 54, colorKey: 'io:activation', phase: 'attention' },
      { id: 'router', label: 'moe_router', typeLabel: 'sqrtsoftplus · top-k = 6 · scale 1.5', kind: 'op', x: 420, y: 1090, width: 330, height: 58, colorKey: 'sem:gate', phase: 'moe', parent: 'moe-block' },
      { id: 'expert-select', label: 'expert_dispatch', typeLabel: '256 routed + 1 shared expert', kind: 'op', x: 820, y: 1090, width: 330, height: 58, colorKey: 'sem:comm', phase: 'moe', parent: 'moe-block' },
      { id: 'shared-expert', label: 'shared_expert', typeLabel: 'SwiGLU · intermediate 2,048', kind: 'op', x: 390, y: 1230, width: 300, height: 58, colorKey: 'sem:mlp', phase: 'moe', parent: 'moe-block' },
      { id: 'routed-experts', label: 'routed_experts', typeLabel: '6 active experts / token · FP4 experts', kind: 'op', x: 850, y: 1230, width: 330, height: 58, colorKey: 'sem:mlp', phase: 'moe', parent: 'moe-block' },
      { id: 'expert-swiglu', label: 'SwiGLU + grouped GEMM', typeLabel: 'gate/up → SiLU → multiply → down', kind: 'op', x: 620, y: 1370, width: 360, height: 62, colorKey: 'sem:mlp', phase: 'moe', parent: 'moe-block' },
      { id: 'combine', label: 'weighted_combine', typeLabel: 'Route weights → hidden output', kind: 'op', x: 620, y: 1480, width: 320, height: 58, colorKey: 'sem:gate', phase: 'moe', parent: 'moe-block' },
      { id: 'mhc', label: 'mHC', typeLabel: 'Manifold-constrained hyper-connection', kind: 'op', x: 620, y: 1660, width: 360, height: 62, colorKey: 'sem:norm', phase: 'mhc', parent: 'mhc-block' },
      { id: 'layer-output', label: 'next_hidden_states', typeLabel: 'BF16 · repeated for 43 layers', kind: 'tensor', x: 620, y: 1810, width: 330, height: 54, colorKey: 'io:activation', phase: 'mhc' },
      { id: 'final-norm', label: 'final_rmsnorm', typeLabel: 'RMSNorm · hidden 4,096', kind: 'op', x: 620, y: 1940, width: 300, height: 58, colorKey: 'sem:norm', phase: 'output', parent: 'output' },
      { id: 'lm-head', label: 'lm_head', typeLabel: 'Hidden 4,096 → Vocab 129,280', kind: 'op', x: 620, y: 2030, width: 330, height: 58, colorKey: 'sem:head', phase: 'output', parent: 'output' },
      { id: 'logits', label: 'logits', typeLabel: 'Output · [batch, seq, 129280]', kind: 'tensor', x: 620, y: 2120, width: 280, height: 54, colorKey: 'io:output', phase: 'output' },
    ],
    edges: [
      { source: 'tokens', target: 'embedding-op', tag: 'token ids' },
      { source: 'embedding-op', target: 'layer-input', tag: 'BF16 hidden' },
      { source: 'layer-input', target: 'pre-norm' },
      { source: 'pre-norm', target: 'indexer', tag: 'normalized input' },
      { source: 'pre-norm', target: 'compressor', tag: 'normalized input' },
      { source: 'pre-norm', target: 'qkv-proj', tag: 'normalized input' },
      { source: 'qkv-proj', target: 'indexer', tag: 'qr', dashed: true },
      { source: 'qkv-proj', target: 'rope-cache', tag: 'KV 写入' },
      { source: 'qkv-proj', target: 'sparse-attn', tag: 'Q' },
      { source: 'indexer', target: 'sparse-attn', tag: 'top-k index', dashed: true },
      // 压缩 KV 和 window KV 共用同一块 buffer：compressor.kv_cache = kv_cache[:, win:]（model.py:491）
      { source: 'compressor', target: 'rope-cache', tag: '压缩块写入', dashed: true },
      { source: 'compressor', target: 'sparse-attn', tag: 'compressed K/V', dashed: true },
      { source: 'rope-cache', target: 'sparse-attn', tag: 'cache 读取', dashed: true },
      { source: 'sparse-attn', target: 'out-proj', tag: 'attention 输出' },
      { source: 'out-proj', target: 'attn-output', tag: 'CSA / HCA output' },
      { source: 'attn-output', target: 'router', tag: 'attention output' },
      { source: 'router', target: 'expert-select', tag: 'indices + weights' },
      { source: 'router', target: 'shared-expert', tag: 'shared path', dashed: true },
      { source: 'expert-select', target: 'routed-experts', tag: '6 experts/token' },
      { source: 'shared-expert', target: 'expert-swiglu', tag: 'shared output' },
      { source: 'routed-experts', target: 'expert-swiglu', tag: 'grouped GEMM' },
      { source: 'expert-swiglu', target: 'combine', tag: 'expert outputs' },
      { source: 'router', target: 'combine', tag: 'route weights', dashed: true },
      { source: 'combine', target: 'mhc', tag: 'MoE output' },
      { source: 'layer-input', target: 'mhc', tag: 'residual', dashed: true },
      { source: 'mhc', target: 'layer-output', tag: 'next layer' },
      { source: 'layer-output', target: 'pre-norm', tag: '×43', dashed: true },
      { source: 'layer-output', target: 'final-norm', tag: 'after layer 43' },
      { source: 'final-norm', target: 'lm-head', tag: 'normalized hidden' },
      { source: 'lm-head', target: 'logits', tag: 'vocabulary logits' },
    ],
  };

  const phaseMeta = {
    all: ['DeepSeek-V4 Flash · Full Network', '43 × Decoder Layer · Hybrid Attention · DeepSeekMoE · mHC', '完整链路', '官方结构参数驱动的整网结构视图；节点为网络模块与主要数据状态。'],
    embedding: ['Embedding · Input', 'Vocab 129,280 → Hidden 4,096', '输入边界', '输入 token 经 embedding 后进入 43 层 Decoder。'],
    layer: ['Decoder Layer · Pre-Norm', 'RMSNorm → Hybrid Attention → MoE → mHC', '层结构', '每层共享主结构，部分 attention 压缩比例由 compress_ratios 指定。'],
    attention: ['Hybrid Attention · CSA / HCA', 'Indexer top-k 512 → compressed attention → RoPE/KV cache', '混合注意力', 'DeepSeek-V4 Flash 使用压缩注意力路径，需结合 prefill/decode 分别评估。'],
    moe: ['DeepSeekMoE · Router + Experts', '256 routed experts · 1 shared expert · 6 active/token', 'MoE 子图', 'Router 产生 expert indices/weights，随后执行 shared 与 routed expert 路径并合并。'],
    mhc: ['mHC · Residual Connection', 'Manifold-constrained hyper-connection · 43-layer carry', '跨层连接', 'mHC 是 V4 相对传统残差连接的结构变化，需作为独立模块观察。'],
    output: ['Output · Final Norm + LM Head', 'Hidden 4,096 → Vocabulary 129,280', '输出边界', '最终 RMSNorm 与词表投影产生 logits。'],
  };

  const phaseOrder = ['all', 'embedding', 'attention', 'moe', 'mhc', 'output'];
  const facts = [
    ['Decoder', '43 层'],
    ['Total / Active', '284B / 13B'],
    ['Hidden', '4,096'],
    ['Attention', '64 Q / 1 KV'],
    ['MoE', '256 routed + 1 shared'],
    ['Top-k', '6 experts/token'],
    ['Context', '1,048,576'],
    ['Precision', 'FP4 expert / FP8 mixed'],
  ];

  const drillSpecs = {
    'embedding-op': { description: '并行词表切片 Embedding；多卡场景下先屏蔽非本 rank 的 token，再通过 AllReduce 合并。', steps: [['mask token ids', 'Shard range check · INT64', 'sem:act'], ['F.embedding', '[batch, seq] → [batch, seq, 4096]', 'sem:linear'], ['zero out remote rows', 'Tensor mask · distributed', 'sem:act'], ['all_reduce embedding', 'TP combine · optional', 'sem:comm']] },
    'pre-norm': { description: 'Decoder 层输入归一化；模型主干使用 FP32 归约后回到激活 dtype。', steps: [['cast FP32', 'Precision boundary', 'sem:linear'], ['square + mean', 'Last-dim reduction', 'sem:act'], ['rsqrt(var + eps)', 'RMS reciprocal · eps 1e-6', 'sem:norm'], ['multiply gamma', 'Elementwise scale · BF16/FP8', 'sem:norm']] },
    indexer: { description: 'CSA 的 lightning indexer：对压缩 KV 做加权打分并选择 top-k 压缩块。', steps: [['wq_b projection', 'Q low-rank → 64 × 128', 'sem:linear'], ['RoPE + Hadamard', 'Position rotate · FP4 path', 'sem:rope'], ['fp4_act_quant', 'FP4 simulation · block 32', 'sem:quant'], ['einsum q · k', 'Indexer score · FP32', 'sem:attention'], ['relu × weights', 'Weighted score reduction', 'sem:act'], ['topk + causal mask', 'top-k 512 · index offset', 'sem:gate']] },
    compressor: { description: 'CSA/HCA 共用的压缩器：学习 gated pooling，将连续 KV 压缩后写入压缩缓存。', steps: [['wkv projection', 'FP32 · value candidate', 'sem:linear'], ['wgate projection', 'FP32 · learned gate', 'sem:linear'], ['window reshape', 'ratio 4 / 128 · overlap aware', 'sem:act'], ['softmax gated pooling', 'Token window reduction', 'sem:norm'], ['RMSNorm + RoPE', 'Compressed KV normalization', 'sem:rope'], ['quant + cache write', 'FP8/FP4 · KV cache update', 'sem:comm']] },
    'qkv-proj': { description: 'MLA 前处理：Q 低秩投影与归一化、latent KV 投影、RoPE 与 KV Cache 写入。对应 npu_mla_prolog_v3 的覆盖范围（model.py:496-506、530）。', steps: [['wq_a + q_norm', 'Q low-rank → rank 1,024', 'sem:linear'], ['wq_b + q reshape', '64 heads × head_dim 512', 'sem:linear'], ['q per-head rsqrt', '无权重归一化 · model.py:498', 'sem:norm'], ['apply_rotary_emb(q)', 'rope_head_dim 64 · YaRN', 'sem:rope'], ['wkv + kv_norm', 'Latent KV · 1 KV head', 'sem:linear'], ['RoPE + act_quant(kv)', 'FP8 non-rope · scale ue8m0', 'sem:quant'], ['KV cache write', 'window 128 · ring / paged', 'io:state']] },
    'sparse-attn': { description: '稀疏注意力主体：先拼出 window 与压缩块的 top-k 索引，再在选中的 KV block 上做注意力。对应 npu_sparse_flash_attention（model.py:507-535）。', steps: [['window topk indices', 'Sliding window 128', 'sem:gate'], ['compress topk idxs', 'indexer 或 静态索引', 'sem:gate'], ['cache read', 'Window + 压缩块 KV', 'io:state'], ['sparse_attn', 'attn_sink · softmax_scale', 'sem:attention'], ['apply RoPE inverse', 'Attention 输出反旋转', 'sem:rope']] },
    'out-proj': { description: '分组输出投影：先按 o_groups 分组做 einsum，再经 wo_b 回到 hidden（model.py:538-543）。', steps: [['reshape to groups', '8 组 × head 输出', 'sem:act'], ['wo_a einsum', 'bsgd,grd→bsgr · rank 1,024', 'sem:linear'], ['wo_b projection', 'Grouped output → hidden 4,096', 'sem:linear']] },
    'rope-cache': { description: '滑窗 KV Cache 状态：由 mla_prolog 写入、sparse_attn 读取。Prefill 整段写入，Decode 单槽环形写入。', steps: [['paged KV write', 'kv_cache[:, start_pos % 128]', 'io:state'], ['compressed cache read', 'CSA/HCA 压缩块', 'io:state']] },
    router: { description: 'DeepSeekMoE 路由器：计算 256 个专家分数，选择每 token 的 6 个 routed experts。', steps: [['router linear', 'Hidden 4096 → 256 scores', 'sem:linear'], ['sqrtsoftplus', 'sqrt(softplus(score))', 'sem:act'], ['add correction bias', 'Non-hash layers · FP32', 'sem:gate'], ['topk / hash lookup', '6 experts/token', 'sem:gate'], ['gather + normalize', 'Route weights × scale 1.5', 'sem:norm']] },
    'expert-select': { description: '按照 token-to-expert 路由结果组织输入；分布式 Expert Parallel 时承担跨 rank dispatch。', steps: [['flatten route map', 'Token × top-k indices', 'sem:act'], ['bincount experts', 'Load count · 256 experts', 'sem:gate'], ['where matching ids', 'Per-expert token index', 'sem:comm'], ['gather expert inputs', 'Routed token tiles', 'io:activation'], ['dispatch / all_reduce', 'EP communication · optional', 'sem:comm']] },
    'shared-expert': { description: '每个 token 必经的共享 SwiGLU Expert，与 routed experts 并行计算。', steps: [['w1 linear', 'Hidden 4096 → intermediate 2048', 'sem:linear'], ['w3 linear', 'Up projection · FP4/FP8', 'sem:linear'], ['clamp + SiLU', 'swiglu_limit 10', 'sem:act'], ['gate × up', 'SwiGLU elementwise', 'sem:mlp'], ['w2 linear', 'Intermediate → hidden 4096', 'sem:linear']] },
    'routed-experts': { description: '激活专家的 grouped execution；逻辑上每 token 选择 6/256 个专家，实际实现可进一步融合为 Grouped GEMM。', steps: [['select local experts', 'TP/EP shard filter', 'sem:comm'], ['gather routed tokens', 'Variable tokens per expert', 'io:activation'], ['w1 + w3 grouped GEMM', 'FP4 expert weights', 'sem:linear'], ['clamp + SiLU × up', 'SwiGLU activation', 'sem:mlp'], ['w2 grouped GEMM', 'Expert output projection', 'sem:linear'], ['scatter weighted outputs', 'Route weight × token combine', 'sem:comm']] },
    'expert-swiglu': { description: '共享与 routed Expert 共用的细粒度 SwiGLU 计算链。', steps: [['load gate / up', 'FP32 accumulation', 'io:input'], ['clamp activation', 'Limit 10.0', 'sem:act'], ['SiLU gate', 'Sigmoid × gate', 'sem:act'], ['multiply up', 'SwiGLU product', 'sem:mlp'], ['apply route weight', 'Optional expert weight', 'sem:gate'], ['down projection', 'Intermediate 2048 → hidden 4096', 'sem:linear']] },
    combine: { description: '合并 6 个 routed expert 输出与 1 个 shared expert 输出，恢复为层级 hidden state。', steps: [['weighted routed sum', 'Top-k route weights', 'sem:gate'], ['shared path add', 'Always-on shared expert', 'sem:act'], ['TP all_reduce', 'Distributed expert output', 'sem:comm'], ['cast to input dtype', 'FP32 → BF16/FP8', 'sem:linear'], ['reshape hidden', '[batch, seq, hc, 4096]', 'io:output']] },
    mhc: { description: 'mHC 残差连接：通过 Sinkhorn 约束的 pre/post/comb 权重混合多份 hidden state。', steps: [['flatten HC copies', 'hc_mult × hidden', 'sem:act'], ['rsqrt + F.linear', 'Mixing logits · FP32', 'sem:linear'], ['hc_split_sinkhorn', '20 Sinkhorn iterations', 'sem:norm'], ['pre weighted sum', 'HC → single hidden', 'sem:act'], ['post + residual combine', 'Single hidden → HC copies', 'sem:act'], ['cast output', 'Next-layer hidden state', 'io:output']] },
    'final-norm': { description: '最终 Decoder 输出的 RMSNorm，为词表投影准备归一化 hidden。', steps: [['cast final hidden', 'Input precision normalize', 'sem:linear'], ['square + mean', 'Last-dim reduction', 'sem:act'], ['rsqrt + eps', 'Final RMS reciprocal', 'sem:norm'], ['multiply final gamma', 'Normalized hidden 4096', 'sem:norm']] },
    'lm-head': { description: 'mHC Head 与词表投影；多卡场景下对词表分片 logits 做 AllGather。', steps: [['hc_head mix', 'Sigmoid mixing weights', 'sem:norm'], ['final RMSNorm', 'Head input normalization', 'sem:norm'], ['vocab linear', 'Hidden 4096 → 129280', 'sem:head'], ['all_gather logits', 'Vocab shard combine · optional', 'sem:comm']] },
  };

  graph.nodes.forEach((node) => {
    if (drillSpecs[node.id]) node.collapsed = true;
  });

  let controller = null;
  let initialized = false;
  let activePhase = 'all';
  let configLoaded = false;
  let activeDrill = null;
  let currentGraph = graph;

  function qs(selector) {
    return document.querySelector(selector);
  }

  function setChrome() {
    const card = qs('[data-model-id="' + MODEL_ID + '"]');
    document.querySelectorAll('[data-model-id]').forEach((item) => {
      const active = item === card;
      item.classList.toggle('is-active', active);
      item.setAttribute('aria-current', String(active));
      item.setAttribute('aria-selected', String(active));
      const status = item.querySelector('em');
      if (status) status.textContent = active ? '已加载' : '可视化';
    });
    const selectorTitle = qs('[data-model-selector-title]');
    const selectorSubtitle = qs('[data-model-selector-subtitle]');
    if (selectorTitle) selectorTitle.textContent = card?.querySelector('b')?.textContent || 'DeepSeek V4 Flash';
    if (selectorSubtitle) selectorSubtitle.textContent = card?.querySelector('small')?.textContent || 'MoE · CSA/HCA · 官方结构';
    document.querySelectorAll('[data-model-selector-icon]').forEach((icon) => { icon.hidden = icon.dataset.modelSelectorIcon !== MODEL_ID; });
    const factsBody = qs('#modelFactsBody');
    if (factsBody) factsBody.innerHTML = facts.map((row) => '<div><dt>' + row[0] + '</dt><dd>' + row[1] + '</dd></div>').join('');
    const title = qs('.kf-model-toolbar h1');
    if (title) title.textContent = phaseMeta.all[0];
    const status = qs('#modelCanvasStatus');
    if (status) status.textContent = 'DeepSeek V4 Flash 官方结构已加载';
    const command = qs('.kf-command');
    if (command) command.textContent = 'MODEL · DeepSeek V4 Flash 架构可视化';
    const inspectorTitle = qs('#modelInspectorTitle');
    if (inspectorTitle) inspectorTitle.textContent = 'DeepSeek V4 Flash';
  }

  async function loadOfficialConfig() {
    try {
      const response = await fetch(CONFIG_URL, { cache: 'no-store' });
      if (!response.ok) throw new Error('HTTP ' + response.status);
      const config = await response.json();
      if (window.PtoModelArchitectureState?.active !== MODEL_ID) return;
      const layer = graph.nodes.find((node) => node.id === 'layer-input');
      const attention = graph.nodes.find((node) => node.id === 'qkv-proj');
      const sparse = graph.nodes.find((node) => node.id === 'sparse-attn');
      const outProj = graph.nodes.find((node) => node.id === 'out-proj');
      const cache = graph.nodes.find((node) => node.id === 'rope-cache');
      const router = graph.nodes.find((node) => node.id === 'router');
      const experts = graph.nodes.find((node) => node.id === 'routed-experts');
      if (layer) layer.typeLabel = 'BF16 · [batch, seq, ' + config.dim + ']';
      if (attention) attention.typeLabel = 'Q/KV 投影 + RMSNorm + RoPE + Cache · ' + config.n_heads + ' Q / 1 KV';
      if (sparse) sparse.typeLabel = '稀疏注意力 · top-k ' + config.index_topk + ' KV block';
      if (outProj) outProj.typeLabel = 'wo_a 分组 einsum → wo_b · ' + config.o_groups + ' 组 × rank ' + config.o_lora_rank;
      if (cache) cache.typeLabel = 'Window ' + config.window_size + ' · paged / ring buffer';
      if (router) router.typeLabel = config.score_func + ' · top-k = ' + config.n_activated_experts + ' · scale ' + config.route_scale;
      if (experts) experts.typeLabel = config.n_activated_experts + ' active experts / token · ' + String(config.expert_dtype || config.dtype).toUpperCase() + ' experts';
      facts[0][1] = config.n_layers + ' 层';
      facts[1][1] = '284B / 13B';
      facts[2][1] = String(config.dim);
      facts[3][1] = config.n_heads + ' Q / 1 KV';
      facts[4][1] = config.n_routed_experts + ' routed + ' + config.n_shared_experts + ' shared';
      facts[5][1] = config.n_activated_experts + ' experts/token';
      facts[6][1] = '1,048,576';
      facts[7][1] = String(config.expert_dtype || config.dtype).toUpperCase() + ' expert / ' + String(config.dtype).toUpperCase() + ' mixed';
      configLoaded = true;
      setChrome();
      render();
    } catch (error) {
      const status = qs('#modelCanvasStatus');
      if (status) status.textContent = 'DeepSeek V4 Flash 结构已加载 · 配置文件读取失败，使用内置摘要';
      console.warn('DeepSeek V4 Flash config unavailable:', error);
    }
  }


  /* ---------------- 融合方案叠加 ----------------
   * 这张图是模块级的：L2 节点本身就已经是「一个融合算子该长的样子」。
   * 所以只有跨模块方案（一个 torch_npu API 吃掉多个 L2 节点）在图上才有真实
   * 拓扑变化；模块内方案在这张图上永远是同一个节点，强行画 before/after
   * 等于把模块展开再合回去，是循环论证。模块内的关系变化交给面板的代码对照。
   */

  let fusionPlan = null;     // { id, kind: 'cross'|'intra', label, sub, nodes: [] }
  let fusionMode = 'before'; // 'before' | 'after'，只对 cross 生效

  const STEP_GAP = 66;
  const FUSED_ID = '__fusion__';

  const planIsCross = () => !!fusionPlan && fusionPlan.kind === 'cross' && (fusionPlan.nodes || []).length > 1;

  const overlapsBox = (a, b) => (
    Math.min(a.x + a.width / 2, b.x + b.width / 2) - Math.max(a.x - a.width / 2, b.x - b.width / 2) > 0
    && Math.min(a.y + a.height / 2, b.y + b.height / 2) - Math.max(a.y - a.height / 2, b.y - b.height / 2) > 0
  );

  /** 把一个跨模块方案覆盖的多个 L2 节点合并成一个融合算子节点，边重新接 */
  function mergeOnePlan(base, plan, fusedId) {
    const ids = new Set((plan.nodes || []).filter((id) => base.nodes.some((n) => n.id === id)));
    if (ids.size < 2) return base;
    const merged = base.nodes.filter((n) => ids.has(n.id));
    const avg = (key) => merged.reduce((s, n) => s + n[key], 0) / merged.length;
    const fused = {
      id: fusedId,
      label: plan.label || '融合算子',
      typeLabel: plan.sub || (merged.length + ' 个模块合并为 1 个'),
      kind: 'op',
      x: avg('x'),
      y: avg('y'),
      width: 400,
      height: 70,
      colorKey: 'sem:gate',
      phase: merged[0].phase,
      parent: merged[0].parent,
      fusionNode: true,
    };
    const kept = base.nodes.filter((n) => !ids.has(n.id));
    // 融合节点落在被合并节点的几何中心，但那里未必空着（例如 MoE 合并后会压到
    // 没被合并的 shared_expert 上）。撞上就往下让，直到腾出空位。
    for (let guard = 0; guard < 40; guard += 1) {
      const clash = kept.find((n) => overlapsBox(fused, n));
      if (!clash) break;
      fused.y = clash.y + clash.height / 2 + fused.height / 2 + 16;
    }
    const seen = new Set();
    const edges = [];
    base.edges.forEach((edge) => {
      const source = ids.has(edge.source) ? fusedId : edge.source;
      const target = ids.has(edge.target) ? fusedId : edge.target;
      if (source === target) return;            // 模块之间的内部依赖被融合吃掉
      const key = source + '>' + target;
      if (seen.has(key)) return;                // 合并后重复的外部边去重
      seen.add(key);
      edges.push({ ...edge, source, target, originalKey: edge.source + '->' + edge.target });
    });
    return { ...base, nodes: kept.concat(fused), edges };
  }

  /** 依次叠加多个方案（跨模块方案彼此不相交，顺序应用即可） */
  function mergePlans(base, plans) {
    return (plans || []).reduce((g, plan, i) => mergeOnePlan(g, plan, '__fusion_' + i + '__'), base);
  }

  function buildMergedGraph(base) {
    if (!planIsCross() || fusionMode !== 'after') return base;
    return mergeOnePlan(base, fusionPlan, FUSED_ID);
  }

  function buildDrillDefinition(nodeId) {
    const spec = drillSpecs[nodeId];
    const target = graph.nodes.find((node) => node.id === nodeId);
    if (!spec || !target) return null;
    const stepGap = STEP_GAP;
    const height = 74 + spec.steps.length * stepGap;
    const top = target.y - target.height / 2;
    const nodes = spec.steps.map((step, index) => ({
      id: nodeId + '__' + index,
      label: step[0],
      typeLabel: step[1],
      kind: step[2].startsWith('io:') ? 'tensor' : 'op',
      x: target.x,
      y: top + 58 + index * stepGap,
      width: index === 0 || index === spec.steps.length - 1 ? 300 : 340,
      height: 50,
      colorKey: step[2],
      parent: 'drill-' + nodeId,
      phase: target.phase,
      drillOwner: nodeId,
    }));
    const base = {
      target,
      spec,
      height,
      delta: height - target.height + 34,
      cluster: { id: 'drill-' + nodeId, label: target.label + ' · L3 operators', x: target.x - 230, y: top, width: 460, height, parent: target.parent, drillOwner: nodeId },
      nodes,
      entry: nodes[0].id,
      exit: nodes[nodes.length - 1].id,
      edges: nodes.slice(0, -1).map((node, index) => ({ source: node.id, target: nodes[index + 1].id })),
    };
    return base;
  }

  // The default graph renderer routes edges by their bounding boxes. That is
  // fine for a simple spine, but the V4 graph has residual and auxiliary
  // paths that jump over several modules. Give those paths dedicated lanes so
  // dashed control/residual edges remain visually separate from solid data
  // edges.
  const routeHints = {
    'pre-norm->compressor': { side: 'right', lane: 1010 },
    'qkv-proj->indexer': { axis: 'vertical', sourceSide: 'bottom', targetSide: 'top', sourceDx: -110, targetDx: 0 },
    'qkv-proj->rope-cache': { side: 'right', lane: 1015 },
    'indexer->sparse-attn': { axis: 'vertical', sourceSide: 'bottom', targetSide: 'top', sourceDx: 0, targetDx: -100 },
    'compressor->sparse-attn': { axis: 'vertical', sourceSide: 'bottom', targetSide: 'top', sourceDx: 0, targetDx: 100 },
    'rope-cache->sparse-attn': { axis: 'horizontal' },
    'router->shared-expert': { side: 'left', lane: 165 },
    'router->combine': { side: 'right', lane: 1190 },
    'layer-input->mhc': { side: 'left', lane: 105 },
    'layer-output->pre-norm': { axis: 'outer-top', sourceSide: 'left', targetSide: 'top', lane: 52, targetDx: -72, turnY: 360 },
  };

  function routeGraph(graphData) {
    const nodeById = new Map(graphData.nodes.map((node) => [node.id, node]));
    const outgoing = new Map();
    const incoming = new Map();
    graphData.edges.forEach((edge) => {
      if (!outgoing.has(edge.source)) outgoing.set(edge.source, []);
      if (!incoming.has(edge.target)) incoming.set(edge.target, []);
      outgoing.get(edge.source).push(edge);
      incoming.get(edge.target).push(edge);
    });
    outgoing.forEach((edges) => edges.sort((a, b) => (nodeById.get(a.target)?.x || 0) - (nodeById.get(b.target)?.x || 0)));
    incoming.forEach((edges) => edges.sort((a, b) => (nodeById.get(a.source)?.x || 0) - (nodeById.get(b.source)?.x || 0)));

    function portOffset(edge, collection, node, axis) {
      const list = collection.get(axis === 'source' ? edge.source : edge.target) || [];
      if (list.length < 2) return 0;
      const index = list.indexOf(edge);
      const raw = (index - (list.length - 1) / 2) * 24;
      return Math.max(-(node.width / 2 - 24), Math.min(node.width / 2 - 24, raw));
    }

    const edges = graphData.edges.map((edge) => {
      const source = nodeById.get(edge.source);
      const target = nodeById.get(edge.target);
      if (!source || !target) return { ...edge };
      const key = edge.routeKey || edge.originalKey || `${edge.source}->${edge.target}`;
      const hint = routeHints[key];
      const routed = { ...edge, waypoints: undefined, curve: undefined, route: 'rounded', cornerRadius: 14 };
      const dx = target.x - source.x;
      const dy = target.y - source.y;

      if (hint) {
        const sourceSide = hint.sourceSide || hint.side;
        const targetSide = hint.targetSide || hint.side;
        const sourceDx = Number(hint.sourceDx) || 0;
        const targetDx = Number(hint.targetDx) || 0;
        routed.sourceAnchor = hint.sourceSide ? { side: sourceSide, dx: sourceDx } : { side: sourceSide, dy: 0 };
        routed.targetAnchor = hint.targetSide ? { side: targetSide, dx: targetDx } : { side: targetSide, dy: 0 };
        if (hint.axis === 'outer-top') {
          const startY = source.y;
          const endX = target.x + targetDx;
          const endY = target.y - target.height / 2;
          const turnY = Number.isFinite(Number(hint.turnY)) ? Number(hint.turnY) : endY - 32;
          routed.waypoints = [
            { x: hint.lane, y: startY },
            { x: hint.lane, y: turnY },
            { x: endX, y: turnY },
          ];
          routed.routeClass = 'outer-return-lane';
        } else if (hint.axis === 'vertical') {
          const startY = source.y + (sourceSide === 'bottom' ? source.height / 2 : sourceSide === 'top' ? -source.height / 2 : 0);
          const endY = target.y + (targetSide === 'top' ? -target.height / 2 : targetSide === 'bottom' ? target.height / 2 : 0);
          const startX = source.x + sourceDx;
          const endX = target.x + targetDx;
          const midY = (startY + endY) / 2;
          routed.waypoints = [{ x: startX, y: midY }, { x: endX, y: midY }];
          routed.routeClass = 'vertical-branch-lane';
        } else {
          routed.waypoints = [{ x: hint.lane, y: source.y }, { x: hint.lane, y: target.y }];
          routed.routeClass = 'side-lane';
        }
        return routed;
      }

      // Keep reverse and very long links on the outer sides of the graph.
      if (dy < -40 || Math.abs(dy) > 360) {
        const useLeft = (source.x + target.x) / 2 < graphData.width / 2;
        const lane = useLeft ? 185 : graphData.width - 105;
        const side = useLeft ? 'left' : 'right';
        routed.sourceAnchor = side;
        routed.targetAnchor = side;
        routed.waypoints = [{ x: lane, y: source.y }, { x: lane, y: target.y }];
        routed.routeClass = 'side-lane';
        return routed;
      }

      // Branches that are almost level should leave from the side of a node;
      // this prevents a horizontal path from cutting through the vertical
      // spine or an adjacent dashed branch.
      if (Math.abs(dy) <= 54 && Math.abs(dx) > 40) {
        const sourceSide = dx > 0 ? 'right' : 'left';
        const targetSide = dx > 0 ? 'left' : 'right';
        routed.sourceAnchor = sourceSide;
        routed.targetAnchor = targetSide;
        if (Math.abs(dy) > 4) {
          const midX = (source.x + target.x) / 2;
          routed.waypoints = [{ x: midX, y: source.y }, { x: midX, y: target.y }];
        } else {
          routed.curve = 'straight';
          routed.route = undefined;
        }
        routed.routeClass = 'horizontal-lane';
        return routed;
      }

      const sourceDx = portOffset(edge, outgoing, source, 'source');
      const targetDx = portOffset(edge, incoming, target, 'target');
      routed.sourceAnchor = { side: dy >= 0 ? 'bottom' : 'top', dx: sourceDx };
      routed.targetAnchor = { side: dy >= 0 ? 'top' : 'bottom', dx: targetDx };
      const startY = source.y + (dy >= 0 ? source.height / 2 : -source.height / 2);
      const endY = target.y + (dy >= 0 ? -target.height / 2 : target.height / 2);
      const midY = (startY + endY) / 2;
      routed.waypoints = [
        { x: source.x + sourceDx, y: midY },
        { x: target.x + targetDx, y: midY },
      ];
      routed.routeClass = Math.abs(dx) < 32 ? 'spine-lane' : 'branch-lane';
      return routed;
    });
    return { ...graphData, edges };
  }

  function buildExpandedGraph(nodeId) {
    const drill = buildDrillDefinition(nodeId);
    if (!drill) return {
      ...graph,
      nodes: graph.nodes.map((node) => ({ ...node })),
      edges: graph.edges.map((edge) => ({ ...edge })),
      clusters: graph.clusters.map((cluster) => ({ ...cluster })),
    };
    const threshold = drill.target.y;
    const shouldShift = (node) => node.y > threshold || (node.y === threshold && Math.abs(node.x - drill.target.x) < (drill.cluster.width + node.width) / 2);
    const nodes = graph.nodes.filter((node) => node.id !== nodeId).map((node) => ({ ...node, y: shouldShift(node) ? node.y + drill.delta : node.y }));
    const clusters = graph.clusters.map((cluster) => {
      const containsTarget = cluster.y <= threshold && cluster.y + cluster.height >= threshold;
      return {
        ...cluster,
        y: cluster.y > threshold ? cluster.y + drill.delta : cluster.y,
        height: containsTarget ? cluster.height + drill.delta : cluster.height,
      };
    });
    clusters.push(drill.cluster);
    const edges = graph.edges.map((edge) => ({
      ...edge,
      routeKey: `${edge.source}->${edge.target}`,
      source: edge.source === nodeId ? drill.exit : edge.source,
      target: edge.target === nodeId ? drill.entry : edge.target,
    }));
    return {
      ...graph,
      height: graph.height + drill.delta,
      nodes: [...nodes, ...drill.nodes],
      clusters,
      edges: [...edges, ...drill.edges],
      activeDrill: nodeId,
    };
  }

  function focusDrillViewport(nodeId) {
    const drill = buildDrillDefinition(nodeId);
    if (!controller || !drill) return;
    const stage = qs('#' + stageId);
    const rect = stage.getBoundingClientRect();
    const padding = 34;
    const zoom = Math.max(.18, Math.min(1.05, (rect.width - padding * 2) / drill.cluster.width, (rect.height - padding * 2) / drill.cluster.height));
    controller.setTransform({
      zoom,
      tx: (rect.width - drill.cluster.width * zoom) / 2 - drill.cluster.x * zoom,
      ty: (rect.height - drill.cluster.height * zoom) / 2 - drill.cluster.y * zoom,
    });
    const readout = qs('#modelZoomReadout');
    if (readout) readout.textContent = Math.round(zoom * 100) + '%';
  }

  function applyPhase(phase) {
    activePhase = phase;
    const meta = phaseMeta[phase] || phaseMeta.all;
    const title = qs('.kf-model-toolbar h1');
    const summary = qs('#modelPhaseSummary');
    if (title) title.textContent = meta[0];
    if (summary) summary.textContent = meta[1];
    document.querySelectorAll('[data-model-phase]').forEach((button, index) => {
      const key = phaseOrder[index] || 'all';
      const item = phaseMeta[key];
      const b = button.querySelector('b');
      const small = button.querySelector('small');
      button.classList.toggle('is-active', key === phase);
      if (b && item) b.textContent = item[2];
      if (small && item) small.textContent = item[1];
    });
    document.querySelectorAll('#' + stageId + ' .pto-model-graphviz-node').forEach((element) => {
      const node = currentGraph.nodes.find((item) => item.id === element.dataset.nodeId);
      const visible = phase === 'all' || node?.phase === phase || (phase === 'layer' && node?.phase === 'layer');
      element.classList.toggle('is-phase-muted', !visible);
      element.classList.toggle('is-phase-active', phase !== 'all' && visible);
    });
    document.querySelectorAll('#' + stageId + ' .pto-model-graphviz-edge, #' + stageId + ' .pto-model-graphviz-edge-tag').forEach((element) => {
      const source = currentGraph.nodes.find((node) => node.id === element.dataset.source);
      const target = currentGraph.nodes.find((node) => node.id === element.dataset.target);
      const visible = phase === 'all' || source?.phase === phase || target?.phase === phase || (phase === 'layer' && (source?.phase === 'layer' || target?.phase === 'layer'));
      element.classList.toggle('is-phase-muted', !visible);
      element.classList.toggle('is-phase-active', phase !== 'all' && visible);
    });
    const body = qs('#modelInspectorBody');
    if (body) body.innerHTML = '<div class="kf-model-inspector__hero"><span>' + meta[2] + '</span><b>' + meta[0] + '</b><p>' + meta[3] + '</p></div><dl class="kf-model-node-detail"><div><dt>来源</dt><dd>Data/DeepSeek-V4-Flash-Official</dd></div><div><dt>结构文件</dt><dd>model.py · inference-config.json</dd></div></dl>';
  }

  function selectNode(nodeId) {
    const node = currentGraph.nodes.find((item) => item.id === nodeId);
    if (!node) return;
    const body = qs('#modelInspectorBody');
    if (body) {
      const owner = node.drillOwner ? graph.nodes.find((item) => item.id === node.drillOwner) : null;
      const source = node.drillOwner ? 'model.py · ' + (owner?.label || node.drillOwner) : 'model.py / inference-config.json';
      body.innerHTML = '<div class="kf-model-inspector__hero"><span>' + (node.drillOwner ? 'L3 OPERATOR' : 'OFFICIAL STRUCTURE') + '</span><b>' + node.label + '</b><p>' + node.typeLabel + '</p></div><dl class="kf-model-node-detail"><div><dt>阶段</dt><dd>' + (phaseMeta[node.phase]?.[2] || node.phase) + '</dd></div><div><dt>父模块</dt><dd>' + (owner?.label || 'DeepSeek-V4 Flash') + '</dd></div><div><dt>来源</dt><dd>' + source + '</dd></div></dl>';
    }
    document.querySelectorAll('#' + stageId + ' .pto-model-graphviz-node').forEach((element) => element.classList.toggle('is-model-selected', element.dataset.nodeId === nodeId));
  }

  function isPristineTransform(transform) {
    return !transform || (
      Math.abs(Number(transform.tx) || 0) < 0.01
      && Math.abs(Number(transform.ty) || 0) < 0.01
      && Math.abs((Number(transform.zoom) || 1) - 1) < 0.001
    );
  }

  function render() {
    const stage = qs('#' + stageId);
    const pattern = window.PtoModelGraphvizPattern;
    if (!stage || !pattern) return;
    const savedTransform = controller?.getTransform?.();
    // createController schedules its first fit in requestAnimationFrame. A
    // second synchronous render can therefore observe the temporary default
    // transform (0, 0, 1); do not mistake that pristine value for a user's
    // viewport and accidentally disable the initial fit.
    const preserveTransform = savedTransform && !isPristineTransform(savedTransform);
    controller?.destroy?.();
    currentGraph = routeGraph(buildMergedGraph(buildExpandedGraph(activeDrill)));
    controller = pattern.renderController(stage, currentGraph, {
      ariaLabel: 'DeepSeek V4 Flash official model architecture',
      colormap: pattern.modelArchitectureColormap(currentGraph),
      fitMode: 'full',
      viewportPadding: 36,
      autoFit: !preserveTransform,
      interaction: { panZoom: true, selectableClusters: false },
      overlays: { edgeTags: true },
      onSelect: ({ nodeId }) => selectNode(nodeId),
    });
    // renderController 同步建好 DOM，融合叠加就在这里补类：不能只放进下面的 rAF。
    // 配置文件加载完成会再触发一次 render，页面在后台时 rAF 被暂停，
    // 那次重建就会把融合视图静默清掉。
    applyFusionClasses();
    requestAnimationFrame(() => {
      applyPhase(activePhase);
      applyFusionClasses();
      if (preserveTransform) {
        controller.setTransform({ ...savedTransform });
        const readout = qs('#modelZoomReadout');
        if (readout) readout.textContent = Math.round(savedTransform.zoom * 100) + '%';
        return;
      }
      const fittedTransform = controller.getTransform?.();
      const readout = qs('#modelZoomReadout');
      if (readout && fittedTransform) readout.textContent = Math.round(fittedTransform.zoom * 100) + '%';
      if (activeDrill) {
        const drill = buildDrillDefinition(activeDrill);
        const owner = graph.nodes.find((node) => node.id === activeDrill);
        const body = qs('#modelInspectorBody');
        if (body && drill) body.innerHTML = '<div class="kf-model-inspector__hero"><span>L3 EXPANDED</span><b>' + (owner?.label || activeDrill) + '</b><p>' + drill.spec.description + '</p></div><dl class="kf-model-node-detail"><div><dt>层级</dt><dd>Module → logical operators</dd></div><div><dt>操作</dt><dd>点击容器右上角 − 收起</dd></div><div><dt>来源</dt><dd>model.py · inference-config.json</dd></div></dl>';
        focusDrillViewport(activeDrill);
      }
    });
  }

  function handleDrillToggle(event) {
    if (window.PtoModelArchitectureState?.active !== MODEL_ID) return;
    const toggle = event.target.closest('.pto-model-graphviz-toggle, .pto-model-graphviz-toggle-icon');
    if (!toggle) return;
    const nodeGroup = toggle.closest('.pto-model-graphviz-node');
    const clusterGroup = toggle.closest('.pto-model-graphviz-cluster');
    const nodeId = nodeGroup?.dataset.nodeId;
    const clusterId = clusterGroup?.dataset.clusterId;
    if (nodeId && drillSpecs[nodeId]) {
      event.preventDefault();
      event.stopImmediatePropagation();
      activeDrill = nodeId;
      activePhase = graph.nodes.find((node) => node.id === nodeId)?.phase || activePhase;
      render();
      return;
    }
    if (clusterId?.startsWith('drill-')) {
      event.preventDefault();
      event.stopImmediatePropagation();
      activeDrill = null;
      render();
    }
  }

  function handleCanvasSelectionClear(event) {
    if (window.PtoModelArchitectureState?.active !== MODEL_ID) return;
    if (event.target.closest('.pto-model-graphviz-node, .pto-model-graphviz-edge, .pto-model-graphviz-edge-tag')) return;
    document.querySelectorAll('#' + stageId + ' .pto-model-graphviz-node').forEach((element) => element.classList.remove('is-model-selected', 'is-fusion-focus'));
  }

  function selectPhase(phase) {
    if (window.PtoModelArchitectureState?.active !== MODEL_ID) return;
    activePhase = phase;
    if (activeDrill) {
      activeDrill = null;
      render();
      return;
    }
    applyPhase(phase);
  }

  function init() {
    if (initialized) return;
    const stage = qs('#' + stageId);
    if (!stage || !window.PtoModelGraphvizPattern) return;
    setChrome();
    stage.addEventListener('click', handleDrillToggle, true);
    stage.addEventListener('pointerdown', (event) => {
      if (!event.target.closest('.pto-model-graphviz-node, .pto-model-graphviz-edge, .pto-model-graphviz-edge-tag')) {
        document.querySelectorAll('#' + stageId + ' .pto-model-graphviz-node').forEach((element) => element.classList.remove('is-model-selected'));
      }
    });
    stage.addEventListener('pointerdown', handleCanvasSelectionClear, true);
    document.querySelectorAll('[data-model-phase]').forEach((button, index) => button.addEventListener('click', () => selectPhase(phaseOrder[index] || 'all')));
    document.querySelector('[data-model-fit]')?.addEventListener('click', () => {
      if (window.PtoModelArchitectureState?.active !== MODEL_ID) return;
      controller?.fit();
      const readout = qs('#modelZoomReadout');
      if (readout) readout.textContent = '适应';
    });
    document.querySelector('[data-model-zoom="in"]')?.addEventListener('click', () => {
      if (window.PtoModelArchitectureState?.active !== MODEL_ID) return;
      const current = controller?.getTransform?.();
      if (current) {
        const zoom = Math.min(2.6, current.zoom * 1.18);
        controller.setTransform({ zoom });
        const readout = qs('#modelZoomReadout');
        if (readout) readout.textContent = Math.round(zoom * 100) + '%';
      }
    });
    document.querySelector('[data-model-zoom="out"]')?.addEventListener('click', () => {
      if (window.PtoModelArchitectureState?.active !== MODEL_ID) return;
      const current = controller?.getTransform?.();
      if (current) {
        const zoom = Math.max(.18, current.zoom / 1.18);
        controller.setTransform({ zoom });
        const readout = qs('#modelZoomReadout');
        if (readout) readout.textContent = Math.round(zoom * 100) + '%';
      }
    });
    initialized = true;
    render();
  }

  // 融合推荐面板回跳用：一个候选可能横跨多个模块（例如 mHC → layer-output → pre-norm），
  // 所以除了选中首个节点，其余节点也要标出来，否则开发者看不出融合边界覆盖了哪几块。
  function highlightNodes(nodeIds) {
    const ids = new Set(nodeIds.filter((id) => graph.nodes.some((node) => node.id === id)));
    document.querySelectorAll('#' + stageId + ' .pto-model-graphviz-node').forEach((element) => {
      element.classList.toggle('is-fusion-focus', ids.has(element.dataset.nodeId));
    });
  }

  /** 渲染器不认识自定义字段，融合相关的类在渲染完成后按 nodeId 补上 */
  function applyFusionClasses() {
    const fused = new Set(currentGraph.nodes.filter((n) => n.fusionNode).map((n) => n.id));
    const scope = fusionPlan ? new Set(fusionPlan.nodes || []) : new Set();
    document.querySelectorAll('#' + stageId + ' .pto-model-graphviz-node').forEach((element) => {
      const id = element.dataset.nodeId;
      element.classList.toggle('is-fusion-node', fused.has(id));
      element.classList.toggle('is-fusion-scope', scope.has(id));
    });
    // 融合前：被方案覆盖的模块之间的边，就是这个 API 会吃掉的依赖
    document.querySelectorAll('#' + stageId + ' .pto-model-graphviz-edge').forEach((element) => {
      const inside = scope.has(element.dataset.source) && scope.has(element.dataset.target);
      element.classList.toggle('is-fusion-inner', inside);
    });
  }

  /** 把视口框到方案覆盖的那几个模块上 */
  function focusPlanViewport() {
    if (!controller || !fusionPlan) return;
    const ids = new Set(fusionPlan.nodes || []);
    const targets = currentGraph.nodes.filter((n) => ids.has(n.id) || (n.fusionNode && fusionMode === 'after'));
    if (!targets.length) return;
    const minX = Math.min(...targets.map((n) => n.x - n.width / 2));
    const maxX = Math.max(...targets.map((n) => n.x + n.width / 2));
    const minY = Math.min(...targets.map((n) => n.y - n.height / 2));
    const maxY = Math.max(...targets.map((n) => n.y + n.height / 2));
    const pad = 120;
    const boxW = maxX - minX + pad * 2;
    const boxH = maxY - minY + pad * 2;
    const stage = qs('#' + stageId);
    if (!stage) return;
    const rect = stage.getBoundingClientRect();
    const zoom = Math.max(.18, Math.min(1.1, rect.width / boxW, rect.height / boxH));
    controller.setTransform({
      zoom,
      tx: rect.width / 2 - ((minX + maxX) / 2) * zoom,
      ty: rect.height / 2 - ((minY + maxY) / 2) * zoom,
    });
    const readout = qs('#modelZoomReadout');
    if (readout) readout.textContent = Math.round(zoom * 100) + '%';
  }

  /**
   * 供推荐面板调用。模块内方案不再下钻——那只会把模块展开再合回去，
   * 图上看不出任何真实变化；它只标位置，关系变化由面板的代码对照负责。
   */
  function setFusionPlan(plan) {
    fusionPlan = plan || null;
    activeDrill = null;
    show();
    const first = fusionPlan && (fusionPlan.nodes || []).map((id) => graph.nodes.find((n) => n.id === id)).find(Boolean);
    if (first && first.phase) activePhase = first.phase;
    render();
    const apply = () => {
      applyPhase(activePhase);
      applyFusionClasses();
      if (fusionPlan) highlightNodes(fusionPlan.nodes || []);
      focusPlanViewport();
    };
    apply();
    setTimeout(apply, 0);
    return planIsCross();
  }

  function setFusionMode(mode) {
    fusionMode = mode === 'after' ? 'after' : 'before';
    if (!planIsCross()) return fusionMode;
    render();
    const apply = () => {
      applyPhase(activePhase);
      applyFusionClasses();
      focusPlanViewport();
    };
    apply();
    setTimeout(apply, 0);
    return fusionMode;
  }

  /* ---------------- 替换前后对比视图 ----------------
   * 左右两张整网图：左边替换前、右边替换后（默认叠加全部可落地方案）。
   * 三件事要联动：视口、选中、下钻。图形控制器没有 transform 回调，
   * 视口只能靠「交互后读取 + 定时兜底」；选中走 onSelect 回调互相镜像；
   * 下钻由这里持有状态、两边一起重建。全程不用 rAF——页面在后台时会被暂停。
   */

  let compare = null;

  const FUSED_PREFIX = '__fusion_';

  function comparePlans() {
    if (!compare) return [];
    return compare.scope === 'all'
      ? compare.plans
      : compare.plans.filter((p) => p.id === compare.activeId);
  }

  /** 融合节点 ↔ 它吃掉的原始模块，选中互映和连线都靠这张表 */
  function buildFusedMap(plans) {
    const map = new Map();
    plans.forEach((plan, i) => map.set(FUSED_PREFIX + i + '__', plan.nodes || []));
    return map;
  }

  /** 左图节点 → 右图对应节点（被吃掉的映射到融合节点） */
  function mapBeforeToAfter(id) {
    for (const [fusedId, originals] of compare.fusedMap) {
      if (originals.includes(id)) return fusedId;
    }
    return id;
  }

  /** 右图节点 → 左图对应节点集合（融合节点展开成它吃掉的那几个） */
  function mapAfterToBefore(id) {
    if (compare.fusedMap.has(id)) return compare.fusedMap.get(id);
    return [id];
  }

  function compareStage(side) {
    return compare.root.querySelector('[data-cmp-stage="' + side + '"]');
  }

  /** 选中互映：source 标成 sync，避免两边来回触发 */
  function syncSelection(side, nodeId) {
    if (!compare || compare.syncingSel) return;
    compare.syncingSel = true;
    try {
      if (side === 'before') {
        const mate = nodeId ? mapBeforeToAfter(nodeId) : null;
        // 点中的模块如果被合并掉了，同组的其它模块也一起标出来——
        // 它们在右图是同一个融合节点，连线也是连向同一个点
        compare.selectedBefore = mate && compare.fusedMap.has(mate)
          ? compare.fusedMap.get(mate).slice()
          : (nodeId ? [nodeId] : []);
        compare.selectedAfter = mate ? [mate] : [];
        compare.after?.selectNode?.(mate, { source: 'sync' });
      } else {
        compare.selectedAfter = nodeId ? [nodeId] : [];
        const mates = nodeId ? mapAfterToBefore(nodeId) : [];
        compare.selectedBefore = mates;
        compare.before?.selectNode?.(mates[0] || null, { source: 'sync' });
      }
    } finally {
      compare.syncingSel = false;
    }
    // 融合节点对应多个原始模块，控制器只能选中一个，其余用附加类标出来
    const beforeHost = compareStage('before');
    beforeHost.querySelectorAll('.pto-model-graphviz-node').forEach((el) => {
      el.classList.toggle('is-cmp-mate', compare.selectedBefore.includes(el.dataset.nodeId));
    });
    drawCompareLinks();
  }

  /** 两个视图之间的虚线连接：把选中的算子在左右图里连起来 */
  function drawCompareLinks() {
    if (!compare) return;
    const svg = compare.root.querySelector('[data-cmp-links]');
    if (!svg) return;
    while (svg.firstChild) svg.removeChild(svg.firstChild);
    const pairs = [];
    (compare.selectedAfter || []).forEach((afterId) => {
      mapAfterToBefore(afterId).forEach((beforeId) => pairs.push([beforeId, afterId]));
    });
    if (!pairs.length) return;

    const box = compare.root.querySelector('.kf-cmp-body').getBoundingClientRect();
    svg.setAttribute('viewBox', `0 0 ${box.width} ${box.height}`);
    const beforeHost = compareStage('before');
    const afterHost = compareStage('after');
    const beforeRect = beforeHost.getBoundingClientRect();
    const afterRect = afterHost.getBoundingClientRect();

    pairs.forEach(([beforeId, afterId]) => {
      const a = beforeHost.querySelector(`[data-node-id="${CSS.escape(beforeId)}"]`);
      const b = afterHost.querySelector(`[data-node-id="${CSS.escape(afterId)}"]`);
      if (!a || !b) return;
      const ra = a.getBoundingClientRect();
      const rb = b.getBoundingClientRect();
      // 节点可能被画布裁掉，端点夹到各自画布边界上，线不会飞出去
      const ay = Math.min(Math.max(ra.top + ra.height / 2, beforeRect.top + 4), beforeRect.bottom - 4);
      const by = Math.min(Math.max(rb.top + rb.height / 2, afterRect.top + 4), afterRect.bottom - 4);
      const ax = Math.min(ra.right, beforeRect.right);
      const bx = Math.max(rb.left, afterRect.left);
      const x1 = ax - box.left;
      const y1 = ay - box.top;
      const x2 = bx - box.left;
      const y2 = by - box.top;
      const mid = (x1 + x2) / 2;
      const path = document.createElementNS('http://www.w3.org/2000/svg', 'path');
      path.setAttribute('d', `M ${x1} ${y1} C ${mid} ${y1}, ${mid} ${y2}, ${x2} ${y2}`);
      path.setAttribute('class', 'kf-cmp-link');
      svg.appendChild(path);
      [[x1, y1], [x2, y2]].forEach(([cx, cy]) => {
        const dot = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
        dot.setAttribute('cx', cx);
        dot.setAttribute('cy', cy);
        dot.setAttribute('r', '3.5');
        dot.setAttribute('class', 'kf-cmp-linkdot');
        svg.appendChild(dot);
      });
    });
  }

  function renderCompareStage(host, graphData, ariaLabel, side) {
    const pattern = window.PtoModelGraphvizPattern;
    return pattern.renderController(host, routeGraph(graphData), {
      ariaLabel,
      colormap: pattern.modelArchitectureColormap(graphData),
      fitMode: 'full',
      viewportPadding: 28,
      autoFit: true,
      interaction: { panZoom: true, selectableClusters: false },
      overlays: { edgeTags: true },
      onSelect: ({ nodeId, source }) => {
        if (source === 'sync') return;
        syncSelection(side, nodeId);
      },
    });
  }

  function syncViewport(source, target) {
    if (!compare || compare.syncing || !source || !target) return;
    const t = source.getTransform && source.getTransform();
    if (!t) return;
    const cur = target.getTransform && target.getTransform();
    const same = cur && Math.abs(cur.tx - t.tx) < 0.5 && Math.abs(cur.ty - t.ty) < 0.5
      && Math.abs(cur.zoom - t.zoom) < 0.001;
    if (!same) {
      compare.syncing = true;
      target.setTransform({ ...t });
      compare.syncing = false;
      const readout = compare.root.querySelector('[data-cmp-zoom]');
      if (readout) readout.textContent = Math.round(t.zoom * 100) + '%';
    }
    drawCompareLinks();
  }

  function renderCompare() {
    if (!compare) return;
    const plans = comparePlans();
    compare.fusedMap = buildFusedMap(plans);
    const clone = () => ({ ...graph, nodes: graph.nodes.map((n) => ({ ...n })), edges: graph.edges.map((e) => ({ ...e })) });

    // 下钻：被合并吃掉的模块在右图里已经不存在，那一侧就不展开，只保留合并结果
    const drill = compare.drill;
    const drillMerged = !!drill && plans.some((p) => (p.nodes || []).includes(drill));
    const beforeGraph = drill ? buildExpandedGraph(drill) : clone();
    const afterGraph = mergePlans(drillMerged || !drill ? clone() : buildExpandedGraph(drill), plans);

    compare.before?.destroy?.();
    compare.after?.destroy?.();
    const beforeHost = compareStage('before');
    const afterHost = compareStage('after');
    compare.before = renderCompareStage(beforeHost, beforeGraph, '替换前整网结构', 'before');
    compare.after = renderCompareStage(afterHost, afterGraph, '替换后整网结构', 'after');

    afterHost.querySelectorAll('.pto-model-graphviz-node').forEach((el) => {
      el.classList.toggle('is-fusion-node', String(el.dataset.nodeId || '').startsWith(FUSED_PREFIX));
    });
    const scope = new Set(plans.flatMap((p) => p.nodes || []));
    beforeHost.querySelectorAll('.pto-model-graphviz-node').forEach((el) => {
      el.classList.toggle('is-fusion-scope', scope.has(el.dataset.nodeId));
    });
    beforeHost.querySelectorAll('.pto-model-graphviz-edge').forEach((el) => {
      el.classList.toggle('is-fusion-inner', scope.has(el.dataset.source) && scope.has(el.dataset.target));
    });

    const stat = (id, text) => {
      const el = compare.root.querySelector('[data-cmp-' + id + ']');
      if (el) el.textContent = text;
    };
    stat('before-count', beforeGraph.nodes.length + ' 个模块');
    stat('after-count', afterGraph.nodes.length + ' 个模块');
    const removed = beforeGraph.edges.length - afterGraph.edges.length;
    stat('summary', plans.length
      ? `${plans.length} 个方案 · 模块 ${beforeGraph.nodes.length} → ${afterGraph.nodes.length} · 依赖边减少 ${removed} 条`
      : '当前方案不改变整网拓扑（模块内融合）');
    stat('drill', drill ? `已下钻 ${drill}${drillMerged ? ' · 右图该模块已被合并，不展开' : ''}` : '');
    compare.root.querySelectorAll('[data-cmp-scope]').forEach((b) => {
      b.classList.toggle('is-active', b.dataset.cmpScope === compare.scope);
    });
    const title = compare.root.querySelector('[data-cmp-title]');
    if (title) {
      const active = compare.plans.find((p) => p.id === compare.activeId);
      title.textContent = compare.scope === 'all' ? '全部可落地方案' : (active ? active.label : '当前方案');
    }

    // 重建后把选中恢复回去
    if ((compare.selectedAfter || []).length) {
      syncSelection('after', compare.selectedAfter[0]);
    } else if ((compare.selectedBefore || []).length) {
      syncSelection('before', compare.selectedBefore[0]);
    }
    syncViewport(compare.before, compare.after);
  }

  /** 下钻按钮在两边都能点，状态由这里持有，点完两个图一起重建 */
  function handleCompareDrill(event, side) {
    const toggle = event.target.closest('.pto-model-graphviz-toggle, .pto-model-graphviz-toggle-icon');
    if (!toggle) return;
    const nodeId = toggle.closest('.pto-model-graphviz-node')?.dataset.nodeId;
    const clusterId = toggle.closest('.pto-model-graphviz-cluster')?.dataset.clusterId;
    if (nodeId && drillSpecs[nodeId]) {
      event.preventDefault();
      event.stopPropagation();
      compare.drill = nodeId;
      renderCompare();
      return;
    }
    if (clusterId && clusterId.startsWith('drill-')) {
      event.preventDefault();
      event.stopPropagation();
      compare.drill = null;
      renderCompare();
    }
  }

  function openCompare(options) {
    const host = document.getElementById('modelArchitectureView');
    if (!host || !window.PtoModelGraphvizPattern) return false;
    const opts = options || {};
    if (!compare) {
      const root = document.createElement('div');
      root.className = 'kf-cmp-root';
      root.id = 'modelCompare';
      root.innerHTML = `
        <header class="kf-cmp-head">
          <div class="kf-cmp-title">
            <span class="kf-eyebrow">替换前后对比</span>
            <b data-cmp-title></b>
            <small data-cmp-summary></small>
          </div>
          <div class="kf-cmp-scope" role="group" aria-label="对比范围">
            <button type="button" data-cmp-scope="all">全部方案</button>
            <button type="button" data-cmp-scope="plan">单个方案</button>
          </div>
          <div class="kf-cmp-actions">
            <small data-cmp-drill></small>
            <span data-cmp-zoom>100%</span>
            <button type="button" data-cmp-fit>适应画布</button>
            <button type="button" data-cmp-close aria-label="关闭对比视图">✕</button>
          </div>
        </header>
        <div class="kf-cmp-body">
          <section><h4>替换前<em data-cmp-before-count></em></h4><div class="kf-cmp-stage pto-model-graphviz-stage" data-cmp-stage="before"></div></section>
          <section><h4>替换后<em data-cmp-after-count></em></h4><div class="kf-cmp-stage pto-model-graphviz-stage" data-cmp-stage="after"></div></section>
          <svg class="kf-cmp-links" data-cmp-links aria-hidden="true"></svg>
        </div>`;
      host.appendChild(root);
      compare = {
        root, before: null, after: null, plans: [], activeId: null, scope: 'all',
        drill: null, selectedBefore: [], selectedAfter: [], fusedMap: new Map(),
        timer: null, syncing: false, syncingSel: false,
      };

      root.addEventListener('click', (event) => {
        if (event.target.closest('[data-cmp-close]')) { closeCompare(); return; }
        if (event.target.closest('[data-cmp-fit]')) {
          compare.before?.fit();
          syncViewport(compare.before, compare.after);
          return;
        }
        const sc = event.target.closest('[data-cmp-scope]');
        if (sc) { compare.scope = sc.dataset.cmpScope; renderCompare(); }
      });

      // 下钻要抢在图形库自己的点击处理之前
      compareStage('before').addEventListener('click', (e) => handleCompareDrill(e, 'before'), true);
      compareStage('after').addEventListener('click', (e) => handleCompareDrill(e, 'after'), true);

      // 视口：这些监听器建在控制器之前，同一次派发里排在图形库前面，
      // 直接读 transform 会读到旧值，用 setTimeout 挪到派发之后
      const later = (from, to) => () => setTimeout(() => syncViewport(from(), to()), 0);
      ['wheel', 'pointermove', 'pointerup', 'dblclick'].forEach((type) => {
        compareStage('before').addEventListener(type, later(() => compare.before, () => compare.after), { passive: true });
        compareStage('after').addEventListener(type, later(() => compare.after, () => compare.before), { passive: true });
      });
    }
    compare.plans = opts.plans || [];
    compare.activeId = opts.activeId || (compare.plans[0] && compare.plans[0].id) || null;
    compare.scope = opts.scope || 'all';
    compare.drill = null;
    compare.selectedBefore = [];
    compare.selectedAfter = [];
    compare.root.hidden = false;
    host.classList.add('is-comparing');
    renderCompare();
    clearInterval(compare.timer);
    compare.timer = setInterval(() => syncViewport(compare.before, compare.after), 140);
    return true;
  }

  function closeCompare() {
    if (!compare) return;
    clearInterval(compare.timer);
    compare.timer = null;
    compare.before?.destroy?.();
    compare.after?.destroy?.();
    compare.before = null;
    compare.after = null;
    compare.root.hidden = true;
    document.getElementById('modelArchitectureView')?.classList.remove('is-comparing');
    window.PtoFusionAdvisor?.onCompareClosed?.();
  }

  function clearFusionPlan() {
    if (!fusionPlan) return;
    fusionPlan = null;
    fusionMode = 'before';
    activeDrill = null;
    render();
    setTimeout(() => { applyPhase(activePhase); applyFusionClasses(); clearFusionHighlight(); }, 0);
  }

  function clearFusionHighlight() {
    document.querySelectorAll('#' + stageId + ' .is-fusion-focus').forEach((element) => element.classList.remove('is-fusion-focus'));
  }

  function focusNodes(nodeIds) {
    const ids = (Array.isArray(nodeIds) ? nodeIds : [nodeIds]).filter(Boolean);
    const target = graph.nodes.find((node) => ids.includes(node.id));
    if (!target) return false;
    // 有下钻展开时先收起，否则融合边界里的节点可能被 L3 展开替换掉
    if (activeDrill) activeDrill = null;
    show();
    const apply = () => {
      applyPhase(target.phase || activePhase);
      controller?.selectNode?.(target.id, { source: 'fusion' });
      selectNode(target.id);
      highlightNodes(ids);
    };
    // render() 是同步重建 DOM 的，这里直接标记即可；再补一次延后执行，
    // 是为了盖过 render 自己排在下一帧的 applyPhase。不依赖 rAF：
    // 页面在后台时 rAF 会被暂停，定位就会静默失败。
    apply();
    setTimeout(apply, 0);
    return true;
  }

  function show() {
    window.PtoModelArchitectureState = { active: MODEL_ID };
    const wasInitialized = initialized;
    init();
    setChrome();
    // init() already rendered the first view. Avoid an immediate second
    // render before the renderer's first auto-fit frame has run.
    if (wasInitialized) render();
    if (!configLoaded) loadOfficialConfig();
    // active 在函数开头就已切到本模型，入口卡可以直接同步，不必等下一帧
    window.PtoFusionAdvisor?.syncEntry?.();
    requestAnimationFrame(() => applyPhase(activePhase));
  }

  window.PtoDeepSeekV4ModelViz = {
    show,
    fit: () => controller?.fit(),
    setPhase: (phase) => applyPhase(phase),
    focusNode: (nodeId) => focusNodes([nodeId]),
    focusNodes,
    clearFusionHighlight,
    setFusionPlan,
    setFusionMode,
    clearFusionPlan,
    openCompare,
    closeCompare,
    isComparing: () => !!compare && !compare.root.hidden,
    getFusionMode: () => fusionMode,
    graph,
    drillSpecs,
  };
})();
