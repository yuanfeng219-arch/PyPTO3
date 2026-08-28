/**
 * 融合算子推荐 · torch_npu 替换方案数据层
 *
 * 口径来自 skills/cannbot-skills-master-model-model-infer-fusion（model-infer-fusion）：
 *   第一步 拆解模块（含 Prefill/Decode 分支差异）· 第二步 匹配仓库参考链路
 *   第三步 查算子文档确认可用性 · 第四步 分析阶段审查 · 第五步 逐模块替换验证
 *
 * 与前一版的根本差别：推荐单元不再是「子图融合候选 + FusionScore」，
 * 而是「模块 → 候选 torch_npu API → 适配状态」。这个 skill 不定义打分公式，
 * 所以这里也不造一个——排序按「能不能直接开工」和覆盖范围，不编分数。
 *
 * 分两类表达（方向 C）：
 *   cross —— 一个 API 吃掉图上多个 L2 模块，整网结构图上是真实可见的合并
 *   intra —— 融合发生在模块内部，整网图上没有拓扑变化，只标位置，
 *            关系变化用「当前 PyTorch 调用 → 替换后 torch_npu 调用」代码对照讲
 *
 * 所有 before[] 的行号都对着 Data/DeepSeek-V4-Flash-Official/model.py，可复核。
 */
(function registerFusionRuleData() {
  'use strict';

  /* ---------------- 官方结构参数 ---------------- */

  const CONFIG = {
    vocab_size: 129280, dim: 4096, moe_inter_dim: 2048, n_layers: 43, n_hash_layers: 3,
    n_heads: 64, n_routed_experts: 256, n_shared_experts: 1, n_activated_experts: 6,
    score_func: 'sqrtsoftplus', route_scale: 1.5, swiglu_limit: 10.0,
    q_lora_rank: 1024, head_dim: 512, rope_head_dim: 64, o_groups: 8, o_lora_rank: 1024,
    window_size: 128, index_n_heads: 64, index_head_dim: 128, index_topk: 512,
    hc_mult: 4, hc_sinkhorn_iters: 20, dtype: 'fp8', expert_dtype: 'fp4',
    compress_ratios: [0, 0, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 0],
  };

  // 压缩注意力只在 compress_ratio 非 0 的层生效。注意官方 inference-config.json 里
  // compress_ratios 有 44 项、比 n_layers = 43 多一项，这里按 n_layers 截断。
  const ratios = CONFIG.compress_ratios.slice(0, CONFIG.n_layers);
  CONFIG.compressLayers = ratios.filter((r) => r > 0).length;
  // model.py:468 —— indexer 只在 compress_ratio == 4 的层构造，128 的层是 compressor-only
  CONFIG.indexerLayers = ratios.filter((r) => r === 4).length;
  CONFIG.expertsPerToken = CONFIG.n_activated_experts + CONFIG.n_shared_experts;

  const SOURCE = 'Data/DeepSeek-V4-Flash-Official/model.py';
  const DOC_BASE = 'https://gitcode.com/Ascend/op-plugin/tree/7.3.0/docs/context/';

  /* ---------------- 适配状态（替代旧的推荐等级） ---------------- */

  const STATUS = {
    ready: { id: 'ready', label: '可直接替换', rank: 0, desc: '命中参考链路，无需前置改造，可直接进入替换验证' },
    prep: { id: 'prep', label: '需前置改造', rank: 1, desc: '候选算子成立，但要先完成 RoPE 收敛、PA 改造等前置项' },
    constraint: { id: 'constraint', label: '约束待确认', rank: 2, desc: '关键参数或语义与算子文档不一致，需查文档核实后才能定' },
    none: { id: 'none', label: '无现成算子', rank: 3, desc: '算子总表中无对应语义，转新增融合算子需求' },
    blocked: { id: 'blocked', label: '不适配', rank: 4, desc: '存在硬约束且无法通过合理前置改造解决，附证据' },
  };

  // 进「推荐方案」列表的只有能落地的两档，其余进「未推荐原因」
  const ACTIONABLE = ['ready', 'prep'];
  const isActionable = (item) => ACTIONABLE.includes(item.status);

  const KIND = {
    cross: { id: 'cross', label: '跨模块超级融合', desc: '一个 API 吃掉整网图上多个模块，结构图上可见合并' },
    intra: { id: 'intra', label: '模块内融合', desc: '融合发生在模块内部，整网拓扑不变，看代码级调用对照' },
  };

  const WORKLOAD = {
    prefill: { id: 'prefill', label: 'Prefill', note: 'start_pos == 0', cares: '大矩阵吞吐、GM 带宽、长序列 block 流水' },
    decode: { id: 'decode', label: 'Decode', note: 'start_pos > 0', cares: 'launch/sync、KV cache 访问、实际 batch、短 token latency' },
  };

  /* ---------------- torch_npu 算子登记表 ----------------
   * listed: 是否出现在 skill 附带的 144 个算子总表里。
   * 总表是版本快照，skill 明确要求进候选后必须用本地 docstring 或在线文档核对，
   * 所以这里只记「表里有没有」，不等于「当前版本可用」。
   */

  const APIS = {
    npu_mla_prolog_v3: { listed: true, beta: true, note: 'MLA 前处理：四条并行计算路径，融合 Q/KV 投影 + RMSNorm + RoPE + Cache 写入' },
    npu_kv_rmsnorm_rope_cache: { listed: true, note: '融合 MLA 结构中 RMSNorm + RoPE + ScatterUpdate(KVCache 写入)' },
    npu_lightning_indexer: { listed: true, note: '基于一系列操作得到每个 token 的 Top-k 位置' },
    npu_quant_lightning_indexer: { listed: true, note: 'SFA 前处理，量化实现存 8 算 8', alias: 'npu_lightning_indexer_quant（参考文档写法，与总表不一致）' },
    npu_sparse_flash_attention: { listed: true, note: '大序列长度推理的稀疏注意力，只计算关键部分' },
    npu_kv_quant_sparse_flash_attention: { listed: true, note: 'SFA 基础上支持 Per-Token-Head-Tile-128 量化输入', alias: 'npu_sparse_flash_attention_antiquant（参考文档写法，与总表不一致）' },
    npu_rms_norm: { listed: true, note: 'RMSNorm 归一化' },
    npu_add_rms_norm: { listed: false, note: '三份 attention 参考链路都在用，但不在 144 个算子总表内，必须先核实是否存在' },
    npu_swiglu: { listed: true, beta: true, note: 'Swish 门控线性单元激活' },
    npu_dequant_swiglu_quant: { listed: true, note: '反量化 + SwiGLU + 量化' },
    npu_grouped_matmul_swiglu_quant_v2: { listed: true, note: 'Grouped MatMul + SwiGLU + 量化融合' },
    npu_grouped_matmul: { listed: true, note: '多个 matmul 分组批量计算，减少访存与调度开销' },
    npu_moe_gating_top_k: { listed: true, note: '对输入做 Sigmoid/SoftMax，分组排序后选 top-k 专家' },
    npu_moe_init_routing_v2: { listed: true, note: 'MoE routing 计算，支持不量化和动态量化' },
    npu_moe_finalize_routing: { listed: true, note: 'MoE 计算最后合并 FFN 输出' },
    npu_moe_re_routing: { listed: true, note: 'AlltoAll 后按专家顺序重排 token' },
    npu_moe_distribute_dispatch_v2: { listed: true, note: 'EP 并行 token dispatch（MC2）' },
    npu_moe_distribute_combine_v2: { listed: true, note: 'EP 并行 token combine（MC2）' },
    npu_moe_distribute_combine_add_rms_norm: { listed: true, note: 'combine 原路返回后再做 add_rms_norm' },
    npu_transpose_batchmatmul: { listed: true, note: '带转置序列的三维张量矩阵乘' },
    npu_rotary_mul: { listed: true, note: 'RoPE 旋转位置编码' },
    npu_interleave_rope: { listed: true, note: 'interleave 模式 RoPE' },
    npu_scatter_nd_update_: { listed: true, note: '按 slot 更新 cache（Decode 写入）' },
    npu_scatter_pa_kv_cache: { listed: true, note: '更新 KVCache 中指定位置的 key/value' },
    npu_dynamic_quant: { listed: true, note: '动态量化' },
    npu_top_k_top_p_sample: { listed: true, note: 'top-k / top-p 采样' },
    npu_gather_sparse_index: { listed: true, note: '按稀疏索引 gather' },
    npu_mm_all_reduce_base: { listed: true, note: 'matmul + all_reduce 融合，通信计算流水并行' },
  };

  const apiDoc = (name) => DOC_BASE + 'torch_npu-' + name + '.md';

  /* ---------------- 模块替换方案 ---------------- */

  const c = CONFIG;

  const MODULES = [
    /* ========== 跨模块超级融合：整网图上真实合并 ========== */
    {
      id: 'mod-mla-prolog',
      kind: 'cross',
      title: 'MLA Prolog 超级融合',
      module: 'RMSNorm + Q/KV 投影 + RoPE + KV Cache 写入',
      nodes: ['pre-norm', 'qkv-proj'],
      apis: ['npu_mla_prolog_v3'],
      reference: 'cann-recipes-infer/models/deepseek-v3.2-exp（MLA + Indexer 链路）',
      layers: c.n_layers,
      status: 'prep',
      covers: ['attn_norm（RMSNorm）', 'wq_a / wq_b 低秩 Q 投影', 'wkv latent KV 投影 + kv_norm', 'Q/K RoPE', 'KV Cache 写入'],
      uncovered: ['indexer 打分（另有 npu_lightning_indexer）', 'sparse_attn 本体', 'wo_a / wo_b 输出投影'],
      before: [
        ['x = self.attn_norm(x)', 'model.py:691'],
        ['qr = q = self.q_norm(self.wq_a(x))', 'model.py:496'],
        ['q = self.wq_b(q).unflatten(-1, (n_heads, head_dim))', 'model.py:497'],
        ['q *= torch.rsqrt(q.square().mean(-1, keepdim=True) + eps)', 'model.py:498'],
        ['apply_rotary_emb(q[..., -rd:], freqs_cis)', 'model.py:499'],
        ['kv = self.kv_norm(self.wkv(x))', 'model.py:502-503'],
        ['apply_rotary_emb(kv[..., -rd:], freqs_cis)', 'model.py:504'],
        ['act_quant(kv[..., :-rd], 64, scale_fmt, ...)', 'model.py:506'],
        ['self.kv_cache[:bsz, start_pos % win] = kv.squeeze(1)', 'model.py:530'],
      ],
      after: [
        ['q_nope, q_pe, _, qr, _ = npu_mla_prolog_v3(', ''],
        ['    token_x=hidden_states, weight_dq=..., weight_uq_qr=...,', ''],
        ['    weight_uk=kv_b_proj_w_k, weight_dkv_kr=...,', ''],
        ['    kv_cache=nope_cache, kr_cache=rope_cache,', ''],
        ['    cache_mode="PA_BSND", cache_index=slot_mapping,', ''],
        ['    weight_quant_mode=1)', ''],
      ],
      stages: {
        prefill: 'cache_mode="PA_BSND"；CP size > 1 时用 "BSND" + 后续手动 scatter_update_ 写 cache',
        decode: 'cache_mode="PA_BSND" + cache_index=slot_mapping',
      },
      prep: [
        'KV Cache 从当前的环形缓冲（kv_cache[:, start_pos % window_size]，model.py:530）改造为 PA + slot_mapping',
        'RoPE 预计算与取值收敛到模型级统一入口，产出稳定的 cos/sin（见 skill 的 rotary-embedding-pattern.md）',
      ],
      constraints: [
        { item: 'V4 在 wq_b 之后多一次 per-head RMS 归一化（无权重）', why: 'model.py:498 的 q *= rsqrt(q.square().mean(-1))，标准 MLA prolog 链路里没有这一步，需确认算子是否覆盖或可外提', checked: false },
        { item: 'head_dim = 512、rope_head_dim = 64 是否在算子支持范围', why: '需查 npu_mla_prolog_v3 文档的 shape 约束', checked: false },
        { item: 'weight_quant_mode 与 V4 的 FP8（scale_fmt=ue8m0）是否匹配', why: '参考链路用 weight_quant_mode=1 仅 qb 量化', checked: false },
      ],
      verify: '先只替换 Q 侧投影 + RoPE，KV cache 写入维持原样，确认 q_nope/q_pe 与参考实现逐元素对齐后再接 cache',
      fallback: '退回分步：npu_rms_norm + 原生投影 + npu_kv_rmsnorm_rope_cache 处理 KV 侧',
    },
    {
      id: 'mod-moe-grouped',
      kind: 'cross',
      title: 'MoE 路由 + 专家计算 + 合并',
      module: 'Gate 之后的整条 MoE 执行链',
      nodes: ['expert-select', 'routed-experts', 'expert-swiglu', 'combine'],
      apis: ['npu_moe_init_routing_v2', 'npu_grouped_matmul', 'npu_moe_finalize_routing', 'npu_moe_distribute_dispatch_v2', 'npu_moe_distribute_combine_v2'],
      reference: 'skill MoE 判断树 · framework_moe_parallel.md（本次 zip 未包含该文件）',
      layers: c.n_layers,
      status: 'prep',
      covers: ['token → 专家的 routing 重排', '每专家 GEMM（w1/w3/w2）', '加权合并回 hidden'],
      uncovered: ['shared_expert 分支（可与 routed 并行）', 'gate 打分本身（见 MoE Gate 方案）'],
      before: [
        ['counts = torch.bincount(indices.flatten(), minlength=n_routed_experts).tolist()', 'model.py:634'],
        ['for i in range(experts_start_idx, experts_end_idx):', 'model.py:635'],
        ['    if counts[i] == 0: continue', 'model.py:636-637'],
        ['    idx, top = torch.where(indices == i)', 'model.py:639'],
        ['    y[idx] += expert(x[idx], weights[idx, top, None])', 'model.py:640'],
        ['dist.all_reduce(y)', 'model.py:642'],
      ],
      after: [
        ['# Prefill（纯 TP）', ''],
        ['x_sorted, idx, counts = npu_moe_init_routing_v2(x, indices, ...)', ''],
        ['y = npu_grouped_matmul([x_sorted], [w], group_list=counts, ...)', ''],
        ['out = npu_moe_finalize_routing(y, ..., weights, idx)', ''],
        ['# Decode（EP，MC2）', ''],
        ['t = npu_moe_distribute_dispatch_v2(x, indices, ep_group, ...)', ''],
        ['y = npu_grouped_matmul(...)', ''],
        ['out = npu_moe_distribute_combine_v2(y, ...)', ''],
      ],
      stages: {
        prefill: '纯 TP：init_routing_v2 → grouped_matmul → finalize_routing；EP 时中间插 AllToAll + npu_moe_re_routing',
        decode: 'EP dispatch/combine 走 MC2：dispatch_v2 → grouped_matmul → combine_v2',
      },
      prep: [
        '当前是 Python for-loop 逐专家执行（model.py:635-640），要先把专家权重按 grouped 布局拼好',
        '专家负载分布需给出 P99 上界，供 grouped_matmul 的 group_list 使用',
      ],
      constraints: [
        { item: 'A2 常规 MC2 要求每 rank 专家数 ≤ 24', why: '本模型 256 routed experts，需 ep_world_size − shared_expert_rank_num ≥ 11 才满足；否则要走 double-routing 等回退', checked: false },
        { item: 'FP4 专家权重（expert_dtype=fp4，model.py:625）能否进 grouped_matmul', why: '需查 npu_grouped_matmul 的 dtype 约束', checked: false },
        { item: 'MoE 实施细节依赖 framework_moe_parallel.md', why: 'skill 把 MoE 详细参数组合外包给该文档，本次 zip 未包含', checked: false },
      ],
      verify: '先在单卡 TP 下替换 init_routing_v2 + grouped_matmul，与 for-loop 版本对拍专家输出；EP/MC2 路径单独验证',
      fallback: '只替换单专家的 w1/w3/w2 为 grouped_matmul，routing 仍走原逻辑',
    },
    {
      id: 'mod-compress-tail',
      kind: 'intra',
      title: '压缩 KV 尾链三合一',
      module: 'compressor 尾部 RMSNorm + RoPE + 压缩 cache 写入',
      nodes: ['compressor'],
      apis: ['npu_kv_rmsnorm_rope_cache'],
      reference: 'deepseek_r1 的 KV RMSNorm+RoPE+Cache 三合一写法',
      layers: CONFIG.compressLayers,
      status: 'prep',
      covers: ['压缩 KV 的 RMSNorm', '压缩 KV 的 RoPE', '压缩 cache 写入'],
      uncovered: ['前面的 gated pooling 压缩（无现成算子，见「压缩池化」）'],
      before: [
        ['kv = self.norm(kv.to(dtype))', 'model.py:361'],
        ['apply_rotary_emb(kv[..., -rd:], freqs_cis)', 'model.py:367'],
        ['act_quant(kv[..., :-rd], 64, scale_fmt, scale_dtype, True)', 'model.py:371'],
        ['self.kv_cache[:bsz, start_pos // ratio] = kv.squeeze(1)', 'model.py:376'],
      ],
      after: [
        ['k_rope, k_nope = npu_kv_rmsnorm_rope_cache(', ''],
        ['    latent_cache, norm.weight, cos, sin,', ''],
        ['    slot_mapping, rope_cache, nope_cache,', ''],
        ['    cache_mode="PA_BSND", is_output_kv=True)', ''],
      ],
      stages: {
        prefill: '整段压缩后一次写入 kv_cache[:, :seqlen // ratio]（model.py:374）',
        decode: '仅在 (start_pos + 1) % ratio == 0 时触发，写单槽（model.py:376）',
      },
      prep: ['压缩 cache 改造为 PA + slot_mapping', 'RoPE cos/sin 统一入口'],
      constraints: [
        { item: 'decode 下该链路是条件执行（duty cycle 1/ratio）', why: 'model.py:344 should_compress =(start_pos+1)%ratio==0，融合算子在不触发的 step 要能整体跳过', checked: false },
        { item: 'ratio=4 的层带 overlap_transform，ratio=128 不带', why: 'model.py:290 overlap = compress_ratio == 4，两类层可能要分别特化', checked: false },
      ],
      verify: '按 compress_ratio = 4 与 128 两类层分别对拍压缩 cache 内容',
      fallback: '只融合 RMSNorm + RoPE，量化与 cache 写入保持独立',
    },

    /* ========== 模块内融合：整网拓扑不变，看代码对照 ========== */
    {
      id: 'mod-rmsnorm',
      kind: 'intra',
      title: 'RMSNorm',
      module: 'attn_norm / ffn_norm / final_norm',
      nodes: ['pre-norm', 'final-norm'],
      apis: ['npu_rms_norm'],
      reference: '所有参考链路通用',
      layers: c.n_layers,
      status: 'ready',
      covers: ['float 上转', '平方 + 均值归约', 'rsqrt(var + eps)', '乘 weight 并回落 dtype'],
      uncovered: [],
      before: [
        ['x = x.float()', 'model.py:193'],
        ['var = x.square().mean(-1, keepdim=True)', 'model.py:194'],
        ['x = x * torch.rsqrt(var + self.eps)', 'model.py:195'],
        ['return (self.weight * x).to(dtype)', 'model.py:196'],
      ],
      after: [
        ['x = torch_npu.npu_rms_norm(x, self.weight, epsilon=self.eps)[0]', ''],
      ],
      stages: { prefill: '无差异', decode: '无差异' },
      prep: [],
      constraints: [
        { item: 'weight 在 checkpoint 里是 bf16、参数存成 fp32（model.py:189）', why: '需确认算子对 weight dtype 的要求，避免多一次 cast', checked: false },
      ],
      verify: '单模块逐元素误差对拍，覆盖极值输入',
      fallback: '保持原实现',
      note: '这一条是「模块内融合」的典型：eager 下是 8 个 kernel，融合成 1 个，但整网结构图上 input_rmsnorm 本来就是一个节点，拓扑没有变化。',
    },
    {
      id: 'mod-indexer',
      kind: 'intra',
      title: 'Lightning Indexer',
      module: 'indexer 投影 + RoPE + 量化 + top-k',
      nodes: ['indexer'],
      apis: ['npu_lightning_indexer', 'npu_quant_lightning_indexer'],
      reference: 'deepseek-v3.2-exp 的 Indexer 链路',
      layers: CONFIG.indexerLayers,
      status: 'constraint',
      covers: ['indexer q 投影', 'RoPE + Hadamard', '激活量化', 'einsum 打分 + top-k'],
      uncovered: ['indexer 自带的 compressor 调用（model.py:417）'],
      before: [
        ['q = ...wq_b projection', 'model.py:406-410'],
        ['apply_rotary_emb + rotate_activation', 'model.py:411-413'],
        ['fp4_act_quant(q, fp4_block_size, True)', 'model.py:414'],
        ['index_score = torch.einsum("bshd,btd->bsht", q, self.kv_cache[...])', 'model.py:420'],
        ['topk_idxs = ...topk + causal mask', 'model.py:424-431'],
      ],
      after: [
        ['topk_indices = npu_lightning_indexer(q, k, weights, ...)', ''],
        ['# 量化路径：npu_quant_lightning_indexer(...)', ''],
      ],
      stages: {
        prefill: '建 mask 后做 top-k（model.py:425）；Indexer KV Cache 用 scatter_update_',
        decode: '走 start_pos 分支不建 mask；Indexer KV Cache 用 npu_scatter_nd_update_',
      },
      prep: [],
      constraints: [
        { item: 'V4 用 FP4 激活量化（fp4_act_quant，block 32）', why: 'npu_quant_lightning_indexer 文档写的是「存 8 算 8」INT8 路径，FP4 支持情况必须先确认', checked: false },
        { item: '参考文档写 npu_lightning_indexer_quant，总表是 npu_quant_lightning_indexer', why: '两者命名不一致，需以本地 docstring 为准', checked: false },
        { item: 'index_topk = 512、index_head_dim = 128 的 shape 约束', why: '需查算子文档', checked: false },
      ],
      verify: 'top-k 索引集合一致率（不是只看分数误差）',
      fallback: '只替换 RoPE 部分为 npu_rotary_mul，打分与 top-k 保持原实现',
    },
    {
      id: 'mod-sparse-attn',
      kind: 'intra',
      title: '稀疏 Flash Attention',
      module: 'sparse_attn 本体',
      nodes: ['sparse-attn'],
      apis: ['npu_sparse_flash_attention', 'npu_kv_quant_sparse_flash_attention'],
      reference: 'deepseek-v3.2-exp：layout_query TND / layout_kv PA_BSND / sparse_mode 3',
      layers: c.n_layers,
      status: 'prep',
      covers: ['稀疏 top-k KV 选择下的注意力计算'],
      uncovered: ['V absorb（后接 matmul）', '输出 RoPE 反旋转（model.py:535）'],
      before: [
        ['o = sparse_attn(q, self.kv_cache[:bsz], self.attn_sink, topk_idxs, self.softmax_scale)', 'model.py:533'],
      ],
      after: [
        ['output = npu_sparse_flash_attention(', ''],
        ['    query=q_nope, key=k_latent, value=k_latent,', ''],
        ['    query_rope=q_pe, key_rope=k_pe,', ''],
        ['    sparse_indices=topk_indices,', ''],
        ['    layout_query="TND", layout_kv="PA_BSND", sparse_mode=3)', ''],
      ],
      stages: {
        prefill: '吃当前算出的 kv（可能 concat 上 kv_compress），model.py:527',
        decode: '吃整个环形 cache self.kv_cache[:bsz]，model.py:533',
      },
      prep: ['KV Cache 改造为 PA_BSND'],
      constraints: [
        { item: 'V4 传了 attn_sink 参数', why: 'model.py:533 的 sparse_attn 带 attention sink，需确认 npu_sparse_flash_attention 是否支持', checked: false },
        { item: 'sparse_attn 已是 kernel 模块里的自定义算子（model.py:12 import）', why: '不是 eager 小算子拼的，替换属于 API 平移，收益要靠实测而非「减少 launch」推断', checked: false },
      ],
      verify: '先跑 Baseline B：现有 sparse_attn 与 npu_sparse_flash_attention 直接对比，确认有正收益再继续',
      fallback: '保持现有 sparse_attn',
    },
    {
      id: 'mod-swiglu',
      kind: 'intra',
      title: 'Expert SwiGLU',
      module: '专家 FFN 的 clamp + SiLU + 乘法',
      nodes: ['expert-swiglu', 'shared-expert'],
      apis: ['npu_swiglu', 'npu_dequant_swiglu_quant', 'npu_grouped_matmul_swiglu_quant_v2'],
      reference: 'skill 未匹配模块提示：Dense / Gated FFN 检查 activation 类融合算子',
      layers: c.n_layers,
      status: 'constraint',
      covers: ['SiLU(gate) × up'],
      uncovered: ['clamp(swiglu_limit) 上下界', 'w1 / w3 / w2 投影本身'],
      before: [
        ['gate = self.w1(x).float()', 'model.py:598'],
        ['up = self.w3(x).float()', 'model.py:599'],
        ['up = torch.clamp(up, min=-limit, max=limit)', 'model.py:601'],
        ['gate = torch.clamp(gate, max=limit)', 'model.py:602'],
        ['x = F.silu(gate) * up', 'model.py:603'],
        ['return self.w2(x.to(dtype))', 'model.py:606'],
      ],
      after: [
        ['x = torch_npu.npu_swiglu(torch.cat([gate, up], dim=-1))', ''],
        ['# 量化路径：npu_dequant_swiglu_quant / npu_grouped_matmul_swiglu_quant_v2', ''],
      ],
      stages: { prefill: '无差异', decode: '无差异' },
      prep: [],
      constraints: [
        { item: 'swiglu_limit = 10.0 的非对称 clamp', why: 'model.py:601-602：up 是双边 clamp、gate 只 clamp 上界。标准 swiglu 算子是否支持这个语义要先确认，不支持则 clamp 只能留在外面', checked: false },
        { item: 'FP4 专家权重', why: 'expert_dtype = fp4（model.py:625），量化版 swiglu 算子多为 INT8 路径', checked: false },
        { item: '计算在 float32 下进行（model.py:598-599 .float()）', why: '需确认算子的 accumulation dtype', checked: false },
      ],
      verify: 'clamp 边界值（±10.0 附近）逐元素对拍',
      fallback: 'clamp 保持在外，只把 SiLU × up 换成 npu_swiglu',
    },
    {
      id: 'mod-o-proj',
      kind: 'intra',
      title: '分组输出投影',
      module: 'wo_a 分组 einsum + wo_b 投影',
      nodes: ['out-proj'],
      apis: ['npu_transpose_batchmatmul'],
      reference: 'MLA absorb 链路用它做 V absorb',
      layers: c.n_layers,
      status: 'constraint',
      covers: ['分组矩阵乘'],
      uncovered: ['wo_b 的 RowParallelLinear + all_reduce'],
      before: [
        ['o = o.view(bsz, seqlen, self.n_local_groups, -1)', 'model.py:538'],
        ['o = torch.einsum("bsgd,grd->bsgr", o, wo_a)', 'model.py:542'],
        ['x = self.wo_b(o.flatten(2))', 'model.py:543'],
      ],
      after: [
        ['o = torch_npu.npu_transpose_batchmatmul(o, wo_a, perm_x1=..., perm_x2=...)', ''],
      ],
      stages: { prefill: '无差异', decode: '无差异' },
      prep: [],
      constraints: [
        { item: 'npu_transpose_batchmatmul 仅支持三维 Tensor', why: '当前是 4 维 grouped einsum（bsgd,grd->bsgr），需先 reshape 或确认能否映射', checked: false },
        { item: 'wo_a 在 checkpoint 里是 FP8', why: 'model.py:540 注释指出可做 FP8 einsum，当前用 BF16', checked: false },
      ],
      verify: '与 einsum 版本逐元素对拍，重点看 o_groups = 8 的分组边界',
      fallback: '保持 torch.einsum',
    },
    {
      id: 'mod-lm-head',
      kind: 'intra',
      title: 'LM Head 混合 + 归一化',
      module: 'hc_head 混合 + final RMSNorm + 词表投影',
      nodes: ['lm-head', 'final-norm'],
      apis: ['npu_rms_norm', 'npu_top_k_top_p_sample'],
      reference: 'skill 未匹配模块提示：LM Head 在算子总表中搜索',
      layers: 1,
      status: 'ready',
      covers: ['final RMSNorm', '（可选）采样'],
      uncovered: ['hc_head 的 sigmoid 混合链', '词表投影 + all_gather'],
      before: [
        ['rsqrt = torch.rsqrt(x.square().mean(-1, keepdim=True) + eps)', 'model.py:731'],
        ['mixes = F.linear(x, hc_fn) * rsqrt', 'model.py:732'],
        ['pre = torch.sigmoid(mixes * hc_scale + hc_base) + self.hc_eps', 'model.py:733'],
        ['logits = self.get_logits(norm(x))', 'model.py:720'],
      ],
      after: [
        ['x = torch_npu.npu_rms_norm(x, norm.weight, epsilon=eps)[0]', ''],
        ['# 采样侧可评估 npu_top_k_top_p_sample', ''],
      ],
      stages: { prefill: '通常只取最后一个位置的 logits', decode: '每 step 都命中' },
      prep: [],
      constraints: [
        { item: '词表 129,280 的 logits 无法驻留片上', why: '融合边界只能停在投影之前', checked: false },
        { item: '后接 all_gather（model.py:722-724）', why: '跨通信域，不能继续向后融合', checked: false },
      ],
      verify: 'logits 逐元素误差 + top-1/top-5 一致率',
      fallback: '只替换 final RMSNorm',
    },

    /* ========== 无现成算子 → 新增融合算子需求 ========== */
    {
      id: 'mod-compress-pool',
      kind: 'intra',
      title: '压缩池化（gated pooling）',
      module: 'compressor 的窗口加权池化',
      nodes: ['compressor'],
      apis: [],
      reference: '无匹配参考链路',
      layers: CONFIG.compressLayers,
      status: 'none',
      covers: [],
      uncovered: ['wkv / wgate 双投影', '窗口 reshape + overlap_transform', 'softmax 加权池化'],
      before: [
        ['kv = self.wkv(x); score = self.wgate(x)', 'model.py:322-323'],
        ['kv = kv.unflatten(1, (-1, ratio)); score = score.unflatten(1, (-1, ratio)) + self.ape', 'model.py:337-338'],
        ['kv = self.overlap_transform(kv, 0)', 'model.py:340'],
        ['kv = (kv * score.softmax(dim=2)).sum(dim=2)', 'model.py:343'],
      ],
      after: [],
      stages: {
        prefill: '整段窗口一次性池化（model.py:325-343）',
        decode: '维护 kv_state / score_state 滚动缓冲，按 duty cycle 触发（model.py:344-359）',
      },
      prep: [],
      constraints: [],
      handoff: {
        why: '144 个算子总表里没有「学习式门控池化压缩 KV」这个语义，最接近的只覆盖到尾部 RMSNorm+RoPE+cache',
        payload: '子图语义：wkv/wgate 双投影 → 窗口 reshape（ratio 4/128，ratio=4 带 overlap）→ softmax 加权求和 → 压缩 KV；decode 下带 kv_state/score_state 滚动状态与 duty cycle 条件执行',
        next: '按《昇腾大模型推理融合规则目录》走新增融合算子的范围分析与开发链路',
      },
      verify: '—',
      fallback: '保持原实现，只把尾链交给 npu_kv_rmsnorm_rope_cache',
    },
    {
      id: 'mod-mhc',
      kind: 'intra',
      title: 'mHC 残差连接',
      module: 'hc_pre / hc_post + Sinkhorn',
      nodes: ['mhc'],
      apis: [],
      reference: '无匹配参考链路',
      layers: c.n_layers,
      status: 'none',
      covers: [],
      uncovered: ['hc_split_sinkhorn 迭代归一化', 'pre / post / comb 三路加权'],
      before: [
        ['x, post, comb = self.hc_pre(x, hc_attn_fn, hc_attn_scale, hc_attn_base)', 'model.py:690'],
        ['pre, post, comb = hc_split_sinkhorn(mixes, hc_scale, hc_base, hc_mult, hc_sinkhorn_iters, hc_eps)', 'model.py:679'],
        ['x = self.hc_post(x, residual, post, comb)', 'model.py:693'],
      ],
      after: [],
      stages: { prefill: '无差异', decode: '无差异' },
      prep: [],
      constraints: [],
      handoff: {
        why: 'Sinkhorn 迭代归一化在算子总表中无对应语义；hc_split_sinkhorn 本身已是 kernel 模块的自定义算子（model.py:12）',
        payload: '子图语义：hc_mult=4 份 hidden 副本展平 → rsqrt + F.linear 出 mixing logits → 20 轮 Sinkhorn 行列交替归一化 → pre/post/comb 三路加权；每层出现两次（attn 前后、ffn 前后）',
        next: '按《昇腾大模型推理融合规则目录》走新增融合算子的范围分析与开发链路',
      },
      verify: '—',
      fallback: '保持原实现',
    },

    /* ========== 不适配（附硬约束证据） ========== */
    {
      id: 'mod-moe-gate',
      kind: 'intra',
      title: 'MoE Gate 打分 + top-k',
      module: 'router 打分与专家选择',
      nodes: ['router'],
      apis: ['npu_moe_gating_top_k'],
      reference: 'skill MoE 判断树：deepseek 系列用 npu_moe_gating_top_k',
      layers: c.n_layers - c.n_hash_layers,
      status: 'blocked',
      covers: [],
      uncovered: ['sqrtsoftplus 打分', 'top-k 选择', '路由权重归一化'],
      before: [
        ['scores = linear(x.float(), self.weight.float())', 'model.py:565'],
        ['scores = F.softplus(scores).sqrt()   # score_func = "sqrtsoftplus"', 'model.py:571'],
        ['scores = scores + self.bias', 'model.py:575'],
        ['indices = scores.topk(self.topk, dim=-1)[1]', 'model.py:579'],
        ['weights /= weights.sum(dim=-1, keepdim=True); weights *= self.route_scale', 'model.py:582-584'],
      ],
      after: [],
      stages: { prefill: '无差异', decode: '无差异' },
      prep: [],
      constraints: [],
      blockers: [
        { code: 'SEMANTIC_MISMATCH', note: 'npu_moe_gating_top_k 文档明确是「对输入 x 做 Sigmoid/SoftMax 计算」，而 V4 的 score_func = sqrtsoftplus（model.py:571 的 F.softplus(scores).sqrt()），不在算子支持的打分函数内' },
        { code: 'CONTROL_FLOW', note: '前 n_hash_layers 层走 tid2eid 查表路由而非打分 top-k（model.py:578），同一算子无法覆盖两种路由' },
      ],
      verify: '—',
      fallback: '保持原实现；若确有收益可作为新增算子需求提出「sqrtsoftplus 打分 + top-k」变体',
    },
    {
      id: 'mod-moe-combine-norm',
      kind: 'cross',
      title: 'MoE combine + 残差归一化',
      module: 'combine 之后接 add_rms_norm',
      nodes: ['combine', 'mhc'],
      apis: ['npu_moe_distribute_combine_add_rms_norm'],
      reference: 'MC2 路径：dispatch_v2 + combine_add_rms_norm 配套',
      layers: c.n_layers,
      status: 'blocked',
      covers: [],
      uncovered: ['combine + 残差 + 归一化'],
      before: [
        ['x = self.hc_post(x, residual, post, comb)   # mHC 加权残差合并', 'model.py:700'],
        ['# 而不是标准的 residual + x 后接 RMSNorm', ''],
      ],
      after: [],
      stages: { prefill: '无差异', decode: '无差异' },
      prep: [],
      constraints: [],
      blockers: [
        { code: 'SEMANTIC_MISMATCH', note: 'V4 用 mHC 取代普通残差加法：残差在 hc_post 里按 Sinkhorn 出来的 post/comb 权重合并（model.py:683-686、700），不是 add_rms_norm 的 residual + x 语义' },
        { code: 'STRUCTURE_MISMATCH', note: 'mHC 每层出现两次（attn 前后、ffn 前后，model.py:688-700），与 combine 的一次性收尾对不上' },
      ],
      verify: '—',
      fallback: 'combine 用 npu_moe_distribute_combine_v2，mHC 保持独立',
      note: '这条同时解释了为什么整条「Residual + Norm」类融合在 V4 上普遍不成立——包括参考链路里高频使用的 npu_add_rms_norm。',
    },
    {
      id: 'mod-embedding',
      kind: 'intra',
      title: 'Embedding 分片查表',
      module: 'ParallelEmbedding',
      nodes: ['embedding-op'],
      apis: ['npu_gather_sparse_index'],
      reference: '无匹配参考链路',
      layers: 1,
      status: 'constraint',
      covers: ['稀疏索引 gather'],
      uncovered: ['分片掩码 + all_reduce'],
      before: [
        ['mask = (x < self.vocab_start_idx) | (x >= self.vocab_end_idx)', 'model.py:96-100'],
        ['y = F.embedding(x, self.weight)', 'model.py:101'],
        ['y[mask] = 0', 'model.py:103'],
        ['dist.all_reduce(y)', 'model.py:104'],
      ],
      after: [
        ['y = torch_npu.npu_gather_sparse_index(self.weight, x)   # 待确认是否适用', ''],
      ],
      stages: { prefill: '整段 seq 查表', decode: '单 token 查表' },
      prep: [],
      constraints: [
        { item: 'npu_gather_sparse_index 是否覆盖分片掩码语义', why: '需查文档；掩码与 all_reduce 大概率仍要留在外面', checked: false },
        { item: '整网只执行一次，绝对收益很小', why: '排序时不应挤占前列', checked: false },
      ],
      verify: '多卡分片下输出一致性',
      fallback: '保持原实现',
    },
  ];

  /* ---------------- 计算 ---------------- */

  /** 方案在整网图上是否真的会改变拓扑：跨模块 + 覆盖到 2 个及以上 L2 节点 */
  function changesTopology(item) {
    return item.kind === 'cross' && item.nodes.length > 1;
  }

  function decorate(item) {
    const status = STATUS[item.status] || STATUS.constraint;
    const apis = (item.apis || []).map((name) => ({
      name,
      ...(APIS[name] || { listed: false, note: '未登记' }),
      doc: apiDoc(name),
    }));
    return {
      ...item,
      statusMeta: status,
      kindMeta: KIND[item.kind],
      apiList: apis,
      // skill 第四步要求每个候选 API 都查过官方文档；这里没有 torch_npu 环境，
      // 一律记为未核对，由「算子核对」页签显式暴露，而不是假装已经确认过
      unchecked: (item.constraints || []).filter((x) => !x.checked).length,
      topology: changesTopology(item),
    };
  }

  function evaluate() {
    const items = MODULES.map(decorate)
      .sort((a, b) => (a.statusMeta.rank - b.statusMeta.rank)
        || (b.topology - a.topology)
        || (b.nodes.length - a.nodes.length)
        || ((b.layers || 0) - (a.layers || 0)));

    const actionable = items.filter(isActionable);
    const deferred = items.filter((x) => !isActionable(x));
    const summary = {
      total: items.length,
      actionable: actionable.length,
      cross: actionable.filter((x) => x.topology).length,
      intra: actionable.filter((x) => !x.topology).length,
      byStatus: Object.keys(STATUS).reduce((acc, key) => {
        acc[key] = items.filter((x) => x.status === key).length;
        return acc;
      }, {}),
      apiCount: new Set(items.flatMap((x) => x.apis || [])).size,
      // 需要重点核对的：总表里没有的，或参考链路与总表命名不一致的
      riskyApis: Object.keys(APIS).filter((n) => APIS[n].listed === false || APIS[n].alias).length,
      handoff: items.filter((x) => x.status === 'none').length,
    };
    return { items, actionable, deferred, summary };
  }

  /** 算子核对清单：skill 第三步/第四步的硬要求 */
  function apiChecklist() {
    const used = new Map();
    MODULES.forEach((m) => (m.apis || []).forEach((name) => {
      if (!used.has(name)) used.set(name, []);
      used.get(name).push(m.title);
    }));
    // 参考链路里出现、但总表没有或命名对不上的算子，即使当前没挂到模块上
    // 也要进核对清单——它们正是最容易照抄出错的地方
    Object.keys(APIS).forEach((name) => {
      if (used.has(name)) return;
      if (APIS[name].listed === false || APIS[name].alias) used.set(name, []);
    });
    return [...used.entries()].map(([name, modules]) => ({
      name,
      modules,
      ...(APIS[name] || { listed: false, note: '未登记' }),
      doc: apiDoc(name),
    })).sort((a, b) => (a.listed === b.listed ? a.name.localeCompare(b.name) : (a.listed ? 1 : -1)));
  }

  window.PtoFusionRules = {
    CONFIG, STATUS, KIND, WORKLOAD, APIS, SOURCE, ACTIONABLE,
    modules: MODULES,
    isActionable,
    changesTopology,
    evaluate,
    apiChecklist,
    apiDoc,
  };
})();
