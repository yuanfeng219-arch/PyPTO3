/**
 * 整网融合推荐器 · 数据层
 *
 * 口径来源：Product_Planning/昇腾大模型推理融合规则目录.md（2026-08-28）
 * 与 fusion-rules-data.js（model-infer-fusion skill）完全不同的一套逻辑：
 *
 *   skill 那套  ——「现有 torch_npu 算子能替换掉哪段模型代码」，产物是替换方案
 *   本文件      ——「这张图里还有哪些子图值得做成一个融合算子」，产物是融合候选
 *
 * 所以本文件的作用对象是 torch_npu 部署态整网（替换已经发生之后的图），
 * 回答的是文档 §23 的那条链路：
 *   模型子图 → 推荐融合边界 → 融合形态 → 节省的 GM/launch/sync
 *   → 需要满足的条件 → 硬件代价 → 可回退实现 → 验证计划
 */
(function registerFusionCatalog() {
  'use strict';

  const SOURCE = 'Product_Planning/昇腾大模型推理融合规则目录.md';
  const DOC_DATE = '2026-08-28';

  /* 与 Data/DeepSeek-V4-Flash-Official/inference-config.json 对齐 */
  const CONFIG = {
    dim: 4096,
    moe_inter_dim: 2048,
    n_layers: 43,
    n_heads: 64,
    head_dim: 512,
    rope_head_dim: 64,
    q_lora_rank: 1024,
    o_groups: 8,
    o_lora_rank: 1024,
    n_routed_experts: 256,
    n_shared_experts: 1,
    n_activated_experts: 6,
    n_hash_layers: 3,
    route_scale: 1.5,
    index_n_heads: 64,
    index_head_dim: 128,
    index_topk: 512,
    window_size: 128,
    vocab_size: 129280,
    hc_mult: 4,
    hc_sinkhorn_iters: 20,
    swiglu_limit: 10.0,
    dtype: 'fp8',
    expert_dtype: 'fp4',
  };

  /* ---------------- 文档 §3.1 推荐等级 ---------------- */

  const LEVELS = {
    strong: { id: 'strong', label: '强推荐', desc: '收益明确、实现路径成熟、阻断条件少', rank: 0 },
    conditional: { id: 'conditional', label: '条件推荐', desc: '收益较好，但依赖 shape、layout、dtype 或场景', rank: 1 },
    cautious: { id: 'cautious', label: '谨慎推荐', desc: '理论收益存在，但可能损失并行度、精度或稳定性', rank: 2 },
    reject: { id: 'reject', label: '不推荐', desc: '收益低、风险高或存在硬阻断', rank: 3 },
    unknown: { id: 'unknown', label: '信息不足', desc: '关键 shape、场景或硬件信息缺失', rank: 4 },
  };

  /** 进入主推荐列表的等级；谨慎/不推荐/信息不足进「未推荐」页签并给原因码 */
  const ACTIONABLE = ['strong', 'conditional'];

  /* ---------------- 文档 §3.2 融合形态 ---------------- */

  const SHAPES = {
    local_dataflow: { label: 'Local dataflow', desc: '相邻算子形成稳定的数据流短链' },
    composite_model_op: { label: 'Composite model op', desc: '多个基础算子对外封装为一个模型语义算子' },
    stage_fusion: { label: 'Stage fusion', desc: '一个模型阶段内的多个子步骤共享数据访问或中间状态' },
    multi_branch: { label: 'Multi-branch fusion', desc: '共享输入、汇合输出或多输出关系共同评估' },
    schedule_synergy: { label: 'Schedule/communication synergy', desc: '不强行合并算子，按阶段做分块、流水或重叠' },
    subgraph_orchestration: { label: 'Subgraph orchestration', desc: '保留多个执行步骤，以统一依赖、buffer 和回退路径交付' },
  };

  /* ---------------- 文档 §2.2 实现路径 ---------------- */

  const MODES = {
    custom_fused_op: { label: 'custom_fused_op', desc: '可实现为一个面向模型子图的自定义融合算子' },
    composite_op: { label: 'composite_op', desc: '由多个 PyPTO/CANN 算子组成，但对模型层提供统一算子接口' },
    orchestrated_subgraph: { label: 'orchestrated_subgraph', desc: '保留多个阶段，通过统一编排、buffer 和依赖交付' },
    overlap_schedule: { label: 'overlap_schedule', desc: '保留通信/计算边界，以分块、流水或调度重叠获得收益' },
  };

  const PYPTO_STATUS = {
    not_checked: '未校验',
    draft: '草图',
    ir_verified: 'IR 已验证',
    runtime_verified: 'Runtime 已验证',
  };

  /* ---------------- 文档 §19 失败原因码 ---------------- */

  const REASONS = {
    MULTI_CONSUMER: '中间结果有多个消费者',
    SHAPE_MISMATCH: 'shape 或广播关系不满足',
    DYNAMIC_SHAPE_UNBOUNDED: '动态 shape 缺少稳定上界',
    REDUCTION_CONFLICT: '归约维度与分块/并行冲突',
    LAYOUT_CONFLICT: 'layout 转换成本或约束不可接受',
    MEMORY_PRESSURE: '片上 buffer 估算超限或压力过高',
    CUBE_VECTOR_BOUNDARY: 'Cube/Vector 交接成本过高或缺少实现路径',
    COMMUNICATION_BOUNDARY: '跨通信域，依赖或同步不适合融合',
    NUMERICAL_RISK: '归约顺序、舍入或近似函数风险',
    PARALLELISM_LOSS: '融合后并行度明显下降',
    LIBRARY_BASELINE_STRONG: '现有库实现可能更优',
    MISSING_BACKEND_SUPPORT: '目标后端缺少必要算子或模板',
    INSUFFICIENT_PROFILE: '缺少实测数据，无法排序',
  };

  /* ---------------- 文档 §5.3 排序公式 ---------------- */

  const WEIGHTS = [
    { key: 'gm', w: 4, label: 'GM traffic saved', sign: 1 },
    { key: 'launch', w: 3, label: 'launch saved', sign: 1 },
    { key: 'reuse', w: 2, label: 'reuse / overlap benefit', sign: 1 },
    { key: 'sync', w: 3, label: 'extra synchronization', sign: -1 },
    { key: 'onchip', w: 3, label: 'on-chip pressure', sign: -1 },
    { key: 'parallel', w: 2, label: 'parallelism loss', sign: -1 },
    { key: 'numeric', w: 2, label: 'numerical risk', sign: -1 },
    { key: 'complexity', w: 1, label: 'implementation complexity', sign: -1 },
  ];

  /* ---------------- 文档 §16 场景 ---------------- */

  const SCENARIOS = {
    prefill: {
      id: 'prefill',
      label: 'Prefill',
      cares: '大矩阵吞吐、GM 带宽、长序列 block 流水',
      // execution_weight（文档 §5.5）。没有真实 workload，这里是明示的假设值
      tokens: 4096,
      weightNote: '假设单请求 4,096 token prompt',
    },
    decode: {
      id: 'decode',
      label: 'Decode',
      cares: 'launch/sync、KV cache 访问、实际 batch、短 token latency',
      tokens: 32,
      weightNote: '假设实际 batch 32 · 每步 1 token',
    },
  };

  /* ---------------- 文档 §20 验证闭环四道门 ---------------- */

  const GATES = [
    { id: 'graph', label: '图级语义验证', desc: '融合前后输出、shape、依赖和 side effect 一致' },
    { id: 'kernel', label: '算子级精度验证', desc: '随机输入、边界值、长序列、极值和不同 dtype' },
    { id: 'perf', label: '实现级性能验证', desc: '比较原始子图、局部融合、组合融合和编排方案' },
    { id: 'e2e', label: '整网级收益验证', desc: '按层数、场景和 workload 观察 TTFT/TPOT/吞吐/峰值内存' },
  ];

  const BASELINES = [
    'Baseline A：原始模型图',
    'Baseline B：已有 CANN/库算子',
    'Candidate A：局部融合',
    'Candidate B：更大范围融合',
    'Candidate C：融合 + tile/layout/并行调优',
  ];

  /* ---------------- §5.4 中间量 ----------------
   * 统一为「单次执行量」：单 token 单次调用。
   * gm_saved_bytes = producer_output_bytes × eliminated_materializations × 2
   * 乘 2 是因为一次物化同时省掉一次写回和一次读回。
   */

  const BF16 = 2;
  const FP8 = 1;
  const gmSaved = (elems, bytes, materializations) => elems * bytes * materializations * 2;

  const c = CONFIG;

  /* ---------------- 融合候选 ----------------
   * site 指向部署态整网上的位置：module 是 L2 模块，steps 是它 L3 展开里的步骤下标。
   * 文档的规则匹配的是算子级模式，所以候选落在 L3，而不是模块本身。
   */

  const CANDIDATES = [
    {
      id: 'cand-swiglu',
      ruleId: 'FUS-FFN-002',
      category: 'ffn',
      priority: 'P0',
      title: 'shared_expert SwiGLU 局部融合',
      pattern: 'gate_proj(x) → SiLU → multiply(up_proj(x))',
      modelContext: 'transformer_block.moe.shared_expert',
      roles: ['projection', 'activation', 'projection'],
      site: { module: 'shared-expert', steps: [0, 1, 2, 3] },
      level: 'strong',
      shape: 'multi_branch',
      mode: 'custom_fused_op',
      pyptoStatus: 'not_checked',
      materialization: 'eliminate',
      consumers: { multi: false, crossStage: false },
      boundary: {
        start: 'shared_expert.input（attention_output 经 router 分流后的全量 token）',
        end: 'gate × up 的逐元素乘积',
        preserve: [],
        shared: [],
      },
      semantics: 'routed experts 已经由 npu_moe_init_routing_v2 + npu_grouped_matmul 接管，'
        + 'shared_expert 是这张图里唯一还保持原生形态的 SwiGLU。它每个 token 必过，'
        + '两路 projection 在逐元素乘法处汇合 —— 文档 §12 点名的 FFN 最高优先级候选。',
      benefits: [
        'w1 / w3 两路 projection 的输出在乘法处汇合，中间 activation 不必落 GM',
        'clamp + SiLU + multiply 三个 Vector 步骤合并为一次 tile 内计算',
        '4 次 kernel launch 降为 1 次；每层每 token 必发生',
      ],
      requirements: [
        'gate / up 输出 shape 一致（均为 intermediate ' + c.moe_inter_dim + '）',
        '两路输出按相同 tile/block 对齐',
        'SiLU 只作用于 gate 分支，up 分支保持原值',
        'multiply 结果没有额外分支消费者（当前只流向 w2）',
        'dtype 与 accumulation 规则明确：FP4 expert 权重 / FP32 累加',
      ],
      risks: [
        'swiglu_limit = ' + c.swiglu_limit + ' 的 clamp 位置一旦改变会改变舍入点',
        'FP4 权重下 gate 与 up 的量化 scale 需要在同一 tile 内可读',
      ],
      traffic: {
        gm: gmSaved(c.moe_inter_dim, BF16, 2),
        gmNote: 'gate 输出与 SiLU 输出两次物化 · ' + c.moe_inter_dim + ' × BF16 × 写回+读回',
        launchBefore: 4,
        launchAfter: 1,
        liveSet: c.moe_inter_dim * BF16 * 2,
        liveNote: 'gate tile + up tile 同时在片上',
        syncAdded: 1,
        syncNote: 'Cube（两路 projection）→ Vector（SiLU × multiply）一次交接',
      },
      invocations: c.n_layers,
      invocationNote: '每层一次 · shared expert 对每个 token 都执行',
      factors: {
        prefill: { gm: 4, launch: 2, reuse: 4, sync: 1, onchip: 2, parallel: 1, numeric: 1, complexity: 2 },
        decode: { gm: 3, launch: 4, reuse: 3, sync: 1, onchip: 1, parallel: 1, numeric: 1, complexity: 2 },
      },
      integrationCost: 2,
      validationCost: 2,
      validation: {
        graph: '融合前后 shared_expert 输出逐元素对齐，确认 clamp 边界值（±10）行为一致',
        kernel: '随机输入 + 极值（触发 clamp）+ FP4/BF16 双精度路径的逐元素误差',
        perf: '对比 4 算子编排 / gate+up 合并 / 全链融合三个版本',
        e2e: '43 层全开后的 TPOT 与峰值内存变化',
      },
      fallback: [
        'gate/up projection 合并 + SiLU·multiply 复合算子（两段）',
        'w1 / w3 / SiLU / multiply 多算子编排，仅共享 tile 布局',
        '回退到原生 PyTorch SwiGLU',
      ],
      evidence: { status: 'experience', refs: ['repo/pto/examples/models/01_ffn.py', SOURCE + ' §12 FUS-FFN-002'] },
    },

    {
      id: 'cand-router',
      ruleId: 'FUS-MOE-001',
      category: 'moe',
      priority: 'P1',
      title: 'MoE Router 打分与 top-k 前处理链',
      pattern: 'router logits → sqrtsoftplus → correction bias → top-k → route weight normalize',
      modelContext: 'transformer_block.moe.router',
      roles: ['projection', 'activation', 'routing'],
      site: { module: 'router', steps: [0, 1, 2, 3, 4] },
      level: 'conditional',
      shape: 'local_dataflow',
      mode: 'composite_op',
      pyptoStatus: 'not_checked',
      materialization: 'eliminate',
      consumers: { multi: false, crossStage: true },
      boundary: {
        start: 'attention_output',
        end: 'route weights + expert indices（dispatch 元数据）',
        preserve: ['expert indices（被 npu_moe_init_routing_v2 消费）'],
        shared: ['route table 同时被 combine 阶段读取'],
      },
      semantics: 'dispatch / expert GEMM / combine 已经被 npu_moe_* 系列吃掉，'
        + '剩下的正好是文档 §14 建议的边界：只融合局部逐元素与 top-k 前处理，'
        + '不把大规模 token reorder 和通信封进同一个算子。',
      benefits: [
        '256 维 logits 的 sqrtsoftplus、bias、归一化都是小张量，逐次往返 GM 的代价占比高',
        'top-k 前的逐元素链可以在一个 Vector tile 内完成',
        '5 次 launch 降为 1 次，decode 场景下占比显著',
      ],
      requirements: [
        'expert 数（' + c.n_routed_experts + '）与 top-k（' + c.n_activated_experts + '）静态',
        'correction bias 只在非 hash 层参与，需要两个静态变体而不是运行时分支',
        'route table 作为跨阶段复用张量必须保留物化（materialization = keep_for_reuse）',
        'route_scale = ' + c.route_scale + ' 的乘法与归一化顺序固定',
      ],
      risks: [
        'hash 层走 lookup 而非 top-k，会引入第二个 shape 特化',
        'top-k 的 tie-breaking 顺序变化会改变专家选择，属于可观察输出',
      ],
      traffic: {
        gm: gmSaved(c.n_routed_experts, 4, 3),
        gmNote: 'logits / softplus 输出 / bias 后结果三次物化 · 256 × FP32',
        launchBefore: 5,
        launchAfter: 1,
        liveSet: c.n_routed_experts * 4 * 2,
        liveNote: '256 维 FP32 打分向量 + top-k 工作区',
        syncAdded: 1,
        syncNote: 'Cube（router linear）→ Vector（打分链）一次交接',
      },
      invocations: c.n_layers,
      invocationNote: '每层一次 · 每 token 都要路由',
      factors: {
        prefill: { gm: 2, launch: 3, reuse: 2, sync: 1, onchip: 1, parallel: 2, numeric: 2, complexity: 2 },
        decode: { gm: 2, launch: 5, reuse: 2, sync: 1, onchip: 1, parallel: 2, numeric: 2, complexity: 2 },
      },
      integrationCost: 2,
      validationCost: 3,
      validation: {
        graph: '融合前后 expert indices 与 route weights 完全一致（含 tie-breaking）',
        kernel: 'hash 层 / 非 hash 层两条路径分别验证；构造分数相等的极端输入',
        perf: 'decode 小 batch 下对比 5 算子编排与融合版本的 launch/sync 占比',
        e2e: '专家负载分布与整网 TPOT，确认路由结果未漂移',
      },
      fallback: [
        'router linear 保持独立，只融合 sqrtsoftplus → bias → normalize',
        '逐元素段与 top-k 段拆成两个算子编排',
        '回退到原生实现',
      ],
      evidence: { status: 'experience', refs: [SOURCE + ' §14 FUS-MOE-001'] },
    },

    {
      id: 'cand-final-norm',
      ruleId: 'FUS-NORM-002',
      category: 'norm',
      priority: 'P0',
      title: 'final RMSNorm 内部归约链',
      pattern: 'x → square → reduce_mean → add epsilon → rsqrt → multiply weight',
      modelContext: 'model.final_norm',
      roles: ['normalization'],
      site: { module: 'final-norm', steps: [0, 1, 2, 3] },
      level: 'conditional',
      shape: 'composite_model_op',
      mode: 'custom_fused_op',
      pyptoStatus: 'not_checked',
      materialization: 'eliminate',
      consumers: { multi: false, crossStage: false },
      boundary: {
        start: 'next_hidden_states（最后一层输出）',
        end: '归一化后的 hidden，直接进入 lm_head',
        preserve: [],
        shared: [],
      },
      semantics: '层内的 pre-norm 已经被 npu_mla_prolog_v3 吃掉，整网上只剩这一个裸露的 RMSNorm。'
        + '文档 §9 要求把它整体当作一个 Norm 模板评估，不要逐节点推荐。',
      benefits: [
        'square / reduce / rsqrt / scale 四步在一个 tile 内完成，中间量不落 GM',
        'FP32 归约只在片上发生，避免精度边界来回转换',
      ],
      requirements: [
        '归约维度 hidden = ' + c.dim + ' 与 tile 分块一致，或给出明确的跨 tile reduction 方案',
        'epsilon、权重和 accumulation dtype 明确（FP32 归约后回到激活 dtype）',
        '归一化结果不会被其他分支独立复用',
      ],
      risks: [
        'hidden ' + c.dim + ' 跨多个 tile 时需要跨 tile 归约，是文档点名的高风险情况',
        '后面紧接 lm_head 的大矩阵与 logits，属于误差敏感位置',
      ],
      traffic: {
        gm: gmSaved(c.dim, BF16, 3),
        gmNote: 'square / 归约结果 / rsqrt 后中间量三次物化 · 4096 × BF16',
        launchBefore: 4,
        launchAfter: 1,
        liveSet: c.dim * 4,
        liveNote: 'FP32 累加器 + hidden tile',
        syncAdded: 1,
        syncNote: '跨 tile 归约需要一次显式 reduction',
      },
      invocations: 1,
      invocationNote: '整网只有一次 —— 局部收益不小，但整网贡献被 invocation_count 压低',
      factors: {
        prefill: { gm: 4, launch: 3, reuse: 2, sync: 1, onchip: 1, parallel: 0, numeric: 2, complexity: 2 },
        decode: { gm: 3, launch: 3, reuse: 1, sync: 1, onchip: 1, parallel: 0, numeric: 2, complexity: 2 },
      },
      integrationCost: 1,
      validationCost: 2,
      validation: {
        graph: '融合前后 hidden 输出逐元素对齐',
        kernel: '长序列 + 极值输入下的 FP32 归约累计误差；对比逐 tile 与整体归约',
        perf: '对比库 RMSNorm 与自定义融合算子',
        e2e: 'logits 分布与困惑度回归',
      },
      fallback: [
        '沿用 npu_rms_norm（替换方案面板里的模块内方案）',
        'square+reduce 与 rsqrt+scale 拆成两段',
        '回退原生实现',
      ],
      evidence: { status: 'experience', refs: [SOURCE + ' §9 FUS-NORM-002'] },
    },

    {
      id: 'cand-sparse-attn',
      ruleId: 'FUS-ATTN-001',
      category: 'attention',
      priority: 'P0',
      title: '稀疏 Attention 主链编排',
      pattern: 'cache read → sparse QKᵀ/softmax/PV → RoPE inverse',
      modelContext: 'transformer_block.attention.sparse_core',
      roles: ['attention_core', 'state_update'],
      site: { module: 'sparse-attn', steps: [2, 3, 4] },
      level: 'conditional',
      shape: 'stage_fusion',
      mode: 'orchestrated_subgraph',
      pyptoStatus: 'not_checked',
      materialization: 'eliminate',
      consumers: { multi: false, crossStage: true },
      boundary: {
        start: 'top-k 索引 + KV cache 读取',
        end: 'attention 输出（反旋转后）',
        preserve: ['KV cache 本身（跨请求状态，属于融合边界）'],
        shared: ['压缩块 KV 同时被 indexer 与 compressor 使用'],
      },
      semantics: '文档 §10 把完整 Attention 主链定为高收益高复杂度。这里选择只融合'
        + '「取 KV → 稀疏注意力 → 反旋转」这一段，把 top-k 选块留在边界外：'
        + 'top-k ' + c.index_topk + ' 是动态稀疏控制，整体单 kernel 会命中 §4.3 的阻断条件。',
      benefits: [
        'score / masked score / probability 三个大中间矩阵不落 GM',
        'online / block softmax 可以在选中的 KV block 上流式完成',
        '选中的 KV tile 在局部循环内复用',
      ],
      requirements: [
        '必须分别识别 prefill 与 decode：两者的 KV 读取形态完全不同',
        'top-k 索引在进入融合边界前已确定，边界内不做动态选块',
        'window ' + c.window_size + ' 与压缩块两路 KV 的 layout 统一',
        'attn_sink 与 softmax_scale 的处理位置固定',
      ],
      risks: [
        'softmax 的 max/sum 状态需要跨 block 合并，归约顺序变化带来数值差异',
        'decode 场景下 KV cache 访问可能比计算更占主导，融合收益被访存掩盖',
        'head_dim ' + c.head_dim + ' 偏大，片上驻留压力高',
      ],
      traffic: {
        gm: gmSaved(c.index_topk * c.window_size / 4, BF16, 2),
        gmNote: 'score 与 probability 两个中间矩阵 · 按 top-k 选中块估算',
        launchBefore: 5,
        launchAfter: 2,
        liveSet: c.head_dim * BF16 * 8,
        liveNote: 'Q tile + KV block + softmax 累加器',
        syncAdded: 3,
        syncNote: 'QK（Cube）↔ softmax（Vector）↔ PV（Cube）多次交接',
      },
      invocations: c.n_layers,
      invocationNote: '每层一次',
      factors: {
        prefill: { gm: 5, launch: 2, reuse: 5, sync: 3, onchip: 4, parallel: 2, numeric: 3, complexity: 5 },
        decode: { gm: 2, launch: 3, reuse: 2, sync: 3, onchip: 2, parallel: 2, numeric: 3, complexity: 5 },
      },
      integrationCost: 4,
      validationCost: 4,
      validation: {
        graph: '固定 top-k 索引下融合前后输出一致；确认 causal mask 语义不变',
        kernel: '长序列、极值、attn_sink 边界；block softmax 与整体 softmax 的误差对比',
        perf: 'prefill 长序列与 decode 小 batch 分别 benchmark，不能互相外推',
        e2e: 'TTFT 与 TPOT 分别观察，确认 KV cache 命中行为未变',
      },
      fallback: [
        '只融合 softmax + PV（文档 FUS-ATTN-003）',
        '只融合 QKᵀ + scale + mask（文档 FUS-ATTN-002）',
        '保留四段编排，只统一 buffer 与依赖',
      ],
      evidence: { status: 'hypothesis', refs: ['repo/pto/examples/models/03_flash_attention.py', SOURCE + ' §10'] },
    },

    {
      id: 'cand-compress-cache',
      ruleId: 'FUS-KV-001',
      category: 'kvcache',
      priority: 'P1',
      title: '压缩 KV 归一化 + RoPE + 量化写入',
      pattern: 'gated pooling → RMSNorm → RoPE → quant → compressed cache write',
      modelContext: 'transformer_block.attention.compressor',
      roles: ['normalization', 'state_update'],
      site: { module: 'compressor', steps: [3, 4, 5] },
      level: 'conditional',
      shape: 'stage_fusion',
      mode: 'custom_fused_op',
      pyptoStatus: 'not_checked',
      materialization: 'eliminate',
      consumers: { multi: false, crossStage: true },
      boundary: {
        start: 'gated pooling 之后的压缩 KV 候选',
        end: '压缩 KV cache 写入完成',
        preserve: ['压缩 KV cache（跨 token 复用的状态）'],
        shared: ['压缩块同时被 indexer 打分与 sparse_attn 读取'],
      },
      semantics: 'npu_mla_prolog_v3 接管的是主 KV cache 写入；compressor 这条写的是 CSA/HCA 的'
        + '压缩 cache，两者互不重叠。文档 §11 FUS-KV-001 正是这个形态。',
      benefits: [
        '归一化、旋转、量化三步的中间张量都不落 GM，直接以最终 dtype 写 cache',
        '写 cache 的地址计算与量化 scale 可以共用一次索引',
      ],
      requirements: [
        'cache 写入地址连续性明确；paged 形态下需要 block table',
        '量化 scale 的粒度与压缩块边界对齐（FP8 · ue8m0）',
        '写入不跨 rank / stream，否则命中通信边界阻断',
        'compress_ratio 4 与 128 两种层需要各自的静态特化',
      ],
      risks: [
        '量化位置改变会改变 cache 中的数值，影响后续所有 token 的注意力结果',
        '状态写入属于副作用，融合边界内不得重排',
      ],
      traffic: {
        gm: gmSaved(c.dim / 4, FP8, 2),
        gmNote: '归一化输出与旋转输出两次物化 · 压缩后维度 × FP8',
        launchBefore: 3,
        launchAfter: 1,
        liveSet: (c.dim / 4) * 4,
        liveNote: '压缩块 tile + FP32 归约累加器',
        syncAdded: 1,
        syncNote: 'Vector 内部完成，仅一次 cache 写入 fence',
      },
      invocations: 41,
      invocationNote: '压缩层 41 层（compress_ratio 非 0 的层）',
      factors: {
        prefill: { gm: 4, launch: 2, reuse: 2, sync: 2, onchip: 1, parallel: 1, numeric: 4, complexity: 3 },
        decode: { gm: 3, launch: 3, reuse: 2, sync: 2, onchip: 1, parallel: 1, numeric: 4, complexity: 3 },
      },
      integrationCost: 3,
      validationCost: 4,
      validation: {
        graph: '融合前后压缩 cache 的内容逐元素一致，确认写入时序不变',
        kernel: 'FP8 量化边界值；compress_ratio 4 与 128 两种层分别验证',
        perf: '对比三算子编排与融合写入的 cache 更新耗时',
        e2e: '长上下文下压缩块命中率与注意力输出漂移',
      },
      fallback: [
        '只融合 RMSNorm + RoPE，量化与写入保持独立',
        '沿用 npu_kv_rmsnorm_rope_cache（替换方案面板里的对应方案）',
        '回退原生实现',
      ],
      evidence: { status: 'experience', refs: [SOURCE + ' §11 FUS-KV-001'] },
    },

    {
      id: 'cand-out-proj',
      ruleId: 'FUS-MM-001',
      category: 'matmul_epilogue',
      priority: 'P0',
      title: '分组输出投影的 reshape + wo_a einsum',
      pattern: 'reshape to groups → grouped einsum（bsgd,grd→bsgr）',
      modelContext: 'transformer_block.attention.out_projection',
      roles: ['projection'],
      site: { module: 'out-proj', steps: [0, 1] },
      level: 'conditional',
      shape: 'local_dataflow',
      mode: 'composite_op',
      pyptoStatus: 'not_checked',
      materialization: 'eliminate',
      consumers: { multi: false, crossStage: false },
      boundary: {
        start: 'sparse attention 输出',
        end: 'wo_a 分组结果（进入 wo_b 之前）',
        preserve: [],
        shared: [],
      },
      semantics: '整段 out_projection 是两个串联的重量级 MatMul，文档 §4.3 明确把'
        + '「多个重量级 MatMul 且没有明确 blocking/streaming 方案」列为默认阻断。'
        + '所以候选只取前半段：reshape 是纯 layout 变换，可以吃进 wo_a 的 tile 装载路径。',
      benefits: [
        'reshape 只改变 layout，可以在 einsum 的 tile 装载阶段完成，省掉一次完整搬运',
        '分组维度 ' + c.o_groups + ' 与 rank ' + c.o_lora_rank + ' 静态，tile 划分稳定',
      ],
      requirements: [
        '分组后的 layout 能被 einsum 的 tile 装载直接消费',
        'wo_a 输出 dtype 与 accumulation dtype 明确',
        'wo_b 保持独立算子，不纳入本候选边界',
      ],
      risks: [
        'layout 转换若无法融进装载路径，收益归零（LAYOUT_CONFLICT）',
        '收益只来自一次搬运，接近文档 §4.3「收益只来自一次很小的搬运」的阻断线',
      ],
      traffic: {
        gm: gmSaved(c.n_heads * c.head_dim / c.o_groups, BF16, 1),
        gmNote: 'reshape 后的中间张量一次物化',
        launchBefore: 2,
        launchAfter: 1,
        liveSet: c.o_lora_rank * BF16 * c.o_groups,
        liveNote: '分组 tile 驻留',
        syncAdded: 0,
        syncNote: '同在 Cube 侧，无额外交接',
      },
      invocations: c.n_layers,
      invocationNote: '每层一次',
      factors: {
        prefill: { gm: 2, launch: 1, reuse: 2, sync: 0, onchip: 2, parallel: 1, numeric: 0, complexity: 2 },
        decode: { gm: 1, launch: 2, reuse: 1, sync: 0, onchip: 1, parallel: 1, numeric: 0, complexity: 2 },
      },
      integrationCost: 1,
      validationCost: 1,
      validation: {
        graph: '融合前后 wo_a 输出一致',
        kernel: '不同 batch/seq 下的 layout 正确性',
        perf: '确认收益不是被 layout 转换吃掉',
        e2e: '整网 attention 段耗时变化',
      },
      fallback: [
        '保持 reshape 与 einsum 两个算子，只统一 layout 约定',
        '回退原生实现',
      ],
      evidence: { status: 'hypothesis', refs: [SOURCE + ' §8 FUS-MM-001'] },
    },

    {
      id: 'cand-indexer-score',
      ruleId: 'FUS-ELE-001',
      category: 'elementwise',
      priority: 'P0',
      title: 'indexer 打分尾链',
      pattern: 'einsum score → relu × weights → top-k + causal mask',
      modelContext: 'transformer_block.attention.indexer',
      roles: ['activation', 'routing'],
      site: { module: 'indexer', steps: [4, 5] },
      level: 'conditional',
      shape: 'local_dataflow',
      mode: 'composite_op',
      pyptoStatus: 'not_checked',
      materialization: 'eliminate',
      consumers: { multi: false, crossStage: false },
      boundary: {
        start: 'indexer 打分矩阵（einsum q·k 输出）',
        end: 'top-k ' + c.index_topk + ' 压缩块索引',
        preserve: ['top-k 索引（sparse_attn 的输入）'],
        shared: [],
      },
      semantics: 'relu、加权归约与 causal mask 都是逐元素/局部归约，紧接一次 top-k。'
        + '文档 §7 的逐元素链默认强推荐，但这里尾部带 top-k，降为条件推荐。',
      benefits: [
        'relu、权重乘、mask 三步在一个 Vector tile 内完成',
        '打分矩阵不必以完整形态落 GM，只输出 top-k 索引',
      ],
      requirements: [
        'top-k = ' + c.index_topk + ' 静态；causal mask 可以按 block 生成',
        '打分矩阵没有其他消费者（当前只喂 top-k）',
        'FP32 打分与 FP4 量化路径的边界明确',
      ],
      risks: [
        'top-k 的 tie-breaking 决定选中哪些 KV block，属于可观察输出',
        '打分矩阵随序列增长，片上驻留需要 block 化',
      ],
      traffic: {
        gm: gmSaved(c.index_n_heads * c.index_topk, 4, 2),
        gmNote: 'relu 前后两次物化 · 打分矩阵 × FP32',
        launchBefore: 3,
        launchAfter: 1,
        liveSet: c.index_topk * 4 * 4,
        liveNote: '打分 block + top-k 工作区',
        syncAdded: 1,
        syncNote: 'Cube（einsum）→ Vector（打分链）一次交接',
      },
      invocations: 21,
      invocationNote: 'indexer 只在 compress_ratio = 4 的 21 层存在',
      factors: {
        prefill: { gm: 3, launch: 2, reuse: 2, sync: 1, onchip: 3, parallel: 1, numeric: 2, complexity: 2 },
        decode: { gm: 2, launch: 3, reuse: 2, sync: 1, onchip: 2, parallel: 1, numeric: 2, complexity: 2 },
      },
      integrationCost: 2,
      validationCost: 3,
      validation: {
        graph: '融合前后选中的压缩块索引完全一致',
        kernel: '构造打分相等的输入，验证 tie-breaking',
        perf: '长序列下 block 化打分的耗时',
        e2e: '压缩块命中分布与注意力质量',
      },
      fallback: [
        '只融合 relu × weights，top-k 保持独立',
        '多算子编排，共享打分 block 布局',
        '回退原生实现',
      ],
      evidence: { status: 'hypothesis', refs: [SOURCE + ' §7 FUS-ELE-001'] },
    },

    {
      id: 'cand-embed-mask',
      ruleId: 'FUS-ELE-002',
      category: 'elementwise',
      priority: 'P0',
      title: 'Embedding 分片掩码链',
      pattern: 'mask token ids → F.embedding → zero out remote rows',
      modelContext: 'model.embedding',
      roles: ['activation'],
      site: { module: 'embedding-op', steps: [0, 1, 2] },
      level: 'strong',
      shape: 'local_dataflow',
      mode: 'custom_fused_op',
      pyptoStatus: 'not_checked',
      materialization: 'eliminate',
      consumers: { multi: false, crossStage: false },
      boundary: {
        start: 'input_ids',
        end: '本 rank 的 embedding 结果（AllReduce 之前）',
        preserve: [],
        shared: [],
      },
      semantics: '掩码、查表、置零三步是标准的逐元素短链，文档 §7 默认强推荐。'
        + '但它整网只执行一次 —— 正好演示文档 §5.5 的要求：局部收益要乘 invocation_count 才是整网贡献。',
      benefits: [
        '掩码与置零可以在查表的写出路径上完成，省掉两次完整 hidden 张量往返',
        '3 次 launch 降为 1 次',
      ],
      requirements: [
        '掩码与置零作用于同一 logical shape',
        '中间结果没有额外消费者',
        'AllReduce 保持在融合边界之外',
      ],
      risks: [
        '整网只调用一次，收益乘不上层数',
        'vocab 分片策略变化会改变掩码逻辑',
      ],
      traffic: {
        gm: gmSaved(c.dim, BF16, 2),
        gmNote: '掩码后 ids 与查表输出两次物化',
        launchBefore: 3,
        launchAfter: 1,
        liveSet: c.dim * BF16,
        liveNote: '单 token hidden tile',
        syncAdded: 0,
        syncNote: '全在 Vector 侧',
      },
      invocations: 1,
      invocationNote: '整网一次 —— net_benefit 会把它排到后面',
      factors: {
        prefill: { gm: 3, launch: 3, reuse: 1, sync: 0, onchip: 1, parallel: 0, numeric: 0, complexity: 1 },
        decode: { gm: 2, launch: 3, reuse: 1, sync: 0, onchip: 1, parallel: 0, numeric: 0, complexity: 1 },
      },
      integrationCost: 1,
      validationCost: 1,
      validation: {
        graph: '融合前后 embedding 输出一致，含非本 rank token 的置零行为',
        kernel: '越界 / 边界 token id',
        perf: '单次调用的 launch 收益',
        e2e: '基本无整网可观测变化，作为低成本收口',
      },
      fallback: [
        'mask + zero out 合并，查表保持独立',
        '回退原生实现',
      ],
      evidence: { status: 'validated', refs: [SOURCE + ' §7 FUS-ELE-002'] },
    },
  ];

  /* ---------------- 未推荐（文档 §19 原因码） ---------------- */

  const DEFERRED = [
    {
      id: 'defer-swiglu-down',
      ruleId: 'FUS-FFN-003',
      title: 'SwiGLU + down projection 整体融合',
      pattern: 'SwiGLU → w2 down projection',
      site: { module: 'shared-expert', steps: [0, 1, 2, 3, 4] },
      level: 'cautious',
      reasons: ['PARALLELISM_LOSS', 'MEMORY_PRESSURE'],
      why: '两个重量级 MatMul 之间夹一次逐元素乘法。文档 §12 FUS-FFN-003 默认谨慎推荐，'
        + '只有存在明确的分块矩阵乘与流式消费方案时才提升等级。',
      next: '先落地 cand-swiglu，拿到 intermediate ' + c.moe_inter_dim + ' 的实测驻留数据后再评估是否把 w2 纳入',
      mode: 'orchestrated_subgraph',
    },
    {
      id: 'defer-mhc',
      ruleId: '—',
      title: 'mHC Sinkhorn 混合链',
      pattern: 'flatten → mixing logits → Sinkhorn ×' + c.hc_sinkhorn_iters + ' → pre/post 加权',
      site: { module: 'mhc', steps: [1, 2, 3] },
      level: 'cautious',
      reasons: ['NUMERICAL_RISK', 'REDUCTION_CONFLICT'],
      why: 'Sinkhorn 是 ' + c.hc_sinkhorn_iters + ' 次固定迭代的行列归一化，融合会改变归约顺序。'
        + '文档 §23 允许把带循环累加器的子图作为模型阶段候选，但要重点评估状态复用，'
        + '不能按普通逐元素链处理。',
      next: '先做归约顺序敏感性实验；若误差可控，按 orchestrated_subgraph 交付而不是单 kernel',
      mode: 'orchestrated_subgraph',
    },
    {
      id: 'defer-lmhead-comm',
      ruleId: 'FUS-COMM-001',
      title: 'vocab projection + AllGather logits',
      pattern: 'MatMul(hidden → vocab) → AllGather',
      site: { module: 'lm-head', steps: [2, 3] },
      level: 'cautious',
      reasons: ['COMMUNICATION_BOUNDARY'],
      why: '文档 §15 要求先判断「是否可以重叠」，而不是判断「是否可以合成一个 kernel」。'
        + 'vocab ' + c.vocab_size.toLocaleString('en-US') + ' 的分片 logits 跨 rank 汇聚，'
        + '融合会隐藏通信失败与超时的定位信息。',
      next: '按 overlap_schedule 评估：vocab 分块产出 + 分块 AllGather 重叠',
      mode: 'overlap_schedule',
    },
    {
      id: 'defer-attn-whole',
      ruleId: 'FUS-ATTN-001',
      title: '稀疏 Attention 整体单 kernel（含 top-k 选块）',
      pattern: 'topk 选块 → cache read → sparse attention → RoPE inverse',
      site: { module: 'sparse-attn', steps: [0, 1, 2, 3, 4] },
      level: 'reject',
      reasons: ['DYNAMIC_SHAPE_UNBOUNDED', 'MEMORY_PRESSURE'],
      why: '文档 §10「不应直接整体融合的情况」明确列出：mask 或 top-k 引入复杂动态稀疏控制。'
        + 'top-k ' + c.index_topk + ' 决定了 KV 读取范围，融合后没有稳定的 tile shape 上界。',
      next: '拆成 cand-sparse-attn（选块留在边界外）已在推荐列表中',
      mode: 'orchestrated_subgraph',
    },
    {
      id: 'defer-residual-norm',
      ruleId: 'FUS-NORM-001',
      title: 'ResidualAdd + RMSNorm',
      pattern: 'residual + hidden → RMSNorm',
      site: null,
      level: 'reject',
      reasons: ['LIBRARY_BASELINE_STRONG'],
      why: '这条 P0 规则在本模型上不成立：残差已被 mHC 替代（不是一次简单 add），'
        + '层内的 pre-norm 又已经被 npu_mla_prolog_v3 吃掉。图上找不到 residual → norm 这个模式。',
      next: '若后续 mHC 退化为普通残差，此规则重新生效',
      mode: null,
    },
    {
      id: 'defer-qkv-rope',
      ruleId: 'FUS-QKV-002 / FUS-KV-001',
      title: 'Q/K projection + RoPE + 主 KV cache 写入',
      pattern: 'projection → split → RoPE → cache write',
      site: { module: '__npu_mod-mla-prolog__', steps: [] },
      level: 'reject',
      reasons: ['LIBRARY_BASELINE_STRONG'],
      why: '这段已经由 npu_mla_prolog_v3 覆盖。文档 §4.3 把「目标算子已有成熟高性能库实现」'
        + '列为默认阻断条件 —— 再做一个自定义融合算子没有净收益。',
      next: '维持 torch_npu 实现；若该算子的约束核对不通过，再回到这条规则',
      mode: null,
    },
    {
      id: 'defer-moe-expert',
      ruleId: 'FUS-MOE-003',
      title: 'Expert GEMM + Combine',
      pattern: 'dispatch → grouped expert GEMM → weighted combine',
      site: { module: '__npu_mod-moe-grouped__', steps: [] },
      level: 'reject',
      reasons: ['LIBRARY_BASELINE_STRONG', 'COMMUNICATION_BOUNDARY'],
      why: '已由 npu_grouped_matmul + npu_moe_finalize_routing 覆盖；跨 rank 部分由'
        + ' npu_moe_distribute_dispatch_v2 / combine_v2 承担。文档 §14 本身也只对跨 rank 整体融合给谨慎推荐。',
      next: '跨 rank AllToAll 与 expert compute 的重叠（overlap_schedule）可作为后续课题',
      mode: 'overlap_schedule',
    },
  ];

  /* ---------------- 评分（文档 §5.3 / §5.4 / §5.5） ---------------- */

  function scoreOf(candidate, scenarioId) {
    const f = candidate.factors[scenarioId] || candidate.factors.prefill;
    const terms = WEIGHTS.map((w) => ({
      key: w.key,
      label: w.label,
      weight: w.sign * w.w,
      factor: f[w.key] || 0,
      value: w.sign * w.w * (f[w.key] || 0),
    }));
    return { terms, total: terms.reduce((s, t) => s + t.value, 0) };
  }

  /** §5.5 整网收益聚合 */
  function netBenefitOf(candidate, scenarioId, score) {
    const scen = SCENARIOS[scenarioId];
    const per = score.total;
    // 文档 §5.5 的 net_benefit。这里的因子是序数不是 bytes/次数，
    // 所以不把 token 数乘进来假装成物理量 —— execution_weight 只作为场景说明展示。
    const net = per * candidate.invocations - candidate.integrationCost - candidate.validationCost;
    return {
      perInvocation: per,
      invocations: candidate.invocations,
      execWeight: scen.tokens,
      execNote: scen.weightNote,
      integrationCost: candidate.integrationCost,
      validationCost: candidate.validationCost,
      net: Math.round(net),
    };
  }

  function evaluate(scenarioId) {
    const scenario = SCENARIOS[scenarioId] ? scenarioId : 'prefill';
    const items = CANDIDATES.map((x) => {
      const score = scoreOf(x, scenario);
      const netBenefit = netBenefitOf(x, scenario, score);
      return {
        ...x,
        levelMeta: LEVELS[x.level],
        shapeMeta: SHAPES[x.shape],
        modeMeta: MODES[x.mode],
        pyptoLabel: PYPTO_STATUS[x.pyptoStatus],
        score,
        netBenefit,
        // §5.4：缺 profile 与 capability 信息时分数只能是 provisional
        provisional: true,
      };
    }).sort((a, b) => b.netBenefit.net - a.netBenefit.net);

    const deferred = DEFERRED.map((x) => ({ ...x, levelMeta: LEVELS[x.level] }))
      .sort((a, b) => a.levelMeta.rank - b.levelMeta.rank);

    const summary = {
      scenario,
      total: items.length,
      strong: items.filter((x) => x.level === 'strong').length,
      conditional: items.filter((x) => x.level === 'conditional').length,
      deferred: deferred.length,
      launchSaved: items.reduce((s, x) => s + (x.traffic.launchBefore - x.traffic.launchAfter) * x.invocations, 0),
      gmSaved: items.reduce((s, x) => s + x.traffic.gm * x.invocations, 0),
      modes: MODES,
    };

    return { scenario, scenarioMeta: SCENARIOS[scenario], items, deferred, summary };
  }

  window.PtoFusionCatalog = {
    SOURCE,
    DOC_DATE,
    CONFIG,
    LEVELS,
    ACTIONABLE,
    SHAPES,
    MODES,
    PYPTO_STATUS,
    REASONS,
    WEIGHTS,
    SCENARIOS,
    GATES,
    BASELINES,
    evaluate,
  };
})();
