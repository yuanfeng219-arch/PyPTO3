/* openPangu-Ultra-MoE-718B-V1.1 计算轴架构图 · 图数据 builder（SPEC §3）
   只画健康拓扑。布局对齐 model-graphviz / DeepSeek V3.2：单列从上到下、禁止折叠模块间多条平行代表边。

   设计约定（本轮重构）：
   - 节点 label 只用专业/学术名称；解释性说明放 node.desc（页面侧 hover tip 展示），不进 label。
   - 路由专家 / 共享专家 = 块内左右并列两条泳道，共同汇入 All-to-All 汇聚（DeepSeek moe_combine 范式）。
   - 连线几何：引擎按 |Δy| >= |Δx| 判定竖直（pattern.js:323）→ 所有前向边保证竖直落差 >= 水平偏移，
     从顶/底进出，不连侧面；权重边（dashed parameter）允许从侧面进。
   - 两级折叠：L1 moeExpanded（MoE 层折叠盒 ↔ 展开内部）；L2 expertExpanded（路由专家代表节点 ↔ 32 EP 桶网格）。
   - 256 路由专家 = 运行时 32 EP 桶 × 8（每桶 8 专家）。EP 桶是「这一层 256 专家的运行时切分」，
     不是 32 个层、也不等于 32 张卡 —— 展开态用点阵（每格 8 点）+ caption 可视化。
   - All-to-All 只用代表边；需抑制标签的通信边用 tag:' '（引擎 trim 后为空 → 跳过绘制）。 */
window.buildUltraMoE718BGraph = (function () {
  const CX = 620, COMM = ' ';  // COMM = 抑制标签的空格

  // ── 输入行（收进「模型输入」parent cluster，居中对齐主轴）──
  const inputs = [
    { id: 'token_ids',    label: 'Token IDs',      typeLabel: 'Input', kind: 'tensor', x: 420, y: 88, width: 150, height: 46, colorKey: 'io:input', desc: '输入 token 序列的整数 ID（词表索引）。' },
    { id: 'position_ids', label: 'Position IDs',   typeLabel: 'Input', kind: 'tensor', x: 620, y: 88, width: 150, height: 46, colorKey: 'io:input', desc: '位置索引，供 RoPE 旋转位置编码使用。' },
    { id: 'attn_mask',    label: 'Attention Mask', typeLabel: 'Input', kind: 'tensor', x: 820, y: 88, width: 160, height: 46, colorKey: 'io:input', desc: '注意力掩码，标记可见 / 填充位置。' },
  ];
  const inputsCluster = { id: 'inputs_group', label: 'Model Inputs', x: 258, y: 44, width: 736, height: 96, colorKey: 'module:model' };

  // ── 主干上半：Embedding → Dense×3 ──
  const trunkTop = [
    { id: 'embedding',   label: 'Parallel Embedding', typeLabel: 'Op', kind: 'op', x: CX, y: 210, width: 250, height: 56, colorKey: 'sem:embedding', desc: 'Parallel Embedding：按词表维切分的词嵌入查表，输出 hidden_states。' },
    { id: 'dense_block', label: 'Dense Decoder Layers × 3', typeLabel: 'Module', kind: 'op', x: CX, y: 308, width: 320, height: 60, colorKey: 'module:decoder', desc: '前 3 层稠密解码层（num_dense_layers=3）：MLA 注意力 + 稠密 SwiGLU MLP，不做专家路由。' },
    { id: 'embedding_weight', label: 'Embedding Weight', typeLabel: 'Parameter', kind: 'tensor', x: 270, y: 210, width: 188, height: 48, colorKey: 'io:parameter', desc: '词嵌入权重矩阵 [vocab, hidden]；V1.1 中 tie_word_embeddings=false，LM Head 权重单独存在。' },
  ];
  const trunkTopEdges = [
    { source: 'token_ids',    target: 'embedding',  tag: 'IDS', edgeType: 'activation' },
    { source: 'embedding_weight', target: 'embedding', tag: 'W', edgeType: 'parameter', dashed: true },
    { source: 'embedding',    target: 'dense_block', tag: 'ACT', edgeType: 'activation' },
  ];

  // ── 主干下半：Final RMSNorm → LM Head → conditional Logits All-Gather → Logits ──
  function trunkBottom(yTop) {
    const yNorm = yTop, yHead = yTop + 118, yGather = yTop + 226, yOut = yTop + 334;
    return {
      nodes: [
        { id: 'final_norm', label: 'Final RMSNorm', typeLabel: 'Op',     kind: 'op',     x: CX, y: yNorm,   width: 210, height: 54, colorKey: 'sem:norm', desc: 'Final RMSNorm：输出投影前的最终归一化。' },
        { id: 'lm_head',    label: 'LM Head',        typeLabel: 'Op',     kind: 'op',     x: CX, y: yHead,   width: 200, height: 54, colorKey: 'sem:head', desc: 'LM Head：hidden → vocab logits 的线性投影。' },
        { id: 'logits_allgather', label: 'Logits All-Gather', typeLabel: 'Comm', kind: 'op', x: CX, y: yGather, width: 230, height: 50, colorKey: 'sem:comm', desc: 'Logits All-Gather：embed_tp_size > 1 时跨词表分片收集 logits；并行度为 1 时等价 no-op。' },
        { id: 'logits',     label: 'Logits',         typeLabel: 'Output', kind: 'tensor', x: CX, y: yOut,    width: 170, height: 48, colorKey: 'io:output', desc: '下一个 token 的预测分布（vocab 维 logits）。' },
        { id: 'lm_head_weight', label: 'LM Head Weight', typeLabel: 'Parameter', kind: 'tensor', x: 335, y: yHead, width: 188, height: 48, colorKey: 'io:parameter', desc: 'LM Head 权重 [hidden, vocab]；V1.1 中 tie_word_embeddings=false，不与 Embedding 权重共享。' },
      ],
      edges: [
        { source: 'final_norm', target: 'lm_head',    tag: 'NEXT', edgeType: 'activation' },
        { source: 'lm_head',    target: 'logits_allgather', tag: 'gather', edgeType: 'communication' },
        { source: 'logits_allgather', target: 'logits', tag: 'LOGITS', edgeType: 'activation' },
        { source: 'lm_head_weight', target: 'lm_head', tag: 'W', edgeType: 'parameter', dashed: true },
      ],
    };
  }

  // ── 折叠态：MoE 解码层 = 一个可展开盒 ──
  function buildCollapsed() {
    const moeBlock = { id: 'moe_block', label: 'MoE Decoder Layers × 58', typeLabel: 'Module', kind: 'op', x: CX, y: 408, width: 320, height: 64, colorKey: 'module:moe', collapsed: true, desc: '中间 58 层 MoE 解码层（每层：MLA 注意力 → Router Top-8 → 256 路由专家 + 1 共享专家 → 合并）。点 + 展开代表层内部。' };
    const bottom = trunkBottom(540);
    const nodes = [...inputs, ...trunkTop, moeBlock, ...bottom.nodes];
    const edges = [
      ...trunkTopEdges,
      { source: 'position_ids', target: 'dense_block', tag: 'POS',  edgeType: 'activation' },
      { source: 'attn_mask',    target: 'dense_block', tag: 'MASK', edgeType: 'activation' },
      { source: 'dense_block',  target: 'moe_block',   tag: 'ACT',  edgeType: 'activation' },
      { source: 'moe_block',    target: 'final_norm',  tag: 'ACT',  edgeType: 'activation' },
      ...bottom.edges,
    ];
    return { width: 1160, height: 836, clusters: [inputsCluster], nodes, edges };
  }

  // ── 展开态：MoE 代表层单列 · 路由/共享并列 · 专家可再下钻 ──
  function buildExpanded(expertExpanded) {
    // MoE 列上半（两种专家态共用），主干中线 CX
    const headNodes = [
      { id: 'mla',         label: 'MLA Attention', typeLabel: 'Module', kind: 'op', x: CX, y: 430, width: 300, height: 56, colorKey: 'sem:attention', desc: 'Multi-head Latent Attention：对 Q / KV 做低秩压缩（q_lora=1536，kv_lora=512）以压缩 KV cache。' },
      { id: 'moe_prenorm', label: 'Pre-RMSNorm',  typeLabel: 'Op', kind: 'op', x: CX, y: 512, width: 200, height: 50, colorKey: 'sem:norm', desc: '进入 MoE 前的 RMSNorm（sandwich-norm 的前半）。' },
      { id: 'gate',        label: 'Router Top-8', typeLabel: 'Op', kind: 'op', x: CX, y: 602, width: 200, height: 52, colorKey: 'sem:gate', desc: 'Router 门控：为每个 token 打分选 Top-8 专家；routed_scaling=2.5，norm_topk_prob 归一化；负载均衡损失约束各专家利用率。' },
      { id: 'w_gate',      label: 'W_gate', typeLabel: 'Parameter', kind: 'tensor', x: 360, y: 602, width: 120, height: 46, colorKey: 'io:parameter', desc: 'Router 权重 [hidden, 256]，给 256 个路由专家逐一打分。' },
      { id: 'a2a_dispatch',label: 'All-to-All Dispatch', typeLabel: 'Comm', kind: 'op', x: CX, y: 694, width: 230, height: 50, colorKey: 'sem:comm', desc: 'All-to-All 分发：按路由结果把每个 token 发往承载目标专家的 expert-parallel rank（通信事件，投影到物理轴）。' },
    ];
    const headEdges = [
      ...trunkTopEdges,
      { source: 'position_ids', target: 'mla', tag: 'POS',  edgeType: 'activation' },
      { source: 'attn_mask',    target: 'mla', tag: 'MASK', edgeType: 'activation' },
      { source: 'dense_block',  target: 'mla', tag: 'ACT',  edgeType: 'activation' },
      { source: 'mla',          target: 'moe_prenorm', tag: 'ACT',  edgeType: 'activation' },
      { source: 'moe_prenorm',  target: 'gate',        tag: 'NORM', edgeType: 'activation' },
      { source: 'w_gate',       target: 'gate',        tag: 'W',    edgeType: 'parameter', dashed: true },
      { source: 'gate',         target: 'a2a_dispatch',tag: 'dispatch', edgeType: 'communication' },
    ];

    const sharedDesc = '共享专家（SwiGLU MLP）：始终激活，不经 Router，每个 token 都会经过，作为稳定基座，与 256 路由专家的输出相加。';
    let expertNodes, expertEdges, clusters, yCombine, width, gridMeta = null;

    if (!expertExpanded) {
      // 专家折叠：路由代表节点（中线）｜ 共享专家（右并列），共同汇入 Combine
      yCombine = 1030;
      expertNodes = [
        { id: 'expert_mlp',    label: 'Routed Expert Bank', typeLabel: '256 experts', kind: 'op', x: CX,  y: 806, width: 300, height: 62, colorKey: 'sem:moe', collapsed: true, desc: '256 个路由专家（SwiGLU MLP）。每个 token 仅激活 Top-8；运行时切成 32 个 EP 桶、每桶 8 专家。点 + 展开为 32 EP 桶点阵。' },
        { id: 'shared_expert', label: 'Shared Expert MLP', typeLabel: 'Module', kind: 'op', x: 890, y: 806, width: 270, height: 62, colorKey: 'sem:mlp', desc: sharedDesc },
      ];
      expertEdges = [
        { source: 'a2a_dispatch',  target: 'expert_mlp',   tag: COMM,  edgeType: 'communication' },
        { source: 'expert_mlp',    target: 'a2a_combine',  tag: COMM,  edgeType: 'communication' },
        { source: 'moe_prenorm',   target: 'shared_expert',tag: 'ACT', edgeType: 'activation' },
        { source: 'shared_expert', target: 'a2a_combine',  tag: 'ACT', edgeType: 'activation' },
      ];
      clusters = [inputsCluster, { id: 'moe_layer', label: 'MoE Decoder Layer (representative · ×57 folded)', x: 260, y: 400, width: 720, height: 760, colorKey: 'module:moe', repeat: 58 }];
      width = 1160;
    } else {
      // 专家展开：256 路由专家 = 32 EP 桶 × 8（点阵在页面侧注入）｜ 共享专家右并列
      const COLS = 8, ROWS = 4, COLP = 58, ROW0 = 812, ROWP = 50, GW = 54, GH = 40;
      const colCenters = []; for (let c = 0; c < COLS; c++) colCenters.push(CX - (COLS - 1) * COLP / 2 + c * COLP);
      const groups = [];
      for (let i = 0; i < 32; i++) {
        const col = i % COLS, row = Math.floor(i / COLS), ep = String(i).padStart(2, '0');
        groups.push({ id: 'expert_group_' + ep, label: 'EP' + ep, typeLabel: '8 experts', kind: 'op',
          x: colCenters[col], y: ROW0 + row * ROWP, width: GW, height: GH, colorKey: 'sem:moe', epRank: ep,
          desc: 'EP 桶 ' + ep + '：承载 256 个路由专家中的 8 个，放在一组 expert-parallel rank 上。EP 桶是这一层 256 专家的运行时切分 —— 不是一层，也不是一张卡。' });
      }
      gridMeta = { noteY: 742, splitLine: true };  // 常驻「≠层≠卡」说明 + dispatch/combine 竖线在专家池分两段（页面侧注入）
      yCombine = 1235;
      expertNodes = [
        ...groups,
        { id: 'shared_expert', label: 'Shared Expert MLP', typeLabel: 'Module', kind: 'op', x: 990, y: 900, width: 270, height: 62, colorKey: 'sem:mlp', desc: sharedDesc },
        { id: 'w_expert',      label: 'Expert Weights', typeLabel: 'Parameter', kind: 'tensor', x: 300, y: 900, width: 170, height: 50, colorKey: 'io:parameter', desc: '256 个路由专家的权重组（每专家一组 SwiGLU 投影），按 EP 桶切分放置。' },
      ];
      expertEdges = [
        // dispatch→pool→combine 的居中竖线在页面侧分两段画（进/出专家池），此处不发引擎边以免穿过网格
        { source: 'w_expert',      target: 'expert_group_16', tag: 'W', edgeType: 'parameter', dashed: true },
        { source: 'moe_prenorm',   target: 'shared_expert',tag: 'ACT', edgeType: 'activation' },
        { source: 'shared_expert', target: 'a2a_combine',  tag: 'ACT', edgeType: 'activation' },
      ];
      clusters = [
        inputsCluster,
        { id: 'moe_layer', label: 'MoE Decoder Layer (representative · ×57 folded)', x: 240, y: 400, width: 840, height: 970, colorKey: 'module:moe', repeat: 58 },
        { id: 'expert_pool', label: 'Routed Expert Pool · 256 experts → 32 EP buckets (8 each)', x: 360, y: 754, width: 520, height: 240, colorKey: 'module:mlp' },
      ];
      width = 1180;
    }

    const tailNodes = [
      { id: 'a2a_combine',  label: 'All-to-All Combine', typeLabel: 'Comm', kind: 'op', x: CX, y: yCombine, width: 230, height: 50, colorKey: 'sem:comm', desc: 'All-to-All 汇聚：把各 rank 算完的专家输出按 token 收回并加权合并（通信事件，投影到物理轴）。' },
      { id: 'moe_residual', label: 'Post-MLP RMSNorm', typeLabel: 'Op', kind: 'op', x: CX, y: yCombine + 84, width: 220, height: 52, colorKey: 'sem:norm', desc: 'Post-MLP RMSNorm：源码通过 fused npu_add_rms_norm 处理 residual add + RMSNorm，不单独画 Add 节点。' },
    ];
    const yTrunk = yCombine + 84 + 120;
    const bottom = trunkBottom(yTrunk);
    const nodes = [...inputs, ...trunkTop, ...headNodes, ...expertNodes, ...tailNodes, ...bottom.nodes];
    const edges = [
      ...headEdges, ...expertEdges,
      { source: 'a2a_combine',  target: 'moe_residual', tag: 'ACT', edgeType: 'activation' },
      { source: 'moe_residual', target: 'final_norm',   tag: 'ACT', edgeType: 'activation' },
      ...bottom.edges,
    ];
    return { width, height: yTrunk + 290, clusters, nodes, edges, gridMeta };
  }

  return function buildUltraMoE718BGraph(opts) {
    opts = opts || {};
    if (!opts.moeExpanded) return buildCollapsed();
    return buildExpanded(!!opts.expertExpanded);
  };
})();
