/* qk_plan semantic graph data adapter. No DOM, layout, selection or camera ownership here.
 * Both Structure and Instance scales use the shared CSA Semantic Graph schema and renderer. */
(function (global) {
  'use strict';

  const defaultBase = global.PTO_CSA_SEMANTIC_GRAPH;
  const logical = global.PtoCsaQkPlanModel;
  const cache = new Map();
  const branchLabels = { plan: 'Candidate Planning', ropecs: 'Inverse RoPE Metadata' };
  const ports = (inputs = ['default'], outputs = ['default']) => ({
    inputs: Object.fromEntries(inputs.map(id => [id, {}])),
    outputs: Object.fromEntries(outputs.map(id => [id, {}])),
  });
  const mapping = (node, port = 'default', relation = 'view') => ({ node, port, relation });
  const source = (file, start, end) => ({ file, ranges: [{ start, end }], symbols: [] });
  const estWidth = (label, expandable) =>
    Math.max(194, Math.ceil([...String(label)].reduce((w, ch) => w + (/[\u2e80-\u9fff\uff00-\uffef]/.test(ch) ? 15.5 : 8.8), 0)) + (expandable ? 96 : 66));

  function node(id, type, name, description, src, payload) {
    const w = ['operation', 'composite'].includes(type) ? estWidth(name, type === 'composite') : undefined;
    return {
      id, type, name, description, source: src, ports: ports(), ...(w ? { w } : {}), [type]: payload,
      provenance: [{ kind: 'resolved', origin: 'qk_plan 源码与现有计算模型', fields: [], note: '静态计算关系；循环与块级实例不表示实测核内时间。' }],
    };
  }

  function tensor(id, name, src, shape, dtype, role = 'intermediate', storage) {
    return node(id, 'data', name, name, src, {
      role, granularity: 'tensor', entity: id,
      ...(shape ? { shape: { allocated: shape } } : {}),
      ...(dtype ? { dtype } : {}),
      ...(storage ? { storage: { space: storage } } : {}),
    });
  }

  function stateNode(id, name, description, src, options = {}) {
    return node(id, 'state', name, description, src, {
      initial: options.initial || '任务尚未完成',
      update: description,
      granularity: options.granularity || 'task',
    });
  }

  const edge = (s, t, label, type = 'data', sourcePort = 'default', targetPort = 'default') =>
    ({ id: `${s}>${t}:${label}:${type}`, s, t, label, type, sourcePort, targetPort });

  function tileAttrs(info, item) {
    if (info.type === 'qkplan') {
      return {
        'idx-topk': [[8, 512], 'INT32', 'GM'],
        'pos-ids': [[8, 1], 'INT32', 'GM'],
        'win-idx': [[8, 128], 'INT32', 'GM'],
        'cmp-indices': [[8, 512], 'INT32', 'GM'],
        'valid-mask': [['T_DYN', 5], 'INT32', 'GM'],
        'sparse-bias': [[8, 640], 'FP32', 'GM'],
        'qk-order': [[1280], 'INT32', 'GM'],
      }[item.op];
    }
    return {
      'freqs-cos': [[8, 32], 'BF16', 'GM'],
      'freqs-sin': [[8, 32], 'BF16', 'GM'],
      'rope-swap': [[16, 64], 'INT32', 'GM'],
      'rope-cos': [[8, 64], 'FP32', 'GM'],
      'rope-sin': [[8, 64], 'FP32', 'GM'],
    }[item.op];
  }

  function kernelGraph(info, instances) {
    const graph = logical.operationModel(info, instances);
    const nodes = [], edges = [];
    const byId = new Map(graph.nodes.map(n => [n.id, n]));

    for (const item of graph.nodes) {
      const src = source(item.file || info.file, ...item.lines);
      const description = [item.detail, item.condition].filter(Boolean).join(' · ') || item.title;
      const ctx = {
        stage: 'sparse/attention/plan',
        kernel: info.name,
        branch: info.branch,
        branchLabel: branchLabels[info.branch] || info.branch,
        operation: item.op,
        coreType: info.coreType,
        ...(item.step != null ? { iteration: item.step } : {}),
      };
      let n;
      if (item.role === 'operation') {
        n = node(item.id, 'operation', item.title, description, src, { semantics: description, category: 'compute', api: [] });
      } else {
        const attrs = tileAttrs(info, item);
        n = tensor(item.id, item.title, src, attrs?.[0] || null, attrs?.[1] || null, item.role === 'activation' ? 'input' : 'output', attrs?.[2]);
        n.description = description;
      }
      n.stageContext = ctx;
      nodes.push(n);
    }

    for (const link of graph.edges) {
      const from = byId.get(link.from);
      const type = link.type === 'state' ? 'state' : from?.lane === 'Control' ? 'control' : 'data';
      const label = link.type === 'state' ? 'qk_wcur cursor' : from?.title || 'data';
      edges.push(edge(link.from, link.to, label, type));
    }

    const inputs = graph.nodes.filter(n => n.role === 'activation');
    const outputs = graph.nodes.filter(n => n.role === 'aggregate');
    const ins = Object.fromEntries(inputs.map(n => [n.id, mapping(n.id)]));
    const outs = Object.fromEntries(outputs.map(n => [n.id, mapping(n.id)]));
    ins.default = mapping(inputs[0]?.id || nodes[0].id);
    ins.ready = mapping(graph.nodes.find(n => n.role === 'operation')?.id || nodes[0].id, 'default', 'control');
    outs.default = mapping(outputs.at(-1)?.id || nodes.at(-1).id);
    outs.tid = mapping(outputs.at(-1)?.id || nodes.at(-1).id, 'default', 'control');

    const count = logical.blockCount(info, 256);
    const n = node(
      info.name,
      'composite',
      info.name,
      `${info.title} · ${info.coreType} · ${count} logical Block (pl.at) · ${info.file}:${info.first}–${info.last}`,
      source(info.file, info.first, info.last),
      {
        semantics: `${info.title} · 源码计算与核内分块；展开查看 Operation / Tensor。`,
        category: 'module',
        api: [],
        subgraph: { schemaVersion: 2, id: `${info.name}-${instances ? 'instances' : 'structure'}`, nodes, edges, groups: [] },
        portMappings: { inputs: ins, outputs: outs },
      }
    );
    n.stageContext = {
      stage: 'sparse/attention/plan',
      kernel: info.name,
      branch: info.branch,
      branchLabel: branchLabels[info.branch] || info.branch,
      coreType: info.coreType,
    };
    n.ports = ports(Object.keys(ins), Object.keys(outs));
    return n;
  }

  function planBranchGraph(instances) {
    const f = 'decode_sparse_attn_csa.py';
    const src = (a, b) => source(f, a, b);
    const k = kernelGraph(logical.kernels.get('csa_slots_build_valid_qk_plan'), instances);
    const nodes = [
      tensor('idx-topk', 'idx_topk', src(111, 111), ['T_DYN', 512], 'INT32', 'input', 'GM'),
      tensor('position-ids', 'position_ids', src(112, 112), ['T_DYN', 1], 'INT32', 'input', 'GM'),
      tensor('window-indices', 'window_swa_indices', src(108, 108), ['T_DYN', 128], 'INT32', 'input', 'GM'),
      k,
      tensor('cmp-sparse-indices', 'cmp_sparse_indices', src(139, 158), ['T_DYN', 512], 'INT32', 'output', 'GM'),
      tensor('valid-block-mask', 'valid_block_mask', src(140, 167), ['T_DYN', 5], 'INT32', 'output', 'GM'),
      tensor('sparse-bias', 'sparse_bias', src(138, 180), ['T_DYN', 640], 'FP32', 'output', 'GM'),
      tensor('qk-order', 'qk_order', src(141, 196), [1280], 'INT32', 'output', 'GM'),
      stateNode('qk-plan-tid', 'qk_plan_tid', '单核规划完成后释放给 24-Core qk_pv 派发依赖。', src(143, 143), { initial: '规划任务未完成', granularity: 'task' }),
    ];
    nodes.find(item => item.id === 'cmp-sparse-indices').description = '有效压缩槽位，−1 表示无效（供 qk_pv Gather）';
    nodes.find(item => item.id === 'valid-block-mask').description = 'SB0 滑窗恒有效，SB1–4 由压缩候选 mask 归约决定';
    nodes.find(item => item.id === 'sparse-bias').description = '有效槽位 0 / 无效槽位 −1e20 加性 QK Bias';
    nodes.find(item => item.id === 'qk-order').description = 'Pass 1 非空块前置 + Pass 2 空块追加的 1,280 项工作队列';

    const edges = [
      edge('idx-topk', 'csa_slots_build_valid_qk_plan', 'Top-k slots', 'data', 'default', 'idx-topk'),
      edge('position-ids', 'csa_slots_build_valid_qk_plan', 'position bound', 'data', 'default', 'pos-ids'),
      edge('window-indices', 'csa_slots_build_valid_qk_plan', 'window slots', 'data', 'default', 'win-idx'),
      edge('csa_slots_build_valid_qk_plan', 'cmp-sparse-indices', 'masked slots', 'data', 'cmp-indices'),
      edge('csa_slots_build_valid_qk_plan', 'valid-block-mask', 'block validity', 'data', 'valid-mask'),
      edge('csa_slots_build_valid_qk_plan', 'sparse-bias', 'additive bias', 'data', 'sparse-bias'),
      edge('csa_slots_build_valid_qk_plan', 'qk-order', 'valid-first queue', 'data', 'qk-order'),
      edge('csa_slots_build_valid_qk_plan', 'qk-plan-tid', 'plan ready → qk_pv', 'control', 'tid'),
    ];

    const mod = node(
      'plan',
      'composite',
      'Candidate Planning',
      '单核 AIV (pl.at) 串行执行 32 个 8-token 向量块与 2-Pass 标量压缩，生成 valid_block_mask、sparse_bias 与 qk_order。',
      src(138, 196),
      {
        semantics: 'Candidate Planning 与 Valid-First 队列构建分支。',
        category: 'module',
        api: [],
        subgraph: { schemaVersion: 2, id: 'qk-plan-branch-' + (instances ? 'instances' : 'structure'), nodes, edges, groups: [] },
        portMappings: { inputs: {}, outputs: {} },
      }
    );
    mod.ports = ports([], []);
    return mod;
  }

  function ropeBranchGraph(instances) {
    const f = 'decode_sparse_attn_csa.py';
    const src = (a, b) => source(f, a, b);
    const k = kernelGraph(logical.kernels.get('rope_cs'), instances);
    const nodes = [
      tensor('freqs-cos', 'freqs_cos', src(113, 113), ['T_DYN', 64], 'BF16', 'input', 'GM'),
      tensor('freqs-sin', 'freqs_sin', src(114, 114), ['T_DYN', 64], 'BF16', 'input', 'GM'),
      k,
      tensor('rope-swap-idx', 'rope_swap_idx', src(321, 330), [16, 64], 'INT32', 'output', 'GM'),
      tensor('rope-cos-il', 'rope_cos_il', src(317, 344), ['T_PAD', 64], 'FP32', 'output', 'GM'),
      tensor('rope-sin-signed', 'rope_sin_signed', src(318, 346), ['T_PAD', 64], 'FP32', 'output', 'GM'),
      stateNode('rope-tid', 'rope_tid', '逆 RoPE 系数就绪后释放给下游 csa_merge_pack_publish。', src(322, 322), { initial: 'RoPE 元数据未就绪', granularity: 'task' }),
    ];
    nodes.find(item => item.id === 'rope-swap-idx').description = 'j^1 通道交换索引，供下游 Merge 逆旋转 Gather';
    nodes.find(item => item.id === 'rope-cos-il').description = '交错展开的 FP32 Cos 表（×32 Token Tiles）';
    nodes.find(item => item.id === 'rope-sin-signed').description = '折叠共轭符号 [+1, −1, ...] 的 FP32 Sin 表';

    const edges = [
      edge('freqs-cos', 'rope_cs', 'local Cos [:, 0:32]', 'data', 'default', 'freqs-cos'),
      edge('freqs-sin', 'rope_cs', 'local Sin [:, 0:32]', 'data', 'default', 'freqs-sin'),
      edge('rope_cs', 'rope-swap-idx', 'j^1 swap index', 'data', 'rope-swap'),
      edge('rope_cs', 'rope-cos-il', 'interleaved Cos', 'data', 'rope-cos'),
      edge('rope_cs', 'rope-sin-signed', 'conjugate Sin', 'data', 'rope-sin'),
      edge('rope_cs', 'rope-tid', 'rope ready → Merge', 'control', 'tid'),
    ];

    const mod = node(
      'rope',
      'composite',
      'Inverse RoPE Metadata',
      '单核 AIV (pl.at) 预计算逆 RoPE 交错 Cos / 共轭 Sin 与 Swap 索引，独立于 qk_plan_tid 并行运行。',
      src(317, 346),
      {
        semantics: 'Inverse RoPE 元数据预计算分支。',
        category: 'module',
        api: [],
        subgraph: { schemaVersion: 2, id: 'qk-plan-rope-' + (instances ? 'instances' : 'structure'), nodes, edges, groups: [] },
        portMappings: { inputs: {}, outputs: {} },
      }
    );
    mod.ports = ports([], []);
    return mod;
  }

  function getModel({ base = defaultBase, focused = false, instances = false } = {}) {
    const key = `${base?.id || 'default'}:${focused}:${instances}`;
    if (cache.has(key)) return cache.get(key);
    const model = structuredClone(base);
    if (focused) {
      const stage = model.nodes.find(n => n.id === 'sparse');
      const planBranch = planBranchGraph(instances);
      const ropeBranch = ropeBranchGraph(instances);
      stage.name = 'qk_plan';
      stage.description = 'Candidate Planning (csa_slots_build_valid_qk_plan) 与 Inverse RoPE Metadata (rope_cs)';
      stage.ports = ports([], []);
      stage.composite.portMappings = { inputs: {}, outputs: {} };
      stage.composite.subgraph = {
        schemaVersion: 2,
        id: 'qk-plan-stage-' + (instances ? 'instances' : 'structure'),
        nodes: [planBranch, ropeBranch],
        edges: [],
        groups: [],
      };
      model.nodes = [stage];
      model.edges = [];
      model.id = 'csa-qk-plan';
    }
    function evidence(list) {
      for (const item of list) {
        if (item.provenance?.[0]?.origin === 'qk_plan 源码与现有计算模型') {
          const payload = item[item.type];
          item.provenance[0].fields = ['data', 'state'].includes(item.type) ? Object.keys(payload).map(k => `${item.type}.${k}`) : [];
        }
        if (item.composite) evidence(item.composite.subgraph.nodes);
      }
    }
    evidence(model.nodes);
    global.PTO_SEMANTIC_COMPUTATION_GRAPH.validate(model);
    cache.set(key, model);
    return model;
  }

  global.PtoCsaQkPlanSemantic = Object.freeze({ getModel });
})(window);
