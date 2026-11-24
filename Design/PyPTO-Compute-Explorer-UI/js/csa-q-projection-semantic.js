/* Q Projection data adapter. No DOM, layout, selection or camera ownership here.
 * Both Structure and Instance scales use the shared CSA Semantic Graph schema and renderer. */
(function (global) {
  'use strict';
  const defaultBase = global.PTO_CSA_SEMANTIC_GRAPH;
  const logical = global.PtoCsaQProjectionModel;
  const cache = new Map();
  const branchLabels = { qr: 'QR Projection', q: 'Q Projection', rope: 'RoPE Prepare' };
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
      provenance: [{ kind: 'resolved', origin: 'Q Projection 源码与现有计算模型', fields: [], note: '静态计算关系；循环与 Split-K 实例不表示实测核内时间。' }],
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
  const edge = (s, t, label, type = 'data', sourcePort = 'default', targetPort = 'default') =>
    ({ id: `${s}>${t}:${label}:${type}`, s, t, label, type, sourcePort, targetPort });

  function tileAttrs(info, item, instances) {
    if (info.type === 'qr') {
      return {
        zero: [[16, 128], 'FP32', 'GM'],
        'load-x': [[16, instances ? 256 : 2048], 'BF16', 'L1'],
        'load-w': [[instances ? 256 : 2048, 128], 'BF16', 'L1'],
        partial: [[16, 128], 'FP32', 'L0C'],
        qr: [[16, 128], 'FP32', 'GM'],
      }[item.op];
    }
    if (info.type === 'qmat') {
      return {
        'load-x': [[64, instances ? 128 : 1024], 'INT8', 'L1'],
        'load-w': [[instances ? 128 : 1024, 512], 'INT8', 'L1'],
        i32: [[64, 512], 'INT32', 'GM'],
      }[item.op];
    }
    if (info.type === 'seed') {
      return {
        'seed-region': [['QR_T_MATMUL', 1024], 'FP32', 'GM'],
        'qr-seeded': [[16, 128], 'FP32', 'GM'],
      }[item.op];
    }
    if (info.type === 'quant') {
      return {
        qr: [[8, 1024], 'FP32', 'GM'],
        gamma: [[1024], 'BF16', 'GM'],
        'qr-scale': [[8, 1], 'FP32', 'GM'],
        'qr-int8': [[8, instances ? 256 : 1024], 'INT8', 'GM'],
      }[item.op];
    }
    if (info.type === 'post') {
      return {
        'q-i32': [[8, 2048], 'INT32', 'GM'],
        'qr-scale': [[8, 1], 'FP32', 'GM'],
        'weight-scale': [[2048], 'FP32', 'GM'],
        rope: [[8, 64], 'FP32', 'GM'],
        q: [[8, instances ? 512 : 2048], 'BF16', 'GM'],
      }[item.op];
    }
    if (info.type === 'qrope' || info.type === 'kvrope') {
      return {
        'cos-sin': [[8, 64], 'BF16', 'GM'],
        params: [[8, 64], 'FP32', 'GM'],
      }[item.op];
    }
    return null;
  }

  function opLabel(info, item, instances) {
    if (info.type === 'qr') {
      const splitTag = item.split != null ? `Split ${item.split}` : null;
      const stepTag = item.step != null ? `K ${item.step}` : null;
      const suffix = [splitTag, stepTag].filter(Boolean).join(' · ');
      return suffix ? `${item.title} · ${suffix}` : item.title;
    }
    if (item.step == null) return item.title;
    const prefix = info.type === 'qmat' ? 'K'
      : info.type === 'post' ? 'Head'
      : ['seed', 'quant'].includes(info.type) ? 'N'
      : 'Iteration';
    return `${item.title} · ${prefix} ${item.step}`;
  }

  function kernelGraph(info, instances) {
    const graph = logical.operationModel(info, instances);
    const nodes = [], edges = [];
    const byId = new Map(graph.nodes.map(n => [n.id, n]));
    const isMatmulKernel = info.type === 'qr' || info.type === 'qmat';

    if (isMatmulKernel) {
      const qr = info.type === 'qr';
      const loadSrc = source(info.file, ...(qr ? [284, 294] : [380, 387]));
      const xIn = tensor('x', qr ? 'X' : 'QR INT8', loadSrc, qr ? [16, 4096] : [64, 1024], qr ? 'BF16' : 'INT8', 'input', 'GM');
      const wIn = tensor('w', qr ? 'Wq_a' : 'Wq_b', loadSrc, qr ? [4096, 128] : [1024, 512], qr ? 'BF16' : 'INT8', 'input', 'GM');
      xIn.description = qr ? 'Rank-local x_normed_t M Tile（按 Split-K 切片）' : '量化后 QR INT8 M Tile（按 K128 切片）';
      wIn.description = qr ? 'Wq_a N Tile 权重（按 Split-K 切片）' : 'Wq_b INT8 N Tile 权重（按 K128 切片）';
      xIn.stageContext = { stage: 'input/q', kernel: info.name, branch: info.branch, branchLabel: branchLabels[info.branch], coreType: info.coreType };
      wIn.stageContext = { stage: 'input/q', kernel: info.name, branch: info.branch, branchLabel: branchLabels[info.branch], coreType: info.coreType };
      nodes.push(xIn, wIn);
      const loadNodeMap = new Map();
      for (const item of graph.nodes) {
        if (item.op === 'load-w') continue;
        const src = source(item.file || info.file, ...item.lines);
        const description = [item.detail, item.condition].filter(Boolean).join(' · ') || item.title;
        const ctx = {
          stage: 'input/q', kernel: info.name, branch: info.branch, branchLabel: branchLabels[info.branch],
          operation: item.op, coreType: info.coreType,
          ...(item.split != null ? { split: item.split } : {}),
          ...(item.step != null ? { iteration: item.step } : {}),
          ...(item.nameRange ? { kRange: item.nameRange } : {}),
        };
        if (item.op === 'load-x') {
          const suffix = [item.split != null && qr ? `S${item.split}` : null, item.step != null ? `K${item.step}` : null].filter(Boolean).join(' ');
          const loadTitle = (qr ? 'Load X / Wq_a' : 'Load QR / Wq_b') + (suffix ? ` · ${suffix}` : '');
          const loadId = item.id.replace(/^load-x:/, 'load:');
          const n = node(loadId, 'operation', loadTitle, `${qr ? 'BF16' : 'INT8'} Tile Slice · ${description}`, src, { semantics: description, category: 'compute', api: [] });
          n.stageContext = { ...ctx, operation: 'load-x' };
          nodes.push(n);
          edges.push(edge('x', loadId, qr ? 'X slice' : 'QR slice'), edge('w', loadId, qr ? 'Wq_a slice' : 'Wq_b slice'));
          loadNodeMap.set(item.id, loadId);
          loadNodeMap.set(item.id.replace(/^load-x:/, 'load-w:'), loadId);
          continue;
        }
        let n;
        if (item.role === 'operation') {
          n = node(item.id, 'operation', opLabel(info, item, instances), description, src, { semantics: description, category: 'compute', api: [] });
        } else {
          const attrs = tileAttrs(info, item, instances);
          n = tensor(item.id, item.title, src, attrs?.[0] || null, attrs?.[1] || null, item.role === 'activation' ? 'input' : 'output', attrs?.[2]);
          n.description = description;
        }
        n.stageContext = ctx;
        nodes.push(n);
        if (item.op === 'matmul') {
          const accShape = qr ? [16, 128] : [64, 512];
          const accDtype = qr ? 'FP32' : 'INT32';
          const acc = tensor(`${item.id}-acc`, qr ? 'q_acc' : 'col_acc', src, accShape, accDtype, 'intermediate', 'L0C');
          acc.description = `${accDtype} L0C accumulator · ${description}`;
          acc.stageContext = ctx;
          nodes.push(acc);
          edges.push(edge(item.id, acc.id, `${accDtype} accumulator`));
        }
      }
      const seenLinks = new Set();
      for (const link of graph.edges) {
        const from = byId.get(link.from);
        const to = byId.get(link.to);
        const mappedFrom = loadNodeMap.get(link.from) || (from?.op === 'matmul' ? `${from.id}-acc` : link.from);
        const key = `${mappedFrom}>${link.to}`;
        if (seenLinks.has(key)) continue;
        seenLinks.add(key);
        const carrying = from?.op === 'matmul' && to?.op === 'matmul';
        const type = carrying ? 'state' : link.type === 'state' ? 'state' : 'data';
        const label = carrying ? 'accumulate' : loadNodeMap.has(link.from) ? 'L1 tiles' : from?.op === 'atomic' ? 'atomic add' : from?.title || 'data';
        edges.push(edge(mappedFrom, link.to, label, type));
      }
      const ins = qr ? { x: mapping('x'), w: mapping('w'), zero: mapping('zero'), default: mapping('x'), ready: mapping(nodes.find(n => n.type === 'operation').id, 'default', 'control') }
        : { x: mapping('x'), w: mapping('w'), default: mapping('x'), ready: mapping(nodes.find(n => n.type === 'operation').id, 'default', 'control') };
      const outId = qr ? 'qr' : 'i32';
      const outs = { [outId]: mapping(outId), default: mapping(outId), tid: mapping(outId, 'default', 'control') };
      const count = logical.blockCount(info, 256);
      const n = node(
        info.name, 'composite', info.name,
        `${info.title} · ${info.coreType} · ${count} logical Blocks · ${info.file}:${info.first}–${info.last}`,
        source(info.file, info.first, info.last),
        {
          semantics: `${info.title} · 源码计算与核内分块；展开查看 Operation / Tensor。`,
          category: 'module', api: [],
          subgraph: { schemaVersion: 2, id: `${info.name}-${instances ? 'instances' : 'structure'}`, nodes, edges, groups: [] },
          portMappings: { inputs: ins, outputs: outs },
        }
      );
      n.stageContext = { stage: 'input/q', kernel: info.name, branch: info.branch, branchLabel: branchLabels[info.branch], coreType: info.coreType };
      n.ports = ports(Object.keys(ins), Object.keys(outs));
      return n;
    }

    for (const item of graph.nodes) {
      const src = source(item.file || info.file, ...item.lines);
      const description = [item.detail, item.condition].filter(Boolean).join(' · ') || item.title;
      const ctx = {
        stage: 'input/q',
        kernel: info.name,
        branch: info.branch,
        branchLabel: branchLabels[info.branch] || info.branch,
        operation: item.op,
        coreType: info.coreType,
        ...(item.split != null ? { split: item.split } : {}),
        ...(item.step != null ? { iteration: item.step } : {}),
        ...(item.nameRange ? { kRange: item.nameRange } : {}),
      };
      let n;
      if (item.role === 'operation') {
        n = node(item.id, 'operation', opLabel(info, item, instances), description, src, { semantics: description, category: 'compute', api: [] });
      } else {
        const attrs = tileAttrs(info, item, instances);
        const label = item.role === 'activation' && info.type === 'qr' && item.split != null
          ? `${item.title.replace(/^Load /, '')} · S${item.split}${item.step != null ? ` K${item.step}` : ''}`
          : item.role === 'activation' && info.type === 'qmat' && item.step != null
          ? `${item.title.replace(/^Load /, '')} · K${item.step}`
          : item.step != null && item.role === 'aggregate'
          ? `${item.title} · ${info.type === 'post' ? 'H' : 'N'}${item.step}`
          : item.title;
        n = tensor(item.id, label, src, attrs?.[0] || null, attrs?.[1] || null, item.role === 'activation' ? 'input' : 'output', attrs?.[2]);
        n.description = description;
      }
      n.stageContext = ctx;
      nodes.push(n);

      // Expose the L0C accumulator between matmul_acc K-tile iterations.
      if (isMatmulKernel && item.op === 'matmul') {
        const accShape = info.type === 'qr' ? [16, 128] : [64, 512];
        const accDtype = info.type === 'qr' ? 'FP32' : 'INT32';
        const accName = info.type === 'qr' ? 'q_acc' : 'col_acc';
        const acc = tensor(`${item.id}-acc`, accName, src, accShape, accDtype, 'intermediate', 'L0C');
        acc.description = `${accDtype} L0C accumulator · ${description}`;
        acc.stageContext = ctx;
        nodes.push(acc);
        edges.push(edge(item.id, acc.id, `${accDtype} accumulator`));
      }
    }

    for (const link of graph.edges) {
      const from = byId.get(link.from);
      const to = byId.get(link.to);
      const fromMatmul = isMatmulKernel && from?.op === 'matmul';
      const carrying = fromMatmul && to?.op === 'matmul';
      const type = carrying ? 'state' : link.type === 'state' ? 'state' : 'data';
      const label = carrying ? 'accumulate' : from?.op === 'atomic' ? 'atomic add' : from?.title || 'data';
      edges.push(edge(fromMatmul ? `${from.id}-acc` : link.from, link.to, label, type));
    }

    const inputs = graph.nodes.filter(n => n.role === 'activation');
    const outputs = graph.nodes.filter(n => n.role === 'aggregate');
    const ins = Object.fromEntries(inputs.map(n => [n.id, mapping(n.id)]));
    const outs = Object.fromEntries(outputs.map(n => [n.id, mapping(n.id)]));
    ins.default = mapping(inputs[0]?.id || nodes[0].id);
    ins.ready = mapping(graph.nodes.find(n => n.role === 'operation')?.id || nodes[0].id, 'default', 'control');
    if (info.type === 'qr') {
      ins.x = mapping(inputs.find(n => n.op === 'load-x')?.id || inputs[0].id);
      ins.w = mapping(inputs.find(n => n.op === 'load-w')?.id || inputs[0].id);
      ins.zero = mapping('zero');
      outs.qr = mapping('qr');
    } else if (info.type === 'qmat') {
      ins.x = mapping(inputs.find(n => n.op === 'load-x')?.id || inputs[0].id);
      ins.w = mapping(inputs.find(n => n.op === 'load-w')?.id || inputs[0].id);
      outs.i32 = mapping('i32');
    } else if (info.type === 'quant') {
      outs['qr-scale'] = mapping('qr-scale');
      outs['qr-int8'] = mapping(outputs.find(n => n.op === 'qr-int8')?.id || outputs.at(-1).id);
    } else if (info.type === 'post') {
      outs.q = mapping(outputs.find(n => n.op === 'q')?.id || outputs.at(-1).id);
    }
    outs.default = mapping(outputs.at(-1)?.id || nodes.at(-1).id);
    outs.tid = mapping(outputs.at(-1)?.id || nodes.at(-1).id, 'default', 'control');

    const count = logical.blockCount(info, 256);
    const n = node(
      info.name, 'composite', info.name,
      `${info.title} · ${info.coreType} · ${count} logical Blocks · ${info.file}:${info.first}–${info.last}`,
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
      stage: 'input/q',
      kernel: info.name,
      branch: info.branch,
      branchLabel: branchLabels[info.branch] || info.branch,
      coreType: info.coreType,
    };
    n.ports = ports(Object.keys(ins), Object.keys(outs));
    return n;
  }

  function qBranchGraph(original, instances) {
    const n = structuredClone(original);
    const k = name => kernelGraph(logical.kernels.get(name), instances);
    const qFile = 'qkv_proj_rope.py', caller = 'decode_csa.py';
    const srcQ = (a, b) => source(qFile, a, b);
    const srcC = (a, b) => source(caller, a, b);

    const nodes = [
      tensor('x', 'x_normed_t', srcC(273, 275), ['T_DYN', 4096], 'BF16', 'input', 'GM'),
      tensor('wqa', 'Wq_a', srcQ(245, 249), [4096, 1024], 'BF16', 'input', 'GM'),
      k('qr_proj_seed'),
      k('qr_proj_matmul'),
      tensor('qr-fp32', 'qr_fp32', srcQ(270, 299), ['QR_T_MATMUL', 1024], 'FP32', 'intermediate', 'GM'),
      tensor('gamma', 'Gamma cq', srcQ(249, 249), [1024], 'BF16', 'input', 'GM'),
      k('qr_rms_norm_quant'),
      tensor('qr-i8', 'qr_i8_matmul', srcQ(303, 303), ['QPROJ_T_MATMUL', 1024], 'INT8', 'intermediate', 'GM'),
      tensor('qr-scale-pad', 'qr_scale_pad_store', srcQ(305, 305), ['QPROJ_T_MATMUL', 1], 'FP32', 'intermediate', 'GM'),
      tensor('wqb', 'Wq_b', srcQ(246, 248), [1024, 32768], 'INT8', 'input', 'GM'),
      k('qproj_matmul'),
      tensor('q-i32', 'q_proj_i32', srcQ(371, 371), ['QPROJ_T_MATMUL', 32768], 'INT32', 'intermediate', 'GM'),
      tensor('wscale', 'Wq_b Scale', srcQ(247, 248), [32768], 'FP32', 'input', 'GM'),
      tensor('rope-local', 'Q-local Cos / Sin', srcC(304, 307), ['T_DYN', 64], 'BF16', 'input', 'GM'),
      k('q_rope_prepare_0'),
      tensor('qparams', 'Q-local RoPE Params', srcQ(123, 125), ['T_DYN', 64], 'FP32', 'intermediate', 'GM'),
      k('qproj_dequant_rms_nope_rope'),
    ];
    nodes.find(item => item.id === 'qr-fp32').description = 'FP32 QR Scratch · Split-K Atomic Add 累加目标';
    nodes.find(item => item.id === 'qr-scale-pad').description = 'FP32 per-row Dequant Scale（manual_dep 显式跨任务传递）';

    const edges = [];
    const add = (...args) => edges.push(edge(...args));
    add('x', 'qr_proj_matmul', 'BF16 input', 'data', 'default', 'x');
    add('wqa', 'qr_proj_matmul', 'BF16 weight', 'data', 'default', 'w');
    add('qr_proj_seed', 'qr_proj_matmul', 'zero seed', 'state', 'default', 'zero');
    add('qr_proj_matmul', 'qr-fp32', 'Atomic Add FP32', 'data', 'qr');
    add('qr-fp32', 'qr_rms_norm_quant', 'QR FP32', 'data', 'default', 'qr');
    add('gamma', 'qr_rms_norm_quant', 'RMSNorm weight', 'data', 'default', 'gamma');
    if (instances) {
      for (let i = 0; i < 4; i++) add('qr_rms_norm_quant', 'qr-i8', `N [${i * 256}, ${(i + 1) * 256})`, 'data', `qr-int8:${i}`);
    } else {
      add('qr_rms_norm_quant', 'qr-i8', 'INT8 quantized QR', 'data', 'qr-int8');
    }
    add('qr_rms_norm_quant', 'qr-scale-pad', 'dequant scale', 'data', 'qr-scale');
    add('qr-i8', 'qproj_matmul', 'INT8 activation', 'data', 'default', 'x');
    add('wqb', 'qproj_matmul', 'INT8 weight', 'data', 'default', 'w');
    add('qproj_matmul', 'q-i32', 'INT32 acc', 'data', 'i32');
    add('rope-local', 'q_rope_prepare_0', 'local Cos / Sin', 'data', 'default', 'cos-sin');
    add('q_rope_prepare_0', 'qparams', 'interleaved RoPE', 'data', 'params');
    add('q-i32', 'qproj_dequant_rms_nope_rope', 'Q INT32', 'data', 'default', 'q-i32');
    add('qr-scale-pad', 'qproj_dequant_rms_nope_rope', 'manual_dep scale', 'state', 'default', 'qr-scale');
    add('wscale', 'qproj_dequant_rms_nope_rope', 'weight scale', 'data', 'default', 'weight-scale');
    add('qparams', 'qproj_dequant_rms_nope_rope', 'RoPE params', 'data', 'default', 'rope');

    n.name = 'Q LoRA & RoPE';
    n.description = 'Rank-local Q 投影：QR Split-K Matmul → RMSNorm + INT8 Quant → Q INT8 Matmul → Dequant / RoPE。';
    n.w = estWidth(n.name, true);
    n.composite.semantics = n.description;
    n.composite.subgraph = { schemaVersion: 2, id: 'q-projection-' + (instances ? 'instances' : 'structure'), nodes, edges, groups: [] };
    n.composite.portMappings = {
      inputs: {
        local: mapping('x'),
        position: mapping('rope-local'),
      },
      outputs: {
        q: mapping('qproj_dequant_rms_nope_rope', 'q'),
        qr: mapping('qr-i8'),
        qr_scale: mapping('qr-scale-pad'),
      },
    };
    return n;
  }

  function kvRopeBranchGraph(instances) {
    const qFile = 'qkv_proj_rope.py', caller = 'decode_csa.py';
    const srcQ = (a, b) => source(qFile, a, b);
    const srcC = (a, b) => source(caller, a, b);
    const ropeKernel = kernelGraph(logical.kernels.get('q_rope_prepare'), instances);
    const nodes = [
      tensor('rope-kv-in', 'KV Group Cos / Sin', srcC(299, 302), ['KV_T_DYN', 64], 'BF16', 'input', 'GM'),
      ropeKernel,
      tensor('rope-kv-params', 'KV Group RoPE Params', srcQ(123, 125), ['KV_T_DYN', 64], 'FP32', 'output', 'GM'),
    ];
    const edges = [
      edge('rope-kv-in', 'q_rope_prepare', 'full-group Cos / Sin', 'data', 'default', 'cos-sin'),
      edge('q_rope_prepare', 'rope-kv-params', '→ KV Projection', 'data', 'params'),
    ];
    const mod = node(
      'rope-kv', 'composite', 'KV RoPE Prepare',
      '全组 KV 共享位置参数预计算（q_rope_prepare · 64 Blocks），供下游 KV Projection 使用。',
      srcQ(120, 241),
      {
        semantics: '全组 KV RoPE 参数预计算分支。',
        category: 'module',
        api: [],
        subgraph: { schemaVersion: 2, id: 'kv-rope-' + (instances ? 'instances' : 'structure'), nodes, edges, groups: [] },
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
    const stage = model.nodes.find(n => n.id === 'input');
    stage.composite.subgraph.nodes = stage.composite.subgraph.nodes.map(n =>
      n.id === 'q' ? qBranchGraph(n, instances) : n
    );
    if (focused) {
      const kvRope = kvRopeBranchGraph(instances);
      const keep = new Set(['q', 'q-tensor', 'qr-tensor', 'qr-scale']);
      stage.name = 'Q Projection';
      stage.description = 'Rank-local Q LoRA（Split-K QR · INT8 Q Matmul · Dequant / RoPE）与 KV RoPE Prepare';
      stage.ports = ports([], []);
      stage.composite.portMappings = { inputs: {}, outputs: {} };
      const keptNodes = stage.composite.subgraph.nodes.filter(n => keep.has(n.id));
      const qIndex = keptNodes.findIndex(n => n.id === 'q');
      keptNodes.splice(qIndex + 1, 0, kvRope);
      stage.composite.subgraph.nodes = keptNodes;
      stage.composite.subgraph.edges = stage.composite.subgraph.edges.filter(e => keep.has(e.s) && keep.has(e.t));
      model.nodes = [stage];
      model.edges = [];
      model.id = 'csa-q-projection';
    }
    function evidence(list) {
      for (const item of list) {
        if (item.provenance?.[0]?.origin === 'Q Projection 源码与现有计算模型') {
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

  global.PtoCsaQProjectionSemantic = Object.freeze({ getModel });
})(window);
