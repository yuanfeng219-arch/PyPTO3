/* CSA Operator Lens · L2 Kernel 链的语义图模型（Explorer-UI schema v2），与 L1 用同一套渲染与图元。
 * 节点：Kernel = 计算节点；链外传进来的 TaskId、链交给外部的 TaskId = 数据节点（role metadata）。
 * 边：源码显式写出的 TaskId 依赖 = control；由张量读写推出的依赖 = data，标签是张量名（见 kernel-scan.js）。 */
(function (root) {
  'use strict';
  const D = root.LensData, SRC = root.LENS_SOURCE;
  const NOTE = '按源码静态整理：TaskId 依赖取自 deps / as 写法，数据依赖按张量读写推出；不代表运行采集结果。';
  const sourceFiles = {};
  Object.keys(SRC.files).forEach(function (f) { sourceFiles[f] = { lineCount: SRC.files[f].text.replace(/\n$/, '').split('\n').length }; });
  const PORTS = function () { return { inputs: { default: {} }, outputs: { default: {} } }; };
  const nodeId = function (k) { return k.replace(/[^\w.-]/g, '_'); };

  function kernelNode(K) {
    return { id: nodeId(K.id), type: 'operation', name: K.name, description: K.what,
      source: { file: K.file, ranges: [{ start: K.range[0], end: K.range[1] }], symbols: [K.name] },
      provenance: [{ kind: 'fact', origin: 'source', fields: [], note: NOTE }], ports: PORTS(),
      operation: { semantics: K.what, category: K.dispatch === 'spmd' ? 'spmd' : 'core_group', api: [K.dispatch === 'spmd' ? 'pl.spmd' : 'pl.at'] } };
  }
  function tidNode(id, name, K, role) {
    const data = { role: 'metadata', granularity: 'scalar', symbol: name, dtype: 'TASK_ID' };
    return { id: id, type: 'data', name: name + (role === 'in' ? ' · 外部' : ' · 交出'),
      description: role === 'in' ? K.name + ' 等待的 TaskId 来自本模块之外：' + name + '。' : K.name + ' 用 as 捕获 ' + name + '，交给本模块之外的消费者。',
      source: { file: K.file, ranges: [{ start: K.line, end: K.line }], symbols: [name] },
      provenance: [{ kind: 'fact', origin: 'source', fields: Object.keys(data).map(function (k) { return 'data.' + k; }), note: NOTE }],
      ports: PORTS(), data: data };
  }
  function edge(s, t, label, type) {
    return { id: s + '>' + t + ':' + label, s: s, t: t, type: type, sourcePort: 'default', targetPort: 'default', label: label };
  }

  function build(mod) {
    const ch = D.CHAINS[mod], M = D.MODULES.filter(function (m) { return m.id === mod; })[0];
    if (!ch) return null;
    const nodes = ch.order.map(function (k) { return kernelNode(D.KERNELS[k]); });
    const edges = [], consumed = {};
    ch.edges.forEach(function (e) { edges.push(edge(nodeId(e[0]), nodeId(e[1]), e[2], 'control')); consumed[e[2]] = 1; });
    (ch.flow || []).forEach(function (e) { edges.push(edge(nodeId(e[0]), nodeId(e[1]), e[2].length > 2 ? e[2].slice(0, 2).join(', ') + ' …' : e[2].join(', '), 'data')); });
    // 链外传进来的 TaskId：源码里 deps 列出、但不是本链 Kernel 产生的
    const ext = {};
    ch.order.forEach(function (k) {
      const K = D.KERNELS[k];
      (K.deps || []).forEach(function (d) {
        if (ch.order.some(function (x) { return D.KERNELS[x].produces === d; })) return;
        const id = 'in_' + d.replace(/\W/g, '_');
        if (!ext[id]) { ext[id] = 1; nodes.unshift(tidNode(id, d, K, 'in')); }
        edges.push(edge(id, nodeId(k), d, 'control'));
      });
    });
    // 交给外部的 TaskId：as 捕获、本链没人等，且不是 _ 开头的占位名
    ch.order.forEach(function (k) {
      const K = D.KERNELS[k];
      if (!K.produces || /^_/.test(K.produces) || consumed[K.produces]) return;
      const id = 'out_' + K.produces.replace(/\W/g, '_');
      nodes.push(tidNode(id, K.produces, K, 'out'));
      edges.push(edge(nodeId(k), id, K.produces, 'control'));
    });
    const first = D.KERNELS[ch.order[0]];
    return {
      schemaVersion: 2, id: 'chain-' + mod, sourceFiles: sourceFiles, constants: {}, externalConstants: {}, runtimeDims: [],
      nodes: nodes, edges: edges,
      groups: [{ id: 'g_' + mod, type: 'region', name: (M ? M.label : mod) + ' · ' + ch.order.length + ' 个 Kernel', description: '本模块派发的 Kernel', members: ch.order.map(nodeId),
        source: { file: first.file, ranges: [{ start: first.fn[0], end: first.fn[1] }], symbols: [] },
        provenance: [{ kind: 'resolved', origin: 'source', fields: ['members'], note: '按模块调用的函数分组。' }] }]
    };
  }
  // 图节点 id ↔ Kernel id
  root.LensChainGraph = {
    build: build,
    nodeOf: nodeId,
    kernelOf: function (id) { const hit = Object.keys(D.KERNELS).filter(function (k) { return nodeId(k) === id; })[0]; return hit || null; }
  };
})(window);
