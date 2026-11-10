/* CSA Operator Lens · L1 语义计算图模型。
 * 采用 Design/PyPTO-Compute-Explorer-UI 的语义图 schema v2（semantic-computation-graph.js 会严格校验），
 * 交给同一套 materialize → renderScene → AI Infra 图元皮肤渲染。
 * 行号与 Shape 取自 Data/DeepseekV4/deepseek_v4_flash_dspark/ 当前源码（2026-11-10）；
 * 模块说明与 lens-data.js 共用同一份文字。 */
(function (root) {
  'use strict';
  const CSA = 'decode_csa.py', SPARSE = 'decode_sparse_attn_csa.py';
  const L = root.LensData;
  const MOD = {}; L.MODULES.forEach(function (m) { MOD[m.id] = m; });

  // 源码段：[[file, a, b], ...] → schema 的 source 对象（一个节点只能落在一个文件）
  function source(list, symbols) {
    const file = list[0][0];
    return { file: file, ranges: list.filter(function (s) { return s[0] === file; }).map(function (s) { return { start: s[1], end: s[2] }; }), symbols: symbols || [] };
  }
  const NOTE = '按 decode_csa.py 源码静态整理；不代表运行采集结果。';
  function evidence(type, payload) {
    const fields = Object.keys(payload).map(function (k) { return type + '.' + k; });
    return fields.length ? [{ kind: 'fact', origin: 'source', fields: fields, note: NOTE }] : [{ kind: 'fact', origin: 'source', fields: [], note: NOTE }];
  }
  const PORTS = function () { return { inputs: { default: {} }, outputs: { default: {} } }; };
  function D(id, name, src, symbols, description, data) {
    return { id: id, type: 'data', name: name, description: description, source: source(src, symbols), provenance: evidence('data', data), ports: PORTS(), data: data };
  }
  function S(id, name, src, symbols, description, state) {
    return { id: id, type: 'state', name: name, description: description, source: source(src, symbols), provenance: evidence('state', state), ports: PORTS(), state: state };
  }
  function O(id, name, src, api, description, category, symbols) {
    return { id: id, type: 'operation', name: name, description: description, source: source(src, symbols), provenance: evidence('operation', {}), ports: PORTS(),
      operation: { semantics: description, category: category, api: api } };
  }
  function C(id, name, src, api, description, category, child) {
    const n = O(id, name, src, api, description, category);
    n.type = 'composite';
    delete n.operation;
    n.composite = { semantics: description, category: category, api: api, subgraph: child.graph, portMappings: child.portMappings };
    return n;
  }
  function E(s, t, label, type) {
    return { id: s + '>' + t + ':' + label, s: s, t: t, type: type || 'data', sourcePort: 'default', targetPort: 'default', label: label };
  }
  function frame(id, nodes, edges, first, last) {
    return { graph: { schemaVersion: 2, id: id, nodes: nodes, edges: edges, groups: [] },
      portMappings: { inputs: { default: { node: first, port: 'default', relation: 'view' } }, outputs: { default: { node: last, port: 'default', relation: 'view' } } } };
  }
  // 模块（计算 / 通信 / 依赖）直接复用 lens-data 的文字
  function op(id, category, api) {
    const m = MOD[id];
    return O(id, m.label, m.src, api, m.what + ' ' + m.how, category);
  }

  // ── Sparse Attention 内部：四个 Kernel（decode_sparse_attn_csa.py） ──
  const attentionChild = frame('sparse-attention', [
    O('kv_touch', 'kv_touch', [[SPARSE, 127, 132]], ['pl.at'], 'WAR 标记：一行自拷贝，让 ori_kv 带上 add_inout，保住与外层 KV 写回的依赖边。', 'synchronization'),
    O('qk_plan', 'qk_plan', [[SPARSE, 134, 196]], ['pl.at', 'pl.row_max', 'pl.write'], '生成 sparse_bias 与块有效标记，把 T×5 个 (token, 稀疏块) 排成 qk_order：非空块在前、空块追加。', 'planning'),
    D('qk_order', 'qk_order', [[SPARSE, 139, 139]], ['qk_order'], '排序后的 Work Item 队列：非空块在前、空块在尾，空块不删除。',
      { role: 'metadata', granularity: 'tensor', symbol: 'qk_order', dtype: 'INT32', shape: { allocated: ['QK_ITEMS'] }, storage: { space: 'GM' } }),
    O('qk_pv', 'qk_pv · SPMD × 24', [[SPARSE, 198, 313]], ['pl.spmd', 'pl.matmul', 'pl.pipeline'], '24 条 Lane grid-stride 领取 Work Item；非空块拼 KV → Q·Kᵀ → 局部 softmax → P·V，空块只写初值。', 'matmul'),
    O('rope_cs', 'rope_cs', [[SPARSE, 315, 346]], ['pl.at', 'pl.gather'], '为 merge_norm 的输出 RoPE 生成 cos / sin 与 j^1 交换索引。', 'position')
  ], [E('kv_touch', 'qk_plan', 'WAR', 'control'), E('qk_plan', 'qk_order', 'qk_order'), E('qk_order', 'qk_pv', 'qk_plan_tid'), E('qk_pv', 'rope_cs', '返回 qk_tid · rope_tid', 'control')], 'kv_touch', 'rope_cs');

  // ── Merge · Publish 内部（decode_csa.py 395–477） ──
  const publishChild = frame('merge-publish', [
    O('merge', 'Merge 5 个稀疏块', [[CSA, 412, 430]], ['pl.pipeline', 'pl.maximum', 'pl.exp'], '按 online softmax 规则合并 5 个稀疏块的 m / l / o。', 'reduction'),
    O('normalize', 'Sink · Normalize', [[CSA, 432, 436]], ['pl.row_expand_div'], '加上 attn_sink 后除以分母，得到归一化输出。', 'normalization'),
    O('out_rope', 'Output RoPE', [[CSA, 438, 444]], ['pl.gather', 'pl.col_expand_mul'], '对 RoPE 维做逆旋转，再与 NoPE 维拼回 BF16。', 'position'),
    O('pack', 'Pack by Group', [[CSA, 446, 453]], ['pl.unroll'], '按 O Group 把每个头写进 attention_grouped 的对应行列。', 'reshape'),
    O('put', 'Put + Notify', [[CSA, 455, 477]], ['pld.tensor.put', 'pld.system.notify'], '把 Group 行直接写进目标 Rank 的 attention_window，再 AtomicAdd 通知对端。', 'communication')
  ], [E('merge', 'normalize', 'm / l / o'), E('normalize', 'out_rope', 'n_full'), E('out_rope', 'pack', 'BF16 heads'), E('pack', 'put', 'attention_grouped')], 'merge', 'put');

  const nodes = [
    D('x_hc', 'x_hc', [[CSA, 153, 153]], ['x_hc'], '本 Rank 的 HC 多路 hidden 输入。',
      { role: 'input', granularity: 'tensor', symbol: 'x_hc', dtype: 'FP32', shape: { allocated: ['T', 'HC_MULT', 'D'] }, storage: { space: 'GM' } }),
    op('pre', 'normalization', ['hc_pre', 'rms_norm']),
    D('x_normed_t', 'x_normed_t · Local', [[CSA, 273, 275]], ['x_normed_t'], '本 Rank 的归一化 token 行：Q Projection 与 Indexer 只读这一份。',
      { role: 'activation', granularity: 'tensor', symbol: 'x_normed_t', dtype: 'BF16', shape: { allocated: ['T', 'D'] }, storage: { space: 'GM' } }),
    op('rope', 'position', ['pl.at', 'pl.gather']),
    op('gather', 'communication', ['decode_cp_token_allgather_step']),
    D('x_normed_full', 'x_normed_full · Gathered', [[CSA, 279, 289]], ['x_normed_full'], 'AllGather 后的整组 token：行 [0,T) 来自 Rank 0，[T,2T) 来自 Rank 1。KV 与两路 Compressor 读这一份。',
      { role: 'activation', granularity: 'tensor', symbol: 'x_normed_full', dtype: 'BF16', shape: { allocated: ['2T', 'D'] }, storage: { space: 'GM' } }),
    op('q', 'projection', ['rope_prepare', 'q_proj_rope']),
    D('q_tensor', 'Q · qr', [[CSA, 221, 221], [CSA, 292, 293]], ['q', 'qr', 'qr_scale'], 'Q 供稀疏注意力使用；量化 qr / qr_scale 供 Indexer 打分。',
      { role: 'activation', granularity: 'tensor', symbol: 'q', dtype: 'BF16', shape: { allocated: ['T', 'H', 'HEAD_DIM'] }, storage: { space: 'GM' } }),
    op('kv', 'projection', ['rope_prepare', 'kv_proj_rope']),
    D('kv_full', 'kv_full', [[CSA, 291, 291], [CSA, 315, 319]], ['kv_full'], '整组 token 的新 KV 行，随后按 slot mapping 写进 Original KV Cache。',
      { role: 'activation', granularity: 'tensor', symbol: 'kv_full', dtype: 'BF16', shape: { allocated: ['2T', 'HEAD_DIM'] }, storage: { space: 'GM' } }),
    op('write', 'store', ['pl.spmd', 'pl.read']),
    S('original', 'Original KV Cache', [[CSA, 186, 186]], ['kv_cache'], MOD.original.what + ' ' + MOD.original.why,
      { initial: '前序 decode step 写入的分页 KV', update: 'KV Writeback 按 ori_slot_mapping 写入本 step 的 2T 行', dtype: 'BF16', symbols: ['kv_cache'] }),
    op('main', 'compress', ['compressor_ratio4']),
    S('compressed', 'Compressed KV Cache', [[CSA, 187, 188]], ['cmp_kv', 'cmp_block_table'], MOD.compressed.what + ' ' + MOD.compressed.why,
      { initial: '前序 step 压缩出的 KV 页', update: 'Main Compressor 每凑满 4 个 token 写一行压缩 KV', dtype: 'BF16', symbols: ['cmp_kv'] }),
    op('idxcomp', 'compress', ['indexer_compressor']),
    S('indexcache', 'Indexer KV Cache', [[CSA, 189, 191]], ['idx_kv_cache', 'idx_kv_scale'], MOD.indexcache.what + ' ' + MOD.indexcache.why,
      { initial: '前序 step 的量化索引页', update: 'Indexer Compressor 写入 INT8 索引与 FP32 scale', dtype: 'INT8', symbols: ['idx_kv_cache', 'idx_kv_scale'] }),
    op('indexer', 'topk', ['indexer']),
    D('idx_topk', 'idx_topk', [[CSA, 222, 223], [CSA, 375, 382]], ['idx_topk', 'idx_topk_scores'], '每个本地 token 选出的 512 个压缩位置，决定 qk_pv 的 SB1–SB4 读哪些行。',
      { role: 'metadata', granularity: 'tensor', symbol: 'idx_topk', dtype: 'INT32', shape: { allocated: ['T', 'IDX_TOPK'] }, storage: { space: 'GM' } }),
    op('ready', 'synchronization', ['pl.system.task_dummy']),
    C('attention', 'Sparse Attention', [[CSA, 384, 393]], ['sparse_attn_csa'], MOD.attention.what + ' ' + MOD.attention.how, 'attention', attentionChild),
    D('sparse_blk', 'Partial · m / l / o', [[CSA, 384, 393]], ['sparse_blk_mi', 'sparse_blk_li', 'sparse_blk_oi'], '每个 (token, 头块, 稀疏块) 的局部 softmax 结果，等 Merge 合并。',
      { role: 'intermediate', granularity: 'tile', symbol: 'sparse_blk_oi', dtype: 'FP32', shape: { allocated: ['T·4·5·16', 'HEAD_DIM'] }, storage: { space: 'GM' } }),
    C('publish', 'Merge · Publish', [[CSA, 395, 477]], ['pl.spmd', 'pld.tensor.put'], MOD.publish.what + ' ' + MOD.publish.how, 'communication', publishChild),
    D('attention_window', 'attention_window', [[CSA, 209, 209]], ['attention_window'], '各 Rank 收到的 Group 行：本 Rank 4 个 O Group × 整组 token。',
      { role: 'activation', granularity: 'tensor', symbol: 'attention_window', dtype: 'BF16', shape: { allocated: ['ATTENTION_WINDOW_ROWS', 'O_GROUP_IN'] }, storage: { space: 'GM' } }),
    op('a2a', 'communication', ['o_group_a2a']),
    op('oproj', 'communication', ['o_proj_reduce_scatter']),
    D('attn_out', 'attn_out', [[CSA, 296, 296]], ['attn_out'], 'ReduceScatter 后回到本 Rank 的注意力输出。',
      { role: 'activation', granularity: 'tensor', symbol: 'attn_out', dtype: 'BF16', shape: { allocated: ['T', 'D'] }, storage: { space: 'GM' } }),
    op('post', 'normalization', ['hc_post']),
    D('x_out', 'x_out', [[CSA, 206, 206]], ['x_out'], '本层输出：本 Rank 的 T 行。',
      { role: 'output', granularity: 'tensor', symbol: 'x_out', dtype: 'FP32', shape: { allocated: ['T', 'HC_MULT', 'D'] }, storage: { space: 'GM' } })
  ];

  const edges = [
    E('x_hc', 'pre', 'x_hc'), E('pre', 'x_normed_t', 'x_normed_t'), E('x_normed_t', 'gather', 'T 行'), E('gather', 'x_normed_full', '2T 行'),
    E('x_normed_t', 'q', 'Local'), E('q', 'q_tensor', 'q · qr'),
    E('x_normed_full', 'kv', 'Gathered'), E('kv', 'kv_full', 'kv_full'), E('kv_full', 'write', 'kv_full'), E('write', 'original', 'ori_slot_mapping', 'state'),
    E('x_normed_full', 'main', 'Gathered'), E('main', 'compressed', 'cmp_kv', 'state'),
    E('x_normed_full', 'idxcomp', 'Gathered'), E('idxcomp', 'indexcache', 'idx_kv_cache', 'state'),
    E('rope', 'kv', 'late_dep', 'control'), E('rope', 'main', 'late_dep', 'control'), E('rope', 'idxcomp', 'late_dep', 'control'),
    E('x_normed_t', 'indexer', 'Local'), E('q_tensor', 'indexer', 'qr'), E('indexcache', 'indexer', 'idx_kv_cache', 'state'), E('idxcomp', 'indexer', 'idx_cache_write_tid', 'control'),
    E('indexer', 'idx_topk', 'idx_topk'),
    E('write', 'ready', 'ori_cache_write_tid', 'control'), E('main', 'ready', 'cmp_cache_write_tid', 'control'),
    E('ready', 'attention', 'cache_ready_dep', 'control'), E('q_tensor', 'attention', 'q'), E('idx_topk', 'attention', 'idx_topk'),
    E('original', 'attention', 'SB0 滑窗', 'state'), E('compressed', 'attention', 'SB1–4 压缩', 'state'),
    E('attention', 'sparse_blk', 'm / l / o'), E('sparse_blk', 'publish', 'qk_tid'), E('publish', 'attention_window', 'Put → 目标 Rank'),
    E('attention_window', 'a2a', 'Group 行'), E('a2a', 'oproj', 'attention_local_groups'), E('oproj', 'attn_out', 'ReduceScatter'),
    E('attn_out', 'post', 'attn_out'), E('x_hc', 'post', 'residual'), E('post', 'x_out', 'x_out')
  ];

  function region(id, name, members, a, b) {
    return { id: id, type: 'region', name: name, description: name, members: members, source: { file: CSA, ranges: [{ start: a, end: b }], symbols: [] },
      provenance: [{ kind: 'resolved', origin: 'source', fields: ['members'], note: '按算子语义组织的阶段分组，不是硬件或时间边界。' }] };
  }
  const groups = [
    region('s1', '输入准备 · Input Preparation', ['x_hc', 'pre', 'x_normed_t', 'rope', 'gather', 'x_normed_full'], 218, 289),
    region('s2', 'KV 与索引准备 · KV & Index', ['q', 'q_tensor', 'kv', 'kv_full', 'write', 'original', 'main', 'compressed', 'idxcomp', 'indexcache', 'indexer', 'idx_topk', 'ready'], 291, 382),
    region('s3', '稀疏注意力 · Sparse Attention', ['attention', 'sparse_blk'], 384, 393),
    region('s4', '输出重分布 · Redistribution', ['publish', 'attention_window', 'a2a'], 395, 486),
    region('s5', '输出投影 · Output Projection', ['oproj', 'attn_out', 'post', 'x_out'], 487, 499)
  ];

  const lines = function (file) { return (root.LENS_SOURCE.files[file].text.replace(/\n$/, '').split('\n')).length; };
  const sourceFiles = {}; sourceFiles[CSA] = { lineCount: lines(CSA) }; sourceFiles[SPARSE] = { lineCount: lines(SPARSE) };
  const c = function (value, line) { return { value: value, line: line }; };
  root.LENS_CSA_GRAPH = {
    schemaVersion: 2, id: 'decode-csa', sourceFiles: sourceFiles,
    constants: { T: c(256, 101), D: c(4096, 103), H: c(64, 104), HEAD_DIM: c(512, 105), HC_MULT: c(4, 111), IDX_TOPK: c(512, 116) },
    externalConstants: {}, runtimeDims: ['2T', 'QK_ITEMS', 'T·4·5·16', 'ATTENTION_WINDOW_ROWS', 'O_GROUP_IN'],
    nodes: nodes, edges: edges, groups: groups
  };
  // 图节点 → 详情：模块直接用 lens-data，数据 / 状态 / 子节点用模型自己的说明
  root.LENS_GRAPH_INFO = function (id) {
    const rootId = String(id).split('/')[0], childId = String(id).split('/')[1];
    const owner = nodes.filter(function (n) { return n.id === rootId; })[0];
    let n = owner;
    if (childId && owner && owner.composite) n = owner.composite.subgraph.nodes.filter(function (x) { return x.id === childId; })[0] || owner;
    if (!n) return null;
    return { id: id, rootId: rootId, node: n, module: !childId && MOD[rootId] ? MOD[rootId] : null };
  };
})(window);
