/* Read-only UI projection of the semantic graph. No geometry or semantic defaults.
 *
 * 本文件只做「语义数据 → 展示数据」的投影，不新增语义、不修改输入模型：
 *  - node / edge / inspector：既有投影，保持原字段与含义；
 *  - card：DATA / STATE 卡片要显示的一行 Shape 与一行元信息（Step 2）；
 *  - region：Allocated / Valid 的二维有效区域示意（只在可可靠解释时给出）；
 *  - tensorDetail：面向 Tensor 理解的结构化 Inspector 数据（Step 2）；
 *  - transformation：从子图节点、端口与依赖边推导的输入 → 操作 → 输出 Shape 关系。
 * 所有形状都来自语义节点已声明的字段：constants 把常量名换成静态求值结果，
 * runtimeDims 中的符号保留原名（不套用典型配置），缺失字段一律照缺省处理。
 * 这里不对表达式求值、不使用 eval，也不为缺失字段补默认值。 */
(function (root) {
  'use strict';
  const own = (value, key) => Object.prototype.hasOwnProperty.call(value, key);
  const human = (value) => {
    if (Array.isArray(value)) return value.map(human).join(' · ');
    if (value && typeof value === 'object') return Object.entries(value).map(([key, item]) => `${key}: ${human(item)}`).join(' · ');
    return String(value);
  };
  function sourceRanges(node) {
    if (node.source?.ranges) return node.source.ranges.map(({ start, end }) => [start, end]);
    if (Array.isArray(node.sourceRanges)) return node.sourceRanges.map((range) => Array.isArray(range) ? range : [range.start, range.end]);
    const lines = String(node.lines || '').match(/\d+(?:\s*[-–]\s*\d+)?/g) || [];
    return lines.map((range) => range.split(/\s*[-–]\s*/).map(Number)).map((pair) => pair.length === 1 ? [pair[0], pair[0]] : pair);
  }
  const sourceLine = (node) => sourceRanges(node)[node.source?.primaryRange || 0]?.[0];
  function formatSource(source) {
    if (!source) return '';
    const ranges = source.ranges || [];
    return `${source.file}${ranges.length ? ':' + ranges.map(({ start, end }) => start === end ? start : `${start}–${end}`).join(', ') : ''}`;
  }

  /* ================= Tensor / Tile 形状投影 ================= */

  /* 数据角色是源码里的英文枚举，展示时给出中文含义并保留原值。 */
  const ROLE_LABELS = {
    input: '输入', intermediate: '中间结果', cache: 'Cache', output: '输出',
    'input-view': '输入切片', 'cache-view': 'Cache 切片', metadata: '索引 / 控制数据',
  };
  const roleLabel = (role) => (ROLE_LABELS[role] ? `${ROLE_LABELS[role]}（${role}）` : role);
  const GRANULARITY_LABELS = {
    tensor: 'Tensor', tile: 'Tile', page: 'Page', token: 'Token', stack: 'Stack',
    scalar: 'Scalar', bundle: 'Bundle',
  };
  const granularityLabel = (value) => (GRANULARITY_LABELS[value] ? GRANULARITY_LABELS[value] : value);
  /* 只用于把常量折叠成可读文本：不做表达式求值，取不到就照抄原文。 */
  const numberText = (value) => (Number.isInteger(value) ? String(value) : String(Number(value.toFixed(6))));
  const constantOf = (model, name) => (model && model.constants && model.constants[name]) || null;
  const isRuntime = (model, name) => !!model && Array.isArray(model.runtimeDims) && model.runtimeDims.includes(name);
  function dimension(model, value) {
    if (typeof value === 'number') return { name: String(value), text: numberText(value), value, origin: 'literal', runtime: false };
    const entry = constantOf(model, value);
    if (entry) return { name: String(value), text: numberText(entry.value), value: entry.value, line: entry.line, origin: 'constant', runtime: false };
    return { name: String(value), text: String(value), value: null, origin: isRuntime(model, value) ? 'runtime' : 'symbol', runtime: isRuntime(model, value) };
  }
  /* 一个维度轴：常量给数值，运行时符号保留名字，其余照抄。 */
  function axis(model, values) {
    if (!Array.isArray(values) || !values.length) return null;
    const dims = values.map((value) => dimension(model, value));
    return { dims, rank: dims.length,
      text: `[${dims.map((item) => item.text).join(', ')}]`,
      symbol: `[${dims.map((item) => item.name).join(', ')}]`,
      numeric: dims.every((item) => item.value !== null),
      runtime: dims.filter((item) => item.runtime).map((item) => item.name) };
  }
  /* 同时给出 allocated 与 valid 两个轴；源码只声明其一时只返回那一个。 */
  function axes(model, payload) {
    if (!payload || !payload.shape) return null;
    const allocated = axis(model, payload.shape.allocated);
    const valid = axis(model, payload.shape.valid);
    return allocated || valid ? { allocated, valid } : null;
  }
  /* 卡片上优先显示 allocated（Tensor / Tile 的形状），并保留 valid 的说明。 */
  function shapeLine(view) {
    if (!view) return null;
    const primary = view.allocated || view.valid;
    const parts = [];
    if (view.allocated) parts.push(`allocated ${view.allocated.symbol} = ${view.allocated.text}`);
    if (view.valid) parts.push(`valid ${view.valid.symbol} = ${view.valid.text}`);
    return { text: primary.text, symbol: primary.symbol, title: parts.join('；'), numeric: primary.numeric, runtime: primary.runtime };
  }
  /* DATA / STATE 卡片内容：Shape 一行 + dtype·粒度·内存一行。 */
  function card(semantic, model) {
    const type = semantic.type, payload = semantic[type] || {};
    if (type !== 'data' && type !== 'state') return null;
    const line = shapeLine(axes(model, payload));
    const meta = [payload.dtype, payload.granularity && granularityLabel(payload.granularity), payload.storage && payload.storage.space].filter(Boolean).join(' · ');
    return { shapeText: line ? line.text : null, shapeSymbol: line ? line.symbol : null, shapeTitle: line ? line.title : '',
      meta, tensor: type === 'data', undeclared: !line };
  }
  /* Overlay 属性只读现有语义节点与依赖；开关留在 UI，不参与布局。 */
  function overlays(item, { model, qualifiedId } = {}) {
    const semantic = item.semantic || item, payload = semantic[semantic.type] || {};
    const view = axes(model, payload), shape = shapeLine(view);
    const found = locate(model, qualifiedId || item.id || semantic.id);
    const transform = transformation(item, { model, qualifiedId });
    const resultNodes = found ? (found.graph.edges || []).filter((edge) => edge.s === found.node.id && edge.type === 'data')
      .map((edge) => found.graph.nodes.find((node) => node.id === edge.t)).filter(Boolean) : [];
    const outputs = transform ? transform.results.map((entry) => entry.dtype).filter(Boolean)
      : resultNodes.filter((node) => node.type === 'data').map((node) => node.data.dtype).filter(Boolean);
    const inputTypes = semantic.type === 'operation' && (payload.api || []).includes('pl.cast') && found
      ? (found.graph.edges || []).filter((edge) => edge.t === found.node.id && edge.type === 'data').map((edge) => found.graph.nodes.find((node) => node.id === edge.s)?.data?.dtype).filter(Boolean) : [];
    const precision = inputTypes.length && outputs.length ? `${[...new Set(inputTypes)].join('/')} → ${[...new Set(outputs)].join('/')}` : payload.dtype ? payload.dtype : ['operation', 'composite'].includes(semantic.type)
      ? [...new Set(outputs)].map((dtype) => `输出 ${dtype}`).join(' · ') : '';
    const incomingTypes = found ? (found.graph.edges || []).filter((edge) => edge.t === found.node.id || edge.s === found.node.id).map((edge) => edge.type) : [];
    return { shape: shape?.text || null, shapeTitle: shape?.title || '',
      valid: view?.valid && view.valid.text !== view.allocated?.text ? view.valid.text : null,
      granularity: payload.granularity ? granularityLabel(payload.granularity) : '',
      precision, memory: payload.storage?.space || '', executionUnit: payload.executionUnit || '',
      control: semantic.type === 'state' || payload.role === 'metadata' || (payload.relations || []).some((entry) => entry.kind === 'control') || incomingTypes.some((type) => ['control', 'state', 'feedback'].includes(type)),
      evidence: (semantic.provenance || []).map((entry) => ({ kind: entry.kind, fields: [...entry.fields], note: entry.note || entry.origin })) };
  }
  /* 二维有效区域：只在 Allocated 与 Valid 都取到数值、尺度一致时才可画。
     动态符号维度会被明确拒绝——不同尺度的两个范围画成比例图就是错的。 */
  function regionView(model, payload) {
    const view = axes(model, payload);
    if (!view || !view.allocated || !view.valid) return null;
    if (view.allocated.rank !== 2) return { skip: '有效区域示意只针对二维 Tensor / Tile。' };
    const unknown = [...view.allocated.dims, ...view.valid.dims].filter((item) => item.value === null);
    if (unknown.length) {
      const runtime = unknown.filter((item) => item.runtime).map((item) => item.name);
      return { skip: `${[...new Set(unknown.map((item) => item.name))].join('、')} 在源码里${runtime.length ? '是运行时维度' : '不是可静态求解的常量'}，与 Valid 的尺度不同，按比例绘制会给出错误的比例；请以文字 Shape 为准。` };
    }
    const rows = view.allocated.dims[0], cols = view.allocated.dims[1];
    const validRows = view.valid.dims[0], validCols = view.valid.dims[1];
    if (!(rows.value > 0 && cols.value > 0) || validRows.value > rows.value || validCols.value > cols.value) {
      return { skip: 'Valid 超出 Allocated，无法绘制有效区域。' };
    }
    const r = rows.value, c = cols.value, vr = validRows.value, vc = validCols.value;
    return {
      rows: r, cols: c, validRows: vr, validCols: vc, full: vr === r && vc === c,
      headline: `Allocated ${r} × ${c} · Valid ${vr} × ${vc}`,
      rowLine: vr === r
        ? `行 0–${r - 1} 全部有效（${rows.name} = ${r}）。`
        : `行 0–${vr - 1} 属于当前计算有效区域（${validRows.name} = ${vr} / ${rows.name} = ${r}）；行 ${vr}–${r - 1} 不属于当前计算。`,
      colLine: vc === c
        ? `列 0–${c - 1} 全部有效（${cols.name} = ${c}）。`
        : `列 0–${vc - 1} 属于当前计算有效区域（${validCols.name} = ${vc} / ${cols.name} = ${c}）。`,
      legend: { valid: '当前计算有效区域', invalid: '非当前计算有效区域' },
      disclaimer: '非有效区域不等于 Padding，也不默认为 0；本算子没有声明这些行列的取值。',
    };
  }

  /* ================= 语义图遍历 ================= */
  function flatten(graph, prefix = '') {
    if (!graph || !Array.isArray(graph.nodes)) return [];
    return graph.nodes.flatMap((node) => [
      { node, id: prefix + node.id, prefix },
      ...(node.composite ? flatten(node.composite.subgraph, `${prefix}${node.id}/`) : []),
    ]);
  }
  const holdsMatmul = (payload) => !!payload && Array.isArray(payload.api) && payload.api.some((api) => api.indexOf('pl.matmul') === 0);
  /* 某个子图在整张模型里的限定前缀；用于区分「同一实体在父图与子图里的两个节点」。 */
  function graphPrefix(model, graph) {
    const walk = (current, prefix) => {
      if (current === graph) return prefix;
      for (const child of current.nodes || []) {
        if (!child.composite?.subgraph) continue;
        const hit = walk(child.composite.subgraph, `${prefix}${child.id}/`);
        if (hit !== null) return hit;
      }
      return null;
    };
    return model && graph ? walk(model, '') || '' : '';
  }

  /* ================= 节点 / 边投影 ================= */
  function node(semantic, context = {}) {
    const p = semantic[semantic.type] || {}, type = semantic.type;
    const sub = type === 'data' ? [p.dtype, p.granularity].filter(Boolean).slice(0, 2).join(' · ') :
      type === 'state' ? [p.dtype, p.granularity && `跨 ${p.granularity}`].filter(Boolean).slice(0, 2).join(' · ') :
      type === 'composite' ? [p.category, p.subgraph && `${p.subgraph.nodes.length} 个节点`].filter(Boolean).join(' · ') :
      [p.category, p.api?.[0]].filter(Boolean).slice(0, 2).join(' · ');
    const id = context.id || semantic.id;
    const stats = context.model ? card(semantic, context.model) : null;
    /* 语义色键：数据/状态按 role 归入中性 IO 层，操作/复合按 category 分层。
       键名沿用 model-graphviz 的 sem:* 约定，取值留空时退回操作层的默认色。 */
    const sem = type === 'data' || type === 'state'
      ? `io:${p.role || type}`
      : `op:${p.category || type}`;
    return { ...semantic, semantic, id, parentId: context.parentId, sem,
      semanticType: type, type, kind: type === 'data' || type === 'state' ? 'data' : 'operation',
      badge: type.toUpperCase(), title: semantic.name, sub, detail: semantic.description,
      api: Array.isArray(p.api) ? p.api.join(' · ') : '', lines: (semantic.source?.ranges || []).map(({ start, end }) => start === end ? String(start) : `${start}–${end}`).join('、'),
      sourceRanges: sourceRanges(semantic), sourceLine: sourceLine(semantic),
      ...(stats ? { card: stats } : {}),
      overlay: overlays(semantic, { model: context.model, qualifiedId: id }) };
  }
  function edge(semantic, context = {}) {
    return { ...semantic, semantic, id: context.id || semantic.id,
      s: context.sourceId || semantic.s, t: context.targetId || semantic.t,
      internal: !!context.internal,
      route: semantic.route || (semantic.type === 'feedback' ? 'feedback' : semantic.type) };
  }
  function locate(model, qualifiedId) {
    if (!model || !qualifiedId) return null;
    const parts = qualifiedId.split('/'); let graph = model, found;
    for (let i = 0; i < parts.length; i++) {
      found = graph.nodes?.find((item) => item.id === parts[i]);
      if (!found) return null;
      if (i < parts.length - 1) graph = found.composite?.subgraph;
    }
    return { node: found, graph };
  }
  /* 依赖方向一律从边推导，不在数据节点里重复保存。 */
  function connectionRows(graph, id) {
    if (!graph) return [];
    const label = (nodeId) => {
      const found = (graph.nodes || []).find((item) => item.id === nodeId);
      return found && found.name !== nodeId ? `${found.name}（${nodeId}）` : nodeId;
    };
    const incoming = (graph.edges || []).filter((item) => item.t === id);
    const outgoing = (graph.edges || []).filter((item) => item.s === id);
    const rows = [];
    const producers = incoming.filter((item) => item.type === 'data').map((item) => label(item.s));
    const consumers = outgoing.filter((item) => item.type === 'data').map((item) => label(item.t));
    if (producers.length) rows.push({ label: 'Producer', value: producers.join('、') });
    if (consumers.length) rows.push({ label: 'Consumer', value: consumers.join('、') });
    const other = [...incoming.map((item) => ({ item, side: 'in' })), ...outgoing.map((item) => ({ item, side: 'out' }))]
      .filter(({ item }) => item.type !== 'data')
      .map(({ item, side }) => `${item.type}：${label(side === 'in' ? item.s : item.t)}（${item.label}）`);
    if (other.length) rows.push({ label: '控制 / 状态', value: other.join('、') });
    return rows;
  }
  function inspector(item, { model, qualifiedId } = {}) {
    const semantic = item.semantic || item;
    const p = semantic[semantic.type] || {};
    const sections = [];
    const add = (title, entries) => {
      const rows = entries.filter(([, value]) => value !== undefined && value !== null && value !== '' &&
        !(Array.isArray(value) && !value.length) && !(typeof value === 'object' && !Object.keys(value).length)).map(([label, value]) => ({ label, value: human(value) }));
      if (rows.length) sections.push({ title, rows });
    };
    add('语义', [['类型', semantic.type], ['名称', semantic.name], ['描述', semantic.description]]);
    // Render every populated payload key; new engineering attributes remain visible.
    add('属性', Object.entries(p).filter(([key]) => !['subgraph', 'portMappings', 'producer', 'consumers', 'inputs', 'outputs'].includes(key)));
    if (semantic.ports) add('端口', ['inputs', 'outputs'].flatMap((side) => Object.entries(semantic.ports[side] || {}).map(([name, value]) => [`${side}.${name}`, Object.keys(value).length ? value : name])));
    if (p.portMappings) add('端口映射', ['inputs', 'outputs'].flatMap((side) => Object.entries(p.portMappings[side] || {}).map(([name, value]) => [`${side}.${name}`, value])));
    if (semantic.source) add('源码', [['位置', formatSource(semantic.source)], ['符号', semantic.source.symbols?.length ? semantic.source.symbols : undefined]]);
    if (semantic.provenance?.length) add('来源依据', semantic.provenance.map((entry, i) => [`${i + 1}. ${entry.kind}`, { origin: entry.origin, fields: entry.fields, ...(own(entry, 'note') ? { note: entry.note } : {}) }]));
    const found = locate(model, qualifiedId || item.id);
    if (found) {
      const id = found.node.id, edges = found.graph.edges || [];
      const incoming = edges.filter((edge) => edge.t === id);
      const outgoing = edges.filter((edge) => edge.s === id);
      if (semantic.type === 'state') add('状态演化', [
        ['含义', semantic.description], ['更新公式', p.update], ['反馈', p.feedback],
        ['输入来源', incoming.map((edge) => `${found.graph.nodes.find((node) => node.id === edge.s)?.name || edge.s} · ${edge.label}`)],
        ['输出去向', outgoing.map((edge) => `${found.graph.nodes.find((node) => node.id === edge.t)?.name || edge.t} · ${edge.label}`)],
        ['数据类型', '源码声明与静态推导；未选择具体 Task 时不表示运行时数值'],
      ]);
      add('依赖', [
        ['生产者', incoming.filter((edge) => edge.type === 'data').map((edge) => edge.s)],
        ['消费者', outgoing.filter((edge) => edge.type === 'data').map((edge) => edge.t)],
        ['输入', incoming.map((edge) => `${edge.s}.${edge.sourcePort} → ${edge.targetPort} (${edge.type})`)],
        ['输出', outgoing.map((edge) => `${edge.sourcePort} → ${edge.t}.${edge.targetPort} (${edge.type})`)]
      ]);
    }
    return sections;
  }

  function inspectorEdge(edge, { model } = {}) {
    const parts = edge.id.split('/');
    let graph = model;
    for (const id of parts.slice(0, -1)) graph = graph?.nodes.find((node) => node.id === id)?.composite?.subgraph;
    if (!graph) return [];
    const semantic = edge.semantic || edge;
    const source = graph.nodes.find((node) => node.id === semantic.s);
    const target = graph.nodes.find((node) => node.id === semantic.t);
    const rows = [
      { label: '含义', value: edge.type === 'feedback' ? '更新后的状态经循环传给下一 Stack 的 Previous' : '状态参与当前计算' },
      { label: '来源', value: source?.name || semantic.s },
      { label: '去向', value: target?.name || semantic.t },
      { label: '依赖', value: semantic.label },
      { label: '更新公式', value: source?.state?.update || target?.state?.update || source?.description },
      { label: '数据类型', value: '源码声明与静态推导；不表示运行时状态数值' },
    ].filter((row) => row.value);
    return [{ title: edge.type === 'feedback' ? '反馈' : '状态依赖', rows },
      { title: '源码', rows: [source, target].filter(Boolean).map((node) => ({
        label: node.name, value: formatSource(node.source),
      })) }];
  }

  /* ================= Tensor Inspector（结构化数据） ================= */
  /* 供 Inspector 使用：只装配内容，不生成 HTML。 */
  function tensorDetail(item, options = {}) {
    const { model, qualifiedId } = options;
    const semantic = item.semantic || item;
    const type = semantic.type, payload = semantic[type] || {};
    const found = locate(model, qualifiedId || item.id);
    const id = found ? found.node.id : semantic.id;
    const graph = found ? found.graph : null;
    const view = type === 'data' || type === 'state' ? axes(model, payload) : null;
    const line = shapeLine(view);
    const groups = [];
    if (type === 'data' || type === 'state') {
      const rows = [
        { label: '数据角色', value: payload.role ? roleLabel(payload.role) : '跨 Stack 状态（不是单轮张量数据）' },
        { label: 'Granularity', value: payload.granularity ? granularityLabel(payload.granularity) : '源码未声明' },
      ];
      if (view?.allocated) rows.push({ label: 'Allocated Shape', value: view.allocated.text, note: view.allocated.symbol !== view.allocated.text ? view.allocated.symbol : '', mono: true });
      if (view?.valid) rows.push({ label: 'Valid Shape', value: view.valid.text, note: view.valid.symbol !== view.valid.text ? view.valid.symbol : '', mono: true });
      if (!view) rows.push({ label: 'Shape', value: '源码未声明' });
      rows.push({ label: 'Dtype', value: payload.dtype || '源码未声明' });
      if (payload.storage) rows.push({ label: 'Memory Space', value: payload.storage.space });
      if (payload.symbol) rows.push({ label: '源码变量', value: payload.symbol, mono: true });
      groups.push({ id: 'metrics', title: 'Shape · Dtype · Memory', rows });
      const shapeOrigin = (semantic.provenance || [])
        .filter((entry) => (entry.fields || []).some((field) => field === `${type}.shape` || field.startsWith(`${type}.shape.`)))
        .map((entry) => ({ label: `Shape 来源（${entry.kind}）`, value: entry.note || entry.origin }));
      const dynamic = view?.allocated?.runtime || [];
      const originRows = [
        ...(payload.relations || []).map((relation, index) => ({ label: `来源关系 ${index + 1}（${relation.kind}）`, value: `${relation.entity} — ${relation.description}` })),
        ...shapeOrigin,
        ...(dynamic.length ? [{ label: '动态维度', value: `${dynamic.join('、')} 由运行时输入决定，不写成固定值。` }] : []),
        ...connectionRows(graph, id),
      ];
      if (originRows.length) groups.push({ id: 'origin', title: '数据来源与关系', rows: originRows });
    }
    const region = type === 'data' || type === 'state' ? regionView(model, payload) : null;
    /* 不绘图时给出可核对的原因，避免读者以为漏了一行。 */
    const regionNote = !view ? '' : (region && region.skip) ? region.skip
      : view.allocated && !view.valid ? '源码只声明了 Allocated Shape，没有声明 Valid Shape，因此不绘制有效区域。'
        : !view.allocated && view.valid ? '源码只声明了 Valid Shape，没有声明 Allocated Shape，因此不绘制有效区域。' : '';
    const evidence = (semantic.provenance || []).map((entry, index) => ({ index: index + 1, kind: entry.kind, origin: entry.origin, fields: entry.fields || [], note: entry.note }));
    const advanced = [
      ...Object.entries(payload.addressing || {}).map(([key, value]) => ({ label: `addressing.${key}`, value })),
      ...Object.entries(payload)
        .filter(([key]) => !['addressing', 'shape', 'storage', 'relations', 'dtype', 'granularity', 'role', 'entity', 'symbol'].includes(key))
        .map(([key, value]) => ({ label: `${type}.${key}`, value: human(value) })),
      ...(semantic.source?.symbols?.length ? [{ label: 'source.symbols', value: semantic.source.symbols.join('、') }] : []),
      ...(semantic.provenance || []).map((entry, index) => ({ label: `provenance[${index}] ${entry.kind}`, value: `${entry.origin} · ${(entry.fields || []).join('、')}${entry.note ? ' — ' + entry.note : ''}` })),
    ];
    return { id, type, badge: type.toUpperCase(), title: semantic.name, symbol: payload.symbol || '',
      lead: semantic.description, granularity: payload.granularity ? granularityLabel(payload.granularity) : '', groups,
      shapeTitle: line ? line.title : '', region: region && !region.skip ? region : null, regionNote,
      evidence, advanced, transformation: transformation(item, options) };
  }

  /* ================= Shape Transformation ================= */
  /* 输入 / 操作 / 输出全部取自当前作用域的依赖边与数据节点；
     形状规则由两侧操作数维名收缩推导，并与结果节点已声明的 Shape 互校。 */
  function transformation(item, options = {}) {
    const { model, qualifiedId } = options;
    const semantic = item.semantic || item;
    const found = locate(model, qualifiedId || item.id);
    if (!found) return null;
    const target = found.node;
    let operation = null, scope = found.graph;
    if (target.type === 'operation' && holdsMatmul(target.operation)) operation = target;
    else if (target.type === 'composite') {
      const candidates = (target.composite.subgraph.nodes || []).filter((child) => child.type === 'operation' && holdsMatmul(child.operation));
      if (candidates.length === 1) { operation = candidates[0]; scope = target.composite.subgraph; }
    }
    if (!operation) return null;
    const payload = operation.operation;
    const transpose = payload.transposedInput || null;
    const child = new Map((scope.nodes || []).map((entry) => [entry.id, entry]));
    const axisOf = (entry, side) => {
      const childAxes = entry ? axes(model, entry[entry.type] || {}) : null;
      const allocated = childAxes?.allocated || null;
      const valid = childAxes?.valid || null;
      if (!allocated) return null;
      const flipped = side === 'operand' && transpose && transpose.node === entry.id;
      const dims = flipped ? [...allocated.dims].reverse() : allocated.dims;
      return { text: `[${dims.map((d) => d.text).join(', ')}]`, symbol: `[${dims.map((d) => d.name).join(', ')}]`, dims, valid, numeric: allocated.numeric };
    };
    const operands = (scope.edges || []).filter((edge) => edge.t === operation.id && edge.type === 'data').map((edge) => {
      const entry = child.get(edge.s) || null;
      return { id: edge.s, name: entry ? entry.name : edge.s, label: edge.label,
        transposed: !!(transpose && transpose.node === edge.s),
        dtype: entry ? (entry[entry.type] || {}).dtype || null : null,
        shape: axisOf(entry, 'operand'), note: entry ? entry.description : '' };
    });
    const results = (scope.edges || []).filter((edge) => edge.s === operation.id && edge.type === 'data').map((edge) => {
      const entry = child.get(edge.t) || null;
      return { id: edge.t, name: entry ? entry.name : edge.t, label: edge.label,
        dtype: entry ? (entry[entry.type] || {}).dtype || null : null,
        shape: axisOf(entry, 'result'), note: entry ? entry.description : '',
        relation: entry ? (entry[entry.type] || {}).relations?.[0] || null : null };
    });
    const rule = matmulRule(operands, results);
    const notes = [];
    for (const operand of operands) {
      const valid = operand.shape?.valid;
      if (!valid || !operand.shape) continue;
      if (operand.shape.text === valid.text) continue;
      notes.push({ label: '有效范围', value: `${operand.name} 参与本次计算的只有 Valid Shape ${valid.text}（${valid.symbol}）；Allocated ${operand.shape.text} 的其余行列不属于当前计算。` });
    }
    for (const result of results) {
      /* 写入语义（例如「每页 Scores 只写进当前 Stack 的列区间」）直接来自节点描述。 */
      if (result.note) notes.push({ label: `${result.name} 写入语义`, value: result.note });
      const relation = result.relation || null;
      if (!relation) continue;
      /* 结果实体在父图里另有同名数据节点时，说明两者粒度不同、不能互相替代。
         父图节点必须在当前作用域之外，否则找到的会是结果节点自己。 */
      const scopePrefix = graphPrefix(model, scope);
      const inScope = scopePrefix ? (entry) => entry.id.indexOf(scopePrefix) === 0 : (entry) => entry.id === result.id;
      const parent = flatten(model).find((entry) => entry.node.type === 'data' && entry.node.data.entity === relation.entity && entry.node.data.shape && !inScope(entry));
      if (!parent) continue;
      const parentAxes = axes(model, parent.node.data);
      const parentShape = shapeLine(parentAxes);
      const resultNode = child.get(result.id);
      const resultGranularity = resultNode?.data?.granularity;
      notes.push({ label: '粒度区别', value: `${result.name} 是实体 ${relation.entity} 在 ${result.shape ? result.shape.text : '单页'} 上的${relation.kind === 'aggregate' ? '聚合' : '单页'}切片（粒度 ${resultGranularity ? granularityLabel(resultGranularity) : relation.kind}）；父图 ${parent.node.name}（${parent.node.data.symbol || parent.id}）粒度是 ${granularityLabel(parent.node.data.granularity)}，Allocated ${parentShape.text}。两者不是同一个 Tensor。` });
    }
    /* Composite 与内部算子同名时（QK Matmul / PV Matmul）不再重复一遍名字。 */
    const title = target.type === 'composite' && target.name !== operation.name ? `${target.name} · ${operation.name}` : operation.name;
    return { id: operation.id, name: operation.name, title,
      operation: { name: operation.name, category: payload.category, api: payload.api, semantics: payload.semantics },
      transposeNote: transpose ? transpose.note : '', operands, results, rule, notes,
      boundary: target.type === 'composite' ? boundaryOf(found, target) : [] };
  }
  /* 二维 Matmul 的形状规则：A 的末维与 B 的首维收缩，结果维取 A[0] 与 B[1]。
     只在两侧都是二维、且维名确实相等时给出；否则不生成公式。 */
  function matmulRule(operands, results) {
    if (operands.length !== 2) return null;
    const [left, right] = operands;
    if (!left.shape || !right.shape || left.shape.dims.length !== 2 || right.shape.dims.length !== 2) return null;
    const contract = left.shape.dims[1];
    if (contract.name !== right.shape.dims[0].name) return null;
    const dims = [left.shape.dims[0], right.shape.dims[1]];
    const text = `[${dims.map((d) => d.text).join(', ')}]`;
    const symbol = `[${dims.map((d) => d.name).join(', ')}]`;
    const declared = results[0]?.shape?.text || null;
    return { contract: contract.name, text, symbol, declared, consistent: declared ? declared === text : null,
      steps: [`${left.name} 的末维 ${contract.name} 与 ${right.name} 的首维 ${contract.name} 收缩。`,
        `结果维取 ${left.name}[0] = ${dims[0].name}、${right.name}[1] = ${dims[1].name} → ${symbol} = ${text}。`] };
  }
  /* Composite 的边界端口来自父图依赖边，用于说明展开后的输入输出对应关系。 */
  function boundaryOf(found, target) {
    const graph = found.graph;
    return (graph.edges || []).filter((edge) => edge.s === target.id || edge.t === target.id).map((edge) => {
      const outgoing = edge.s === target.id;
      const peerId = outgoing ? edge.t : edge.s;
      const peer = (graph.nodes || []).find((entry) => entry.id === peerId);
      return { direction: outgoing ? 'out' : 'in', type: edge.type, label: edge.label,
        peer: peer ? peer.name : peerId, peerId, transposed: false };
    });
  }

  function synchronization(event, model) {
    const stage = (end) => model.nodes.find((node) => node.id === end.node)?.name || end.node;
    return {
      title: `${event.name} · Event ${event.eventId}`,
      direction: `${event.producer.coreType} → ${event.consumer.coreType}`,
      rows: [
        { label: '同步类型', value: 'Cross-core Synchronization' },
        { label: 'Producer', value: `${stage(event.producer)} / ${event.producer.coreType}` },
        { label: 'Set Pipe', value: event.producer.pipe },
        { label: 'Consumer', value: `${model.nodes.find(node => node.id === (event.consumer.boundary || event.consumer.node))?.name || event.consumer.boundary || event.consumer.node} / ${event.consumer.coreType}` },
        { label: 'WAIT 位置', value: `${event.consumer.boundary || event.consumer.node} · ${event.consumer.anchor}` },
        { label: 'Wait Pipe', value: event.consumer.pipe },
        { label: 'Transfer Buffer', value: `${event.buffer} · GM`, mono: true },
        { label: 'Data', value: `${event.dtype} · ${flatten(model).find((entry) => entry.id === event.tensor)?.node.name || event.tensor}` },
        { label: 'ffts_mode', value: `set: ${event.producer.fftsMode}；wait: 源码未传入` },
        { label: 'Reason', value: event.explanation },
        { label: '证据边界', value: '静态 READY 次序；逻辑 Tick 不表示耗时，WAIT 不表示已发生 Stall。Buffer Slot 复用与 State Feedback 分别建模。' },
      ],
      sources: [['set', event.producer], ['wait', event.consumer]].flatMap(([action, end]) =>
        end.source.ranges.map((range, index) => ({ action, index, range, source: end.source,
          label: `sync_${action} · ${end.source.file}:${range.start}–${range.end}` }))),
    };
  }

  root.PTO_COMPUTATION_PRESENTATION = { node, edge, sourceRanges, sourceLine, formatSource, inspector, inspectorEdge,
    flatten, shapeAxes: axes, shapeLine, card, regionView, tensorDetail, transformation, matmulRule,
    roleLabel, granularityLabel, overlays, synchronization };
})(typeof window === 'undefined' ? globalThis : window);
