/* Source-grounded semantic model. Display strings are derived by the presentation
 * adapter; graph identity, ports and evidence are independent of geometry. */
(function (root) {
  'use strict';
  const FILE = 'paged_attention_pypto.py';
  const parseRanges = (ranges) => ranges.split('、').map((part) => {
    const [start, end = start] = part.split('–').map(Number);
    return { start, end };
  });
  const source = (ranges, symbols = []) => ({ file: FILE, ranges: parseRanges(ranges), symbols });
  /* 源码第 17–50 行声明的编译期常量。Tensor / Tile 的 shape 用这些常量名表达，
     这里给出静态求值结果与声明行号：下游 Tensor Shape 视图可直接解析，
     测试会重新从内嵌源码求值并逐条互校，避免常量与源码漂移。 */
  const CONSTANTS = {
    BATCH: { value: 16, line: 17 }, NUM_HEADS: { value: 40, line: 18 }, NUM_KV_HEADS: { value: 8, line: 19 },
    GROUP: { value: 5, line: 20 }, HEAD_DIM: { value: 128, line: 21 }, BLOCK_SIZE: { value: 128, line: 22 },
    STACK_PAGES: { value: 4, line: 23 }, STACK_TOKENS: { value: 512, line: 24 }, KV_HIDDEN: { value: 1024, line: 25 },
    SEQ_LEN: { value: 3584, line: 26 }, MAX_BLOCKS: { value: 28, line: 27 }, NUM_BLOCKS: { value: 448, line: 28 },
    NUM_TASKS: { value: 128, line: 29 }, ATTN_SPMD_BLOCKS: { value: 24, line: 30 }, PRE_LAUNCH: { value: 2, line: 31 },
    TRANSFER_SLOTS: { value: 3, line: 32 }, FFTS_WORKSPACE_ELEMENTS: { value: 256, line: 33 },
    QK_READY_EVENT: { value: 0, line: 34 }, SOFTMAX_READY_EVENT: { value: 1, line: 35 }, PV_READY_EVENT: { value: 2, line: 36 },
    ROW_TILE: { value: 16, line: 37 }, AIV_ROW_TILE: { value: 8, line: 38 }, AIV_LANE0_ROWS: { value: 2, line: 39 },
    TRANSFER_ROWS: { value: 1152, line: 40 }, SCALE: { value: 1 / Math.sqrt(128), line: 41 }, KCACHE_ROWS: { value: 57344, line: 42 },
    HALF_DIM: { value: 64, line: 43 }, HIDDEN: { value: 5120, line: 44 }, ROPE_CORES: { value: 32, line: 48 },
    ROPE_ITEMS_PER_CORE: { value: 4, line: 49 }, K_RED_ROWS: { value: 8, line: 50 },
  };
  /* 由 constants.py 提供、本文件无法静态求值的常量：只记录来源表达式，不断言数值。
     Q_HEAD_PAD 另由第 63 行 assert 约束为 16。 */
  const EXTERNAL_CONSTANTS = {
    Q_HEAD_PAD: { expression: 'M.q_head_pad', line: 45, assert: 'assert Q_HEAD_PAD == 16', assertLine: 63 },
    HEAD_DIM_INV: { expression: 'M.head_dim_inv', line: 46 },
    EPS: { expression: 'M.eps', line: 47 },
  };
  /* 依赖运行时输入的维度名：只可能出现这些符号，不允许写成典型场景的固定数值。 */
  const RUNTIME_DIMS = ['cache_token_rows', 'query_rows', 'active_batch', 'max_blocks_per_seq', 'lane_rows'];
  const RESOLVED_NOTE = '由本算子已声明的输入、计算和视图映射推导；不代表运行采集结果。';
  const evidence = (type, payload, notes = {}, resolved = []) => {
    if (type === 'data') resolved = [...new Set([...resolved, ...['role', 'granularity', 'entity', 'relations', 'addressing'].filter((key) => key in payload)])];
    const fact = Object.keys(payload).filter((key) => !resolved.includes(key));
    return [
      ...(fact.length ? [{ kind: 'fact', origin: 'source', fields: fact.map((key) => `${type}.${key}`), ...(notes.fact ? { note: notes.fact } : {}) }] : []),
      ...(resolved.length ? [{ kind: 'resolved', origin: 'source', fields: resolved.map((key) => `${type}.${key}`), note: notes.resolved || RESOLVED_NOTE }] : []),
    ];
  };
  const ordinaryPorts = (entity) => ({ inputs: { default: {} }, outputs: { default: entity ? { entity } : {} } });
  function node(id, type, name, ranges, symbols, description, payload, resolved = [], notes = {}) {
    return { id, type, name, description, source: source(ranges, symbols),
      provenance: evidence(type, payload, notes, resolved), ports: ordinaryPorts(payload.entity), [type]: payload };
  }
  const D = (id, name, ranges, symbols, description, data, resolved, notes) => node(id, 'data', name, ranges, symbols, description, data, resolved, notes);
  /* 操作节点可以再挂一条「形状依据」（例如 Matmul 的转置操作数）：它只复述源码里
     已经存在的调用与作用对象，不引入新的计算语义，也不参与节点 / 边的判定。 */
  const O = (id, name, ranges, api, description, category, symbols = [], extra = {}, resolved = [], notes = {}) => node(id, 'operation', name, ranges, symbols, description,
    { semantics: description, category, api, ...extra }, resolved, notes);
  const S = (id, name, ranges, symbols, description, state, resolved, notes) => node(id, 'state', name, ranges, symbols, description, state, resolved, notes);
  const E = (s, t, label, type = 'data', extra = {}) => ({ id: `${s}>${t}:${label}`, s, t, type,
    sourcePort: 'default', targetPort: 'default', label, ...extra });
  function frame(id, nodes, edges, inputs, outputs) {
    const ports = { inputs: {}, outputs: {} }, portMappings = { inputs: {}, outputs: {} };
    for (const side of ['inputs', 'outputs']) {
      for (const [name, binding] of Object.entries(side === 'inputs' ? inputs : outputs)) {
        const spec = typeof binding === 'string' ? { node: binding, relation: 'view' } : binding;
        const { entity, ...mapping } = spec;
        ports[side][name] = entity ? { entity } : {};
        portMappings[side][name] = { port: 'default', ...mapping };
      }
    }
    return { graph: { schemaVersion: 2, id, nodes, edges, groups: [] }, ports, portMappings };
  }
  function C(id, name, ranges, symbols, description, category, api, child) {
    const composite = { semantics: description, category, api, subgraph: child.graph, portMappings: child.portMappings };
    return { ...node(id, 'composite', name, ranges, symbols, description, composite), ports: child.ports };
  }
  const view = (node, entity) => ({ node, entity, relation: 'view' });
  const slice = (node, entity, description) => ({ node, entity, relation: 'slice', description });
  const control = (node, entity, description) => ({ node, entity, relation: 'control', description });
  const aggregate = (node, entity, description) => ({ node, entity, relation: 'aggregate', description });
  /* 分页 Cache 的寻址表达式：读路径取页表 + 页内偏移，写路径取层基址 + slot。
     物理页号由运行时 block_table 决定。 */
  const cacheAccess = {
    logicalPage: 'stack * STACK_PAGES + page_in_stack',
    physicalPage: 'block_table_2d[batch, logical_page]',
    row: 'cache_base + physical_page * BLOCK_SIZE + token_offset',
    column: 'kv_head * HEAD_DIM + dim',
    writeRow: 'cache_row = layer_cache_base_token_rows + write_slot',
    writeColumn: 'cache_col : cache_col + HEAD_DIM = kv_head * HEAD_DIM',
  };
  const cachePayload = (key) => ({ role: 'cache', granularity: 'tensor', entity: `cache.${key}`,
    symbol: key === 'key' ? 'key_cache_bsnd' : 'value_cache_bsnd',
    shape: { allocated: ['cache_token_rows', 'KV_HIDDEN'] },
    storage: { space: 'GM' },
    addressing: { ...cacheAccess } });
  const keyCache = (id, name = 'K Cache') => D(id, name, '17–25、101–107、194–198、261–295',
    ['key_cache_bsnd', 'block_table_2d', 'cache_base', 'col'],
    '分页 Key Cache 的二维访问视图（外部 pl.Tensor，位于 GM）。逻辑页经 block_table 映射到物理页；写入载荷为 BF16（第 198 行），但 Cache 元素类型未在签名或 Tile 标注中声明，故不声明 dtype。',
    cachePayload('key'), ['storage'],
    { fact: 'symbol 与 shape 取自第 101–102 行的 pl.reshape(key_cache, [cache_token_rows, KV_HIDDEN])：KV_HIDDEN = 1024 是源码常量，cache_token_rows 由运行时传入的 Cache 尺寸决定，不能写成固定行数。',
      resolved: 'storage：外部 pl.Tensor 位于 GM（第 253–255 行以 GM writes 描述外部张量写入）；addressing 取自第 194–198、261–295 行的页表与行列寻址；role / granularity / entity 为语义归类。' });
  const valueCache = (id, name = 'V Cache') => D(id, name, '17–25、101–107、210–213、364–429',
    ['value_cache_bsnd', 'block_table_2d', 'cache_base', 'col'],
    '分页 Value Cache 的二维访问视图（外部 pl.Tensor，位于 GM）。与 K Cache 使用相同的逻辑页表和行列寻址；dtype 由第 386–429 行 v0..v3 的 pl.Tile[[BLOCK_SIZE, HEAD_DIM], pl.BF16, …] 标注与第 210–213 行的 BF16 写入推导，函数签名本身未声明。',
    { ...cachePayload('value'), dtype: 'BF16' }, ['storage', 'dtype'],
    { fact: 'symbol 与 shape 取自第 103 行的 pl.reshape(value_cache, [cache_token_rows, KV_HIDDEN])；cache_token_rows 由运行时传入的 Cache 尺寸决定。',
      resolved: 'dtype：第 391–429 行的 v0..v3 以 pl.BF16 Tile 直接从 value_cache_bsnd 载入，第 210–213 行也写入 BF16 载荷；storage：外部 pl.Tensor 位于 GM；addressing / role / granularity / entity 为语义归类。' });
  const normFrame = (isQ) => frame(isQ ? 'q-rmsnorm' : 'k-rmsnorm', [
    O('sum', 'Row Sum', isQ ? '229–229' : '175–175', ['pl.row_sum', 'pl.mul'], '按行求平方和。', 'reduction', [isQ ? 'q_ss' : 'k_ss']),
    O('inv', 'Inverse RMS', isQ ? '230–230' : '176–176', ['pl.recip', 'pl.sqrt', 'pl.add', 'pl.mul'], '由平方和、HEAD_DIM_INV 和 EPS 求逆 RMS。', 'normalization', [isQ ? 'q_inv' : 'k_inv']),
    O('weight', 'Norm Weight', isQ ? '231–234' : '177–180', ['pl.row_expand_mul', 'pl.col_expand_mul'], '应用 head 权重与逆 RMS。', 'normalization', [isQ ? 'q_heads' : 'k_normed']),
  ], [E('sum', 'inv', 'sum'), E('inv', 'weight', 'inv_rms')],
  { default: view('sum', isQ ? 'projection.q' : 'projection.k') }, { default: view('weight', isQ ? 'normalized.q' : 'normalized.k') });

  const subgraphs = {
    'q-prep': frame('q-preparation', [
      C('norm', 'Q RMSNorm', '215–234', ['q_raw', 'q_heads'], '按 head 对 Q 应用 inv_rms、RMSNorm 与 q_norm_w。', 'normalization', ['pl.row_sum', 'pl.recip', 'pl.row_expand_mul'], normFrame(true)),
      O('rope', 'Q RoPE', '235–246', ['pl.concat', 'pl.col_expand_mul', 'pl.sub', 'pl.add'], 'cos/sin 旋转 Q 的半维。', 'position', ['q_rotated']),
      O('write', '写入 Q', '247–251', ['pl.cast'], 'BF16 Q 写入 q2d；不写 KV Cache。', 'store', ['q2d', 'q_rotated']),
    ], [E('norm', 'rope', 'normalized Q'), E('rope', 'write', 'rotated Q')],
    { Q: view('norm', 'projection.q') }, { Q: view('write', 'attention.q') }),
    'k-prep': frame('k-preparation', [
      C('norm', 'K RMSNorm', '162–180', ['k_raw', 'k_normed'], '按 head 对 K 应用 inv_rms、RMSNorm 与 k_norm_w。', 'normalization', ['pl.row_sum', 'pl.recip', 'pl.row_expand_mul'], normFrame(false)),
      O('rope', 'K RoPE', '181–193', ['pl.concat', 'pl.col_expand_mul', 'pl.sub', 'pl.add'], 'cos/sin 旋转 K 的半维。', 'position', ['k_rotated']),
      O('cast', 'BF16 K', '194–198', ['pl.cast'], 'K 转 BF16 后写入 Key Cache。', 'cast', ['k_rotated']),
    ], [E('norm', 'rope', 'normalized K'), E('rope', 'cast', 'rotated K')],
    { K: view('norm', 'projection.k') }, { K: view('cast', 'attention.k') }),
    'v-scale': frame('v-scaling', [
      O('scale', 'V × inv_rms', '200–207', ['pl.mul'], 'V 投影与 inv_rms 相乘。', 'elementwise', ['v_proj', 'inv_rms']),
      O('cast', 'BF16 V', '200–213', ['pl.cast'], '缩放后的 V 转 BF16。', 'cast', ['v_row_bf16']),
    ], [E('scale', 'cast', 'scaled V')], { V: view('scale', 'projection.v') }, { V: view('cast', 'attention.v') }),
    'cache-append': frame('cache-append', [
      D('slot', '写入位置', '146–155、194–194', ['slot_mapping', 'write_slot', 'cache_row'], 'slot_mapping 先钳制到非负值，再加当前层基址，决定当前 token 的 Cache 行。',
        { role: 'metadata', granularity: 'scalar', symbol: 'write_slot', entity: 'metadata.slot', dtype: 'INDEX',
          addressing: { source: 'slot_mapping[batch_idx]',
            clamped: 'write_slot = pl.max(pl.cast(pl.tensor.read(slot_mapping, [batch_idx]), pl.INDEX), 0)',
            cacheRow: 'cache_row = layer_cache_base_token_rows + write_slot' } },
        [], { fact: 'symbol 与 dtype 取自第 152–155 行 write_slot 的 pl.cast(…, pl.INDEX) 定义。',
          resolved: 'addressing 取自第 146–155 行与第 194 行的 slot_mapping → write_slot → cache_row 计算；role / granularity / entity 为语义归类。' }),
      O('write-k', '追加 K', '194–198', [], 'K 载荷写入 Key Cache；这是索引赋值。', 'store', ['key_cache_bsnd', 'k_rotated']),
      O('write-v', '追加 V', '210–213', [], 'V 载荷写入 Value Cache；这是索引赋值。', 'store', ['value_cache_bsnd', 'v_row_bf16']),
      keyCache('key', 'Key Cache'), valueCache('value', 'Value Cache'),
    ], [E('slot', 'write-k', 'slot', 'control'), E('slot', 'write-v', 'slot', 'control'), E('write-k', 'key', 'K'), E('write-v', 'value', 'V')],
    { K: view('write-k', 'attention.k'), V: view('write-v', 'attention.v'), slot: view('slot', 'metadata.slot') },
    { K: view('key', 'cache.key'), V: view('value', 'cache.value') }),
    qk: frame('qk', [
      D('q-tile', 'Q Tile', '269–276', ['q_tile', 'q2d'], '从 q2d 按 [qp_row, 0] 加载 ROW_TILE 行到 Mat；valid_shape 只保留当前 KV head 的 GROUP 行（第 269–276 行），其余行属于相邻 KV head 的 Q。外部 Tensor 的 dtype 未由此 load 显式声明。',
        { role: 'input-view', granularity: 'tile', entity: 'attention.q', symbol: 'q_tile',
          shape: { allocated: ['ROW_TILE', 'HEAD_DIM'], valid: ['GROUP', 'HEAD_DIM'] },
          storage: { space: 'Mat' },
          relations: [{ kind: 'slice', entity: 'attention.q', description: 'q_tile 是 q2d 张量（[query_rows, HEAD_DIM]，第 106 行）从行 qp_row = batch * NUM_HEADS + kv_head * GROUP 起的 ROW_TILE 行切片，其中 GROUP 行为当前 KV head 的有效 Q。' }],
          addressing: { rowBase: 'qp_row = batch * NUM_HEADS + kv_head * GROUP' } },
        [], { fact: 'shape / symbol 取自第 270–276 行 pl.load 的 [ROW_TILE, HEAD_DIM] 与 valid_shape=[GROUP, HEAD_DIM]；ROW_TILE = 16 = Q_HEAD_PAD，GROUP = 5；storage 取自 target_memory=pl.MemorySpace.Mat。' }),
      D('k-page', 'K Page', '261–267、282–295', ['qk_k_page', 'qk_ph', 'block_table_2d'], '通过逻辑页表读物理页号，再将一个 KV head 的 K Page 加载为 Mat Tile。Page 不是 SPMD Block；load 未声明 valid_shape，整页有效。',
        { role: 'cache-view', granularity: 'page', entity: 'cache.key.page', symbol: 'qk_k_page',
          shape: { allocated: ['BLOCK_SIZE', 'HEAD_DIM'] },
          storage: { space: 'Mat' },
          relations: [{ kind: 'slice', entity: 'cache.key', description: 'qk_k_page 是 Key Cache 中物理页 qk_ph 的 [BLOCK_SIZE, HEAD_DIM] 页切片，列 col = kv_head * HEAD_DIM。' }],
          addressing: { ...cacheAccess } },
        [], { fact: 'shape / symbol 取自第 290–295 行 pl.load 的 [BLOCK_SIZE, HEAD_DIM] 与 target_memory=pl.MemorySpace.Mat；页号 qk_ph 取自第 286–289 行的 block_table_2d 读取。' }),
      O('matmul', 'QK Matmul', '296–300', ['pl.matmul', 'pl.tile.transpose_view'], 'Q Tile 与 K Page 转置相乘，显式输出 FP32。', 'matmul', ['qk_score_page'],
        { transposedInput: { node: 'k-page', api: 'pl.tile.transpose_view', line: 298,
          note: '第 298 行把 K Page 经 pl.tile.transpose_view 作为 matmul 的第二个操作数，收缩维因此是 K Page 的 HEAD_DIM。' } },
        ['transposedInput'],
        { resolved: 'transposedInput：第 296–299 行的 pl.matmul(q_tile, pl.tile.transpose_view(qk_k_page), out_dtype=pl.FP32) 已包含这两次调用，这里只把「右操作数是 K Page 的转置视图」记下来，供 Shape 关系展示使用。' }),
      D('score', 'FP32 Scores', '296–305', ['qk_score_page', 'score_transfer'], '每 Page 的 FP32 Scores 写入 score_transfer 的当前 Stack 列区间；它是父图 Stack Scores 的切片，不是整个 Stack。',
        { role: 'intermediate', granularity: 'page', entity: 'attention.scores', symbol: 'qk_score_page', dtype: 'FP32',
          shape: { allocated: ['ROW_TILE', 'BLOCK_SIZE'] },
          relations: [{ kind: 'slice', entity: 'attention.scores', description: 'pl.store(..., [produce_row, qk_page_offset * BLOCK_SIZE], score_transfer) 将每页写入当前 Stack。' }] },
        ['shape'],
        { fact: 'dtype 取自第 296–300 行 pl.matmul(..., out_dtype=pl.FP32)。',
          resolved: 'shape：qk_score_page = q_tile(ROW_TILE × HEAD_DIM) × transpose_view(k_page)(HEAD_DIM × BLOCK_SIZE) → [ROW_TILE, BLOCK_SIZE]；源码未在 matmul 处声明 valid，故不声明 valid；granularity / role / entity 为语义归类。' }),
    ], [E('q-tile', 'matmul', 'Q'), E('k-page', 'matmul', 'Kᵀ'), E('matmul', 'score', 'Scores')],
    { Q: view('q-tile', 'attention.q'), K: slice('k-page', 'cache.key', '从 K Cache 取当前逻辑页对应的 KV head 切片。'), block: control('k-page', 'metadata.block-table', '页表提供物理页号，不是 K 载荷。') },
    { Scores: aggregate('score', 'attention.scores', '每页 FP32 Scores 按 page offset 写入同一个 Stack 的 score_transfer。') }),
    'scale-mask': frame('scaling-mask', [
      O('scale', 'Scaling', '777–777', ['pl.tile.muls'], 'Scores 乘 SCALE = 1/√HEAD_DIM。', 'elementwise', ['score_scaled', 'SCALE']),
      O('mask', 'Valid Columns Mask', '778–788', ['pl.set_validshape', 'pl.fillpad'], '根据 seq_len 处理当前 Stack 的有效列和填充值。', 'masking', ['score_masked', 'valid_cols']),
    ], [E('scale', 'mask', 'scaled Scores')],
    { Scores: view('scale', 'attention.scores'), seq: control('mask', 'metadata.seq-lens', 'seq_len 决定当前 Stack 的有效列数。') }, { Scores: view('mask', 'attention.masked-scores') }),
    softmax: frame('online-softmax', [
      D('scores', 'Masked Scores', '770–788', ['score_masked'], '当前 Stack 的 score_aiv 经过 SCALE 与有效列处理后的视图；valid 列数由 seq_len 决定（第 778–788 行）。',
        { role: 'intermediate', granularity: 'stack', entity: 'attention.masked-scores', symbol: 'score_masked',
          shape: { allocated: ['AIV_ROW_TILE', 'STACK_TOKENS'], valid: ['lane_rows', 'STACK_TOKENS'] },
          storage: { space: 'Vec' } },
        [], { fact: 'shape 取自第 770–776 行 pl.load 的 [AIV_ROW_TILE, STACK_TOKENS] 与 valid_shape=[lane_rows, STACK_TOKENS]，以及第 782–788 行的 pl.set_validshape；storage 取自 target_memory=pl.MemorySpace.Vec。' }),
      S('previous', 'Previous · m_old / l_old', '745–760', ['m_iter', 'l_iter'], '上轮 Stack 的 running max 和归一化和。m_old 对应 m_iter，l_old 对应 l_iter；首轮读取源码初始化值，不表示采集到的运行数值。',
        { entity: 'softmax.state', granularity: 'stack', dtype: 'FP32', initial: 'm = -3.0e38; l_sum = 0', update: '读取上轮 m_after / l_after', feedback: 'updated → previous，下一轮 Stack' }),
      O('local-max', 'Row Maximum', '789–789', ['pl.row_max'], '计算当前 Stack 的局部最大值。', 'reduction', ['local_m']),
      O('next-max', 'Update Maximum', '790–790', ['pl.maximum'], 'next_m = max(local_m, m_iter)。', 'reduction', ['next_m']),
      O('rescale', 'Rescale', '791–791', ['pl.exp', 'pl.sub'], 'rescale = exp(m_iter − next_m)。', 'elementwise', ['rescale']),
      O('exp', 'Exponential', '792–792', ['pl.exp', 'pl.row_expand_sub'], '计算未除以全局 l_sum 的概率权重。', 'elementwise', ['probability_aiv']),
      D('probability', 'Probability P', '802–815', ['probability_bf16', 'probability_transfer'], '当前 Stack 的 BF16 概率权重写入 probability_transfer，供 PV 消费；probability_bf16 由算内 probability_aiv 转换而来，其所在存储空间未显式声明，故不声明 storage。',
        { role: 'intermediate', granularity: 'stack', entity: 'attention.probability', symbol: 'probability_bf16', dtype: 'BF16' },
        [], { fact: 'dtype 取自第 802–806 行 pl.cast(..., target_type=pl.BF16, mode="rint")；symbol 取自第 802、807–816 行的 set_validshape 与 pl.store。' }),
      O('sum', 'Normalization Sum', '793–801', ['pl.row_sum', 'pl.add', 'pl.mul'], 'next_l = rescale × l_iter + row_sum(probability_reduce)。', 'reduction', ['next_l']),
      S('updated', 'Updated · m_new / l_new', '791–801、823–825、856–861', ['m_after', 'l_after', 'rescale_after'], 'm_new = max(m_old, row_max(Scores))；l_new = exp(m_old − m_new) × l_old + sum(exp(Scores − m_new))。更新后反馈到下一 Stack；rescale 进入延迟 FIFO。',
        { entity: 'softmax.state', granularity: 'stack', dtype: 'FP32', initial: 'm = -3.0e38; l_sum = 0; pending rescale = 1', update: 'm_after=next_m; l_after=next_l; rescale_after=rescale', feedback: 'updated → previous；rescale → Pending Rescale FIFO' }),
    ], [E('scores', 'local-max', 'Scores'), E('scores', 'exp', 'Scores'), E('local-max', 'next-max', 'local_m'),
      E('previous', 'next-max', 'm_iter', 'state'), E('previous', 'rescale', 'm_iter', 'state'), E('next-max', 'rescale', 'next_m'), E('next-max', 'exp', 'next_m'),
      E('exp', 'probability', 'P'), E('exp', 'sum', 'ΣP'), E('rescale', 'sum', 'rescale'), E('previous', 'sum', 'l_iter', 'state'),
      E('next-max', 'updated', 'next_m', 'state'), E('sum', 'updated', 'next_l', 'state'), E('updated', 'previous', 'next stack', 'feedback', { route: 'feedback' })],
    { Scores: view('scores', 'attention.masked-scores') }, { P: view('probability', 'attention.probability'), state: view('updated', 'softmax.state') }),
    pv: frame('pv', [
      D('probability', 'Probability P', '435–446', ['probability0', 'probability_transfer'], '从 probability_transfer 加载当前 Page 的 BF16 概率 Tile 到 Mat：allocated [ROW_TILE, BLOCK_SIZE]、valid [GROUP, BLOCK_SIZE]。',
        { role: 'input-view', granularity: 'page', entity: 'attention.probability', symbol: 'probability0', dtype: 'BF16',
          shape: { allocated: ['ROW_TILE', 'BLOCK_SIZE'], valid: ['GROUP', 'BLOCK_SIZE'] },
          storage: { space: 'Mat' },
          relations: [{ kind: 'slice', entity: 'attention.probability', description: 'probability0..3 是本 Stack 概率列区间（第 k 个 Page 对应列 k * BLOCK_SIZE）的 [ROW_TILE, BLOCK_SIZE] 切片，valid 行数为 GROUP。' }] },
        [], { fact: 'shape / dtype / symbol 取自第 435–446 行 pl.Tile[[ROW_TILE, BLOCK_SIZE], pl.BF16, …] 与 pl.load(..., valid_shape=[GROUP, BLOCK_SIZE], target_memory=pl.MemorySpace.Mat)；probability1..3 同型（第 452–502 行）。' }),
      D('v-page', 'V Page', '364–407', ['v0', 'pv_ph0', 'block_table_2d'], '通过 block_table 读物理页号，加载一个 KV head 的 V Page 到 Mat；v0 显式声明 BF16。load 未声明 valid_shape，整页有效。',
        { role: 'cache-view', granularity: 'page', entity: 'cache.value.page', symbol: 'v0', dtype: 'BF16',
          shape: { allocated: ['BLOCK_SIZE', 'HEAD_DIM'] },
          storage: { space: 'Mat' },
          relations: [{ kind: 'slice', entity: 'cache.value', description: 'v0..v3 是 Value Cache 中物理页 pv_ph0..3 的 [BLOCK_SIZE, HEAD_DIM] 页切片，列 col = kv_head * HEAD_DIM。' }],
          addressing: { ...cacheAccess } },
        [], { fact: 'shape / dtype / symbol 取自第 386–429 行 pl.Tile[[BLOCK_SIZE, HEAD_DIM], pl.BF16, …] 与 pl.load(..., target_memory=pl.MemorySpace.Mat)；页号取自第 370–385 行的 block_table_2d 读取。' }),
      O('matmul', 'PV Matmul', '447–718', ['pl.matmul', 'pl.matmul_acc'], 'P 与 V Page 相乘；同一 Stack 的 Page 结果在 FP32 accumulator 中累加。', 'matmul', ['pv0', 'pv3']),
      D('partial', 'PV Partial', '447–451、503–718', ['pv3', 'pv_transfer'], '当前 Stack 的 FP32 Context partial 写入 pv_transfer；这是单轮数据结果，不是跨 Stack 状态。',
        { role: 'intermediate', granularity: 'stack', entity: 'context.partial', symbol: 'pv_transfer', dtype: 'FP32',
          shape: { allocated: ['ROW_TILE', 'HEAD_DIM'] },
          storage: { space: 'GM' } },
        ['shape', 'storage'],
        { fact: 'dtype 取自第 447–451 行 pl.matmul(..., out_dtype=pl.FP32)；symbol 取自第 503 行 pl.store(pv3, ..., pv_transfer)。',
          resolved: 'shape：pv3 = probability(ROW_TILE × BLOCK_SIZE) × v(BLOCK_SIZE × HEAD_DIM) → [ROW_TILE, HEAD_DIM]；storage：pv_transfer 是外部 pl.Tensor，位于 GM；granularity / role / entity 为语义归类。' }),
    ], [E('probability', 'matmul', 'P'), E('v-page', 'matmul', 'V'), E('matmul', 'partial', 'partial')],
    { P: slice('probability', 'attention.probability', '从当前 Stack 的 probability_transfer 取一个 Page 的概率列。'), V: slice('v-page', 'cache.value', '从 V Cache 取当前物理页的 KV head 切片。') }, { partial: view('partial', 'context.partial') }),
    'context-acc': frame('context-accumulate', [
      D('partial', 'PV Partial', '835–841', ['pv_aiv', 'pv_transfer'], '从当前 Stack 的 PV 传输区加载局部 partial 到 Vec；load 未显式声明外部 buffer dtype。',
        { role: 'input-view', granularity: 'stack', entity: 'context.partial', symbol: 'pv_aiv', dtype: 'FP32',
          shape: { allocated: ['AIV_ROW_TILE', 'HEAD_DIM'], valid: ['lane_rows', 'HEAD_DIM'] },
          storage: { space: 'Vec' } },
        ['dtype'],
        { fact: 'shape / symbol 取自第 835–841 行 pl.load 的 [AIV_ROW_TILE, HEAD_DIM]、valid_shape=[lane_rows, HEAD_DIM]；storage 取自 target_memory=pl.MemorySpace.Vec。',
          resolved: 'dtype：第 835–841 行的 load 未声明类型，由第 447–451、503 行的 FP32 生产端与第 748–753 行的 FP32 累加器推导；granularity / role / entity 为语义归类。' }),
      S('pending', 'Delayed Rescale · pending0', '754–760、848–861', ['pending0_iter', 'pending1_iter', 'rescale_after'], 'rescale_after 先进入 pending1，下一次移到 pending0，再由 Context 消费；延迟使缩放因子与所累加的 PV partial 对齐。源码只给出静态传递关系，不提供运行值。',
        { entity: 'context.pending-rescale', granularity: 'stack', initial: 'pending0 = 1; pending1 = 1', update: 'pending0_after=pending1_iter; pending1_after=rescale_after', feedback: 'pl.range init_values 与 pl.yield_ 传递到下一 tick' }),
      S('old', 'Previous · o_old', '748–760、842–861', ['o_iter', 'o_after'], '跨 Stack 保存的 Context accumulator。o_old 对应 o_iter；首轮初始化为 0，之后读取上轮 o_after。',
        { entity: 'context.o', granularity: 'stack', dtype: 'FP32',
          shape: { allocated: ['AIV_ROW_TILE', 'HEAD_DIM'], valid: ['lane_rows', 'HEAD_DIM'] },
          initial: 'o = 0', update: 'o_after = o_iter × pending0_iter + pv_aiv', feedback: 'add → old，下一轮 Stack' },
        [], { fact: 'shape 与 dtype 取自第 748–753 行的 pl.tile.full([AIV_ROW_TILE, HEAD_DIM], dtype=pl.FP32) 与 pl.set_validshape(o_init, lane_rows, HEAD_DIM)；initial / update / feedback 取自第 748–760、842–861 行的 pl.range / pl.yield_ 传递。' }),
      O('rescale', 'Rescale Context', '842–843', ['pl.row_expand_mul'], '旧 Context 乘 pending0。', 'elementwise', ['o_iter', 'pending0_iter']),
      O('add', 'Accumulate', '842–844', ['pl.add'], '把当前 Stack 的 PV partial 加到重缩放后的 Context。', 'accumulation', ['o_after']),
      S('updated', 'Updated · o_new', '842–844、856–861', ['o_after'], 'o_new = o_old × pending0 + PV partial；经 pl.yield_ 传给下一 Stack 的 o_old。',
        { entity: 'context.o', granularity: 'stack', dtype: 'FP32', initial: 'o = 0', update: 'o_after = o_iter × pending0_iter + pv_aiv', feedback: 'updated → previous，下一轮 Stack' }),
    ], [E('pending', 'rescale', 'pending0', 'state'), E('old', 'rescale', 'o', 'state'), E('rescale', 'add', 'rescaled o'), E('partial', 'add', 'PV partial'), E('add', 'updated', 'o_after', 'state'), E('updated', 'old', 'next stack o', 'feedback', { route: 'feedback' })],
    { partial: view('partial', 'context.partial'), state: aggregate('pending', 'softmax.state', 'rescale_after 经双槽 FIFO 延迟为 pending0_iter/pending1_iter。') }, { Context: view('updated', 'context.o') }),
    'output-norm': frame('output-normalization', [
      O('normalize', 'Final Normalize', '864–865', ['pl.recip', 'pl.row_expand_mul'], 'Context o 除以最终 l_sum。', 'normalization', ['o', 'l_sum']),
      O('cast', 'Cast BF16', '864–868', ['pl.cast'], '归一化 Context 转 BF16。', 'cast', ['ctx']),
      O('store', 'Store Output', '869–873', ['pl.store'], '按当前 Batch/KV 组及 AIV lane 写回 Attention Output。', 'store', ['ctx', 'out2d']),
    ], [E('normalize', 'cast', 'Context'), E('cast', 'store', 'BF16')],
    { Context: control('normalize', 'context.normalization-state', '输出归一化同时读取最终 o 与 l_sum。') }, { Output: view('store', 'attention.output') }),
  };

  // The layout reads these phase memberships; node identity and dependencies stay in the graph.
  subgraphs.softmax.graph.stateEvolution = {
    previous: ['previous', 'scores'], current: ['local-max', 'next-max', 'rescale', 'exp', 'sum', 'probability'], updated: ['updated'] };
  subgraphs['context-acc'].graph.stateEvolution = {
    previous: ['old', 'pending', 'partial'], current: ['rescale', 'add'], updated: ['updated'] };

  const nodes = [
    D('q-proj', 'Q Projection', '68–94、215–227', ['q_proj'], '外部 Q 投影张量（GM）。签名只声明 pl.Tensor，未声明 dtype 与完整 shape；源码只按下标 [batch_idx, kv_head * GROUP * HEAD_DIM + q] 取 [1, GROUP * HEAD_DIM] 的行切片，列上界为 NUM_HEADS * HEAD_DIM，行数由运行时决定，故不声明 shape 与 dtype。',
      { role: 'input', granularity: 'tensor', entity: 'projection.q', symbol: 'q_proj',
        storage: { space: 'GM' },
        addressing: { row: 'batch_idx : batch_idx + 1', column: 'kv_head * GROUP * HEAD_DIM : (kv_head + 1) * GROUP * HEAD_DIM', columnUpperBound: 'NUM_HEADS * HEAD_DIM = HIDDEN' } },
      ['storage'],
      { fact: 'symbol 取自第 78 行函数签名；dtype 与完整 shape 源码未声明，两个字段缺省。',
        resolved: 'storage：外部 pl.Tensor 位于 GM（第 253–255 行以 GM writes 描述外部张量写入）；addressing 取自第 219–222 行的行 / 列切片；role / granularity / entity 为语义归类。' }),
    D('k-proj', 'K Projection', '68–94、162–173', ['k_proj'], '外部 K 投影张量（GM）。签名只声明 pl.Tensor，未声明 dtype 与完整 shape；源码按下标 [batch_idx, cache_col + k] 取 [1, HEAD_DIM] 的行切片，列上界为 NUM_KV_HEADS * HEAD_DIM = KV_HIDDEN，行数由运行时决定，故不声明 shape 与 dtype。',
      { role: 'input', granularity: 'tensor', entity: 'projection.k', symbol: 'k_proj',
        storage: { space: 'GM' },
        addressing: { row: 'batch_idx : batch_idx + 1', column: 'cache_col : cache_col + HEAD_DIM = kv_head * HEAD_DIM : … + HEAD_DIM', columnUpperBound: 'KV_HIDDEN = NUM_KV_HEADS * HEAD_DIM' } },
      ['storage'],
      { fact: 'symbol 取自第 79 行函数签名；dtype 与完整 shape 源码未声明，两个字段缺省。',
        resolved: 'storage：外部 pl.Tensor 位于 GM；addressing 取自第 165–168 行的行 / 列切片（cache_col = kv_head * HEAD_DIM，第 161 行）；role / granularity / entity 为语义归类。' }),
    D('v-proj', 'V Projection', '68–94、200–207', ['v_proj'], '外部 V 投影张量（GM）。签名只声明 pl.Tensor，未声明 dtype 与完整 shape；源码按下标 [batch_idx, cache_col + v] 取 [1, HEAD_DIM] 的行切片，列上界为 KV_HIDDEN，行数由运行时决定，故不声明 shape 与 dtype。',
      { role: 'input', granularity: 'tensor', entity: 'projection.v', symbol: 'v_proj',
        storage: { space: 'GM' },
        addressing: { row: 'batch_idx : batch_idx + 1', column: 'cache_col : cache_col + HEAD_DIM = kv_head * HEAD_DIM : … + HEAD_DIM', columnUpperBound: 'KV_HIDDEN = NUM_KV_HEADS * HEAD_DIM' } },
      ['storage'],
      { fact: 'symbol 取自第 80 行函数签名；dtype 与完整 shape 源码未声明，两个字段缺省。',
        resolved: 'storage：外部 pl.Tensor 位于 GM；addressing 取自第 202–205 行的行 / 列切片；role / granularity / entity 为语义归类。' }),
    C('q-prep', 'Q Norm → RoPE', '215–251', ['q_raw', 'q_heads', 'q_rotated', 'q2d'], 'Q 乘 inv_rms、按 head 归一化，再施加 RoPE，写入 q2d。', 'normalization', ['pl.row_sum', 'pl.recip', 'pl.concat'], subgraphs['q-prep']),
    C('k-prep', 'K Norm → RoPE', '162–198', ['k_raw', 'k_normed', 'k_rotated'], 'K 乘 inv_rms、按 head 归一化，再施加 RoPE。', 'normalization', ['pl.row_sum', 'pl.recip', 'pl.concat'], subgraphs['k-prep']),
    C('v-scale', 'V Scaling', '200–213', ['v_row_bf16'], 'V 按 inv_rms 缩放并转 BF16。', 'elementwise', ['pl.mul', 'pl.cast'], subgraphs['v-scale']),
    D('q', 'Q', '247–251', ['q_rotated', 'q2d'], 'Q 在 q2d 中的张量视图（GM）：q2d = pl.reshape(q_tnd_flat, [query_rows, HEAD_DIM])（第 106 行），本 kernel 只写入 (batch, kv_head) 组对应的 GROUP 行（第 247–251 行）。归一化在 Q_HEAD_PAD = 16 行的填充 tile 上完成，只有前 GROUP 行是有效 Q。dtype 描述写入载荷（pl.cast → BF16），不声明 q2d 的存储类型。',
      { role: 'intermediate', granularity: 'tensor', entity: 'attention.q', symbol: 'q2d', dtype: 'BF16',
        shape: { allocated: ['query_rows', 'HEAD_DIM'], valid: ['GROUP', 'HEAD_DIM'] },
        storage: { space: 'GM' },
        addressing: { rowBase: 'q_row = batch_idx * NUM_HEADS + kv_head * GROUP', rowExtent: 'q_rotated[0:GROUP, :]' } },
      ['shape', 'storage'],
      { fact: 'dtype 取自第 248–251 行 pl.cast(..., target_type=pl.BF16)；symbol（q2d）与 source.symbols 取自第 247–251 行。',
        resolved: 'shape.allocated 取自第 106 行的 pl.reshape(q_tnd_flat, [query_rows, HEAD_DIM])，query_rows = pl.tensor.dim(q_tnd_flat, 0) 是运行时量；shape.valid 取自第 248–251 行的写入切片 q2d[q_row : q_row + GROUP, :]；storage：外部张量位于 GM；granularity：本节点描述整个 q2d 张量，Tile 形状见 qk/q-tile。' }),
    D('k', 'K', '181–198', ['k_rotated'], '当前 token 单个 KV head 的 K 载荷。归一化在 K_RED_ROWS = 8 行的填充 tile 上完成（行 1..7 为 0 填充，第 162–172 行），随后取 k_full = k_normed[0:1, :]，RoPE 后转 BF16 追加到 Key Cache（第 181–198 行）。',
      { role: 'intermediate', granularity: 'token', entity: 'attention.k', symbol: 'k_rotated', dtype: 'BF16',
        shape: { allocated: [1, 'HEAD_DIM'] },
        addressing: { reductionTile: 'k_raw = pl.reshape(pl.concat(k_proj[batch_idx : batch_idx + 1, cache_col : cache_col + HEAD_DIM], k_red_pad), [K_RED_ROWS, HEAD_DIM])，行 1..7 为 0 填充', payload: 'k_full = k_normed[0:1, :]' } },
      ['shape'],
      { fact: 'dtype 取自第 198 行 pl.cast(k_rotated, target_type=pl.BF16)；symbol 取自第 181–193 行的 k_full / k_rotated。',
        resolved: 'shape：K 载荷是 1 行 × HEAD_DIM 的行切片（第 181、193 行），不是整个 [K_RED_ROWS, HEAD_DIM] 归约 tile；addressing 取自第 162–172、181 行；granularity / role / entity 为语义归类。' }),
    D('v', 'V', '200–213', ['v_row_bf16'], '当前 token 单个 KV head 的 BF16 V 载荷：v_proj 的 [1, HEAD_DIM] 行切片乘 inv_rms 后转 BF16，再按 cache_row / cache_col 写入 Value Cache（第 200–213 行）。',
      { role: 'intermediate', granularity: 'token', entity: 'attention.v', symbol: 'v_row_bf16', dtype: 'BF16',
        shape: { allocated: [1, 'HEAD_DIM'] },
        addressing: { payload: 'pl.cast(pl.mul(v_proj[batch_idx : batch_idx + 1, cache_col : cache_col + HEAD_DIM], inv_rms), target_type=pl.BF16)', writeRow: 'cache_row = layer_cache_base_token_rows + write_slot' } },
      ['shape'],
      { fact: 'dtype 与 symbol 取自第 200–213 行的 pl.cast(..., target_type=pl.BF16) 与 v_row_bf16 写入。',
        resolved: 'shape：载荷是 v_proj 的单行切片（第 202–205 行）→ [1, HEAD_DIM]；addressing 取自第 200–213 行；granularity / role / entity 为语义归类。' }),
    D('aux', 'Aux Inputs', '68–94、146–155、261–267', ['seq_lens', 'block_table', 'slot_mapping'], '运行时索引与控制输入（外部 pl.Tensor，GM）。三者的元素类型与行数都未在源码中声明：seq_lens 决定有效长度与 active_batch；slot_mapping 给出当前 token 的写入槽位（第 146–155 行先钳制到非负再加层基址）；block_table 提供逻辑页到物理页映射，并 reshape 为 [active_batch, max_blocks_per_seq]。',
      { role: 'metadata', granularity: 'bundle', entity: 'metadata.bundle', symbol: 'seq_lens, block_table, slot_mapping',
        storage: { space: 'GM' },
        addressing: { activeBatch: 'active_batch = pl.tensor.dim(seq_lens, 0)', blockTable: 'block_table_2d = pl.reshape(block_table, [active_batch, max_blocks_per_seq])，max_blocks_per_seq = pl.tensor.dim(block_table, 0) // active_batch', slotMapping: 'write_slot = pl.max(pl.cast(pl.tensor.read(slot_mapping, [batch_idx]), pl.INDEX), 0)', seqLength: 'seq_len = pl.read(seq_lens, [batch])' } },
      ['storage'],
      { fact: 'symbols 取自第 73、72、75 行函数签名；三个张量的 dtype 与 shape 源码未声明，两个字段缺省。',
        resolved: 'storage：外部 pl.Tensor 位于 GM；addressing 取自第 98、104–105、146–155、264 行的读取与 reshape 表达式；role / granularity / entity 为语义归类。' }),
    C('cache-append', 'Cache Append', '194–198、210–213', ['key_cache_bsnd', 'value_cache_bsnd'], '当前 token 的 K/V 载荷按 cache_row 写入对应 Cache。', 'store', [], subgraphs['cache-append']),
    keyCache('k-cache'), valueCache('v-cache'),
    C('qk', 'QK Matmul', '269–276、285–305、306–356', ['q_tile', 'qk_k_page', 'qk_score_page', 'score_transfer'], 'Q 与映射后的 K Page 相乘；每页 FP32 结果写入当前 Stack 的 Scores 区间。', 'matmul', ['pl.load', 'pl.matmul'], subgraphs.qk),
    D('scores', 'Scores', '278–305、770–776', ['score_transfer', 'score_aiv'], '多个 Page 的 QK Scores 按列区间写入 score_transfer（GM），再按整个 Stack 加载到 AIV。score_transfer 的形状由写入地址推导，不代表运行场景的固定配置。',
      { role: 'intermediate', granularity: 'stack', entity: 'attention.scores', symbol: 'score_transfer', dtype: 'FP32',
        shape: { allocated: ['TRANSFER_ROWS', 'STACK_TOKENS'] },
        storage: { space: 'GM' },
        relations: [{ kind: 'aggregate', entity: 'attention.scores', description: 'qk/score 每页结果按 qk_page_offset * BLOCK_SIZE 写入本 Stack。' }] },
      ['shape', 'storage'],
      { fact: 'dtype 取自第 296–300 行 pl.matmul(..., out_dtype=pl.FP32)；symbol 取自第 304、771 行。',
        resolved: 'shape：签名未声明 score_transfer 的形状，由第 268、283 行的写入行址 transfer_base + (stack % TRANSFER_SLOTS) * ROW_TILE 与第 303 行的列址 qk_page_offset * BLOCK_SIZE 推导 → [TRANSFER_ROWS, STACK_TOKENS]，其中 TRANSFER_ROWS = ATTN_SPMD_BLOCKS * TRANSFER_SLOTS * ROW_TILE = 1152、STACK_TOKENS = 512；storage：外部 pl.Tensor 位于 GM；granularity / role / entity 为语义归类。' }),
    C('scale-mask', 'Scaling + Mask', '777–788', ['score_scaled', 'score_masked', 'valid_cols'], '缩放 Scores，再根据 seq_len 处理末 Stack 的无效列。', 'masking', ['pl.tile.muls', 'pl.set_validshape', 'pl.fillpad'], subgraphs['scale-mask']),
    C('softmax', 'Online Softmax', '745–825、856–861', ['m_iter', 'l_iter', 'probability_bf16', 'm_after', 'l_after'], '跨 Stack 更新 running max 和归一化和，输出 BF16 概率权重及 rescale 状态。', 'normalization', ['pl.row_max', 'pl.exp', 'pl.row_sum'], subgraphs.softmax),
    D('probability', 'Probability P', '802–815', ['probability_bf16', 'probability_transfer'], '本 Stack 的 BF16 权重尚未除以全局 l_sum，供 PV 消费。概率写在 probability_transfer 的两个 Page 列区间内。',
      { role: 'intermediate', granularity: 'stack', entity: 'attention.probability', symbol: 'probability_bf16', dtype: 'BF16',
        shape: { allocated: ['TRANSFER_ROWS', 'STACK_TOKENS'] },
        storage: { space: 'GM' } },
      ['shape', 'storage'],
      { fact: 'dtype 取自第 802–806 行 pl.cast(..., target_type=pl.BF16, mode="rint")；symbol 取自第 802、814 行。',
        resolved: 'shape：由第 283、364–367 行的写入行址与第 812–816 行 pl.store(..., [produce_row + lane_row, 0], probability_transfer) 推导 → [TRANSFER_ROWS, STACK_TOKENS]；storage：probability_transfer 是外部 pl.Tensor，位于 GM；granularity / role / entity 为语义归类。' }),
    C('pv', 'PV Matmul', '364–718', ['probability0', 'v0', 'pv3', 'pv_transfer'], 'P 与映射后的 V Page 相乘，Stack 内各页结果累加为 FP32 partial。', 'matmul', ['pl.load', 'pl.matmul', 'pl.matmul_acc'], subgraphs.pv),
    C('context-acc', 'Context Accumulate', '827–861', ['pv_aiv', 'o_iter', 'o_after', 'pending0_iter'], '旧 Context 乘延迟 rescale，再加当前 Stack 的 PV partial。', 'accumulation', ['pl.row_expand_mul', 'pl.add'], subgraphs['context-acc']),
    S('context', 'Context', '745–760、842–861', ['o', 'l_sum'], '跨 Stack 累积的 o 与归一化和 l_sum，是最终输出归一化的状态，不是单轮 PV 数据。o 是 [AIV_ROW_TILE, HEAD_DIM] 的 FP32 累加器（valid 行 lane_rows，第 748–753 行，形状见 context-acc/old）；l_sum 由行归约产生，列布局未显式声明，两个状态形状不同，故本节点不声明 shape。',
      { entity: 'context.normalization-state', granularity: 'stack', dtype: 'FP32', initial: 'o = 0; l_sum = 0', update: 'o = o_after; l_sum = l_after', feedback: 'pl.range init_values / pl.yield_ 跨 tick 传递' }),
    C('output-norm', 'Output Normalize', '864–873', ['o', 'l_sum', 'ctx', 'out2d'], '所有 Stack 完成后，以最终 l_sum 归一化 o，转 BF16 并写回 out2d。', 'normalization', ['pl.recip', 'pl.row_expand_mul', 'pl.cast', 'pl.store'], subgraphs['output-norm']),
    D('output', 'Attention Output', '864–873', ['ctx', 'out2d'], '最终 BF16 Attention 载荷按 Batch、KV head、GROUP 和 lane 对应行写回 out2d；out2d = pl.reshape(out, [query_rows, HEAD_DIM])（第 108 行），本次调用只写 lane_rows 行（第 869–873 行）。dtype 描述 ctx 载荷。',
      { role: 'output', granularity: 'tensor', entity: 'attention.output', symbol: 'out2d', dtype: 'BF16',
        shape: { allocated: ['query_rows', 'HEAD_DIM'], valid: ['lane_rows', 'HEAD_DIM'] },
        storage: { space: 'GM' },
        addressing: { rowBase: 'batch * NUM_HEADS + kv_head * GROUP + lane_row' } },
      ['shape', 'storage'],
      { fact: 'dtype 取自第 864–868 行 pl.cast(..., target_type=pl.BF16)；symbol（out2d）与 source.symbols 取自第 869–873 行。',
        resolved: 'shape.allocated 取自第 108 行的 pl.reshape(out, [query_rows, HEAD_DIM])；shape.valid 继承第 748–753 行 o 的 pl.set_validshape(o_init, lane_rows, HEAD_DIM)；storage：外部张量位于 GM；granularity / role / entity 为语义归类。' }),
  ];
  const aux = nodes.find((node) => node.id === 'aux');
  aux.ports.outputs = { slot: { entity: 'metadata.slot' }, block: { entity: 'metadata.block-table' }, seq: { entity: 'metadata.seq-lens' } };
  const ports = {
    'q-proj>q-prep': { targetPort: 'Q' }, 'q-prep>q': { sourcePort: 'Q' }, 'k-proj>k-prep': { targetPort: 'K' }, 'k-prep>k': { sourcePort: 'K' },
    'v-proj>v-scale': { targetPort: 'V' }, 'v-scale>v': { sourcePort: 'V' }, 'k>cache-append': { targetPort: 'K' }, 'v>cache-append': { targetPort: 'V' },
    'aux>cache-append': { sourcePort: 'slot', targetPort: 'slot' }, 'cache-append>k-cache': { sourcePort: 'K' }, 'cache-append>v-cache': { sourcePort: 'V' },
    'aux>qk': { sourcePort: 'block', targetPort: 'block' }, 'q>qk': { targetPort: 'Q' }, 'k-cache>qk': { targetPort: 'K' },
    'qk>scores': { sourcePort: 'Scores' }, 'scores>scale-mask': { targetPort: 'Scores' }, 'aux>scale-mask': { sourcePort: 'seq', targetPort: 'seq' },
    'scale-mask>softmax': { sourcePort: 'Scores', targetPort: 'Scores' }, 'softmax>probability': { sourcePort: 'P' },
    'softmax>context-acc': { sourcePort: 'state', targetPort: 'state' }, 'softmax>context': { sourcePort: 'state' },
    'probability>pv': { targetPort: 'P' }, 'v-cache>pv': { targetPort: 'V' }, 'pv>context-acc': { sourcePort: 'partial', targetPort: 'partial' },
    'context-acc>context': { sourcePort: 'Context' }, 'context>output-norm': { targetPort: 'Context' }, 'output-norm>output': { sourcePort: 'Output' },
  };
  const edge = (s, t, label, type = 'data', extra = {}) => E(s, t, label, type, { ...ports[`${s}>${t}`], ...extra });
  const edges = [
    edge('q-proj', 'q-prep', 'Q Projection'), edge('q-prep', 'q', 'Q'), edge('k-proj', 'k-prep', 'K Projection'), edge('k-prep', 'k', 'K'), edge('v-proj', 'v-scale', 'V Projection'), edge('v-scale', 'v', 'V'),
    edge('k', 'cache-append', 'K'), edge('v', 'cache-append', 'V'), edge('aux', 'cache-append', 'slot_mapping', 'control'), edge('aux', 'qk', 'block_table', 'control'), edge('aux', 'scale-mask', 'seq_lens', 'control'),
    edge('cache-append', 'k-cache', 'K'), edge('cache-append', 'v-cache', 'V'), edge('q', 'qk', 'Q'), edge('k-cache', 'qk', 'K · block_table'), edge('qk', 'scores', 'Scores'), edge('scores', 'scale-mask', 'Scores'),
    edge('scale-mask', 'softmax', 'scaled / masked Scores'), edge('softmax', 'probability', 'P'), edge('probability', 'pv', 'P'), edge('v-cache', 'pv', 'V · block_table'), edge('pv', 'context-acc', 'PV partial'),
    edge('softmax', 'context-acc', 'rescale', 'state', { route: 'state' }), edge('softmax', 'context', 'l_sum', 'state', { route: 'state' }),
    edge('context-acc', 'context', 'Context', 'state'), edge('context', 'output-norm', 'Context', 'state'), edge('output-norm', 'output', 'BF16 Output'),
  ];
  // Boundary contracts inherit only known properties of the same semantic entity.
  // Slice / aggregate mappings then validate the refinement inside each Composite.
  edges.forEach((edge) => {
    const from = nodes.find((node) => node.id === edge.s), to = nodes.find((node) => node.id === edge.t);
    for (const [boundary, peer, side, port] of [[from, to, 'outputs', edge.sourcePort], [to, from, 'inputs', edge.targetPort]]) {
      if (boundary.type !== 'composite' || !peer.data) continue;
      const contract = boundary.ports[side][port];
      if (contract.entity !== peer.data.entity) continue;
      for (const key of ['dtype', 'granularity']) if (peer.data[key]) contract[key] = peer.data[key];
    }
  });
  for (const [id, primaryRange] of [['qk', 1], ['k-cache', 2], ['v-cache', 2], ['context', 1]]) {
    nodes.find((node) => node.id === id).source.primaryRange = primaryRange;
  }
  const region = (id, name, members) => ({ id, type: 'region', name, description: name, members, source: source('68–874'), provenance: [{ kind: 'resolved', origin: 'source', fields: ['members'], note: '按算子语义组织的现有区域分组；不是硬件执行分区。' }] });
  const model = { schemaVersion: 2, id: 'paged-attention', sourceFiles: { [FILE]: { lineCount: (root.PTO_PAGED_ATTENTION_PYPTO_SOURCE || '').trimEnd().split('\n').length } },
    constants: CONSTANTS, externalConstants: EXTERNAL_CONSTANTS, runtimeDims: RUNTIME_DIMS, nodes, edges, groups: [
    region('qkv-preparation', 'QKV Preparation', ['q-proj', 'k-proj', 'v-proj', 'q-prep', 'k-prep', 'v-scale', 'q', 'k', 'v']),
    region('paged-attention', 'Paged Attention', ['cache-append', 'k-cache', 'v-cache', 'qk', 'scores', 'scale-mask', 'softmax', 'probability', 'pv', 'context-acc']),
    region('output-region', 'Output', ['context', 'output-norm', 'output']),
  ] };
  // 源码中 Phase 0、Softmax 与输出归一化位于 split_aiv；QK / PV 位于 split_aic。
  // 只给计算节点标执行单元，DATA / STATE 的存储位置仍由 storage 单独描述。
  const executionUnits = {
    'q-prep': 'AIV', 'k-prep': 'AIV', 'v-scale': 'AIV', 'cache-append': 'AIV',
    qk: 'AIC', 'scale-mask': 'AIV', softmax: 'AIV', pv: 'AIC',
    'context-acc': 'AIV', 'output-norm': 'AIV',
  };
  const annotateExecution = (graph, unit) => {
    for (const item of graph.nodes) {
      const current = graph === model ? executionUnits[item.id] : unit;
      if (current && (item.type === 'operation' || item.type === 'composite')) {
        item[item.type].executionUnit = current;
        item.provenance.push({ kind: 'resolved', origin: 'source', fields: [`${item.type}.executionUnit`],
          note: '依据源码中的 split_aic / split_aiv 执行分支归属。' });
      }
      if (item.type === 'composite') annotateExecution(item.composite.subgraph, current);
    }
  };
  annotateExecution(model);
  // Execution constraints are orthogonal to the mathematical graph and Journey edges.
  // QK's load/wait boundary precedes Scaling + Mask, even though Softmax is the consumer stage.
  const endpoint = (node, anchor, coreType, pipe, ranges, extra = {}) =>
    ({ node, anchor, coreType, pipe, source: source(ranges), ...extra });
  model.synchronizations = [
    { id: 'qk-ready', eventId: 0, name: 'QK_READY', tensor: 'scores', buffer: 'score_transfer', dtype: 'FP32',
      producer: endpoint('qk', 'qk/score', 'AIC', 'FIX', '357–362', { fftsMode: 2 }),
      consumer: endpoint('softmax', 'scale-mask/scale', 'AIV', 'MTE2', '765–769', { boundary: 'scale-mask' }),
      explanation: 'QK 将当前 Stack 各 Page 的 FP32 Scores 写入 GM score_transfer 后 set。Softmax 的 AIV 在加载 Scores 前 wait，随后才进行 Scaling、Mask 和 Online Softmax，避免读取尚未就绪的数据。' },
    { id: 'softmax-ready', eventId: 1, name: 'SOFTMAX_READY', tensor: 'probability', buffer: 'probability_transfer', dtype: 'BF16',
      producer: endpoint('softmax', 'softmax/probability', 'AIV', 'MTE3', '817–822', { fftsMode: 2 }),
      consumer: endpoint('pv', 'pv/probability', 'AIC', 'MTE2', '430–434、522–526、576–580、662–666', { waitPageCounts: [4, 1, 2, 3] }),
      explanation: 'Softmax 将当前 Stack 的 BF16 概率权重写入 GM probability_transfer 后 set。PV 的 AIC 在加载概率 Tile 前 wait。四处 wait 属于 Stack 有效页数的不同条件分支，不是同一 Stack 连续等待四次；权重尚未除以最终 l_sum。' },
    { id: 'pv-ready', eventId: 2, name: 'PV_READY', tensor: 'pv/partial', buffer: 'pv_transfer', dtype: 'FP32',
      producer: endpoint('pv', 'pv/partial', 'AIC', 'FIX', '719–724', { fftsMode: 2 }),
      consumer: endpoint('context-acc', 'context-acc/partial', 'AIV', 'MTE2', '830–834'),
      explanation: 'PV 将当前 Stack 的 FP32 partial 写入 GM pv_transfer 后 set。Context Accumulate 的 AIV 在加载 partial 前 wait，再将它加入经过延迟 rescale 的旧 Context，避免读取未完成的 PV 结果。' },
  ];
  root.PTO_PYPA_SEMANTIC_GRAPH = model;
  root.PTO_PYPA_SEMANTIC_SUBGRAPHS = Object.fromEntries(Object.entries(subgraphs).map(([id, bundle]) => [id, bundle.graph]));
})(typeof window === 'undefined' ? globalThis : window);
