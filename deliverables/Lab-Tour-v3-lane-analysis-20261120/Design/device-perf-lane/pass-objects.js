// Structural views of the curated snapshots. Positions are diagram layout, not timing.
(() => {
  const n = (id, title, sub, y, state = 'same', match = [], x = 220, width = 280) =>
    ({ id, title, sub, y, state, match, x, width });
  const chain = nodes => nodes.slice(1).map((node, i) => ({ from: nodes[i].id, to: node.id }));
  const pane = (label, nodes, edges = chain(nodes)) => ({ label, nodes, edges });
  const scene = (title, delta, before, after, note, sources = null) => ({ title, delta, before, after, note, sources });
  const load = (sub, y = 20, state = 'same') => n('load', '载入 oi 分片', sub, y, state, ['m_cur_oi_inline1216__tile:']);
  const csa = [
    scene('P29 保留了两阶段流水请求', ['请求深度 2 → 2', 'oi 分片 32 KiB'],
      pane('P28 · 输入', [n('loop', '合并后续 4 份结果', 'pipeline · stage=2', 20, 'same', ['pl.pipeline']), load('[16,512] FP32 · 32 KiB', 110), n('acc', '更新累积结果', '当前结果 + 新分片', 200, 'same', ['m_oi_inline1220__tile_1'])]),
      pane('P29 · 输出', [n('loop', '合并后续 4 份结果', 'pipeline · stage=2', 20, 'same', ['pl.pipeline']), load('[16,512] FP32 · 32 KiB', 110), n('acc', '更新累积结果', '当前结果 + 新分片', 200, 'same', ['m_oi_inline1220__tile_1'])]),
      '此循环仍保留请求。stage=2 表示预期流水深度，实际缓冲能力继续看 P35。'),
    scene('P30 把一个循环体展开成两个迭代阶段', ['循环步长 1 → 2', '迭代阶段 1 → 2'],
      pane('P29 · 一个循环体', [n('loop', 'pipeline(1,5)', 'stage=2', 20, 'same', ['pl.pipeline']), load('同一份循环体', 110), n('acc', '更新累积结果', '下一轮读取该结果', 200, 'same', ['m_oi_inline1220'])]),
      pane('P30 · 展开后', [n('loop', 'pipeline(1,5,2)', '降低后 stage=1', 20, 'changed', ['pl.pipeline']), n('load', '阶段 0 · 加载与合并', 'pipeline_membership 0:0', 110, 'changed', ['"0:0"']), n('stage1', '阶段 1 · 加载与合并', 'pipeline_membership 0:1', 200, 'added', ['"0:1"'])]),
      '图示是迭代结构，箭头不表示实测重叠。stage=1 不能单独证明流水退化。'),
    scene('P35 将这三个阶段 0 的值复用到同一缓冲', ['选中 3 个值：缓冲 3 → 1', '另一阶段仍有独立 mem_vec_18'],
      pane('P34 · 三个值分别存放', [load('mem_vec_15 · 32 KiB'), n('weighted', '加权后的分片', 'mem_vec_28 · 32 KiB', 110, 'same', ['t__tile_5:']), n('acc-next', '下一份累积结果', 'mem_vec_29 · 32 KiB', 200, 'same', ['m_oi_inline1220__tile_1:'])]),
      pane('P35 · 存储复用', [load('mem_vec_15 · 32 KiB', 20, 'reused'), n('weighted', '加权后的分片', 'mem_vec_15 · 覆盖复用', 110, 'reused', ['t__tile_5:']), n('acc-next', '下一份累积结果', 'mem_vec_15 · 覆盖复用', 200, 'reused', ['m_oi_inline1220__tile_1:'])]),
      '这里只展示三个值的复用实例。PH-MR-001 另报告深度 2 仅满足 1/2；尚未将告警定位到具体变量。'),
    scene('P50 在每次 Put 后新增 GM Fence', ['显式 GM Fence 0 → 2 处', '随 pack_work 循环重复'],
      pane('P49 · 插入前', [n('put0', 'Put 第一组数据', 'mem_vec_66 @99584 · 64 KiB', 20, 'same', ['tput_stage:', 'tput_stage,']), n('put1', 'Put 第二组数据', 'mem_vec_66 @99584 · 64 KiB', 148, 'same', ['tput_stage_1:', 'tput_stage_1,']), n('notify', 'Notify', 'pack_work 循环之外', 276, 'same', ['pld.system.notify'])]),
      pane('P50 · 插入后', [n('put0', 'Put 第一组数据', 'mem_vec_66 @99584 · 64 KiB', 20, 'same', ['tput_stage:', 'tput_stage,']), n('fence0', 'GM Fence', '发布数据的可见性边界', 84, 'added', ['pl.system.fence']), n('put1', 'Put 第二组数据', 'mem_vec_66 @99584 · 64 KiB', 148, 'same', ['tput_stage_1:', 'tput_stage_1,']), n('fence1', 'GM Fence', '发布数据的可见性边界', 212, 'added', ['pl.system.fence']), n('notify', 'Notify', '数据先可见，再通知', 276, 'same', ['pld.system.notify'])]),
      'Fence 保障通信发布顺序。共享 staging 不等于 Fence 的插入原因；图中 Notify 不在每次 pack_work 内。'),
  ];
  function dependency(stage) {
    const graph = (label, type, state) => {
      const nodes = [n('qk', 'qk_pv', '任务依赖 · qk_tid', 20, 'same', ['qk_tid'], 115, 190), n('rope', 'rope_cs', '任务依赖 · attn_rope_tid', 20, 'same', ['attn_rope_tid'], 325, 190), n('publish', 'merge_pack_publish', type, 162, state, ['csa_merge_pack_publish', 'publish_tid']), n('blocks', '48 个 AIV block', '整任务依赖保留', 270, 'same', ['48', 'core_num'])];
      return pane(label, nodes, [{ from: 'qk', to: 'publish', label: 'deps' }, { from: 'rope', to: 'publish', label: 'deps' }, { from: 'publish', to: 'blocks' }]);
    };
    const options = {
      P00: ['前端已写入两个显式依赖', '前端 · 输入', 'spmd scope', '前端 · 已有结构', 'spmd scope', 'same', [0]],
      P09: ['P09 外提 kernel，两个依赖边保留', '前端', 'scope 内联计算', 'P09 · OutlineIncoreScopes', 'scope → kernel 调用', 'changed', [0, 1]],
      P10: ['P10 形成 SPMD 提交，两个依赖边保留', 'P09 · scope 调用', 'kernel 调用', 'P10 · OutlineClusterScopes', 'spmd_submit(wrapper)', 'changed', [1, 2]],
      P40: ['P40 仍保留已有显式依赖', 'P39 · 输入', 'spmd_submit · deps', 'P40 · 输出', 'spmd_submit · deps', 'same', [3, 4]],
    };
    const [title, a, at, b, bt, state, sources] = options[stage] || options.P09;
    return scene(title, ['显式前驱 2 → 2', '提交规模 48 blocks'], graph(a, at, 'same'), graph(b, bt, state), '这些边约束整任务就绪；实际等待时长由当前 rank 的 block 记录确认。', sources);
  }
  const matmul = [
    scene('P09 把 64 行计算外提为 kernel', ['每块 64 行 → 64 行', '矩阵规模保留'], pane('前端', [n('slice', '切出 64 行', '[64,128] BF16', 20, 'same', ['slice', '64']), n('mul', '张量矩阵乘', '[64,128] × [128,128]', 120, 'same', ['matmul']), n('store', '组装输出', '[64,128] FP32', 220, 'same', ['assemble'])]), pane('P09 · kernel', [n('slice', '按 block_idx 切片', '[64,128] BF16', 20, 'changed', ['get_block_idx', 'slice']), n('mul', '张量矩阵乘', '相同计算规模', 120, 'same', ['matmul']), n('store', '组装输出', '[64,128] FP32', 220, 'same', ['assemble'])]), '64 行切分来自前端，外提保留该工作量。'),
    scene('P11 降为 tile 操作；P17 后形状保持', ['Tile 加载 2 个', 'Matmul 1 次'], pane('P11', [n('loadA', '加载 A / B tile', '16 KiB + 32 KiB → Mat', 20, 'same', ['tile.load']), n('mul', 'tile.matmul', '64 × 128 × 128', 120, 'same', ['tile.matmul']), n('store', 'tile.store', '32 KiB FP32', 220, 'same', ['tile.store'])]), pane('P17 · AutoTileMatmulL0', [n('loadA', '加载 A / B tile', '16 KiB + 32 KiB → Mat', 20, 'same', ['tile.load']), n('mul', 'tile.matmul', '64 × 128 × 128', 120, 'same', ['tile.matmul']), n('store', 'tile.store', '32 KiB FP32', 220, 'same', ['tile.store'])]), '此处直接对比 P11 与 P17。该 kernel 的 tile 形状未变化。'),
    scene('P19 为矩阵乘增加 Left / Right 搬运', ['tile.move 0 → 2', 'Mat → Left / Right'], pane('P18', [n('loadA', 'A / B 位于 Mat', 'BF16 tile', 20, 'same', ['tile.load']), n('mul', '矩阵乘', 'Mat 输入', 200, 'same', ['tile.matmul'])]), pane('P19', [n('loadA', 'A / B 位于 Mat', 'BF16 tile', 20, 'same', ['tile.load']), n('move', '搬到 Left / Right', 'A → Left · B → Right', 110, 'added', ['tile.move']), n('mul', '矩阵乘 → Acc', 'Left × Right', 200, 'changed', ['tile.matmul'])]), '最终 PTO 的块布局与子布局可在对象关联代码中查看；布局不同不等于多余转置。'),
    scene('最终代码保留加载、搬运与写回路径', ['输入内层 256B', 'PH001 建议 ≥512B'], pane('Tile IR', [n('loadA', '两次 tile.load', 'BF16[128] · 每行 256B', 20, 'same', ['tile.load', 'tload']), n('move', '两次 tile.move', 'Mat → Left / Right', 110, 'same', ['tile.move', 'tmov']), n('mul', 'tile.matmul', 'Acc 输出', 200, 'same', ['tile.matmul', 'tmatmul']), n('store', 'tile.store', '写回 GM', 290, 'same', ['tile.store', 'tstore'])]), pane('最终 C++', [n('loadA', '2 × TLOAD', '256B 内层跨度', 20, 'warning', ['TLOAD', 'tile.load', 'tload']), n('move', '2 × TMOV', 'Mat → Left / Right', 110, 'changed', ['TMOV']), n('mul', 'TMATMUL', 'Acc 输出', 200, 'same', ['TMATMUL']), n('store', 'TSTORE', '写回 GM', 290, 'same', ['TSTORE'])]), '256B 是静态内层跨度，不能换算成实测带宽或效率。'),
  ];
  const quant = [
    scene('前端已安排两轮读取同一输入', ['唯一输入 32 KiB / block', '代码加载量 64 KiB / block'], pane('输入对象', [n('input', '64 × 128 FP32', '32 KiB 唯一数据', 20, 'same', ['qh_acc_gm']), n('scale', '求量化 scale', '先知道整行最大值', 130, 'same', ['qh_amax', 'qh_scale']), n('quant', '量化为 INT8', '用 scale 转换输入', 240, 'same', ['qh_q_', 'qh_i8'])]), pane('前端两轮循环', [n('read0', '第一轮 · 读取两片', '2 × [64,64] · 32 KiB', 20, 'same', ['h0', 'qh_a_f32']), n('scale', '求最大值 / scale', '[64,1] FP32', 110, 'same', ['qh_scale', 'qh_amax']), n('read1', '第二轮 · 再读两片', '同一输入 · 32 KiB', 200, 'warning', ['h1', 'qh_q_f32']), n('quant', '量化并写回', '[64,128] INT8 · 8 KiB', 290, 'same', ['qh_i8'])]), '两轮访问来自前端。静态加载字节数不等于实际 GM 流量。'),
    scene('P11 显式生成加载、转换与写回', ['舍入转换链保留', '两轮加载保留'], pane('P09 · Tensor', [n('read1', 'Tensor slice', '[64,64] FP32', 20, 'same', ['tensor.slice']), n('cast0', 'FP32 → INT32', 'rint', 100, 'same', ['INT32']), n('cast1', 'INT32 → FP16', 'round', 180, 'same', ['FP16']), n('cast2', 'FP16 → INT8', 'trunc', 260, 'same', ['INT8'])]), pane('P11 · Tile', [n('read1', 'tile.load', '[64,64] FP32', 20, 'changed', ['tile.load']), n('cast0', 'tile.cast → INT32', 'rint', 100, 'changed', ['target_type=pl.INT32']), n('cast1', 'tile.cast → FP16', 'round', 180, 'changed', ['target_type=pl.FP16']), n('cast2', 'tile.cast → INT8', 'trunc', 260, 'changed', ['target_type=pl.INT8'])]), '转换带有不同舍入语义，不能直接假定可合并。'),
    scene('P35 复用转换缓冲；两轮加载仍在', ['这条转换链缓冲 4 → 1', '两轮读取 2 → 2'], pane('P34 · 独立缓冲', [n('read1', '第二轮加载', 'mem_vec_14 · 16 KiB', 20, 'same', ['qh_q_f32']), n('cast0', 'INT32 结果', 'mem_vec_16 · 16 KiB', 100, 'same', ['qh_q_i32']), n('cast1', 'FP16 结果', 'mem_vec_17 · 8 KiB', 180, 'same', ['qh_q_half']), n('cast2', 'INT8 结果', 'mem_vec_18 · 4 KiB', 260, 'same', ['qh_i8'])]), pane('P35 · 同址复用', [n('read1', '第二轮加载仍保留', 'mem_vec_7 · 16 KiB', 20, 'reused', ['qh_q_f32']), n('cast0', 'INT32 结果', 'mem_vec_7 · 覆盖复用', 100, 'reused', ['qh_q_i32']), n('cast1', 'FP16 结果', 'mem_vec_7 · 覆盖复用', 180, 'reused', ['qh_q_half']), n('cast2', 'INT8 结果', 'mem_vec_7 · 覆盖复用', 260, 'reused', ['qh_i8'])]), '复用地址节省缓冲；不会自动将第一轮输入保留到第二轮。'),
    scene('最终读写粒度落到三个观测对象', ['输入 256B / 行', 'scale 4B / 行', 'INT8 64B / 行'], pane('最终 PTO · tile 对象', [n('read1', 'FP32 输入分片', '[64,64] · row_major', 20, 'same', ['tload']), n('scale', 'scale 输出', '[64,1] · col_major', 130, 'same', ['qh_scale_dq']), n('quant', 'INT8 输出分片', '[64,64] · row_major', 240, 'same', ['qh_i8'])], []), pane('PH001 · 对应读写点', [n('read1', 'TLOAD · 两轮', '内层 256B', 20, 'warning', ['TLOAD', 'tload', 'tile.load']), n('scale', 'TSTORE · scale', '内层 4B', 130, 'warning', ['qh_scale_dq', 'TSTORE', 'tile.store']), n('quant', 'TSTORE · INT8', '内层 64B', 240, 'warning', ['qh_i8', 'TSTORE', 'tile.store'])], []), '每行跨度来自静态形状与 dtype；片上布局不等于 GM 连续布局。'),
  ];
  const escape = value => String(value).replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' })[c]);
  function graph(p, selected) {
    const nodes = new Map(p.nodes.map(item => [item.id, item]));
    const edges = p.edges.map(edge => {
      const a = nodes.get(edge.from), b = nodes.get(edge.to);
      const x1 = a.x, y1 = a.y + 54, x2 = b.x, y2 = b.y;
      return '<path class="object-edge" d="M' + x1 + ' ' + y1 + ' C' + x1 + ' ' + ((y1 + y2) / 2) + ',' + x2 + ' ' + ((y1 + y2) / 2) + ',' + x2 + ' ' + (y2 - 5) + '"/><path class="object-arrow" d="M' + (x2 - 4) + ' ' + (y2 - 9) + ' L' + x2 + ' ' + (y2 - 4) + ' L' + (x2 + 4) + ' ' + (y2 - 9) + '"/>'
        + (edge.label ? '<text class="object-edge-label" x="' + ((x1 + x2) / 2 + 8) + '" y="' + ((y1 + y2) / 2) + '">' + escape(edge.label) + '</text>' : '');
    }).join('');
    return '<article class="object-pane"><h4>' + escape(p.label) + '</h4><svg viewBox="0 0 440 350" role="group" aria-label="' + escape(p.label) + ' 结构图">' + edges + p.nodes.map(node =>
      '<g role="button" tabindex="0" aria-pressed="' + (selected === node.id) + '" aria-label="' + escape(node.title + ' · ' + node.sub)
      + '" class="pass-object ' + node.state + (selected === node.id ? ' selected' : '') + '" data-pass-object="' + node.id + '"><rect x="' + (node.x - node.width / 2) + '" y="' + node.y + '" width="' + node.width + '" height="54" rx="7"/><text x="' + node.x + '" y="' + (node.y + 22) + '">' + escape(node.title) + '</text><text class="object-sub" x="' + node.x + '" y="' + (node.y + 41) + '">' + escape(node.sub) + '</text></g>').join('') + '</svg></article>';
  }
  window.PERF_PASS_OBJECTS = {
    sceneFor(kernel, mode, index, stage) {
      if (mode === 'setup') return index === 2 ? dependency(stage) : null;
      return ({ csa_merge_pack_publish: csa, qr_hadamard_matmul: matmul, qr_hadamard_quant: quant })[kernel]?.[index] || null;
    },
    object(scene, id) { return scene.after.nodes.find(n => n.id === id) || scene.before.nodes.find(n => n.id === id); },
    render(scene, selected) {
      return '<div class="object-verdict">' + escape(scene.title) + '</div><div class="object-deltas">' + scene.delta.map(text => '<span>' + escape(text) + '</span>').join('') + '</div>'
        + '<div class="object-legend"><span>□ 保留</span><span class="added">＋ 新增</span><span class="changed">↻ 改写</span><span class="reused">▧ 复用</span><span class="warning">! 告警</span><span>结构图 · 点击对象联动快照与代码</span></div>'
        + '<div class="object-panes">' + graph(scene.before, selected) + graph(scene.after, selected) + '</div><p class="object-note">' + escape(scene.note) + '</p>';
    },
  };
})();
