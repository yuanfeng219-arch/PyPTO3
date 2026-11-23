/* Main ratio-4 compressor projection. Source contract: decode_compressor_ratio4.py:62–69, 94–136. */
(function (global) {
  'use strict';
  const data = global.PTO_CSA_COMPRESSOR_DATA;
  const stage = global.PTO_CSA_RUNTIME_DATA.stages.find(item => item.id === 'cache/main');
  const phases = ['generation', 'ordering', 'assignment'];
  const phaseNames = {generation:'Work Generation', ordering:'Block Ordering', assignment:'Core Assignment'};
  const reduceMotion = global.matchMedia('(prefers-reduced-motion: reduce)').matches;
  const [mainSource, indexSource] = ['decode_compressor_ratio4.py', 'decode_indexer_compressor.py'];
  const definitions = {
    kv_score_proj: [mainSource, 110, 136, [[118,'K Tile Pipeline'],[128,'双投影累加'],[135,'KV / Score 写回']]],
    scatter_softmax_pool: [mainSource, 145, 215, [[148,'定位压缩边界'],[154,'读取历史 Ring 与当前投影'],[209,'Online Softmax 累加与输出']]],
    compress_state_commit: [mainSource, 221, 233, [[223,'遍历请求 Token'],[226,'提交本步投影']]],
    rmsnorm_rope_cache_write: [mainSource, 237, 287, [[246,'RMSNorm'],[254,'NoPE / RoPE'],[279,'按 Slot 写入主 Cache']]],
    kv_score_proj_0: [indexSource, 112, 142, [[120,'K Tile Pipeline'],[130,'双投影累加'],[137,'KV / Score 写回']]],
    scatter_softmax_pool_0: [indexSource, 148, 225, [[151,'定位压缩边界'],[157,'读取历史 Ring 与当前投影'],[212,'Online Softmax 累加与输出']]],
    compress_state_commit_0: [indexSource, 227, 244, [[229,'遍历请求 Token'],[232,'提交本步投影']]],
    rmsnorm_rope: [indexSource, 245, 294, [[255,'RMS 归约'],[263,'NoPE / RoPE']]],
    kv_hadamard: [indexSource, 297, 304, [[301,'Hadamard 变换']]],
    kv_and_cache_write: [indexSource, 306, 335, [[308,'量化与 Scale'],[325,'按 Slot 写入 Indexer Cache']]],
  };
  const localObjects = {
    scatter_softmax_pool: ['每个 Block 对应一个请求', '当前 KV / Score、APE、历史状态 Ring', '跨窗口在线 Softmax 池化后的 KV'],
    scatter_softmax_pool_0: ['每个 Block 对应一个请求', 'Indexer KV / Score、APE、Indexer 状态 Ring', 'Indexer 池化后的 KV'],
    compress_state_commit: ['每个 Block 对应一个请求', '本步投影、APE、state_slot_mapping', '主 Compressor 状态 Ring'],
    compress_state_commit_0: ['每个 Block 对应一个请求', 'Indexer 投影、APE、state_slot_mapping', 'Indexer 状态 Ring'],
    rmsnorm_rope_cache_write: ['每个 Block 处理 16 行 Tile', '池化 KV、Norm 权重、RoPE、Slot 映射', '归一化 KV 与主 Cache'],
    rmsnorm_rope: ['每个 Block 处理 16 行 Tile', 'Indexer 池化 KV、Norm 权重、RoPE', '供 Hadamard 消费的 KV'],
    kv_hadamard: ['每个 Block 处理 16 行 Tile', '归一化 KV、Hadamard 权重', '变换后的 Indexer KV'],
    kv_and_cache_write: ['每个 Block 处理 16 行 Tile', '变换后的 KV、Indexer Slot 映射', 'INT8 Indexer Cache 与反量化 Scale'],
  };
  const isProjection = name => name.startsWith('kv_score_proj');
  const kernelByName = new Map(stage.kernels.map(k => [k.name, k]));
  const rows = 16, columns = 64, inner = 512, iterations = 4096 / inner;
  const columnTiles = 1024 / columns;
  const ops = {
    x: ['Load X', 120, 121, 'MTE2', 'pto.tload'],
    wkv: ['Load Wkv', 122, 126, 'MTE2', 'pto.tload'],
    wgate: ['Load Wgate', 122, 127, 'MTE2', 'pto.tload'],
    kv: ['KV Matmul', 128, 132, 'M', 'pto.tmatmul'],
    score: ['Score Matmul', 128, 133, 'M', 'pto.tmatmul'],
    kvOut: ['Write KV', 135, 135, 'FIX', 'pto.tstore'],
    scoreOut: ['Write Score', 136, 136, 'FIX', 'pto.tstore'],
  };
  const escape = value => String(value).replace(/[&<>"']/g, char =>
    ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'})[char]);
  const tile = (block, name) => {
    const n = name === 'kv_score_proj_0' ? 32 : columns;
    const nTiles = name === 'kv_score_proj_0' ? 8 : columnTiles;
    return { row: Math.floor(block / nTiles) * rows, col: block % nTiles * n, n };
  };
  const range = (start, size) => `[${start}, ${start + size})`;
  const ticks = count => Array.from({ length: count }, (_, i) => i);
  const button = (label, attr, value, selected = false) =>
    `<button type="button" class="btn btn-sm${selected ? ' is-selected' : ''}" ${attr}="${escape(value)}">${escape(label)}</button>`;

  function operationGraph(expanded, name) {
    const count = expanded ? iterations : 1, nodes = [], edges = [];
    const add = (op, step, col, row) => {
      const repeated = step != null;
      const accumulates = expanded && (op === 'kv' || op === 'score') && step > 0;
      const node = {
        id: repeated ? `${op}:${step}` : op, op, step: repeated && expanded ? step : null,
        col, row, title: ops[op][0] + (accumulates ? ' Acc' : ''),
        detail: repeated ? (expanded ? `K ${range(step * inner, inner)}` : `×${iterations} · K Tile`) : `FP32 · 16 × ${tile(0, name).n}`,
      };
      nodes.push(node);
      return node.id;
    };
    const link = (from, to, state = false) => edges.push({ from, to, state });
    for (let i = 0; i < count; i++) {
      const x = add('x', i, i, 2);
      const kv = add('kv', i, i, 1), score = add('score', i, i, 3);
      link(add('wkv', i, i, 0), kv);
      link(add('wgate', i, i, 4), score);
      link(x, kv);
      link(x, score);
      if (i > 0) {
        link(`kv:${i - 1}`, kv, true);
        link(`score:${i - 1}`, score, true);
      }
    }
    link(`kv:${count - 1}`, add('kvOut', null, count, 1));
    link(`score:${count - 1}`, add('scoreOut', null, count, 3));
    return { nodes, edges, columns: count + 1 };
  }

  function glyph(title, detail, role, unit, attributes) {
    return `<button type="button" class="viz2-instance-node csa-kvsp-node" data-role="${role}" ${attributes}>
      <span class="viz2-inst-head"><b>${escape(title)}</b>${unit ? `<em class="viz2-inst-unit" data-unit="${unit.toLowerCase()}">${unit}</em>` : ''}</span>
      <small>${escape(detail)}</small></button>`;
  }

  function graphMarkup(graph, selected) {
    const width = graph.columns * 184, height = 372;
    const x = n => n.col * 184 + 16, y = n => n.row * 72 + 16;
    const byId = new Map(graph.nodes.map(node => [node.id, node]));
    const selectedIds = new Set(graph.nodes.filter(n => selected?.op === n.op && (n.step === null || selected.step == null || selected.step === n.step)).map(n => n.id));
    const neighbors = new Set(selectedIds);
    const paths = graph.edges.map(edge => {
      const active = selectedIds.has(edge.from) || selectedIds.has(edge.to);
      if (active) { neighbors.add(edge.from); neighbors.add(edge.to); }
      const kind = edge.state ? 'accum' : 'data';
      const attrs = `data-kind="${kind}" class="${active ? 'is-related' : ''}"`;
      const a = byId.get(edge.from), b = byId.get(edge.to);
      if (a.col === b.col) {
        const down = a.row < b.row, offset = a.op === 'x' ? 88 : 60;
        return `<path ${attrs} d="M${x(a)+offset} ${y(a)+(down?48:0)} V${y(b)+(down?0:48)}"/>`;
      }
      return `<path ${attrs} d="M${x(a)+152} ${y(a)+24} H${x(b)}"/>`;
    }).join('');
    return `<div class="csa-kvsp-graph ${selectedIds.size ? 'has-focus' : ''}" style="width:${width}px;height:${height}px">
      <svg width="${width}" height="${height}" aria-hidden="true"><defs><marker id="csa-kvsp-arrow" viewBox="0 0 8 8" markerWidth="5" markerHeight="5" refX="7" refY="4" orient="auto"><path d="M0 0L8 4L0 8Z"/></marker></defs>${paths}</svg>
      ${graph.nodes.map(n => {
        const role = n.op.endsWith('Out') ? 'aggregate' : n.row % 2 === 0 ? 'activation' : 'operation';
        return glyph(n.title, n.detail, role, role === 'operation' ? 'AIC' : '',
          `style="left:${x(n)}px;top:${y(n)}px" data-kvsp-op="${n.id}" data-related="${neighbors.has(n.id)}" aria-pressed="${selectedIds.has(n.id)}"`);
      }).join('')}</div>`;
  }

  function mount(host, { onSource, onRank, onExit }) {
    let rank = 0, mode = 'assignment', scope = 'inter', scale = 'structure', block = 0, kernel = 'kv_score_proj';
    let sliceId = null, selected = null, blockSelected = false, inspectorOpen = false;
    let phase = 'generation', timer = null, loop = false, zoom = 1;
    const stop = () => { if (timer != null) global.clearInterval(timer); timer = null; };
    function play() {
      stop();
      timer = global.setInterval(() => {
        const next = phases.indexOf(phase) + 1;
        if (next === phases.length && !loop) stop();
        else phase = phases[next % phases.length];
        render(true);
      }, 1500);
    }
    function setKernel(name) {
      kernel = name; block = 0; selected = null; sliceId = null; blockSelected = false;
    }
    const phaseButton = (name, attr, value, pressed) =>
      button(name, attr, value, pressed).replace('<button ', `<button aria-pressed="${pressed}" `);
    const current = () => data.ranks[rank];
    const blocks = () => kernelByName.get(kernel).blockNum;

    function blockGrid(indices, label, name = kernel) {
      return `<div class="csa-kvsp-blocks" role="group" aria-label="${label}">${indices.map(index =>
        `<button type="button" class="btn btn-sm" data-kvsp-block="${index}" data-kvsp-block-kernel="${name}" aria-label="查看 Block ${index}" title="${isProjection(name) ? `M ${range(tile(index, name).row, rows)} · N ${range(tile(index, name).col, tile(index, name).n)}` : `逻辑 Block ${index}`}">${index}</button>`).join('')}</div>`;
    }

    function assignment() {
      if (scope === 'intra') {
        if (!isProjection(kernel)) return localAssignment();
        const { row, col, n } = tile(block, kernel);
        return `<h3>Block ${block} · Tile 与存储</h3>
          <p>M ${range(row, rows)} · N ${range(col, n)} · ${iterations} × K${inner}</p>
          <div class="csa-sv-table-wrap"><table class="csa-sv-table"><thead><tr><th>对象</th><th>逻辑 Tile</th><th>存储路径</th></tr></thead><tbody>
          <tr><td>X</td><td>BF16 · 16 × 512 / K Tile</td><td>GM → L1 (Mat) → L0A</td></tr>
          <tr><td>Wkv / Wgate</td><td>BF16 · ${n} × 512 / K Tile，各一份</td><td>GM → L1 (Mat) → L0B · 转置权重</td></tr>
          <tr><td>KV / Score 累加器</td><td>FP32 · 16 × ${n}，各一份</td><td>L0C → GM · 两个输出 Tile</td></tr>
          </tbody></table></div>${button('查看源码', 'data-kvsp-source', '110:136')}`;
      }
      return assignmentCanvas();
    }

    function assignmentCanvas() {
      const count = blocks(), projection = isProjection(kernel);
      const perRow = projection ? (kernel === 'kv_score_proj_0' ? 8 : 16) : 16;
      const groupCount = Math.ceil(count / perRow), sequenceColumns = 32;
      const records = current().kernels[kernel];
      const cores = new Map();
      for (const record of records) {
        if (!cores.has(record[1])) cores.set(record[1], []);
        cores.get(record[1]).push(record);
      }
      const byCore = [...cores].sort((a,b)=>a[0]-b[0]);
      const showMeasured = phase === 'assignment';
      const ordered = phase !== 'generation';
      const gridColumns = ordered ? sequenceColumns : perRow;
      const width = 1080, step = 27, rowStep = 23, left = 115, top = 68;
      const taskHeight = Math.ceil(count/gridColumns)*rowStep+top+14;
      const measuredTop = top;
      const height = showMeasured ? measuredTop+byCore.length*rowStep+18 : Math.max(310,taskHeight+22);
      const ownership = projection ? `${groupCount} M Tiles × ${perRow} N Tiles` : /pool|commit/.test(kernel) ? `${count} requests × 1 Block` : `${count} 个 16 行 Tiles × 1 Block`;
      const caption = phase === 'generation' ? `${ownership} → ${count} Work Items` :
        phase === 'ordering' ? `SPMD Block 0–${count-1} · 按 block_idx 编号` :
        `${count} 逻辑 Blocks · ${records.length} 实测切片 · ${cores.size} 个物理核`;
      return `<div class="csa-qk-canvas-head csa-compressor-controls">
        <div><h2>${phaseNames[phase]}</h2><p>${caption}</p></div>
        <div class="csa-qk-canvas-controls">
          ${button(timer == null ? '播放' : '暂停','data-kvsp-play','')}
          ${button('重播','data-kvsp-replay','')}
          <div class="segmented-control segmented-control-muted" role="group" aria-label="任务分配阶段">
            ${phases.map((p,i)=>phaseButton(['Generation','Ordering','Assignment'][i],'data-kvsp-phase',p,p===phase)).join('')}
          </div><label><input type="checkbox" data-kvsp-loop ${loop?'checked':''}> 循环</label>
        </div></div>
        <div class="csa-compressor-sheet" style="--compressor-zoom:${zoom}">
          <svg class="csa-compressor-assignment" width="${width}" height="${height}" viewBox="0 0 ${width} ${height}" role="group" aria-label="Compressor Work Assignment">
          <text x="22" y="26" class="csa-qk-canvas-axis">${escape(phaseNames[phase] + ' · ' + caption)}</text>
          ${!showMeasured ? ticks(Math.ceil(count/gridColumns)).map(i=>`<g class="csa-qk-canvas-group">
            <rect x="100" y="${top+i*rowStep-3}" width="${gridColumns*step+20}" height="22" rx="4"/>
            <text x="20" y="${top+i*rowStep+10}">${!ordered&&projection?'M '+i:'B '+i*gridColumns}</text></g>`).join('') : ''}
          ${!showMeasured ? ticks(count).map(i=>{
            const x=left+(i%gridColumns)*step,y=top+Math.floor(i/gridColumns)*rowStep;
            const active=blockSelected&&block===i;
            return `<g class="csa-qk-canvas-item is-valid ${active?'is-selected':''}" role="button" tabindex="0"
              data-kvsp-logical="${i}" data-x="${x}" data-y="${y}" style="transform:translate(${x}px,${y}px)" aria-pressed="${active}" aria-label="逻辑 Block ${i}">
              <rect width="22" height="16" rx="3"/><title>Block ${i}</title></g>`;
          }).join('') : ''}
          ${showMeasured ? `<text x="22" y="${measuredTop-22}" class="csa-qk-canvas-axis">按物理核分组 · 每个色块是一条实测切片</text>
            ${byCore.map(([core, entries],i)=>`<g class="csa-qk-canvas-lane">
              <rect x="100" y="${measuredTop+i*rowStep-3}" width="916" height="22" rx="4"/>
              <text x="20" y="${measuredTop+i*rowStep+10}">${core<24?'AIC':'AIV'} ${core}</text></g>
              ${entries.map(([id,,time,dur],j)=>`<rect class="csa-compressor-slice" role="button" tabindex="0" data-kvsp-slice="${id}" data-unit="${core<24?'aic':'aiv'}"
                aria-pressed="${sliceId===id}" aria-label="物理切片 ${id}，Core ${core}" x="${left+j*Math.min(step,875/entries.length)}" y="${measuredTop+i*rowStep}" width="${Math.min(22,800/entries.length)}" height="16" rx="3"><title>#${id} · ${time} µs · ${dur} µs</title></rect>`).join('')}`).join('')}` : ''}
          </svg></div>`;
    }

    const classify = op => /load/.test(op) ? 'Load' : /store/.test(op) ? 'Store' :
      /extract|mov/.test(op) ? 'Tile Move / Extract' : /matmul/.test(op) ? 'Cube' :
      /flag|barrier/.test(op) ? 'Sync' : 'Vector / Scalar';

    function irTable(withSource) {
      return `<div class="csa-sv-table-wrap"><table class="csa-sv-table"><thead><tr><th>PTO IR 行</th><th>指令</th><th>类别</th>${withSource ? '<th>源码</th>' : ''}</tr></thead><tbody>
        ${data.ir[kernel].map(([op,line,py])=>`<tr><td>${line}</td><td>${op}</td><td>${classify(op)}</td>
          ${withSource ? `<td>${py ? button('L'+py, 'data-kvsp-source', py+':'+py) : '—'}</td>` : ''}</tr>`).join('')}</tbody></table></div>`;
    }

    function localAssignment() {
      const groups = new Map();
      for (const [op] of data.ir[kernel]) {
        const type = classify(op);
        groups.set(type, (groups.get(type) || 0) + 1);
      }
      return `<h3>${kernel} · Block ${block} · ${kernelByName.get(kernel).coreType}</h3>
        <dl class="csa-sv-dl">${['Block 归属','读取','写入'].map((label,i)=>`<dt>${label}</dt><dd>${localObjects[kernel][i]}</dd>`).join('')}</dl>
        <div class="csa-kvsp-cores">${[...groups].map(([name, count])=>`<section class="csa-sv-card"><b>${name}</b><small>${count} 个编译指令位置</small></section>`).join('')}</div>`;
    }

    function stageGraph() {
      const positions = [
        ['kv_score_proj',0,0],['scatter_softmax_pool',1,0],['compress_state_commit',2,1],['rmsnorm_rope_cache_write',2,0],
        ['kv_score_proj_0',0,2],['scatter_softmax_pool_0',1,2],['compress_state_commit_0',2,3],['rmsnorm_rope',2,2],
        ['kv_hadamard',3,2],['kv_and_cache_write',4,2],
      ];
      const xy = new Map(positions.map(([name,c,r])=>[name,[c*224+16,r*82+38+(r>=2?38:0)]]));
      const pairs = new Map();
      for (const edge of stage.internalEdges) {
        const key = edge.from+'>'+edge.to;
        if (!pairs.has(key)) pairs.set(key,{...edge,sources:[]});
        pairs.get(key).sources.push(edge.source);
      }
      const paths = [...pairs.values()].map(edge=>{
        const [x,y] = xy.get(edge.from), [tx,ty] = xy.get(edge.to);
        const offset = /commit/.test(edge.to) ? 10 : 0;
        const start = y+24+offset, finish = ty+24;
        const route = tx-x>224
          ? `M${x+190} ${start} H${x+198} Q${x+206} ${start} ${x+206} ${start+8}
            V${finish+7} Q${x+206} ${finish+15} ${x+214} ${finish+15} H${tx}`
          : `M${x+190} ${start} C${x+210} ${start} ${tx-20} ${finish} ${tx} ${finish}`;
        return `<path class="${edge.sources.includes('tensormap') ? '' : 'is-state'}" d="${route}"/>`;
      });
      return `<p>Main Compressor 与 Indexer Compressor · 实线是 Tensor 关系，虚线是显式依赖</p>
        <div class="csa-kvsp-graph csa-compressor-stage-graph" style="width:1130px;height:390px"><svg width="1130" height="390" aria-hidden="true">
        <rect class="csa-compressor-region" x="5" y="5" width="1110" height="171" rx="9"/><text class="csa-qp-row-label" x="16" y="25">Main Compressor</text>
        <rect class="csa-compressor-region" x="5" y="200" width="1110" height="177" rx="9"/><text class="csa-qp-row-label" x="16" y="225">Indexer Compressor</text>
        <defs><marker id="csa-kvsp-arrow" viewBox="0 0 8 8" markerWidth="5" markerHeight="5" refX="7" refY="4" orient="auto"><path d="M0 0L8 4L0 8Z"/></marker></defs>${paths.join('')}</svg>
        ${positions.map(([name])=>{const [x,y]=xy.get(name);const k=kernelByName.get(name);
          return glyph(name, `${k.blockNum} Blocks`, 'operation', k.coreType,
            `style="left:${x}px;top:${y}px" data-kvsp-kernel="${name}" aria-pressed="${kernel===name}"`);
        }).join('')}</div>`;
    }

    function computation() {
      if (scope === 'intra' && isProjection(kernel)) return graphMarkup(operationGraph(scale === 'instances', kernel), selected);
      if (scope === 'intra') return localGraph();
      return stageGraph() + (scale === 'instances'
        ? `<h3>逻辑执行实例</h3><p>按编译任务展开。选择 Block 查看它的核内计算。</p>
          ${stage.kernels.map(k => `<details class="csa-sv-card" ${k.name === kernel ? 'open' : ''}>
            <summary>${k.name} · ${k.blockNum} Blocks</summary>${blockGrid(ticks(k.blockNum), k.name + ' SPMD 实例', k.name)}</details>`).join('')}`
        : `<p>选中 Kernel 查看分配与源码，或进入核内查看选定 Block。</p>`);
    }

    function localGraph() {
      const [, , end, stages] = definitions[kernel];
      const expanded = scale === 'instances';
      const unit=kernelByName.get(kernel).coreType;
      const width=stages.length*225+24;
      return `<h3>${kernel} · Block ${block}</h3>
        ${expanded ? '<p>按结构节点展开编译指令位置。实际循环次数与条件分支随输入变化。</p>' : ''}
        <div class="csa-kvsp-graph csa-compressor-stage-graph" style="width:${width}px;height:80px">
        <svg width="${width}" height="80" aria-hidden="true"><defs><marker id="csa-kvsp-arrow" viewBox="0 0 8 8" markerWidth="5" markerHeight="5" refX="7" refY="4" orient="auto"><path d="M0 0L8 4L0 8Z"/></marker></defs>
        ${stages.slice(1).map((_,i)=>`<path d="M${206+i*225} 40H${241+i*225}"/>`).join('')}</svg>
        ${stages.map(([line,label],i)=>glyph(label,'源码 L'+line,'operation',unit,
          `style="left:${16+i*225}px;top:16px" data-kvsp-site-group="${i}" aria-pressed="${selected?.group===i}"`)).join('')}</div>
        ${expanded ? `<div class="csa-compressor-flow">${stages.map(([first,label],i)=>{
          const last=stages[i+1]?stages[i+1][0]-1:end;
          const sites=data.ir[kernel].filter(([, , line])=>line>=first&&line<=last);
          return `<details class="csa-sv-card" ${selected?.group===i||selected==null?'open':''}><summary>${label} · ${sites.length}</summary>
            <ul>${sites.map(([op,line])=>`<li>${button(op+' · IR L'+line,'data-kvsp-site',line,selected?.site===line)}</li>`).join('')}</ul></details>`;
        }).join('')}</div>` : ''}`;
    }

    function localPipeline() {
      const sites = data.ir[kernel], count = sites.length;
      const lanes = ['Load', 'Tile Move / Extract', 'Vector / Scalar', 'Cube', 'Store', 'Sync']
        .filter(lane => sites.some(([op]) => classify(op) === lane));
      return `<h3>${kernel} · Block ${block} · 静态编译顺序</h3>
        <p>横轴是 PTO IR 指令位置序号。循环与分支随输入变化，核内指令时间戳未采集。</p>
        <svg class="csa-kvsp-timeline" viewBox="0 0 1100 ${lanes.length*34+36}" role="img" aria-label="按指令类别分组的编译位置">
          ${lanes.map((lane,i)=>`<text x="10" y="${30+i*34}">${lane}</text><path d="M155 ${34+i*34}H1080"/>`).join('')}
          ${sites.map(([op,line],i)=>`<rect role="button" tabindex="0" data-kvsp-site="${line}" aria-label="${op}，IR L${line}" aria-pressed="${selected?.site===line}" x="${158+i*914/Math.max(1,count)}" y="${18+lanes.indexOf(classify(op))*34}" width="${Math.min(11,Math.max(1,840/Math.max(1,count)))}" height="14" rx="2"><title>${op} · IR L${line}</title></rect>`).join('')}
        </svg><details class="csa-sv-card"><summary>PTO IR 与源码行</summary>${irTable(true)}</details>`;
    }

    function projectionPipeline() {
      const laneNames = ['MTE2 · GM → L1','MTE1 · L1 → L0','M · Cube','FIX · L0C → GM'];
      const laneY = i => 75+i*63, colX = i => 175+i*145;
      const opsByLane = ['x','extract','kv','kvOut'];
      const active = (op,i) => selected?.op===op && (selected.step==null || selected.step===i);
      const cell = (i,lane,labels) => {
        const op=opsByLane[lane], key=lane===3?op:`${op}:${i}`;
        return `<g class="csa-qp-node ${active(op,i)?'is-selected':''}" role="button" tabindex="0" data-kvsp-op="${key}" data-unit="${lane}" aria-label="${labels.join(' ')}">
          <rect x="${colX(i)}" y="${laneY(lane)}" width="123" height="38" rx="5"/>
          <text x="${colX(i)+9}" y="${laneY(lane)+15}">${labels[0]}</text><text class="csa-qp-boundary-note" x="${colX(i)+9}" y="${laneY(lane)+29}">${labels[1]}</text></g>`;
      };
      const edges=ticks(iterations).map(i=>{
        const x=colX(i)+61,acc=i+1<iterations?`<path class="csa-qp-timeline-reuse" d="M${colX(i)+123} ${laneY(2)+19}H${colX(i+1)}"/>`:'';
        return `<path class="csa-qp-timeline-data" d="M${x} ${laneY(0)+38}V${laneY(1)} M${x} ${laneY(1)+38}V${laneY(2)}"/>${acc}`;
      }).join('');
      return `<h3>Block ${block} · K Tile 因果泳道</h3><p>源码双级 Pipeline · 横轴为逻辑 K 索引</p>
        <div class="csa-compressor-sheet" style="--compressor-zoom:${zoom}"><svg class="csa-qp-svg csa-compressor-causal" width="1480" height="340" viewBox="0 0 1480 340" role="group" aria-label="投影核内因果泳道">
          <defs><marker id="csa-compressor-pipe-arrow" viewBox="0 0 8 8" markerWidth="5" markerHeight="5" refX="7" refY="4" orient="auto"><path d="M0 0L8 4L0 8Z"/></marker></defs>
          ${laneNames.map((name,i)=>`<text class="csa-qp-row-label" x="16" y="${laneY(i)+23}">${name}</text><path class="csa-compressor-lane-rule" d="M16 ${laneY(i)+48}H1472"/>`).join('')}
          ${ticks(iterations).map(i=>`<text class="csa-qp-row-label" x="${colX(i)}" y="46">K${i} · ${range(i*inner,inner)}</text>`).join('')}
          ${edges}<path class="csa-qp-timeline-data" d="M${colX(7)+123} ${laneY(2)+19} Q${colX(8)+60} ${laneY(2)+19} ${colX(8)+61} ${laneY(3)}"/>
          ${ticks(iterations).map(i=>cell(i,0,['Load X / Weights','BF16 · GM → L1'])+cell(i,1,['Tile Extract','2 × K256 / 投影'])+cell(i,2,[i?'KV / Score Acc':'KV / Score Init','FP32 · L0C'])).join('')}
          ${cell(iterations,3,['KV / Score Store','FP32 · L0C → GM'])}
        </svg></div>`;
    }

    function pipeline() {
      if (scope === 'intra') {
        if (!isProjection(kernel)) return localPipeline();
        return projectionPipeline();

      }
      const slices = Object.entries(current().kernels).flatMap(([name, entries]) => entries.map(e => [...e, name]));
      const start = Math.min(...slices.map(s => s[2])), end = Math.max(...slices.map(s => s[2] + s[3]));
      const toX = time => 74 + (time - start) / (end - start) * 1000;
      return `<h3>实测执行泳道 · Rank ${rank}</h3><p>横轴为合并记录时间（µs）。选择切片查看所属 Kernel、物理核和耗时。</p>
        <svg class="csa-kvsp-timeline" viewBox="0 0 1100 914" role="group" aria-label="物理 AIC 与 AIV 泳道">
        ${ticks(6).map(i => { const x = 74 + 200 * i; return `<text x="${x}" y="16">${(start+(end-start)*i/5).toFixed(1)}</text><path d="M${x} 24V902"/>`; }).join('')}
        ${ticks(72).map(i => `<text x="12" y="${40 + i * 12}">${i<24?'AIC':'AIV'} ${i}</text><path d="M72 ${42+i*12}H1074"/>`).join('')}
        ${slices.map(([id, core, time, duration, , name]) => `<rect role="button" tabindex="0" data-kvsp-slice="${id}" data-unit="${core<24?'aic':'aiv'}" data-highlight="${kernel===name}" aria-label="事件 ${id}，${name}，Core ${core}，${time} 微秒" aria-pressed="${sliceId === id}" x="${toX(time)}" y="${29+core*12}" width="${Math.max(1,toX(time+duration)-toX(time))}" height="10" rx="2"><title>#${id} · ${name} · Core ${core} · ${duration} µs</title></rect>`).join('')}
        </svg><small>${escape(current().source)} · Compressor</small>`;
    }

    function detail() {
      if (scope === 'inter' && sliceId != null && mode !== 'graph') {
        const [id, core, time, duration, elapsed] = Object.values(current().kernels).flat().find(s => s[0] === sliceId);
        return `<h3>${kernel} · 事件 #${id}</h3><dl class="csa-sv-dl"><dt>物理核</dt><dd>${core<24?'AIC':'AIV'} ${core}</dd><dt>起点</dt><dd>${time} µs</dd><dt>切片耗时</dt><dd>${duration} µs</dd><dt>Kernel 耗时</dt><dd>${elapsed} µs</dd></dl><p>记录未提供逻辑 Block ID。</p>`;
      }
      if (mode === 'assignment' && scope === 'inter' && blockSelected) {
        const [file,start,end]=definitions[kernel];
        const t=tile(block,kernel);
        const ownership=isProjection(kernel)?`M ${range(t.row,rows)} · N ${range(t.col,t.n)}`:/pool|commit/.test(kernel)?`Request ${block}`:`M ${range(block*rows,rows)}`;
        return `<h3>${kernel} · Block ${block}</h3><dl class="csa-sv-dl"><dt>Rank</dt><dd>${rank}</dd><dt>逻辑归属</dt><dd>${ownership}</dd><dt>身份</dt><dd>rank${rank}/${kernel}/block${block}</dd></dl>
          ${button('进入核内','data-kvsp-block',block)}${button('查看生成规则','data-kvsp-source',start+':'+end)}`;
      }
      if (scope !== 'intra') {
        const records = kernelByName.get(kernel);
        return `<h3>${kernel}</h3><p>${records.coreType} · ${records.blockNum} Blocks</p>${button('进入核内', 'data-kvsp-block', 0)}`;
      }
      if (!isProjection(kernel)) {
        const [, , end, stages]=definitions[kernel];
        const group=stages[selected?.group], site=data.ir[kernel].find(([,line])=>line===selected?.site);
        const line=site?.[2]||group?.[0];
        return `<h3>${site?.[0]||group?.[1]||kernel} · Block ${block}</h3>
          <dl class="csa-sv-dl">${['归属','输入','输出'].map((label,i)=>`<dt>${label}</dt><dd>${localObjects[kernel][i]}</dd>`).join('')}
          ${site?`<dt>PTO IR</dt><dd>${kernel}.pto L${site[1]}</dd>`:''}</dl>
          ${line?button('查看源码','data-kvsp-source',line+':'+(site?line:stages[selected.group+1]?.[0]-1||end)):''}`;
      }
      const { row, col, n } = tile(block, kernel);
      if (mode==='pipeline' && selected?.op==='extract') {
        const [,start,end]=definitions[kernel];
        const sites=data.ir[kernel].filter(([op])=>op==='pto.textract');
        return `<h3>Tile Extract · K${selected.step}</h3><dl class="csa-sv-dl"><dt>K 范围</dt><dd>${range(selected.step*inner,inner)}</dd>
          <dt>执行管线</dt><dd>MTE1 · L1 → L0A / L0B</dd><dt>数据关系</dt><dd>每个投影将 K512 拆为 2 × K256，供 Cube 累加</dd>
          <dt>PTO IR</dt><dd>${kernel}.pto · pto.textract · ${sites.length} 个静态位置</dd></dl>${button('查看源码','data-kvsp-source',start+':'+end)}`;
      }
      const rawOp = ['graph','pipeline'].includes(mode) && selected && ops[selected.op];
      const sourceOffset = selected?.op === 'kvOut' || selected?.op === 'scoreOut' ? 5 : 2;
      const op = rawOp && kernel === 'kv_score_proj_0' ? [rawOp[0],rawOp[1]+sourceOffset,rawOp[2]+sourceOffset,rawOp[3],rawOp[4]] : rawOp;
      const ir = op && (selected.step > 0 && op[4] === 'pto.tmatmul' ? 'pto.tmatmul.acc' : op[4]);
      const expanded = (scale === 'instances' || mode === 'pipeline') && selected?.step != null;
      const related = rawOp ? operationGraph(expanded, kernel) : null;
      const node = related?.nodes.find(n => n.op === selected.op && (!expanded || n.step === selected.step));
      const connections = key => related.edges.filter(edge => edge[key] === node.id).map(edge =>
        related.nodes.find(n => n.id === edge[key === 'to' ? 'from' : 'to']).title).join('、') || 'Block 边界';
      return `<h3>逻辑 Block ${block}</h3><dl class="csa-sv-dl"><dt>输出 M</dt><dd>${range(row, rows)}</dd><dt>输出 N</dt><dd>${range(col, n)}</dd></dl>
        ${op && node ? `<h3>${node.title}</h3><dl class="csa-sv-dl"><dt>对象 ID</dt><dd>rank${rank}/block${block}/${node.id}</dd>
        <dt>K 范围</dt><dd>${expanded ? range(selected.step * inner, inner) : node.detail}</dd><dt>输入</dt><dd>${connections('to')}</dd><dt>输出</dt><dd>${connections('from')}</dd>
        <dt>执行管线</dt><dd>${op[3]}</dd><dt>PTO IR</dt><dd>${ir} · ${kernel}.pto</dd></dl>
        ${button('查看源码', 'data-kvsp-source', op[1]+':'+op[2])}` : ''}`;
    }

    function render(preserve = false) {
      const previous = host.querySelector('.csa-sv-main');
      const offset = preserve && previous ? [previous.scrollLeft, previous.scrollTop] : null;
      const positions = mode==='assignment' && preserve ? new Map([...host.querySelectorAll('[data-kvsp-logical]')].map(el=>[el.dataset.kvspLogical,[Number(el.dataset.x),Number(el.dataset.y)]])) : null;
      const inspector = inspectorOpen ? detail() : '';
      const screen = ({assignment,graph:computation,pipeline})[mode]();
      const canZoom = mode !== 'assignment' || scope === 'inter';
      const content = mode === 'graph' || mode === 'pipeline' && scope === 'inter'
        ? `<div class="csa-compressor-zoom-content" style="--compressor-zoom:${zoom}">${screen}</div>` : screen;
      host.innerHTML = `<div class="csa-sv-shell"><header class="csa-sv-header"><div class="csa-sv-title-group">
        <div class="csa-sv-title-row"><h2>Compressor</h2></div><p class="csa-sv-sub">Main + Indexer · ${stage.kernels.length} Kernels · ${Object.values(current().kernels).flat().length} 实测切片 / Rank</p></div>
        <div class="csa-sv-controls">
        <div class="segmented-control segmented-control-muted" role="group" aria-label="Rank">${[0,1].map(r => button(`Rank ${r}`, 'data-kvsp-rank', r, r === rank)).join('')}</div>
        <div class="segmented-control segmented-control-muted" role="group" aria-label="核间核内">${button('核间', 'data-kvsp-scope', 'inter', scope === 'inter')}${button('核内', 'data-kvsp-scope', 'intra', scope === 'intra')}</div>
        <label>Kernel <select data-kvsp-kernel-picker>${stage.kernels.map(k=>`<option ${k.name===kernel?'selected':''}>${k.name}</option>`).join('')}</select></label>
        ${scope === 'intra' ? `<label>Block <select data-kvsp-block-picker>${ticks(blocks()).map(i => `<option ${i === block ? 'selected' : ''}>${i}</option>`).join('')}</select></label>` : ''}
        ${mode === 'graph' ? `<div class="segmented-control segmented-control-muted" role="group" aria-label="Graph 观察尺度">${button('Structure', 'data-kvsp-scale', 'structure', scale === 'structure')}${button('Instances', 'data-kvsp-scale', 'instances', scale === 'instances')}</div>` : ''}
        ${button('返回 CSA', 'data-kvsp-exit', '')}</div></header>
        <div class="csa-compressor-canvas"><div class="csa-sv-main csa-kvsp-main" tabindex="0" aria-label="Compressor 画布">${content}</div>
        ${canZoom?`<footer class="csa-compressor-footer"><span>${mode==='pipeline'&&scope==='inter'?'实测 µs · AIC / AIV':mode==='assignment'?'逻辑任务 / 实测分配':'静态计算'}</span>
          <div class="viz2-instance-tools">${button('−','data-kvsp-zoom','out')}<output>${Math.round(zoom*100)}%</output>${button('＋','data-kvsp-zoom','in')}${button('适应','data-kvsp-zoom','fit')}</div></footer>`:''}</div>
        <aside class="pto-ide-frame__pane csa-sv-inspector" ${inspector ? '' : 'hidden'}>
          <header class="pto-ide-frame__pane-header"><b>对象详情</b><button type="button" class="btn btn-ghost btn-sm" data-kvsp-close aria-label="关闭详情">×</button></header>
          <div class="pto-ide-frame__pane-body">${inspector}</div></aside></div>`;
      if (offset) host.querySelector('.csa-sv-main').scrollTo(...offset);
      if (positions && !reduceMotion) for (const el of host.querySelectorAll('[data-kvsp-logical]')) {
        const before=positions.get(el.dataset.kvspLogical), after=el.style.transform;
        if (before && (before[0]!==Number(el.dataset.x) || before[1]!==Number(el.dataset.y))) {
          el.animate([{transform:`translate(${before[0]}px,${before[1]}px)`},{transform:after}],{duration:420,easing:'cubic-bezier(.2,.7,.2,1)'});
        }
      }
    }

    function inspectSlice(id) {
      sliceId = Number(id);
      inspectorOpen = true;
      blockSelected = false;
      kernel = Object.keys(current().kernels).find(name => current().kernels[name].some(slice => slice[0] === sliceId));
      block = 0;
      selected = null;
      render(true);
    }
    host.addEventListener('keydown', event => {
      if (event.target.matches('[data-kvsp-slice],[data-kvsp-logical],[data-kvsp-op],[data-kvsp-site]') && event.target instanceof SVGElement && ['Enter', ' '].includes(event.key)) {
        event.preventDefault();
        const attr=['data-kvsp-slice','data-kvsp-logical','data-kvsp-op','data-kvsp-site'].find(name=>event.target.hasAttribute(name));
        const value=event.target.getAttribute(attr);
        event.target.dispatchEvent(new MouseEvent('click',{bubbles:true}));
        host.querySelector(`[${attr}="${value}"]`)?.focus();
      }
    });
    host.addEventListener('click', event => {
      const action = event.target.closest('button,[data-kvsp-slice],[data-kvsp-logical],[data-kvsp-op],[data-kvsp-site]');
      if (!action) return;
      const d = action.dataset;
      if (d.kvspExit != null) { stop(); return onExit(); }
      if (d.kvspClose != null) { inspectorOpen=false; selected=null; sliceId=null; blockSelected=false; return render(true); }
      if (d.kvspPhase) { stop(); phase=d.kvspPhase; return render(true); }
      if (d.kvspPlay != null) { if (timer==null) { if (phase==='assignment') phase='generation'; play(); } else stop(); return render(true); }
      if (d.kvspReplay != null) { phase='generation'; play(); return render(true); }
      if (d.kvspZoom) {
        const scene=host.querySelector('.csa-kvsp-graph,.csa-compressor-causal,.csa-kvsp-timeline,.csa-compressor-assignment');
        const viewport=host.querySelector('.csa-kvsp-main');
        zoom=d.kvspZoom==='fit'?Math.max(.25,Math.min(1,
          (viewport.clientWidth-48)/(scene?.getBoundingClientRect().width/zoom||viewport.clientWidth),
          (viewport.clientHeight-36)/(scene?.getBoundingClientRect().height/zoom||viewport.clientHeight)))
          :Math.max(.25,Math.min(2.5,zoom*(d.kvspZoom==='in'?1.25:.8)));
        return render(d.kvspZoom!=='fit');
      }
      if (d.kvspLogical != null) {
        block=Number(d.kvspLogical); blockSelected=true; sliceId=null; inspectorOpen=true; return render(true);
      }
      if (d.kvspSource) {
        const [file,start,end] = definitions[kernel];
        return onSource(file, ...(['110:136','118:136','108:136'].includes(d.kvspSource) ? [start,end] : d.kvspSource.split(':').map(Number)));
      }
      if (d.kvspSlice) return inspectSlice(d.kvspSlice);
      if (d.kvspRank != null) { rank = Number(d.kvspRank); sliceId = null; onRank(rank); }
      if (d.kvspScope) { if (scope!==d.kvspScope) zoom=1; scope=d.kvspScope; stop(); inspectorOpen=scope==='intra' && blockSelected; }
      if (d.kvspKernel) { setKernel(d.kvspKernel); inspectorOpen=true; }
      if (d.kvspScale) scale = d.kvspScale;
      if (d.kvspBlock != null) {
        if (d.kvspBlockKernel && kernel !== d.kvspBlockKernel) {
          kernel = d.kvspBlockKernel; selected = null; sliceId = null;
        }
        block = Number(d.kvspBlock); scope = 'intra'; stop(); inspectorOpen=true;
      }
      if (d.kvspSiteGroup != null || d.kvspSite != null) {
        const site=d.kvspSite==null?null:Number(d.kvspSite);
        const py=data.ir[kernel].find(([,line])=>line===site)?.[2];
        const region=definitions[kernel][3];
        const group=d.kvspSiteGroup!=null?Number(d.kvspSiteGroup):region.reduce((hit,[line],i)=>py>=line?i:hit,0);
        selected={group,site}; inspectorOpen=true; return render(true);
      }
      if (d.kvspOp) {
        const [op, index] = d.kvspOp.split(':');
        inspectorOpen=true;
        selected = { op, step: (mode==='pipeline' || scale === 'instances') && index != null ? Number(index) : selected?.op === op ? selected.step : null };
        return render(true);
      }
      render();
    });
    let drag = null;
    host.addEventListener('pointerdown', event => {
      const viewport=event.target.closest('.csa-kvsp-main');
      if (!viewport || event.button!==0 || event.target.closest('button, summary, select, table, [role="button"]')
        || !viewport.querySelector('svg')) return;
      drag={viewport,id:event.pointerId,x:event.clientX,y:event.clientY,left:viewport.scrollLeft,top:viewport.scrollTop};
      viewport.setPointerCapture(event.pointerId);
      viewport.classList.add('is-panning');
    });
    host.addEventListener('pointermove', event => {
      if (drag?.id!==event.pointerId) return;
      drag.viewport.scrollTo(drag.left+drag.x-event.clientX,drag.top+drag.y-event.clientY);
    });
    host.addEventListener('lostpointercapture', () => {
      drag?.viewport.classList.remove('is-panning');
      drag=null;
    });
    host.addEventListener('change', event => {
      if (event.target.matches('[data-kvsp-loop]')) { loop=event.target.checked; return; }
      if (event.target.matches('[data-kvsp-kernel-picker]')) {
        setKernel(event.target.value);
      } else if (event.target.matches('[data-kvsp-block-picker]')) block = Number(event.target.value);
      else return;
      render();
    });
    return { update(nextMode, nextRank) {
      if (rank !== nextRank) sliceId = null;
      if (mode !== nextMode) { stop(); zoom=1; }
      mode = nextMode;
      rank = nextRank;
      render();
    }, pause: stop };
  }

  global.PtoCsaCompressor = Object.freeze({ mount });
})(window);
