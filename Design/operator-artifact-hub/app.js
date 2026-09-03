(function () {
  'use strict';

  const stages = [
    { id: 'context', label: '上下文与来源', count: 1, size: '7.6 KB', path: 'kernel_config.py', note: '配置已采集；Commit 与环境指纹缺失', viewer: '配置查看', complete: '部分完整' },
    { id: 'compile', label: '编译过程', count: 44, size: '13.25 MB', path: 'passes_dump/ · report/', note: '42 个有序 Pass、内存报告和性能提示', viewer: 'Pass Diff', complete: '完整' },
    { id: 'codegen', label: '生成代码', count: 156, size: '9.70 MB', path: 'ptoas/ · kernels/ · orchestration/', note: '38 个 PTO 模块生成 39 个 callable kernel', viewer: '源码查看', complete: '完整' },
    { id: 'runtime', label: '运行时证据', count: 4, size: '10.56 MB', path: 'dfx_outputs/', note: '426 tasks、1,369 edges、50,556 events', viewer: 'Timeline / Graph', complete: '完整' },
    { id: 'diagnostics', label: '诊断与结论', count: 2, size: '45.3 KB', path: 'report/', note: '16 条性能提示；Right 空间达到 100%', viewer: '诊断查看', complete: '有告警' },
    { id: 'replay', label: '复现与验证', count: 1, size: '4.7 KB', path: 'debug/run.py', note: '可 Replay；无随机种子、Golden 或 Oracle', viewer: '脚本查看', complete: '部分完整' }
  ];

  const chain = [
    { id: 'pass', type: 'Pass 41', name: 'MaterializeRuntimeScopes', detail: '最终 IR 快照把运行时 Scope 物化为可生成的调度结构。' },
    { id: 'logical', type: 'Logical Kernel', name: 'fa_fused', detail: '一个逻辑 Kernel 在 SplitVectorKernel 后分裂为 AIC 与 AIV 两个 callable。' },
    { id: 'callable', type: 'func_id 22 / 23', name: 'fa_fused_aic · aiv', detail: 'kernel_config.py 与 name_map 共同提供 callable ID 到最终源码的稳定映射。' },
    { id: 'task', type: 'Runtime Task', name: 'task_id 8589…', detail: 'deps.json 通过 kernel_ids、task_id、Tensor 参数和依赖边描述运行时实例。' },
    { id: 'trace', type: 'Trace Event', name: 'AICore slice', detail: 'Swimlane 与 merged trace 使用 task_id 把一次 Core 执行关联回调度任务。' }
  ];

  const artifacts = [
    { stage: 'context', name: '运行与 Kernel 配置', path: 'kernel_config.py', count: '1', size: '7.6 KB', status: '部分完整', viewer: '配置查看' },
    { stage: 'compile', name: 'IR 逐 Pass 快照', path: 'passes_dump/00_frontend.py → 41_after_MaterializeRuntimeScopes.py', count: '42', size: '13.21 MB', status: '完整', viewer: 'Pass Diff' },
    { stage: 'compile', name: '编译诊断报告', path: 'report/', count: '2', size: '45.3 KB', status: '有告警', viewer: 'Report' },
    { stage: 'codegen', name: 'PTO 中间产物', path: 'ptoas/', count: '76', size: '1.24 MB', status: '完整', viewer: '源码查看' },
    { stage: 'codegen', name: 'AIC Kernel', path: 'kernels/aic/', count: '16', size: '1.62 MB', status: '完整', viewer: '源码 / Binary' },
    { stage: 'codegen', name: 'AIV Kernel', path: 'kernels/aiv/', count: '62', size: '5.70 MB', status: '完整', viewer: '源码 / Binary' },
    { stage: 'codegen', name: 'Orchestration', path: 'orchestration/', count: '2', size: '1.14 MB', status: '完整', viewer: '源码 / Binary' },
    { stage: 'runtime', name: '任务依赖图', path: 'dfx_outputs/deps.json', count: '1', size: '319 KB', status: '完整', viewer: 'Task Graph' },
    { stage: 'runtime', name: 'L2 Swimlane 记录', path: 'dfx_outputs/l2_swimlane_records.json', count: '1', size: '215 KB', status: '完整', viewer: 'Swimlane' },
    { stage: 'runtime', name: '合并运行时 Trace', path: 'dfx_outputs/merged_swimlane_20260625_185006.json', count: '1', size: '10.02 MB', status: '完整', viewer: 'Timeline' },
    { stage: 'runtime', name: 'Callable 名称映射', path: 'dfx_outputs/name_map__jit_decode_fwd_layers_20260625_184941.json', count: '1', size: '1.0 KB', status: '完整', viewer: 'JSON' },
    { stage: 'replay', name: 'Debug Replay', path: 'debug/run.py', count: '1', size: '4.7 KB', status: '部分完整', viewer: '脚本查看' }
  ];

  const graphNodes = [
    { id: 'p32', type: 'PASS', label: '32 · AllocateMemoryAddr', meta: 'IR snapshot', x: 4, y: 12, links: ['k36', 'diag'] },
    { id: 'p41', type: 'PASS', label: '41 · RuntimeScopes', meta: 'IR snapshot', x: 4, y: 57, links: ['k22', 'orch'] },
    { id: 'k36', type: 'CALLABLE', label: 'down_proj', meta: 'func_id 36 · AIC', x: 31, y: 6, links: ['t36', 'diag'] },
    { id: 'k22', type: 'CALLABLE', label: 'fa_fused_aic', meta: 'func_id 22 · AIC', x: 31, y: 45, links: ['t22'] },
    { id: 'orch', type: 'ORCHESTRATION', label: 'decode_fwd_layers', meta: 'aicpu entry', x: 31, y: 76, links: ['t22'] },
    { id: 't36', type: 'RUNTIME TASK', label: '85 task instances', meta: 'deps.json', x: 59, y: 6, links: ['trace'] },
    { id: 't22', type: 'RUNTIME TASK', label: 'task_id 8589…', meta: 'deps.json', x: 59, y: 51, links: ['trace'] },
    { id: 'trace', type: 'TRACE', label: 'AICore slices', meta: '50,556 events', x: 81, y: 31, links: [] },
    { id: 'diag', type: 'DIAGNOSTIC', label: 'Right · 100%', meta: 'memory report', x: 59, y: 78, links: [] }
  ];

  const policies = [
    { name: 'Manifest、摘要与结论', note: '身份、门禁、哈希、清理记录', value: '永久保留' },
    { name: 'Pass 与生成源码', note: '保留基线与失败 Run；普通 Run 可归档', value: '基线永久' },
    { name: 'Object 与 Shared Library', note: '可由同环境重建，但复现包需要', value: '保留 90 天' },
    { name: 'Runtime Trace', note: '体积最大，保留聚合摘要与异常切片', value: '冷存储 30 天' },
    { name: 'Tensor Dump', note: '敏感且高成本，按采集等级单独管理', value: '默认不采集' }
  ];

  const stageNames = Object.fromEntries(stages.map(item => [item.id, item.label]));
  const state = { view: 'overview', stage: 'compile', chain: 'pass', filter: 'all', query: '' };
  const $ = (selector, root) => (root || document).querySelector(selector);
  const $$ = (selector, root) => Array.from((root || document).querySelectorAll(selector));
  const esc = value => String(value).replace(/[&<>"']/g, char => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[char]));

  function renderStages() {
    $('#stage-track').innerHTML = stages.map(stage => `
      <button class="stage-card${state.stage === stage.id ? ' active' : ''}" data-stage="${stage.id}">
        <span class="stage-dot"></span>
        <strong>${esc(stage.label)}</strong>
        <small>${stage.count} 项</small>
        <em>${esc(stage.size)}</em>
      </button>`).join('');
  }

  function renderChain() {
    $('#evidence-chain').innerHTML = chain.map((item, index) => `${index ? '<span class="chain-arrow"></span>' : ''}
      <button class="chain-node${state.chain === item.id ? ' active' : ''}" data-chain="${item.id}">
        <small>${esc(item.type)}</small><strong>${esc(item.name)}</strong>
      </button>`).join('');
    const item = chain.find(entry => entry.id === state.chain) || chain[0];
    $('#chain-detail').innerHTML = `<strong>${esc(item.name)}</strong> · ${esc(item.detail)}`;
  }

  function renderFilters() {
    const items = [{ id: 'all', label: '全部 206' }].concat(stages.map(stage => ({ id: stage.id, label: stage.label })));
    $('#artifact-filters').innerHTML = items.map(item => `<button class="filter-chip${state.filter === item.id ? ' active' : ''}" data-filter="${item.id}">${esc(item.label)}</button>`).join('');
  }

  function renderArtifactRows() {
    const query = state.query.trim().toLowerCase();
    const rows = artifacts.filter(item => (state.filter === 'all' || item.stage === state.filter) && (!query || `${item.name} ${item.path} ${item.viewer}`.toLowerCase().includes(query)));
    $('#artifact-rows').innerHTML = rows.length ? rows.map((item, index) => `
      <tr data-artifact="${artifacts.indexOf(item)}">
        <td><div class="artifact-name"><span><svg><use href="#i-file"></use></svg></span><div><strong>${esc(item.name)}</strong><small>${esc(item.path)}</small></div></div></td>
        <td>${esc(stageNames[item.stage])}</td><td>${esc(item.count)}</td><td>${esc(item.size)}</td>
        <td><span class="table-status" style="color:${item.status === '有告警' ? 'var(--amber)' : item.status === '部分完整' ? '#9fb1ff' : 'var(--green)'}">${esc(item.status)}</span></td>
        <td><span class="viewer-pill">${esc(item.viewer)}</span></td>
      </tr>`).join('') : '<tr><td class="empty-row" colspan="6">没有匹配的产物</td></tr>';
  }

  function renderInspector(stageId) {
    const stage = stages.find(item => item.id === stageId) || stages[1];
    state.stage = stage.id;
    $('#inspector-title').textContent = stage.label;
    $('#inspector-body').innerHTML = `
      <section class="inspector-section">
        <div class="inspector-hero"><span><svg><use href="#i-box"></use></svg></span><div><strong>${esc(stage.label)}</strong><small>${esc(stage.count)} artifacts · ${esc(stage.size)}</small></div></div>
      </section>
      <section class="inspector-section"><h3>集合信息</h3><dl class="kv-list">
        <dt>位置</dt><dd><code>${esc(stage.path)}</code></dd>
        <dt>完整性</dt><dd>${esc(stage.complete)}</dd>
        <dt>查看方式</dt><dd>${esc(stage.viewer)}</dd>
        <dt>数据属性</dt><dd>原始证据 · 只读</dd>
      </dl></section>
      <section class="inspector-section"><h3>分析摘要</h3><div class="inspector-note">${esc(stage.note)}</div></section>
      <section class="inspector-section"><h3>可关联对象</h3><ul class="relation-list">
        <li><svg><use href="#i-link"></use></svg>Run 20260625_184941</li>
        <li><svg><use href="#i-link"></use></svg>39 个 callable kernel</li>
        <li><svg><use href="#i-link"></use></svg>426 个 runtime task</li>
      </ul></section>`;
    renderStages();
    openInspector();
  }

  function renderArtifactInspector(item) {
    $('#inspector-title').textContent = item.name;
    $('#inspector-body').innerHTML = `
      <section class="inspector-section"><div class="inspector-hero"><span><svg><use href="#i-file"></use></svg></span><div><strong>${esc(item.name)}</strong><small>${esc(item.count)} 项 · ${esc(item.size)}</small></div></div></section>
      <section class="inspector-section"><h3>Artifact metadata</h3><dl class="kv-list">
        <dt>阶段</dt><dd>${esc(stageNames[item.stage])}</dd><dt>路径</dt><dd><code>${esc(item.path)}</code></dd><dt>完整性</dt><dd>${esc(item.status)}</dd><dt>Viewer</dt><dd>${esc(item.viewer)}</dd>
      </dl></section>
      <section class="inspector-section"><h3>管理原则</h3><div class="inspector-note">文件保持不可变。产品中的摘要、Diff 和关系图需要记录 derived_from、分析器版本与生成时间。</div></section>`;
    openInspector();
  }

  function renderGraph() {
    const canvas = $('#relation-canvas');
    canvas.innerHTML = `<svg class="relation-lines" viewBox="0 0 1000 445" preserveAspectRatio="none"></svg>` + graphNodes.map(node => `
      <button class="graph-node" data-node="${node.id}" style="left:${node.x}%;top:${node.y}%"><small>${esc(node.type)}</small><strong>${esc(node.label)}</strong><em>${esc(node.meta)}</em></button>`).join('');
    requestAnimationFrame(drawGraphLines);
  }

  function drawGraphLines() {
    const canvas = $('#relation-canvas');
    const svg = $('.relation-lines', canvas);
    if (!canvas || !svg) return;
    const box = canvas.getBoundingClientRect();
    svg.setAttribute('viewBox', `0 0 ${box.width} ${box.height}`);
    const paths = [];
    graphNodes.forEach(source => {
      const sourceEl = $(`[data-node="${source.id}"]`, canvas);
      source.links.forEach(targetId => {
        const targetEl = $(`[data-node="${targetId}"]`, canvas);
        if (!sourceEl || !targetEl) return;
        const a = sourceEl.getBoundingClientRect();
        const b = targetEl.getBoundingClientRect();
        const x1 = a.right - box.left;
        const y1 = a.top + a.height / 2 - box.top;
        const x2 = b.left - box.left;
        const y2 = b.top + b.height / 2 - box.top;
        const mid = x1 + (x2 - x1) * .52;
        paths.push(`<path data-edge="${source.id} ${targetId}" d="M${x1},${y1} C${mid},${y1} ${mid},${y2} ${x2},${y2}"/>`);
      });
    });
    svg.innerHTML = paths.join('');
  }

  function focusGraph(id) {
    const selected = graphNodes.find(node => node.id === id);
    const related = new Set([id, ...(selected ? selected.links : [])]);
    graphNodes.forEach(node => { if (node.links.includes(id)) related.add(node.id); });
    $$('.graph-node').forEach(node => {
      node.classList.toggle('focus', node.dataset.node === id);
      node.classList.toggle('muted', !related.has(node.dataset.node));
    });
    $$('.relation-lines path').forEach(path => {
      const ids = path.dataset.edge.split(' ');
      path.style.opacity = ids.includes(id) ? '1' : '.15';
      path.style.stroke = ids.includes(id) ? 'rgba(109,141,255,.78)' : 'rgba(126,148,205,.28)';
    });
    if (selected) {
      $('#inspector-title').textContent = selected.label;
      $('#inspector-body').innerHTML = `<section class="inspector-section"><div class="inspector-hero"><span><svg><use href="#i-branch"></use></svg></span><div><strong>${esc(selected.label)}</strong><small>${esc(selected.type)}</small></div></div></section><section class="inspector-section"><h3>Relation identity</h3><dl class="kv-list"><dt>实体类型</dt><dd>${esc(selected.type)}</dd><dt>索引摘要</dt><dd>${esc(selected.meta)}</dd><dt>直接下游</dt><dd>${selected.links.length} 个对象</dd></dl></section><section class="inspector-section"><h3>关联原则</h3><div class="inspector-note">关联使用显式 ID 与 Manifest 关系，不依赖文件名猜测；原始文件只作为证据载体。</div></section>`;
      openInspector();
    }
  }

  function renderPolicies() {
    $('#policy-list').innerHTML = policies.map(policy => `<div class="policy-row"><span class="policy-icon"><svg><use href="#i-box"></use></svg></span><div><strong>${esc(policy.name)}</strong><small>${esc(policy.note)}</small></div><span class="policy-select">${esc(policy.value)}</span></div>`).join('');
  }

  function switchView(view) {
    state.view = view;
    $$('.view-tab').forEach(tab => { const active = tab.dataset.view === view; tab.classList.toggle('active', active); tab.setAttribute('aria-selected', String(active)); });
    $$('.view').forEach(panel => { const active = panel.dataset.panel === view; panel.classList.toggle('active', active); panel.hidden = !active; });
    if (view === 'relations') requestAnimationFrame(drawGraphLines);
  }

  function openInspector() {
    $('.app-shell').classList.remove('inspector-closed');
    $('#inspector').classList.remove('closed');
  }

  function closeInspector() {
    $('.app-shell').classList.add('inspector-closed');
    $('#inspector').classList.add('closed');
  }

  function showToast(message) {
    const toast = $('#toast');
    toast.textContent = message;
    toast.classList.add('show');
    window.clearTimeout(showToast.timer);
    showToast.timer = window.setTimeout(() => toast.classList.remove('show'), 2600);
  }

  function showModal() { $('#modal').hidden = false; }
  function hideModal() { $('#modal').hidden = true; }

  function bindEvents() {
    document.addEventListener('click', event => {
      const stage = event.target.closest('[data-stage]');
      if (stage) renderInspector(stage.dataset.stage);
      const chainNode = event.target.closest('[data-chain]');
      if (chainNode) { state.chain = chainNode.dataset.chain; renderChain(); }
      const tab = event.target.closest('[data-view]');
      if (tab && tab.classList.contains('view-tab')) switchView(tab.dataset.view);
      const switcher = event.target.closest('[data-switch]');
      if (switcher) switchView(switcher.dataset.switch);
      const filter = event.target.closest('[data-filter]');
      if (filter) { state.filter = filter.dataset.filter; renderFilters(); renderArtifactRows(); }
      const row = event.target.closest('[data-artifact]');
      if (row) renderArtifactInspector(artifacts[Number(row.dataset.artifact)]);
      const graphNode = event.target.closest('[data-node]');
      if (graphNode) focusGraph(graphNode.dataset.node);
      const gate = event.target.closest('[data-gate]');
      if (gate) showToast(`${gate.dataset.gate}门禁：当前状态来自 Manifest 摘要，而不是由“运行完成”自动推断。`);
    });

    $('#artifact-search').addEventListener('input', event => { state.query = event.target.value; renderArtifactRows(); });
    $('#global-search').addEventListener('keydown', event => {
      if (event.key === 'Enter') { switchView('artifacts'); state.query = event.target.value; $('#artifact-search').value = state.query; renderArtifactRows(); }
    });
    $('#compare-button').addEventListener('click', showModal);
    $('#modal-close').addEventListener('click', hideModal);
    $('#modal-confirm').addEventListener('click', hideModal);
    $('#modal').addEventListener('click', event => { if (event.target === $('#modal')) hideModal(); });
    $('#manifest-button').addEventListener('click', () => showToast('Demo：将导出 manifest.json、summary.json 与 relations.json。'));
    $('#open-gaps').addEventListener('click', () => { renderInspector('replay'); showToast('已定位到复现与验证缺口。'); });
    $('#close-inspector').addEventListener('click', closeInspector);
    $('#reset-graph').addEventListener('click', () => { $$('.graph-node').forEach(node => node.classList.remove('focus','muted')); $$('.relation-lines path').forEach(path => { path.style.opacity = ''; path.style.stroke = ''; }); });
    $('#simulate-policy').addEventListener('click', event => { $('.savings-meter i').style.width = '58%'; event.currentTarget.textContent = '已模拟 · 原始文件未变更'; showToast('仅模拟策略：没有删除或移动任何产物。'); });
    window.addEventListener('resize', () => { if (state.view === 'relations') drawGraphLines(); });
    document.addEventListener('keydown', event => {
      if ((event.metaKey || event.ctrlKey) && event.key.toLowerCase() === 'k') { event.preventDefault(); $('#global-search').focus(); }
      if (event.key === 'Escape') hideModal();
    });
  }

  renderStages();
  renderChain();
  renderFilters();
  renderArtifactRows();
  renderGraph();
  renderPolicies();
  renderInspector('compile');
  bindEvents();
}());
