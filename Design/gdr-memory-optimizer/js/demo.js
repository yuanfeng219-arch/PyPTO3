(() => {
  'use strict';

  const $ = (selector, root = document) => root.querySelector(selector);
  const $$ = (selector, root = document) => Array.from(root.querySelectorAll(selector));

  const scenarios = {
    bwd: {
      label: 'gdr_bwd', baseline: 5024, workspace: 519.69, runtime: 1094.49, io: 589.13,
      devProgram: 105.48, dynamic: 96, metadata: 123.14, rootInner: 488.34, staticOutcast: 25,
    },
    fwd: {
      label: 'gdr_fwd', baseline: 2223, workspace: 382.73, runtime: 980.73, io: 322.63,
      devProgram: 96.55, dynamic: .06, metadata: 24.97, rootInner: 210.60, staticOutcast: 136.62,
    },
  };

  const recommendations = [
    {
      id: 'MEM-001', title: '性能缓冲按开关惰性分配', short: '关闭采集时不创建 76 × 10.3 MB perf buffer',
      priority: 'P0', defaultEnabled: true, saving: () => 782.90, confidence: '确定性',
      evidence: 'DeviceRunner::Init 在非 SIM 模式无条件分配 AICore 性能缓冲，实测 782.90 MB，与算子无关。',
      acceptance: 'AC-01：默认非采集运行中不产生该分配；开启后性能数据可正常采集。',
      steps: ['将 InitPerfData 绑定到明确的采集开关', '首次启用时线程安全地惰性创建', '进程内跨算子复用并记录实际分配量'],
    },
    {
      id: 'MEM-002', title: 'CtrlflowCache 按实际编码大小分配', short: '移除 DevAscendProgram 固定 100 MB cacheData',
      priority: 'P0', defaultEnabled: true, saving: (mode) => mode === 'eager' ? 95.37 : 72.97, confidence: 'Eager 确定',
      evidence: 'ctrlflowCache.cacheData 当前恒为 100,000,000 B（95.37 MB）；Eager 无提前发射时可直接消除。',
      acceptance: 'AC-02：无控制流缓存需求时不再出现固定 cacheData 预留；ACLGraph 按 Host 计算结果分配。',
      steps: ['消费 Host 控制流计算结果', '无需求时容量归零', '计算不可用时记录 fallback_reason 并回退'],
    },
    {
      id: 'MEM-003', title: 'RingBuffer 执行模式感知', short: 'Eager count=1，ACLGraph 按提前发射深度',
      priority: 'P0', defaultEnabled: false, saving: () => 0, confidence: '待采集',
      evidence: 'Eager 不存在控制流提前发射，但当前 RingBuffer 没有按模式收缩。Issue 未给出独立可归因的节省量。',
      acceptance: 'AC-03/04：Eager count=1；ACLGraph 按深度分配，相关回归全部通过。',
      steps: ['显式传递执行模式', '根据模式计算 count 与上限', '补充 Eager/ACLGraph 模式矩阵测试'],
    },
    {
      id: 'MEM-004', title: 'Eager 统一进入 Torch 内存池', short: 'RingBuffer 与 Workspace 共用生命周期',
      priority: 'P0', defaultEnabled: false, saving: () => 0, confidence: '待采集',
      evidence: '目前 Runtime 与 Torch 双通道分配造成峰值口径割裂；池化收益需结合 reserved/allocated 再测量。',
      acceptance: 'Eager 路径统一池化，功能与性能回归通过，跨 Launch 复用和释放边界明确。',
      steps: ['定义统一生命周期', '复用 RTS TilingData 随路通路', '补齐 reserved/allocated 峰值对账'],
    },
    {
      id: 'MEM-007', title: 'Profiling 增加 PyPTO 归因字段', short: '补齐 I/O、Runtime 直接申请与共享项',
      priority: 'P1', defaultEnabled: false, saving: () => 0, confidence: '观测能力',
      evidence: 'operator_memory 只能看到 atten::empty，memory_record 的 Total Allocated 不包含 PyPTO 直接申请。',
      acceptance: 'AC-07：可按 PyPTO 算子筛选并查看 I/O、Workspace、Runtime 直接申请和共享项。',
      steps: ['透传稳定 operator_id', '统一 allocation_source/lifecycle/component', '进程级共享项单列，避免重复累计'],
    },
  ];

  const ledgerBase = [
    { component: 'perf_buffer', lifecycle: 'process', source: 'rtMalloc', estimated: 782.90, requested: 782.90, state: '固定' },
    { component: 'device_args', lifecycle: 'process', source: 'rtMalloc', estimated: 1.35, requested: 1.35, state: '正常' },
    { component: 'backend_server.so', lifecycle: 'process', source: 'rtMalloc', estimated: .95, requested: .95, state: '正常' },
    { component: 'dev_ascend_program', lifecycle: 'operator', source: 'rtMalloc', estimated: 105.48, requested: 105.48, state: '偏大' },
    { component: 'ctrlflow_cache', lifecycle: 'operator', source: 'rtMalloc', estimated: 95.37, requested: 95.37, state: '固定' },
    { component: 'dynamic_cell_match', lifecycle: 'operator', source: 'rtMalloc', estimated: 96.00, requested: 96.00, state: '正常' },
    { component: 'workspace', lifecycle: 'launch', source: 'torch.empty', estimated: 519.69, requested: 519.69, state: '正常' },
    { component: 'user_io', lifecycle: 'launch', source: 'torch.empty', estimated: 589.13, requested: 589.13, state: '未归因' },
    { component: 'torch_reserved', lifecycle: 'launch', source: 'torch.pool', estimated: 2826.95, requested: 3708.00, state: '池化差异' },
  ];

  const state = {
    kernel: 'bwd', mode: 'eager', hardware: 'a3', tab: 'overview', selected: 'MEM-001',
    enabled: new Set(recommendations.filter((item) => item.defaultEnabled).map((item) => item.id)),
    profileMode: 'current', chart: null,
  };

  function fmt(value, digits = 2) {
    return Number(value).toLocaleString('zh-CN', { minimumFractionDigits: digits, maximumFractionDigits: digits });
  }

  function scenario() { return scenarios[state.kernel]; }

  function selectedSaving() {
    return recommendations.reduce((sum, item) => sum + (state.enabled.has(item.id) ? item.saving(state.mode, state.kernel) : 0), 0);
  }

  function projected() { return Math.max(0, scenario().baseline - selectedSaving()); }

  function setTab(tab) {
    state.tab = tab;
    $$('.tab-control-item').forEach((button) => {
      const active = button.dataset.tab === tab;
      button.classList.toggle('is-selected', active);
      button.setAttribute('aria-selected', String(active));
    });
    $$('.mo-view').forEach((view) => {
      const active = view.dataset.view === tab;
      view.classList.toggle('is-active', active);
      view.hidden = !active;
    });
    $$('.mo-tree-row').forEach((row) => row.classList.toggle('is-selected', row.dataset.openTab === tab));
    const titles = {
      overview: ['内存预算与优化模拟', '从固定预留到按需分配'],
      ledger: ['统一分配账本', 'Host budget ↔ Device actual'],
      profiling: ['Profiling 对账', 'operator_memory · memory_record'],
    };
    $('#viewTitle').textContent = titles[tab][0];
    $('#viewMeta').textContent = tab === 'overview' ? `${scenario().label} · ${state.mode} · ${state.hardware}` : titles[tab][1];
    if (tab === 'overview') requestAnimationFrame(renderChart);
  }

  function renderRecommendations() {
    $('#recommendationList').innerHTML = recommendations.map((item) => {
      const enabled = state.enabled.has(item.id);
      const selected = state.selected === item.id;
      const saving = item.saving(state.mode, state.kernel);
      return `<div class="mo-rec-row${selected ? ' is-selected' : ''}" data-rec-row="${item.id}">
        <input type="checkbox" aria-label="启用 ${item.title}" data-rec-toggle="${item.id}" ${enabled ? 'checked' : ''}>
        <button class="mo-rec-main" type="button" data-rec-select="${item.id}"><b>${item.id} · ${item.title}</b><small>${item.short}</small></button>
        <span class="mo-rec-saving${saving ? '' : ' pending'}">${saving ? `-${fmt(saving, 1)} MB` : item.confidence}</span>
      </div>`;
    }).join('');
    $('#enabledCount').textContent = `${state.enabled.size} / ${recommendations.length}`;
  }

  function renderInspector() {
    const item = recommendations.find((candidate) => candidate.id === state.selected) || recommendations[0];
    const saving = item.saving(state.mode, state.kernel);
    $('#inspectorId').textContent = item.id;
    $('#inspector').innerHTML = `
      <section class="mo-inspector-section"><span class="mo-inspector-kicker">${item.priority} · ${item.confidence}</span><h2>${item.title}</h2><p class="mo-inspector-lead">${item.evidence}</p><div class="mo-inspector-impact"><span>当前场景预计影响</span><strong>${saving ? `-${fmt(saving, 2)} MB` : '待采集'}</strong></div></section>
      <section class="mo-inspector-section"><span class="mo-inspector-kicker">CONTEXT</span><dl class="mo-inspector-dl"><div><dt>Kernel</dt><dd>${scenario().label}</dd></div><div><dt>Mode</dt><dd>${state.mode}</dd></div><div><dt>Hardware</dt><dd>${state.hardware.toUpperCase()}</dd></div><div><dt>Lifecycle</dt><dd>${item.id === 'MEM-001' ? 'process' : item.id === 'MEM-007' ? 'cross-source' : 'operator / launch'}</dd></div></dl></section>
      <section class="mo-inspector-section"><span class="mo-inspector-kicker">IMPLEMENTATION</span><ol class="mo-inspector-list">${item.steps.map((step) => `<li>${step}</li>`).join('')}</ol></section>
      <section class="mo-inspector-section"><span class="mo-inspector-kicker">ACCEPTANCE</span><p class="mo-inspector-lead">${item.acceptance}</p></section>
      <section class="mo-inspector-section mo-inspector-actions"><button class="btn btn-solid" type="button" data-inspector-toggle="${item.id}">${state.enabled.has(item.id) ? '从模拟中移除' : '加入预算模拟'}</button><button class="btn" type="button" data-open-tab="ledger">查看相关分配</button></section>`;
  }

  function renderMetrics() {
    const data = scenario();
    const saving = selectedSaving();
    const after = projected();
    const delta = data.baseline ? saving / data.baseline * 100 : 0;
    $('#metricBaseline').textContent = (data.baseline / 1000).toFixed(2);
    $('#metricProjected').textContent = (after / 1000).toFixed(2);
    $('#metricSaving').textContent = fmt(saving, 1);
    $('#metricWorkspace').textContent = fmt(data.workspace, 2);
    $('#impactDelta').textContent = `-${delta.toFixed(1)}%`;
    $('#treePeak').textContent = `${(data.baseline / 1000).toFixed(2)} GB`;
    $('#viewMeta').textContent = `${data.label} · ${state.mode} · ${state.hardware}`;
  }

  function renderChart() {
    if (state.tab !== 'overview' || !window.PtoTrainingMetricsChart) return;
    state.chart?.destroy?.();
    const data = scenario();
    const active = recommendations.filter((item) => state.enabled.has(item.id) && item.saving(state.mode, state.kernel) > 0);
    const steps = Array.from({ length: active.length + 1 }, (_, index) => index);
    const baseline = steps.map(() => data.baseline);
    const optimized = [data.baseline];
    active.forEach((item) => optimized.push(optimized[optimized.length - 1] - item.saving(state.mode, state.kernel)));
    state.chart = window.PtoTrainingMetricsChart.render('#impactChart', {
      steps,
      series: [
        { id: 'baseline', label: 'baseline', key: 'baseline', colorVar: '--highlight-copy-blue-source', axis: 'left' },
        { id: 'optimized', label: 'optimized', key: 'optimized', colorVar: '--highlight-ub-green-source', axis: 'left', emphasis: true },
      ],
      data: { baseline, optimized },
      cursor: steps[steps.length - 1],
      options: { height: 230 },
    });
  }

  function renderLifecycle() {
    const data = scenario();
    const saving = selectedSaving();
    const process = Math.max(0, 785.2 - (state.enabled.has('MEM-001') ? 782.9 : 0));
    const operator = data.devProgram + data.metadata + data.dynamic - (state.enabled.has('MEM-002') ? recommendations[1].saving(state.mode, state.kernel) : 0);
    const launch = data.workspace + data.io;
    const rows = [
      { label: '优化前总量', value: data.baseline, parts: [data.runtime, data.devProgram + data.metadata + data.dynamic, data.workspace + data.io] },
      { label: '优化后预估', value: data.baseline - saving, parts: [process, Math.max(0, operator), launch] },
      { label: '可解释分配', value: process + Math.max(0, operator) + launch, parts: [process, Math.max(0, operator), launch] },
    ];
    const max = Math.max(...rows.map((row) => row.value));
    $('#lifecycleBars').innerHTML = rows.map((row) => {
      const scale = row.value / max * 100;
      const sum = row.parts.reduce((a, b) => a + b, 0) || 1;
      const widths = row.parts.map((part) => part / sum * scale);
      return `<div class="mo-lifecycle-row"><span>${row.label}</span><div class="mo-bar-track"><i class="process" style="width:${widths[0]}%"></i><i class="operator" style="width:${widths[1]}%"></i><i class="launch" style="width:${widths[2]}%"></i></div><span>${fmt(row.value, 1)} MB</span></div>`;
    }).join('');
  }

  function ledgerRows() {
    const data = scenario();
    return ledgerBase.map((row) => {
      const next = { ...row };
      if (row.component === 'device_args') next.estimated = next.requested = state.kernel === 'bwd' ? 1.35 : 1.41;
      if (row.component === 'dev_ascend_program') next.estimated = next.requested = data.devProgram;
      if (row.component === 'dynamic_cell_match') next.estimated = next.requested = data.dynamic;
      if (row.component === 'workspace') next.estimated = next.requested = data.workspace;
      if (row.component === 'user_io') next.estimated = next.requested = data.io;
      if (row.component === 'torch_reserved') {
        next.estimated = state.kernel === 'bwd' ? 2826.95 : 898.64;
        next.requested = state.kernel === 'bwd' ? 3708 : 1040;
      }
      if (row.component === 'ctrlflow_cache' && state.mode === 'aclgraph') next.state = '需精确化';
      return next;
    });
  }

  function renderLedger() {
    const query = ($('#ledgerSearch').value || '').trim().toLowerCase();
    const lifecycle = $('#lifecycleFilter').value;
    const rows = ledgerRows().filter((row) => (lifecycle === 'all' || row.lifecycle === lifecycle) && `${row.component} ${row.source}`.toLowerCase().includes(query));
    $('#ledgerBody').innerHTML = rows.map((row) => {
      const delta = row.requested - row.estimated;
      const stateClass = row.state === '正常' ? 'success' : row.state === '未归因' || row.state === '固定' || row.state === '需精确化' ? 'warning' : '';
      return `<tr><td>${row.component}</td><td>${row.lifecycle}</td><td>${row.source}</td><td>${fmt(row.estimated)} MB</td><td>${fmt(row.requested)} MB</td><td>${delta ? `+${fmt(delta)}` : '0.00'} MB</td><td><span class="mo-state-text ${stateClass}">${row.state}</span></td></tr>`;
    }).join('');
  }

  function renderProfiling() {
    const adapted = state.profileMode === 'adapted';
    const coverage = adapted ? 100 : 58;
    $('#profileCoverage').textContent = `${coverage}%`;
    $('#coverageBar').style.width = `${coverage}%`;
    $('#treeCoverage').textContent = `${coverage}%`;
    $('#profileState').textContent = adapted ? '可归因' : '缺少标识';
    $('#profileState').className = `mo-state-text ${adapted ? 'success' : 'warning'}`;
    const fields = [
      ['Torch Workspace', true], ['Runtime 直接申请', adapted], ['Input / Output', adapted], ['进程级共享项', adapted], ['operator_id', adapted],
    ];
    $('#coverageList').innerHTML = fields.map(([label, ok]) => `<div class="mo-coverage-row"><span>${label}</span><span class="${ok ? 'ok' : 'missing'}">${ok ? 'COVERED' : 'MISSING'}</span></div>`).join('');
    $('#profilePreview').textContent = adapted
      ? `[PyPTO][gdr_bwd]\noperator_id = pypto:gdr_bwd:varlen64\nallocation_source = torch.empty\nlifecycle = launch\ncomponent = workspace\nbytes_requested = 544,935,936\nbytes_reserved = 3,888,087,040\ninput_bytes = 617,747,333\nshared_process_bytes = 821,978,726`
      : `[aten::empty]\noperator = unknown\nallocation_source = torch.empty\ncomponent = unknown\nbytes_requested = 544,935,936\n\n# Missing\nPyPTO operator identity\nRuntime direct allocations\nInput / Output bytes\nProcess-shared attribution`;
  }

  function renderAll() {
    renderMetrics();
    renderRecommendations();
    renderInspector();
    renderLifecycle();
    renderLedger();
    renderProfiling();
    requestAnimationFrame(renderChart);
  }

  let toastTimer = null;
  function toast(message) {
    const node = $('#toast');
    node.textContent = message;
    node.hidden = false;
    clearTimeout(toastTimer);
    toastTimer = setTimeout(() => { node.hidden = true; }, 2600);
  }

  function toggleRecommendation(id) {
    if (state.enabled.has(id)) state.enabled.delete(id); else state.enabled.add(id);
    state.selected = id;
    renderAll();
  }

  function bindEvents() {
    $('#kernelSelect').addEventListener('change', (event) => { state.kernel = event.target.value; $('#statusText').textContent = `${scenario().label} 基线已加载`; renderAll(); });
    $('#modeSelect').addEventListener('change', (event) => { state.mode = event.target.value; renderAll(); });
    $('#hardwareSelect').addEventListener('change', (event) => { state.hardware = event.target.value; toast(`${event.target.selectedOptions[0].textContent} Profile 已切换；metadata 参数等待真实校准`); renderAll(); });

    document.addEventListener('click', (event) => {
      const tab = event.target.closest('[data-tab], [data-open-tab]');
      if (tab) { setTab(tab.dataset.tab || tab.dataset.openTab); return; }
      const select = event.target.closest('[data-rec-select]');
      if (select) { state.selected = select.dataset.recSelect; renderRecommendations(); renderInspector(); return; }
      const inspectorToggle = event.target.closest('[data-inspector-toggle]');
      if (inspectorToggle) { toggleRecommendation(inspectorToggle.dataset.inspectorToggle); return; }
      const profile = event.target.closest('[data-profile-mode]');
      if (profile) {
        state.profileMode = profile.dataset.profileMode;
        $$('[data-profile-mode]').forEach((button) => button.classList.toggle('is-selected', button === profile));
        renderProfiling();
      }
    });

    $('#recommendationList').addEventListener('change', (event) => {
      const toggle = event.target.closest('[data-rec-toggle]');
      if (toggle) toggleRecommendation(toggle.dataset.recToggle);
    });
    $('#ledgerSearch').addEventListener('input', renderLedger);
    $('#lifecycleFilter').addEventListener('change', renderLedger);
    $('#runSimulation').addEventListener('click', () => {
      renderAll();
      $('#statusText').textContent = `模拟完成：确定性节省 ${fmt(selectedSaving(), 1)} MB`;
      toast(`预算模拟完成：${scenario().label} 预计从 ${(scenario().baseline / 1000).toFixed(2)} GB 降至 ${(projected() / 1000).toFixed(2)} GB`);
    });
    $('#exportReport').addEventListener('click', () => toast('UX Demo：预算报告已生成，包含估算公式、fallback_reason 与验收项'));
    $('#commandTrigger').addEventListener('click', () => { setTab('ledger'); $('#ledgerSearch').focus(); });
    $('#consoleAction').addEventListener('click', () => toast('[MEMLOG] 90 allocations · estimate delta 0 MB · 1 baseline discrepancy'));
    $('#resetDemo').addEventListener('click', () => {
      state.kernel = 'bwd'; state.mode = 'eager'; state.hardware = 'a3'; state.tab = 'overview'; state.selected = 'MEM-001'; state.enabled = new Set(['MEM-001', 'MEM-002']); state.profileMode = 'current';
      $('#kernelSelect').value = state.kernel; $('#modeSelect').value = state.mode; $('#hardwareSelect').value = state.hardware; $('#ledgerSearch').value = ''; $('#lifecycleFilter').value = 'all';
      $$('[data-profile-mode]').forEach((button) => button.classList.toggle('is-selected', button.dataset.profileMode === 'current'));
      setTab('overview'); renderAll(); toast('Demo 已重置为 GDR bwd / Eager / A3 基线');
    });
  }

  function init() {
    bindEvents();
    setTab(state.tab);
    renderAll();
    try { window.PtoIdeFrame?.initAll(); } catch (error) { console.warn('Shared IDE frame enhancement unavailable.', error); }
  }

  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', init, { once: true });
  else init();
})();
