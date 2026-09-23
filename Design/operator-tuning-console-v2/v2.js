/* V2 makes an Investigation the primary object. The original console stays
 * authoritative for raw evidence, views, and the experiment ledger. */
(() => {
  'use strict';
  const $ = (s, root = document) => root.querySelector(s);
  const $$ = (s, root = document) => Array.from(root.querySelectorAll(s));
  const esc = (v) => String(v == null ? '' : v).replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
  const byId = (run, id) => (run.findings || []).find((f) => f.id === id);
  const has = (run, id) => Boolean(byId(run, id));
  const state = { caseKey: '', taskId: '', section: 'overview' };

  function activeRun() {
    const key = $('[data-bind="caseChip"]') && $('[data-bind="caseChip"]').textContent.trim();
    const runs = window.TUNING_RUNS || {};
    return runs[key] || Object.values(runs).find((run) => run.case && (
      run.case.program === key || run.case.label === key || run.case.id === key
    )) || Object.values(runs)[0];
  }

  function taskData(run) {
    if (Array.isArray(run.investigations) && run.investigations.length) return run.investigations;
    const program = run.case && run.case.program || '当前程序';
    const f1 = byId(run, 'F1'), f2 = byId(run, 'F2'), f3 = byId(run, 'F3');
    const f4 = byId(run, 'F4'), f5 = byId(run, 'F5'), f6 = byId(run, 'F6');
    const f7 = byId(run, 'F7'), f8 = byId(run, 'F8'), f9 = byId(run, 'F9');
    const first = (run.findings || [])[0];
    const task = (id, title, status, target, findings, hypotheses, experiments) => ({
      id, title, status, target, findings: findings.filter(Boolean), hypotheses, experiments,
      baseline: '当前 run · ' + program, owner: '待分派',
    });
    const primary = f3 && f1
      ? task('INV-024', '解释 rank 间尾部延迟', '需要实验',
        '缩短关键路径上的通信等待，并验证是否改善端到端尾部。',
        ['F3', 'F1', has(run, 'F7') && 'F7', has(run, 'F6') && 'F6'],
        [
          { id: 'H-01', title: 'rank 启动错位放大集合通信等待', level: '强支持', claim: 'F3 的 launch skew 与 F1 的关键路径 wait 有同一条跨层时序锚点；它解释等待来源，但不排除其他成因。', evidence: ['F3', 'F1'], need: '在不改通信算法的前提下，缩小启动错位后 wait span 是否同步下降。' },
          { id: 'H-02', title: '调度队列可能是额外贡献因素', level: '待区分', claim: 'F7/F6 同时出现，只能作为竞争解释，不能升级为 H-01 的因果前提。', evidence: ['F7', 'F6'], need: '比较局部 dispatch 调整前后 ready>0 占比、complete 次数与 device wall。' },
        ],
        [{ id: 'EXP-024-01', status: '待执行', name: '只调整关键 rank 的启动 / dispatch 时序', change: '不改通信算法、Tile、融合边界', measures: 'rank start skew · wait span · device wall', guardrail: '结果校验、吞吐、host bind 时间' }])
      : f7 && f6
        ? task('INV-031', '解释 AIC 任务已就绪但未派发', '需要实验',
          '确认 ready queue 堵塞是否对 device wall 有可观测贡献。',
          ['F7', 'F6', has(run, 'F9') && 'F9'],
          [
            { id: 'H-01', title: '关键路径上的依赖 / dispatch 时机造成队列积压', level: '待验证', claim: 'ready queue 指标说明 AIC 任务可运行但未派发；不能单独证明调度器是根因。', evidence: ['F7'], need: '仅改变关键路径任务的 dispatch 时机，重测 queue 与 device wall。' },
            { id: 'H-02', title: 'AICPU complete 开销是竞争解释', level: '待区分', claim: 'complete 阶段的工作量可能延后派发，需要独立测量而不是与 queue 告警合并。', evidence: ['F6'], need: '记录 complete 次数和单位任务开销是否与等待窗口同向变化。' },
          ],
          [{ id: 'EXP-031-01', status: '待执行', name: '关键路径局部提前 dispatch', change: '不改变任务数量与 fusion 边界', measures: 'ready>0 · AIC util · device wall', guardrail: '吞吐与正确性同基线回归' }])
        : task('INV-001', '建立首个可证伪的性能假设', '待分诊',
          '用一个可回滚改动验证最高影响发现是否能改善端到端结果。',
          [first && first.id],
          [{ id: 'H-01', title: '最高优先级发现值得进一步验证', level: '待建模', claim: '当前只有同层观测，尚未形成跨层解释。', evidence: [first && first.id], need: '先绑定端到端指标和最小改动，再开始实验。' }],
          [{ id: 'EXP-001-01', status: '待规划', name: '定义最小单变量实验', change: '待选择', measures: '局部指标 · device wall · 正确性', guardrail: '保持同一基线与采样条件' }]);

    const backlog = [];
    if (f2 || f9) backlog.push(task('INV-025', '评估混合核的任务边界', '待分诊',
      '判断 hand-off 或块内串行是否值得以融合 / 解耦方式处理。',
      [f2 && 'F2', f9 && 'F9', f8 && 'F8'],
      [{ id: 'H-01', title: '任务边界引入可避免的 hand-off 或串行段', level: '待建模', claim: '先区分任务领取、依赖等待和核内串行，再选择 fusion 或 FIFO 方案。', evidence: [f2 && 'F2', f9 && 'F9'], need: '定位最小 scope 并做一个只改变边界的对照实验。' }],
      [{ id: 'EXP-025-01', status: '待规划', name: '选择一个 kernel scope 建立对照', change: '待确认', measures: '任务 span · hand-off · util', guardrail: '避免形成新的独占核' }]));
    if (f4 || f5) backlog.push(task('INV-026', '处理 Tile 与流水资源约束', '需要补证',
      '确认搬运粒度改动是否会触发流水深度回退。',
      [f4 && 'F4', f5 && 'F5'],
      [{ id: 'H-01', title: '粒度与流水深度受同一 L0 / UB 预算约束', level: '约束耦合', claim: 'F5 的修改可能触发 F4；它是 guardrail 关系，尚不是性能因果结论。', evidence: [f4 && 'F4', f5 && 'F5'], need: '在同一个 source scope 对比 Tile 预算、PH-MR-001 和 MTE 时间。' }],
      [{ id: 'EXP-026-01', status: '待规划', name: '同 scope 的 Tile 预算对照', change: '只调整末维或 pipeline depth 之一', measures: 'PH 提示 · MTE · L0/UB 预算', guardrail: '不接受深度回退换来的局部收益' }]));
    return [primary].concat(backlog);
  }

  function current() {
    const tasks = taskData(activeRun());
    return { tasks, task: tasks.find((item) => item.id === state.taskId) || tasks[0] };
  }

  function findingButton(id, run) {
    const item = byId(run, id);
    if (!item) return '';
    return '<button class="v2-ref" type="button" data-v2-evidence="' + esc(id) + '"><b>' + esc(id) + '</b><span>' + esc(item.level) + '</span><em>' + esc(item.metric) + '</em></button>';
  }

  function stateTone(status) {
    if (/需要实验/.test(status)) return 'action';
    if (/补证|分诊/.test(status)) return 'pending';
    return 'neutral';
  }

  function renderExplorer() {
    const root = $('#v2TaskExplorer');
    const data = current(), tasks = data.tasks, task = data.task;
    root.innerHTML = '<div class="v2-nav-head"><span>INVESTIGATIONS</span><button type="button" class="v2-icon-btn" title="按状态筛选">⌄</button></div><div class="v2-nav-filter"><span class="is-active">进行中 ' + tasks.filter((t) => t.status !== '待分诊').length + '</span><span>全部 ' + tasks.length + '</span></div><div class="v2-task-list">' + tasks.map((item) =>
      '<button class="v2-task-item ' + (item.id === task.id ? 'is-selected' : '') + '" type="button" data-v2-task="' + esc(item.id) + '"><span class="v2-task-item__top"><b>' + esc(item.id) + '</b><i class="v2-state v2-state--' + stateTone(item.status) + '">' + esc(item.status) + '</i></span><strong>' + esc(item.title) + '</strong><small>' + esc(item.findings.join(' · ') || '尚未绑定证据') + '</small></button>').join('') + '</div><div class="v2-nav-divider"></div><div class="v2-nav-meta"><span>当前 run</span><b>' + esc((activeRun().case || {}).program) + '</b><small>原始五层视图在“打开证据”后使用</small></div>';
    $$('[data-v2-task]', root).forEach((button) => button.addEventListener('click', () => { state.taskId = button.dataset.v2Task; state.section = 'overview'; renderProduct(); }));
  }

  function renderRail(task) {
    const root = $('#v2InvestigationRail');
    root.innerHTML = '<section class="v2-rail-section"><span class="v2-label">当前任务</span><h3>' + esc(task.id) + '</h3><p>' + esc(task.title) + '</p><div class="v2-rail-status"><i class="v2-state v2-state--' + stateTone(task.status) + '">' + esc(task.status) + '</i><span>' + esc(task.owner) + '</span></div></section><section class="v2-rail-section"><span class="v2-label">完成条件</span><p>一个假设经同基线实验被接受或驳回；端到端、守卫条件和正确性均已记录。</p></section><section class="v2-rail-section"><span class="v2-label">任务进度</span><ol class="v2-checklist"><li class="is-done">已建立基线</li><li class="is-done">已关联证据</li><li class="is-current">选择单变量实验</li><li>回归并更新结论</li></ol></section>';
  }

  function header(task) {
    return '<header class="v2-investigation-header"><div class="v2-breadcrumb">RUN / INVESTIGATIONS / ' + esc(task.id) + '</div><div class="v2-title-row"><div><h1>' + esc(task.title) + '</h1><p>' + esc(task.target) + '</p></div><div class="v2-title-actions"><i class="v2-state v2-state--' + stateTone(task.status) + '">' + esc(task.status) + '</i><button class="btn btn-secondary btn-sm" type="button" data-v2-section="evidence">查看关联证据</button></div></div><dl class="v2-facts"><div><dt>基线</dt><dd>' + esc(task.baseline) + '</dd></div><div><dt>负责人</dt><dd>' + esc(task.owner) + '</dd></div><div><dt>已绑定证据</dt><dd>' + task.findings.length + ' 项</dd></div><div><dt>实验</dt><dd>' + task.experiments.length + ' 个待办</dd></div></dl></header>';
  }

  function nav() {
    const tabs = [['overview', '概览'], ['hypotheses', '假设'], ['experiments', '实验'], ['evidence', '证据']];
    return '<nav class="v2-detail-tabs">' + tabs.map((item) => '<button type="button" data-v2-section="' + item[0] + '" class="' + (state.section === item[0] ? 'is-selected' : '') + '">' + item[1] + '</button>').join('') + '</nav>';
  }

  function overview(task, run) {
    const main = task.hypotheses[0], exp = task.experiments[0];
    return '<div class="v2-overview-grid"><section class="v2-panel v2-panel--decision"><span class="v2-label">现在要做的决定</span><h2>' + esc(exp.name) + '</h2><p>这是当前唯一建议执行的实验。其余发现保留为假设或待分诊任务，避免在一轮中同时改变多个变量。</p><dl class="v2-decision-spec"><div><dt>变更范围</dt><dd>' + esc(exp.change) + '</dd></div><div><dt>成功信号</dt><dd>' + esc(exp.measures) + '</dd></div><div><dt>守卫条件</dt><dd>' + esc(exp.guardrail) + '</dd></div></dl><button class="btn btn-primary btn-sm" type="button" data-v2-open-experiment="' + esc((main.evidence || [])[0] || '') + '">在原工作台创建实验</button></section><section class="v2-panel"><span class="v2-label">当前首要假设</span><h2>' + esc(main.id + ' · ' + main.title) + '</h2><p>' + esc(main.claim) + '</p><div class="v2-ref-list">' + main.evidence.map((id) => findingButton(id, run)).join('') + '</div><div class="v2-question"><b>还需证明</b><span>' + esc(main.need) + '</span></div></section></div><section class="v2-panel v2-panel--activity"><span class="v2-label">任务日志</span><ol><li><time>现在</time><div><b>等待选择实验</b><span>还没有实验结果；AI 结论保持为假设，不写入最终结论。</span></div></li><li><time>基线</time><div><b>已载入当前 run 的结构化证据</b><span>层级页面、Pass 轨迹与发现列表仍可作为原始证据透镜。</span></div></li></ol></section>';
  }

  function hypotheses(task, run) {
    return '<div class="v2-stack">' + task.hypotheses.map((item) => '<article class="v2-panel v2-hypothesis"><header><div><span class="v2-label">' + esc(item.id) + '</span><h2>' + esc(item.title) + '</h2></div><i class="v2-state v2-state--' + (item.level === '强支持' ? 'action' : 'pending') + '">' + esc(item.level) + '</i></header><dl><div><dt>主张</dt><dd>' + esc(item.claim) + '</dd></div><div><dt>证据</dt><dd class="v2-ref-list">' + item.evidence.map((id) => findingButton(id, run)).join('') + '</dd></div><div><dt>区分方式</dt><dd>' + esc(item.need) + '</dd></div></dl></article>').join('') + '</div>';
  }

  function experiments(task) {
    return '<section class="v2-panel v2-experiment-table"><header><div><span class="v2-label">实验计划</span><h2>一轮只验证一个假设</h2></div><button class="btn btn-secondary btn-sm" type="button" data-v2-open-experiment="' + esc((task.hypotheses[0].evidence || [])[0] || '') + '">打开原始实验台账</button></header><table><thead><tr><th>实验</th><th>状态</th><th>变更</th><th>验证指标</th><th>守卫条件</th></tr></thead><tbody>' + task.experiments.map((item) => '<tr><td><b>' + esc(item.id) + '</b><span>' + esc(item.name) + '</span></td><td><i class="v2-state v2-state--pending">' + esc(item.status) + '</i></td><td>' + esc(item.change) + '</td><td>' + esc(item.measures) + '</td><td>' + esc(item.guardrail) + '</td></tr>').join('') + '</tbody></table><p class="v2-table-note">执行入口仍使用原控制台的实验台账，避免 V2 与原版产生两套实验记录。</p></section>';
  }

  function evidence(task, run) {
    const items = task.findings.map((id) => byId(run, id)).filter(Boolean);
    return '<section class="v2-panel v2-evidence-table"><header><div><span class="v2-label">关联证据</span><h2>按任务相关性组织，而不是按层独立排队</h2></div><span>' + items.length + ' 项</span></header><div class="v2-evidence-rows">' + items.map((item) => '<button type="button" data-v2-evidence="' + esc(item.id) + '"><span class="v2-evidence-id">' + esc(item.id) + '</span><span class="v2-evidence-level">' + esc(item.level) + '</span><span><b>' + esc(item.title) + '</b><small>' + esc(item.metric) + '</small></span><span class="v2-evidence-open">打开原始视图 →</span></button>').join('') + '</div><p class="v2-table-note">打开后保留原来的 E2E、L2、L1、Compiler、ISA 视图、Inspector 与实验台账。</p></section>';
  }

  function renderCenter() {
    const run = activeRun(), task = current().task, host = $('#v2TaskCenter');
    const body = { overview, hypotheses, experiments, evidence }[state.section](task, run);
    host.innerHTML = '<div class="v2-investigation">' + header(task) + nav() + '<main class="v2-detail-body">' + body + '</main></div>';
    $$('[data-v2-section]', host).forEach((button) => button.addEventListener('click', () => { state.section = button.dataset.v2Section; renderProduct(); }));
    $$('[data-v2-evidence]', host).forEach((button) => button.addEventListener('click', () => openEvidence(button.dataset.v2Evidence)));
    $$('[data-v2-open-experiment]', host).forEach((button) => button.addEventListener('click', () => openEvidence(button.dataset.v2OpenExperiment, true)));
  }

  function ensureChrome() {
    const explorer = $('#explorerPane .pto-ide-frame__pane-body');
    if (explorer && !$('#v2TaskExplorer')) { const el = document.createElement('div'); el.id = 'v2TaskExplorer'; el.className = 'v2-task-explorer'; explorer.appendChild(el); }
    const inspector = $('#inspector');
    if (inspector && !$('#v2InvestigationRail')) { const el = document.createElement('div'); el.id = 'v2InvestigationRail'; el.className = 'v2-investigation-rail'; inspector.appendChild(el); }
  }

  function showProduct() {
    ensureChrome();
    const stage = $('#stage'), toolbar = $('#viewToolbar'), explorer = $('#explorerPane .pto-ide-frame__pane-body'), inspector = $('#inspector');
    if (!stage) return;
    stage.hidden = true; toolbar.hidden = true; $('#v2TaskCenter').hidden = false;
    explorer.classList.add('v2-product-active'); inspector.classList.add('v2-product-active');
    renderProduct();
    $$('#levelTabs .tab-control-item').forEach((tab) => tab.classList.toggle('is-selected', tab.id === 'v2TaskTab'));
  }

  function hideProduct() {
    const explorer = $('#explorerPane .pto-ide-frame__pane-body'), inspector = $('#inspector');
    $('#v2TaskCenter').hidden = true; $('#stage').hidden = false;
    if (explorer) explorer.classList.remove('v2-product-active');
    if (inspector) inspector.classList.remove('v2-product-active');
    const tab = $('#v2TaskTab'); if (tab) tab.classList.remove('is-selected');
  }

  function renderProduct() {
    ensureChrome();
    const data = current();
    state.taskId = data.task.id;
    renderExplorer(); renderRail(data.task); renderCenter();
  }

  function openEvidence(id, experiment) {
    if (!id) return;
    const target = $$('#findingList .tc-finding').find((button) => $('.id', button) && $('.id', button).textContent.trim() === id);
    if (target) target.click();
    hideProduct();
    if (experiment) requestAnimationFrame(() => {
      const button = $$('#inspector button').find((item) => /开始实验|新建实验/.test(item.textContent));
      if (button) button.focus();
    });
  }

  function addTab() {
    const host = $('#levelTabs');
    if (!host || $('#v2TaskTab')) return;
    const tab = document.createElement('button');
    tab.id = 'v2TaskTab'; tab.type = 'button'; tab.className = 'tab-control-item v2-task-tab'; tab.textContent = 'Investigations';
    tab.title = '按调优任务管理目标、假设、实验和证据'; tab.addEventListener('click', showProduct); host.prepend(tab);
  }

  function boot() {
    const stage = $('#stage'); if (!stage || $('#v2TaskCenter')) return;
    const host = document.createElement('div'); host.id = 'v2TaskCenter'; host.className = 'v2-task-center'; stage.before(host);
    addTab();
    const run = activeRun(); state.caseKey = run.case && run.case.id || ''; state.taskId = taskData(run)[0].id;
    showProduct();
    new MutationObserver(() => {
      addTab();
      const originalSelected = $('#levelTabs .tab-control-item.is-selected:not(#v2TaskTab)');
      if (originalSelected && !$('#v2TaskCenter').hidden) hideProduct();
    }).observe($('#levelTabs'), { childList: true, subtree: true, attributes: true, attributeFilter: ['class'] });
    const chip = $('[data-bind="caseChip"]');
    if (chip) new MutationObserver(() => {
      const next = activeRun(), key = next.case && next.case.id || '';
      if (key !== state.caseKey) { state.caseKey = key; state.taskId = taskData(next)[0].id; state.section = 'overview'; }
      if (!$('#v2TaskCenter').hidden) renderProduct();
    }).observe(chip, { childList: true, subtree: true, characterData: true });
  }
  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', boot); else boot();
})();
