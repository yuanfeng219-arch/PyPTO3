/* ==========================================================================
   PyPTO Autonomous Operator Workbench (v2.2 Canvas-First)
   Dark Default · Plain-Spoken Copy · Unified Contract & Budget Page ·
   Visual-Only Stall Expression · Canvas Marquee & Multi-Select Compare
   ========================================================================== */

(function () {
  'use strict';

  const D = window.PTO_V2_DATA;
  const $ = (id) => document.getElementById(id);

  function cloneContract(c) {
    return { ...c, certainty: { ...(c.certainty || {}) } };
  }

  // ---------- Reactive Application State ----------
  const state = {
    theme: 'dark',
    act: 4,
    activeView: 'canvas', // 'contract' | 'canvas' | 'diff' | 'gate'
    globalPaused: false,
    isPlaying: false,
    playbackTimer: null,

    // Active contract & draft on unified 目标与约束 page
    contract: cloneContract({ ...D.INITIAL_CONTRACT_V1, ...D.CONTRACT_V2_PATCH }),
    contractDraft: cloneContract({ ...D.INITIAL_CONTRACT_V1, ...D.CONTRACT_V2_PATCH }),
    contractEpochX: 7.45,
    contractHistory: [],
    nlPromptText: D.DEFAULT_NL_PROMPT,
    nlEditing: false,

    // Graph & Experiments
    lanes: [],
    runs: [],
    humanAnchors: [],
    reflexionRules: [],
    auditEvents: [],
    pendingRefutation: null,

    // Progressive Disclosure Floating Inspector & Canvas Controls
    inspectorOpen: false,
    selectedRunId: '#11',
    selectedLaneId: null,
    activeExceptionId: null,
    canvasScheme: 'landscape', // 'landscape' | 'bullseye' | 'swimlane'
    landscapeLens: 'perf',     // 'perf' | 'gen' | 'mem'
    landscapeAxisFilter: null, // null | 'incore' | 'memory' | 'dispatch' | 'human'
    bullseyeGateFilter: null,  // null | 0..5
    bullseyeFocusId: '#11',
    canvasFilter: 'all', // 'all' | 'valid' | 'alerts'
    canvasZoom: 'fit',   // 'fit' | 'wide'
    lensState: 'compact', // 'compact' | 'expanded' | 'collapsed'
    canvasAbortCtrl: null,

    // Marquee & Multi-Select Compare State
    multiSelectMode: false,
    selectedCompareIds: [],

    // Diff & Gate View States
    gateCandidateId: '#11',
    diffRunIds: ['#0', '#10', '#11'],
    openHunks: { 'hunk-dispatch': true, 'hunk-incore-fastpath': true, 'hunk-memory-precision': false },

    // Floating Bottom Agent Composer State
    threadOpen: false,
    composerCompact: false,
    attachedChips: [],
    threadMessages: [],

    nextSeq: 12,
  };

  // ---------- Scenario Snapshots (Act 0 ~ Act 4) ----------
  function loadActSnapshot(actNum, options = {}) {
    state.act = actNum;
    state.activeExceptionId = null;
    state.landscapeAxisFilter = null;
    state.bullseyeGateFilter = null;
    state.pendingRefutation = null;
    state.selectedLaneId = null;
    state.selectedCompareIds = [];
    if (!options.keepInspector) state.inspectorOpen = false;

    if (actNum <= 3) {
      state.contract = cloneContract(D.INITIAL_CONTRACT_V1);
      state.contractDraft = cloneContract(D.INITIAL_CONTRACT_V1);
      state.contractEpochX = null;
      state.contractHistory = [
        { version: 'v1', ts: '14:02:08', summary: '立项约束：Ascend 910B2 (24 CG)，主目标 ≥2.0×，小 Shape [1,128] 取默认 ≤10%。' },
      ];
    } else {
      const merged = { ...D.INITIAL_CONTRACT_V1, ...D.CONTRACT_V2_PATCH };
      state.contract = cloneContract(merged);
      state.contractDraft = cloneContract(merged);
      state.contractEpochX = 7.45;
      state.contractHistory = [
        { version: 'v1', ts: '14:02:08', summary: '立项约束：Ascend 910B2 (24 CG)，主目标 ≥2.0×，小 Shape [1,128] 取默认 ≤10%。' },
        { version: 'v2', ts: '14:19:42', summary: '收紧约束：[1,128] 退化 ≤2.0%、精度裕度 ≥20%、Workspace ≤64KB；#10 标记为 v2 失效。' },
      ];
    }

    state.lanes = D.BASE_LANES
      .map((l) => ({ ...l, status: l.statusByAct[actNum] || 'hidden' }))
      .filter((l) => l.status !== 'hidden');

    state.runs = D.BASE_RUNS
      .filter((r) => r.actIntro <= actNum || r.runningInAct === actNum)
      .map((r) => ({ ...r, isRunning: r.runningInAct === actNum && r.actIntro > actNum }));

    state.humanAnchors = [];
    if (actNum >= 2) {
      state.humanAnchors.push({
        id: 'ha-refute',
        x: 5.98,
        laneId: 'lane-dispatch',
        nearRunId: '#10',
        shortLabel: '反证转编排',
        detail: 'Perf Lane 实测 CG 0~11 与 CG 12~23 极差 2.5×，剪枝停滞的 InCore 路径并转向编排与派发。',
        ts: '14:14:20',
      });
    }
    if (actNum >= 3) {
      state.humanAnchors.push({
        id: 'ha-fork',
        x: 7.98,
        laneId: 'lane-fork',
        nearRunId: '#11',
        shortLabel: 'Fork 快路径',
        detail: '保留 #10 错峰分核，为 t_dim<=4 派生直通快路径并压缩 Workspace 至 32KB。',
        ts: '14:20:05',
      });
    }

    state.reflexionRules = [
      { id: 'rx-1', source: '失败回退 (#5)', rule: '三级预取导致 Vec (UB) 越界 +42KB，禁止同时驻留三份 FP32 得分 Tile。' },
      { id: 'rx-2', source: '失败回退 (#4)', rule: 'Acc 累加与 inverse-RoPE 禁止提前转 BF16，必须保持 FP32 后 rint 舍入。' },
    ];
    if (actNum >= 2) {
      state.reflexionRules.unshift({
        id: 'rx-3',
        source: '手动反证',
        rule: '24 核组极差 >1.5× 时，停止在 InCore 微调预取，优先排查 pl.spmd 分核同余碰撞。',
      });
    }
    if (actNum >= 4) {
      state.reflexionRules.unshift({
        id: 'rx-4',
        source: '约束变更 (v2)',
        rule: '引入 qk_order 轮转表时，必须为 t_dim<=4 保留直通快路径，避免小 Shape 退化。',
      });
    }

    state.auditEvents = [
      { ts: '14:02:08', kind: 'contract', title: '约束 v1 立项生效', detail: '锁定 Ascend 910B2 指纹 910b2-cg24-a5c8e41d，预算上限 30 轮。' },
      { ts: '14:03:12', kind: 'system', title: '#0 基线实测完成', detail: '[32,4096] 482.0μs，Roofline 34.2%，24 核组极差 2.50×。' },
    ];
    if (actNum >= 1) {
      state.auditEvents.push(
        { ts: '14:06:44', kind: 'success', title: '#3 成为当前最优 (398.0μs)', detail: 'InCore 双缓冲预取降延迟 -17.4%。' },
        { ts: '14:11:09', kind: 'system', title: '连续 3 轮集中在 InCore 且无新收益', detail: '#6→#8→#9 耗时 395→396→393μs，核组极差仍 2.46×；#4/#5 失败回退。' }
      );
    }
    if (actNum >= 2) {
      state.auditEvents.push(
        { ts: '14:14:20', kind: 'human', title: '◆ 手动反证：剪枝 InCore 并转编排派发', detail: 'Perf Lane 框选 12慢/12快双峰，Boost 2× 启动编排与分核派发。' }
      );
    }
    if (actNum >= 3) {
      state.auditEvents.push(
        { ts: '14:17:52', kind: 'contract', title: '#10 存在风险挂起 (206.1μs)', detail: '主 Shape -57.2%，但 [1,128] 退化 +5.5%、精度裕度 17%。' }
      );
    }
    if (actNum >= 4) {
      state.auditEvents.push(
        { ts: '14:19:42', kind: 'contract', title: '约束变更 v1 → v2 生效', detail: '锁定 [1,128] ≤2.0% 与裕度 ≥20%，#10 标记为 v2 失效。' },
        { ts: '14:20:05', kind: 'human', title: '◆ 从 #10 Fork 小 Shape 快路径', detail: '保留错峰分核，压缩 Workspace 至 32KB。' },
        { ts: '14:23:18', kind: 'success', title: '#11 全门禁通过 (214.0μs)', detail: '6/6 门禁全通过，[1,128] -1.8%，精度裕度 74%。' }
      );
    }

    state.threadMessages = [
      { role: 'system', ts: '14:02', text: `已锁定 ${state.contract.chip} · 指纹 ${state.contract.fingerprint} · 约束 ${state.contract.version}` },
    ];
    if (actNum === 0) {
      state.threadMessages.push({ role: 'agent', ts: '14:02', text: '#0 基线 482.0μs（核组极差 2.50×）。可在「目标与约束」确认推断项后启动 3 路并行探索。' });
    } else if (actNum === 1 || actNum === 2) {
      state.threadMessages.push(
        { role: 'agent', ts: '14:11', text: 'InCore 路径连续 3 轮耗时 395→396→393μs 几乎无变化，24 核组极差仍 2.46×。' },
        { role: 'agent', ts: '14:12', text: '建议在 Perf Lane 核查双峰分布，或直接输入「剪枝」释放槽位转向编排与派发。' }
      );
    } else if (actNum === 3) {
      state.threadMessages.push(
        { role: 'agent', ts: '14:18', text: '编排分核 #10 将 [32,4096] 降至 206.1μs (-57.2%)，但 [1,128] 退化 +5.5%、精度裕度 17%，已挂起。' },
        { role: 'agent', ts: '14:18', text: '可前往「目标与约束」收紧小 Shape 约束，或输入「Fork」从 #10 派生快路径修复。' }
      );
    } else {
      state.threadMessages.push(
        { role: 'agent', ts: '14:23', text: `#11 历史记录符合 ${state.contract.version} 全部 6 项门禁：[32,4096] 214.0μs (2.25×)，[1,128] -1.8%，Workspace 32KB。可前往验收门禁核查。` }
      );
    }

    if (actNum === 0) {
      state.selectedRunId = '#0';
      state.bullseyeFocusId = '#0';
      state.gateCandidateId = '#0';
    } else if (actNum === 1 || actNum === 2) {
      state.selectedRunId = '#9';
      state.bullseyeFocusId = '#9';
      state.gateCandidateId = '#3';
    } else if (actNum === 3) {
      state.selectedRunId = '#10';
      state.bullseyeFocusId = '#10';
      state.gateCandidateId = '#10';
      state.diffRunIds = ['#0', '#3', '#10'];
    } else {
      state.selectedRunId = '#11';
      state.bullseyeFocusId = '#11';
      state.gateCandidateId = '#11';
      state.diffRunIds = ['#0', '#10', '#11'];
    }

    if (!options.keepView) {
      state.activeView = options.defaultView || 'canvas';
    }
  }

  // ---------- Computed Helpers ----------
  function getIncumbentBestRun() {
    const completed = state.runs.filter((r) => !r.isRunning && r.buildOk && r.testsOk);
    let best = completed.find((r) => r.id === '#0') || completed[0];
    completed.forEach((r) => {
      const ev = D.evaluateRun(r, state.contract);
      if (ev.overall === 'pass' || ev.overall === 'progress') {
        if (r.p0Latency < best.p0Latency) best = r;
      }
    });
    return best;
  }

  function getActivePathCount() {
    if (state.globalPaused) return 0;
    return state.lanes.filter((l) => ['active', 'running', 'plateau'].includes(l.status)).length;
  }

  function getRunsUsedCount() { return state.runs.length; }


  function getSpotlightRun() {
    return state.runs.find((r) => r.id === state.gateCandidateId && !r.isRunning) || getIncumbentBestRun();
  }

  function getTopGateDims() {
    const ref = getSpotlightRun();
    if (!ref || ref.id === '#0') {
      return { hardware: 'pass', correctness: 'pass', precision: 'pass', generalization: 'pending', performance: 'pending', engineering: 'pass' };
    }
    return D.evaluateRun(ref, state.contract).dims;
  }

  function navigateToContract(focusGroup) {
    // Structurally dismiss node popover so zero overlap is possible
    state.inspectorOpen = false;
    state.activeView = 'contract';
    state.contractDraft = cloneContract(state.contract);
    renderAll();
    if (focusGroup) {
      const card = document.querySelector(`[data-constraint-group="${focusGroup}"]`);
      if (card) card.scrollIntoView({ behavior: 'smooth', block: 'center' });
    }
  }

  function showToast(msg) {
    const box = $('toastContainer');
    box.innerHTML = '';
    const el = document.createElement('div');
    el.className = 'toast';
    el.textContent = msg;
    box.appendChild(el);
    clearTimeout(box._timer);
    box._timer = setTimeout(() => {
      el.style.opacity = '0';
      el.style.transition = 'opacity 180ms ease';
      setTimeout(() => { if (el.parentNode) el.remove(); }, 200);
    }, 2200);
  }

  // ---------- Render: Single Top Rail & Shell ----------
  function renderShell() {
    document.documentElement.dataset.theme = state.theme;
    const canvas = state.activeView === 'canvas';
    $('canvasMain').hidden = !canvas;
    $('bullseyeGateBar').hidden = !canvas || state.canvasScheme !== 'bullseye';
    $('bullseyeCornerHud').hidden = !canvas || state.canvasScheme !== 'bullseye';
    $('structuredMain').hidden = canvas;
    $('editorialIdentity').hidden = !canvas;
    $('subviewIdentity').hidden = canvas;
    ['contract','diff','gate'].forEach(v => { $('view-'+v).hidden = state.activeView !== v; });
    document.querySelectorAll('[data-view]').forEach(b => {
      b.classList.toggle('is-active',b.dataset.view === state.activeView);
      b.setAttribute('aria-current',b.dataset.view === state.activeView ? 'page' : 'false');
    });
    document.querySelectorAll('[data-canvas-scheme]').forEach(b => {
      b.hidden = !canvas;
      b.classList.toggle('is-active', b.dataset.canvasScheme === state.canvasScheme);
      b.setAttribute('aria-pressed', String(b.dataset.canvasScheme === state.canvasScheme));
    });
    ['canvasSchemeSwitcher','canvasCtxSep','canvasCtxDropdown','canvasLegend','zoomControls'].forEach(id => $(id).hidden = !canvas);
    $('themeToggleBtn').setAttribute('aria-label',state.theme === 'dark' ? '切换为浅色主题' : '切换为深色主题');
    const best = getIncumbentBestRun();
    $('edScopeLine').textContent = `PYPTO · [32, 4096] · 910B2 · FP16/BF16`;
    $('edFocalLatency').textContent = best ? best.p0Latency.toFixed(1)+' μs' : '待采样';
    $('edFocalSpeedup').textContent = best ? `${(482/best.p0Latency).toFixed(2)}× · 基线 482.0 μs` : '基线待确认';
    $('edFocalClock').textContent = `device_wall_us · ${state.contract.version} · ${getRunsUsedCount()}/${state.contract.maxRuns} 次实验`;
    $('subviewContractBadge').textContent = state.contract.version;
    const dims = getTopGateDims();
    $('topGateDots').innerHTML = D.GATE_KEYS.map((key,i) => `<button class="gate-dot-btn" data-goto-group="${key}" data-status="${dims[key]}" data-tip="${D.GATE_LABELS[i]}：${dims[key] === 'pass' ? '通过' : dims[key] === 'fail' ? '未通过' : '待确认'}"><span class="gate-dot-core"></span></button>`).join('')+`<span class="mono">${Object.values(dims).filter(v=>v==='pass').length}/6 门禁</span>`;
    $('badgeGateDot').className = 'tab-status-dot';
    $('seqTrack').innerHTML = ['立项','局部探索','核查','收紧','验收'].map((v,i)=>`<button class="seq-step-btn ${state.act===i?'is-active':''}" data-act="${i}" data-tip="回放 Act ${i} · ${v}">${i}</button>`).join('');
    $('playbackBtn').setAttribute('aria-label', state.isPlaying?'暂停故事回放':'播放故事回放');
    $('playbackBtn').innerHTML = state.isPlaying ? '<svg viewBox="0 0 16 16"><path d="M5 3v10M11 3v10"/></svg>' : '<svg viewBox="0 0 16 16"><path d="M5 3.5 12.5 8 5 12.5Z" fill="currentColor"/></svg>';
    $('stepPrevBtn').disabled = state.act === 0;
    $('stepNextBtn').disabled = state.act === 4;
    $('multiSelectToggleBtn').setAttribute('aria-pressed',String(state.multiSelectMode));
    $('auditBadge').textContent = String(D.computeDiagnostics(state.runs,state.lanes,state.contract,state.act).exceptions.length);
    renderCompareFloatBar(); renderComposerDock();
  }


  function buildHighlightedNlHtml(text) {
    let html = escapeHtml(text);
    D.NL_PHRASES.forEach((ph) => {
      const esc = escapeHtml(ph.text);
      if (html.includes(esc)) {
        html = html.replace(esc, `<span class="nl-phrase" data-nl-key="${ph.key}">${esc}</span>`);
      }
    });
    return html;
  }

  function renderContractView() {
    const container = $('view-contract');
    const draft = state.contractDraft;
    const cert = draft.certainty;

    const certVals = Object.values(cert);
    const explicitCount = certVals.filter((v) => v === 'explicit').length;
    const uncertainCount = certVals.length - explicitCount;

    const dims = getTopGateDims();
    const passCount = Object.values(dims).filter((v) => v === 'pass').length;
    const alertCount = Object.values(dims).filter((v) => v === 'fail' || v === 'warn').length;

    const hasDraftDiff =
      draft.maxRuns !== state.contract.maxRuns ||
      draft.maxWorkers !== state.contract.maxWorkers ||
      draft.plateauThresholdRounds !== state.contract.plateauThresholdRounds ||
      draft.p0TargetSpeedup !== state.contract.p0TargetSpeedup ||
      draft.p1MaxRegressionPct !== state.contract.p1MaxRegressionPct ||
      draft.minPrecisionMarginPct !== state.contract.minPrecisionMarginPct ||
      draft.maxWorkspaceKB !== state.contract.maxWorkspaceKB ||
      draft.p1GuardrailEnabled !== state.contract.p1GuardrailEnabled ||
      JSON.stringify(draft.certainty) !== JSON.stringify(state.contract.certainty);

    // Deterministic impact preview
    const completed = state.runs.filter((r) => !r.isRunning && r.buildOk && r.testsOk && r.id !== '#0');
    const newlyInvalid = [];
    let compliantBest = null;
    completed.forEach((r) => {
      const prevEv = D.evaluateRun(r, state.contract);
      const nextEv = D.evaluateRun(r, draft);
      if (nextEv.overall === 'invalid' && prevEv.overall !== 'invalid') newlyInvalid.push(r.id);
      if (nextEv.overall === 'pass' || nextEv.overall === 'progress') {
        if (!compliantBest || r.p0Latency < compliantBest.p0Latency) compliantBest = r;
      }
    });

    const nextVer = `v${state.contract.epochNum + 1}`;
    const impactText = newlyInvalid.length
      ? `应用 <strong class="mono">${nextVer}</strong> 后，历史实验 <strong class="mono" style="color:var(--danger);">${newlyInvalid.join('、')}</strong> 将标为 ${nextVer} 失效（历史读数保留）；合规最优切至 <strong class="mono" style="color:var(--success);">${compliantBest ? `${compliantBest.id} (${compliantBest.p0Latency.toFixed(1)}μs)` : '#0 基线'}</strong>。`
      : hasDraftDiff
      ? `应用 <strong class="mono">${nextVer}</strong> 后，画布将插入竖向约束变更线，并按新阈值重评全部历史实验（当前合规最优 <strong class="mono" style="color:var(--success);">${compliantBest ? `${compliantBest.id} (${compliantBest.p0Latency.toFixed(1)}μs)` : '待探索'}</strong>）。`
      : `当前生效版本 <strong class="mono">${state.contract.version}</strong> · 点击任意虚线项的「确认」或调整滑块即可预览重评影响。`;

    function badgeFor(cState) {
      if (cState === 'explicit') return `<span class="certainty-badge is-explicit">● 已指定</span>`;
      if (cState === 'inferred') return `<span class="certainty-badge is-inferred">◇ Agent 推断</span>`;
      return `<span class="certainty-badge is-default">◌ 默认值</span>`;
    }

    function confirmBtn(key, cState) {
      if (cState === 'explicit') return '';
      return `<button class="btn btn-xs btn-warning-outline" type="button" data-confirm-rule="${key}">确认</button>`;
    }

    function ruleClass(cState) {
      return cState === 'explicit' ? 'is-explicit' : 'is-uncertain';
    }

    container.innerHTML = `
      <!-- 1. Top Overview & Inline Live Budget Strip -->
      <div class="contract-overview-strip">
        <div class="contract-overview-left">
          <span class="pill pill-primary mono" style="height:24px;font-size:12px;">${state.contract.version}</span>
          <span class="pill pill-neutral scenario-pill" data-nl-target="scenario" title="当前优化任务场景">场景：Decode 稀疏注意力 <span class="mono">[32,4096]</span></span>
          <span class="certainty-badge is-explicit">● 已指定 ${explicitCount} 项</span>
          <span class="certainty-badge is-inferred">◇ 推断/默认 ${uncertainCount} 项</span>
          <span class="pill ${alertCount === 0 && passCount === 6 ? 'pill-success' : 'pill-warning'} mono">${passCount}/6 通过${alertCount ? ` · ${alertCount} 异常` : ''}</span>
          <button class="btn btn-xs btn-ghost" type="button" id="openAuditFromContractBtn">变更记录 (${state.contractHistory.length})</button>
        </div>

        <div class="inline-budget-controls" aria-label="上板预算与自动暂停策略">
          <label class="inline-slider-group">
            <span>上板上限</span>
            <input type="range" id="inlineMaxRuns" min="16" max="60" step="2" value="${draft.maxRuns}" />
            <strong id="inlineMaxRunsVal">${draft.maxRuns} 轮</strong>
          </label>
          <label class="inline-slider-group">
            <span>并发路径</span>
            <input type="range" id="inlineMaxWorkers" min="1" max="6" step="1" value="${draft.maxWorkers}" />
            <strong id="inlineMaxWorkersVal">${draft.maxWorkers} 路</strong>
          </label>
          <label class="inline-slider-group">
            <span>无收益暂停</span>
            <input type="range" id="inlinePlateau" min="2" max="6" step="1" value="${draft.plateauThresholdRounds}" />
            <strong id="inlinePlateauVal">连 ${draft.plateauThresholdRounds} 轮</strong>
          </label>
          <button class="btn btn-xs ${state.globalPaused ? 'btn-primary' : ''}" type="button" id="inlinePauseToggleBtn">${state.globalPaused ? '恢复调度' : '暂停调度'}</button>
        </div>
      </div>

      <!-- 2. Natural Language Input -> Structured Constraints Mapping -->
      <div class="nl-source-panel">
        <div class="nl-source-head">
          <div style="display:flex;align-items:center;gap:10px;min-width:0;">
            <strong style="font-size:12.5px;">用户自然语言输入</strong>
            <span class="nl-source-status" id="nlLinkStatus">悬停下方任意规则，可溯源至输入短语或查看 Agent 推断来源</span>
          </div>
          <div style="display:flex;align-items:center;gap:6px;">
            ${state.nlEditing
              ? `<button class="btn btn-xs btn-primary" type="button" id="reparseNlBtn">重新解析</button><button class="btn btn-xs btn-ghost" type="button" id="cancelEditNlBtn">取消</button>`
              : `<button class="btn btn-xs" type="button" id="editNlBtn">编辑输入</button>`}
          </div>
        </div>
        ${state.nlEditing
          ? `<textarea class="nl-edit-textarea" id="nlEditTextarea" aria-label="编辑用户自然语言输入">${escapeHtml(state.nlPromptText)}</textarea>`
          : `<div class="nl-prompt-display" id="nlPromptDisplay">${buildHighlightedNlHtml(state.nlPromptText)}</div>`}
      </div>

      <!-- 3. All 6 Constraint Groups in Compact 3x2 Grid (Solid Explicit vs Dashed Inferred/Default) -->
      <div class="constraints-grid-6">
        <!-- Group 1: Hardware -->
        <div class="constraint-card ${ruleClass(cert.hardware)}" data-constraint-group="hardware">
          <div class="constraint-card-head">
            <span class="constraint-card-title">1. 硬件</span>
            ${badgeFor(cert.hardware)}
          </div>
          <div class="rule-row ${ruleClass(cert.hardware)}" data-nl-target="hardware">
            <div class="rule-row-top">
              <span class="rule-label">目标芯片与核组</span>
              ${confirmBtn('hardware', cert.hardware)}
            </div>
            <div class="rule-val mono">${draft.chip} · ${draft.topology}</div>
            <div style="font-size:11px;color:var(--foreground-muted);" class="mono">声明指纹 ${draft.fingerprint} · 校验 Vec/Mat 容量</div>
          </div>
        </div>

        <!-- Group 2: Correctness -->
        <div class="constraint-card ${ruleClass(cert.correctness)}" data-constraint-group="correctness">
          <div class="constraint-card-head">
            <span class="constraint-card-title">2. 正确性</span>
            ${badgeFor(cert.correctness)}
          </div>
          <div class="rule-row ${ruleClass(cert.correctness)}" data-nl-target="correctness">
            <div class="rule-row-top">
              <span class="rule-label">Golden 对拍全通过</span>
              <span class="certainty-badge is-explicit">● 已指定</span>
            </div>
            <div class="rule-val mono">${draft.goldenRef}</div>
          </div>
          <div class="rule-row ${ruleClass(cert.correctness)}" data-nl-target="fallback">
            <div class="rule-row-top">
              <span class="rule-label">边界与尾块保护</span>
              <span class="certainty-badge is-explicit">● 已指定</span>
            </div>
            <div class="rule-val">空输入 / 尾块 / 禁止隐式 Fallback</div>
          </div>
        </div>

        <!-- Group 3: Precision (Mixed Explicit + Inferred Margin) -->
        <div class="constraint-card ${ruleClass(cert.precisionMargin)}" data-constraint-group="precision">
          <div class="constraint-card-head">
            <span class="constraint-card-title">3. 精度</span>
            ${badgeFor(cert.precisionMargin)}
          </div>
          <div class="rule-row is-explicit" data-nl-target="precision">
            <div class="rule-row-top">
              <span class="rule-label">类型链与误差</span>
              <span class="certainty-badge is-explicit">● 已指定</span>
            </div>
            <div class="rule-val mono">BF16→FP32→BF16 · rtol 1/128</div>
          </div>
          <div class="rule-row ${ruleClass(cert.precisionMargin)}" data-nl-target="inferred">
            <div class="rule-row-top">
              <span class="rule-label">极端值精度裕度（防溢出/漂移）</span>
              <div style="display:flex;align-items:center;gap:5px;">
                ${badgeFor(cert.precisionMargin)}
                ${confirmBtn('precisionMargin', cert.precisionMargin)}
              </div>
            </div>
            <div class="rule-slider-row">
              <input type="range" id="draftPrecMargin" min="10" max="40" step="2" value="${draft.minPrecisionMarginPct}" />
              <span class="rule-val mono">≥ ${draft.minPrecisionMarginPct}%</span>
            </div>
          </div>
        </div>

        <!-- Group 4: Generalization (Default/Uncertain in v1 -> Solid Confirmed in v2!) -->
        <div class="constraint-card ${ruleClass(cert.p1Regression)}" data-constraint-group="generalization">
          <div class="constraint-card-head">
            <span class="constraint-card-title">4. 泛化</span>
            ${badgeFor(cert.p1Regression)}
          </div>
          <div class="rule-row ${ruleClass(cert.p1Regression)}" data-nl-target="default">
            <div class="rule-row-top">
              <span class="rule-label">非主 Shape 防回退</span>
              <div style="display:flex;align-items:center;gap:5px;">
                ${badgeFor(cert.p1Regression)}
                ${confirmBtn('p1Regression', cert.p1Regression)}
              </div>
            </div>
            <div class="rule-slider-row">
              <input type="range" id="draftP1Reg" min="0.5" max="10" step="0.5" value="${draft.p1MaxRegressionPct}" />
              <span class="rule-val mono">[1,128] ≤ ${draft.p1MaxRegressionPct.toFixed(1)}%</span>
            </div>
          </div>
          <div class="rule-row ${ruleClass(cert.sparseBucket)}" data-nl-target="default">
            <div class="rule-row-top">
              <span class="rule-label">多稀疏度分布验证</span>
              <div style="display:flex;align-items:center;gap:5px;">
                ${badgeFor(cert.sparseBucket)}
                ${confirmBtn('sparseBucket', cert.sparseBucket)}
              </div>
            </div>
            <div class="rule-val">${draft.sparseCoverageMode}</div>
          </div>
        </div>

        <!-- Group 5: Performance -->
        <div class="constraint-card ${ruleClass(cert.workspace)}" data-constraint-group="performance">
          <div class="constraint-card-head">
            <span class="constraint-card-title">5. 性能</span>
            ${badgeFor(cert.workspace)}
          </div>
          <div class="rule-row is-explicit" data-nl-target="perf">
            <div class="rule-row-top">
              <span class="rule-label">目标加速比</span>
              <span class="certainty-badge is-explicit">● 已指定</span>
            </div>
            <div class="rule-slider-row">
              <input type="range" id="draftSpeedup" min="1.5" max="2.5" step="0.1" value="${draft.p0TargetSpeedup}" />
              <span class="rule-val mono">≥ ${draft.p0TargetSpeedup.toFixed(1)}× / ≤${Math.round(482 / draft.p0TargetSpeedup)}μs</span>
            </div>
          </div>
          <div class="rule-row ${ruleClass(cert.workspace)}" data-nl-target="inferred">
            <div class="rule-row-top">
              <span class="rule-label">额外显存上限</span>
              <div style="display:flex;align-items:center;gap:5px;">
                ${badgeFor(cert.workspace)}
                ${confirmBtn('workspace', cert.workspace)}
              </div>
            </div>
            <div class="rule-slider-row">
              <input type="range" id="draftWorkspace" min="32" max="128" step="16" value="${draft.maxWorkspaceKB}" />
              <span class="rule-val mono">Workspace ≤${draft.maxWorkspaceKB}KB</span>
            </div>
          </div>
        </div>

        <!-- Group 6: Standards & Merge Rules -->
        <div class="constraint-card ${ruleClass(cert.topology)}" data-constraint-group="engineering">
          <div class="constraint-card-head">
            <span class="constraint-card-title">6. 规范</span>
            ${badgeFor(cert.topology)}
          </div>
          <div class="rule-row ${ruleClass(cert.topology)}" data-nl-target="inferred">
            <div class="rule-row-top">
              <span class="rule-label">接口与合入规范</span>
              <div style="display:flex;align-items:center;gap:5px;">
                ${badgeFor(cert.topology)}
                ${confirmBtn('topology', cert.topology)}
              </div>
            </div>
            <div class="rule-val">不破对外接口 · 仅用合规 PyPTO API · Lint 零警告</div>
            <div style="font-size:11px;color:var(--foreground-muted);">允许编排派发 / InCore / 内存与 Ring 调优</div>
          </div>
        </div>
      </div>

      <!-- Apply & Impact Preview Bar -->
      <div class="contract-apply-bar ${hasDraftDiff ? 'has-changes' : ''}">
        <div style="font-size:12px;color:var(--foreground-secondary);flex:1;min-width:260px;">${impactText}</div>
        <div style="display:flex;align-items:center;gap:8px;flex-shrink:0;">
          ${state.contract.epochNum < 2 ? `<button class="btn btn-sm btn-warning-outline" type="button" id="presetStrictV2Btn">一键套用推荐 v2 严格约束</button>` : ''}
          <button class="btn btn-sm btn-primary" type="button" id="applyContractPageBtn" ${hasDraftDiff ? '' : 'disabled'}>应用变更 (${state.contract.version} → ${nextVer})</button>
        </div>
      </div>
    `;

    bindContractViewEvents();
  }

  function bindContractViewEvents() {
    // Budget and constraints commit together through the versioned apply action.
    [['inlineMaxRuns','maxRuns'],['inlineMaxWorkers','maxWorkers'],['inlinePlateau','plateauThresholdRounds']].forEach(([id,key])=>{
      $(id).addEventListener('input',e=>{state.contractDraft[key]=Number(e.target.value);renderContractView();});
    });
    $('inlinePauseToggleBtn').addEventListener('click', () => {
      state.globalPaused = !state.globalPaused;
      showToast(state.globalPaused ? '已暂停自主调度' : '已恢复自主调度');
      renderAll();
    });
    $('openAuditFromContractBtn').addEventListener('click', openAuditDrawer);

    // NL Edit & Re-parse
    const editBtn = $('editNlBtn');
    if (editBtn) editBtn.addEventListener('click', () => { state.nlEditing = true; renderContractView(); });
    const cancelBtn = $('cancelEditNlBtn');
    if (cancelBtn) cancelBtn.addEventListener('click', () => { state.nlEditing = false; renderContractView(); });
    const reparseBtn = $('reparseNlBtn');
    if (reparseBtn) {
      reparseBtn.addEventListener('click', () => {
        const val = $('nlEditTextarea').value.trim();
        if (val) state.nlPromptText = val;
        // Detect if user added small-shape or workspace constraint in NL
        if (val.includes('128') || val.includes('小 Shape') || val.includes('2%')) {
          state.contractDraft.p1MaxRegressionPct = 2.0;
          state.contractDraft.p1GuardrailEnabled = true;
          state.contractDraft.certainty.p1Regression = 'explicit';
        }
        state.nlEditing = false;
        showToast('已重新解析自然语言输入并更新约束映射');
        renderContractView();
      });
    }

    // Bidirectional Hover Linkage: Rule -> NL highlight or "未在输入中提及"
    const statusEl = $('nlLinkStatus');
    document.querySelectorAll('#view-contract [data-nl-target]').forEach((row) => {
      row.addEventListener('mouseenter', () => {
        const key = row.dataset.nlTarget;
        document.querySelectorAll('#view-contract .nl-phrase').forEach((sp) => {
          sp.classList.toggle('is-active', sp.dataset.nlKey === key);
        });
        if (!statusEl) return;
        if (key === 'inferred' || key === 'default') {
          statusEl.textContent = '未在输入中提及 · 由 Agent 推断 / 取默认值（点击「确认」可转为已确认）';
          statusEl.className = 'nl-source-status is-inferred-hint';
        } else {
          statusEl.textContent = '已溯源至上方高亮的自然语言输入短语';
          statusEl.className = 'nl-source-status is-linked-hint';
        }
      });
      row.addEventListener('mouseleave', () => {
        document.querySelectorAll('#view-contract .nl-phrase').forEach((sp) => sp.classList.remove('is-active'));
        if (statusEl) {
          statusEl.textContent = '悬停下方任意规则，可溯源至输入短语或查看 Agent 推断来源';
          statusEl.className = 'nl-source-status';
        }
      });
    });

    // One-click [确认] converts dashed inferred/default -> solid explicit!
    document.querySelectorAll('#view-contract [data-confirm-rule]').forEach((btn) => {
      btn.addEventListener('click', () => {
        const key = btn.dataset.confirmRule;
        state.contractDraft.certainty[key] = 'explicit';
        state.contract.certainty[key] = 'explicit';
        showToast('已确认该约束，转为明确规则');
        renderContractView();
      });
    });

    // Inline Rule Sliders
    $('draftPrecMargin').addEventListener('input', (e) => {
      state.contractDraft.minPrecisionMarginPct = Number(e.target.value);
      state.contractDraft.certainty.precisionMargin = 'explicit';
      renderContractView();
    });
    $('draftP1Reg').addEventListener('input', (e) => {
      state.contractDraft.p1MaxRegressionPct = Number(e.target.value);
      state.contractDraft.p1GuardrailEnabled = state.contractDraft.p1MaxRegressionPct <= 5.0;
      state.contractDraft.certainty.p1Regression = 'explicit';
      renderContractView();
    });
    $('draftSpeedup').addEventListener('input', (e) => {
      state.contractDraft.p0TargetSpeedup = Number(e.target.value);
      state.contractDraft.p0TargetLatencyUs = 482 / state.contractDraft.p0TargetSpeedup;
      renderContractView();
    });
    $('draftWorkspace').addEventListener('input', (e) => {
      state.contractDraft.maxWorkspaceKB = Number(e.target.value);
      state.contractDraft.certainty.workspace = 'explicit';
      renderContractView();
    });

    const presetBtn = $('presetStrictV2Btn');
    if (presetBtn) {
      presetBtn.addEventListener('click', () => {
        state.contractDraft.p1GuardrailEnabled = true;
        state.contractDraft.p1MaxRegressionPct = 2.0;
        state.contractDraft.minPrecisionMarginPct = 20;
        state.contractDraft.maxWorkspaceKB = 64;
        state.contractDraft.certainty.p1Regression = 'explicit';
        state.contractDraft.certainty.precisionMargin = 'explicit';
        state.contractDraft.certainty.workspace = 'explicit';
        renderContractView();
      });
    }

    $('applyContractPageBtn').addEventListener('click', () => {
      const nextEpoch = state.contract.epochNum + 1;
      const nextVer = `v${nextEpoch}`;
      state.contract = cloneContract({ ...state.contractDraft, version: nextVer, epochNum: nextEpoch });
      state.contractDraft = cloneContract(state.contract);
      state.contractEpochX = 7.45;
      state.contractHistory.push({
        version: nextVer,
        ts: nowTs(),
        summary: `预算 ${state.contract.maxRuns} 次 / 并发 ${state.contract.maxWorkers} 路 / 停滞 ${state.contract.plateauThresholdRounds} 轮；约束更新：[1,128] ≤${state.contract.p1MaxRegressionPct.toFixed(1)}%，精度裕度 ≥${state.contract.minPrecisionMarginPct}%，Workspace ≤${state.contract.maxWorkspaceKB}KB。`,
      });
      state.auditEvents.push({
        ts: nowTs(),
        kind: 'contract',
        title: `约束变更 ${nextVer} 生效`,
        detail: '预算与约束已统一生效，并按新规则重评历史实验。',
      });
      showToast(`已应用约束 ${nextVer}，历史实验已重新评估`);
      renderAll();
    });
  }

  // ---------- VIEW 1: Full-Width Exploration Canvas (3-Card HUD + Visual-Only Stall + Marquee) ----------
  function renderCanvasView() { renderOpenStage(); }


  function svgEl(tag, attrs = {}, parent = null) {
    const e = document.createElementNS('http://www.w3.org/2000/svg', tag);
    Object.entries(attrs).forEach(([k, v]) => e.setAttribute(k, String(v)));
    if (parent) parent.appendChild(e);
    return e;
  }

  function toggleCompareSelection(runId) {
    const run = state.runs.find((r) => r.id === runId);
    if (!run || run.isRunning || !run.buildOk || !run.testsOk) {
      showToast('仅支持选择已通过编译与校验的实验进行对照');
      return;
    }
    const idx = state.selectedCompareIds.indexOf(runId);
    if (idx >= 0) {
      state.selectedCompareIds.splice(idx, 1);
    } else {
      if (state.selectedCompareIds.length >= 4) state.selectedCompareIds.shift();
      state.selectedCompareIds.push(runId);
    }
    // Keep sorted by experiment seq
    state.selectedCompareIds.sort((a, b) => {
      const ra = state.runs.find((r) => r.id === a);
      const rb = state.runs.find((r) => r.id === b);
      return (ra?.seq ?? 99) - (rb?.seq ?? 99);
    });
    renderCanvasView();
    renderCompareFloatBar();
  }

  function renderCompareFloatBar() {
    const bar = $('compareSelectionBar');
    if (state.activeView !== 'canvas' || state.selectedCompareIds.length < 2) {
      bar.hidden = true;
      return;
    }
    const n = state.selectedCompareIds.length;
    $('compareSelectionText').innerHTML = `已选 ${n} 个实验：<span class="mono" style="color:var(--primary);">${state.selectedCompareIds.join(' · ')}</span>`;
    $('compareSelectedGoBtn').textContent = `对比所选 ${n} 轮 →`;
    bar.hidden = false;
  }

  function renderInspectorPopover() {
    const pop = $('inspectorPanel');
    pop.hidden = !state.inspectorOpen;
    $('inspectorToggle').setAttribute('aria-expanded', String(state.inspectorOpen));
    if (!state.inspectorOpen) return;
    const r = state.runs.find(r=>r.id===state.selectedRunId) || state.runs[0];
    if (!r) return;
    const ev = D.evaluateRun(r,state.contract);
    const lane = state.lanes.find(l=>l.id===r.laneId);
    const value = r.isRunning || r.p0Latency == null ? '未采样' : `${r.p0Latency.toFixed(1)} μs`;
    const src = r.sourceRef || {};
    $('inspectorPanelContent').innerHTML = `<div class="fp-head"><div><small class="mono">实验 ${r.id} · 波次 ${r.round ?? 0}</small><h2 class="fp-title">${escapeHtml(r.shortLabel || r.title)}</h2></div><button class="text-action-btn" id="closeInsPopBtn">关闭</button></div>
      <p>${escapeHtml(r.summary || r.title || '')}</p><p class="mono">${r.parentId ? r.parentId+' → '+r.id : '基线'} · ${D.AXIS_META[r.axis]?.name || '基线'}</p><p class="lane-status">路径状态：${lane?.status==='pruned'?'已剪枝 · 停止后续探索，保留历史记录':lane?.status==='paused'?'已暂停':lane?.status==='archived'?'已归档':lane?.status==='converged'?'已收敛':'可继续核查'}</p>
      <div class="ins-reading mono"><strong>${value}</strong><span>${r.isRunning?'待采样候选':r.buildOk?'历史样本记录':'编译拦截'}</span></div>
      <p class="mono">${r.latencyRangeUs ? 'min–max ['+r.latencyRangeUs.join(', ')+'] μs' : '无耗时包络'}<br>Vec ${r.vecKB ?? '—'} / ${state.contract.vecLimitKB} KB · Workspace ${r.workspaceKB ?? '—'} KB</p>
      <div class="ins-gates">${D.GATE_KEYS.map((k,i)=>`<span>${D.GATE_LABELS[i]} <b>${r.isRunning?'待核查':ev.dims[k]==='pass'?'通过':ev.dims[k]==='fail'?'未通过':'待核查'}</b></span>`).join('')}</div>
      <p class="evidence-note">来源：历史场景记录；原始设备采集与声明 revision 尚未核实。${r.isRunning?'新候选尚无编译或设备结果。':''}</p>
      <details><summary>源码与推导依据</summary><pre>${escapeHtml(r.codeSnippet || '未绑定代码片段')}</pre><p>${escapeHtml(JSON.stringify(src))}</p><p>源码锚点为场景声明，尚未验证该版本的索引映射。</p></details>
      <div class="ins-actions-grid"><button class="btn" id="openCandidateFork">派生分支</button><button class="btn" id="inspectCompare">加入对照</button><button class="btn" id="inspectBudget">调整预算</button>${lane?`<button class="btn" id="inspectPrune" ${lane.status==='pruned'?'disabled':''}>${lane.status==='pruned'?'路径已剪枝':'剪枝路径'}</button>`:''}<button class="btn" id="inspectCode">编辑代码</button><button class="btn" id="inspectHandoff">复制调查上下文</button></div>`;
    $('closeInsPopBtn').onclick=()=>{state.inspectorOpen=false;renderInspectorPopover();};
    $('openCandidateFork').onclick=()=>openForkModal(r.id,r.axis);
    $('inspectCompare').onclick=()=>toggleCompareSelection(r.id);
    $('inspectBudget').onclick=()=>navigateToContract();
    if ($('inspectPrune')) $('inspectPrune').onclick=()=>executePruneLane(lane.id,'人工核查后止损');
    $('inspectCode').onclick=()=>openCodeAnchorModal(r);
    $('inspectHandoff').onclick=async()=>{
      const payload={scope:D.CASE_CONTRACT.scope,id:r.id,round:r.round,parentId:r.parentId,contract:state.contract,sourceStatus:'unverified-scenario-record',measurements:r.isRunning?null:{latencyUs:r.p0Latency,rangeUs:r.latencyRangeUs,vecKB:r.vecKB},question:'核对源码映射、原始采集与当前门禁'};
      try { await navigator.clipboard.writeText(JSON.stringify(payload,null,2));showToast('已复制调查上下文'); } catch { openGenericModal('调查上下文', '<pre>'+escapeHtml(JSON.stringify(payload,null,2))+'</pre>', false); }
    };
  }


  function renderDiffView() {
    const container = $('view-diff');
    const available = state.runs.filter((r) => !r.isRunning && r.buildOk && r.testsOk);
    const ids = state.diffRunIds.map((id) => (available.some((r) => r.id === id) ? id : available[available.length - 1]?.id || '#0'));
    const cols = ids.map((id) => available.find((r) => r.id === id) || state.runs[0]);
    const rBase = cols[0];
    const rFocus = cols[cols.length - 1];
    const speedDelta = rBase && rFocus ? ((1 - rFocus.p0Latency / rBase.p0Latency) * 100).toFixed(1) : '0.0';
    const focusEv = D.evaluateRun(rFocus, state.contract);

    let verdictText = `${rFocus.id} 较 ${rBase.id} 主场景 [32,4096] ${speedDelta >= 0 ? '-' : '+'}${Math.abs(speedDelta)}% · ${focusEv.badge}`;

    const selectOpts = (selId) => available.map((r) =>
      `<option value="${r.id}" ${r.id === selId ? 'selected' : ''}>${r.id} · ${r.p0Latency.toFixed(1)}μs</option>`
    ).join('');

    const shapeRows = [
      { label: '[32, 4096] 稀疏 20%', level: '主场景', key: 'p0Latency', unit: 'μs', base: rBase.p0Latency },
      { label: '[4, 2048] 稀疏 20%', level: '主场景', key: 'p0MidLatency', unit: 'μs', base: rBase.p0MidLatency },
      { label: '[1, 128] 小批量', level: `退化 ≤${state.contract.p1MaxRegressionPct}%`, key: 'p1Latency', unit: 'μs', base: rBase.p1Latency, isP1: true },
      { label: '精度裕度', level: `≥${state.contract.minPrecisionMarginPct}%`, key: 'precisionMarginPct', unit: '%', isMargin: true },
      { label: '额外 Workspace', level: `≤${state.contract.maxWorkspaceKB}KB`, key: 'workspaceKB', unit: 'KB', isWorkspace: true },
    ];

    const matrixHtml = shapeRows.map((row) => {
      const anyAlert = row.isP1 && cols.some((c) => c.p1RegressionPct > state.contract.p1MaxRegressionPct);
      const cells = cols.map((c, idx) => {
        const val = c[row.key];
        if (row.isMargin) {
          const ok = val >= state.contract.minPrecisionMarginPct;
          return `<td><div class="margin-bar-wrap"><div class="margin-bar"><div class="margin-bar-fill" style="width:${Math.min(100, val)}%;background:${ok ? 'var(--success)' : 'var(--danger)'}"></div></div><span class="mono ${ok ? '' : 'delta-bad'}">${val}%</span></div></td>`;
        }
        if (row.isWorkspace) {
          const ok = val <= state.contract.maxWorkspaceKB;
          return `<td class="mono ${ok ? '' : 'delta-bad'}">+${val} KB ${ok ? '✓' : '✗'}</td>`;
        }
        if (idx === 0) return `<td class="mono">${val.toFixed(2)} ${row.unit}</td>`;
        const pct = ((val - row.base) / row.base) * 100;
        const cls = pct < -1 ? 'delta-good' : pct > (row.isP1 ? state.contract.p1MaxRegressionPct : 2) ? 'delta-bad' : pct > 0.5 ? 'delta-warn' : '';
        return `<td class="mono">${val.toFixed(2)} ${row.unit} <span class="${cls}">(${pct >= 0 ? '+' : ''}${pct.toFixed(1)}%)</span></td>`;
      }).join('');
      return `<tr class="${anyAlert ? 'row-alert' : ''}"><td><strong>${row.label}</strong></td><td><span class="pill ${row.isP1 ? 'pill-warning' : 'pill-neutral'}">${row.level}</span></td>${cells}</tr>`;
    }).join('');

    const counterDefs = [
      {key:'aiv_time',label:'aiv_time'}, {key:'core_skew',label:'24 核组极差'},
      {key:'vec_resc_cflt',label:'vec_resc_cflt'}, {key:'mte2',label:'mte2'},
      {key:'mte3',label:'mte3'}, {key:'icache_miss',label:'icache_miss'},
    ];
    const countersHtml = counterDefs.map(cd=>`<div class="counter-item"><span>${cd.label}</span><span class="mono">${rBase.counters?.[cd.key] ?? '—'} → ${rFocus.counters?.[cd.key] ?? '—'}</span><small>原始字段值 · 单位待核验</small></div>`).join('');
    const effRows=state.lanes.map(lane=>{
      const runs=state.runs.filter(r=>r.laneId===lane.id);
      const valid=runs.filter(r=>!r.isRunning&&r.buildOk&&r.testsOk&&Number.isFinite(r.p0Latency));
      const best=valid.sort((a,b)=>a.p0Latency-b.p0Latency)[0];
      const status={pruned:'已剪枝',paused:'已暂停',active:'可继续',running:'待验证',converged:'已收敛',plateau:'停滞',excluded:'无实验',idle:'未开始',archived:'已归档'}[lane.status]||lane.status;
      return {name:lane.name,axis:lane.axis,rounds:runs.length+' 次',out:(best?best.p0Latency.toFixed(1)+' μs · ':'')+(status||'待核查')};
    });

    const hunksHtml = D.CODE_DIFF_HUNKS.map((h) => {
      const ax = D.AXIS_META[h.axis];
      const isOpen = !!state.openHunks[h.id];
      const lines = h.lines.map((l) => `
        <div class="code-line is-${l.type}">
          <span class="code-ln">${l.ln}</span>
          <span class="code-sign">${l.type === 'add' ? '+' : l.type === 'del' ? '-' : ' '}</span>
          <span>${escapeHtml(l.text)}</span>
        </div>
      `).join('');
      return `
        <div class="code-hunk">
          <button class="code-hunk-head" type="button" data-toggle-hunk="${h.id}">
            <div style="display:flex;align-items:center;gap:8px;min-width:0;">
              <span class="mono" style="color:var(--foreground-muted);">${isOpen ? '▼' : '▶'}</span>
              <span class="pill ${ax.pillClass}">${ax.name}</span>
              <strong style="font-size:12px;">${h.title}</strong>
              <span class="mono" style="font-size:11px;color:var(--foreground-muted);">${h.file}</span>
            </div>
            <span class="pill pill-success mono">${h.impact}</span>
          </button>
          <div class="code-hunk-body" ${isOpen ? '' : 'hidden'}>${lines}</div>
        </div>
      `;
    }).join('');

    container.innerHTML = `
      <div class="verdict-bar">
        <div style="display:flex;align-items:center;gap:10px;min-width:0;">
          <span class="pill ${focusEv.overall === 'pass' ? 'pill-success' : 'pill-warning'}">${focusEv.badge}</span>
          <strong style="white-space:nowrap;overflow:hidden;text-overflow:ellipsis;">${verdictText}</strong>
        </div>
        <div style="display:flex;align-items:center;gap:6px;flex-shrink:0;">
          <button class="btn btn-xs" type="button" data-action="open-compute-explorer">Compute Explorer (Graph/Assignment/Pipeline) ↗</button>
          <button class="btn btn-xs" type="button" data-action="open-perf-lane">Perf Lane ↗</button>
        </div>
      </div>

      <div class="panel">
        <div class="panel-head">
          <div class="diff-top-bar" style="width:100%;">
            <div style="display:flex;align-items:center;gap:6px;flex-wrap:wrap;">
              <h3 class="panel-title">多 Shape 与约束矩阵</h3>
              <button class="btn btn-xs" type="button" data-diff-preset="#0,#10,#11">#0 / #10 / #11</button>
              <button class="btn btn-xs" type="button" data-diff-preset="#10,#11">#10 vs #11</button>
              <button class="btn btn-xs" type="button" data-diff-preset="#9,#10">InCore(#9) vs 编排(#10)</button>
            </div>
            <div class="diff-pickers">
              ${cols.map((c, i) => `<select class="diff-select" data-diff-slot="${i}" aria-label="对比实验 ${i + 1}">${selectOpts(c.id)}</select>`).join('<span style="color:var(--foreground-muted);font-size:11px;">vs</span>')}
            </div>
          </div>
        </div>
        <div style="overflow-x:auto;">
          <table class="data-table">
            <thead>
              <tr>
                <th>Shape / 约束</th>
                <th>要求</th>
                ${cols.map((c) => {
                  const ev = D.evaluateRun(c, state.contract);
                  const cls = ev.overall === 'pass' ? 'pill-success' : ev.overall === 'invalid' ? 'pill-danger' : ev.overall === 'risk' ? 'pill-warning' : 'pill-neutral';
                  return `<th><span class="mono">${c.id}</span> <span class="pill ${cls}">${ev.badge}</span></th>`;
                }).join('')}
              </tr>
            </thead>
            <tbody>${matrixHtml}</tbody>
          </table>
        </div>
      </div>

      <div class="diff-two-col">
        <div class="panel">
          <div class="panel-head">
            <h3 class="panel-title">采集信号对照 · 单位与原始采集待核验</h3>
            <span class="panel-sub mono">${rBase.id} → ${rFocus.id}</span>
          </div>
          <div class="panel-body"><div class="counter-list">${countersHtml}</div></div>
        </div>

        <div class="panel">
          <div class="panel-head">
            <h3 class="panel-title">各方向尝试统计</h3>
          </div>
          <div style="overflow-x:auto;">
            <table class="data-table">
              <thead><tr><th>优化路径</th><th>方向</th><th>轮数</th><th>结果</th></tr></thead>
              <tbody>
                ${effRows.map((er) => {
                  const ax = D.AXIS_META[er.axis];
                  return `<tr style="${er.axis === 'host' ? 'opacity:0.55;' : ''}">
                    <td><strong>${er.name}</strong></td>
                    <td><span class="pill ${ax.pillClass}">${ax.short}</span></td>
                    <td class="mono">${er.rounds}</td>
                    <td class="mono">${er.out}</td>
                  </tr>`;
                }).join('')}
              </tbody>
            </table>
          </div>
        </div>
      </div>

      <div class="panel">
        <div class="panel-head">
          <h3 class="panel-title">PyPTO 语义代码变更</h3>
          <span class="panel-sub">按调优方向折叠 · 关联历史记录</span>
        </div>
        <div class="panel-body">${hunksHtml}</div>
      </div>
    `;

    document.querySelectorAll('#view-diff [data-diff-preset]').forEach((b) => {
      b.addEventListener('click', () => { state.diffRunIds = b.dataset.diffPreset.split(','); renderDiffView(); });
    });
    document.querySelectorAll('#view-diff [data-diff-slot]').forEach((sel) => {
      sel.addEventListener('change', () => { state.diffRunIds[Number(sel.dataset.diffSlot)] = sel.value; renderDiffView(); });
    });
    document.querySelectorAll('#view-diff [data-toggle-hunk]').forEach((b) => {
      b.addEventListener('click', () => { state.openHunks[b.dataset.toggleHunk] = !state.openHunks[b.dataset.toggleHunk]; renderDiffView(); });
    });
  }

  // ---------- VIEW 3: Acceptance Gate (Scenario A #11 vs Scenario D #10) ----------
  function renderGateView() {
    const container = $('view-gate');
    const candidates = ['#11', '#10', '#3'].map((id) => state.runs.find((r) => r.id === id && !r.isRunning)).filter(Boolean);
    state.runs.filter((r) => r.isCustom && !r.isRunning && r.buildOk && r.testsOk).forEach((cr) => candidates.push(cr));

    let cand = candidates.find((c) => c.id === state.gateCandidateId) || candidates[0] || state.runs[0];
    if (!cand) return;
    state.gateCandidateId = cand.id;

    const ev = D.evaluateRun(cand, state.contract);
    const isScenarioD = ev.overall !== 'pass';
    const speedup = (482.0 / cand.p0Latency).toFixed(2);
    const latDrop = ((1 - cand.p0Latency / 482.0) * 100).toFixed(1);

    const candSwitchHtml = candidates.map((c) => {
      const cev = D.evaluateRun(c, state.contract);
      const dotCol = cev.overall === 'pass' ? 'var(--success)' : (cev.overall === 'invalid' || cev.overall === 'risk') ? 'var(--danger)' : 'var(--warning)';
      const tag = c.id === '#11' ? '全通过' : c.id === '#10' ? '有风险' : c.id === '#3' ? '保守' : '派生';
      return `<button class="cand-tab ${c.id === cand.id ? 'is-active' : ''}" type="button" data-gate-cand="${c.id}">
        <span class="axis-dot" style="background:${dotCol}"></span>
        <span class="mono">${c.id}</span>
        <span style="color:var(--foreground-muted);font-size:11px;">${tag} · ${c.p0Latency.toFixed(0)}μs</span>
      </button>`;
    }).join('');

    const sealSvg = isScenarioD
      ? `<svg viewBox="0 0 56 56" width="52" height="52"><circle cx="28" cy="28" r="23" fill="var(--pill-danger-bg)" stroke="var(--danger)" stroke-width="2.5" stroke-dasharray="6 4" /><text x="28" y="34" text-anchor="middle" fill="var(--danger)" font-size="22" font-weight="700">!</text></svg>`
      : `<svg viewBox="0 0 56 56" width="52" height="52"><circle cx="28" cy="28" r="23" fill="var(--pill-success-bg)" stroke="var(--success)" stroke-width="2.5" /><path d="M18 28.5 25 35.5 39 21.5" fill="none" stroke="var(--success)" stroke-width="3" stroke-linecap="round" stroke-linejoin="round" /></svg>`;

    const riskTriageHtml = isScenarioD ? `
      <div class="panel" style="border-color:color-mix(in srgb, var(--danger) 36%, var(--border-subtle));">
        <div class="panel-head">
          <h3 class="panel-title">
            <span style="color:var(--danger);">1 · 2 · 3 · 4 风险分诊 (${cand.id})</span>
            <span class="panel-sub">已拦截自动晋升</span>
          </h3>
          <div style="display:flex;gap:8px;flex-wrap:wrap;">
            ${state.runs.some((r) => r.id === '#11' && !r.isRunning) ? `<button class="btn btn-xs btn-primary" type="button" data-gate-cand="#11">切至全通过 #11</button>` : ''}
            <button class="btn btn-xs btn-warning-outline" type="button" data-action="goto-contract">调整目标与约束</button>
            <button class="btn btn-xs btn-danger-outline" type="button" id="openWaiverBtn">知情豁免</button>
          </div>
        </div>
        <div class="panel-body">
          <div class="risk-triage-grid">
            ${D.RISK_BUCKETS_FAST.map((rb) => {
              let meterHtml = '';
              if (rb.meterType === 'dots') {
                meterHtml = `<div class="meter-dots">${Array.from({ length: rb.totalDots }, (_, i) => `<span class="meter-dot ${i < rb.filledDots ? 'is-filled' : ''}"></span>`).join('')}</div>`;
              } else {
                const col = rb.sev === 'high' ? 'var(--danger)' : 'var(--warning)';
                meterHtml = `<div class="meter-track"><div class="meter-fill" style="width:${rb.valuePct}%;background:${col}"></div><div class="meter-threshold" style="left:${rb.thresholdPct}%" title="${rb.limit}"></div></div>`;
              }
              return `
                <div class="risk-meter-card" data-sev="${rb.sev}">
                  <div class="risk-card-top">
                    <span style="display:inline-flex;align-items:center;gap:6px;">
                      <span class="exc-num is-${rb.sev === 'high' ? 'danger' : 'warning'}">${rb.num}</span>
                      <strong style="font-size:12px;">${rb.title}</strong>
                    </span>
                  </div>
                  <div style="display:flex;justify-content:space-between;align-items:baseline;">
                    <span class="mono" style="font-size:14px;font-weight:700;color:${rb.sev === 'high' ? 'var(--danger)' : 'var(--warning)'};">${rb.metric}</span>
                    <span class="mono" style="font-size:10.5px;color:var(--foreground-muted);">${rb.limit}</span>
                  </div>
                  ${meterHtml}
                  <div style="font-size:11px;color:var(--foreground-secondary);">${rb.caption}</div>
                </div>
              `;
            }).join('')}
          </div>
        </div>
      </div>
    ` : '';

    const gateItems = [
      { key: 'hardware', name: '1. 硬件', val: state.contract.chip.replace('Ascend ', '') },
      { key: 'correctness', name: '2. 正确性', val: 'Golden ✓ · 0 Fallback' },
      { key: 'precision', name: '3. 精度', val: `裕度 ${cand.precisionMarginPct}%` },
      { key: 'generalization', name: '4. 泛化', val: `[1,128] ${cand.p1RegressionPct >= 0 ? '+' : ''}${cand.p1RegressionPct.toFixed(1)}%` },
      { key: 'performance', name: '5. 性能', val: `${speedup}× · +${cand.workspaceKB}KB` },
      { key: 'engineering', name: '6. 规范', val: '合规 API · 0 警告' },
    ];

    container.innerHTML = `
      <div class="gate-hero ${isScenarioD ? 'is-risk' : 'is-pass'}">
        <div class="gate-hero-left">
          ${sealSvg}
          <div>
            <div style="display:flex;align-items:center;gap:8px;">
              <strong style="font-size:17px;">${isScenarioD ? `${cand.id} · 存在风险拦截` : `${cand.id} · 历史记录符合当前门禁`}</strong>
              <span class="pill ${isScenarioD ? 'pill-danger' : 'pill-success'} mono">${state.contract.version}</span>
            </div>
            <div class="gate-cand-row" style="margin-top:8px;">${candSwitchHtml}</div>
          </div>
        </div>
        <div style="display:flex;align-items:center;gap:8px;">
          <button class="btn btn-sm" type="button" data-action="goto-diff">多轮对照</button>
          <button class="btn btn-sm btn-primary" type="button" id="generatePrReportBtn" ${isScenarioD ? 'disabled' : ''}>生成验收报告</button>
        </div>
      </div>

      ${riskTriageHtml}

      <div class="kpi-grid-4">
        <div class="kpi-card">
          <div class="kpi-label">主场景 [32,4096] 延迟</div>
          <div class="kpi-val">${cand.p0Latency.toFixed(1)} <span style="font-size:13px;font-weight:500;">μs</span></div>
          <div class="kpi-sub delta-good">-${latDrop}% · ${speedup}× vs #0</div>
        </div>
        <div class="kpi-card">
          <div class="kpi-label">Roofline & 核组均衡</div>
          <div class="kpi-val">${cand.rooflinePct.toFixed(1)}%</div>
          <div class="kpi-sub mono">24 CG 极差 ${cand.coreSkew.toFixed(2)}×</div>
        </div>
        <div class="kpi-card">
          <div class="kpi-label">小 Shape [1,128]</div>
          <div class="kpi-val ${cand.p1RegressionPct > state.contract.p1MaxRegressionPct ? 'delta-bad' : 'delta-good'}">${cand.p1RegressionPct >= 0 ? '+' : ''}${cand.p1RegressionPct.toFixed(1)}%</div>
          <div class="kpi-sub mono">${cand.p1Latency.toFixed(2)}μs · 上限 ≤${state.contract.p1MaxRegressionPct}%</div>
        </div>
        <div class="kpi-card">
          <div class="kpi-label">精度裕度 & Workspace</div>
          <div class="kpi-val ${cand.precisionMarginPct < state.contract.minPrecisionMarginPct ? 'delta-bad' : ''}">${cand.precisionMarginPct}%</div>
          <div class="kpi-sub mono">+${cand.workspaceKB}KB · 编译 ${cand.compileSec}s</div>
        </div>
      </div>

      <div class="gate-pills-6">
        ${gateItems.map((g) => {
          const st = ev.dims[g.key] || 'pass';
          const ico = st === 'pass' ? '✓' : st === 'warn' ? '⚠' : '✗';
          const col = st === 'pass' ? 'var(--success)' : st === 'warn' ? 'var(--warning)' : 'var(--danger)';
          return `<div class="gate-mini-card" data-status="${st}"><div style="display:flex;justify-content:space-between;align-items:center;"><span style="font-size:11px;font-weight:600;">${g.name}</span><span class="mono" style="color:${col};font-weight:700;">${ico}</span></div><div class="mono" style="font-size:12px;color:var(--foreground-secondary);">${g.val}</div></div>`;
        }).join('')}
      </div>

      <div class="panel">
        <div class="panel-head">
          <h3 class="panel-title">Shape & 边界覆盖矩阵</h3>
          <span class="panel-sub mono">${state.contract.fingerprint} · 8 组场景覆盖记录 · 原始采集待核验</span>
        </div>
        <div class="panel-body">
          <div class="shape-dot-grid">
            ${D.COVERAGE_SHAPES.map((cs) => {
              const failP1 = isScenarioD && cs.shape === '[1, 128]';
              const blindSparse = isScenarioD && cs.shape.includes('5%');
              const cls = failP1 ? 'pill-danger' : blindSparse ? 'pill-warning' : 'pill-success';
              const txt = failP1 ? '+5.5% ✗' : blindSparse ? '未测' : 'PASS';
              return `<div class="shape-tile"><div><div class="mono" style="font-weight:600;">${cs.shape}</div><div style="font-size:10.5px;color:var(--foreground-muted);">${cs.bucket}</div></div><span class="pill ${cls} mono">${txt}</span></div>`;
            }).join('')}
          </div>
        </div>
      </div>
    `;

    document.querySelectorAll('#view-gate [data-gate-cand]').forEach((btn) => {
      btn.addEventListener('click', () => {
        state.gateCandidateId = btn.dataset.gateCand;
        state.selectedRunId = btn.dataset.gateCand;
        renderShell();
        renderGateView();
        if (state.inspectorOpen) renderInspectorPopover();
      });
    });

    const prBtn = $('generatePrReportBtn');
    if (prBtn) prBtn.addEventListener('click', () => openPrReportModal(cand));
    const wvBtn = $('openWaiverBtn');
    if (wvBtn) wvBtn.addEventListener('click', () => openWaiverModal(cand));
  }

  // ---------- Floating Bottom Agent Composer Dock ----------
  function renderComposerDock() {
    $('composerPopover').hidden = false;
    $('composerHistory').hidden = !state.threadOpen;
    $('composerHeaderToggle').setAttribute('aria-expanded',String(state.threadOpen));
    $('composerPopScope').textContent = `${state.selectedRunId} · ${state.contract.version}`;
    $('composerThreadBody').innerHTML = state.threadMessages.map(m=>`<p class="thread-msg"><b>${m.role==='user'?'你':'调优记录'}</b> ${escapeHtml(m.text)}</p>`).join('');
    $('composerChips').innerHTML = state.attachedChips.map(ch=>`<span class="agent-context-chip">${escapeHtml(ch.label)}<button type="button" data-remove-context="${escapeHtml(ch.id)}" aria-label="移除上下文">×</button></span>`).join('');
    updateComposerSendVisual();
  }


  function updateComposerSendVisual() { $('composerSubmitBtn').disabled = !$('composerInput').value.trim(); }


  function addComposerChip(chip) {
    if (!state.attachedChips.some((c) => c.id === chip.id)) {
      state.attachedChips.push(chip);
      renderComposerDock();
    }
  }

  function handleComposerSubmit(rawText) {
    const text=rawText.trim();if(!text)return;
    state.threadMessages.push({role:'user',ts:nowTs(),text});
    if(text.includes('剪枝')) executePruneLane('lane-incore','指令止损');
    else if(/预算|约束/.test(text))navigateToContract();
    else {
      const requested=text.match(/#\d+/)?.[0]||state.selectedRunId;
      const axis=/Ring|内存/i.test(text)?'memory':/编排|派发/.test(text)?'dispatch':'incore';
      spawnQuickHypothesisRun(requested,axis,text);
      state.threadMessages.push({role:'agent',ts:nowTs(),text:'已记录分支假设。等待外部编译与设备采样，尚无性能或验收结论。'});
    }
    state.threadOpen=true;renderAll();
  }


  function executePruneLane(laneId, reason) {
    const lane = state.lanes.find((l) => l.id === laneId);
    if (!lane || lane.status === 'pruned') return;
    lane.status = 'pruned';
    lane.pruneReason = reason;
    state.reflexionRules.unshift({ id: `rx-${Date.now().toString().slice(-3)}`, source: `剪枝 ${lane.name}`, rule: `停止「${lane.name}」（${reason}），释放并发槽位。` });
    state.auditEvents.push({ ts: nowTs(), kind: 'human', title: `◆ 剪枝「${lane.name}」`, detail: `原因：${reason}。` });
    showToast(`已剪枝「${lane.name}」，释放 1 路并发槽位`);
    renderAll();
  }

  function ensureDispatchLaneRunning() {
    let lane = state.lanes.find((l) => l.id === 'lane-dispatch');
    if (!lane) {
      lane = { ...D.BASE_LANES.find((l) => l.id === 'lane-dispatch'), status: 'running', budgetMode: 'boost' };
      state.lanes.push(lane);
    } else {
      lane.status = 'running';
      lane.budgetMode = 'boost';
    }
    if (!state.humanAnchors.some((a) => a.id === 'ha-refute')) {
      state.humanAnchors.push({ id: 'ha-refute', x: 5.98, laneId: 'lane-dispatch', nearRunId: '#10', shortLabel: '反证转编排', detail: '剪枝停滞路径并转向编排与分核派发。', ts: nowTs() });
    }
    if (!state.runs.some(r=>r.id==='#10')) {
      state.runs.push({...D.BASE_RUNS.find(r=>r.id==='#10'),isRunning:false});
      state.act=Math.max(3,state.act);
      showToast('已载入 #10 历史回放记录，当前未触发设备执行');
    }
  }


  function spawnQuickHypothesisRun(sourceRunId, axis, hyp) {
    if (state.runs.length >= state.contract.maxRuns) {showToast('实验预算已用尽，请先调整预算');return;}
    const src=state.runs.find(r=>r.id===sourceRunId)||getIncumbentBestRun()||state.runs[0];
    const seq=state.nextSeq++, id=`#${seq}`, laneId=`lane-custom-${seq}`;
    state.lanes.push({id:laneId,name:`${D.AXIS_META[axis]?.name || '定向'} Fork`,axis,status:'paused',hypothesis:hyp});
    state.runs.push({id,seq,round:Math.max(...state.runs.map(r=>r.round??0))+1,parentId:src.id,laneId,axis,shortLabel:hyp,title:hyp,summary:`从 ${src.id} 派生，等待编译与设备采样。`,isRunning:true,isCustom:true,evidenceGrade:'candidate',p0Latency:null,latencyRangeUs:null,vecKB:null,workspaceKB:null,buildOk:true,testsOk:true,x:src.x+1});
    state.selectedRunId=id;state.bullseyeFocusId=id;state.activeView='canvas';state.inspectorOpen=true;
    state.auditEvents.push({ts:nowTs(),kind:'human',title:`派生 ${id}`,detail:'待验证候选；尚未连接设备执行。'});
    showToast(`已从 ${src.id} 派生 ${id}，等待采样`);renderAll();
  }


  function openAuditDrawer() {
    $('auditDrawerVerPill').textContent = state.contract.version;
    $('auditDrawerBody').innerHTML = `
      <div class="drawer-section">
        <div class="drawer-section-title">
          <span>约束版本记录</span>
          <div style="display:flex;gap:6px;">
            <button class="btn btn-xs" type="button" id="exportAuditJsonBtn">导出 JSON</button>
            <button class="btn btn-xs" type="button" id="exportAuditMdBtn">导出 Markdown</button>
          </div>
        </div>
        <table class="data-table">
          <thead><tr><th>版本</th><th>时间</th><th>变更说明</th></tr></thead>
          <tbody>${state.contractHistory.map((c) => `<tr><td class="mono"><strong>${c.version}</strong></td><td class="mono">${c.ts}</td><td>${c.summary}</td></tr>`).join('')}</tbody>
        </table>
      </div>
      <div class="drawer-section">
        <div class="drawer-section-title"><span>决策与手动介入时间线</span><span class="mono">${state.auditEvents.length}</span></div>
        <div>${[...state.auditEvents].reverse().map((ev) => `<div class="timeline-row"><span class="mono" style="color:var(--foreground-muted);">${ev.ts}</span><span class="tl-dot is-${ev.kind}"></span><div><strong>${ev.title}</strong><div style="color:var(--foreground-secondary);font-size:11.5px;margin-top:2px;">${ev.detail}</div></div></div>`).join('')}</div>
      </div>
      <div class="drawer-section">
        <div class="drawer-section-title"><span>已沉淀经验禁区</span><span class="mono">${state.reflexionRules.length}</span></div>
        ${state.reflexionRules.map((rx) => `<div style="padding:7px 0;border-bottom:1px solid var(--border-subtle);font-size:11.5px;"><span class="pill pill-human" style="margin-bottom:4px;">${rx.source}</span><div>${rx.rule}</div></div>`).join('')}
      </div>
    `;
    $('auditDrawerBackdrop').hidden = false;

    $('exportAuditJsonBtn').onclick = () => {
      const payload = JSON.stringify({ operator: 'decode_sparse_attn_csa', contract: state.contract, history: state.contractHistory, events: state.auditEvents }, null, 2);
      openGenericModal('决策记录 JSON', `<pre class="code-editor-area" style="margin:0;white-space:pre-wrap;">${escapeHtml(payload)}</pre>`, true, payload);
    };
    $('exportAuditMdBtn').onclick = () => {
      const md = `# PyPTO 调优记录 · decode_sparse_attn_csa\n- 当前约束：${state.contract.version}\n` + state.contractHistory.map((c) => `- **${c.version}** (${c.ts}): ${c.summary}`).join('\n');
      openGenericModal('决策复盘 Markdown', `<pre class="code-editor-area" style="margin:0;white-space:pre-wrap;">${escapeHtml(md)}</pre>`, true, md);
    };
  }

  function openForkModal(sourceRunId, defaultAxis) {
    const src = state.runs.find((r) => r.id === sourceRunId) || getIncumbentBestRun() || state.runs[0];
    $('forkSourceBadge').textContent = `${src.id} (${src.p0Latency ? src.p0Latency.toFixed(1) + 'μs' : '基线'})`;
    const initAxis = defaultAxis || (src.id === '#10' ? 'incore' : 'dispatch');

    $('forkModalBody').innerHTML = `
      <div class="form-field">
        <label for="forkAxisSelect">目标 PyPTO 优化方向</label>
        <div id="forkAxisSelect" class="fork-axis-options" role="group" aria-label="调优方向" data-value="${initAxis}">${['dispatch','incore','memory'].map(axis=>`<button class="btn ${axis===initAxis?'btn-primary':''}" type="button" data-fork-axis="${axis}">${D.AXIS_META[axis].name}</button>`).join('')}</div>
      </div>
      <div class="form-field" style="margin-bottom:0;">
        <label for="forkHypothesisInput">探索方向说明</label>
        <input id="forkHypothesisInput" type="text" value="${initAxis === 'dispatch' ? '错峰轮转分核映射，消除 24 核组双峰碰撞' : '保留错峰分核，为 t_dim<=4 增加直通快路径并压缩 Workspace'}" />
      </div>
    `;
    $('forkModalBackdrop').hidden = false;
    $('forkAxisSelect').querySelectorAll('button').forEach(b=>b.onclick=()=>{$('forkAxisSelect').dataset.value=b.dataset.forkAxis;$('forkAxisSelect').querySelectorAll('button').forEach(t=>t.classList.toggle('btn-primary',t===b));});

    $('confirmForkBtn').onclick = () => {
      const axis = $('forkAxisSelect').dataset.value;
      const hyp = $('forkHypothesisInput').value.trim() || '定向优化尝试';
      $('forkModalBackdrop').hidden = true;
      spawnQuickHypothesisRun(src.id, axis, hyp);
      showToast(`已从 ${src.id} 派生候选，等待编译与采样`);
    };
  }

  function closeToolModal() {
    $('toolModalBackdrop').hidden = true;
    const iframe = $('toolModalIframe');
    if (iframe) iframe.src = 'about:blank';
  }

  function openSingleToolModal(toolKey) {
    const urls = {
      'perf-lane': '../../PyPTO-Compute-Explorer-UI/perf-lane-slz/result.html',
      'compute-explorer': '../../PyPTO-Compute-Explorer-UI/csa.html',
    };
    const key = toolKey === 'compute-explorer' ? 'compute-explorer' : 'perf-lane';
    const iframe = $('toolModalIframe');

    function setTool(k) {
      document.querySelectorAll('#toolModalSwitch .seg-btn').forEach((b) => b.classList.toggle('is-active', b.dataset.tool === k));
      requestAnimationFrame(() => { iframe.src = urls[k]; });
      $('toolHypothesisText').textContent = k === 'perf-lane'
        ? 'InCore 双缓冲未掩盖搬运（连续多轮集中在 InCore）'
        : 'Compute Explorer · Graph / Assignment / Pipeline 诊断视角';
    }

    $('toolModalBackdrop').hidden = false;
    setTool(key);
    document.querySelectorAll('#toolModalSwitch .seg-btn').forEach((b) => { b.onclick = () => setTool(b.dataset.tool); });

    $('refuteWithEvidenceBtn').onclick = () => {
      closeToolModal();
      state.pendingRefutation = {
        sourceTool: 'Device Perf Lane',
        refutedHypothesis: 'InCore 双缓冲未掩盖搬运',
        summaryMetrics: 'CG 0~11: 382μs vs CG 12~23: 153μs（核组极差 2.50×）',
      };
      state.activeView = 'canvas';
      state.selectedLaneId = null;
      state.inspectorOpen = true;
      showToast('已带双峰实测返回画布，可在右侧浮层一键转向编排');
      renderAll();
    };
  }

  function openCodeAnchorModal(run) {
    $('codeAnchorNodeBadge').textContent=run.id;
    $('codeEditorTextarea').value=run.codeSnippet||'';
    $('codeModalBackdrop').hidden=false;
    $('saveCodeAnchorBtn').onclick=()=>{
      const text=$('codeEditorTextarea').value, note=$('codeAnchorNote').value.trim()||'代码调整';
      $('codeModalBackdrop').hidden=true;spawnQuickHypothesisRun(run.id,run.axis||'incore',note);
      const created=state.runs.find(r=>r.id===state.selectedRunId);if(created?.isCustom)created.codeSnippet=text;
    };
  }


  function openPrReportModal(cand) {
    const speedup = (482.0 / cand.p0Latency).toFixed(2);
    const md = [
      `## [PyPTO Agent] decode_sparse_attn_csa — ${cand.id} (${speedup}× · ${state.contract.version} 全通过)`,
      `- 硬件：${state.contract.chip} (${state.contract.topology}) · 指纹 \`${state.contract.fingerprint}\``,
      `- 主场景 \`[32,4096]\`: 482.0μs → **${cand.p0Latency.toFixed(1)}μs** (${speedup}×)`,
      `- 小 Shape \`[1,128]\`: 3.25μs → **${cand.p1Latency.toFixed(2)}μs** (${cand.p1RegressionPct.toFixed(1)}% ✓)`,
      ``,
      `### 核心改动`,
      `1. **编排与派发**：\`qk_pv\` 24 核组错峰轮转映射，核组极差 \`2.50× → ${cand.coreSkew.toFixed(2)}×\`。`,
      `2. **InCore 调优**：\`t_dim <= 4\` 直通快路径消除小 Shape 同步开销。`,
      `3. **内存与 Ring**：复用 \`scope-depth ring\` 压缩 Workspace 至 \`+${cand.workspaceKB}KB\`，保持 \`Acc\` FP32 累加裕度 \`${cand.precisionMarginPct}%\`。`,
    ].join('\n');
    openGenericModal(`PR 与验收摘要 · ${cand.id}`, `<pre class="code-editor-area" style="margin:0;white-space:pre-wrap;min-height:240px;">${escapeHtml(md)}</pre>`, true, md);
  }

  function openWaiverModal(cand) {
    $('genericModalTitle').textContent = `知情豁免 · ${cand.id}`;
    $('genericModalBody').innerHTML = `
      <div class="impact-preview-box" style="margin-bottom:10px;border-left-color:var(--danger);">
        ${cand.id} 存在 [1,128] +${cand.p1RegressionPct.toFixed(1)}% 退化与精度裕度 ${cand.precisionMarginPct}% 风险，豁免理由将写入决策记录。
      </div>
      <div class="form-field" style="margin-bottom:0;">
        <label for="waiverReasonInput">豁免理由</label>
        <textarea id="waiverReasonInput" rows="3">仅部署于大批量离线集群（B≥16），不覆盖 [1,128] 单请求场景。</textarea>
      </div>
    `;
    $('genericModalFoot').innerHTML = `<button class="btn btn-sm" type="button" id="cancelWaiverBtn">取消</button><button class="btn btn-sm btn-danger-outline" type="button" id="confirmWaiverBtn">确认豁免</button>`;
    $('genericModalBackdrop').hidden = false;
    $('cancelWaiverBtn').onclick = () => { $('genericModalBackdrop').hidden = true; };
    $('confirmWaiverBtn').onclick = () => {
      const reason = $('waiverReasonInput').value.trim();
      if (!reason) return;
      state.auditEvents.push({ ts: nowTs(), kind: 'human', title: `◆ 知情豁免 (${cand.id})`, detail: reason });
      $('genericModalBackdrop').hidden = true;
      showToast(`已记录 ${cand.id} 知情豁免理由`);
      renderAll();
    };
  }

  function openGenericModal(title, bodyHtml, copyable, rawText) {
    $('genericModalTitle').textContent = title;
    $('genericModalBody').innerHTML = bodyHtml;
    $('genericModalFoot').innerHTML = `${copyable ? `<button class="btn btn-sm btn-primary" type="button" id="copyGenericBtn">复制</button>` : ''}<button class="btn btn-sm" type="button" id="okGenericBtn">关闭</button>`;
    $('genericModalBackdrop').hidden = false;
    $('okGenericBtn').onclick = () => { $('genericModalBackdrop').hidden = true; };
    if (copyable) {
      $('copyGenericBtn').onclick = () => {
        if (navigator.clipboard && rawText) navigator.clipboard.writeText(rawText).catch(() => {});
        showToast('已复制');
      };
    }
  }

  // ---------- Unified Action Dispatcher ----------
  function handleAction(action) {
    switch (action) {
      case 'inspect-stall': state.selectedRunId='#9';state.inspectorOpen=true;renderInspectorPopover();break;
      case 'inspect-run-10': state.selectedRunId='#10';state.inspectorOpen=true;renderInspectorPopover();break;
      case 'start-act1': loadActSnapshot(1); showToast('已启动 3 路并行探索'); renderAll(); break;
      case 'prune-and-dispatch': executePruneLane('lane-incore', '连续集中在 InCore 无新收益'); ensureDispatchLaneRunning(); renderAll(); break;
      case 'dispatch-assign': openForkModal('#3', 'dispatch'); break;
      case 'focus-rollbacks': state.activeExceptionId = 'exc-rollback'; renderCanvasView(); break;
      case 'open-perf-lane': openSingleToolModal('perf-lane'); break;
      case 'open-compute-explorer': openSingleToolModal('compute-explorer'); break;
      case 'goto-contract': navigateToContract('generalization'); break;
      case 'open-fork-10': openForkModal(state.runs.some((r) => r.id === '#10') ? '#10' : '#3', 'incore'); break;
      case 'open-fork-11': openForkModal('#11', 'incore'); break;
      case 'view-risk-radar': state.gateCandidateId = '#10'; state.activeView = 'gate'; state.inspectorOpen = false; renderAll(); break;
      case 'goto-diff': state.activeView = 'diff'; state.inspectorOpen = false; renderAll(); break;
      case 'goto-gate-best': state.gateCandidateId = '#11'; state.activeView = 'gate'; state.inspectorOpen = false; renderAll(); break;
      default: break;
    }
  }

  // ---------- Live Playback ----------
  function togglePlayback() {
    if (state.isPlaying) { stopPlayback(); return; }
    state.isPlaying = true;
    let cur = state.act >= 4 ? 0 : state.act;
    loadActSnapshot(cur, { defaultView: 'canvas' });
    renderAll();
    showToast('开始演进回放：Act 0 → Act 4');
    state.playbackTimer = setInterval(() => {
      cur += 1;
      if (cur > 4) { stopPlayback(); showToast('演进完成：#11 已通过 v2 全部 6 项门禁'); return; }
      loadActSnapshot(cur, { defaultView: 'canvas' });
      renderAll();
    }, 2200);
  }

  function stopPlayback() {
    state.isPlaying = false;
    if (state.playbackTimer) { clearInterval(state.playbackTimer); state.playbackTimer = null; }
    renderShell();
  }

  // ---------- Master Render ----------
  function renderAll() {
    renderShell();
    if (state.activeView === 'contract') renderContractView();
    else if (state.activeView === 'canvas') renderCanvasView();
    else if (state.activeView === 'diff') renderDiffView();
    else if (state.activeView === 'gate') renderGateView();
    if (state.activeView !== 'canvas') $('view-'+state.activeView).insertAdjacentHTML('afterbegin','<p class="evidence-note">记录来源 data.js · 原始设备采集、硬件与源码声明待核验</p>');
    renderInspectorPopover();
  }

  function nowTs() {
    const d = new Date();
    return [d.getHours(), d.getMinutes(), d.getSeconds()].map((n) => String(n).padStart(2, '0')).join(':');
  }

  function escapeHtml(str) {
    return String(str).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
  }

  // ---------- Global Static Wiring ----------
  // Open stage: one shared object model, one camera per scheme, no layout docks.
  const stageCameras = Object.fromEntries(['landscape','bullseye','swimlane'].map(k=>[k,{x:0,y:0,k:1}]));
  let stagePositions = {}, stageSize={w:0,h:0}, activeMetric='latency';
  const C={normal:'var(--primary)',adopted:'var(--ss-pass)',regress:'var(--ss-anomaly)',muted:'var(--ss-text-muted)',text:'var(--ss-text)',line:'var(--ss-boundary)'};
  const numeric=r=>!r.isRunning && Number.isFinite(r.p0Latency) && r.p0Latency>0;
  const measure=(r,key)=>key==='vec'?(Number.isFinite(r.vecKB)?r.vecKB:null):numeric(r)?key==='speedup'?482/r.p0Latency:r.p0Latency:null;
  function runColor(r) {
    if(r.isRunning)return C.muted;
    if(r.id==='#11' && D.evaluateRun(r,state.contract).overall==='pass')return C.adopted;
    if(!r.buildOk || !r.testsOk || ['#4','#7'].includes(r.id) || (state.contract.p1GuardrailEnabled && r.p1RegressionPct>state.contract.p1MaxRegressionPct))return C.regress;
    return C.normal;
  }
  function stageText(root,x,y,content,options={}) {
    const t=svgEl('text',{x,y,fill:C.muted,'font-size':11,'font-family':'ui-monospace, SFMono-Regular, monospace',...options},root);t.textContent=content;return t;
  }
  function stageLine(root,x1,y1,x2,y2,options={}) {return svgEl('line',{x1,y1,x2,y2,stroke:C.line,'stroke-width':1,...options},root);}
  function selectStageRun(r,e={}) {
    if(e.shiftKey||state.multiSelectMode)toggleCompareSelection(r.id);
    else {state.selectedRunId=r.id;state.selectedLaneId=null;state.bullseyeFocusId=r.id;state.inspectorOpen=true;renderOpenStage();renderInspectorPopover();}
  }
  function isPrunedRun(r) { return state.lanes.some(l=>l.id===r.laneId && l.status==='pruned'); }
  const seenBubbles = new Set();
  function bubbleRadius(r) { return 28 + Math.min(20, Math.max(0,r.seq ?? 0)) * 1.8; }
  function placeLandscapeBubbles(bounds) {
    const placed=[];
    const column=(bounds.right-bounds.left)/Math.max(7,...state.runs.map(r=>r.round??0));
    // Display circles can move; the quantitative anchors, curves and waves never do.
    [...state.runs].sort((a,b)=>(b.seq??0)-(a.seq??0)).forEach(r=>{
      const p=stagePositions[r.id],radius=bubbleRadius(r);
      let best=null;
      for(let step=0;step<=12;step++) {
        const offsets=step===0?[0]:[-step*14,step*14];
        offsets.forEach(dy=>[0,-column*.28,column*.28].forEach(dx=>{
          const x=Math.max(bounds.left-radius*.15,Math.min(bounds.right+radius*.15,p.x+dx));
          const y=Math.max(bounds.top+radius,Math.min(bounds.bottom+28,p.y+dy));
          const overlap=placed.reduce((sum,q)=>sum+Math.pow(Math.max(0,radius+q.radius+7-Math.hypot(x-q.x,y-q.y)),2),0);
          const score=overlap*100+Math.pow(x-p.x,2)+Math.pow(y-p.y,2);
          if(!best||score<best.score)best={x,y,radius,score};
        }));
      }
      p.bubbleX=best.x;p.bubbleY=best.y;placed.push(best);
    });
  }
  function drawLandscapeBubble(root,r,p,{selected,dim}) {
    const radius=bubbleRadius(r),x=p.bubbleX??p.x,y=p.bubbleY??p.y;
    const distance=Math.hypot(x-p.x,y-p.y);
    if(distance>radius+3) {
      const ratio=(distance-radius)/distance;
      stageLine(root,p.x,p.y,p.x+(x-p.x)*ratio,p.y+(y-p.y)*ratio,{stroke:runColor(r),opacity:.6});
      svgEl('circle',{cx:p.x,cy:p.y,r:2.6,fill:runColor(r)},root);
    }
    const value=measure(r,activeMetric);
    const reading=value==null?'无读数':activeMetric==='speedup'?value.toFixed(2)+'×':value.toFixed(activeMetric==='vec'?0:1)+(activeMetric==='vec'?' KB':' μs');
    const status=r.isRunning?' · 待采样':!r.buildOk||!r.testsOk?' · 失败':r.id==='#11'?' · 采纳':'';
    const g=svgEl('g',{transform:`translate(${x},${y})`,class:'canvas-node landscape-bubble',tabindex:0,role:'button','aria-label':`${r.id} ${r.shortLabel||''} ${reading}${status}${isPrunedRun(r)?' · 路径已剪枝':''}`,'aria-pressed':String(selected),opacity:dim?.22:isPrunedRun(r)?.5:1},root);
    const fresh=!seenBubbles.has(r.id);seenBubbles.add(r.id);
    const core=svgEl('g',{class:'bubble-core'+(fresh?' bubble-enter':''),style:`--bubble-delay:${Math.min(r.seq??0,12)*32}ms`},g);
    svgEl('circle',{r:radius,fill:runColor(r),stroke:'none',class:'bubble-disc'},core);
    const ink='var(--bubble-ink)';
    const mark=r.isRunning?' ?':!r.buildOk||!r.testsOk?' ×':r.id==='#11'?' ✓':'';
    stageText(core,0,-7,r.id+mark,{'text-anchor':'middle',fill:ink,'font-size':radius<34?11:13,'font-weight':650,'text-decoration':isPrunedRun(r)?'line-through':selected?'underline':'none',class:'bubble-id'});
    stageText(core,0,13,reading,{'text-anchor':'middle',fill:ink,'font-size':radius<34?10:12,'font-weight':550,class:'bubble-reading'});
    g.addEventListener('click',e=>selectStageRun(r,e));
    g.addEventListener('keydown',e=>{if(e.key==='Enter'||e.key===' '){e.preventDefault();selectStageRun(r,e);}});
    g.addEventListener('pointerenter',e=>{root.appendChild(g);showStageTip(r,e);});
    g.addEventListener('pointerleave',()=>{$('demoHoverTip').hidden=true;});
    g.addEventListener('focus',()=>root.appendChild(g));
  }
  function drawStageNode(root,r,p,{compact=false,dim=false}={}) {
    const selected=state.selectedCompareIds.includes(r.id)||(state.inspectorOpen&&state.selectedRunId===r.id);
    if (!compact) { drawLandscapeBubble(root,r,p,{selected,dim}); return; }
    const g=svgEl('g',{transform:`translate(${p.x},${p.y})`,class:'canvas-node',tabindex:0,role:'button','aria-label':`${r.id} ${r.shortLabel||''} ${numeric(r)?r.p0Latency+' μs':'无耗时读数'}${isPrunedRun(r)?' · 路径已剪枝':''}`,opacity:dim?.22:isPrunedRun(r)?.5:1},root);
    svgEl('circle',{r:compact?8:10,fill:runColor(r),stroke:'none'},g);
    if(!r.buildOk||!r.testsOk)stageText(g,0,4,'×',{'text-anchor':'middle',fill:'var(--ss-bg)','font-size':16});
    else if(r.id==='#11'&&!r.isRunning)stageText(g,0,4,'✓',{'text-anchor':'middle',fill:'var(--ss-bg)','font-size':13});
    else if(r.isRunning)stageText(g,0,4,'?',{'text-anchor':'middle',fill:'var(--ss-bg)','font-size':12});
    const near=Object.entries(stagePositions).filter(([id,q])=>id!==r.id && Math.abs(q.x-p.x)<12 && Math.abs(q.y-p.y)<76);
    const lower=near.some(([,q])=>q.y<p.y), upper=!lower&&near.length>0;
    stageText(g,0,compact?-16:lower?27:upper?-38:-20,r.id,{'text-anchor':'middle',fill:C.text,'font-size':12,'font-weight':600,'text-decoration':isPrunedRun(r)?'line-through':selected?'underline':'none'});
    if(!compact)stageText(g,0,lower?44:upper?-22:29,measure(r,activeMetric)==null?'无读数':activeMetric==='speedup'?measure(r,activeMetric).toFixed(2)+'×':measure(r,activeMetric).toFixed(activeMetric==='vec'?0:1)+(activeMetric==='vec'?' KB':' μs'),{'text-anchor':'middle','font-size':11});
    g.addEventListener('click',e=>selectStageRun(r,e));
    g.addEventListener('keydown',e=>{if(e.key==='Enter'||e.key===' '){e.preventDefault();selectStageRun(r,e);}});
    g.addEventListener('pointerenter',e=>showStageTip(r,e));g.addEventListener('pointerleave',()=>{$('demoHoverTip').hidden=true;});
  }
  function showStageTip(r,e) {
    const tip=$('demoHoverTip');tip.textContent=`${r.id} · ${r.shortLabel||r.title} · ${numeric(r)?r.p0Latency.toFixed(1)+' μs':'未获得耗时读数'}`;
    tip.hidden=false;tip.style.left=Math.min(e.clientX+14,innerWidth-310)+'px';tip.style.top=Math.min(e.clientY+14,innerHeight-70)+'px';
  }
  function renderOpenStage() {
    const svg=$('stageSvg'), scene=$('scene');if(!svg||!scene)return;
    const w=innerWidth,h=innerHeight;stageSize={w,h};svg.setAttribute('viewBox',`0 0 ${w} ${h}`);scene.innerHTML='';stagePositions={};
    $('canvasMain').dataset.scheme=state.canvasScheme;
    $('bullseyeCornerHud').hidden=state.canvasScheme!=='bullseye';
    $('bullseyeGateBar').hidden=state.canvasScheme!=='bullseye';
    if(state.canvasScheme==='landscape')drawLandscape(scene,w,h);
    else if(state.canvasScheme==='bullseye')drawBullseye(scene,w,h);
    else drawSwimlane(scene,w,h);
    updateStageCamera();renderStageControls();renderCompareFloatBar();
    $('canvasMain').classList.toggle('is-marquee-mode',state.multiSelectMode);
  }
  function plotBounds(w,h) {
    return {left:w<700?58:92,right:w-(w<1000?160:310),top:w<700?240:230,bottom:Math.max(340,h-265)};
  }
  function drawLandscape(root,w,h) {
    const b=plotBounds(w,h), maxRound=Math.max(7,...state.runs.map(r=>r.round??0));
    const measured=state.runs.map(r=>measure(r,activeMetric)).filter(v=>v!=null);
    const target=activeMetric==='vec'?state.contract.vecLimitKB:activeMetric==='speedup'?482/D.getTargetLatencyUs(state.contract):D.getTargetLatencyUs(state.contract);
    let min=activeMetric==='speedup'?.8:0,max=activeMetric==='vec'?Math.max(256,...measured)*1.05:activeMetric==='speedup'?Math.max(2.6,...measured)*1.04:Math.max(520,...measured)*1.03;
    const y=v=>activeMetric==='speedup'?b.bottom-(v-min)/(max-min)*(b.bottom-b.top):b.top+(v-min)/(max-min)*(b.bottom-b.top);
    const x=round=>b.left+(b.right-b.left)*round/maxRound;
    const ticks=activeMetric==='vec'?[0,64,128,192,256]:activeMetric==='speedup'?[1,1.5,2,2.5]:[0,100,200,300,400,500];
    stageText(root,b.left,b.top-28,activeMetric==='vec'?'片上 Vec 占用 · KB':activeMetric==='speedup'?'相对 #0 加速比 · ×':'端到端耗时 · μs',{fill:C.text,'font-size':12});
    ticks.forEach(v=>{stageLine(root,b.left,y(v),b.right,y(v),{'stroke-dasharray':'2 6',opacity:.5});stageText(root,b.left-14,y(v)+4,activeMetric==='speedup'?v.toFixed(1):String(v),{'text-anchor':'end'});});
    for(let round=0;round<=maxRound;round++) {
      stageLine(root,x(round),b.top,x(round),b.bottom,{opacity:.25});
      stageText(root,x(round),b.bottom+93,round===0?'R0 基线':`Round ${round}`,{'text-anchor':'middle'});
    }
    stageLine(root,b.left,y(target),b.right,y(target),{stroke:activeMetric==='vec'?C.regress:C.adopted,'stroke-dasharray':'6 6',opacity:.7});
    stageText(root,b.left+12,y(target)-12,activeMetric==='vec'?`容量上限 ${target} KB`:`${state.contract.version} 耗时目标 ${D.getTargetLatencyUs(state.contract).toFixed(1)} μs · ${(482/D.getTargetLatencyUs(state.contract)).toFixed(2)}×`,{'text-anchor':'start',fill:activeMetric==='vec'?C.regress:C.adopted});
    state.runs.forEach(r=>{const v=measure(r,activeMetric);stagePositions[r.id]={x:x(r.round??0),y:v==null?b.bottom+23:y(v),missing:v==null};});
    placeLandscapeBubbles(b);
    const stall=['#6','#8','#9'].map(id=>stagePositions[id]).filter(Boolean);
    if(stall.length===3) {
      const top=Math.min(...stall.map(p=>p.y))-43, bottom=Math.max(...stall.map(p=>p.y))+43;
      const left=stall[0].x-38,right=stall[2].x+38;
      svgEl('path',{d:`M ${left} ${(top+bottom)/2} C ${left} ${top-10}, ${left+48} ${top}, ${(left+right)/2} ${top} S ${right} ${top-8}, ${right} ${(top+bottom)/2} S ${right-48} ${bottom}, ${(left+right)/2} ${bottom} S ${left} ${bottom+10}, ${left} ${(top+bottom)/2}`,fill:'var(--ss-stall-fill)',stroke:'none'},root);
      stageText(root,left+18,Math.min(top-8,...['#6','#8','#9'].map(id=>stagePositions[id].bubbleY-bubbleRadius(state.runs.find(r=>r.id===id))-13)),'停滞 · #6 / #8 / #9 样本包络重叠',{fill:'var(--ss-warning)','font-size':10});
    }
    state.runs.forEach(r=>{
      const a=stagePositions[r.parentId],p=stagePositions[r.id];if(!a)return;
      svgEl('path',{d:`M ${a.x} ${a.y} C ${(a.x+p.x)/2} ${a.y}, ${(a.x+p.x)/2} ${p.y}, ${p.x} ${p.y}`,fill:'none',stroke:runColor(r),'stroke-width':1.7,'stroke-dasharray':p.missing?'4 5':'none',opacity:.72},root);
    });
    state.runs.forEach(r=>{
      const p=stagePositions[r.id];
      if(numeric(r)&&r.latencyRangeUs && activeMetric!=='vec') {
        const range=r.latencyRangeUs.map(v=>activeMetric==='speedup'?482/v:v).map(y);
        stageLine(root,p.x,range[0],p.x,range[1],{stroke:runColor(r),'stroke-width':3,opacity:.4});
        range.forEach(yy=>stageLine(root,p.x-5,yy,p.x+5,yy,{stroke:runColor(r),opacity:.6}));
      }
      drawStageNode(root,r,p);
    });
    // Callout labels are recomputed from terminal coordinates after metric or Fork changes.
    const ends=[['#7','内存与 Ring · 片上驻留复用'],['#9','InCore 调优 · 双缓冲与切分'],['#10','编排与派发 · 分核细化'],['#11','编排与派发 · 尾块快路径'],...state.runs.filter(r=>r.isCustom).map(r=>[r.id,`${D.AXIS_META[r.axis]?.name||'定向'} · 待采样分支`])].filter(([id])=>stagePositions[id]);
    const ordered=ends.map(([id,label])=>({id,label,p:stagePositions[id]})).sort((a,c)=>a.p.y-c.p.y);
    const gap=44;let prior=b.top-48;
    ordered.forEach((a,i)=>{a.labelY=Math.max(a.p.y,prior+gap);prior=a.labelY;});
    if(ordered.length && ordered.at(-1).labelY>b.bottom+30){const shift=ordered.at(-1).labelY-(b.bottom+30);ordered.forEach(a=>a.labelY-=shift);}
    ordered.forEach(a=>{
      const lx=b.right+62,ly=a.labelY,run=state.runs.find(r=>r.id===a.id),startX=(a.p.bubbleX??a.p.x)+bubbleRadius(run)+4,startY=a.p.bubbleY??a.p.y;
      svgEl('path',{d:`M ${startX} ${startY} L ${Math.max(startX+8,b.right+44)} ${ly} H ${lx+8}`,fill:'none',stroke:C.muted,'stroke-width':.8,opacity:.65},root);
      stageText(root,lx+14,ly+4,a.id+' '+(w<1400?a.label.split(' · ')[0]:a.label)+(isPrunedRun(state.runs.find(r=>r.id===a.id))?' · 已剪枝':''),{'font-size':w<1000?9:10});
    });
    $('canvasLegend').innerHTML='<span style="color:var(--primary)">●</span> 正常演进　<span style="color:var(--ss-pass)">●</span> 最终采纳　<span style="color:var(--ss-anomaly)">●</span> 性能回退 / 违例　<span style="color:var(--ss-warning)">▬</span> 停滞<br><small>气泡大小随实验编号递增 · 细点 / 引线对应原始坐标 · min–max 样本范围</small>';
  }
  function drawBullseye(root,w,h) {
    const cx=w*.49, cy=Math.max(360,(h+110)/2), R=Math.max(130,Math.min(w*.29,(h-320)/2));
    const radius=us=>R*us/482;
    [100,D.getTargetLatencyUs(state.contract),300,400,482].forEach(us=>{
      svgEl('circle',{cx,cy,r:radius(us),fill:'none',stroke:us===D.getTargetLatencyUs(state.contract)?C.adopted:C.line,'stroke-width':1,'stroke-dasharray':us===D.getTargetLatencyUs(state.contract)?'6 5':'none',opacity:.65},root);
      stageText(root,cx+radius(us)+12,cy-8,`${us.toFixed(0)} μs`,{fill:us===D.getTargetLatencyUs(state.contract)?C.adopted:C.muted});
    });
    stageLine(root,cx-R-24,cy,cx+R+24,cy,{opacity:.6});stageLine(root,cx,cy-R-24,cx,cy+R+24,{opacity:.6});
    const names=[[-1,-1,'InCore 调优'],[1,-1,'内存与 Ring'],[1,1,'编排与派发'],[-1,1,'Host 常驻 · 无实验']];
    names.forEach(([dx,dy,label])=>stageText(root,cx+dx*(R+25),cy+dy*(R*.72),label,{'text-anchor':dx<0?'end':'start',fill:C.text,'font-size':12}));
    stageText(root,cx,cy-5,'耗时原点',{'text-anchor':'middle','font-size':10});stageText(root,cx,cy+13,'0 μs',{'text-anchor':'middle','font-size':10});
    const byAxis={};state.runs.forEach(r=>(byAxis[r.id==='#0'?'baseline':r.axis||'baseline'] ||= []).push(r));
    Object.entries(byAxis).forEach(([axis,runs])=>runs.forEach((r,i)=>{
      const angle=(axis==='incore'?205+i*12:axis==='memory'?310+i*14:axis==='dispatch'?30+i*20:axis==='human'?65:180)*Math.PI/180;
      const rr=numeric(r)?radius(r.p0Latency):R+42;
      stagePositions[r.id]={x:cx+Math.cos(angle)*rr,y:cy+Math.sin(angle)*rr,missing:!numeric(r)};
    }));
    state.runs.forEach(r=>{const p=stagePositions[r.id],a=stagePositions[r.parentId];if(a)stageLine(root,a.x,a.y,p.x,p.y,{stroke:runColor(r),opacity:.4,'stroke-dasharray':numeric(r)?'none':'4 4'});});
    state.runs.forEach(r=>{
      const dim=state.bullseyeGateFilter!==null && D.evaluateRun(r,state.contract).dims[D.GATE_KEYS[state.bullseyeGateFilter]]==='pass';
      drawStageNode(root,r,stagePositions[r.id],{compact:true,dim});
    });
    stageText(root,cx,cy+R+60,`径向 = 耗时 / 482 μs · 虚线圈 = ${state.contract.version} 耗时目标 · 门禁结果独立核查`,{'text-anchor':'middle','font-size':10});
    $('bullseyeGateBar').innerHTML=D.GATE_LABELS.map((label,i)=>`<button class="cap-tab ${state.bullseyeGateFilter===i?'is-active':''}" data-filter-gate="${i}" aria-pressed="${state.bullseyeGateFilter===i}">${label}</button>`).join('')+`<button class="cap-tab" data-filter-gate="all">全部</button>`;
    drawGateRadar();
    $('canvasLegend').innerHTML='径向越近，耗时越低<br><small>耗时达标不代表六项门禁通过 · 轴外 ? 无耗时样本</small>';
  }
  function drawGateRadar() {
    const svg=$('bsRadarSvg');svg.innerHTML='';
    const ids=state.selectedCompareIds.length?state.selectedCompareIds.slice(-3):['#0',state.selectedRunId];
    const cx=114,cy=88,R=58,pt=(v,i)=>[cx+Math.cos(-Math.PI/2+i*Math.PI/3)*R*v,cy+Math.sin(-Math.PI/2+i*Math.PI/3)*R*v];
    svgEl('polygon',{points:D.GATE_KEYS.map((_,i)=>pt(1,i).join(',')).join(' '),fill:'none',stroke:C.line},svg);
    D.GATE_LABELS.forEach((v,i)=>{const p=pt(1.3,i);stageText(svg,p[0],p[1]+3,v,{'text-anchor':'middle','font-size':9});});
    ids.forEach((id,j)=>{const r=state.runs.find(r=>r.id===id);if(!r)return;const dims=D.evaluateRun(r,state.contract).dims;
      svgEl('polygon',{points:D.GATE_KEYS.map((key,i)=>pt(r.isRunning?.15:dims[key]==='pass'?1:dims[key]==='fail'?.32:.6,i).join(',')).join(' '),fill:'none',stroke:j?runColor(r):C.muted,'stroke-dasharray':j?'none':'4 4','stroke-width':1.5},svg);
    });
    $('bsRadarLegend').textContent=[...new Set(ids)].join(' / ')+' · 通过 1 / 失败 .32 / 待评 .6';
    $('bsCornerTitle').textContent='六项门禁状态轮廓';
  }
  function drawSwimlane(root,w,h) {
    const left=w<900?140:205,right=w-74,top=w<700?240:218,bottom=h-225,maxRound=Math.max(7,...state.runs.map(r=>r.round??0));
    const failed=state.runs.filter(r=>!r.buildOk);
    const rowIds=['SPINE',...state.lanes.map(l=>l.id),...(failed.length?['compile-blocked']:[])], rowH=Math.max(38,Math.min(62,(bottom-top)/rowIds.length));
    const x=round=>left+(right-left)*round/maxRound,y=id=>top+rowIds.indexOf(id)*rowH;
    const bandW=(right-left)/maxRound;
    for(let round=0;round<=maxRound;round++) {
      if(round%2)svgEl('rect',{x:x(round)-bandW/2,y:top-36,width:bandW,height:rowH*rowIds.length+130,fill:'var(--ss-text)',opacity:.025},root);
      stageText(root,x(round),top-27,round===0?'R0 基线':`Round ${round}`,{'text-anchor':'middle'});
    }
    rowIds.forEach((id,i)=>{
      const lane=state.lanes.find(l=>l.id===id),label=id==='compile-blocked'?'编译拦截':(lane?.name||'采纳主干')+(lane?.status==='pruned'?' · 已剪枝':'');
      stageLine(root,left,y(id),right,y(id),{opacity:.3});
      const labelEl=stageText(root,left-24,y(id)+4,label,{'text-anchor':'end','font-size':11,fill:lane?.status==='pruned'?C.muted:C.text});
      if(lane){labelEl.setAttribute('role','button');labelEl.setAttribute('tabindex',0);labelEl.style.cursor='pointer';labelEl.onclick=()=>openLaneControls(lane);labelEl.onkeydown=e=>{if(e.key==='Enter')openLaneControls(lane);};}
      if(lane && D.AXIS_META[lane.axis]?.glyphPath)svgEl('path',{d:D.AXIS_META[lane.axis].glyphPath,transform:`translate(${Math.max(12,left-195)},${y(id)-10}) scale(.7)`,fill:'none',stroke:C.muted,'stroke-width':1.5},root);
    });
    state.runs.forEach(r=>stagePositions[r.id]={x:x(r.round??0),y:y(!r.buildOk?'compile-blocked':rowIds.includes(r.laneId)?r.laneId:'SPINE')});
    state.runs.forEach(r=>{
      const p=stagePositions[r.id],a=stagePositions[r.parentId];if(!a)return;
      const d=a.y===p.y?`M ${a.x+9} ${a.y} H ${p.x-9}`:`M ${a.x+9} ${a.y} H ${(a.x+p.x)/2} V ${p.y} H ${p.x-9}`;
      svgEl('path',{d,fill:'none',stroke:runColor(r),'stroke-width':1.5,opacity:.65,'stroke-dasharray':r.isRunning?'4 5':'none'},root);
    });
    state.runs.forEach(r=>drawStageNode(root,r,stagePositions[r.id],{compact:true}));
    const adopted=state.runs.find(r=>r.id==='#11');if(adopted){const p=stagePositions[adopted.id];stageLine(root,p.x,p.y,p.x,y('SPINE'),{stroke:C.adopted,'stroke-dasharray':'4 4'});stageText(root,p.x+12,y('SPINE')+4,'#11 采纳',{fill:C.adopted});}
    const teleY=top+rowH*rowIds.length+26;
    stageText(root,left-24,teleY,'耗时 μs',{'text-anchor':'end'});stageText(root,left-24,teleY+52,'门禁 通过 / 6',{'text-anchor':'end'});
    // Every telemetry mark retains its experiment's exact Wave x coordinate.
    state.runs.forEach(r=>{
      const xx=x(r.round??0);if(numeric(r)) {const yy=teleY+24-r.p0Latency/482*30;svgEl('circle',{cx:xx,cy:yy,r:3,fill:runColor(r)},root);}
      
      const sameWave=state.runs.filter(t=>(t.round??0)===(r.round??0)).sort((a,b)=>a.seq-b.seq),rank=sameWave.indexOf(r);
      stageText(root,xx,teleY-32+rank*13,r.id+': '+(numeric(r)?r.p0Latency.toFixed(1):'—'),{'font-size':9,'text-anchor':'middle'});
      const n=r.isRunning?null:Object.values(D.evaluateRun(r,state.contract).dims).filter(v=>v==='pass').length;
      stageText(root,xx,teleY+52+rank*13,r.id+' '+(n==null?'待评':n+'/6'),{'text-anchor':'middle','font-size':9,fill:runColor(r)});
    });
    $('canvasLegend').innerHTML='实线：实验派生　虚线：候选 / 采纳主干<br><small>波次为离散序列，非设备时间 · 节点与遥测共用波次轴</small>';
  }
  function openLaneControls(lane) {
    openGenericModal(lane.name,`<p>${escapeHtml(lane.hypothesis||'')}</p><p>状态：${escapeHtml(lane.status)}</p><button class="btn" id="lanePauseAction">${lane.status==='paused'?'恢复路径':'暂停路径'}</button><button class="btn" id="lanePruneAction">剪枝止损</button><button class="btn" id="laneBudgetAction">调整预算</button>`,false);
    $('lanePauseAction').onclick=()=>{lane.status=lane.status==='paused'?'active':'paused';$('genericModalBackdrop').hidden=true;renderAll();};
    $('lanePruneAction').onclick=()=>{$('genericModalBackdrop').hidden=true;executePruneLane(lane.id,'人工止损');};
    $('laneBudgetAction').onclick=()=>{$('genericModalBackdrop').hidden=true;navigateToContract();};
  }
  function updateStageCamera() {
    const cam=stageCameras[state.canvasScheme];$('scene').setAttribute('transform',`translate(${cam.x},${cam.y}) scale(${cam.k})`);$('zoomReadout').textContent=Math.round(cam.k*100)+'%';
  }
  function zoomStage(factor,point={x:innerWidth/2,y:innerHeight/2}) {
    const cam=stageCameras[state.canvasScheme],next=Math.max(.35,Math.min(4,cam.k*factor)),ratio=next/cam.k;
    cam.x=point.x-(point.x-cam.x)*ratio;cam.y=point.y-(point.y-cam.y)*ratio;cam.k=next;updateStageCamera();
  }
  function renderStageControls() {
    const isLandscape=state.canvasScheme==='landscape';$('canvasCtxDropdown').hidden=!isLandscape;
    const labels={latency:'耗时 · μs',vec:'Vec 占用 · KB',speedup:'加速比 · ×'};
    $('canvasCtxValue').textContent=labels[activeMetric];
    $('canvasCtxMenu').innerHTML=Object.entries(labels).map(([key,label])=>`<button role="option" aria-selected="${activeMetric===key}" class="config-option ${activeMetric===key?'is-active':''}" data-stage-metric="${key}">${label}${activeMetric===key?' ✓':''}</button>`).join('');
  }
  function wireStaticEvents() {
    const on=(id,event,fn)=>{if($(id))$(id).addEventListener(event,fn);};
    document.addEventListener('click',e=>{
      const view=e.target.closest('[data-view]');if(view){state.activeView=view.dataset.view;state.inspectorOpen=false;if(state.activeView==='contract')state.contractDraft=cloneContract(state.contract);renderAll();}
      const scheme=e.target.closest('[data-canvas-scheme]');if(scheme){state.canvasScheme=scheme.dataset.canvasScheme;renderAll();}
      const act=e.target.closest('[data-act]');if(act){stopPlayback();loadActSnapshot(Number(act.dataset.act),{keepView:true});renderAll();}
      const group=e.target.closest('[data-goto-group]');if(group)navigateToContract(group.dataset.gotoGroup);
      const metric=e.target.closest('[data-stage-metric]');if(metric){activeMetric=metric.dataset.stageMetric;$('canvasCtxMenu').setAttribute('aria-hidden','true');$('canvasCtxTrigger').setAttribute('aria-expanded','false');renderOpenStage();}
      const gate=e.target.closest('[data-filter-gate]');if(gate){const val=gate.dataset.filterGate;state.bullseyeGateFilter=val==='all'||state.bullseyeGateFilter===Number(val)?null:Number(val);renderOpenStage();}
      const action=e.target.closest('[data-action]');if(action)handleAction(action.dataset.action);
      const quick=e.target.closest('[data-quick-cmd]');if(quick){$('composerInput').value=quick.dataset.quickCmd;updateComposerSendVisual();}
      if(!e.target.closest('#canvasCtxDropdown')){$('canvasCtxMenu').setAttribute('aria-hidden','true');$('canvasCtxTrigger').setAttribute('aria-expanded','false');}
    });
    on('canvasCtxTrigger','click',()=>{const open=$('canvasCtxTrigger').getAttribute('aria-expanded')!=='true';$('canvasCtxTrigger').setAttribute('aria-expanded',String(open));$('canvasCtxMenu').setAttribute('aria-hidden',String(!open));if(open)$('canvasCtxMenu').querySelector('button')?.focus();});
    on('canvasCtxMenu','keydown',e=>{const buttons=[...$('canvasCtxMenu').querySelectorAll('button')],i=buttons.indexOf(document.activeElement);if(['ArrowDown','ArrowUp'].includes(e.key)){e.preventDefault();buttons[(i+(e.key==='ArrowDown'?1:buttons.length-1))%buttons.length].focus();}});
    on('themeToggleBtn','click',()=>{state.theme=state.theme==='dark'?'light':'dark';renderShell();});
    on('playbackBtn','click',togglePlayback);
    ['stepPrevBtn','stepNextBtn'].forEach((id,i)=>on(id,'click',()=>{stopPlayback();loadActSnapshot(Math.max(0,Math.min(4,state.act+(i?1:-1))),{keepView:true});renderAll();}));
    on('multiSelectToggleBtn','click',()=>{state.multiSelectMode=!state.multiSelectMode;renderShell();renderOpenStage();});
    on('compareSelectedGoBtn','click',()=>{state.diffRunIds=[...state.selectedCompareIds];state.activeView='diff';state.inspectorOpen=false;renderAll();});
    on('compareSelectedClearBtn','click',()=>{state.selectedCompareIds=[];renderOpenStage();renderCompareFloatBar();});
    on('zoomInBtn','click',()=>zoomStage(1.2));on('zoomOutBtn','click',()=>zoomStage(1/1.2));on('zoomFitBtn','click',()=>{Object.assign(stageCameras[state.canvasScheme],{x:0,y:0,k:1});renderOpenStage();});
    on('composerHeaderToggle','click',()=>{state.threadOpen=!state.threadOpen;renderComposerDock();});
    on('composerSettingsBtn','click',()=>navigateToContract());
    on('composerAttachBtn','click',()=>{const menu=$('composerAttachMenu');menu.hidden=!menu.hidden;$('composerAttachBtn').setAttribute('aria-expanded',String(!menu.hidden));});
    document.addEventListener('click',e=>{
      const attach=e.target.closest('[data-attach-context]');
      if(attach){const contract=attach.dataset.attachContext==='contract';addComposerChip({id:contract?'active-contract':'node-'+state.selectedRunId,label:contract?'契约 '+state.contract.version:'实验 '+state.selectedRunId});}
      const remove=e.target.closest('[data-remove-context]');if(remove){state.attachedChips=state.attachedChips.filter(ch=>ch.id!==remove.dataset.removeContext);renderComposerDock();}
      if(!e.target.closest('.agent-attach-wrap')){$('composerAttachMenu').hidden=true;$('composerAttachBtn').setAttribute('aria-expanded','false');}
      if(attach){$('composerAttachMenu').hidden=true;$('composerAttachBtn').setAttribute('aria-expanded','false');$('composerInput').focus();}
    });
    on('composerInput','input',updateComposerSendVisual);on('composerForm','submit',e=>{e.preventDefault();const value=$('composerInput').value;$('composerInput').value='';handleComposerSubmit(value);});
    const panel=(id,button)=>{const el=$(id);el.hidden=!el.hidden;$(button)?.setAttribute('aria-expanded',String(!el.hidden));};
    on('inspectorToggle','click',()=>{state.inspectorOpen=!state.inspectorOpen;renderInspectorPopover();});
    on('auditToggle','click',()=>{const diag=D.computeDiagnostics(state.runs,state.lanes,state.contract,state.act);$('auditCardsContainer').innerHTML=diag.exceptions.length?diag.exceptions.map(exc=>`<article class="audit-decision"><h3>${escapeHtml(exc.signal)}</h3><p>${escapeHtml(exc.advice)}</p><button class="btn" data-action="${exc.inspectAction}">${escapeHtml(exc.inspectLabel)}</button><button class="btn" data-action="${exc.actionType}">${escapeHtml(exc.actionLabel)}</button></article>`).join(''):'<p>当前候选未发现契约违例。</p>';$('auditPanelSub').textContent=diag.exceptions.length+' 项待核查';panel('auditPanel','auditToggle');});
    on('closeAuditPanelBtn','click',()=>{$('auditPanel').hidden=true;$('auditToggle').setAttribute('aria-expanded','false');});
    on('infoToggle','click',()=>panel('infoPanel','infoToggle'));on('closeInfoPanelBtn','click',()=>{$('infoPanel').hidden=true;});
    on('sourceToggle','click',()=>{state.inspectorOpen=true;renderInspectorPopover();$('inspectorPanel').querySelector('details').open=true;});
    on('bsCornerToggleBtn','click',()=>{$('bsCornerBody').hidden=!$('bsCornerBody').hidden;$('bsCornerToggleBtn').textContent=$('bsCornerBody').hidden?'展开':'收起';});
    on('usageOpen','click',()=>$('usageWindow').showModal());on('usageClose','click',()=>$('usageWindow').close());
    document.querySelectorAll('.prompt-copy').forEach(b=>b.onclick=async()=>{try{await navigator.clipboard.writeText(b.parentElement.querySelector('pre').textContent);$('usageCopyStatus').textContent='已复制';}catch{$('usageCopyStatus').textContent='请选中文字复制';}});
    on('proposalOpenBtn','click',()=>openGenericModal('开放舞台设计与证据约定',D.PROPOSAL_HTML,false));on('openLedgerBtn','click',openAuditDrawer);
    const closes={closeAuditDrawerBtn:'auditDrawerBackdrop',closeForkModalBtn:'forkModalBackdrop',cancelForkModalBtn:'forkModalBackdrop',closeCodeModalBtn:'codeModalBackdrop',cancelCodeModalBtn:'codeModalBackdrop',closeGenericModalBtn:'genericModalBackdrop'};
    Object.entries(closes).forEach(([id,target])=>on(id,'click',()=>{$(target).hidden=true;}));on('closeToolModalBtn','click',closeToolModal);
    document.addEventListener('keydown',e=>{if(e.key!=='Escape')return;document.querySelectorAll('.modal-backdrop,.floating-panel').forEach(el=>el.hidden=true);state.inspectorOpen=false;state.threadOpen=false;renderComposerDock();$('canvasCtxMenu').setAttribute('aria-hidden','true');$('canvasCtxTrigger').setAttribute('aria-expanded','false');state.bullseyeGateFilter=null;renderOpenStage();});
    wireStageGestures();
    // One tooltip owner; strip native titles on existing and newly rendered fragments.
    const strip=()=>document.querySelectorAll('[title]').forEach(el=>{if(!el.dataset.tip)el.dataset.tip=el.getAttribute('title');el.removeAttribute('title');});
    new MutationObserver(strip).observe(document.body,{childList:true,subtree:true});strip();
    const tip=$('demoHoverTip');
    document.addEventListener('pointerover',e=>{const el=e.target.closest('[data-tip]');if(!el)return;tip.textContent=el.dataset.tip;tip.hidden=false;const box=el.getBoundingClientRect();tip.style.left=Math.min(box.left,innerWidth-320)+'px';tip.style.top=(box.bottom+8>innerHeight-60?box.top-48:box.bottom+8)+'px';});
    document.addEventListener('pointerout',e=>{if(e.target.closest('[data-tip]'))tip.hidden=true;});
    document.addEventListener('focusin',e=>{const el=e.target.closest('[data-tip]');if(el){const b=el.getBoundingClientRect();tip.textContent=el.dataset.tip;tip.hidden=false;tip.style.left=Math.min(b.left,innerWidth-320)+'px';tip.style.top=b.bottom+8+'px';}});document.addEventListener('focusout',()=>tip.hidden=true);
    window.addEventListener('resize',()=>{if(state.activeView==='canvas')renderOpenStage();});
  }
  function wireStageGestures() {
    const svg=$('stageSvg');let gesture=null;
    svg.addEventListener('wheel',e=>{e.preventDefault();zoomStage(Math.exp(-e.deltaY*.001),{x:e.clientX,y:e.clientY});},{passive:false});
    svg.addEventListener('pointerdown',e=>{if(e.target.closest('.canvas-node')||e.button!==0)return;const cam=stageCameras[state.canvasScheme];gesture={x:e.clientX,y:e.clientY,cx:cam.x,cy:cam.y,select:e.shiftKey||state.multiSelectMode};svg.setPointerCapture(e.pointerId);});
    svg.addEventListener('pointermove',e=>{if(!gesture)return;const cam=stageCameras[state.canvasScheme];if(gesture.select){$('screenOverlay').innerHTML='';svgEl('rect',{x:Math.min(e.clientX,gesture.x),y:Math.min(e.clientY,gesture.y),width:Math.abs(e.clientX-gesture.x),height:Math.abs(e.clientY-gesture.y),class:'marquee-box'},$('screenOverlay'));}else{cam.x=gesture.cx+e.clientX-gesture.x;cam.y=gesture.cy+e.clientY-gesture.y;updateStageCamera();}});
    svg.addEventListener('pointerup',e=>{if(!gesture)return;if(gesture.select){const cam=stageCameras[state.canvasScheme],left=Math.min(e.clientX,gesture.x),right=Math.max(e.clientX,gesture.x),top=Math.min(e.clientY,gesture.y),bottom=Math.max(e.clientY,gesture.y);const hits=state.runs.filter(r=>{const p=stagePositions[r.id],x=(p.bubbleX??p.x)*cam.k+cam.x,y=(p.bubbleY??p.y)*cam.k+cam.y;return numeric(r)&&r.buildOk&&r.testsOk&&x>=left&&x<=right&&y>=top&&y<=bottom;});state.selectedCompareIds=hits.slice(-4).map(r=>r.id);$('screenOverlay').innerHTML='';renderOpenStage();renderCompareFloatBar();}gesture=null;});
    svg.addEventListener('pointercancel',()=>{gesture=null;$('screenOverlay').innerHTML='';});
  }

  // ---------- Boot ----------
  wireStaticEvents();
  loadActSnapshot(4, { defaultView: 'canvas' });
  renderAll();
})();
