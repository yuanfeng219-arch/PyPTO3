/* PyPTO Studio — CASE 回放演示
 * 纯前端状态机：单一 state 驱动 IDE VIEW 与 Agent Windows 双视图。
 * 所有事实数据来自 index.html 内嵌的 demo-spec.json（<script id="demo-spec">）。
 * 无外部网络依赖，直接双击 index.html 即可运行。
 */
(function () {
  'use strict';

  // ---------------------------------------------------------------------
  // DOM helpers
  // ---------------------------------------------------------------------
  const $ = (sel, root) => (root || document).querySelector(sel);
  const $$ = (sel, root) => Array.from((root || document).querySelectorAll(sel));

  // ---------------------------------------------------------------------
  // Spec
  // ---------------------------------------------------------------------
  const SPEC = JSON.parse(document.getElementById('demo-spec').textContent);
  const SCENARIOS = SPEC.scenarios;
  const UI = SPEC.ui;
  const MODEL = SPEC.model;
  const AGENTS = MODEL.agents;
  const PHASES = MODEL.phases;

  // V2 presentation metadata keeps replay labels and progressive disclosure
  // separate from the final archived case summary.
  const SCENARIO_UI = {
    correctness: {
      taskId: 'T-2040',
      replayTitle: 'decode_layer logits 全零输出定界',
      inquiry: '首个异常出现在 post_rms_reduce、silu，还是 rms_lm_head？',
      baselineRun: '7c31e2a',
      candidateRun: '4d9a1c6',
      firstAnomalyAt: 's7',
      rootCauseAt: 's8',
      fixAt: 's9',
      evidenceAt: { e1: 's1', e2: 's1', e3: 's1', e4: 's6', e5: 's7', e6: 's7', e7: 's8', e8: 's8' },
      impactAt: { s5: 1, s6: 2, s7: 5, s8: 6 },
    },
    perf: {
      taskId: 'T-2041',
      replayTitle: 'decode_layer 406 秒设备占用归因',
      inquiry: '时间主要消耗在 Queue、Host，还是 Device？',
      baselineRun: 'perf-406s',
      candidateRun: 'golden-8t',
      firstAnomalyAt: 's6',
      rootCauseAt: 's8',
      fixAt: 's9',
      evidenceAt: { e1: 's4', e2: 's6', e3: 's8', e4: 's7', e5: 's7', e6: 's10' },
      impactAt: { s5: 1, s6: 4, s7: 5, s8: 6 },
    },
    runtime: {
      taskId: 'T-2042',
      replayTitle: 'fa_fused timeout 后续错误风暴定界',
      inquiry: '后续错误是独立失败，还是共享 Worker 被前一次失败污染？',
      baselineRun: 'rt-failed',
      candidateRun: 'rt-recovery',
      firstAnomalyAt: 's7',
      rootCauseAt: 's7',
      fixAt: 's8',
      evidenceAt: { e1: 's3', e2: 's3', e3: 's6', e4: 's7', e5: 's8', e6: 's9' },
      impactAt: { s4: 1, s5: 1, s6: 3, s7: 5 },
    },
    schedule: {
      taskId: 'INV-2040',
      replayTitle: 'PyPTO vs CCE 2.3× 差距归因',
      inquiry: '2.3× 差距发生在核内算力、GM transport，还是调度依赖距离？',
      baselineRun: 'cce-20260723',
      candidateRun: 'distance-2',
      firstAnomalyAt: 's5',
      rootCauseAt: 's7',
      fixAt: 's8',
      evidenceAt: { e1: 's1', e2: 's2', e3: 's4', e4: 's4', e5: 's7', e6: 's8', e7: 's9' },
      impactAt: { s5: 1, s6: 2, s7: 5 },
    },
  };

  const AGENT_SHORT_ROLE = {
    lead: '编排',
    triage: '分诊',
    correctness: '正确性',
    runtime: 'Runtime',
    perf: '性能',
  };

  const AGENT_AVATAR_SRC = {
    lead: './assets/agents/lead.png',
    triage: './assets/agents/triage.png',
    correctness: './assets/agents/correctness.png',
    runtime: './assets/agents/runtime.png',
    perf: './assets/agents/perf.png',
  };

  // ---------------------------------------------------------------------
  // State (双视图共享，切换视图不丢步骤/选中)
  // ---------------------------------------------------------------------
  const state = {
    mode: 'ide',           // 'ide' | 'agent'
    scenarioId: 'correctness',
    stepIndex: 0,
    playing: false,
    evidenceFilter: null,  // null | 'E0' | 'E1' | 'E2' | 'E3'
    selectedAgent: null,
    terminalOpen: true,    // IDE 终端收起/展开
    ideToolActive: null,   // IDE 工具面板当前激活工具 id
    toolView: 'tile',      // Agent 工具面板 'tile' | 'full'
    toolFullId: null,      // Agent 工具面板全屏工具 id
    leftOpen: true,        // 左侧面板开合
    rightOpen: true,       // 右侧面板开合
    lastStepByScenario: { correctness: 0, perf: 0, runtime: 0, schedule: 0 },
    confirmed: {},         // 'scenarioId/stepId' -> true（已人工确认）
    rejected: {},          // 'scenarioId/stepId' -> true（已驳回）
    toolCollapsed: {},     // 'toolId' -> true（tile 折叠）
    actOpen: {},           // 'scenarioId/stepId' -> 工具调用行是否展开
    chatTarget: null,      // 对话输入框当前 @ 的 agent（null = 跟随当前负责人）
    messages: { correctness: [], perf: [], runtime: [], schedule: [] }, // 人工插话 + agent 回复
  };
  let autoplayTimer = null;
  let toastTimer = null;

  const scenario = () => SCENARIOS.find((s) => s.scenarioId === state.scenarioId) || SCENARIOS[0];
  const steps = () => scenario().steps;
  const step = () => steps()[state.stepIndex] || steps()[0];
  const phaseIndex = (id) => PHASES.findIndex((p) => p.id === id);
  // MODEL.phases 只有 id/judge，label 在 UI.stepper.phases —— 兜底到 id，避免出现 undefined
  const UI_PHASES = (UI.stepper && UI.stepper.phases) || [];
  const phaseLabel = (p) => {
    const u = UI_PHASES.find((x) => x.id === p.id);
    return (u && u.label) || p.label || p.id;
  };
  const clamp = (i) => Math.max(0, Math.min(steps().length - 1, i));
  const scenarioUi = () => SCENARIO_UI[state.scenarioId] || SCENARIO_UI.correctness;
  const stepIndexById = (id) => steps().findIndex((s) => s.id === id);
  const isAtOrAfter = (id) => {
    const idx = stepIndexById(id);
    return idx >= 0 && state.stepIndex >= idx;
  };
  const displayTitle = () => isAtOrAfter('s10') ? scenario().title : scenarioUi().replayTitle;
  const knownFirstAnomaly = () => isAtOrAfter(scenarioUi().firstAnomalyAt) ? scenario().firstAnomaly : null;
  const knownRootCause = () => isAtOrAfter(scenarioUi().rootCauseAt) ? scenario().rootCause : null;
  const knownFix = () => isAtOrAfter(scenarioUi().fixAt) ? scenario().fix : null;
  const unlockedEvidence = () => (scenario().evidence || []).filter((e) => {
    const at = scenarioUi().evidenceAt[e.id];
    return !at || isAtOrAfter(at);
  });
  const impactRevealCount = () => {
    let count = 0;
    steps().slice(0, state.stepIndex + 1).forEach((s) => {
      if (scenarioUi().impactAt[s.id] != null) count = scenarioUi().impactAt[s.id];
    });
    return count;
  };
  const phaseQuestion = () => {
    const current = PHASES.find((p) => p.id === step().phase);
    return current ? current.judge : '';
  };

  const agentName = (id) => {
    const a = AGENTS.find((x) => x.id === id);
    return a ? a.id + ' · ' + a.name : (id || '-');
  };

  // ---------------------------------------------------------------------
  // Feedback
  // ---------------------------------------------------------------------
  function toast(msg) {
    const t = $('#demo-toast');
    t.textContent = msg;
    t.classList.add('is-visible');
    clearTimeout(toastTimer);
    toastTimer = setTimeout(() => t.classList.remove('is-visible'), 1700);
  }

  // ---------------------------------------------------------------------
  // DOM builders
  // ---------------------------------------------------------------------
  function el(tag, cls, text) {
    const n = document.createElement(tag);
    if (cls) n.className = cls;
    if (text != null) n.textContent = text;
    return n;
  }

  function agentAvatar(agentId, className) {
    const avatar = el('span', className + ' chat-team__avatar--' + agentId);
    const img = el('img', 'agent-avatar__image');
    img.src = AGENT_AVATAR_SRC[agentId] || AGENT_AVATAR_SRC.lead;
    img.alt = '';
    img.width = 64;
    img.height = 64;
    img.decoding = 'async';
    avatar.appendChild(img);
    return avatar;
  }

  // 小图标统一走 SVG，别再用 ▦ / ⛶ 这类字符 —— 各平台字形差异大且对不齐基线
  const ICON_PATHS = {
    grid: '<rect x="3" y="3" width="7.5" height="7.5" rx="1.5"/><rect x="13.5" y="3" width="7.5" height="7.5" rx="1.5"/><rect x="3" y="13.5" width="7.5" height="7.5" rx="1.5"/><rect x="13.5" y="13.5" width="7.5" height="7.5" rx="1.5"/>',
    expand: '<path d="M9 3H4.5A1.5 1.5 0 0 0 3 4.5V9"/><path d="M15 3h4.5A1.5 1.5 0 0 1 21 4.5V9"/><path d="M9 21H4.5A1.5 1.5 0 0 1 3 19.5V15"/><path d="M15 21h4.5a1.5 1.5 0 0 0 1.5-1.5V15"/>',
    collapse: '<path d="M5 9l7 7 7-7"/>',
    unfold: '<path d="M19 15l-7-7-7 7"/>',
  };

  function icon(name) {
    const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
    svg.setAttribute('viewBox', '0 0 24 24');
    svg.setAttribute('class', 'ui-icon');
    svg.setAttribute('aria-hidden', 'true');
    svg.innerHTML = ICON_PATHS[name] || '';
    return svg;
  }

  function badge(text, kind) {
    const b = el('span', 'demo-badge' + (kind ? ' demo-badge--' + kind : ''));
    b.textContent = text;
    return b;
  }

  const evidenceBadge = (level) => badge(level || 'E0', 'ev-' + (level || 'E0'));

  function gateBadge(gate) {
    const g = (!gate || gate === 'null') ? 'not-evaluated' : gate;
    return badge(g, 'gate-' + g);
  }

  function card(title, bodyNode, opts) {
    const c = el('section', 'demo-card' + (opts && opts.cls ? ' ' + opts.cls : ''));
    const h = el('header', 'demo-card__header');
    h.appendChild(el('span', 'demo-card__title', title));
    if (opts && opts.meta) h.appendChild(el('span', 'demo-card__meta', opts.meta));
    c.appendChild(h);
    c.appendChild(bodyNode);
    return c;
  }

  function mono(text) {
    const c = el('code', 'demo-mono');
    c.textContent = text;
    return c;
  }

  function field(label, value, isCode) {
    const f = el('div', 'demo-field-block');
    f.appendChild(el('span', 'demo-field-block__label', label));
    if (isCode) f.appendChild(mono(value));
    else f.appendChild(el('div', 'demo-field-block__value', value));
    return f;
  }

  // ---------------------------------------------------------------------
  // 人工确认卡（代码修改 / 签发基线）
  // ---------------------------------------------------------------------
  function effectiveConfirm(st) {
    if (st.confirmCard) return { kind: 'manual', text: st.confirmCard };
    if (st.phase === 'Deliver' && /签发/.test((st.title || '') + (st.conclusion || ''))) {
      return { kind: 'sign', text: st.title + '。' + st.conclusion };
    }
    return null;
  }

  const confirmKey = (st) => state.scenarioId + '/' + st.id;
  const isConfirmed = (st) => !!state.confirmed[confirmKey(st)];
  const isRejected = (st) => !!state.rejected[confirmKey(st)];

  // 当前步是否构成「未放行的人工门禁」
  function isGateBlocking(st) {
    return !!effectiveConfirm(st) && !isConfirmed(st);
  }

  // 从当前位置到 target 之间，第一个未放行的门禁下标（含当前步）
  function firstBlockerBefore(target) {
    const list = steps();
    for (let i = state.stepIndex; i < Math.min(target, list.length); i += 1) {
      if (isGateBlocking(list[i])) return i;
    }
    return -1;
  }

  function nudgeConfirm() {
    const cards = $$('.demo-card--confirm');
    cards.forEach((c) => {
      c.classList.remove('is-nudge');
      void c.offsetWidth;
      c.classList.add('is-nudge');
      c.scrollIntoView({ block: 'nearest' });
    });
  }

  function buildConfirmCard(st) {
    const info = effectiveConfirm(st);
    const kindLabel = info.kind === 'sign' ? '签发确认' : '人工确认';
    const done = isConfirmed(st);
    const rejected = isRejected(st);
    const body = el('div', 'demo-confirm');
    body.appendChild(el('div', 'demo-confirm__kind', kindLabel + (done ? ' · 已确认' : '')));
    body.appendChild(el('p', 'demo-confirm__text', info.text));

    if (!done) {
      const actions = el('div', 'demo-confirm__actions');
      const ok = el('button', 'btn btn-compact btn-solid', rejected ? '重新确认并继续' : '确认并继续');
      ok.type = 'button';
      ok.addEventListener('click', () => {
        state.confirmed[confirmKey(st)] = true;
        delete state.rejected[confirmKey(st)];
        if (state.stepIndex >= steps().length - 1) {
          render();
          toast('已确认：' + kindLabel + ' · 流程完成');
        } else {
          toast('已确认：' + kindLabel);
          advance();
        }
      });
      const no = el('button', 'btn btn-compact', '驳回 / 暂停');
      no.type = 'button';
      no.addEventListener('click', () => {
        state.rejected[confirmKey(st)] = true;
        stopAutoplay();
        render();
        toast('已驳回 · 流程在此停住，不会继续');
      });
      actions.appendChild(ok);
      actions.appendChild(no);
      body.appendChild(actions);
      if (rejected) {
        body.appendChild(el('div', 'demo-confirm__state', '已驳回：后续步骤不可推进。需重新确认，或退回上一步调整方案。'));
      } else {
        body.appendChild(el('div', 'demo-confirm__hint', '此步为人工门禁：未确认前「下一步 / 自动播放 / 阶段跳转」都不会越过。'));
      }
    }
    return card(kindLabel + ' · 人工确认节点', body, {
      cls: 'demo-card--confirm' + (rejected ? ' is-rejected' : ''),
      meta: st.id,
    });
  }

  // ---------------------------------------------------------------------
  // IDE VIEW — 左：文件树 + 编辑器
  // ---------------------------------------------------------------------
  // ---------------------------------------------------------------------
  // 工作区：与 Toolkit Studio 的 Coding 阶段共用同一个 Qwen3_14b workspace
  // ---------------------------------------------------------------------
  const WORKSPACE = {
    name: 'Qwen3_14b',
    folders: ['.claude', '.github', '3rdparty', 'cmake', 'docs', 'examples',
              'include', 'python', 'runtime', 'src', 'tests', 'toolchain'],
    files: [
      { name: 'AGENTS.md', kind: 'M' },
      { name: 'CMakeLists.txt', kind: 'C' },
      { name: 'pyproject.toml', kind: 'T' },
      { name: 'README.md', kind: 'M' },
    ],
    sources: [
      { name: 'decode_layer.py', kind: 'Py' },
      { name: 'paged_attention_dynamic.py', kind: 'Py' },
      { name: 'matmul.py', kind: 'Py' },
    ],
  };

  // 回放产物挂在 workspace 之下，随步骤逐步「生成」
  function buildArtifacts() {
    const out = [];
    const seen = {};
    steps().slice(0, state.stepIndex + 1).forEach((st) => {
      splitList(st.artifact).forEach((a2) => {
        if (a2 && !seen[a2]) { seen[a2] = 1; out.push(a2); }
      });
    });
    return out;
  }

  function fileTreeData() {
    return {
      root: WORKSPACE.name,
      active: 'decode_layer.py',
      folders: WORKSPACE.folders,
      files: WORKSPACE.files,
      sources: WORKSPACE.sources,
      artifacts: buildArtifacts(),
    };
  }

  function renderFileTree() {
    const target = $('#file-tree');
    target.innerHTML = '';
    const data = fileTreeData();
    target.appendChild(el('div', 'demo-filetree__root', data.root));

    const group = (label) => target.appendChild(el('div', 'demo-filetree__group', label));
    const item = (name, kind, opts) => {
      const it = el('button', 'demo-filetree__item'
        + (opts && opts.active ? ' is-active' : '')
        + (opts && opts.dir ? ' is-dir' : '')
        + (opts && opts.artifact ? ' is-artifact' : ''));
      it.type = 'button';
      it.title = name;
      it.appendChild(el('span', 'demo-filetree__kind', kind));
      it.appendChild(el('span', 'demo-filetree__name', name));
      if (opts && opts.badge) it.appendChild(el('span', 'demo-filetree__badge', opts.badge));
      it.addEventListener('click', () => toast('选中 ' + name));
      target.appendChild(it);
      return it;
    };

    data.folders.forEach((f) => item(f, '▸', { dir: true }));
    data.files.forEach((f) => item(f.name, f.kind, {}));
    group('kernels');
    data.sources.forEach((f, i) => item(f.name, f.kind, {
      active: f.name === data.active,
      badge: i === 0 ? '1' : null,
    }));
    if (data.artifacts.length) {
      group('build_output · 回放产物 ' + data.artifacts.length);
      data.artifacts.forEach((a2) => item(a2, '·', { artifact: true }));
    }
  }

  // ---------------------------------------------------------------------
  // 编辑器：真实 decode_layer.py 切片，按步骤锚定到对应算子
  // ---------------------------------------------------------------------
  // 每一步高亮哪一段源码 —— 让「回放走到哪一步」和「代码里的哪个算子」对得上。
  const STEP_ANCHOR = {
    correctness: {
      s1: 'post_rms_reduce', s2: 'residual_rms_cast', s3: 'post_rms_reduce', s4: 'post_rms_reduce',
      s5: 'silu', s6: 'gate_proj', s7: 'silu', s8: 'silu', s9: 'post_rms_reduce', s10: 'post_rms_reduce',
    },
    perf: {
      s1: 'rms_recip', s2: 'rms_recip', s3: 'fa_fused', s4: 'fa_fused', s5: 'fa_fused',
      s6: 'rms_recip', s7: 'down_proj', s8: 'rms_recip', s9: 'rms_recip', s10: 'rms_recip',
    },
    runtime: {
      s1: 'fa_fused', s2: 'fa_fused', s3: 'fa_fused', s4: 'fa_fused', s5: 'fa_fused',
      s6: 'fa_fused', s7: 'fa_fused', s8: 'fa_fused', s9: 'fa_fused', s10: 'fa_fused',
    },
    schedule: {
      s1: 'fa_fused', s2: 'fa_fused', s3: 'fa_fused', s4: 'silu', s5: 'silu',
      s6: 'silu', s7: 'down_proj', s8: 'down_proj', s9: 'down_proj', s10: 'down_proj',
    },
  };

  const SLICES = window.PTO_DECODE_LAYER_SLICES || [];

  function anchorName() {
    const m = STEP_ANCHOR[state.scenarioId] || STEP_ANCHOR.correctness;
    return m[step().id] || 'post_rms_reduce';
  }

  // 把切片摊平成带真实行号的行。算子块从 `with pl.` 开始、到下一个 `with pl.` 之前结束，
  // 命中当前锚点的整块高亮，便于「回放到哪一步」直接对到代码。
  function editorLines() {
    const target = anchorName();
    const rows = [];
    let prevEnd = null;
    SLICES.forEach((slice) => {
      const start = slice[0];
      const body = slice[1];
      if (prevEnd != null && start > prevEnd + 1) {
        rows.push({ gap: true, t: '⋯ ' + (start - prevEnd - 1) + ' lines' });
      }
      const blocks = [];
      body.forEach((text, i) => {
        if (/^\s*with pl\./.test(text)) blocks.push({ from: i, to: body.length - 1, name: null });
        const hint = text.match(/name_hint="([a-z_0-9]+)"/);
        if (hint && blocks.length) blocks[blocks.length - 1].name = hint[1];
      });
      for (let i = 0; i < blocks.length - 1; i += 1) blocks[i].to = blocks[i + 1].from - 1;
      const hit = blocks.filter((bl) => bl.name === target);
      body.forEach((text, i) => {
        const bl = hit.find((x) => i >= x.from && i <= x.to);
        rows.push({ n: start + i, t: text, anchor: !!bl, isHead: !!bl && /name_hint="/.test(text) });
      });
      prevEnd = start + body.length - 1;
    });
    return rows;
  }

  // Python 关键字 / 字符串 / 注释的轻量着色 —— 只做够读的程度，不引三方高亮库
  function pyText(txt) {
    const frag = document.createDocumentFragment();
    const push = (t, cls) => {
      if (!t) return;
      frag.appendChild(cls ? el('span', 'tok tok--' + cls, t) : document.createTextNode(t));
    };
    const c = txt.indexOf('#');
    const body = c >= 0 ? txt.slice(0, c) : txt;
    const re = /("[^"]*")|\b(with|for|in|as|def|return|import|if|else|range|True|False|None)\b|\b(pl)\.(\w+)|\b(\d+(?:\.\d+)?)\b/g;
    let last = 0, m;
    while ((m = re.exec(body))) {
      push(body.slice(last, m.index));
      if (m[1]) push(m[1], 'str');
      else if (m[2]) push(m[2], 'kw');
      else if (m[3]) { push('pl', 'ns'); push('.'); push(m[4], 'fn'); }
      else push(m[5], 'num');
      last = m.index + m[0].length;
    }
    push(body.slice(last));
    if (c >= 0) push(txt.slice(c), 'cmt');
    return frag;
  }

  function renderEditor() {
    const target = $('#code-editor');
    target.innerHTML = '';
    const anchor = anchorName();
    const head = el('div', 'demo-editor__lang');
    head.appendChild(el('span', null, 'pto-dsl · ' + WORKSPACE.name + '/kernels/' + fileTreeData().active));
    head.appendChild(el('span', 'demo-editor__anchor', '本步聚焦 · ' + anchor));
    target.appendChild(head);
    const code = el('div', 'demo-editor__code');
    let firstAnchorRow = null;
    editorLines().forEach((ln) => {
      if (ln.gap) {
        const g = el('div', 'demo-editor__gap', ln.t);
        code.appendChild(g);
        return;
      }
      const row = el('div', 'demo-editor__line'
        + (ln.anchor ? ' is-anchor' : '') + (ln.isHead ? ' is-anchor-head' : ''));
      row.appendChild(el('span', 'demo-editor__num', String(ln.n)));
      const c = el('span', 'demo-editor__text');
      c.appendChild(pyText(ln.t));
      row.appendChild(c);
      code.appendChild(row);
      if (ln.anchor && !firstAnchorRow) firstAnchorRow = row;
    });
    target.appendChild(code);
    // 跟随步骤滚到当前算子，省得用户自己找
    if (firstAnchorRow) {
      requestAnimationFrame(() => {
        code.scrollTop = Math.max(0, firstAnchorRow.offsetTop - code.clientHeight / 3);
      });
    }
  }

  // ---------------------------------------------------------------------
  // IDE VIEW — 中：当前证据面板（syncsToStep）
  // ---------------------------------------------------------------------
  function buildStepCard() {
    const st = step();
    const body = el('div', 'demo-stepcard');
    const metaRow = el('div', 'demo-stepcard__meta');
    metaRow.appendChild(badge(st.phase, 'phase'));
    metaRow.appendChild(evidenceBadge(st.evidenceLevel));
    metaRow.appendChild(gateBadge(st.gate));
    metaRow.appendChild(badge(agentName(st.agentId), 'agent'));
    body.appendChild(metaRow);
    body.appendChild(el('h3', 'demo-stepcard__title', st.title));
    const grid = el('div', 'demo-stepcard__grid');
    grid.appendChild(field('用户操作', st.userAction));
    grid.appendChild(field('Agent 动作', st.agentAction));
    if (st.toolCall) grid.appendChild(field('工具调用', st.toolCall, true));
    if (st.artifact) grid.appendChild(field('产物', st.artifact));
    body.appendChild(grid);
    body.appendChild(field('结论', st.conclusion));
    return card('步骤 ' + st.id + ' · ' + st.phase, body, { meta: 'syncsToStep · ' + scenario().scenarioId });
  }

  function buildSymptomCard() {
    const ul = el('ul', 'demo-list');
    scenario().symptom.forEach((s) => ul.appendChild(el('li', 'demo-list__item', s)));
    return card('症状卡 · Symptom', ul, { meta: 'scenario.symptom' });
  }

  function buildImpactChainCard() {
    const sc = scenario();
    const body = el('div', 'demo-chain');
    const revealCount = impactRevealCount();
    const revealed = (sc.impactChain || []).slice(0, revealCount);
    if (!revealed.length) {
      const locked = el('div', 'evidence-locked');
      locked.appendChild(el('span', 'evidence-locked__icon', '⌁'));
      locked.appendChild(el('div', 'evidence-locked__copy', '影响链尚未建立'));
      locked.appendChild(el('p', 'demo-note', '继续采集 producer → consumer 证据，链路会随步骤逐段解锁。'));
      body.appendChild(locked);
    }
    revealed.forEach((txt, i) => {
      const node = el('div', 'demo-chain__node');
      node.appendChild(el('span', 'demo-chain__idx', String(i + 1)));
      node.appendChild(el('span', 'demo-chain__text', txt));
      body.appendChild(node);
      if (i < revealed.length - 1) body.appendChild(el('span', 'demo-chain__arrow', '↓'));
    });
    if (revealed.length && revealed.length < (sc.impactChain || []).length) {
      body.appendChild(el('div', 'demo-chain__pending', '后续链路待证据解锁 · ' + ((sc.impactChain || []).length - revealed.length) + ' 个节点'));
    }
    const extras = el('div', 'demo-chain__extras');
    extras.appendChild(el('div', 'demo-chain__extra', '首个异常 · ' + (knownFirstAnomaly() || '待定位')));
    extras.appendChild(el('div', 'demo-chain__extra demo-chain__extra--root', '根因 · ' + (knownRootCause() || '证据尚未收敛')));
    body.appendChild(extras);
    return card('影响链 · Impact Chain', body, { meta: revealCount + '/' + (sc.impactChain || []).length + ' nodes' });
  }

  function tensorTableRows() {
    const sc = scenario();
    const rows = [];
    const e5 = sc.evidence.find((e) => e.id === 'e5');
    const e6 = sc.evidence.find((e) => e.id === 'e6');
    if (e5) e5.name.split('/').forEach((p) => rows.push({ name: p.replace(/\s*正确\s*$/, '').trim(), status: 'correct' }));
    if (e6) rows.push({ name: e6.name.replace(/\s*全零\s*$/, '').trim(), status: 'zero' });
    const s7 = sc.steps.find((s) => s.id === 's7');
    if (s7 && /logits\s*全零/.test((s7.conclusion || '') + (s7.userAction || ''))) {
      rows.push({ name: 'logits', status: 'propagated' });
    }
    return rows;
  }

  function buildTensorTableCard() {
    const body = el('div', 'demo-table-wrap');
    const table = el('table', 'demo-table');
    const thead = el('thead');
    const hr = el('tr');
    ['producer 中间量', 'host-visible 检查', '状态'].forEach((t) => hr.appendChild(el('th', null, t)));
    thead.appendChild(hr);
    table.appendChild(thead);
    const tbody = el('tbody');
    tensorTableRows().forEach((r) => {
      const tr = el('tr');
      tr.appendChild(el('td', 'demo-table__mono', r.name));
      tr.appendChild(el('td', null, r.status === 'zero' || r.status === 'propagated' ? '全零' : '正确'));
      const cell = el('td');
      const tag = r.status === 'correct'
        ? badge('通过', 'gate-pass')
        : (r.status === 'zero' ? badge('首个异常', 'gate-block') : badge('传播零值', 'gate-warn'));
      cell.appendChild(tag);
      tr.appendChild(cell);
      tbody.appendChild(tr);
    });
    table.appendChild(tbody);
    body.appendChild(table);
    body.appendChild(el('p', 'demo-note', '来源：args_dump.json 逐 producer 对拍（E2）· combine 只是传播者'));
    return card('逐 producer tensor 表', body, { meta: 'step s7 · dump_viewer' });
  }

  function buildCacheSchematicCard() {
    const body = el('div', 'demo-cache');
    const stages = [
      { name: 'post_rms_reduce', detail: 'AICore scalar raw GM store 写 post_inv_rms', note: 'pipe_barrier(PIPE_ALL)', cls: '' },
      { name: 'cache line', detail: '未 flush 到 HBM', note: '跨任务消费者读陈旧 post_inv_rms=0', cls: 'demo-cache__stage--danger' },
      { name: 'silu', detail: '读到陈旧 post_inv_rms=0', note: 'deferred RMS scale 整段乘成 0', cls: '' },
      { name: 'down_proj → rms_lm_head', detail: '传播零值', note: 'logits 全零', cls: '' },
    ];
    stages.forEach((s, i) => {
      const st = el('div', 'demo-cache__stage ' + s.cls);
      st.appendChild(el('span', 'demo-cache__name', s.name));
      st.appendChild(el('span', 'demo-cache__detail', s.detail));
      st.appendChild(el('span', 'demo-cache__note', s.note));
      body.appendChild(st);
      if (i < stages.length - 1) body.appendChild(el('span', 'demo-cache__arrow', '→'));
    });
    body.appendChild(el('p', 'demo-note', '根因：' + (knownRootCause() || '待单变量实验确认')));
    return card('缓存可见性示意', body, { meta: 'simulator 无真实 cache 行为' });
  }

  function buildFixDiffCard() {
    const sc = scenario();
    const body = el('div', 'demo-diff');
    const pre = el('div', 'demo-diff__code');
    const lines = [
      { k: 'ctx', t: '// post_rms_reduce kernel：写 post_inv_rms 后' },
      { k: 'ctx', t: 'pipe_barrier(PIPE_ALL);  // 仅保证核内 pipeline 顺序' },
      { k: 'add', t: 'dcci(post_inv_rms_range, CACHELINE_OUT);  // ' + sc.fix },
    ];
    lines.forEach((ln) => {
      const row = el('div', 'demo-diff__line demo-diff__line--' + ln.k);
      row.appendChild(el('span', 'demo-diff__mark', ln.k === 'add' ? '+' : ' '));
      const c = el('span', 'demo-diff__text');
      const di = ln.t.indexOf('dcci(');
      if (di >= 0) {
        const j = ln.t.indexOf(')', di);
        c.appendChild(document.createTextNode(ln.t.slice(0, di)));
        const hl = el('span', 'demo-diff__hl');
        hl.textContent = ln.t.slice(di, j + 1);
        c.appendChild(hl);
        c.appendChild(document.createTextNode(ln.t.slice(j + 1)));
      } else {
        c.textContent = ln.t;
      }
      row.appendChild(c);
      pre.appendChild(row);
    });
    body.appendChild(pre);
    body.appendChild(el('p', 'demo-note', sc.fix));
    return card('修复 diff（Recover）', body, { meta: 'Candidate · 不覆盖 Baseline' });
  }

  function regressionBadge(sc) {
    const m = (sc.regression || '').match(/(\d+\/\d+\s*PASS|bit-identical)/i);
    return m ? m[1] : '已回归';
  }

  function buildRegressionCard() {
    const sc = scenario();
    const body = el('div');
    const res = el('div', 'demo-regression');
    res.appendChild(el('span', 'demo-regression__badge', regressionBadge(sc)));
    res.appendChild(el('span', 'demo-regression__text', sc.regression));
    body.appendChild(res);
    body.appendChild(el('p', 'demo-note', 'Trusted Baseline 已签发 · ' + sc.status));
    return card('回归结果 · Regression', body, { meta: sc.platform });
  }

  function buildBaselineReportCard() {
    const sc = scenario();
    const extra = {
      correctness: { risk: 'simulator 无真实 cache 行为（onboard 验证已闭合）', admission: 'pass · Trusted Baseline 已签发' },
      perf: { risk: '调整线程数可能改变浮点归约顺序（torch.equal 复验 bit-identical）', admission: 'pass · Recipe 已发布' },
      runtime: { risk: 'force reset 具破坏性，需独占 task-submit 锁', admission: 'warn · 原失败仍存在，后续环境已恢复（不整体标绿）' },
    }[sc.scenarioId] || { risk: '-', admission: 'pass' };

    const body = el('div', 'demo-baseline-report');
    const blk = (label, pairs) => {
      const wrap = el('div', 'demo-section');
      wrap.appendChild(el('div', 'demo-section-label', label));
      pairs.forEach((p) => wrap.appendChild(field(p[0], p[1])));
      body.appendChild(wrap);
    };
    blk('Baseline · 可信基线', [
      ['验收规则', '固定输入 + Golden 复现 · 原 Contract 未变'],
      ['适用边界', sc.platform + ' · ' + sc.category],
      ['签发者', 'lead · 主责编排智能体'],
      ['过期条件', '平台 / 版本 / 语义变更时重新评估'],
    ]);
    blk('Report · 准入报告', [
      ['变更', sc.fix],
      ['结果', sc.regression],
      ['风险', extra.risk],
      ['证据', 'RUN + CODE · ' + sc.steps.length + ' 步 · ' + (sc.evidence || []).length + ' 条证据'],
      ['复现步骤', (sc.tools && sc.tools[0] && sc.tools[0].command) || '见 terminal 工具日志'],
      ['准入状态', extra.admission],
    ]);
    return card('Trusted Baseline · 准入报告', body, { meta: sc.scenarioId + ' · Deliver' });
  }

  // --- perf 专用卡 ---
  function parseDecomp() {
    const s6 = scenario().steps.find((s) => s.id === 's6');
    const text = (s6 && s6.agentAction) || '';
    const segs = [];
    const re = /([^，。]+?)\s*(\d+(?:\.\d+)?s|毫秒级)\s*[（(]([^（）()]+)[）)]/g;
    let m;
    while ((m = re.exec(text))) segs.push({ label: m[1].trim(), value: m[2], note: m[3].trim() });
    const totalM = (scenario().symptom[0] || '').match(/约\s*(\d+)\s*秒/);
    return { segs: segs, total: totalM ? Number(totalM[1]) : 406, conclusion: s6 ? s6.conclusion : '' };
  }

  function buildPerfDecompCard() {
    const d = parseDecomp();
    const body = el('div');
    const bar = el('div', 'demo-bar');
    d.segs.forEach((s) => {
      const sec = s.value === '毫秒级' ? 0.5 : (parseFloat(s.value) || 0);
      const w = Math.max(2, Math.round(sec / Math.max(1, d.total) * 100));
      const seg = el('div', 'demo-bar__seg');
      seg.style.width = w + '%';
      seg.title = s.label + ' ' + s.value;
      if (sec >= 300) seg.dataset.dominant = 'true';
      bar.appendChild(seg);
    });
    body.appendChild(el('div', 'demo-bar__scale', 'device wall ≈ ' + d.total + 's'));
    body.appendChild(bar);
    const ul = el('ul', 'demo-list');
    d.segs.forEach((s) => {
      const li = el('li', 'demo-list__item demo-list__item--kv');
      li.appendChild(el('span', 'demo-list__k', s.label));
      li.appendChild(el('span', 'demo-list__v', s.value + ' · ' + s.note));
      ul.appendChild(li);
    });
    body.appendChild(ul);
    body.appendChild(el('p', 'demo-note', d.conclusion));
    return card('406s 阶段分解', body, { meta: 'Queue / Host / Device' });
  }

  function buildPerfHypothesesCard() {
    const s7 = scenario().steps.find((s) => s.id === 's7');
    const text = ((s7 && s7.agentAction) || '').replace(/^.*?否定[:：]?/, '');
    const parts = text.split('、').map((p) => p.replace(/[。.]$/, '').trim()).filter(Boolean);
    const body = el('div', 'demo-hypotheses');
    parts.forEach((p) => {
      const row = el('div', 'demo-hypotheses__item');
      row.appendChild(el('span', 'demo-hypotheses__x', '✕'));
      row.appendChild(el('span', 'demo-hypotheses__text', p));
      body.appendChild(row);
    });
    body.appendChild(el('p', 'demo-note', (s7 && s7.conclusion) || ''));
    return card('否定四个候选假设', body, { meta: '逐条测量' });
  }

  function buildThreadCurveCard() {
    const s8 = scenario().steps.find((s) => s.id === 's8');
    const text = (s8 && s8.userAction) || '';
    const pts = [];
    const re = /(\d+)\s*→\s*(\d+(?:\.\d+)?)s/g;
    let m;
    while ((m = re.exec(text))) pts.push({ threads: Number(m[1]), sec: Number(m[2]) });
    const body = el('div');
    const plot = el('div', 'demo-threadcurve');
    const max = Math.max.apply(null, pts.map((p) => p.sec)) || 1;
    pts.forEach((p) => {
      const col = el('div', 'demo-threadcurve__col');
      col.style.height = Math.max(6, Math.round(p.sec / max * 100)) + '%';
      if (p.threads >= 4 && p.threads <= 16) col.classList.add('is-flat');
      col.title = p.threads + ' 线程 → ' + p.sec + 's';
      col.appendChild(el('span', 'demo-threadcurve__bar'));
      col.appendChild(el('span', 'demo-threadcurve__x', String(p.threads)));
      plot.appendChild(col);
    });
    body.appendChild(plot);
    body.appendChild(el('p', 'demo-note', (s8 && s8.conclusion) || ''));
    return card('thread curve 单变量定位', body, { meta: 'PYPTO_BENCH raw samples' });
  }

  function buildPerfRegressionCard() {
    const sc = scenario();
    const body = el('div');
    body.appendChild(el('span', 'demo-regression__badge', regressionBadge(sc)));
    body.appendChild(el('p', 'demo-note', sc.regression));
    body.appendChild(el('p', 'demo-note', '未采用方案记录见 recipe.md（Golden cache / vectorize / 拆 nightly / skip_golden）'));
    return card('bit-identical 回归 + Recipe', body, { meta: 'torch.equal' });
  }

  // --- runtime 专用卡 ---
  function buildRuntimeExperimentsCard() {
    const sc = scenario();
    const body = el('div', 'demo-experiments');
    ['s4', 's5', 's6'].filter((id) => {
      const idx = stepIndexById(id);
      return idx >= 0 && idx <= state.stepIndex;
    }).forEach((id) => {
      const st = sc.steps.find((s) => s.id === id);
      if (!st) return;
      const m = (st.conclusion || '').match(/结果\s*(\d+\/\d+)\s*(\w+)/);
      const exp = el('div', 'demo-experiment');
      exp.appendChild(el('span', 'demo-experiment__title', st.title));
      exp.appendChild(el('span', 'demo-experiment__result', m ? (m[1] + ' ' + m[2]) : (st.conclusion || '')));
      exp.appendChild(el('p', 'demo-experiment__note', st.conclusion));
      body.appendChild(exp);
    });
    return card('三组最小实验', body, { meta: '实验1 / 实验2 / 实验3' });
  }

  function buildForceResetCard() {
    const sc = scenario();
    const st = sc.steps.find((s) => s.id === 's8') || step();
    const body = el('div');
    body.appendChild(el('p', 'demo-note', st.conclusion));
    const ul = el('ul', 'demo-list');
    (sc.fix || '').split('+').forEach((f) => {
      const t = f.replace(/[。.]$/, '').trim();
      if (t) ul.appendChild(el('li', 'demo-list__item', t));
    });
    body.appendChild(ul);
    return card('force-reset 恢复', body, { meta: '破坏性恢复 · 独占 task-submit 锁' });
  }

  function renderCenterPanel() {
    const target = $('#center-panel');
    target.innerHTML = '';
    const st = step();
    const sc = scenario();
    const conf = effectiveConfirm(st);
    if (conf) target.appendChild(buildConfirmCard(st));
    target.appendChild(buildStepCard());

    const builders = [];
    if (st.id === 's1') builders.push(buildSymptomCard);
    builders.push(buildImpactChainCard);
    if (sc.scenarioId === 'correctness') {
      if (st.id === 's7') builders.push(buildTensorTableCard);
      if (st.phase === 'Judge') builders.push(buildCacheSchematicCard);
      if (st.phase === 'Recover' || st.phase === 'Deliver') builders.push(buildFixDiffCard);
      if (st.phase === 'Deliver') builders.push(buildRegressionCard);
    } else if (sc.scenarioId === 'perf') {
      if (st.id === 's6') builders.push(buildPerfDecompCard);
      if (st.id === 's7') builders.push(buildPerfHypothesesCard);
      if (st.id === 's8') builders.push(buildThreadCurveCard);
      if (st.phase === 'Deliver') builders.push(buildPerfRegressionCard);
    } else if (sc.scenarioId === 'runtime') {
      if (['s4', 's5', 's6', 's7'].indexOf(st.id) >= 0) builders.push(buildRuntimeExperimentsCard);
      if (st.id === 's8' || st.id === 's9') builders.push(buildForceResetCard);
    }
    if (st.phase === 'Deliver') builders.push(buildBaselineReportCard);
    builders.forEach((fn) => target.appendChild(fn()));
    target.scrollTop = 0;
  }

  // ---------------------------------------------------------------------
  // IDE VIEW — 右：证据检查器
  // ---------------------------------------------------------------------
  function confidence(level) {
    return ({
      'E0': '低 · 观察',
      'E1': '中 · 现场',
      'E2': '高 · 强信号',
      'E3': '确认 · 单变量/回归',
    })[level] || '未知';
  }

  function renderFindingCard() {
    const target = $('#finding-card');
    target.innerHTML = '';
    const st = step();
    const sc = scenario();
    const next = steps()[state.stepIndex + 1];
    const body = el('div', 'demo-finding');
    [
      ['当前结论', st.conclusion],
      ['首个异常', knownFirstAnomaly() || '尚未定位'],
      ['证据强度', st.evidenceLevel + ' · ' + confidence(st.evidenceLevel)],
      ['下一验证', next ? next.title : '签发 / 归档'],
      ['当前负责人', agentName(st.agentId)],
    ].forEach((pair) => body.appendChild(field(pair[0], pair[1])));
    target.appendChild(card('Finding · 当前可复核判断', body, { meta: st.id }));
  }

  function renderEvidenceFilter() {
    const target = $('#evidence-filter');
    target.innerHTML = '';
    const mk = (label, value) => {
      const b = el('button', 'demo-filter__chip' + (state.evidenceFilter === value ? ' is-active' : ''), label);
      b.type = 'button';
      b.addEventListener('click', () => {
        state.evidenceFilter = (state.evidenceFilter === value) ? null : value;
        renderEvidenceList();
        toast(state.evidenceFilter ? '过滤证据等级 ' + state.evidenceFilter : '显示全部证据');
      });
      target.appendChild(b);
    };
    ['E0', 'E1', 'E2', 'E3'].forEach((l) => mk(l, l));
    mk('全部', null);
  }

  function renderEvidenceList() {
    renderEvidenceFilter();
    const target = $('#evidence-list');
    target.innerHTML = '';
    const evs = unlockedEvidence();
    const filtered = state.evidenceFilter ? evs.filter((e) => e.level === state.evidenceFilter) : evs;
    if (!filtered.length) {
      target.appendChild(el('li', 'demo-empty', '无匹配证据'));
      return;
    }
    filtered.forEach((e) => {
      const li = el('li', 'demo-evidence__item');
      const top = el('div', 'demo-evidence__top');
      top.appendChild(evidenceBadge(e.level));
      top.appendChild(el('span', 'demo-evidence__name', e.name));
      li.appendChild(top);
      const meta = el('div', 'demo-evidence__meta');
      meta.appendChild(badge(e.layer || 'layer', 'layer'));
      meta.appendChild(el('span', 'demo-evidence__id', e.id));
      li.appendChild(meta);
      li.appendChild(el('p', 'demo-evidence__note', e.note));
      target.appendChild(li);
    });
  }

  function renderTerminalLog() {
    const target = $('#terminal-log');
    target.innerHTML = '';
    const all = steps();
    // 终端是「已执行」的记录，不是待办清单 —— 只渲染到当前步
    all.slice(0, state.stepIndex + 1).forEach((s, i) => {
      const line = el('div', 'demo-terminal__line' + (i === state.stepIndex ? ' is-current' : ' is-past'));
      line.appendChild(el('span', 'demo-terminal__prompt', '$'));
      line.appendChild(el('span', 'demo-terminal__cmd', s.toolCall || '# (无工具调用 — agent 推理)'));
      target.appendChild(line);
      if (s.artifact) target.appendChild(el('div', 'demo-terminal__out', '→ ' + s.artifact));
    });
    const rest = all.length - (state.stepIndex + 1);
    if (rest > 0) {
      const p = el('div', 'demo-terminal__pending');
      p.appendChild(el('span', null, '· 还有 ' + rest + ' 步未执行'));
      target.appendChild(p);
    }
    target.scrollTop = target.scrollHeight;
  }

  // ---------------------------------------------------------------------
  // Agent Windows — 左：花名册
  // ---------------------------------------------------------------------
  function renderRoster() {
    const target = $('#agent-roster');
    target.innerHTML = '';
    const roster = UI.views.agentWindows.columns.left.roster;
    roster.forEach((r) => {
      const agent = AGENTS.find((a) => a.id === r.id) || {};
      const b = el('button', 'demo-roster__item'
        + (r.id === step().agentId ? ' is-active' : '')
        + (state.selectedAgent === r.id ? ' is-selected' : ''));
      b.type = 'button';
      const head = el('div', 'demo-roster__head');
      head.appendChild(el('span', 'demo-roster__id', r.id));
      head.appendChild(badge(r.status || 'live', 'status-live'));
      b.appendChild(head);
      b.appendChild(el('div', 'demo-roster__name', agent.name || r.id));
      b.appendChild(el('div', 'demo-roster__duty', r.duty));
      if (agent.owns && agent.owns.length) {
        const owns = el('div', 'demo-roster__owns');
        agent.owns.forEach((p) => owns.appendChild(el('span', 'demo-roster__own', p)));
        b.appendChild(owns);
      }
      b.addEventListener('click', () => {
        state.selectedAgent = state.selectedAgent === r.id ? null : r.id;
        toast(state.selectedAgent ? '聚焦 ' + r.id : '取消聚焦');
        renderAgentView();
      });
      target.appendChild(b);
    });
  }

  // ---------------------------------------------------------------------
  // Agent Windows — 中：计划 + 时间线
  // ---------------------------------------------------------------------
  function renderPlan() {
    const target = $('#agent-plan');
    target.innerHTML = '';
    const sc = scenario();
    const body = el('div', 'demo-plan');
    body.appendChild(el('div', 'demo-plan__objective', sc.lead));
    body.appendChild(el('div', 'demo-plan__meta', sc.capability + ' · ' + sc.platform));
    const phases = el('div', 'demo-plan__phases');
    const curIdx = phaseIndex(step().phase);
    PHASES.forEach((p, i) => {
      const c = el('span', 'demo-plan__phase'
        + (i === curIdx ? ' is-active' : '')
        + (i < curIdx ? ' is-done' : ''), (i + 1) + '.' + p.id);
      c.title = p.judge;
      phases.appendChild(c);
    });
    body.appendChild(phases);
    target.appendChild(card('计划 · Plan', body, { meta: sc.status }));
  }

  function renderTimeline() {
    const target = $('#agent-timeline');
    target.innerHTML = '';
    steps().forEach((s, i) => {
      const isCur = i === state.stepIndex;
      const isPast = i < state.stepIndex;
      const dimmed = state.selectedAgent && s.agentId !== state.selectedAgent;
      const node = el('div', 'demo-timeline__node'
        + (isCur ? ' is-current' : '')
        + (isPast ? ' is-past' : '')
        + (dimmed ? ' is-dimmed' : ''));
      const head = el('div', 'demo-timeline__head');
      head.appendChild(el('span', 'demo-timeline__phase', s.phase));
      head.appendChild(el('span', 'demo-timeline__id', s.id));
      head.appendChild(el('span', 'demo-timeline__agent', agentName(s.agentId)));
      node.appendChild(head);
      node.appendChild(el('div', 'demo-timeline__title', s.title));
      const chain = el('div', 'demo-timeline__chain');
      chain.appendChild(el('span', 'demo-timeline__chip', 'agent'));
      chain.appendChild(el('span', 'demo-timeline__chip demo-timeline__chip--mono', s.toolCall || '—'));
      chain.appendChild(el('span', 'demo-timeline__chip', s.artifact || 'artifact —'));
      chain.appendChild(el('span', 'demo-timeline__chip', s.conclusion));
      chain.appendChild(gateBadge(s.gate));
      node.appendChild(chain);
      target.appendChild(node);
    });
    const cur = target.querySelector('.is-current');
    if (cur) cur.scrollIntoView({ block: 'nearest' });
  }

  // ---------------------------------------------------------------------
  // Agent Windows — 右：确认卡 + 交接/证据包
  // ---------------------------------------------------------------------
  function renderConfirmCard() {
    const target = $('#confirm-card');
    target.innerHTML = '';
    const st = step();
    const conf = effectiveConfirm(st);
    if (conf) {
      target.appendChild(buildConfirmCard(st));
    } else {
      const body = el('div', 'demo-confirm--none');
      body.appendChild(el('p', 'demo-empty', '当前步骤无人工确认卡'));
      const upcoming = steps().filter((s) => effectiveConfirm(s)).map((s) => s.id);
      if (upcoming.length) body.appendChild(el('p', 'demo-note', '确认节点：' + upcoming.join(' · ')));
      target.appendChild(card('人工确认卡', body, { meta: st.id }));
    }
  }

  function renderHandoff() {
    const target = $('#handoff-package');
    target.innerHTML = '';
    const sc = scenario();
    const body = el('div', 'demo-handoff');
    body.appendChild(el('div', 'demo-handoff__lead', sc.lead));
    const sec = (title, items) => {
      const s = el('div', 'demo-handoff__sec');
      s.appendChild(el('div', 'demo-handoff__sec-title', title));
      const ul = el('ul', 'demo-list');
      (Array.isArray(items) ? items : [items]).filter(Boolean).forEach((it) => ul.appendChild(el('li', 'demo-list__item', it)));
      s.appendChild(ul);
      body.appendChild(s);
    };
    sec('根因', [sc.rootCause]);
    sec('修复', [sc.fix]);
    sec('回归', [sc.regression]);
    sec('可复用规则', sc.reusableRules || []);
    target.appendChild(card('交接 / 证据包', body, { meta: sc.scenarioId + ' · ' + sc.status }));
  }

  // ---------------------------------------------------------------------
  // 工具面板：工具注册表 + 归属 + 渲染
  // ---------------------------------------------------------------------
  const TOOL_META = {
    code:        { label: '代码',            agent: 'lead',         kind: 'code' },
    swimlane:    { label: '泳道图',          agent: 'runtime',      kind: 'viz' },
    deps:        { label: '依赖 / 影响链',   agent: 'runtime',      kind: 'viz' },
    finding:     { label: 'Finding',         agent: 'lead',         kind: 'insight' },
    evidence:    { label: '证据列表',        agent: 'correctness',  kind: 'evidence' },
    symptom:     { label: '症状卡',          agent: 'triage',       kind: 'insight' },
    impact:      { label: '影响链',          agent: 'triage',       kind: 'viz' },
    tensor:      { label: 'Tensor 对拍',     agent: 'correctness',  kind: 'evidence' },
    cache:       { label: '缓存可见性',      agent: 'runtime',      kind: 'viz' },
    fix:         { label: '修复 diff',       agent: 'lead',         kind: 'code' },
    regression:  { label: '回归结果',        agent: 'lead',         kind: 'insight' },
    perfdecomp:  { label: '阶段分解',        agent: 'perf',         kind: 'viz' },
    perfhypo:    { label: '否定假设',        agent: 'perf',         kind: 'insight' },
    threadcurve: { label: '线程曲线',        agent: 'perf',         kind: 'viz' },
    perfreg:     { label: '回归 + Recipe',   agent: 'perf',         kind: 'insight' },
    experiments: { label: '三组实验',        agent: 'runtime',      kind: 'insight' },
    forcereset:  { label: '恢复方案',        agent: 'runtime',      kind: 'insight' },
    baseline:    { label: '基线与准入报告',  agent: 'lead',         kind: 'report' },
    handoff:     { label: '交接 / 证据包',   agent: 'lead',         kind: 'report' },
    contrast:    { label: '配对对照 / 区分实验', agent: 'perf',      kind: 'evidence' },
  };

  // tile 很窄，kind 用两字中文比英文更省位置也更好扫
  const TOOL_KIND_LABEL = {
    viz: '可视化',
    insight: '判断',
    evidence: '证据',
    code: '代码',
    report: '报告',
  };

  function buildTool(id) {
    const b = {
      code: buildCodeTool,
      contrast: buildContrastCard,
      swimlane: buildSwimlaneTool,
      deps: buildDepsTool,
      finding: buildFindingTool,
      evidence: buildEvidenceTool,
      symptom: buildSymptomCard,
      impact: buildImpactChainCard,
      tensor: buildTensorTableCard,
      cache: buildCacheSchematicCard,
      fix: buildFixDiffCard,
      regression: buildRegressionCard,
      perfdecomp: buildPerfDecompCard,
      perfhypo: buildPerfHypothesesCard,
      threadcurve: buildThreadCurveCard,
      perfreg: buildPerfRegressionCard,
      experiments: buildRuntimeExperimentsCard,
      forcereset: buildForceResetCard,
      baseline: buildBaselineReportCard,
      handoff: buildHandoffTool,
    }[id];
    return b ? b() : null;
  }

  // 工具集合只由「回放到第几步」决定 —— 传入 index 便于回溯某个工具最早在哪一步产出
  function toolsForStep(index) {
    const at = index == null ? state.stepIndex : index;
    const sc = scenario();
    const all = steps();
    const st = all[at] || all[0];
    const reached = (id) => {
      const idx = all.findIndex((s) => s.id === id);
      return idx >= 0 && at >= idx;
    };
    const ids = [];
    const push = (id) => { if (ids.indexOf(id) < 0) ids.push(id); };
    push('finding');
    push('evidence');
    if (sc.scenarioId === 'correctness') {
      if (st.id === 's1') push('symptom');
      if (phaseIndex(st.phase) >= phaseIndex('Judge')) push('impact');
      if (st.id === 's7') push('tensor');
      if (reached('s8')) push('cache');
      if (st.phase === 'Recover' || st.phase === 'Deliver') { push('fix'); push('code'); }
      if (st.phase === 'Deliver') { push('regression'); push('baseline'); push('handoff'); }
    } else if (sc.scenarioId === 'perf') {
      if (st.id === 's6') push('perfdecomp');
      if (st.id === 's7') push('perfhypo');
      if (st.id === 's8') push('threadcurve');
      if (reached('s6')) push('swimlane');
      if (st.phase === 'Recover' || st.phase === 'Deliver') push('code');
      if (st.phase === 'Deliver') { push('perfreg'); push('baseline'); push('handoff'); }
    } else if (sc.scenarioId === 'schedule') {
      if (reached('s2')) push('contrast');
      if (phaseIndex(st.phase) >= phaseIndex('Judge')) push('impact');
      if (reached('s6')) push('swimlane');
      if (st.phase === 'Recover' || st.phase === 'Deliver') push('code');
      if (st.phase === 'Deliver') push('handoff');
    } else {
      if (['s4', 's5', 's6', 's7'].indexOf(st.id) >= 0) push('experiments');
      if (st.id === 's8' || st.id === 's9') push('forcereset');
      if (reached('s7')) push('swimlane');
      if (st.phase === 'Recover' || st.phase === 'Deliver') push('code');
      if (st.phase === 'Deliver') { push('baseline'); push('handoff'); }
    }
    return ids.map((id) => ({ id: id, meta: TOOL_META[id] })).filter((t) => t.meta);
  }

  // ---------------------------------------------------------------------
  // 工具归属：谁调的、哪一步调的、现在是不是活跃
  // ---------------------------------------------------------------------
  // 工具面板里的每张卡都由某个 Agent 产出；回溯首次出现的步骤，让用户看得到
  // 「correctness 在 s7 调用 Tensor 对拍」这条因果，而不是一堆无主的卡片。
  function toolOrigin(id) {
    for (let i = 0; i <= state.stepIndex; i += 1) {
      if (toolsForStep(i).some((t) => t.id === id)) return steps()[i];
    }
    return step();
  }

  function toolAttribution(id) {
    const meta = TOOL_META[id];
    const origin = toolOrigin(id);
    const owner = meta.agent;
    const live = step().agentId === owner;
    const fresh = origin.id === step().id;
    return {
      meta: meta,
      owner: owner,
      origin: origin,
      status: live ? 'live' : (fresh ? 'fresh' : 'ready'),
      statusLabel: live ? '正在调用' : (fresh ? '本步产出' : '已产出'),
      caller: origin.agentId,
      hint: (AGENT_SHORT_ROLE[owner] || owner) + ' Agent · ' + meta.kind
        + ' · 首次产出 ' + origin.id + '（' + origin.phase + '）',
    };
  }

  // 工具面板抬头：把「本步一共几个工具 / 分属几个 Agent」讲清楚
  function toolAgentGroups() {
    const groups = [];
    toolsForStep().forEach((t) => {
      let g = groups.find((x) => x.agentId === t.meta.agent);
      if (!g) { g = { agentId: t.meta.agent, tools: [] }; groups.push(g); }
      g.tools.push(t);
    });
    return groups;
  }

  function visibleTools() {
    const tools = toolsForStep();
    if (!state.selectedAgent) return tools;
    return tools.filter((t) => t.meta.agent === state.selectedAgent);
  }

  function buildCodeTool() {
    const st = step();
    const sc = scenario();
    const body = el('div', 'demo-diff');
    const pre = el('div', 'demo-diff__code');
    const lines = [];
    lines.push({ k: 'ctx', t: '// ' + displayTitle() });
    lines.push({ k: 'cmd', t: st.toolCall || '# (无工具调用)' });
    if (knownFix()) {
      lines.push({ k: 'add', t: '+ ' + knownFix() });
    }
    lines.forEach((ln) => {
      const row = el('div', 'demo-diff__line demo-diff__line--' + ln.k);
      row.appendChild(el('span', 'demo-diff__mark', ln.k === 'add' ? '+' : ' '));
      const c = el('span', 'demo-diff__text');
      const txt = ln.t;
      const di = txt.indexOf('dcci(');
      if (di >= 0) {
        const j = txt.indexOf(')', di);
        c.appendChild(document.createTextNode(txt.slice(0, di)));
        const hl = el('span', 'demo-diff__hl');
        hl.textContent = txt.slice(di, j + 1);
        c.appendChild(hl);
        c.appendChild(document.createTextNode(txt.slice(j + 1)));
      } else {
        c.textContent = txt;
      }
      row.appendChild(c);
      pre.appendChild(row);
    });
    body.appendChild(pre);
    return card('代码 · 源码片段', body, { meta: 'agent:lead · ' + fileTreeData().active });
  }

  const SWIM_DATA = {
    correctness: {
      lanes: ['Host', 'AICPU', 'AICore'],
      tasks: [
        { lane: 'AICore', start: 0, dur: 2, label: 'post_rms_reduce', note: '写 post_inv_rms · 未 flush', tone: '' },
        { lane: 'AICore', start: 2, dur: 1, label: '缺 dcci', note: 'cache line 未刷到 HBM', tone: 'danger' },
        { lane: 'AICore', start: 3, dur: 2, label: 'silu', note: '读陈旧 post_inv_rms=0', tone: 'warn' },
        { lane: 'AICore', start: 5, dur: 1, label: 'down_proj', note: '传播零值', tone: 'muted' },
      ],
    },
    perf: {
      lanes: ['Host', 'Device'],
      tasks: [
        { lane: 'Host', start: 0, dur: 9, label: 'golden_decode_layer', note: '359s · 40 层', tone: 'dominant' },
        { lane: 'Device', start: 0, dur: 1, label: 'device run', note: '毫秒级', tone: '' },
      ],
    },
    schedule: {
      lanes: ['same-step', 'distance-2'],
      tasks: [
        { lane: 'same-step', start: 0, dur: 1.5, label: 'produce', note: 'produce_qk(step)', tone: '' },
        { lane: 'same-step', start: 1.5, dur: 2, label: 'wait', note: 'sync_wait(PROB_READY) · 串行等待', tone: 'danger' },
        { lane: 'same-step', start: 3.5, dur: 1.5, label: 'consume', note: 'consume_pv(step) · 27.9 μs/轮', tone: 'warn' },
        { lane: 'distance-2', start: 0, dur: 1.5, label: 'prologue', note: '预产 2 个 slot', tone: 'muted' },
        { lane: 'distance-2', start: 1.5, dur: 2.4, label: 'produce+consume', note: '边产边消费 · 18.8 μs/轮', tone: 'dominant' },
      ],
    },
    runtime: {
      lanes: ['Host', 'AICore'],
      tasks: [
        { lane: 'AICore', start: 0, dur: 2, label: 'fa_fused HANG', note: 'op-timeout', tone: 'danger' },
        { lane: 'AICore', start: 2, dur: 4, label: 'NOOP 级联', note: 'sticky-error 污染', tone: 'warn' },
      ],
    },
  };

  function buildSwimlaneTool() {
    const data = SWIM_DATA[scenario().scenarioId] || SWIM_DATA.correctness;
    const body = el('div', 'swim');
    data.lanes.forEach((lane) => {
      const row = el('div', 'swim__row');
      row.appendChild(el('div', 'swim__lane', lane));
      const track = el('div', 'swim__track');
      data.tasks.filter((t) => t.lane === lane).forEach((t) => {
        const bar = el('div', 'swim__bar swim__bar--' + t.tone);
        bar.style.left = (t.start / 6 * 100) + '%';
        bar.style.width = Math.max(8, t.dur / 6 * 100) + '%';
        bar.title = t.label + ' · ' + t.note;
        bar.appendChild(el('span', 'swim__bar-label', t.label));
        track.appendChild(bar);
      });
      row.appendChild(track);
      body.appendChild(row);
    });
    const legend = el('div', 'swim__legend');
    data.tasks.forEach((t) => {
      const c = el('span', 'swim__legend-item');
      c.appendChild(el('span', 'swim__dot swim__dot--' + t.tone));
      c.appendChild(el('span', null, t.label + ' · ' + t.note));
      legend.appendChild(c);
    });
    body.appendChild(legend);
    return card('泳道图 · 执行时序', body, { meta: 'agent:runtime · 简化示意' });
  }

  function buildDepsTool() {
    return buildImpactChainCard();
  }

  function buildFindingTool() {
    const st = step();
    const sc = scenario();
    const next = steps()[state.stepIndex + 1];
    const body = el('div', 'demo-finding');
    [
      ['当前结论', st.conclusion],
      ['首个异常', knownFirstAnomaly() || '尚未定位'],
      ['证据强度', st.evidenceLevel + ' · ' + confidence(st.evidenceLevel)],
      ['下一验证', next ? next.title : '签发 / 归档'],
      ['当前负责人', agentName(st.agentId)],
    ].forEach((pair) => body.appendChild(field(pair[0], pair[1])));
    return card('Finding · 当前可复核判断', body, { meta: st.id });
  }

  function buildEvidenceTool() {
    const wrap = el('div', 'demo-evidence-tool');
    const allEvidence = scenario().evidence || [];
    const available = unlockedEvidence();
    const progress = el('div', 'evidence-progress');
    const progressText = el('div', 'evidence-progress__text');
    progressText.appendChild(el('strong', null, String(available.length)));
    progressText.appendChild(el('span', null, ' / ' + allEvidence.length + ' 条证据已解锁'));
    progress.appendChild(progressText);
    const track = el('span', 'evidence-progress__track');
    const fill = el('span', 'evidence-progress__fill');
    fill.style.width = (allEvidence.length ? Math.round(available.length / allEvidence.length * 100) : 0) + '%';
    track.appendChild(fill);
    progress.appendChild(track);
    wrap.appendChild(progress);
    const ul = el('ul', 'demo-evidence__list demo-evidence__list--tool');
    available.forEach((e) => {
      const li = el('li', 'demo-evidence__item');
      const top = el('div', 'demo-evidence__top');
      top.appendChild(evidenceBadge(e.level));
      top.appendChild(el('span', 'demo-evidence__name', e.name));
      li.appendChild(top);
      const meta = el('div', 'demo-evidence__meta');
      meta.appendChild(badge(e.layer || 'layer', 'layer'));
      meta.appendChild(el('span', 'demo-evidence__id', e.id));
      li.appendChild(meta);
      li.appendChild(el('p', 'demo-evidence__note', e.note));
      ul.appendChild(li);
    });
    if (!available.length) ul.appendChild(el('li', 'demo-empty', '执行当前步骤后生成第一条证据'));
    wrap.appendChild(ul);
    const lockedCount = Math.max(0, allEvidence.length - available.length);
    if (lockedCount) wrap.appendChild(el('div', 'evidence-lock-count', '还有 ' + lockedCount + ' 条证据将在后续实验中解锁'));
    return card('证据列表 · Evidence', wrap, { meta: available.length + '/' + allEvidence.length + ' available' });
  }

  // 配对对照 + 区分实验：INV-2040 的两组关键数字，按回放进度逐步解锁
  function buildContrastCard() {
    const sc = scenario();
    const body = el('div', 'demo-contrast');
    const ev = (id) => (sc.evidence || []).find((e) => e.id === id);
    const unlocked = (id) => unlockedEvidence().some((e) => e.id === id);

    const pair = (title, meta, rows) => {
      const grp = el('div', 'demo-contrast__group');
      const head = el('div', 'demo-contrast__head');
      head.appendChild(el('span', 'demo-contrast__title', title));
      head.appendChild(el('span', 'demo-contrast__meta', meta));
      grp.appendChild(head);
      rows.forEach((r) => {
        const row = el('div', 'demo-contrast__row' + (r.tone ? ' is-' + r.tone : ''));
        row.appendChild(el('span', 'demo-contrast__name', r.name));
        const track = el('span', 'demo-contrast__track');
        const fill = el('span', 'demo-contrast__fill');
        fill.style.width = Math.max(6, Math.round(r.pct)) + '%';
        track.appendChild(fill);
        row.appendChild(track);
        row.appendChild(el('code', 'demo-contrast__value', r.value));
        grp.appendChild(row);
      });
      body.appendChild(grp);
    };

    if (unlocked('e2')) {
      pair('配对 L2 · 端到端', 'batch 16 · seq_len 3584 · seed 1234 · L2 level 4', [
        { name: 'CCE', value: '1004.74 μs', pct: 99 },
        { name: 'PyPTO', value: '1014.66 μs', pct: 100 },
      ]);
      body.appendChild(el('p', 'demo-note', '端到端只差约 1%，Golden PASS/PASS —— 2.3× 不在整层。'));
    }
    if (unlocked('e3')) {
      pair('Attention Core 执行段', 'merged L2 trace', [
        { name: 'AIC exec', value: '266.42 μs', pct: 100 },
        { name: 'AIV exec', value: '266.48 μs', pct: 100 },
        { name: 'extra launch', value: '57.78 μs', pct: 22, tone: 'danger' },
      ]);
      body.appendChild(el('p', 'demo-note', '核内两侧几乎相同；差异落在边界 / 调度。'));
    }
    if (unlocked('e5')) {
      pair('区分实验 · 唯一变量 = 依赖距离', '各 100 轮 median · Golden PASS/PASS', [
        { name: 'same-step', value: '27.9 μs', pct: 100, tone: 'danger' },
        { name: 'distance-2', value: '18.8 μs', pct: 67, tone: 'good' },
      ]);
      body.appendChild(el('p', 'demo-note', '相同 workspace 与 sync 事件下改变依赖距离 → 1.48×；隔离收益不可外推为全层。'));
    }
    if (!body.childNodes.length) {
      body.appendChild(el('p', 'demo-note', '尚未锁定可横比的配对数据；先建立 Performance Contract。'));
    }
    const e7 = ev('e7');
    if (e7 && unlocked('e7')) {
      const open = el('div', 'demo-contrast__open');
      open.appendChild(el('strong', null, '未关闭边界'));
      open.appendChild(el('span', null, e7.note));
      body.appendChild(open);
    }
    return card('配对对照 / 区分实验', body, { meta: 'INV-2040 · ' + (sc.status || '') });
  }

  function buildHandoffTool() {
    const sc = scenario();
    const body = el('div', 'demo-handoff');
    body.appendChild(el('div', 'demo-handoff__lead', sc.lead));
    const sec = (title, items) => {
      const s = el('div', 'demo-handoff__sec');
      s.appendChild(el('div', 'demo-handoff__sec-title', title));
      const ul = el('ul', 'demo-list');
      (Array.isArray(items) ? items : [items]).filter(Boolean).forEach((it) => ul.appendChild(el('li', 'demo-list__item', it)));
      s.appendChild(ul);
      body.appendChild(s);
    };
    sec('根因', [sc.rootCause]);
    sec('修复', [sc.fix]);
    sec('回归', [sc.regression]);
    sec('可复用规则', sc.reusableRules || []);
    return card('交接 / 证据包', body, { meta: sc.scenarioId + ' · ' + sc.status });
  }

  // ---------------------------------------------------------------------
  // IDE VIEW：目录树 | 编辑器+终端 | 工具面板（tabs）
  // ---------------------------------------------------------------------
  // 工具卡抬头：Agent 头像 + 归属 + 调用状态（tile / 全屏 / IDE 三处共用）
  function toolAttrStrip(id, opts) {
    const attr = toolAttribution(id);
    const strip = el('div', 'tool-attr__row tool-attr__row--' + attr.status);
    strip.appendChild(agentAvatar(attr.owner, 'tool-attr__avatar'));
    const identity = el('div', 'tool-attr__identity');
    const line1 = el('div', 'tool-attr__line');
    line1.appendChild(el('strong', 'tool-attr__agent', AGENT_SHORT_ROLE[attr.owner] || attr.owner));
    line1.appendChild(el('span', 'tool-attr__id', attr.owner));
    line1.appendChild(el('span', 'tool-attr__status', attr.statusLabel));
    identity.appendChild(line1);
    const line2 = el('div', 'tool-attr__line tool-attr__line--sub');
    line2.appendChild(el('span', 'tool-attr__tool', (opts && opts.label) || attr.meta.label));
    const kind = el('span', 'tool-attr__kind tool-attr__kind--' + attr.meta.kind,
      TOOL_KIND_LABEL[attr.meta.kind] || attr.meta.kind);
    kind.title = attr.meta.kind;
    line2.appendChild(kind);
    const origin = el('span', 'tool-attr__origin', attr.origin.id + ' · ' + attr.origin.phase);
    origin.title = '首次产出于 ' + attr.origin.id + '（' + attr.origin.phase + '）';
    line2.appendChild(origin);
    identity.appendChild(line2);
    strip.appendChild(identity);
    strip.title = attr.hint;
    if (opts && opts.actions) strip.appendChild(opts.actions);
    return strip;
  }

  function renderIdeTools() {
    const tools = toolsForStep();
    const tabs = $('#ide-tool-tabs');
    tabs.innerHTML = '';
    const activeId = tools.some((t) => t.id === state.ideToolActive) ? state.ideToolActive : tools[0].id;
    state.ideToolActive = activeId;
    tools.forEach((t) => {
      const attr = toolAttribution(t.id);
      const b = el('button', 'tool-tab tool-tab--' + attr.status + (t.id === activeId ? ' is-active' : ''));
      b.type = 'button';
      b.id = 'tool-tab-' + t.id;
      b.setAttribute('role', 'tab');
      b.setAttribute('aria-selected', String(t.id === activeId));
      b.setAttribute('aria-controls', 'ide-tool-body');
      b.tabIndex = t.id === activeId ? 0 : -1;
      b.title = attr.hint;
      b.appendChild(agentAvatar(t.meta.agent, 'tool-tab__avatar'));
      b.appendChild(el('span', 'tool-tab__label', t.meta.label));
      b.addEventListener('click', () => { state.ideToolActive = t.id; renderIdeTools(); });
      tabs.appendChild(b);
    });
    const active = tools.find((t) => t.id === activeId) || tools[0];
    $('#ide-tool-meta').textContent = toolAgentGroups().length + ' agents · ' + tools.length + ' tools';
    const attrSlot = $('#ide-tool-attr');
    attrSlot.innerHTML = '';
    attrSlot.appendChild(toolAttrStrip(active.id));
    const body = $('#ide-tool-body');
    body.setAttribute('aria-labelledby', 'tool-tab-' + activeId);
    body.innerHTML = '';
    body.appendChild(buildTool(activeId));
  }

  function renderTerminalState() {
    const term = $('#ide-terminal');
    term.classList.toggle('is-collapsed', !state.terminalOpen);
    const btn = $('#terminal-toggle');
    if (btn) {
      btn.textContent = state.terminalOpen ? '⌄' : '⌃';
      btn.setAttribute('aria-expanded', String(state.terminalOpen));
    }
  }

  function toggleTerminal() {
    state.terminalOpen = !state.terminalOpen;
    renderTerminalState();
    toast(state.terminalOpen ? '终端已展开' : '终端已收起');
  }

  // ---------------------------------------------------------------------
  // Agent Windows：任务列表 | 对话流 | 工具面板（平铺/全屏）
  // ---------------------------------------------------------------------
  function renderTaskList() {
    const target = $('#task-list');
    target.innerHTML = '';
    const sc = scenario();
    const ui = scenarioUi();
    const task = el('div', 'task-card is-active');
    task.appendChild(el('div', 'task-card__id', ui.taskId));
    task.appendChild(el('div', 'task-card__title', displayTitle()));
    task.appendChild(badge(sc.capability, 'cap'));
    const taskStatus = isGateBlocking(step())
      ? (step().phase === 'Deliver' ? '等待人工签发' : '等待人工确认')
      : (isAtOrAfter('s10') ? sc.status : ('CASE 回放中 · Step ' + (state.stepIndex + 1) + '/' + steps().length));
    task.appendChild(el('div', 'task-card__status', taskStatus));
    target.appendChild(task);

    const runs = el('div', 'task-section');
    runs.appendChild(el('div', 'task-section__title', 'Runs'));
    [
      { txt: 'baseline ' + ui.baselineRun, cls: '' },
      { txt: 'candidate ' + ui.candidateRun, cls: 'is-candidate' },
    ].forEach((r) => {
      const item = el('div', 'task-run ' + r.cls);
      item.appendChild(el('span', 'task-run__dot'));
      item.appendChild(el('code', 'demo-mono', r.txt));
      runs.appendChild(item);
    });
    target.appendChild(runs);

    const phases = el('div', 'task-section');
    phases.appendChild(el('div', 'task-section__title', '7 阶段门禁'));
    const curIdx = phaseIndex(step().phase);
    PHASES.forEach((p, i) => {
      const item = el('div', 'task-phase' + (i === curIdx ? ' is-active' : '') + (i < curIdx ? ' is-done' : ''));
      item.appendChild(el('span', 'task-phase__num', String(i + 1)));
      item.appendChild(el('span', 'task-phase__id', p.id));
      item.appendChild(el('span', 'task-phase__judge', p.judge));
      phases.appendChild(item);
    });
    target.appendChild(phases);

    const related = el('div', 'task-section');
    related.appendChild(el('div', 'task-section__title', '相关案例 · 切换'));
    SCENARIOS.forEach((s) => {
      const item = el('button', 'task-case' + (s.scenarioId === state.scenarioId ? ' is-active' : ''));
      item.type = 'button';
      item.appendChild(el('span', 'task-case__id', s.scenarioId));
      item.appendChild(el('span', 'task-case__title', s.capability));
      item.addEventListener('click', () => { if (s.scenarioId !== state.scenarioId) switchScenario(s.scenarioId); });
      related.appendChild(item);
    });
    target.appendChild(related);
  }

  // ---------------------------------------------------------------------
  // 对话流：Agent 回合 = 「工具调用行（可展开）+ 结论行」交替
  // ---------------------------------------------------------------------
  // 把命令压成一个人能读的动作名：python -m simpler_setup.tools.dump_viewer → dump_viewer
  function toolCallName(cmd) {
    if (!cmd) return '';
    // 命令里可能带注释行 / 前置 export，取第一段真正会执行的东西
    const seg = String(cmd).split(/[;&|\n]/).map((x) => x.trim())
      .filter((x) => x && x.charAt(0) !== '#' && !/^(export|mkdir|cd|LOGDIR=)/.test(x));
    const first = seg[0] || String(cmd).trim();
    const mod = first.match(/-m\s+([\w.]+)/);
    if (mod) return mod[1].split('.').pop();
    const py = first.match(/([\w./-]+\.py)/);
    if (py) return py[1].split('/').pop();
    const fn = first.match(/^(\w+)\s*\(/);
    if (fn) return fn[1];
    const tok = first.split(/\s+/).filter((t) => t.indexOf('=') < 0)[0] || first;
    return tok.split('/').pop().slice(0, 24);
  }

  const splitList = (txt) => String(txt || '').split(/[、,，]/).map((x) => x.trim()).filter(Boolean);

  // 某一步新产出的工具面板（与上一步相比多出来的那些）
  function newToolsAt(index) {
    const now = toolsForStep(index).map((t) => t.id);
    if (index <= 0) return now;
    const before = toolsForStep(index - 1).map((t) => t.id);
    return now.filter((id) => before.indexOf(id) < 0);
  }

  // 工具调用行的摘要文案：「运行 dump_viewer，产出 Tensor 对拍」
  function callSummary(st, index) {
    const parts = [];
    const name = toolCallName(st.toolCall);
    if (name) parts.push('运行 ' + name);
    const fresh = newToolsAt(index).map((id) => TOOL_META[id] && TOOL_META[id].label).filter(Boolean);
    if (fresh.length) parts.push('产出 ' + fresh.slice(0, 2).join('、') + (fresh.length > 2 ? ' 等 ' + fresh.length + ' 个面板' : ''));
    else {
      const arts = splitList(st.artifact);
      if (arts.length) parts.push('读取 ' + (arts.length > 1 ? arts.length + ' 个产物' : arts[0]));
    }
    return parts.join('，') || st.title;
  }

  const actKey = (st) => state.scenarioId + '/' + st.id;
  const actOpen = (st, index) => {
    const k = actKey(st);
    return state.actOpen[k] != null ? state.actOpen[k] : index === state.stepIndex;
  };

  // 工具调用行 + 展开后的执行细节（命令 / 产物 / 人工动作 / 跳到对应面板）
  function chatActRow(st, index) {
    const wrap = el('div', 'chat-act' + (actOpen(st, index) ? ' is-open' : ''));
    const row = el('button', 'chat-act__row');
    row.type = 'button';
    row.setAttribute('aria-expanded', String(actOpen(st, index)));
    row.appendChild(el('span', 'chat-act__id', st.id));
    row.appendChild(el('span', 'chat-act__text', callSummary(st, index)));
    row.appendChild(el('span', 'chat-act__chevron', '\u203A'));
    const marks = el('span', 'chat-act__marks');
    if (st.evidenceLevel) marks.appendChild(evidenceBadge(st.evidenceLevel));
    if (st.gate) marks.appendChild(gateBadge(st.gate));
    row.appendChild(marks);
    row.addEventListener('click', () => {
      state.actOpen[actKey(st)] = !actOpen(st, index);
      renderChat();
    });
    wrap.appendChild(row);

    const detail = el('div', 'chat-act__detail');
    detail.appendChild(el('div', 'chat-act__task', st.title));
    if (st.userAction) detail.appendChild(el('p', 'chat-act__line', '人工：' + st.userAction));
    if (st.agentAction) detail.appendChild(el('p', 'chat-act__line', 'Agent：' + st.agentAction));
    if (st.toolCall) detail.appendChild(el('code', 'chat-act__cmd', st.toolCall));
    if (st.artifact) detail.appendChild(el('span', 'chat-act__artifact', '→ ' + st.artifact));
    const fresh = newToolsAt(index);
    if (fresh.length) {
      const jump = el('div', 'chat-act__jump');
      jump.appendChild(el('span', 'chat-act__jump-label', '打开面板'));
      fresh.forEach((id) => {
        const meta = TOOL_META[id];
        if (!meta) return;
        const b = el('button', 'chat-act__jump-btn');
        b.type = 'button';
        b.appendChild(agentAvatar(meta.agent, 'chat-act__jump-avatar'));
        b.appendChild(el('span', null, meta.label));
        b.title = '在右侧工具面板全屏查看 ' + meta.label;
        b.addEventListener('click', () => {
          state.selectedAgent = null;
          state.ideToolActive = id;
          state.toolView = 'full';
          state.toolFullId = id;
          renderAgentTools();
          toast('工具面板已切到 ' + meta.label);
        });
        jump.appendChild(b);
      });
      detail.appendChild(jump);
    }
    wrap.appendChild(detail);
    return wrap;
  }

  // 一个 Agent 的连续回合：头像 + 名字，内部是若干组「调用行 + 结论行」
  function chatTurn(group) {
    const agentId = group.agentId;
    const node = el('div', 'chat-turn chat-turn--' + agentId
      + (group.isCurrent ? ' is-current' : '')
      + (state.selectedAgent && agentId !== state.selectedAgent ? ' is-dimmed' : ''));
    node.appendChild(agentAvatar(agentId, 'chat-turn__avatar'));
    const body = el('div', 'chat-turn__body');
    const head = el('div', 'chat-turn__head');
    head.appendChild(el('strong', 'chat-turn__name', AGENT_SHORT_ROLE[agentId] || agentId));
    head.appendChild(el('span', 'chat-turn__id', agentId));
    // 阶段标题就在上一行，首个回合不再重复 phase 标签
    if (!group.newPhase) head.appendChild(el('span', 'chat-turn__phase', group.phase));
    body.appendChild(head);
    if (group.quote) {
      const q = el('div', 'chat-quote');
      q.appendChild(el('span', 'chat-quote__who', group.quoteWho || '人工'));
      q.appendChild(el('span', null, group.quote));
      body.appendChild(q);
    }
    group.items.forEach((it) => {
      if (it.act) body.appendChild(chatActRow(it.act, it.index));
      if (it.say) body.appendChild(el('p', 'chat-turn__say', it.say));
    });
    node.appendChild(body);
    return node;
  }

  function buildPlanMessage() {
    const node = el('div', 'chat-turn chat-turn--lead chat-turn--plan');
    node.appendChild(agentAvatar('lead', 'chat-turn__avatar'));
    const body = el('div', 'chat-turn__body');
    const head = el('div', 'chat-turn__head');
    head.appendChild(el('strong', 'chat-turn__name', AGENT_SHORT_ROLE.lead));
    head.appendChild(el('span', 'chat-turn__id', 'lead'));
    head.appendChild(el('span', 'chat-turn__phase', 'Plan'));
    body.appendChild(head);
    const q = el('div', 'chat-quote');
    q.appendChild(el('span', 'chat-quote__who', '调查目标'));
    q.appendChild(el('span', null, scenarioUi().inquiry));
    body.appendChild(q);
    body.appendChild(el('p', 'chat-turn__say', '按 7 阶段门禁推进，每步只回答本阶段的判据问题。'));
    const chips = el('div', 'chat-plan__phases');
    PHASES.forEach((p, i) => {
      const chip = el('span', 'chat-plan__phase' + (i <= phaseIndex(step().phase) ? ' is-done' : ''), (i + 1) + '.' + p.id);
      chip.title = p.judge;
      chips.appendChild(chip);
    });
    body.appendChild(chips);
    node.appendChild(body);
    return node;
  }

  function chatPhaseHeader(phase) {
    const p = PHASES.find((x) => x.id === phase);
    const row = el('div', 'chat-phase');
    row.appendChild(el('span', 'chat-phase__bar'));
    row.appendChild(el('span', 'chat-phase__label', phase + (p ? ' · ' + p.judge : '')));
    return row;
  }

  function agentActivity(agentId) {
    const owned = steps().map((s, i) => ({ step: s, index: i })).filter((x) => x.step.agentId === agentId);
    const completed = owned.filter((x) => x.index < state.stepIndex).length;
    const isActive = step().agentId === agentId;
    const hasFuture = owned.some((x) => x.index > state.stepIndex);
    let status = 'idle';
    let label = '未参与';
    if (isActive) { status = 'active'; label = '正在执行'; }
    else if (completed && hasFuture) { status = 'waiting'; label = '等待回接'; }
    else if (hasFuture) { status = 'queued'; label = '等待接手'; }
    else if (completed) { status = 'done'; label = '已完成'; }
    return { completed: completed, total: owned.length, status: status, label: label };
  }

  function renderAgentBoard(target) {
    const all = steps();
    const current = step();
    const next = all[state.stepIndex + 1];
    const board = el('section', 'agent-board');
    const overview = el('div', 'agent-board__overview');
    const mission = el('div', 'agent-board__mission');
    mission.appendChild(el('span', 'agent-board__eyebrow', current.phase + ' · 协作态势'));
    mission.appendChild(el('strong', 'agent-board__question', phaseQuestion()));
    overview.appendChild(mission);
    const metrics = el('div', 'agent-board__metrics');
    metrics.appendChild(field('当前负责人', AGENT_SHORT_ROLE[current.agentId] || current.agentId));
    metrics.appendChild(field('证据', unlockedEvidence().length + '/' + (scenario().evidence || []).length));
    metrics.appendChild(field('进度', (state.stepIndex + 1) + '/' + all.length));
    overview.appendChild(metrics);
    board.appendChild(overview);

    const topology = el('div', 'agent-topology');
    AGENTS.forEach((a) => {
      const activity = agentActivity(a.id);
      const node = el('button', 'agent-node agent-node--' + activity.status
        + (state.selectedAgent === a.id ? ' is-selected' : ''));
      node.type = 'button';
      node.setAttribute('aria-pressed', String(state.selectedAgent === a.id));
      node.title = a.name + ' · ' + a.role;
      const head = el('div', 'agent-node__head');
      head.appendChild(agentAvatar(a.id, 'agent-node__avatar'));
      const identity = el('span', 'agent-node__identity');
      identity.appendChild(el('strong', 'agent-node__role', AGENT_SHORT_ROLE[a.id] || a.id));
      identity.appendChild(el('span', 'agent-node__id', a.id));
      head.appendChild(identity);
      node.appendChild(head);
      const progress = el('span', 'agent-node__progress');
      const progressFill = el('span', 'agent-node__progress-fill');
      progressFill.style.width = (activity.total ? Math.round((activity.completed + (activity.status === 'active' ? 0.5 : 0)) / activity.total * 100) : 0) + '%';
      progress.appendChild(progressFill);
      node.appendChild(progress);
      // 状态和步数放同一行的两端，不再和名字挤在头像右边
      const foot = el('div', 'agent-node__foot');
      foot.appendChild(el('span', 'agent-node__status', activity.label));
      foot.appendChild(el('span', 'agent-node__count', activity.completed + '/' + activity.total + ' steps'));
      node.appendChild(foot);
      node.addEventListener('click', () => {
        state.selectedAgent = state.selectedAgent === a.id ? null : a.id;
        toast(state.selectedAgent
          ? '聚焦 ' + a.id + '：对话记录高亮，工具面板只看它的产出'
          : '显示全部 Agent 记录与工具');
        renderAgentView();
      });
      topology.appendChild(node);
    });
    board.appendChild(topology);

    const handoff = el('div', 'agent-handoff');
    const blocked = isGateBlocking(current);
    const route = el('div', 'agent-handoff__route');
    route.appendChild(el('span', 'agent-handoff__agent agent-handoff__agent--current', AGENT_SHORT_ROLE[current.agentId] || current.agentId));
    route.appendChild(el('span', 'agent-handoff__arrow', blocked ? '‖' : (next ? '→' : '✓')));
    route.appendChild(el('span', 'agent-handoff__agent', blocked ? '人工确认' : (next ? (AGENT_SHORT_ROLE[next.agentId] || next.agentId) : '归档')));
    handoff.appendChild(route);
    const copy = el('div', 'agent-handoff__copy');
    const blockedLabel = current.phase === 'Deliver' ? '等待人工签发' : '等待人工确认';
    copy.appendChild(el('strong', null, blocked ? blockedLabel : (next ? (next.agentId === current.agentId ? '继续处理' : '下一次交接') : '调查闭合')));
    copy.appendChild(el('span', null, blocked ? effectiveConfirm(current).text : (next ? next.title : '证据包已具备交付条件')));
    handoff.appendChild(copy);
    handoff.appendChild(el('code', 'agent-handoff__artifact', current.artifact || 'artifact 待生成'));
    board.appendChild(handoff);
    target.appendChild(board);
  }

  // ---------------------------------------------------------------------
  // 对话输入：人工插话 + 按当前回放进度作答（不泄露未解锁事实）
  // ---------------------------------------------------------------------
  const chatTargetAgent = () => state.chatTarget || step().agentId;
  const scenarioMessages = () => {
    if (!state.messages[state.scenarioId]) state.messages[state.scenarioId] = [];
    return state.messages[state.scenarioId];
  };

  function chatSuggestions() {
    const st = step();
    const list = [];
    list.push('这一步为什么这么判断？');
    if (steps()[state.stepIndex + 1]) list.push('下一步准备做什么？');
    list.push('把这一步的证据给我');
    if (isGateBlocking(st)) list.push('这个确认卡的风险在哪？');
    else if (knownRootCause()) list.push('根因和修复怎么对应？');
    return list.slice(0, 4);
  }

  // 回复只用「回放到当前步为止已解锁」的事实拼装 —— 与渐进披露规则保持一致
  function composeReply(text, agentId) {
    const st = step();
    const sc = scenario();
    const short = AGENT_SHORT_ROLE[agentId] || agentId;
    const ownTools = toolsForStep().filter((t) => t.meta.agent === agentId);
    const toolLine = ownTools.length
      ? '相关面板：' + ownTools.map((t) => t.meta.label).join('、')
      : '本步我没有工具产出，可看 ' + (AGENT_SHORT_ROLE[st.agentId] || st.agentId) + ' 的面板。';
    let body;
    if (/为什么|凭什么|why|依据|判断/.test(text)) {
      body = '本 phase 的判据是「' + phaseQuestion() + '」。' + st.id + ' 的动作是：'
        + st.agentAction + ' 证据等级 ' + (st.evidenceLevel || 'E0') + '，结论：' + st.conclusion;
    } else if (/下一步|接下来|然后|next|计划/.test(text)) {
      const nx = steps()[state.stepIndex + 1];
      body = nx
        ? '下一步 ' + nx.id + '（' + nx.phase + '）交给 ' + (AGENT_SHORT_ROLE[nx.agentId] || nx.agentId)
          + '：' + nx.title + '。' + (isGateBlocking(st) ? '需先完成本步人工确认才会启动。' : '')
        : '回放已到最后一步，证据包具备交付条件。';
    } else if (/证据|evidence|日志|log|数据|产物|artifact/.test(text)) {
      body = '本步命令：' + st.toolCall + '；产物：' + (st.artifact || '暂无')
        + '。已解锁证据 ' + unlockedEvidence().length + '/' + (sc.evidence || []).length + ' 条。';
    } else if (/根因|root|修复|fix|怎么改/.test(text)) {
      body = knownRootCause()
        ? '根因：' + knownRootCause() + (knownFix() ? ' 修复：' + knownFix() : ' 修复方案尚未成形。')
        : '根因还没定位。' + (knownFirstAnomaly()
          ? '当前只确认首个异常在：' + knownFirstAnomaly()
          : '现在只能给出症状，继续往下定界才能收敛。');
    } else if (/风险|确认|签发|gate/.test(text)) {
      const conf = effectiveConfirm(st);
      body = conf
        ? '本步是人工门禁：' + conf.text + ' 未确认前我不会把它写进 Baseline。'
        : '本步无需人工确认，gate = ' + (st.gate || 'not-evaluated') + '。';
    } else {
      body = '收到。回放停在 ' + st.id + '（' + st.phase + '）：' + st.title
        + '。当前负责人是 ' + (AGENT_SHORT_ROLE[st.agentId] || st.agentId) + '，结论：' + st.conclusion;
    }
    return { agentId: agentId, short: short, body: body, toolLine: toolLine };
  }

  function sendChatMessage(text) {
    const value = (text || '').trim();
    if (!value) return;
    const target = chatTargetAgent();
    scenarioMessages().push({
      atIndex: state.stepIndex,
      text: value,
      to: target,
      reply: composeReply(value, target),
    });
    stopAutoplay();
    renderChat();
    toast((AGENT_SHORT_ROLE[target] || target) + ' 已回应，工具面板同步高亮');
    const input = $('#chat-input');
    if (input) { input.value = ''; input.style.height = 'auto'; }
  }

  function chatUserMsg(m) {
    const node = el('div', 'chat-turn chat-turn--user');
    node.appendChild(el('span', 'chat-turn__avatar chat-turn__avatar--user', '你'));
    const body = el('div', 'chat-turn__body');
    const head = el('div', 'chat-turn__head');
    head.appendChild(el('strong', 'chat-turn__name', '你'));
    head.appendChild(el('span', 'chat-turn__phase', '@' + m.to));
    body.appendChild(head);
    body.appendChild(el('div', 'chat-bubble', m.text));
    node.appendChild(body);
    return node;
  }

  // 回复也按「引用提问 → 工具调用行 → 结论行」的节奏走，和回放消息同构
  function chatReplyMsg(m) {
    const agentId = m.reply.agentId;
    const node = el('div', 'chat-turn chat-turn--' + agentId + ' chat-turn--reply'
      + (state.selectedAgent && agentId !== state.selectedAgent ? ' is-dimmed' : ''));
    node.appendChild(agentAvatar(agentId, 'chat-turn__avatar'));
    const body = el('div', 'chat-turn__body');
    const head = el('div', 'chat-turn__head');
    head.appendChild(el('strong', 'chat-turn__name', AGENT_SHORT_ROLE[agentId] || agentId));
    head.appendChild(el('span', 'chat-turn__id', agentId));
    head.appendChild(el('span', 'chat-turn__phase', '答复'));
    body.appendChild(head);
    const q = el('div', 'chat-quote');
    q.appendChild(el('span', 'chat-quote__who', '你'));
    q.appendChild(el('span', null, m.text));
    body.appendChild(q);
    const act = el('div', 'chat-act chat-act--static');
    const row = el('div', 'chat-act__row');
    row.appendChild(el('span', 'chat-act__text', m.reply.toolLine));
    act.appendChild(row);
    body.appendChild(act);
    body.appendChild(el('p', 'chat-turn__say', m.reply.body));
    node.appendChild(body);
    return node;
  }

  function renderComposer() {
    const targets = $('#chat-targets');
    if (!targets) return;
    const active = chatTargetAgent();
    targets.innerHTML = '';
    AGENTS.forEach((a) => {
      const isOn = a.id === active;
      const b = el('button', 'chat-target' + (isOn ? ' is-selected' : '')
        + (a.id === step().agentId ? ' is-live' : ''));
      b.type = 'button';
      b.title = '@' + a.id + ' · ' + a.name + '（' + a.role + '）';
      b.setAttribute('aria-pressed', String(isOn));
      b.appendChild(agentAvatar(a.id, 'chat-target__avatar'));
      b.addEventListener('click', () => {
        state.chatTarget = a.id === active ? null : a.id;
        renderComposer();
        const input = $('#chat-input');
        if (input) input.focus();
      });
      targets.appendChild(b);
    });

    const sug = $('#chat-suggest');
    sug.innerHTML = '';
    chatSuggestions().forEach((s) => {
      const b = el('button', 'chat-suggest__chip', s);
      b.type = 'button';
      b.addEventListener('click', () => sendChatMessage(s));
      sug.appendChild(b);
    });

    const input = $('#chat-input');
    input.placeholder = '@' + active + ' 追问本步（' + step().id + ' · ' + step().phase + '）…';
    $('#chat-composer-hint').textContent = '接收：' + (AGENT_SHORT_ROLE[active] || active)
      + ' · ' + active + (state.chatTarget ? '（已手动指定）' : '（跟随当前负责人）')
      + ' · Enter 发送，Shift+Enter 换行';
  }

  // 「正在…已用时 N 秒」这一行 —— 单独刷新，不整块重渲染对话流
  let stepEnteredAt = Date.now();

  function renderLiveLine() {
    const line = document.getElementById('chat-live');
    if (!line) return;
    const all = steps();
    const cur = all[state.stepIndex];
    const next = all[state.stepIndex + 1];
    if (!next) return;
    const secs = Math.max(0, Math.round((Date.now() - stepEnteredAt) / 1000));
    const rest = all.length - (state.stepIndex + 1);
    const nextName = AGENT_SHORT_ROLE[next.agentId] || next.agentId;
    let text;
    if (isGateBlocking(cur)) {
      text = '等待人工确认后继续 · 后续 ' + rest + ' 步未开始 · 已等待 ' + secs + ' 秒…';
    } else if (state.playing) {
      text = nextName + ' 正在接手 ' + next.id + '：' + next.title + '，已用时 ' + secs + ' 秒…';
    } else {
      text = nextName + ' 待接手 ' + next.id + '：' + next.title + ' · 剩余 ' + rest + ' 步';
    }
    line.classList.toggle('is-blocked', isGateBlocking(cur));
    line.classList.toggle('is-running', state.playing && !isGateBlocking(cur));
    line.querySelector('.chat-live__text').textContent = text;
  }

  function renderChat() {
    const team = $('#chat-team');
    team.innerHTML = '';
    renderAgentBoard(team);

    const confirmSlot = $('#chat-confirm');
    confirmSlot.innerHTML = '';
    const conf = effectiveConfirm(step());
    if (conf) confirmSlot.appendChild(buildConfirmCard(step()));

    const flow = $('#agent-chat');
    flow.innerHTML = '';
    flow.appendChild(buildPlanMessage());
    const all = steps();
    // 只渲染到当前步；连续同 Agent 同阶段的步骤合成一个回合，
    // 回合内按「工具调用行 → 结论行」交替，读起来就是一条完成任务的轨迹。
    const groups = [];
    all.slice(0, state.stepIndex + 1).forEach((s2, i) => {
      const last = groups[groups.length - 1];
      if (last && last.agentId === s2.agentId && last.phase === s2.phase && !last.closed) {
        last.items.push({ act: s2, index: i, say: s2.conclusion });
        last.isCurrent = last.isCurrent || i === state.stepIndex;
      } else {
        groups.push({
          agentId: s2.agentId,
          phase: s2.phase,
          newPhase: !last || last.phase !== s2.phase,
          isCurrent: i === state.stepIndex,
          items: [{ act: s2, index: i, say: s2.conclusion }],
        });
      }
      // 人工插话把当前回合截断，后续步骤另起一个回合，时间顺序才不会乱
      const asks = scenarioMessages().filter((m) => m.atIndex === i);
      if (asks.length) {
        groups[groups.length - 1].closed = true;
        asks.forEach((m) => groups.push({ ask: m }));
      }
    });
    groups.forEach((g) => {
      if (g.ask) {
        flow.appendChild(chatUserMsg(g.ask));
        flow.appendChild(chatReplyMsg(g.ask));
        return;
      }
      if (g.newPhase) flow.appendChild(chatPhaseHeader(g.phase));
      flow.appendChild(chatTurn(g));
    });

    const cur0 = all[state.stepIndex];
    const rest = all.length - (state.stepIndex + 1);
    if (rest > 0) {
      const live = el('div', 'chat-live');
      live.id = 'chat-live';
      live.appendChild(el('span', 'chat-live__spinner'));
      live.appendChild(el('span', 'chat-live__text'));
      flow.appendChild(live);
      renderLiveLine();
    }
    renderComposer();
    // 滚动跟随：只渲染到当前步，滚到底即为当前步
    requestAnimationFrame(() => { flow.scrollTop = flow.scrollHeight; });
  }

  // 工具面板顶部的 Agent 过滤条：与对话流的「聚焦 Agent」共用 state.selectedAgent，
  // 点头像即可只看某个 Agent 调起的工具，对话流同步高亮它的发言。
  function renderToolAgentBar() {
    const bar = $('#tool-agent-filter');
    if (!bar) return;
    bar.innerHTML = '';
    const groups = toolAgentGroups();
    const total = toolsForStep().length;

    const allChip = el('button', 'tool-agentchip tool-agentchip--all' + (state.selectedAgent ? '' : ' is-selected'));
    allChip.type = 'button';
    allChip.setAttribute('aria-pressed', String(!state.selectedAgent));
    allChip.appendChild(el('span', 'tool-agentchip__name', '全部'));
    allChip.appendChild(el('span', 'tool-agentchip__count', String(total)));
    allChip.addEventListener('click', () => {
      if (!state.selectedAgent) return;
      state.selectedAgent = null;
      renderAgentView();
    });
    bar.appendChild(allChip);

    groups.forEach((g) => {
      const isSel = state.selectedAgent === g.agentId;
      const isLive = step().agentId === g.agentId;
      const chip = el('button', 'tool-agentchip'
        + (isSel ? ' is-selected' : '') + (isLive ? ' is-live' : ''));
      chip.type = 'button';
      chip.setAttribute('aria-pressed', String(isSel));
      chip.title = (AGENT_SHORT_ROLE[g.agentId] || g.agentId) + ' Agent 调起 '
        + g.tools.length + ' 个工具：' + g.tools.map((t) => t.meta.label).join('、');
      chip.appendChild(agentAvatar(g.agentId, 'tool-agentchip__avatar'));
      chip.appendChild(el('span', 'tool-agentchip__name', AGENT_SHORT_ROLE[g.agentId] || g.agentId));
      chip.appendChild(el('span', 'tool-agentchip__count', String(g.tools.length)));
      chip.addEventListener('click', () => {
        state.selectedAgent = isSel ? null : g.agentId;
        toast(state.selectedAgent
          ? '只看 ' + g.agentId + ' 的工具产出与发言'
          : '显示全部 Agent 的工具与发言');
        renderAgentView();
      });
      bar.appendChild(chip);
    });
  }

  function toolEmptyState() {
    const box = el('div', 'tool-empty');
    box.appendChild(el('div', 'tool-empty__title',
      (AGENT_SHORT_ROLE[state.selectedAgent] || state.selectedAgent) + ' 在本步没有工具产出'));
    box.appendChild(el('div', 'tool-empty__body', '当前由 '
      + (AGENT_SHORT_ROLE[step().agentId] || step().agentId) + ' 执行 ' + step().id + '。'));
    const back = el('button', 'btn tool-empty__back', '显示全部工具');
    back.type = 'button';
    back.addEventListener('click', () => { state.selectedAgent = null; renderAgentView(); });
    box.appendChild(back);
    return box;
  }

  function renderAgentTools() {
    renderToolAgentBar();
    const grid = $('#agent-tool-grid');
    grid.innerHTML = '';
    const tools = visibleTools();
    const tileBtn = $('#tool-view-tile');
    const fullBtn = $('#tool-view-full');
    tileBtn.classList.toggle('is-selected', state.toolView === 'tile');
    fullBtn.classList.toggle('is-selected', state.toolView === 'full');
    tileBtn.setAttribute('aria-pressed', String(state.toolView === 'tile'));
    fullBtn.setAttribute('aria-pressed', String(state.toolView === 'full'));
    grid.classList.toggle('is-full', state.toolView === 'full');
    // 只剩一张卡时铺满整行，避免右半边留一块空白
    grid.classList.toggle('is-single', state.toolView === 'tile' && tools.length === 1);

    const meta = $('#agent-tool-meta');
    if (meta) {
      meta.textContent = state.selectedAgent
        ? state.selectedAgent + ' · ' + tools.length + '/' + toolsForStep().length + ' tools'
        : toolAgentGroups().length + ' agents · ' + tools.length + ' tools';
    }

    if (!tools.length) {
      grid.classList.remove('is-full');
      grid.appendChild(toolEmptyState());
      return;
    }

    if (state.toolView === 'full') {
      const id = tools.some((t) => t.id === state.toolFullId) ? state.toolFullId : tools[0].id;
      state.toolFullId = id;
      const full = el('div', 'tool-full');
      const back = el('button', 'btn btn-icon tool-tile__btn');
      back.appendChild(icon('grid'));
      back.type = 'button';
      back.title = '返回平铺';
      back.setAttribute('aria-label', '返回平铺');
      back.addEventListener('click', () => { state.toolView = 'tile'; renderAgentTools(); });
      full.appendChild(toolAttrStrip(id, { actions: back }));
      full.appendChild(el('div', 'tool-full__body')).appendChild(buildTool(id));
      grid.appendChild(full);
      return;
    }

    tools.forEach((t) => {
      const attr = toolAttribution(t.id);
      const collapsed = !!state.toolCollapsed[t.id];
      const tile = el('section', 'tool-tile tool-tile--' + attr.status
        + ' tool-tile--kind-' + t.meta.kind + (collapsed ? ' is-collapsed' : ''));

      const actions = el('div', 'tool-tile__actions');
      const foldBtn = el('button', 'btn btn-icon tool-tile__btn');
      foldBtn.appendChild(icon(collapsed ? 'unfold' : 'collapse'));
      foldBtn.type = 'button';
      foldBtn.title = (collapsed ? '展开 ' : '收起 ') + t.meta.label;
      foldBtn.setAttribute('aria-expanded', String(!collapsed));
      foldBtn.addEventListener('click', () => {
        state.toolCollapsed[t.id] = !collapsed;
        renderAgentTools();
      });
      actions.appendChild(foldBtn);
      const expandBtn = el('button', 'btn btn-icon tool-tile__btn');
      expandBtn.appendChild(icon('expand'));
      expandBtn.type = 'button';
      expandBtn.title = '全屏查看 ' + t.meta.label;
      expandBtn.setAttribute('aria-label', '全屏查看 ' + t.meta.label);
      expandBtn.addEventListener('click', () => { state.toolView = 'full'; state.toolFullId = t.id; renderAgentTools(); });
      actions.appendChild(expandBtn);

      tile.appendChild(toolAttrStrip(t.id, { actions: actions }));
      if (!collapsed) {
        const body = el('div', 'tool-tile__body');
        body.appendChild(buildTool(t.id));
        tile.appendChild(body);
      }
      grid.appendChild(tile);
    });
  }

  // ---------------------------------------------------------------------
  // 视图组装
  // ---------------------------------------------------------------------
  function renderIdeGate() {
    const slot = $('#ide-gate-slot');
    slot.innerHTML = '';
    const conf = effectiveConfirm(step());
    const show = conf && !isConfirmed(step());
    slot.hidden = !show;
    if (show) slot.appendChild(buildConfirmCard(step()));
  }

  // 状态条跟着当前步走：诊断计数 + 光标落在锚点算子那一行
  function renderStatusStrip() {
    const diag = $('#status-diagnostic');
    const lncol = $('#status-lncol');
    const meta = $('#explorer-meta');
    if (meta) meta.textContent = 'workspace · ' + (buildArtifacts().length + WORKSPACE.sources.length) + ' items';
    if (!diag || !lncol) return;
    const st = step();
    const blocking = isGateBlocking(st);
    diag.textContent = blocking ? '1 diagnostic · 等待人工确认'
      : (st.gate === 'warn' ? '1 warning · ' + st.id : 'DSL 即时诊断运行中');
    diag.classList.toggle('is-warn', blocking || st.gate === 'warn');
    const rows = editorLines();
    const idx = rows.findIndex((r) => r.isHead);
    const line = idx >= 0 ? rows[idx].n : (rows[0] && rows[0].n) || 1;
    lncol.textContent = 'Ln ' + line + ', Col 1';
  }

  function renderIdeView() {
    renderFileTree();
    renderStatusStrip();
    renderIdeGate();
    renderEditor();
    renderTerminalLog();
    renderTerminalState();
    renderIdeTools();
  }

  function renderAgentView() {
    renderTaskList();
    renderChat();
    renderAgentTools();
  }

  // ---------------------------------------------------------------------
  // Top bar / taskbar / stepper / playback
  // ---------------------------------------------------------------------
  function renderModeButtons() {
    $$('.demo-mode-btn').forEach((b) => {
      const active = b.dataset.mode === state.mode;
      b.classList.toggle('is-selected', active);
      b.setAttribute('aria-pressed', String(active));
      b.setAttribute('aria-selected', String(active));
    });
  }

  function initCaseSelect() {
    const sel = $('#case-switch');
    sel.innerHTML = '';
    SCENARIOS.forEach((s) => {
      const ui = SCENARIO_UI[s.scenarioId] || {};
      const opt = el('option', null, (ui.taskId || s.scenarioId) + ' · ' + (ui.replayTitle || s.title));
      opt.value = s.scenarioId;
      opt.title = s.title;
      sel.appendChild(opt);
    });
  }

  function syncCaseSelect() {
    $('#case-switch').value = state.scenarioId;
  }

  function renderTaskbar() {
    $('#task-id').textContent = scenarioUi().taskId;
    $('#task-title').textContent = displayTitle();
  }

  function renderViewCue() {
    const cue = $('#view-cue');
    const preferred = step().view;
    const label = preferred === 'ide' ? 'IDE' : 'Agent';
    // 已经在该视图时不提示 ——「当前步骤 · IDE」只是重复顶栏的 tab 状态；
    // 只有需要切过去看时才出现，这时它是个可点的动作。
    const actionable = (preferred === 'ide' || preferred === 'agent') && state.mode !== preferred;
    cue.hidden = !actionable;
    if (!actionable) return;
    cue.textContent = '转到 ' + label + ' 查看';
    cue.classList.add('is-action');
    cue.disabled = false;
    cue.dataset.targetMode = preferred;
    cue.title = '本步骤主要在 ' + label + ' 视图进行，点击切换';
  }

  function renderStepper() {
    const track = $('#stepper-track');
    track.innerHTML = '';
    const curIdx = phaseIndex(step().phase);
    PHASES.forEach((p, i) => {
      const label = phaseLabel(p);
      const b = el('button', 'demo-step'
        + (i === curIdx ? ' is-active' : '')
        + (i < curIdx ? ' is-done' : ''));
      b.appendChild(el('span', 'demo-step__idx', String(i + 1)));
      b.appendChild(el('span', 'demo-step__label', label));
      b.type = 'button';
      b.setAttribute('role', 'tab');
      b.setAttribute('aria-selected', String(i === curIdx));
      b.tabIndex = i === curIdx ? 0 : -1;
      b.title = label + '：' + p.judge;
      if (i === curIdx) b.setAttribute('aria-current', 'step');
      b.addEventListener('click', () => {
        const target = steps().findIndex((s) => s.phase === p.id);
        const next = target >= 0 ? target : state.stepIndex;
        if (next === state.stepIndex) { toast('已在阶段 ' + label); return; }
        const blocker = firstBlockerBefore(next);
        if (blocker >= 0) {
          setStep(blocker);
          nudgeConfirm();
          toast('需先完成人工确认：' + steps()[blocker].title);
          return;
        }
        setStep(next);
        toast('跳转到阶段 ' + label);
      });
      track.appendChild(b);
    });
  }

  function renderPlayback() {
    $('#playback-counter').textContent = 'Step ' + (state.stepIndex + 1) + ' / ' + steps().length;
    const play = $('#step-play');
    play.textContent = state.playing ? '⏸' : '▶';
    play.title = state.playing ? '暂停' : '自动播放';
    play.setAttribute('aria-label', state.playing ? '暂停' : '自动播放');
    play.classList.toggle('is-selected', state.playing);

    // 门禁读数：把「为什么走不动」写在控制键旁边，而不是只靠 toast
    const gate = $('#transport-gate');
    const st = step();
    const blocked = isGateBlocking(st);
    if (gate) {
      gate.classList.toggle('is-blocked', blocked);
      gate.textContent = blocked
        ? (isRejected(st) ? '⛔ 已驳回 · 需重新确认' : '⏸ 待人工确认')
        : '';
    }
    const next = $('#step-next');
    if (next) {
      next.disabled = blocked;
      next.title = blocked ? '此步需人工确认后才能继续' : '下一步（→）';
    }
  }

  function renderStatus() {
    // 顶栏精简后，环境指纹落在底部轨道 —— 它是「可复现」叙事的必要信息，不能丢
    const fp = (UI.topbar && UI.topbar.envFingerprintChip) || {};
    const env = fp.display || '';
    const fields = fp.fields || {};
    const short = [fields.platform, 'pypto ' + fields.pypto, 'CANN ' + fields.cann, 'dev×' + fields.device_count]
      .filter(Boolean).join(' · ');
    const node = $('#status-readout');
    node.textContent = 'CASE REPLAY · RUN + CODE' + (short ? ' · ' + short : '');
    node.title = '环境指纹 · ' + env + '　|　当前：'
      + (state.mode === 'ide' ? 'IDE VIEW' : 'Agent Windows')
      + ' · ' + scenario().scenarioId + ' · ' + step().phase;
  }

  function syncViews() {
    $$('.demo-view').forEach((v) => {
      v.hidden = v.dataset.view !== state.mode;
    });
  }

  // ---------------------------------------------------------------------
  // 状态迁移
  // ---------------------------------------------------------------------
  function setStep(i) {
    const next = clamp(i);
    if (next !== state.stepIndex) stepEnteredAt = Date.now();
    state.stepIndex = next;
    state.lastStepByScenario[state.scenarioId] = state.stepIndex;
    state.ideToolActive = null;
    render();
  }

  function stopAutoplay() {
    if (autoplayTimer) {
      clearInterval(autoplayTimer);
      autoplayTimer = null;
    }
    state.playing = false;
  }

  function togglePlay() {
    if (state.playing) {
      stopAutoplay();
      renderPlayback();
      toast('自动播放已暂停');
      return;
    }
    if (isGateBlocking(step())) {
      nudgeConfirm();
      toast('此步需人工确认后才能继续');
      return;
    }
    state.playing = true;
    renderPlayback();
    toast('自动播放已启动 · 约 3s/步');
    autoplayTimer = setInterval(() => {
      if (isGateBlocking(step())) {
        stopAutoplay();
        renderPlayback();
        nudgeConfirm();
        toast('遇到人工确认节点，已暂停');
        return;
      }
      if (state.stepIndex >= steps().length - 1) {
        stopAutoplay();
        renderPlayback();
        toast('已播放到末尾');
        return;
      }
      setStep(state.stepIndex + 1);
    }, 3000);
  }

  function advance() {
    if (isGateBlocking(step())) {
      stopAutoplay();
      renderPlayback();
      nudgeConfirm();
      toast('此步需人工确认后才能继续');
      return;
    }
    if (state.stepIndex >= steps().length - 1) {
      toast('已到最后一步');
      return;
    }
    setStep(state.stepIndex + 1);
    toast('下一步：' + step().title);
  }

  function retreat() {
    if (state.stepIndex <= 0) {
      toast('已是第一步');
      return;
    }
    setStep(state.stepIndex - 1);
    toast('上一步：' + step().title);
  }

  function switchScenario(id) {
    if (!SCENARIOS.some((s) => s.scenarioId === id)) return;
    state.lastStepByScenario[state.scenarioId] = state.stepIndex;
    state.scenarioId = id;
    state.stepIndex = state.lastStepByScenario[id] || 0;
    state.selectedAgent = null;
    state.evidenceFilter = null;
    state.ideToolActive = null;
    state.toolView = 'tile';
    state.toolFullId = null;
    stopAutoplay();
    render();
    toast('切换案例：' + scenarioUi().taskId + ' · 已恢复到 Step ' + (state.stepIndex + 1));
  }

  function switchMode(mode) {
    if (mode !== 'ide' && mode !== 'agent') return;
    if (state.mode === mode) {
      toast('已在 ' + (mode === 'ide' ? 'IDE VIEW' : 'Agent Windows'));
      return;
    }
    state.mode = mode;
    render();
    toast('切换到 ' + (mode === 'ide' ? 'IDE VIEW' : 'Agent Windows') + '（当前步骤保持）');
  }

  // 窗口宽度阈值：innerWidth 为 0（离屏 / 预览容器尚未布局）时不按窄屏处理，
  // 否则两侧面板会在正常宽度下被误收起。
  function isNarrowViewport() {
    const w = window.innerWidth || 0;
    return w > 0 && w <= 960;
  }

  function reset() {
    const currentScenario = state.scenarioId;
    state.mode = 'ide';
    state.stepIndex = 0;
    state.lastStepByScenario[currentScenario] = 0;
    state.playing = false;
    state.selectedAgent = null;
    state.evidenceFilter = null;
    state.terminalOpen = true;
    state.ideToolActive = null;
    state.toolView = 'tile';
    state.toolFullId = null;
    state.toolCollapsed = {};
    state.actOpen = {};
    state.chatTarget = null;
    stepEnteredAt = Date.now();
    state.messages[currentScenario] = [];
    state.leftOpen = !isNarrowViewport();
    state.rightOpen = !isNarrowViewport();
    Object.keys(state.confirmed).forEach((key) => { if (key.indexOf(currentScenario + '/') === 0) delete state.confirmed[key]; });
    Object.keys(state.rejected).forEach((key) => { if (key.indexOf(currentScenario + '/') === 0) delete state.rejected[key]; });
    stopAutoplay();
    render();
    toast('已重置当前案例：' + scenarioUi().taskId + ' · Define');
  }

  // ---------------------------------------------------------------------
  // 组装渲染
  // ---------------------------------------------------------------------
  function render() {
    renderModeButtons();
    syncCaseSelect();
    renderTaskbar();
    renderViewCue();
    renderStepper();
    renderPlayback();
    renderIdeView();
    renderAgentView();
    renderStatus();
    syncViews();
    applyPanelStates();
    renderWindowControls();
  }

  function initSplitPanes() {
    if (!window.PtoWorkbenchShell || !window.PtoWorkbenchShell.initResizablePanes) return;
    $$('.ide-layout').forEach((layout) => {
      const panes = Array.from(layout.children).filter((el) => el.classList.contains('ide-pane'));
      if (panes.length < 2) return;
      const isAgent = layout.closest('#agent-view') != null;
      window.PtoWorkbenchShell.initResizablePanes({
        root: layout,
        panes: panes,
        direction: 'horizontal',
        sizes: [18, 54, 28],
        minSize: [180, 400, 300],
        gutterSize: 10,
        storageKey: 'pypto-studio-split-' + (isAgent ? 'agent' : 'ide'),
        gutterLabel: '调整相邻面板宽度',
      });
    });
  }

  function setPaneVisibility(pane, open) {
    if (!pane) return;
    pane.hidden = !open;
    const next = pane.nextElementSibling;
    const prev = pane.previousElementSibling;
    if (next && next.classList.contains('pto-workbench-shell__split-gutter')) next.hidden = !open;
    if (prev && prev.classList.contains('pto-workbench-shell__split-gutter')) prev.hidden = !open;
  }

  function applyPanelStates() {
    $$('.demo-view .ide-layout').forEach((layout) => {
      const panes = Array.from(layout.children).filter((el) => el.classList.contains('ide-pane'));
      if (panes.length >= 3) {
        setPaneVisibility(panes[0], state.leftOpen);
        setPaneVisibility(panes[panes.length - 1], state.rightOpen);
      }
    });
  }

  function renderWindowControls() {
    const left = $('#wc-left');
    const right = $('#wc-right');
    const bottom = $('#wc-bottom');
    if (left) { left.classList.toggle('is-on', state.leftOpen); left.setAttribute('aria-pressed', String(state.leftOpen)); }
    if (right) { right.classList.toggle('is-on', state.rightOpen); right.setAttribute('aria-pressed', String(state.rightOpen)); }
    if (bottom) {
      bottom.classList.toggle('is-on', state.terminalOpen);
      bottom.setAttribute('aria-pressed', String(state.terminalOpen));
      bottom.disabled = state.mode === 'agent';
      bottom.title = state.mode === 'agent' ? '底部面板（终端）仅在 IDE VIEW 可用' : '底部面板（终端）';
    }
  }

  function toggleLeftPanel() {
    state.leftOpen = !state.leftOpen;
    if (state.leftOpen && isNarrowViewport()) state.rightOpen = false;
    applyPanelStates();
    renderWindowControls();
    toast(state.leftOpen ? '左侧面板已打开' : '左侧面板已关闭');
  }

  function toggleRightPanel() {
    state.rightOpen = !state.rightOpen;
    if (state.rightOpen && isNarrowViewport()) state.leftOpen = false;
    applyPanelStates();
    renderWindowControls();
    toast(state.rightOpen ? '右侧面板已打开' : '右侧面板已关闭');
  }

  function init() {
    if (isNarrowViewport()) {
      state.leftOpen = false;
      state.rightOpen = false;
    }
    initCaseSelect();
    render();
    initSplitPanes();
    applyPanelStates();
    renderWindowControls();

    setInterval(renderLiveLine, 1000);

    $$('.demo-mode-btn').forEach((b) => b.addEventListener('click', () => switchMode(b.dataset.mode)));
    $('#case-switch').addEventListener('change', (e) => switchScenario(e.target.value));
    $('#reset-btn').addEventListener('click', reset);
    $('#view-cue').addEventListener('click', (e) => {
      const targetMode = e.currentTarget.dataset.targetMode;
      if (targetMode) switchMode(targetMode);
    });
    $('#step-prev').addEventListener('click', retreat);
    $('#step-play').addEventListener('click', togglePlay);
    $('#step-next').addEventListener('click', advance);
    $('#terminal-toggle').addEventListener('click', toggleTerminal);
    $('#terminal-header').addEventListener('click', (e) => {
      if (e.target.closest('button')) return;
      toggleTerminal();
    });
    $('#tool-view-tile').addEventListener('click', () => { state.toolView = 'tile'; renderAgentTools(); });
    $('#tool-view-full').addEventListener('click', () => { state.toolView = 'full'; state.toolFullId = null; renderAgentTools(); });
    $('#wc-left').addEventListener('click', toggleLeftPanel);
    $('#wc-right').addEventListener('click', toggleRightPanel);

    const composer = $('#chat-composer');
    const chatInput = $('#chat-input');
    composer.addEventListener('submit', (e) => {
      e.preventDefault();
      sendChatMessage(chatInput.value);
    });
    chatInput.addEventListener('keydown', (e) => {
      if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        sendChatMessage(chatInput.value);
      }
    });
    // 输入框随内容长高，最多 4 行，避免把对话流挤没
    chatInput.addEventListener('input', () => {
      chatInput.style.height = 'auto';
      chatInput.style.height = Math.min(chatInput.scrollHeight, 96) + 'px';
    });
    $('#wc-bottom').addEventListener('click', () => {
      if (state.mode !== 'ide') { toast('底部面板（终端）仅在 IDE VIEW 可用'); return; }
      toggleTerminal();
    });

    // 窄屏默认收起两侧面板；跨过断点时重新套用默认值，避免窗口变宽后面板还空着
    const narrowMq = window.matchMedia('(max-width: 960px)');
    const applyBreakpoint = (isNarrow) => {
      state.leftOpen = !isNarrow;
      state.rightOpen = !isNarrow;
      applyPanelStates();
      renderWindowControls();
    };
    if (narrowMq.addEventListener) narrowMq.addEventListener('change', (e) => applyBreakpoint(e.matches));
    else if (narrowMq.addListener) narrowMq.addListener((e) => applyBreakpoint(e.matches));

    window.addEventListener('keydown', (e) => {
      const tag = (e.target && e.target.tagName ? e.target.tagName : '').toLowerCase();
      if (tag === 'input' || tag === 'select' || tag === 'textarea') return;
      if (e.key === 'ArrowLeft') {
        e.preventDefault();
        retreat();
      } else if (e.key === 'ArrowRight') {
        e.preventDefault();
        advance();
      } else if ((e.key === ' ' || e.key === 'Spacebar') && tag !== 'button') {
        e.preventDefault();
        togglePlay();
      }
    });
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init, { once: true });
  } else {
    init();
  }
})();
