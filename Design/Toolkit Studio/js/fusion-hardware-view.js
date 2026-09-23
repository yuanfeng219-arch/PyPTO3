/**
 * 融合的数据流变化 · 昇腾硬件抽象架构图
 *
 * 替换方案面板里「覆盖的子链路」讲的是「这个 API 一次吃掉了哪几段计算」。
 * 这一层再往下问一句：吃掉之后，数据在硬件上少跑了哪些路？
 *
 * 用的是设计系统里现成的 memory-architecture 抽象（GM 存储墙 + AIC/AIV 核），
 * 数据流的变化不是画上去的示意，而是从方案的子链路推出来的：
 *
 *   融合前 —— 每段子链路是一个独立 kernel，各自 GM 读一次、写一次
 *   融合后 —— 整段只有首尾各一次 GM 往返，中间的 Cube ↔ Vector 交接留在片上
 *
 * 每段子链路走 Cube 还是 Vector，由它的语义角色决定（见 fusion-rules-data 的 ROLES.unit）。
 */
(function registerFusionHardwareView() {
  'use strict';

  const CANVAS_ID = 'fusionHwCanvas';

  const state = {
    open: false,
    item: null,
    mode: 'before',   // 'before' | 'after'
    step: 0,
    playing: false,
    speed: 1,
    crossings: 0,     // 本次演示累计穿越 GM 的次数
    lastFocus: null,
  };

  const STEP_MS = 1500;   // 每段停留时长（1× 速度）
  let timer = null;

  let root = null;
  let preset = null;
  let overlay = null;
  let zoom = null;

  const rules = () => window.PtoFusionRules;
  const mem = () => window.PtoMemoryArchitecturePattern;
  const esc = (v) => String(v).replace(/[&<>"']/g, (ch) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[ch]));

  const AIC_ID = 'fusHwAic';
  const AIV_ID = 'fusHwAiv';
  const GM_SELECTOR = '[data-mem950-node="rail:GM"]';
  const CORE_IN = { cube: `#${AIC_ID} [data-aic-node="buffer:L1"]`, vector: `#${AIV_ID} [data-aiv-node="cache:ND-DMA Cache"]` };
  const CORE_OUT = { cube: `#${AIC_ID} [data-aic-node="buffer:L0C"]`, vector: `#${AIV_ID} [data-aiv-node="buffer:UB"]` };
  const CORE_ROOT = { cube: `#${AIC_ID}`, vector: `#${AIV_ID}` };

  /* ---------------- 从方案推出硬件视角的执行序列 ---------------- */

  /**
   * 每段子链路 → { text, role, roleMeta, unit, io }
   * unit 决定它落在 Cube 还是 Vector；io 表示这一段本身就要碰 GM（cache 写入、通信）
   */
  function sequenceOf(item) {
    const covered = (item.coverage && item.coverage.covered) || [];
    return covered.map((e) => ({
      ...e,
      unit: (e.roleMeta && e.roleMeta.unit) || 'vector',
      io: !!(e.roleMeta && e.roleMeta.io),
    }));
  }

  /** 相邻两段跨了执行单元 = 一次 Cube/Vector 交接 */
  function handoffs(seq) {
    let n = 0;
    for (let i = 1; i < seq.length; i += 1) if (seq[i].unit !== seq[i - 1].unit) n += 1;
    return n;
  }

  function metricsOf(seq) {
    const n = seq.length;
    return {
      steps: n,
      launchBefore: n,
      launchAfter: 1,
      // 融合前每段各读一次写一次；融合后整段只有首尾各一次
      gmBefore: n * 2,
      gmAfter: 2,
      eliminated: Math.max(0, n - 1),
      handoffBefore: 0,      // 融合前每次交接都退化成一次 GM 往返
      handoffAfter: handoffs(seq),
      units: [...new Set(seq.map((s) => s.unit))],
    };
  }

  /* ---------------- 硬件预设 ----------------
   * 路由随模式重建：标签上的 ×N 就是这条路径被走过的次数，
   * 「融合前 ×5 / 融合后 ×1」本身就是这次改动的全部内容。
   */

  function buildPreset(item, seq, mode) {
    const m = metricsOf(seq);
    const routes = [];
    const cores = [];

    const useCube = m.units.includes('cube');
    const useVector = m.units.includes('vector');
    if (useCube) cores.push({ id: AIC_ID, kind: 'aic', title: 'AIC · Cube', presetKey: 'aicDraftV1' });
    if (useVector) cores.push({ id: AIV_ID, kind: 'aiv', title: 'AIV · Vector', presetKey: 'aivOfficialV1' });

    const loadRoute = (unit, label) => ({
      id: 'gm-to-' + unit,
      label,
      tone: 'transport',
      from: GM_SELECTOR,
      to: CORE_IN[unit],
      fromSide: 'right',
      toSide: 'left',
      style: 'lane-h-target',
      labelDy: -18,
    });
    const storeRoute = (unit, label) => ({
      id: unit + '-to-gm',
      label,
      tone: 'directReturn',
      from: CORE_OUT[unit],
      to: GM_SELECTOR,
      fromSide: 'left',
      toSide: 'right',
      style: 'lane-h-source',
      labelDy: 18,
    });
    const handoffRoute = (from, to, label) => ({
      id: from + '-to-' + to,
      label,
      tone: 'direct',
      from: CORE_OUT[from],
      to: CORE_IN[to],
      fromSide: 'right',
      toSide: 'right',
      labelDx: 18,
      labelDy: -12,
    });

    if (mode === 'before') {
      // 每个执行单元上有几段子链路，GM 就被穿越几次
      m.units.forEach((unit) => {
        const k = seq.filter((s) => s.unit === unit).length;
        routes.push(loadRoute(unit, `MTE2 · load ×${k}`));
        routes.push(storeRoute(unit, `MTE3 · store ×${k}`));
      });
    } else {
      const first = seq[0];
      const last = seq[seq.length - 1];
      if (first) routes.push(loadRoute(first.unit, 'MTE2 · load ×1'));
      if (last) routes.push(storeRoute(last.unit, 'MTE3 · store ×1'));
      // 中间的跨单元交接从 GM 往返降级为片上传递
      const pairs = new Map();
      for (let i = 1; i < seq.length; i += 1) {
        if (seq[i].unit === seq[i - 1].unit) continue;
        const key = seq[i - 1].unit + '>' + seq[i].unit;
        pairs.set(key, (pairs.get(key) || 0) + 1);
      }
      pairs.forEach((count, key) => {
        const [from, to] = key.split('>');
        routes.push(handoffRoute(from, to, `片上交接 ×${count}`));
      });
    }

    return {
      id: 'fusion-hw-' + item.id + '-' + mode,
      name: item.title + ' · ' + mode,
      rails: [{
        key: 'GM',
        label: 'GM / DDR',
        tone: 'memory-shell',
        grid: { rows: 26, cols: 4, cellSize: 12, gap: 4, shape: 'hex' },
      }],
      cores,
      routes,
      hoverTips: {
        'rail:GM': {
          title: 'GM / DDR',
          body: mode === 'before'
            ? `融合前每段子链路都要在这里读一次、写一次，共 ${m.gmBefore} 次穿越。`
            : '融合后只有整段的输入读入与最终结果写回，共 2 次穿越。',
        },
        'core:AIC': { title: 'AIC · Cube', body: '投影、GEMM、注意力矩阵乘在这里执行。' },
        'core:AIV': { title: 'AIV · Vector', body: '归一化、激活、RoPE、量化与路由打分在这里执行。' },
        'buffer:L1': { title: 'L1', body: 'Cube 侧的输入驻留；融合后上游 tile 可以直接留在这里。' },
        'buffer:L0C': { title: 'L0C', body: 'Cube 的累加输出；融合后不必写回 GM 就能交给 Vector。' },
        'buffer:UB': { title: 'Unified Buffer', body: 'Vector 侧的工作区；融合后整条子链路的中间量都停在这里。' },
        'cache:ND-DMA Cache': { title: 'ND-DMA Cache', body: 'MTE2 从 GM 搬入的落点。' },
      },
    };
  }

  /* ---------------- 数据流动画 ----------------
   * 两层：路径上的行进虚线给方向和节奏，圆点是「一包数据」本身。
   * 圆点用 SVG animateMotion 跟着 mpath 跑 —— 覆盖层重算路径时只改 d，
   * 动画会自动贴上新几何，不需要我们在 rAF 里逐帧算坐标。
   */

  const TONE_COLOR = { transport: "#ffdf1f", directReturn: "#29c7a6", direct: "#4d97ff" };

  function decorateRoutes() {
    const svg = overlay && overlay.svg;
    if (!svg || !preset) return;
    (preset.routes || []).forEach((route) => {
      const group = svg.querySelector(`[data-route-id="${route.id}"]`);
      const path = group && group.querySelector(`path`);
      if (!group || !path || group.querySelector(`.kf-hw-packet`)) return;
      const pathId = `kfhw-${preset.id}-${route.id}`;
      path.setAttribute(`id`, pathId);
      path.classList.add(`kf-hw-line`);
      const color = TONE_COLOR[route.tone] || TONE_COLOR.transport;
      const packet = document.createElementNS(`http://www.w3.org/2000/svg`, `circle`);
      packet.setAttribute(`class`, `kf-hw-packet`);
      packet.setAttribute(`r`, `4.5`);
      packet.setAttribute(`fill`, color);
      const motion = document.createElementNS(`http://www.w3.org/2000/svg`, `animateMotion`);
      motion.setAttribute(`dur`, `1.6s`);
      motion.setAttribute(`repeatCount`, `indefinite`);
      motion.setAttribute(`rotate`, `auto`);
      const mpath = document.createElementNS(`http://www.w3.org/2000/svg`, `mpath`);
      mpath.setAttribute(`href`, `#${pathId}`);
      mpath.setAttributeNS(`http://www.w3.org/1999/xlink`, `xlink:href`, `#${pathId}`);
      motion.appendChild(mpath);
      packet.appendChild(motion);
      // 圆点放在标签之前，避免盖住文字
      group.insertBefore(packet, group.lastElementChild);
    });
    applySpeed();
  }

  /** 速度只改动画时长，不重建 DOM */
  function applySpeed() {
    const svg = overlay && overlay.svg;
    if (!svg) return;
    const dur = (1.6 / state.speed).toFixed(2) + `s`;
    svg.querySelectorAll(`animateMotion`).forEach((m) => m.setAttribute(`dur`, dur));
    svg.style.setProperty(`--kf-hw-dash-dur`, (1.1 / state.speed).toFixed(2) + `s`);
  }

  /** 走完一段要穿越几次 GM —— 融合前每段都是一读一写，融合后只有首尾 */
  function crossingsAt(index, seq) {
    if (state.mode === `before`) return 2;
    const last = seq.length - 1;
    if (last === 0) return 2;
    return (index === 0 || index === last) ? 1 : 0;
  }

  function updateCounter() {
    if (!root) return;
    const el = root.querySelector(`[data-hw-counter]`);
    if (el) el.textContent = String(state.crossings);
    // 自动播放时是整轮累计；手点某一段时就是这一段自己的次数
    const label = root.querySelector(`[data-hw-counter-label]`);
    if (label) label.textContent = state.playing ? '本轮已穿越 GM' : '这一段穿越 GM';
  }

  function stopTimer() {
    if (timer) { clearInterval(timer); timer = null; }
  }

  function play() {
    const seq = sequenceOf(state.item);
    if (!seq.length) return;
    stopTimer();
    state.playing = true;
    // 从头开始一轮，计数器同时归零
    state.crossings = 0;
    goToStep(0, seq);
    timer = setInterval(() => {
      const next = state.step + 1;
      if (next >= seq.length) {
        // 一轮走完，停一拍再从头开始，让「融合前 vs 融合后」的节奏差看得出来
        state.crossings = 0;
        goToStep(0, seq);
        return;
      }
      goToStep(next, seq);
    }, STEP_MS / state.speed);
    syncPlayButton();
  }

  function pause() {
    stopTimer();
    state.playing = false;
    syncPlayButton();
  }

  function syncPlayButton() {
    const btn = root && root.querySelector(`[data-hw-play]`);
    if (btn) {
      btn.textContent = state.playing ? `⏸ 暂停` : `▶ 自动播放`;
      btn.classList.toggle(`is-playing`, state.playing);
    }
    root?.classList.toggle(`is-playing`, state.playing);
  }

  /** 推进到某一段：高亮 + 累计 GM 穿越 */
  function goToStep(index, seq) {
    state.crossings += crossingsAt(index, seq);
    focusStep(index, seq);
    updateCounter();
  }

  /* ---------------- 渲染 ---------------- */

  function renderMetrics(m) {
    const cell = (label, before, after, note, good) => `<div class="kf-hw-metric ${good ? 'is-good' : ''}">
      <span>${esc(label)}</span>
      <b><i class="is-before">${before}</i><em>→</em><i class="is-after">${after}</i></b>
      <small>${esc(note)}</small>
    </div>`;
    return `<div class="kf-hw-metrics">
      ${cell('GM 往返', m.gmBefore + ' 次', m.gmAfter + ' 次', `消除 ${m.eliminated} 段中间物化`, true)}
      ${cell('kernel launch', m.launchBefore + ' 次', m.launchAfter + ' 次', '一次调用完成整段', true)}
      ${cell('Cube/Vector 交接', m.handoffBefore + ' 次片上', m.handoffAfter + ' 次片上', m.handoffAfter ? '原本要绕 GM，现在留在片上' : '整段在同一执行单元内', m.handoffAfter > 0)}
      ${cell('执行单元', m.units.length + ' 个', m.units.length + ' 个', m.units.map((u) => rules().UNITS[u].label).join(' + '), false)}
    </div>`;
  }

  function renderSteps(seq) {
    const last = seq.length - 1;
    return `<ol class="kf-hw-steps">${seq.map((s, i) => {
      const unit = rules().UNITS[s.unit];
      const gm = state.mode === 'before' ? '读 + 写 GM' : (i === 0 ? '读 GM' : i === last ? '写 GM' : '片上');
      const onchip = state.mode === 'after' && i !== 0 && i !== last;
      return `<li class="kf-hw-step ${i === state.step ? 'is-active' : ''} ${onchip ? 'is-onchip' : ''}" style="--role:${s.roleMeta.color}">
        <button type="button" data-hw-step="${i}">
          <i class="kf-hw-step__idx">${i + 1}</i>
          <span class="kf-hw-step__text">${esc(s.text)}</span>
          <em class="kf-hw-step__unit is-${s.unit}">${esc(unit.label)}</em>
          <em class="kf-hw-step__gm ${onchip ? 'is-onchip' : ''}">${esc(gm)}</em>
        </button>
      </li>`;
    }).join('')}</ol>`;
  }

  function shell(item, seq) {
    const m = metricsOf(seq);
    const api = (item.apis && item.apis[0]) || '候选算子';
    return `<div class="kf-hw-shell" role="dialog" aria-modal="true" aria-label="融合数据流 · 昇腾硬件抽象架构">
      <header class="kf-hw-head">
        <div class="kf-hw-title">
          <span class="kf-hw-eyebrow">HARDWARE DATA PATH</span>
          <h2>${esc(item.title)}</h2>
          <p><code>${esc(api)}</code> · GM ⇄ AIC/AIV 抽象架构 · ${m.steps} 段子链路</p>
        </div>
        <div class="kf-hw-headactions">
          <div class="kf-hw-modeswitch" role="group" aria-label="融合前后">
            <button type="button" data-hw-mode="before" class="${state.mode === 'before' ? 'is-active' : ''}">融合前 · ${m.launchBefore} 个 kernel</button>
            <button type="button" data-hw-mode="after" class="${state.mode === 'after' ? 'is-active' : ''}">融合后 · 1 个 kernel</button>
          </div>
          <div class="kf-hw-play" role="group" aria-label="数据流动画">
            <button type="button" class="kf-hw-playbtn" data-hw-play>▶ 自动播放</button>
            <div class="kf-hw-speed">${[0.5, 1, 2].map((v) => `<button type="button" data-hw-speed="${v}" class="${state.speed === v ? 'is-active' : ''}">${v}×</button>`).join('')}</div>
          </div>
          <button class="kf-hw-btn" type="button" data-hw-fit>适应</button>
          <span class="kf-hw-zoom" data-hw-zoom-readout>—</span>
          <button class="kf-hw-close" type="button" data-hw-close aria-label="关闭">✕</button>
        </div>
      </header>
      ${renderMetrics(m)}
      <div class="kf-hw-body">
        <aside class="kf-hw-rail">
          <div class="kf-hw-railhead">子链路执行序列<small>点一段看它在硬件上的路径</small></div>
          <div class="kf-hw-counter">
            <span data-hw-counter-label>这一段穿越 GM</span><b data-hw-counter>0</b><em>次</em>
            <i>整段共 ${state.mode === 'before' ? m.gmBefore : m.gmAfter} 次</i>
          </div>
          ${renderSteps(seq)}
          <p class="kf-hw-note">${state.mode === 'before'
    ? '融合前每段都是独立 kernel：算完必须写回 GM，下一段再读回来。'
    : '融合后整段只在开头读一次、结尾写一次；中间结果停在 L0C / UB，跨执行单元时直接片上传递。'}</p>
        </aside>
        <div class="kf-hw-stage">
          <div class="kf-hw-viewport" data-hw-viewport>
            <div class="kf-hw-sizer" data-hw-sizer>
              <div class="kf-hw-canvas" id="${CANVAS_ID}"></div>
            </div>
          </div>
          <div class="kf-hw-legend">
            <span><i class="is-transport"></i>MTE2 · GM → 片上</span>
            <span><i class="is-return"></i>MTE3 · 片上 → GM</span>
            <span><i class="is-direct"></i>片上交接（不经 GM）</span>
          </div>
        </div>
      </div>
    </div>`;
  }

  /* ---------------- 硬件图 ---------------- */

  function mountDiagram(item, seq) {
    const canvas = root.querySelector('#' + CANVAS_ID);
    const viewport = root.querySelector('[data-hw-viewport]');
    const sizer = root.querySelector('[data-hw-sizer]');
    if (!mem() || !canvas) return;

    preset = buildPreset(item, seq, state.mode);
    mem().renderArchitecture(canvas, preset);
    overlay = mem().createRouteOverlay(canvas, preset);
    mem().attachHoverInteractions?.(canvas, preset, {
      selector: `${GM_SELECTOR}, #${AIC_ID}, #${AIC_ID} [data-aic-node], #${AIV_ID}, #${AIV_ID} [data-aiv-node]`,
    });
    zoom = mem().createZoomController({
      root,
      viewport,
      sizer,
      canvas,
      defaultZoom: 0.34,
      min: 0.16,
      max: 1.2,
      step: 0.08,
      pan: true,
      wheelZoom: false,
      readout: '[data-hw-zoom-readout]',
    });
    fit();
    decorateRoutes();
    state.crossings = 0;
    goToStep(state.step, seq);
    if (state.playing) play();
  }

  function fit() {
    const canvas = root && root.querySelector('#' + CANVAS_ID);
    const viewport = root && root.querySelector('[data-hw-viewport]');
    const graph = canvas && canvas.querySelector('.pto-mem950');
    if (!graph || !viewport || !zoom) return;
    const w = (viewport.clientWidth - 20) / Math.max(graph.scrollWidth, 1);
    const h = (viewport.clientHeight - 20) / Math.max(graph.scrollHeight, 1);
    zoom.setZoom(Math.max(0.16, Math.min(1.05, w, h)));
    zoom.center?.();
    overlay?.render?.();
  }

  /** 当前这一段子链路在硬件上走哪条路 */
  function focusFor(index, seq) {
    const s = seq[index];
    if (!s) return { selectors: [], routes: [] };
    const unit = s.unit;
    const last = seq.length - 1;

    if (state.mode === 'before') {
      // 独立 kernel：读进来、算、写回去
      return {
        selectors: [GM_SELECTOR, CORE_ROOT[unit], CORE_IN[unit], CORE_OUT[unit]],
        routes: ['gm-to-' + unit, unit + '-to-gm'],
      };
    }

    // 只有一段时，它同时是入口和出口
    if (last === 0) {
      return {
        selectors: [GM_SELECTOR, CORE_ROOT[unit], CORE_IN[unit], CORE_OUT[unit]],
        routes: ['gm-to-' + unit, unit + '-to-gm'],
      };
    }
    if (index === 0) return { selectors: [GM_SELECTOR, CORE_ROOT[unit], CORE_IN[unit]], routes: ['gm-to-' + unit] };
    if (index === last) return { selectors: [CORE_ROOT[unit], CORE_OUT[unit], GM_SELECTOR], routes: [unit + '-to-gm'] };

    const prev = seq[index - 1].unit;
    if (prev !== unit) {
      // 跨执行单元：这一步原本要绕 GM，现在是片上交接
      return { selectors: [CORE_ROOT[prev], CORE_OUT[prev], CORE_ROOT[unit], CORE_IN[unit]], routes: [prev + '-to-' + unit] };
    }
    // 同一个单元内接着算，完全不动 GM
    return { selectors: [CORE_ROOT[unit], CORE_OUT[unit]], routes: [] };
  }

  function focusStep(index, seq) {
    const canvas = root && root.querySelector('#' + CANVAS_ID);
    if (!canvas || !preset) return;
    state.step = index;
    mem().setPathFocus(canvas, preset, focusFor(index, seq));
    root.querySelectorAll('[data-hw-step]').forEach((btn) => {
      btn.parentElement.classList.toggle('is-active', Number(btn.dataset.hwStep) === index);
    });
  }

  /* ---------------- 生命周期 ---------------- */

  function render() {
    const item = state.item;
    if (!item || !root) return;
    const seq = sequenceOf(item);
    root.innerHTML = shell(item, seq);
    // renderArchitecture 依赖布局尺寸，等一帧再挂（这里用 setTimeout，
    // 面板在后台时 rAF 会被暂停）
    setTimeout(() => mountDiagram(item, seq), 0);
  }

  function open(item) {
    const host = document.getElementById('modelArchitectureView');
    if (!host || !item || !rules() || !mem()) return false;
    const seq = sequenceOf(item);
    if (!seq.length) return false;
    if (!root) {
      root = document.createElement('div');
      root.className = 'kf-hw-root';
      root.id = 'fusionHardware';
      host.appendChild(root);
      root.addEventListener('click', onClick);
    }
    state.lastFocus = document.activeElement;
    state.item = item;
    state.mode = 'before';
    state.step = 0;
    state.open = true;
    root.hidden = false;
    host.classList.add('is-hardware-open');
    render();
    return true;
  }

  function close() {
    if (!root) return;
    pause();
    state.open = false;
    root.hidden = true;
    document.getElementById('modelArchitectureView')?.classList.remove('is-hardware-open');
    if (state.lastFocus && state.lastFocus.isConnected) state.lastFocus.focus();
  }

  function onClick(event) {
    const t = event.target;
    if (t.closest('[data-hw-close]')) { close(); return; }
    if (t.closest('[data-hw-fit]')) { fit(); return; }

    const mode = t.closest('[data-hw-mode]');
    if (mode) {
      stopTimer();
      state.mode = mode.dataset.hwMode;
      state.step = 0;
      render();   // mountDiagram 挂好后会按 state.playing 续播
      return;
    }

    if (t.closest('[data-hw-play]')) {
      if (state.playing) pause(); else play();
      return;
    }

    const speed = t.closest('[data-hw-speed]');
    if (speed) {
      state.speed = Number(speed.dataset.hwSpeed) || 1;
      root.querySelectorAll('[data-hw-speed]').forEach((b) => b.classList.toggle('is-active', Number(b.dataset.hwSpeed) === state.speed));
      applySpeed();
      if (state.playing) play();
      return;
    }

    const step = t.closest('[data-hw-step]');
    if (step) {
      pause();
      const seq = sequenceOf(state.item);
      state.crossings = 0;
      goToStep(Number(step.dataset.hwStep), seq);
    }
  }

  document.addEventListener('keydown', (event) => {
    if (event.key === 'Escape' && state.open) { event.stopPropagation(); close(); }
  });

  window.PtoFusionHardwareView = { open, close, isOpen: () => state.open };
})();
