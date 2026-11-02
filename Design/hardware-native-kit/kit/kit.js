/* ============================================================================
 * Hardware-Native Kit · 公共入口
 * ----------------------------------------------------------------------------
 *   HNKit.mount(el, { kernel, platform })   挂载三区联动工作台
 *   inst.setKernel(kernel)                  换一个 kernel，骨架不动
 *   HNKit.check(kernel, platform)           只跑校验，返回诊断（给 CI 用）
 *
 * 宿主页面只需要一个空的容器元素；三区的 DOM 由 kit 自己搭。
 * ========================================================================== */
(function (root) {
  'use strict';

  const C = root.HNKitCore, V = root.HNKitView;
  const CHIP_VIEW = { w: 720, h: 470 };

  const SHELL = ''
    + '<div class="kb-app">'
    + '  <header class="kb-top">'
    + '    <span class="kb-brand">HARDWARE-NATIVE LENS</span><span class="kb-mode">KIT</span>'
    + '    <span class="kb-crumb" data-kb="crumb"></span>'
    + '    <span class="kb-spacer"></span>'
    + '    <span class="kb-target" data-kb="target"></span>'
    + '    <span class="kb-switch" data-kb="switch"></span>'
    + '    <button class="kb-toggle" type="button" data-kb="toggle" aria-expanded="false" aria-controls="kbDock">显示判定</button>'
    + '  </header>'
    + '  <main class="kb-main">'
    + '    <section class="kb-pane kb-src">'
    + '      <header class="kb-pane-head"><span>源码</span><small data-kb="src-meta"></small></header>'
    + '      <div class="kb-pane-body" data-kb="src"></div>'
    + '      <div class="kb-code-resize" role="separator" tabindex="0" aria-label="调整源码区宽度" aria-orientation="vertical" title="拖动或使用左右方向键调整源码区宽度"></div>'
    + '    </section>'
    + '    <section class="kb-mid">'
    + '    <section class="kb-pane kb-logic">'
    + '      <header class="kb-pane-head"><span>算子逻辑</span><small data-kb="logic-meta"></small></header>'
    + '      <div class="kb-pane-body kb-svg" data-kb="logic"></div>'
    + '    </section>'
    + '    <div class="kb-splitter" role="separator" tabindex="0" aria-label="调整算子逻辑与硅片落点高度" aria-orientation="horizontal" title="拖动或使用上下方向键调整高度"></div>'
    + '    <section class="kb-pane kb-chip">'
    + '      <header class="kb-pane-head"><span>硅片落点</span><small data-kb="chip-meta"></small></header>'
    + '      <div class="kb-pane-body kb-svg" data-kb="chip"></div>'
    + '    </section>'
    + '    <div class="kb-caption" data-kb="caption" aria-live="polite"></div>'
    + '    </section>'
    + '  <section class="kb-dock" id="kbDock" hidden>'
    + '    <div class="kb-dock-col">'
    + '      <span class="kb-dock-label">参数</span>'
    + '      <div class="kb-knobs" data-kb="knobs"></div>'
    + '    </div>'
    + '    <div class="kb-dock-col kb-dock-grow">'
    + '      <span class="kb-dock-label">片上空间判定 <small>编码期估算 · 未经编译 · 未上板</small></span>'
    + '      <div class="kb-verdicts" data-kb="verdicts"></div>'
    + '    </div>'
    + '  </section>'
    + '  </main>'
    + '  <footer class="kb-status"><span data-kb="status"></span><span data-kb="evidence">编码期估算 · 未经编译 · 未上板</span></footer>'
    + '</div>';

  function mount(host, opts) {
    const platforms = root.HNKitPlatforms;
    host.innerHTML = SHELL;
    const $ = function (name) { return host.querySelector('[data-kb="' + name + '"]'); };
    const app = host.querySelector('.kb-app');
    $('toggle').addEventListener('click', function () {
      const dock = host.querySelector('.kb-dock');
      dock.hidden = !dock.hidden;
      this.setAttribute('aria-expanded', String(!dock.hidden));
      this.textContent = dock.hidden ? '显示判定' : '隐藏判定';
    });

    const codeResize = host.querySelector('.kb-code-resize');
    const main = host.querySelector('.kb-main');
    const codePane = host.querySelector('.kb-src');
    function resizeCode(width) {
      const minMid = window.innerWidth <= 1280 ? 360 : 560;
      const dock = host.querySelector('.kb-dock');
      const dockWidth = !dock.hidden && window.innerWidth > 1280 ? dock.offsetWidth : 0;
      const min = window.innerWidth <= 1280 ? 280 : 340;
      main.style.setProperty('--kb-code-width', Math.max(min, Math.min(Math.max(min, main.clientWidth - minMid - dockWidth), width)) + 'px');
    }
    codeResize.addEventListener('pointerdown', function (ev) {
      const startX = ev.clientX, startWidth = codePane.offsetWidth;
      codeResize.setPointerCapture(ev.pointerId);
      function move(e) { resizeCode(startWidth + e.clientX - startX); }
      function end() { codeResize.removeEventListener('pointermove', move); codeResize.removeEventListener('pointerup', end); codeResize.removeEventListener('pointercancel', end); }
      codeResize.addEventListener('pointermove', move); codeResize.addEventListener('pointerup', end); codeResize.addEventListener('pointercancel', end);
    });
    codeResize.addEventListener('keydown', function (ev) {
      if (ev.key !== 'ArrowLeft' && ev.key !== 'ArrowRight') return;
      ev.preventDefault(); resizeCode(codePane.offsetWidth + (ev.key === 'ArrowLeft' ? -10 : 10));
    });
    const splitter = host.querySelector('.kb-splitter');
    const logicPane = host.querySelector('.kb-logic');
    const mid = host.querySelector('.kb-mid');
    function resizeLogic(height) {
      const max = Math.max(140, mid.clientHeight - 240);
      logicPane.style.flexBasis = Math.max(140, Math.min(max, height)) + 'px';
    }
    splitter.addEventListener('pointerdown', function (ev) {
      const startY = ev.clientY, startHeight = logicPane.offsetHeight;
      splitter.setPointerCapture(ev.pointerId);
      function move(e) { resizeLogic(startHeight + e.clientY - startY); }
      function end() { splitter.removeEventListener('pointermove', move); splitter.removeEventListener('pointerup', end); splitter.removeEventListener('pointercancel', end); }
      splitter.addEventListener('pointermove', move);
      splitter.addEventListener('pointerup', end);
      splitter.addEventListener('pointercancel', end);
    });
    splitter.addEventListener('keydown', function (ev) {
      if (ev.key !== 'ArrowUp' && ev.key !== 'ArrowDown') return;
      ev.preventDefault();
      resizeLogic(logicPane.offsetHeight + (ev.key === 'ArrowUp' ? -10 : 10));
    });

    const state = {
      kernel: null, platform: null, cfg: {}, consts: null,
      anchors: null, idx: null, graph: null, cap: null, boxes: null,
      hover: null, pin: null
    };

    // ── 换 kernel：重算锚点、索引、图布局，源码区重绘一次 ────────────────────
    function setKernel(kernel) {
      state.kernel = kernel;
      state.platform = platforms[kernel.platform];
      if (!state.platform) throw new Error('未知平台 ' + kernel.platform);
      state.cfg = {};
      (kernel.knobs || []).forEach(function (k) {
        state.cfg[k.id] = kernel.consts[k.id] != null ? kernel.consts[k.id] : k.options[0];
      });
      state.anchors = C.resolveAnchors(kernel);
      state.idx = V.buildIndex(kernel, state.platform, state.anchors);
      state.graph = C.layoutGraph(kernel);
      state.hover = null; state.pin = null;

      $('crumb').innerHTML = C.esc(kernel.file.split(/[\\/]/).pop()) + '<i>/</i><b>' + C.esc(kernel.entry) + '</b>';
      $('target').textContent = '目标 ' + state.platform.label;
      $('target').title = state.platform.note || '';
      $('src-meta').textContent = kernel.file.split(/[\\/]/).pop() + ' · '
        + kernel.source[0].no + '–' + kernel.source[kernel.source.length - 1].no + ' · ' + kernel.source.length + ' 行代码';
      $('logic-meta').textContent = kernel.entry + ' · 循环嵌套与计算链';
      $('chip-meta').textContent = kernel.compiledAllocations ? '编译地址 · Vec 地址条按比例 · 源码映射为推断' : '示意 · 不按比例 · Vec 地址条按比例';
      $('evidence').textContent = kernel.compiledAllocations ? '编译地址 · 来源为提供的 pass 文件 · 未上板验证' : '编码期估算 · 未经编译 · 未上板';
      $('knobs').innerHTML = V.renderKnobs(kernel, state.cfg);
      recompute();
    }

    // ── 改参数：只重算容量，逻辑图与硅片重绘 ────────────────────────────────
    function recompute() {
      const k = state.kernel;
      state.consts = C.resolveConsts(k, state.cfg);
      state.cap = C.capacity(k, state.platform, state.consts);
      state.boxes = C.layoutBoxes(state.platform, k, CHIP_VIEW);
      $('src').innerHTML = V.renderSource(k, state.idx, state.consts);
      $('logic').innerHTML = V.renderGraph(state.graph, k, state.cap, state.consts);
      $('chip').innerHTML = V.renderChip(state.boxes, state.platform, state.cap, k, state.consts);
      $('verdicts').innerHTML = V.renderVerdicts(state.cap, state.platform);
      host.querySelector('.kb-dock-label small').textContent = k.compiledAllocations ? '编译地址 · 平台容量为示例口径 · 未上板' : '编码期估算 · 未经编译 · 未上板';
      renderStatus();
      applyHot();
    }

    function renderStatus() {
      const diag = state.anchors.diagnostics.concat(state.cap.diagnostics);
      const worst = state.cap.worst;
      let html = '<span class="kb-st is-' + worst + '">' + V.VERDICT_TEXT[worst] + '</span>';
      if (diag.length) {
        html += '<span class="kb-st is-diag">' + diag.length + ' 条 manifest 校验提示</span>'
          + '<span class="kb-diag">' + diag.slice(0, 3).map(function (d) {
            return C.esc(d.what + '：' + d.msg);
          }).join(' · ') + (diag.length > 3 ? ' …' : '') + '</span>';
      } else {
        html += '<span class="kb-st-dim">manifest 锚点全部命中源码</span>';
      }
      $('status').innerHTML = html;
    }

    // ── 联动：四类对象互相点亮 ──────────────────────────────────────────────
    function applyHot() {
      const f = V.focusFor(state.idx, state.hover || state.pin);
      app.classList.toggle('is-focusing', !!(state.hover || state.pin));
      app.querySelectorAll('.kb-line').forEach(function (el) {
        el.classList.toggle('is-hot', f.lines.indexOf(Number(el.dataset.line)) >= 0);
      });
      app.querySelectorAll('.kb-region').forEach(function (el) {
        el.classList.toggle('is-hot', f.regions.indexOf(el.dataset.region) >= 0);
      });
      app.querySelectorAll('.kb-buf').forEach(function (el) {
        el.classList.toggle('is-hot', f.bufs.indexOf(el.dataset.buf) >= 0);
      });
      app.querySelectorAll('.kb-hit').forEach(function (el) {
        el.classList.toggle('is-hot', f.nodes.indexOf(el.dataset.node) >= 0);
      });
      app.querySelectorAll('.kb-edge').forEach(function (el) {
        el.classList.toggle('is-hot',
          f.nodes.indexOf(el.dataset.from) >= 0 && f.nodes.indexOf(el.dataset.to) >= 0);
      });
      $('caption').innerHTML = V.caption(state.idx, state.cap, state.kernel, state.consts,
        state.hover || state.pin);
    }

    // buf 在 region 里面，node 在 svg 里面 —— 由内向外判定
    function targetOf(el) {
      if (!el || !el.closest) return null;
      const buf = el.closest('[data-buf]');
      if (buf) return { type: 'buf', id: buf.dataset.buf };
      const node = el.closest('[data-node]');
      if (node) return { type: 'node', id: node.dataset.node };
      const region = el.closest('[data-region]');
      if (region) return { type: 'region', id: region.dataset.region };
      const line = el.closest('[data-line]');
      if (line) return { type: 'line', no: Number(line.dataset.line) };
      return null;
    }

    host.addEventListener('mouseover', function (ev) {
      const t = targetOf(ev.target);
      if (!t) { if (state.hover) { state.hover = null; applyHot(); } return; }
      state.hover = t; applyHot();
    });
    host.addEventListener('mouseleave', function () { state.hover = null; applyHot(); });
    host.addEventListener('focusin', function (ev) {
      const t = targetOf(ev.target);
      if (t) { state.hover = t; applyHot(); }
    });
    host.addEventListener('focusout', function (ev) {
      if (!ev.relatedTarget || !targetOf(ev.relatedTarget)) { state.hover = null; applyHot(); }
    });
    host.addEventListener('keydown', function (ev) {
      if (ev.key === 'Escape') { state.pin = null; state.hover = null; applyHot(); return; }
      if ((ev.key === 'Enter' || ev.key === ' ') && targetOf(ev.target) && !ev.target.closest('button, select')) {
        ev.preventDefault();
        const t = targetOf(ev.target);
        state.pin = state.pin && JSON.stringify(state.pin) === JSON.stringify(t) ? null : t;
        applyHot();
      }
    });
    host.addEventListener('click', function (ev) {
      const t = targetOf(ev.target);
      if (!t) return;
      const key = JSON.stringify(t), cur = state.pin ? JSON.stringify(state.pin) : null;
      state.pin = key === cur ? null : t;
      applyHot();
    });
    host.addEventListener('change', function (ev) {
      const sel = ev.target.closest && ev.target.closest('[data-knob]');
      if (!sel) return;
      state.cfg[sel.dataset.knob] = Number(sel.value);
      recompute();
      const back = host.querySelector('[data-knob="' + sel.dataset.knob + '"]');
      if (back) back.focus({ preventScroll: true });
    });

    if (opts && opts.kernels) {
      $('switch').innerHTML = opts.kernels.map(function (k, i) {
        return '<button type="button" data-pick="' + i + '"'
          + (i === 0 ? ' aria-pressed="true"' : ' aria-pressed="false"') + '>'
          + C.esc(k.entry) + '</button>';
      }).join('');
      $('switch').addEventListener('click', function (ev) {
        const b = ev.target.closest('[data-pick]');
        if (!b) return;
        setKernel(opts.kernels[Number(b.dataset.pick)]);
        $('switch').querySelectorAll('[data-pick]').forEach(function (el) {
          el.setAttribute('aria-pressed', String(el === b));
        });
      });
    }

    function setConfig(cfg) {
      (state.kernel.knobs || []).forEach(function (knob) {
        if (Object.prototype.hasOwnProperty.call(cfg, knob.id) && knob.options.indexOf(cfg[knob.id]) >= 0) state.cfg[knob.id] = cfg[knob.id];
      });
      $('knobs').innerHTML = V.renderKnobs(state.kernel, state.cfg);
      recompute();
    }
    setKernel((opts && opts.kernel) || (opts.kernels && opts.kernels[0]));
    return { setKernel: setKernel, setConfig: setConfig, state: state };
  }

  // manifest 体检：锚点是否全部命中源码、形状常量是否都有定义。
  // 源码一改，这里会先报出来，而不是页面上静默高亮错的行。
  function check(kernel, platform) {
    const consts = C.resolveConsts(kernel, {});
    const a = C.resolveAnchors(kernel);
    const cap = C.capacity(kernel, platform || root.HNKitPlatforms[kernel.platform], consts);
    return {
      kernel: kernel.id,
      ok: !a.diagnostics.length && !cap.diagnostics.length,
      diagnostics: a.diagnostics.concat(cap.diagnostics),
      worst: cap.worst
    };
  }

  root.HNKit = { mount: mount, check: check, CHIP_VIEW: CHIP_VIEW };
})(typeof window !== 'undefined' ? window : this);
