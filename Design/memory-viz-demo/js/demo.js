/*
 * Ascend Memory Studio — demo glue.
 *
 * Product-page responsibilities only: domain data, pane content, commands.
 * All chrome/visuals come from PTO patterns:
 *   - ide-frame          → page shell (already initialized by initAll)
 *   - memory-reuse-viewer → the space-time occupancy map (MVP + P1 reuse)
 *   - swimlane-task       → runtime execution trace bars (P4)
 *   - floating-playback   → playback chrome (P4)
 */
(function () {
  'use strict';

  const MV = window.MEMVIZ;
  const KB = MV.KB;

  // Data-viz encodings (documented exemption): tensor kinds + engine lanes.
  const KIND_COLOR = { resident: '#58a6ff', temp: '#3fb950', loop: '#f0883e' };
  const LANE_COLOR = { mte2: '#4369ef', cube: '#7c8db8', vec: '#04a37a', mte3: '#f0883e' };
  const HEALTH_COLOR = { ok: 'var(--success)', warning: 'var(--warning)', error: 'var(--danger)' };

  const TABS = [
    { id: 'mvp', label: 'MVP · 占用图' },
    { id: 'p1', label: 'P1 · 复用与流水' },
    { id: 'p2', label: 'P2 · 对比与源码' },
    { id: 'p4', label: 'P4 · 运行时' },
  ];

  const state = {
    kernelKey: MV.order[0],
    tab: 'mvp',
    diffSide: 'current',
    selected: null, // tensor object
    step: 0,
    playing: false,
    timerId: null,
  };

  // ---- formatting -------------------------------------------------------
  const fmtKB = (b) => ((b / KB) % 1 === 0 ? `${b / KB}KB` : `${(b / KB).toFixed(1)}KB`);
  const fmtHex = (b) => '0x' + b.toString(16).toUpperCase().padStart(5, '0');
  const esc = (s) => String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;');

  const $ = (sel, root = document) => root.querySelector(sel);

  function kernel() {
    return MV.kernels[state.kernelKey];
  }

  function activeMemory() {
    if (state.tab === 'p2') {
      return state.diffSide === 'baseline' ? MV.diff.baseline : MV.diff.current;
    }
    return kernel().memory;
  }

  // ---- stats ------------------------------------------------------------
  function bufferStats(memory) {
    const out = {};
    memory.buffers.forEach((buf) => {
      const ts = memory.tensors.filter((t) => t.buffer === buf.name);
      const usage = new Array(memory.ticks + 1).fill(0);
      ts.forEach((t) => {
        for (let k = t.allocTick; k < t.freeTick; k++) usage[k] += t.size;
      });
      let peak = 0;
      let peakTick = 0;
      usage.forEach((u, i) => { if (u > peak) { peak = u; peakTick = i; } });
      const highWater = ts.reduce((m, t) => Math.max(m, t.offset + t.size), 0);
      out[buf.name] = {
        capacity: buf.capacity,
        peak,
        peakTick,
        highWater,
        util: peak / buf.capacity,
        count: ts.length,
        overflow: highWater > buf.capacity,
        reused: ts.filter((t) => t.reuseOf || (t.reusedBy && t.reusedBy.length)).length,
      };
    });
    return out;
  }

  function liveAt(memory, tick) {
    // returns {bytes, count} of live tensors across all buffers at a tick
    let bytes = 0;
    let count = 0;
    memory.tensors.forEach((t) => {
      if (tick >= t.allocTick && tick < t.freeTick) { bytes += t.size; count += 1; }
    });
    return { bytes, count };
  }

  // ---- the memory-reuse-viewer (MVP + P1 core) --------------------------
  let viewer = null;
  const viewerHost = $('#mv-viewer');

  function mountViewer() {
    const mem = activeMemory();
    if (viewer && viewer.destroy) viewer.destroy();
    viewer = window.PtoMemoryReuseViewer.render(viewerHost, mem, { initialBuffer: mem.buffers[0].name });
    $('#mv-preview-meta').textContent = mem.kernel + (state.tab === 'p2' ? ` · ${activeMemory().ref}` : '');
  }

  // viewer emits bubbling events for host-owned detail / source
  viewerHost.addEventListener('pto-memory-reuse-tensor-select', (e) => {
    state.selected = e.detail.tensor || null;
    renderInspector();
    // map → source: keep the full-file panel in sync when it is open
    if (isSourceOpen() && state.selected) {
      highlightInSource(state.selected);
      scrollSourceTo(state.selected.srcHotLine || state.selected.srcLineStart, true);
    }
  });
  viewerHost.addEventListener('pto-memory-reuse-open-source', (e) => {
    state.selected = e.detail.tensor || state.selected;
    setTab('p2');
    if (state.selected && viewer) viewer.selectTensor(state.selected.id);
  });

  // ---- P2 full-file source panel (bidirectional map ↔ source) -----------
  let srcState = { fileName: null, side: null, lineToTensor: {}, ranges: [] };

  const isSourceOpen = () => { const el = $('#mv-src'); return el && !el.hidden; };
  const setSourceOpen = (open) => { const el = $('#mv-src'); if (el) el.hidden = !open; };

  function findTensor(id) {
    return activeMemory().tensors.find((t) => t.id === id) || null;
  }

  // lightweight CCE/C++ syntax coloring (data-viz-style exemption)
  function hlLine(raw) {
    const e = esc(raw);
    const idx = e.indexOf('//');
    let code = idx >= 0 ? e.slice(0, idx) : e;
    const comment = idx >= 0 ? '<span class="c">' + e.slice(idx) + '</span>' : '';
    code = code
      .replace(/\b(constexpr|inline|for|class|public|private|using|namespace|return|extern|__aicore__|__global__)\b/g, '<span class="k">$1</span>')
      .replace(/\b(LocalTensor|GlobalTensor|GM_ADDR|TPipe|TQue|half|float|int|void|RoundMode|MmadParams|HardEvent)\b/g, '<span class="t">$1</span>')
      .replace(/\b(DataCopy|Mmad|Add|Relu|Cast|Muls|Softmax|EnQue|DeQue|AllocTensor|FreeTensor|Get|SetFlag|WaitFlag|PipeBarrier|InitBuffer|SetGlobalBuffer)\b/g, '<span class="f">$1</span>');
    return code + comment;
  }

  function buildLineMap(mem, file) {
    const map = {};
    const ranges = [];
    mem.tensors.forEach((t) => {
      if (t.srcFile !== file.name) return;
      ranges.push({ id: t.id, start: t.srcLineStart, end: t.srcLineEnd, hot: t.srcHotLine });
      for (let l = t.srcLineStart; l <= t.srcLineEnd; l++) if (!map[l]) map[l] = t.id;
    });
    ranges.sort((a, b) => a.start - b.start);
    srcState = { fileName: file.name, side: state.diffSide, lineToTensor: map, ranges };
  }

  function renderSourceFile(file) {
    const code = $('#mv-src-code');
    const lines = file.content.split('\n');
    if (lines.length && lines[lines.length - 1] === '') lines.pop();
    code.innerHTML = lines.map((ln, i) => {
      const no = i + 1;
      const tid = srcState.lineToTensor[no];
      const t = tid ? findTensor(tid) : null;
      const tag = t ? `<span class="tag" style="color:${KIND_COLOR[t.kind]}">◂ ${esc(t.name)}</span>` : '';
      return `<div class="mv-src-line${tid ? ' mapped' : ''}" data-line="${no}" data-tid="${tid || ''}">`
        + `<span class="gut">${no}</span>`
        + `<span class="txt">${hlLine(ln) || ' '}${tag}</span></div>`;
    }).join('');
    // source → map: clicking a mapped line drives the viewer's own selection
    code.querySelectorAll('.mv-src-line').forEach((el) => {
      el.addEventListener('click', () => {
        const tid = el.dataset.tid;
        if (tid && viewer) viewer.selectTensor(tid);
      });
    });
  }

  function highlightInSource(t) {
    const code = $('#mv-src-code');
    if (!code) return;
    code.querySelectorAll('.mv-src-line').forEach((el) => {
      el.classList.remove('range', 'hot');
      const no = Number(el.dataset.line);
      if (t && no >= t.srcLineStart && no <= t.srcLineEnd) el.classList.add('range');
      if (t && no === t.srcHotLine) el.classList.add('hot');
    });
    const status = $('#mv-src-status');
    if (status) {
      status.innerHTML = t
        ? `<b>${esc(t.name)}</b> · ${t.buffer} · 行 ${t.srcLineStart}-${t.srcLineEnd} · 热行 ${t.srcHotLine}`
        : '选择一个 tensor 以联动源码';
    }
  }

  function scrollSourceTo(line, center) {
    const el = document.querySelector(`.mv-src-line[data-line="${line}"]`);
    if (el) el.scrollIntoView({ block: center ? 'center' : 'nearest', behavior: 'smooth' });
  }

  function openSource(tensor) {
    const mem = activeMemory();
    const file = mem.sourceFiles && mem.sourceFiles[0];
    if (!file) return;
    setSourceOpen(true);
    if (srcState.fileName !== file.name || srcState.side !== state.diffSide) {
      buildLineMap(mem, file);
      renderSourceFile(file);
    }
    $('#mv-src-name').textContent = file.name;
    let t = tensor;
    if (t && t.srcFile !== file.name) t = mem.tensors[0];
    if (t) {
      highlightInSource(t);
      scrollSourceTo(t.srcHotLine || t.srcLineStart, true);
    }
  }

  function navSource(dir) {
    if (!srcState.ranges.length) return;
    const cur = state.selected ? state.selected.id : null;
    let idx = srcState.ranges.findIndex((r) => r.id === cur);
    idx = idx < 0 ? (dir > 0 ? 0 : srcState.ranges.length - 1) : idx + dir;
    idx = (idx + srcState.ranges.length) % srcState.ranges.length;
    if (viewer) viewer.selectTensor(srcState.ranges[idx].id);
  }

  // ---- explorer ---------------------------------------------------------
  function renderExplorer() {
    const host = $('#mv-explorer');
    const kernelRows = MV.order.map((key) => {
      const k = MV.kernels[key];
      const sel = key === state.kernelKey ? ' is-selected' : '';
      return `
        <div class="mv-kernel${sel}" data-kernel="${key}">
          <span class="mv-kernel__dot" style="background:${HEALTH_COLOR[k.health]}"></span>
          <span>
            <span class="mv-kernel__name">${esc(k.title)}</span><br />
            <span class="mv-kernel__sub">${esc(k.subtitle)}</span>
          </span>
        </div>`;
    }).join('');

    const legend = Object.entries({ resident: '常驻', temp: '临时', loop: '跨循环' })
      .map(([kind, label]) => `
        <div class="mv-legend__row"><i class="mv-swatch" style="background:${KIND_COLOR[kind]}"></i>${label}</div>`)
      .join('') +
      `<div class="mv-legend__row"><i class="mv-swatch" style="background:#ff5d5d"></i>峰值线 / 容量墙</div>`;

    host.innerHTML = `
      <div>
        <div class="mv-group-label">Kernel</div>
        <div class="mv-kernel-list">${kernelRows}</div>
      </div>
      <div>
        <div class="mv-group-label">图例</div>
        <div class="mv-legend">${legend}</div>
      </div>
      <div class="inspector-soft-card is-info" style="border-radius:var(--radius-lg)">
        <div style="font-size:12px;line-height:1.6;color:var(--foreground-secondary)">
          横轴 = buffer 地址 offset → 容量；纵轴 = 指令生命周期 tick。
          每个矩形宽 = 占用大小、高 = 存活区间。上下相邻同色块 = 内存复用。
        </div>
      </div>`;

    host.querySelectorAll('.mv-kernel').forEach((el) => {
      el.addEventListener('click', () => selectKernel(el.dataset.kernel));
    });
  }

  function selectKernel(key) {
    if (key === state.kernelKey) return;
    state.kernelKey = key;
    state.selected = null;
    state.step = 0;
    state.playing = false;
    stopRaf();
    renderExplorer();
    mountViewer();
    buildSwimlane();
    syncPlaybackScrubber();
    renderInspector();
    renderStatus();
    renderTerminal();
  }

  // ---- tabs -------------------------------------------------------------
  function renderTabs() {
    const host = $('#mv-tabs');
    host.innerHTML = TABS.map((t) => `
      <button class="tab-control-item${t.id === state.tab ? ' is-selected' : ''}" role="tab" data-tab="${t.id}">${t.label}</button>`).join('');
    host.querySelectorAll('.tab-control-item').forEach((el) => {
      el.addEventListener('click', () => setTab(el.dataset.tab));
    });
  }

  function setTab(id) {
    const prevWasDiff = state.tab === 'p2';
    state.tab = id;
    const nowIsDiff = id === 'p2';
    renderTabs();
    $('#mv-inspector-meta').textContent = TABS.find((t) => t.id === id).label;
    $('#mv-preview-title').textContent = id === 'p2' ? '占用图对比 · baseline ↔ current + 源码' : '空间-时间占用图';
    if (prevWasDiff !== nowIsDiff) {
      // toggle the source column BEFORE remounting so the viewer sizes to the split
      setSourceOpen(nowIsDiff);
      mountViewer(); // diff swaps the dataset
    }
    if (nowIsDiff) openSource(state.selected || activeMemory().tensors[0]);
    renderInspector();
    renderStatus();
  }

  // ---- inspector --------------------------------------------------------
  function section(title, kicker, body) {
    return `
      <div class="inspector-section">
        <div class="inspector-section-head">
          <span class="inspector-section-title">${title}</span>
          ${kicker ? `<span class="inspector-section-kicker">${kicker}</span>` : ''}
        </div>
        ${body}
      </div>`;
  }

  function selectionSection(showSource, showCce) {
    const t = state.selected;
    if (!t) return section('选中 Tensor', '', '<div class="mv-empty">在左图点击任一 tensor 查看详情。</div>');
    const reuseTags = [];
    if (t.reuseOf) reuseTags.push(`<span class="mv-reuse-tag">↥ 复用自 ${nameOf(t)}</span>`);
    (t.reusedBy || []).forEach((r) => reuseTags.push(`<span class="mv-reuse-tag">↧ 被 ${r} 复用</span>`));
    const kv = `
      <div class="mv-kv">
        <span class="k">名称</span><span class="v">${esc(t.name)}</span>
        <span class="k">Buffer</span><span class="v">${t.buffer}</span>
        <span class="k">地址</span><span class="v">${fmtHex(t.offset)} – ${fmtHex(t.offset + t.size)}</span>
        <span class="k">大小</span><span class="v">${fmtKB(t.size)} (${t.size} B)</span>
        <span class="k">存活</span><span class="v">#${t.allocTick} → #${t.freeTick} (${t.freeTick - t.allocTick} ticks)</span>
        <span class="k">分类</span><span class="v">${t.kind}</span>
        <span class="k">源码</span><span class="v">${t.srcFile}:${t.srcLineStart}-${t.srcLineEnd}</span>
      </div>
      ${reuseTags.length ? `<div style="margin-top:8px">${reuseTags.join('')}</div>` : ''}
      ${t.overflow ? '<div class="inspector-soft-card is-danger" style="margin-top:8px;border-radius:var(--radius-md);font-size:12px">⚠ 该 tensor 越过容量边界，是溢出的直接来源。</div>' : ''}`;

    let extra = '';
    if (showSource) {
      const codeHtml = (t.code || '').split('\n').map((ln, i) => {
        const no = (t.srcLineStart || 1) + i;
        const hot = no === t.srcHotLine ? ' class="hot"' : '';
        return `<span${hot}><span class="ln">${no}</span>${esc(ln)}</span>`;
      }).join('\n');
      extra += `
        <div style="font-size:11px;color:var(--foreground-muted);margin-top:10px">源码 · ${t.srcFile}:${t.srcLineStart}</div>
        <pre class="mv-code">${codeHtml}</pre>`;
    }
    if (showCce) {
      const cceHtml = (t.cce || '').split('\n').map((l) => {
        const e = esc(l);
        if (e.trim().startsWith('//')) return `<span class="cm">${e}</span>`;
        return e.replace(/^(\s*)([A-Z_][A-Z0-9_.]+)/, (m, sp, op) => `${sp}<span class="op">${op}</span>`);
      }).join('\n');
      extra += `
        <div style="font-size:11px;color:var(--foreground-muted);margin-top:8px">CCE 指令 · 生成代码</div>
        <pre class="mv-code">${cceHtml}</pre>`;
    }
    return section('选中 Tensor', t.buffer, kv + extra);
  }

  function nameOf(t) {
    const src = (state.tab === 'p2' ? activeMemory() : kernel().memory);
    const ref = src.tensors.find((x) => x.id === t.reuseOf);
    return ref ? ref.name : t.reuseOf;
  }

  function capacitySection() {
    const mem = activeMemory();
    const stats = bufferStats(mem);
    const rows = mem.buffers.map((buf) => {
      const s = stats[buf.name];
      const pct = Math.min(100, Math.round(s.util * 100));
      const color = s.overflow ? 'var(--danger)' : s.util > 0.85 ? 'var(--warning)' : 'var(--success)';
      const capPct = Math.min(100, (buf.capacity / Math.max(buf.capacity, s.highWater)) * 100);
      return `
        <div style="margin-bottom:10px">
          <div class="mv-metric-row"><span>${buf.name}</span><b style="color:${s.overflow ? 'var(--danger)' : 'inherit'}">${fmtKB(s.peak)} / ${fmtKB(buf.capacity)} · ${(s.util * 100).toFixed(0)}%</b></div>
          <div class="mv-meterbar">
            <div class="mv-meterbar__fill" style="width:${pct}%;background:${color}"></div>
            <div class="mv-meterbar__cap" style="left:${capPct}%"></div>
          </div>
          <div style="font-size:11px;color:var(--foreground-muted)">high-water ${fmtHex(s.highWater)} · headroom ${s.overflow ? '−' + fmtKB(s.highWater - buf.capacity) : fmtKB(buf.capacity - s.highWater)}</div>
        </div>`;
    }).join('');
    return section('容量与峰值', 'MVP', rows);
  }

  function overflowBanner() {
    const d = kernel().diagnostics;
    if (state.tab === 'p2' || !d.overflow) return '';
    return `<div class="inspector-soft-card is-danger" style="border-radius:var(--radius-lg);margin-bottom:var(--space-4)">
      <div style="font-weight:500;color:var(--danger)">Buffer 溢出</div>
      <div style="font-size:12px;line-height:1.6;color:var(--foreground-secondary);margin-top:4px">
        <code>${d.overflow.tensor}</code> 使 <b>${d.overflow.space}</b> 越界 ${fmtKB(d.overflow.byBytes)}。编译将报错 / 910B 上可能静默损坏。
      </div>
    </div>`;
  }

  function pipelineSection() {
    const d = kernel().diagnostics;
    const rows = d.pipeline.map((p) => {
      const ok = p.ok;
      const badge = ok
        ? `<span style="color:var(--success)">depth ${p.achieved}</span>`
        : `<span style="color:var(--warning)">${p.requested}→${p.achieved} 降级</span>`;
      return `
        <div class="mv-diff-row" style="grid-template-columns:1fr auto">
          <span class="lbl">${p.space} · slot ${p.slot}<br /><span style="font-size:11px;color:var(--foreground-muted)">${esc(p.note)}</span></span>
          <span class="delta">${badge}</span>
        </div>`;
    }).join('');
    return section('Double-buffer / 流水深度', 'P1', rows);
  }

  function hintsSection() {
    const d = kernel().diagnostics;
    if (state.tab === 'p2') return '';
    const map = { success: 'var(--success)', info: 'var(--primary)', warning: 'var(--warning)', danger: 'var(--danger)' };
    const rows = d.hints.map((h) => `
      <div class="mv-hint" style="margin-bottom:10px">
        <span class="mv-hint__code" style="color:${map[h.level]}">${h.code}</span>
        <span class="mv-hint__msg">${esc(h.msg)}
          <span class="mv-hint__src" data-src-line="${h.srcLine}">${h.srcFile}:${h.srcLine} ↗</span>
        </span>
      </div>`).join('');
    return section('编译诊断', 'P1', rows);
  }

  function diffSection() {
    const base = bufferStats(MV.diff.baseline);
    const cur = bufferStats(MV.diff.current);
    const buffers = MV.diff.current.buffers.map((b) => b.name);
    const rows = buffers.map((name) => {
      // Memory reuse shrinks the allocation footprint (high-water), not live bytes.
      const bp = base[name] ? base[name].highWater : 0;
      const cp = cur[name] ? cur[name].highWater : 0;
      const cap = MV.diff.current.buffers.find((b) => b.name === name).capacity;
      const delta = cp - bp;
      const dc = delta < 0 ? 'var(--success)' : delta > 0 ? 'var(--danger)' : 'var(--foreground-muted)';
      const sign = delta > 0 ? '+' : '';
      const baseOvf = bp > cap ? ' style="color:var(--danger)"' : '';
      const curOvf = cp > cap ? ' style="color:var(--danger)"' : '';
      return `
        <div class="mv-diff-row">
          <span class="lbl">${name} footprint</span>
          <span class="num"${baseOvf}>${fmtKB(bp)}</span>
          <span class="num"${curOvf}>→ ${fmtKB(cp)}</span>
          <span class="delta" style="color:${dc}">${sign}${fmtKB(delta)}</span>
        </div>`;
    }).join('');

    const baseReuse = MV.diff.baseline.tensors.filter((t) => t.reuseOf).length;
    const curReuse = MV.diff.current.tensors.filter((t) => t.reuseOf).length;

    const ab = `
      <div class="segmented-control segmented-control-muted mv-ab" role="group" aria-label="baseline current">
        <button class="segmented-control-item${state.diffSide === 'baseline' ? ' is-selected' : ''}" data-side="baseline">baseline</button>
        <button class="segmented-control-item${state.diffSide === 'current' ? ' is-selected' : ''}" data-side="current">current</button>
      </div>`;

    const body = `
      <div style="font-size:12px;color:var(--foreground-secondary);margin-bottom:8px">
        ${esc(MV.diff.label)}<br />主图当前显示：${ab}
      </div>
      ${rows}
      <div class="inspector-soft-card is-success" style="margin-top:10px;border-radius:var(--radius-md);font-size:12px">
        复用条目 ${baseReuse} → ${curReuse}：<code>reluUb</code> 新复用 <code>wGm_in</code> 空槽，
        UB footprint 从 <b>304KB(溢出)</b> 降到 <b>240KB(容纳)</b>。
      </div>`;
    return section('分支对比', 'P2', body);
  }

  function runtimeSection() {
    const rt = kernel().runtime;
    const live = liveAt(kernel().memory, state.step);
    const overCap = rt.realPeakKB * KB;
    const body = `
      <div class="mv-metric-row"><span>静态峰值 (编译期)</span><b>${rt.staticPeakKB}KB</b></div>
      <div class="mv-metric-row"><span>运行时实测峰值</span><b style="color:${rt.realPeakKB * KB > MV.CAP.UB ? 'var(--danger)' : 'inherit'}">${rt.realPeakKB}KB</b></div>
      <div class="mv-metric-row"><span>流水 overlap</span><b>${rt.overlapPct}%</b></div>
      <div class="mv-metric-row"><span>停顿 (stall)</span><b style="color:${rt.stalls > 1 ? 'var(--warning)' : 'inherit'}">${rt.stalls}</b></div>
      <div class="inspector-soft-card is-info" style="margin-top:10px;border-radius:var(--radius-md);font-size:12px">
        当前 tick <b>#${state.step}</b> · live tensors <b>${live.count}</b> · live bytes <b>${fmtKB(live.bytes)}</b>
        <div style="margin-top:4px;color:var(--foreground-muted)">拖动底部播放条 scrubber 或点播放，观察占用随执行推进。</div>
      </div>`;
    return section('运行时占用与 overlap', 'P4', body);
  }

  function renderInspector() {
    const host = $('#mv-inspector');
    let html = '';
    if (state.tab === 'mvp') {
      html = overflowBanner() + capacitySection() + hintsSection() + selectionSection(false, false);
    } else if (state.tab === 'p1') {
      html = overflowBanner() + pipelineSection() + hintsSection() + selectionSection(false, false);
    } else if (state.tab === 'p2') {
      // full source lives in the editor panel; inspector keeps facts + generated CCE
      html = diffSection() + selectionSection(false, true);
    } else if (state.tab === 'p4') {
      html = runtimeSection() + selectionSection(false, false);
    }
    host.innerHTML = html;

    host.querySelectorAll('[data-src-line]').forEach((el) => {
      el.addEventListener('click', () => {
        // jump to the tensor whose hot line matches, open source panel + select
        const line = Number(el.dataset.srcLine);
        const t = kernel().memory.tensors.find((x) => x.srcHotLine === line || (x.srcLineStart <= line && x.srcLineEnd >= line));
        if (t) {
          state.selected = t;
          setTab('p2');
          if (viewer) viewer.selectTensor(t.id);
        }
      });
    });
    host.querySelectorAll('[data-side]').forEach((el) => {
      el.addEventListener('click', () => {
        state.diffSide = el.dataset.side;
        mountViewer();
        if (isSourceOpen()) openSource(state.selected);
        renderInspector();
        renderStatus();
      });
    });
  }

  // ---- status strip -----------------------------------------------------
  function renderStatus() {
    const mem = activeMemory();
    const stats = bufferStats(mem);
    const primary = stats[mem.buffers[0].name];
    const k = kernel();
    const worst = Object.entries(stats).sort((a, b) => b[1].util - a[1].util)[0];
    $('#mv-status').innerHTML = `
      <span class="item"><i class="mv-swatch" style="background:${HEALTH_COLOR[k.health]};display:inline-block"></i> ${esc(k.title)}</span>
      <span class="item">${mem.buffers[0].name} 容量 <b>${fmtKB(primary.capacity)}</b></span>
      <span class="item is-peak">峰值 <b>${fmtKB(primary.peak)}</b></span>
      <span class="item">利用率 <b>${(primary.util * 100).toFixed(1)}%</b></span>
      <span class="item">最紧张 <b>${worst[0]} ${(worst[1].util * 100).toFixed(0)}%</b></span>
      <span class="spacer"></span>
      <span class="item">Tensors <b>${mem.tensors.length}</b></span>
      <span class="item">复用 <b>${mem.tensors.filter((t) => t.reuseOf).length}</b></span>`;
  }

  // ---- diagnostics terminal log ----------------------------------------
  function renderTerminal() {
    const body = $('#mv-terminal-body');
    if (!body) return;
    const k = kernel();
    const lines = [
      `<span class="lv-ok">[pass]</span> InitMemRef → MaterializeSemanticAliases → MemoryReuse → AllocateMemoryAddr`,
      ...k.diagnostics.hints.map((h) => {
        const cls = h.level === 'danger' ? 'lv-err' : h.level === 'warning' ? 'lv-warn' : 'lv-ok';
        return `<span class="${cls}">[${h.code}]</span> ${esc(h.msg)}  (${h.srcFile}:${h.srcLine})`;
      }),
    ];
    body.innerHTML = lines.map((l) => `<p class="mv-log-line">${l}</p>`).join('');
  }

  // ---- swimlane (P4 runtime trace) -------------------------------------
  const swimHost = $('#mv-swimlane');
  let swimCanvas = null;
  let swimCtx = null;
  const SWIM = { padL: 132, padR: 24, padT: 10, padB: 22, laneH: 26, laneGap: 8 };

  function buildSwimlane() {
    swimHost.innerHTML = '';
    swimCanvas = document.createElement('canvas');
    swimCanvas.style.width = '100%';
    swimCanvas.style.height = '100%';
    swimCanvas.style.display = 'block';
    swimHost.appendChild(swimCanvas);
    swimCtx = swimCanvas.getContext('2d');
    drawSwimlane();
  }

  function drawSwimlane() {
    if (!swimCtx) return;
    const rt = kernel().runtime;
    const rect = swimHost.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;
    const W = Math.max(320, rect.width);
    const H = Math.max(120, rect.height);
    swimCanvas.width = W * dpr;
    swimCanvas.height = H * dpr;
    swimCtx.setTransform(dpr, 0, 0, dpr, 0, 0);
    swimCtx.clearRect(0, 0, W, H);

    const plotX = SWIM.padL;
    const plotW = W - SWIM.padL - SWIM.padR;
    const xOf = (cycle) => plotX + (cycle / rt.cycles) * plotW;
    const font = getComputedStyle(document.body).fontFamily;

    // grid + cycle ruler
    swimCtx.strokeStyle = 'rgba(255,255,255,0.06)';
    swimCtx.fillStyle = 'rgba(255,255,255,0.35)';
    swimCtx.font = `10px ${font}`;
    swimCtx.textAlign = 'center';
    swimCtx.textBaseline = 'alphabetic';
    const step = 16;
    for (let c = 0; c <= rt.cycles; c += step) {
      const x = xOf(c);
      swimCtx.beginPath();
      swimCtx.moveTo(x, SWIM.padT);
      swimCtx.lineTo(x, H - SWIM.padB);
      swimCtx.stroke();
      swimCtx.fillText('#' + c, x, H - 8);
    }

    // lanes + bars
    rt.lanes.forEach((lane, i) => {
      const y = SWIM.padT + i * (SWIM.laneH + SWIM.laneGap);
      // lane label
      swimCtx.fillStyle = 'rgba(255,255,255,0.55)';
      swimCtx.font = `11px ${font}`;
      swimCtx.textAlign = 'left';
      swimCtx.textBaseline = 'middle';
      swimCtx.fillText(lane.label, 8, y + SWIM.laneH / 2);
      // lane baseline
      swimCtx.strokeStyle = 'rgba(255,255,255,0.05)';
      swimCtx.beginPath();
      swimCtx.moveTo(plotX, y + SWIM.laneH);
      swimCtx.lineTo(plotX + plotW, y + SWIM.laneH);
      swimCtx.stroke();

      rt.tasks.filter((t) => t.lane === lane.id).forEach((t) => {
        const x = xOf(t.start);
        const w = Math.max(3, xOf(t.start + t.dur) - x);
        window.PtoSwimlaneTaskPattern.drawTaskBar(swimCtx, {
          x,
          y: y + 2,
          width: w,
          height: SWIM.laneH - 4,
          baseColor: t.status === 'stall' ? '#f0883e' : LANE_COLOR[lane.id],
          fontFamily: font,
          isSelected: false,
          task: {
            label: t.op,
            displayName: t.op,
            laneKind: lane.id,
            totalCycle: t.total,
            clcCycle: t.clc,
            gap: t.gap,
            gapRatio: t.gapRatio,
            status: t.status,
            inputRawMagic: t.in ? [1] : undefined,
            outputRawMagic: t.out ? [1] : undefined,
          },
        });
      });
    });

    // time cursor
    const cx = xOf(Math.min(state.step, rt.cycles));
    swimCtx.strokeStyle = '#ff5d5d';
    swimCtx.lineWidth = 1.5;
    swimCtx.beginPath();
    swimCtx.moveTo(cx, SWIM.padT - 2);
    swimCtx.lineTo(cx, H - SWIM.padB);
    swimCtx.stroke();
    swimCtx.fillStyle = '#ff5d5d';
    swimCtx.beginPath();
    swimCtx.arc(cx, SWIM.padT - 2, 3, 0, Math.PI * 2);
    swimCtx.fill();
    swimCtx.textAlign = 'left';
    swimCtx.textBaseline = 'top';
    swimCtx.font = `10px ${font}`;
    swimCtx.fillText('cycle #' + state.step, Math.min(cx + 6, W - 60), SWIM.padT - 2);
  }

  // ---- playback wiring (P4) --------------------------------------------
  let pb = null;

  function findPb() {
    const mount = $('[data-ide-floating-playback]');
    if (!mount) return null;
    const pick = (id, cls) => document.getElementById(id) || mount.querySelector(cls);
    return {
      mount,
      play: pick('ide-floating-playback-1-play', '.pto-floating-playback__button--primary'),
      back: pick('ide-floating-playback-1-step-back', '.pto-floating-playback__button--step-back'),
      fwd: pick('ide-floating-playback-1-step-fwd', '.pto-floating-playback__button--step-fwd'),
      replay: pick('ide-floating-playback-1-replay', '.pto-floating-playback__button--replay'),
      scrubber: pick('ide-floating-playback-1-scrubber', '.pto-floating-playback__scrubber'),
      label: pick('ide-floating-playback-1-scrubber-label', '.pto-floating-playback__counter'),
      opname: pick('ide-floating-playback-1-scrubber-opname', '.pto-floating-playback__opname'),
    };
  }

  function syncPlaybackScrubber() {
    if (!pb || !pb.scrubber) return;
    const rt = kernel().runtime;
    pb.scrubber.max = String(rt.cycles - 1);
    pb.scrubber.value = String(state.step);
    updatePlaybackLabels();
  }

  function updatePlaybackLabels() {
    if (!pb) return;
    const rt = kernel().runtime;
    if (pb.mount) pb.mount.dataset.playbackState = `cycle #${state.step}`;
    if (pb.label) pb.label.textContent = `${state.step} / ${rt.cycles - 1}`;
    if (pb.opname) pb.opname.textContent = `cycle #${state.step}`;
  }

  function onStepChanged(fromScrub) {
    if (pb && pb.scrubber && !fromScrub) pb.scrubber.value = String(state.step);
    updatePlaybackLabels();
    drawSwimlane();
    if (state.tab === 'p4') renderInspector();
  }

  function stopRaf() {
    if (state.timerId) clearInterval(state.timerId);
    state.timerId = null;
    state.playing = false;
  }

  function startPlay() {
    const rt = kernel().runtime;
    if (state.step >= rt.cycles - 1) state.step = 0;
    state.playing = true;
    state.timerId = setInterval(() => {
      state.step += 1;
      if (state.step >= rt.cycles - 1) {
        state.step = rt.cycles - 1;
        onStepChanged(false);
        stopRaf();
        return;
      }
      onStepChanged(false);
    }, 55);
  }

  function wirePlayback() {
    pb = findPb();
    if (!pb) return;
    if (pb.play) pb.play.addEventListener('click', () => {
      if (state.playing) stopRaf();
      else startPlay();
    });
    if (pb.scrubber) pb.scrubber.addEventListener('input', () => {
      stopRaf();
      state.step = Number(pb.scrubber.value) || 0;
      onStepChanged(true);
    });
    if (pb.back) pb.back.addEventListener('click', () => { stopRaf(); state.step = Math.max(0, state.step - 1); onStepChanged(false); });
    if (pb.fwd) pb.fwd.addEventListener('click', () => { stopRaf(); state.step = Math.min(kernel().runtime.cycles - 1, state.step + 1); onStepChanged(false); });
    if (pb.replay) pb.replay.addEventListener('click', () => { stopRaf(); state.step = 0; onStepChanged(false); });
  }

  // ---- boot -------------------------------------------------------------
  function boot() {
    if (!window.PtoMemoryReuseViewer) return; // pattern not loaded
    renderTabs();
    renderExplorer();
    mountViewer();
    buildSwimlane();
    wirePlayback();
    syncPlaybackScrubber();
    renderInspector();
    renderStatus();
    renderTerminal();

    // source-panel chrome
    const srcClose = $('#mv-src-close');
    if (srcClose) srcClose.addEventListener('click', () => setTab('mvp'));
    const srcPrev = $('#mv-src-prev');
    if (srcPrev) srcPrev.addEventListener('click', () => navSource(-1));
    const srcNext = $('#mv-src-next');
    if (srcNext) srcNext.addEventListener('click', () => navSource(1));

    let rAF;
    window.addEventListener('resize', () => {
      cancelAnimationFrame(rAF);
      rAF = requestAnimationFrame(drawSwimlane);
    });
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', boot);
  } else {
    boot();
  }
})();
