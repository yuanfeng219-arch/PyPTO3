/**
 * 推理性能分析 · 时间线（P3）
 *
 * 所有 span 都从 inference-profile-data.js 的 op 数据推导，不另立数字，保证与算子表同源：
 *   每层 14 个任务的 device time 合计 355.5 μs + barrier 4.5 μs = 360.0 μs
 *   360.0 × 40 层 + 边界 605 μs + Host 195 μs = 15,200 μs = TPOT
 *
 * 调度口径为「零重叠串行」——与算子表的 share 合计 100% 一致。
 * 其中 3 个任务与前驱无数据依赖却被串行调度，单列为「可并行旁路」，
 * 这正是 demo-v2.js 里既有的「计算段无法跨 bn 重叠」叙事在时间线上的体现。
 */
(function registerInferenceTimeline() {
  'use strict';

  const GROUP_COLOR = {
    mlp: 'var(--primary)',
    attn: 'var(--warning)',
    proj: 'var(--tone-blue-strong, #4a90d9)',
    norm: 'var(--tone-green-strong, #4caf7d)',
    boundary: 'color-mix(in srgb, var(--foreground) 42%, transparent)',
    idle: 'color-mix(in srgb, var(--danger) 55%, transparent)',
  };

  /** 层内调度顺序 = decode_layer.py 的任务依赖顺序 */
  const LAYER_CHAIN = [
    'rms-recip', 'qkv-proj', 'qk-norm',
    'fa-work-build', 'rope-qkv', 'fa-fused', 'online-softmax',
    'out-proj', 'residual-cast', 'post-rms-reduce', 'gate-up-proj', 'silu', 'down-proj', 'dcr-xgamma',
  ];

  /** 与前驱无数据依赖、当前却被串行调度的任务 —— 时间线要专门标出来 */
  const PARALLELIZABLE = {
    'rms-recip': '只算 inv_rms 标量，与 qkv_proj 读同一份 normed_in，无先后依赖',
    'fa-work-build': '只读 seq_lens 构建工作表，与 Scope 1 完全无关',
    'post-rms-reduce': '与 residual_rms_cast 消费同一份 out_proj 结果，可并行归约',
  };

  /** 层内 barrier，合计 4.5 μs */
  const BARRIERS = { 'qk-norm': 1.2, 'online-softmax': 1.4, 'dcr-xgamma': 1.9 };

  const TRACKS = [
    { id: 'host', label: 'Host', kind: 'host', hint: 'python dispatch / 等待下一步' },
    { id: 'queue', label: 'Stream', kind: 'queue', hint: '设备队列占用' },
    { id: 'edge', label: '边界', kind: 'task', hint: 'copy_hidden · cast · rms_lm_head' },
    { id: 'scope1', label: 'Scope 1', kind: 'task', hint: 'Input RMS + QKV' },
    { id: 'scope2', label: 'Scope 2', kind: 'task', hint: 'Paged Flash Attention' },
    { id: 'scope3', label: 'Scope 3', kind: 'task', hint: 'Output + MLP' },
    { id: 'aic', label: 'AIC · Cube', kind: 'hw', hint: '矩阵计算单元' },
    { id: 'aiv', label: 'AIV · Vector', kind: 'hw', hint: '向量计算单元' },
    { id: 'mte2', label: 'MTE2 ↓', kind: 'hw', hint: 'HBM → 片上' },
    { id: 'mte3', label: 'MTE3 ↑', kind: 'hw', hint: '片上 → HBM' },
    { id: 'sync', label: '同步 / 空隙', kind: 'gap', hint: 'barrier 与暴露空隙' },
  ];

  const UNIT_TRACK = { cube: 'aic', vector: 'aiv', mte2: 'mte2', mte3: 'mte3' };

  const BOUNDARY_IN = [['copy-hidden', 3], ['x-gamma0', 2]];
  const BOUNDARY_OUT = [['cast-lmhead', 2], ['rms-lm-head', 598]];
  const HOST_LAUNCH = 40;
  const HOST_TAIL = 155;

  let model = null;

  const esc = (v) => String(v).replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
  const fmt = (n, d = 2) => Number(n).toFixed(d);

  function scopeTrack(op) {
    const head = op.scope.split(' · ')[0];
    if (head === 'Scope 1') return 'scope1';
    if (head === 'Scope 2') return 'scope2';
    if (head === 'Scope 3') return 'scope3';
    return 'edge';
  }

  /** 把一个任务 span 按 units 占比铺成硬件流水线上的若干块，视觉上呈现占空比 */
  function hardwareChunks(span, op) {
    const out = [];
    Object.keys(UNIT_TRACK).forEach((unit) => {
      const pct = op.units[unit];
      if (!pct) return;
      const n = Math.max(1, Math.min(9, Math.round(span.dur / 14)));
      const slot = span.dur / n;
      const width = slot * (pct / 100);
      for (let i = 0; i < n; i += 1) {
        out.push({
          track: UNIT_TRACK[unit],
          t0: span.t0 + i * slot + (slot - width) / 2,
          dur: width,
          unit,
          opId: op.id,
          name: op.name,
          pct,
        });
      }
    });
    return out;
  }

  /** 铺一层：返回 { spans, gaps, dur } —— dur 恒为 360.0 μs */
  function layoutLayer(byId, originUs, layerIndex) {
    const spans = [];
    const gaps = [];
    let t = originUs;
    LAYER_CHAIN.forEach((opId) => {
      const op = byId[opId];
      const dur = op.perLayer ? op.perLayer[layerIndex] : op.perLayerUs;
      spans.push({
        id: `${opId}@${layerIndex}`,
        opId,
        name: op.name,
        track: scopeTrack(op),
        group: op.group,
        t0: t,
        dur,
        layer: layerIndex,
        bound: op.boundLabel,
        parallelizable: !!PARALLELIZABLE[opId],
        critical: !PARALLELIZABLE[opId],
      });
      t += dur;
      const barrier = BARRIERS[opId];
      if (barrier) {
        gaps.push({ id: `gap-${opId}@${layerIndex}`, t0: t, dur: barrier, reason: 'barrier', after: op.name, layer: layerIndex });
        t += barrier;
      }
    });
    return { spans, gaps, end: t };
  }

  function build(profile) {
    if (model && model.profileId === profile.id) return model;
    const byId = Object.fromEntries(profile.ops.map((o) => [o.id, o]));

    const spans = [];
    const gaps = [];
    // Host 先 dispatch，设备侧从 HOST_LAUNCH 之后才开始，两段 host 时间串行计入 TPOT
    let t = HOST_LAUNCH;

    spans.push({ id: 'host-launch', opId: null, name: 'graph launch', track: 'host', group: 'idle', t0: 0, dur: HOST_LAUNCH, layer: null, critical: true });

    BOUNDARY_IN.forEach(([opId, dur]) => {
      const op = byId[opId];
      spans.push({ id: opId, opId, name: op.name, track: 'edge', group: op.group, t0: t, dur, layer: null, bound: op.boundLabel, critical: true });
      t += dur;
    });

    const layers = [];
    for (let i = 0; i < profile.meta.layers; i += 1) {
      const start = t;
      const laid = layoutLayer(byId, t, i);
      spans.push(...laid.spans);
      gaps.push(...laid.gaps);
      t = laid.end;
      layers.push({ index: i, t0: start, dur: t - start });
    }

    BOUNDARY_OUT.forEach(([opId, dur]) => {
      const op = byId[opId];
      spans.push({ id: opId, opId, name: op.name, track: 'edge', group: op.group, t0: t, dur, layer: null, bound: op.boundLabel, critical: true });
      t += dur;
    });

    const deviceEnd = t;
    spans.push({ id: 'host-tail', opId: null, name: '等待下一 step', track: 'host', group: 'idle', t0: deviceEnd, dur: HOST_TAIL, layer: null, critical: false });
    gaps.push({ id: 'gap-host', t0: deviceEnd, dur: HOST_TAIL, reason: 'host', after: 'rms_lm_head', layer: null });
    spans.push({ id: 'queue', opId: null, name: '设备队列占用', track: 'queue', group: 'proj', t0: HOST_LAUNCH, dur: deviceEnd - HOST_LAUNCH, layer: null, critical: true });

    const total = deviceEnd + HOST_TAIL;

    // 硬件轨道只在展开单层时生成，整步铺 40 层会有上万个块
    const hardwareFor = (layerIndex) => spans
      .filter((s) => s.layer === layerIndex && byId[s.opId])
      .flatMap((s) => hardwareChunks(s, byId[s.opId]));

    const parallelUs = Object.keys(PARALLELIZABLE).reduce((a, id) => a + byId[id].perLayerUs, 0);
    const chainUs = LAYER_CHAIN.reduce((a, id) => a + byId[id].perLayerUs, 0);

    model = {
      profileId: profile.id,
      spans,
      gaps,
      layers,
      total,
      deviceEnd,
      hardwareFor,
      stats: {
        layerDur: chainUs + 4.5,
        chainUs,
        criticalUs: chainUs - parallelUs,
        parallelUs,
        barrierUs: 4.5,
        parallelStepMs: parallelUs * profile.meta.layers / 1000,
        parallelPct: parallelUs * profile.meta.layers / 1000 / profile.summary.tpot.p50 * 100,
      },
    };
    return model;
  }

  /* ---------------- 渲染 ---------------- */

  function ruler(w0, w1, zoom) {
    const span = w1 - w0;
    const target = 10 * zoom;
    const raw = span / target;
    const mag = 10 ** Math.floor(Math.log10(raw));
    const step = [1, 2, 5, 10].map((m) => m * mag).find((s) => s >= raw) || mag * 10;
    const ticks = [];
    for (let v = Math.ceil(w0 / step) * step; v <= w1; v += step) {
      const label = span >= 2000 ? `${fmt(v / 1000, 2)} ms` : `${fmt(v, step < 1 ? 1 : 0)} μs`;
      ticks.push(`<span class="kf-prof-tltick" style="left:${(v - w0) / span * 100}%"><i></i><b>${label}</b></span>`);
    }
    return ticks.join('');
  }

  /** 裁到窗口内：跨窗口的 span（如覆盖整步的 Stream）否则会画到相邻轨道上 */
  function clamp(t0, dur, w0, span) {
    const rawL = (t0 - w0) / span * 100;
    const rawR = (t0 + dur - w0) / span * 100;
    if (rawL > 100 || rawR < 0) return null;
    const left = Math.max(rawL, 0);
    const width = Math.max(Math.min(rawR, 100) - left, 0.08);
    return { left, width, clippedL: rawL < 0, clippedR: rawR > 100 };
  }

  function spanEl(s, w0, span, opts) {
    const box = clamp(s.t0, s.dur, w0, span);
    if (!box) return '';
    const { left, width } = box;
    const dim = opts.criticalOnly && !s.critical ? ' is-dim' : '';
    const par = s.parallelizable ? ' is-parallel' : '';
    const sel = opts.selected === s.id ? ' is-selected' : '';
    const title = `${s.name}\n${fmt(s.dur, 2)} μs · 起 ${fmt(s.t0, 1)} μs${s.layer !== null && s.layer !== undefined ? ` · L${s.layer}` : ''}${s.parallelizable ? '\n⚠ 与前驱无依赖，当前被串行调度' : ''}`;
    return `<button class="kf-prof-tlspan${dim}${par}${sel}" type="button" style="left:${left}%;width:${width}%;--sp:${GROUP_COLOR[s.group] || 'var(--foreground-muted)'}" data-tl-span="${esc(s.id)}" data-tl-op="${esc(s.opId || '')}" title="${esc(title)}"><b>${esc(s.name)}</b></button>`;
  }

  function hwEl(c, w0, span) {
    const box = clamp(c.t0, c.dur, w0, span);
    if (!box) return '';
    return `<i class="kf-prof-tlhw" style="left:${box.left}%;width:${box.width}%" title="${esc(c.name)} · ${c.unit.toUpperCase()} ${fmt(c.pct, 1)}%"></i>`;
  }

  function gapEl(g, w0, span, threshold) {
    if (g.dur < threshold) return '';
    const box = clamp(g.t0, g.dur, w0, span);
    if (!box) return '';
    const reason = g.reason === 'host' ? 'Host dispatch 等待' : `${g.after} 之后的 barrier`;
    return `<i class="kf-prof-tlgap is-${g.reason}" style="left:${box.left}%;width:${Math.max(box.width, 0.12)}%" title="${esc(`${fmt(g.dur, 2)} μs · ${reason}`)}"></i>`;
  }

  function minimap(m, st, profile) {
    const cells = m.layers.map((l) => `<button class="kf-prof-tlcell${st.layer === l.index ? ' is-active' : ''}" type="button" data-tl-layer="${l.index}" title="L${l.index} · ${fmt(l.dur, 1)} μs" style="left:${l.t0 / m.total * 100}%;width:${l.dur / m.total * 100}%"></button>`).join('');
    const head = `<i class="kf-prof-tledge" style="left:0;width:${5 / m.total * 100}%" title="输入边界 5 μs"></i>`;
    const tail = `<i class="kf-prof-tledge" style="left:${(m.deviceEnd - 600) / m.total * 100}%;width:${600 / m.total * 100}%" title="输出边界 600 μs"></i>`;
    const host = `<i class="kf-prof-tlhost" style="left:${m.deviceEnd / m.total * 100}%;width:${HOST_TAIL / m.total * 100}%" title="Host 等待 155 μs"></i>`;
    const win = st.mode === 'layer' && m.layers[st.layer]
      ? `<i class="kf-prof-tlwin" style="left:${m.layers[st.layer].t0 / m.total * 100}%;width:${Math.max(m.layers[st.layer].dur / m.total * 100, 0.6)}%"></i>` : '';
    return `<div class="kf-prof-tlmini">
      <div class="kf-prof-tlminitrack">${head}${cells}${tail}${host}${win}</div>
      <div class="kf-prof-tlminiaxis"><span>0</span><span>整步 ${fmt(profile.summary.tpot.p50, 1)} ms · 40 层 · 点击层块展开</span><span>${fmt(m.total / 1000, 2)} ms</span></div>
    </div>`;
  }

  function render(profile, st) {
    const m = build(profile);
    const layerMode = st.mode === 'layer';
    const layer = m.layers[st.layer];
    const w0 = layerMode ? layer.t0 : 0;
    const w1 = layerMode ? layer.t0 + layer.dur : m.total;
    const span = w1 - w0;

    const visibleSpans = m.spans.filter((s) => s.t0 + s.dur > w0 && s.t0 < w1);
    const hw = layerMode ? m.hardwareFor(st.layer) : [];
    const shownGaps = m.gaps.filter((g) => g.t0 + g.dur > w0 && g.t0 < w1 && g.dur >= st.gapThreshold);

    const rows = TRACKS.map((track) => {
      let content = '';
      if (track.kind === 'hw') {
        content = layerMode
          ? hw.filter((c) => c.track === track.id).map((c) => hwEl(c, w0, span)).join('')
          : '<span class="kf-prof-tlnote">展开单层后显示硬件流水线</span>';
      } else if (track.kind === 'gap') {
        content = shownGaps.map((g) => gapEl(g, w0, span, st.gapThreshold)).join('')
          || '<span class="kf-prof-tlnote">窗口内无 ≥ 阈值的空隙</span>';
      } else {
        const list = visibleSpans.filter((s) => s.track === track.id);
        content = list.length ? list.map((s) => spanEl(s, w0, span, st)).join('') : '<span class="kf-prof-tlnote">—</span>';
      }
      return `<div class="kf-prof-tltrack" data-track="${track.id}" data-kind="${track.kind}">${content}</div>`;
    }).join('');

    const labels = TRACKS.map((t) => `<div class="kf-prof-tllabel" title="${esc(t.hint)}"><b>${esc(t.label)}</b></div>`).join('');

    const s = m.stats;
    const layerOptions = m.layers.map((l) => `<option value="${l.index}"${l.index === st.layer ? ' selected' : ''}>L${l.index} · ${fmt(l.dur, 1)} μs</option>`).join('');

    const sel = st.selected ? m.spans.find((x) => x.id === st.selected) : null;
    const detail = sel ? `<div class="kf-prof-tlsel">
        <b style="--sp:${GROUP_COLOR[sel.group]}">${esc(sel.name)}</b>
        <span>起 ${fmt(sel.t0 - w0, 2)} μs（窗口内）· 时长 <b>${fmt(sel.dur, 2)} μs</b>${sel.layer !== null && sel.layer !== undefined ? ` · L${sel.layer}` : ''}${sel.bound ? ` · ${esc(sel.bound)}-bound` : ''}</span>
        ${sel.parallelizable ? `<em>⚠ ${esc(PARALLELIZABLE[sel.opId])}</em>` : ''}
        <span class="kf-prof-spacer" style="flex:1"></span>
        ${sel.opId ? `<button class="kf-prof-btn" type="button" data-tl-goto-op="${esc(sel.opId)}">在算子分析中打开</button>` : ''}
        ${sel.opId ? `<button class="kf-prof-btn" type="button" data-goto-graph="${esc(sel.opId)}">↗ 在结构图中定位</button>` : ''}
      </div>` : '<div class="kf-prof-tlsel is-empty"><span>点击任意色块查看该任务的时间细节，并可回跳到算子分析或结构图。</span></div>';

    return `<div class="kf-prof-optools">
        <div class="kf-prof-seg" role="group" aria-label="时间范围">
          <button type="button" class="${layerMode ? '' : 'is-active'}" data-tl-mode="step">整步 15.2 ms</button>
          <button type="button" class="${layerMode ? 'is-active' : ''}" data-tl-mode="layer">展开单层</button>
        </div>
        <select class="kf-prof-search" id="tlLayerPick" ${layerMode ? '' : 'disabled'} aria-label="选择层">${layerOptions}</select>
        <div class="kf-prof-seg" role="group" aria-label="缩放">
          <button type="button" data-tl-zoom="out" aria-label="缩小">−</button>
          <button type="button" data-tl-zoom="fit">${st.zoom === 1 ? '适应' : `${st.zoom}×`}</button>
          <button type="button" data-tl-zoom="in" aria-label="放大">＋</button>
        </div>
        <button class="kf-prof-btn${st.criticalOnly ? ' is-on' : ''}" type="button" data-tl-critical>${st.criticalOnly ? '✓ ' : ''}只看关键路径</button>
        <label class="kf-prof-tlthresh">空隙阈值 <select data-tl-threshold>${[0.5, 1, 2, 5].map((v) => `<option value="${v}"${st.gapThreshold === v ? ' selected' : ''}>≥ ${v} μs</option>`).join('')}</select></label>
        <span class="kf-prof-spacer" style="flex:1"></span>
        <span style="color:var(--foreground-muted);font-size:var(--kf-type-2xs)">窗口 ${fmt(span >= 2000 ? span / 1000 : span, 2)} ${span >= 2000 ? 'ms' : 'μs'} · 检出空隙 ${shownGaps.length} 处</span>
      </div>

      ${minimap(m, st, profile)}

      <section class="kf-prof-card">
        <header><h3>${layerMode ? `Decoder Layer ${st.layer}` : '单个 decode step'}</h3><span>${layerMode ? `${fmt(layer.dur, 1)} μs · 14 个任务串行` : `${fmt(m.total / 1000, 2)} ms · 输入边界 → 40 层 → 输出边界 → Host`}</span></header>
        <div class="kf-prof-tlgrid" style="--tl-zoom:${st.zoom}">
          <div class="kf-prof-tllabels"><div class="kf-prof-tlrulerspacer"></div>${labels}</div>
          <div class="kf-prof-tlscroll" id="tlScroll">
            <div class="kf-prof-tlinner">
              <div class="kf-prof-tlruler">${ruler(w0, w1, st.zoom)}</div>
              ${rows}
            </div>
          </div>
        </div>
        ${detail}
      </section>

      <section class="kf-prof-card">
        <header><h3>时间账</h3><span>每层 ${fmt(s.layerDur, 1)} μs 的去向</span></header>
        <div class="kf-prof-card__body">
          <div class="kf-prof-sol">
            <div class="kf-prof-solrow is-bottleneck" data-unit="mte2">
              <span>关键路径</span>
              <div class="kf-prof-soltrack"><div class="kf-prof-solfill" style="width:${s.criticalUs / s.layerDur * 100}%"></div></div>
              <div class="kf-prof-solval">${fmt(s.criticalUs / s.layerDur * 100, 1)}%</div>
              <div class="kf-prof-soldetail">${fmt(s.criticalUs, 1)} μs · 11 个有真实依赖的任务</div>
            </div>
            <div class="kf-prof-solrow" data-unit="vector">
              <span>可并行旁路</span>
              <div class="kf-prof-soltrack"><div class="kf-prof-solfill" style="width:${s.parallelUs / s.layerDur * 100}%"></div></div>
              <div class="kf-prof-solval">${fmt(s.parallelUs / s.layerDur * 100, 1)}%</div>
              <div class="kf-prof-soldetail">${fmt(s.parallelUs, 1)} μs · rms_recip · fa_work_build · post_rms_reduce</div>
            </div>
            <div class="kf-prof-solrow" data-unit="mte3">
              <span>Barrier</span>
              <div class="kf-prof-soltrack"><div class="kf-prof-solfill" style="width:${s.barrierUs / s.layerDur * 100}%"></div></div>
              <div class="kf-prof-solval">${fmt(s.barrierUs / s.layerDur * 100, 1)}%</div>
              <div class="kf-prof-soldetail">${fmt(s.barrierUs, 1)} μs · Scope 间 3 处</div>
            </div>
          </div>
          <div class="kf-prof-verdict">
            <i>◆</i>
            <b>3 个任务与前驱无数据依赖，却被串行调度</b>
            <p><code>rms_recip</code> 只算 inv_rms 标量、<code>fa_work_build</code> 只读 seq_lens、<code>post_rms_reduce</code> 与 <code>residual_rms_cast</code> 消费同一份输入——三者都可以与相邻任务重叠，
            当前合计占用 <code>${fmt(s.parallelUs, 1)} μs/层</code>。若完全隐藏，可省 <code>${fmt(s.parallelStepMs, 3)} ms/step</code>，即 <code>TPOT −${fmt(s.parallelPct, 2)}%</code>。
            与源码里缺少 <code>pl.pipeline(stage=F)</code> 导致计算段无法跨块重叠是同一类问题。</p>
          </div>
        </div>
      </section>`;
  }

  window.PtoInferenceTimeline = { build, render, TRACKS, LAYER_CHAIN, PARALLELIZABLE };
})();
