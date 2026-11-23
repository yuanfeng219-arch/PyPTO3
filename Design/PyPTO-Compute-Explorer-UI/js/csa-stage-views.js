/* CSA Real-Data Stage Views for Assignment & Pipeline across All Stages and individual stages.
 * Powered by measured DFX outputs from _jit_l3_decode_csa_20260903_010617. */
(function (global) {
  'use strict';

  const esc = value => String(value ?? '').replace(/[&<>"']/g, ch =>
    ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' })[ch]);

  const badgeClass = kernel =>
    kernel.isComm ? 'is-comm' : kernel.coreType === 'MIX' ? 'is-mix' : kernel.coreType === 'AIC' ? 'is-aic' : 'is-aiv';

  const badgeLabel = kernel =>
    kernel.isComm ? 'COMM' : kernel.coreType === 'MIX' ? 'MIX 1:2' : kernel.coreType;

  function mount(host, options) {
    let mode = 'assignment'; // 'assignment' | 'pipeline'
    let stageId = 'all';
    let rank = 0;
    let filter = 'all'; // 'all' | 'AIC' | 'AIV' | 'MIX' | 'COMM'
    let selectedKernel = null;

    function getData() {
      return global.PTO_CSA_RUNTIME_DATA;
    }

    function currentStageEntries() {
      const data = getData();
      if (!data) return { stage: null, kernels: [], stages: [] };
      if (stageId === 'all') {
        const kernels = data.stages.flatMap(st =>
          st.kernels.map(k => ({ ...k, stageId: st.id, stageLabel: st.label, stageSource: st.source, stageHints: st.perfHints }))
        );
        return { stage: null, kernels, stages: data.stages };
      }
      const st = data.stages.find(s => s.id === stageId);
      const kernels = st
        ? st.kernels.map(k => ({ ...k, stageId: st.id, stageLabel: st.label, stageSource: st.source, stageHints: st.perfHints }))
        : [];
      return { stage: st, kernels, stages: data.stages };
    }

    function filteredKernels(kernels) {
      return kernels.filter(k => {
        if (filter === 'all') return true;
        if (filter === 'COMM') return k.isComm;
        return k.coreType === filter;
      });
    }

    function inspectorHtml(kernel, stage, data) {
      if (!kernel) return '';
      const r0 = kernel.ranks?.[0];
      const r1 = kernel.ranks?.[1];
      const cur = kernel.ranks?.[rank] || r0;
      const stageObj = stage || data.stages.find(s => s.id === kernel.stageId);
      const upstream = (stageObj?.internalEdges || []).filter(e => e.to === kernel.name).map(e => e.from);
      const downstream = (stageObj?.internalEdges || []).filter(e => e.from === kernel.name).map(e => e.to);
      const hints = (stageObj?.perfHints || []).slice(0, 3);
      const isQkPv = kernel.stageId === 'sparse/attention/qk-pv';

      return `<header class="pto-ide-frame__pane-header">
        <div>
          <b>${esc(kernel.name)}</b>
          <small>${esc(kernel.stageLabel || '')} · ${esc(badgeLabel(kernel))} · block_num=${kernel.blockNum}</small>
        </div>
        <button type="button" class="btn btn-ghost btn-sm" data-sv-close aria-label="关闭详情">×</button>
      </header>
      <div class="pto-ide-frame__pane-body">
        <dl class="csa-sv-dl">
          <dt>核型与编组</dt><dd>${esc(kernel.coreType)}${kernel.coreType === 'MIX' ? ' (1 AIC + 2 AIV)' : ''}${kernel.isComm ? ' · 跨 Rank 通信' : ''}</dd>
          <dt>SPMD Blocks</dt><dd>${kernel.blockNum} × ${kernel.taskCount} Task</dd>
          <dt>Rank ${rank} 窗口</dt><dd>${cur ? `${cur.startUs} → ${cur.endUs} µs (跨度 ${cur.spanUs} µs)` : '未观测'}</dd>
          <dt>单切片耗时</dt><dd>${cur ? `min ${cur.minDurUs} / med ${cur.medDurUs} / max ${cur.maxDurUs} µs` : '—'}</dd>
          <dt>双卡对比</dt><dd>R0 med ${r0?.medDurUs ?? '—'} µs · R1 med ${r1?.medDurUs ?? '—'} µs</dd>
          <dt>参与核数</dt><dd>${cur ? `${cur.coresUsed} 核 (${cur.aicUsed} AIC / ${cur.aivUsed} AIV · ${cur.slices} 切片)` : '—'}</dd>
          <dt>段内上游</dt><dd>${esc(upstream.join(', ') || '跨阶段输入')}</dd>
          <dt>段内下游</dt><dd>${esc(downstream.join(', ') || '跨阶段输出')}</dd>
        </dl>
        <div class="csa-sv-actions">
          ${isQkPv ? `<button type="button" class="btn btn-sm" data-sv-open-qkpv="${mode}">下钻 qk_pv 精细视图 →</button>` : ''}
          ${stageObj ? `<button type="button" class="btn btn-ghost btn-sm" data-sv-goto-graph="${esc(stageObj.id)}">在 Graph 中定位 ↗</button>` : ''}
          ${(stageObj?.source || []).slice(0, 2).map(([f, a, z]) =>
            `<button type="button" class="btn btn-ghost btn-sm" data-sv-source="${esc(f)}:${a}:${z}">${esc(f)} L${a}–${z} ↗</button>`
          ).join('')}
        </div>
        ${hints.length ? `<div class="csa-sv-hints">
          <b>编译性能诊断（perf_hints.log）</b>
          ${hints.map(h => `<div class="csa-sv-hint-item">
            <div><code>${esc(h.code)}</code> · ${esc(h.rule)}</div>
            <div>${esc(h.detail)}</div>
            <button type="button" class="btn btn-ghost btn-sm" data-sv-source="${esc(h.file)}:${Math.max(1, h.line - 3)}:${h.line + 6}">${esc(h.file)}:${h.line} ↗</button>
          </div>`).join('')}
        </div>` : ''}
      </div>`;
    }

    function renderAssignment(data, stage, kernels) {
      const rMeta = data.meta.ranks[rank];
      const maxAic = Math.max(0, ...kernels.map(k => k.ranks?.[rank]?.aicUsed || 0));
      const maxAiv = Math.max(0, ...kernels.map(k => k.ranks?.[rank]?.aivUsed || 0));
      const totalSlices = kernels.reduce((acc, k) => acc + (k.ranks?.[rank]?.slices || 0), 0);
      const ownership = stage
        ? stage.dataOwnership
        : `TP=2 双卡（Rank 0 与 Rank 1）各配置 24 AIC + 48 AIV 共 72 核，单层执行 ${data.meta.totalTasks} 个 Task、${data.meta.totalTensors} 个 Tensor 与 ${data.meta.totalEdges} 条依赖边。Rank 0 实测总跨度 ${data.meta.ranks[0].spanUs} µs，Rank 1 实测总跨度 ${data.meta.ranks[1].spanUs} µs。`;
      const hostNote = `Host STRACE（Rank ${rank}）：冷启动 ${Math.round(rMeta.coldWallUs / 100) / 10} ms → 热执行 ${Math.round(rMeta.warmWallUs / 100) / 10} ms（Orch ${rMeta.warmOrchUs} µs · Sched ${rMeta.warmSchedUs} µs）。`;

      const rows = kernels.map(k => {
        const rs = k.ranks?.[rank];
        const isSel = selectedKernel === k.name;
        return `<tr class="${isSel ? 'is-selected' : ''}" data-sv-kernel="${esc(k.name)}">
          <td class="csa-sv-kname">${esc(k.name)}</td>
          <td><span class="csa-sv-badge ${badgeClass(k)}">${esc(badgeLabel(k))}</span></td>
          <td>${stage ? '' : esc(k.stageLabel)}</td>
          <td>${k.blockNum} × ${k.taskCount}</td>
          <td>${rs ? `${rs.coresUsed} 核 (${rs.slices} 切片)` : '—'}</td>
          <td>${rs ? `${rs.medDurUs} µs` : '—'}</td>
          <td>${rs ? `${rs.maxDurUs} µs` : '—'}</td>
        </tr>`;
      }).join('');

      return `<div class="csa-sv-cards">
        <div class="csa-sv-card">
          <h3>数据归属与跨 Rank 语义</h3>
          <p>${esc(ownership)}</p>
        </div>
        <div class="csa-sv-card">
          <h3>72 核硬件分配（Rank ${rank} · a2a3 实测）</h3>
          <p>${esc(hostNote)}</p>
          <div class="csa-sv-corebar">
            <div class="csa-sv-coremeter">
              <span>AIC 峰值：${maxAic} / 24</span>
              <div class="csa-sv-coremeter-track"><div class="csa-sv-coremeter-fill is-aic" style="width:${Math.round(maxAic / 24 * 100)}%"></div></div>
            </div>
            <div class="csa-sv-coremeter">
              <span>AIV 峰值：${maxAiv} / 48 · 共 ${totalSlices} 执行切片</span>
              <div class="csa-sv-coremeter-track"><div class="csa-sv-coremeter-fill is-aiv" style="width:${Math.round(maxAiv / 48 * 100)}%"></div></div>
            </div>
          </div>
        </div>
      </div>
      <div class="csa-sv-table-wrap">
        <table class="csa-sv-table">
          <thead>
            <tr>
              <th>Kernel</th>
              <th>类型</th>
              <th>${stage ? '' : '阶段'}</th>
              <th>SPMD Blocks</th>
              <th>Rank ${rank} 核分配</th>
              <th>中位耗时</th>
              <th>最大耗时</th>
            </tr>
          </thead>
          <tbody>${rows}</tbody>
        </table>
      </div>`;
    }

    function renderPipeline(data, stage, kernels) {
      const active = kernels
        .filter(k => k.ranks?.[rank])
        .sort((a, b) => a.ranks[rank].startUs - b.ranks[rank].startUs);
      if (!active.length) {
        return `<div class="csa-sv-card"><p>当前筛选条件下无匹配 Kernel。</p></div>`;
      }
      const minTs = Math.min(...active.map(k => k.ranks[rank].startUs));
      const maxTs = Math.max(...active.map(k => k.ranks[rank].endUs));
      const span = Math.max(1, maxTs - minTs);

      const ticks = [0, 0.25, 0.5, 0.75, 1].map(ratio => {
        const val = minTs + span * ratio;
        return `<span style="left:${ratio * 100}%">${val.toFixed(0)} µs</span>`;
      }).join('');

      const lanes = active.map(k => {
        const rs = k.ranks[rank];
        const leftPct = ((rs.startUs - minTs) / span) * 100;
        const widthPct = Math.max(0.6, ((rs.endUs - rs.startUs) / span) * 100);
        const isSel = selectedKernel === k.name;
        return `<div class="csa-sv-lane${isSel ? ' is-selected' : ''}" data-sv-kernel="${esc(k.name)}" role="button" tabindex="0" title="${esc(`${k.name} · ${rs.startUs}–${rs.endUs} µs · med ${rs.medDurUs} µs · ${rs.slices} 切片`)}">
          <div class="csa-sv-lane-label">
            <span class="csa-sv-badge ${badgeClass(k)}">${esc(badgeLabel(k))}</span>
            <span>${esc(k.name)}</span>
          </div>
          <div class="csa-sv-lane-track">
            <div class="csa-sv-bar ${badgeClass(k)}" style="left:${leftPct.toFixed(2)}%;width:${widthPct.toFixed(2)}%"></div>
          </div>
          <div class="csa-sv-lane-stat">med ${rs.medDurUs} µs · ×${rs.slices}</div>
        </div>`;
      }).join('');

      const rankCompare = `Rank 0 总跨度 ${data.meta.ranks[0].spanUs} µs · Rank 1 总跨度 ${data.meta.ranks[1].spanUs} µs（双卡同步差 ${Math.abs(data.meta.ranks[0].spanUs - data.meta.ranks[1].spanUs).toFixed(1)} µs）`;

      return `<div class="csa-sv-timeline">
        <div class="csa-sv-axis">
          <span>Kernel（按 Rank ${rank} 实测启动排序）</span>
          <div class="csa-sv-axis-ticks">${ticks}</div>
          <span style="text-align:right">中位耗时 · 切片</span>
        </div>
        ${lanes}
        <div class="csa-sv-legend">
          <span><i style="background:#80b7f7"></i>AIC Cube</span>
          <span><i style="background:#b58af6"></i>AIV Vector</span>
          <span><i style="background:#f2c45f"></i>MIX 1:2 混核（点击 qk_pv 可下钻因果泳道）</span>
          <span><i style="background:#f29a62"></i>跨 Rank 通信</span>
          <span style="margin-left:auto">${esc(rankCompare)}</span>
        </div>
      </div>`;
    }

    function render({ preserveScroll = false } = {}) {
      const scrollTop = preserveScroll ? host.querySelector('.csa-sv-main')?.scrollTop : null;
      const data = getData();
      if (!data) return;
      const { stage, kernels } = currentStageEntries();
      const visible = filteredKernels(kernels);
      const selectedObj = visible.find(k => k.name === selectedKernel) || kernels.find(k => k.name === selectedKernel) || null;
      const rInfo = stage ? stage.ranks[rank] : {
        spanUs: data.meta.ranks[rank].spanUs,
        slices: data.meta.ranks[rank].workerSlices,
        hbViolations: data.meta.ranks[rank].hbViolations,
      };
      const title = stage ? stage.full : 'DeepSeek V4 Decode CSA · 整层实测总览（8 Stages · 64 Kernels）';
      const subtitle = stage ? stage.short : `数据来源：_jit_l3_decode_csa_20260903_010617（a2a3 · chip_swimlane level 4 · ${data.meta.totalTasks} Tasks · ${data.meta.totalEdges} 依赖边）`;

      host.innerHTML = `<div class="csa-sv-shell">
        <div class="csa-sv-header">
          <div class="csa-sv-title-group">
            <div class="csa-sv-title-row">
              <h2>${esc(title)}</h2>
              <span class="csa-sv-pill">Rank ${rank} · ${rInfo.spanUs} µs</span>
              <span class="csa-sv-pill">${rInfo.slices} 实测切片</span>
              ${rInfo.hbViolations ? `<span class="csa-sv-pill is-warn">${rInfo.hbViolations} hb_violation</span>` : ''}
            </div>
            <p class="csa-sv-sub">${esc(subtitle)}</p>
          </div>
          <div class="csa-sv-controls">
            <div class="csa-sv-seg" role="group" aria-label="切换 Rank">
              <button type="button" class="${rank === 0 ? 'is-selected' : ''}" data-sv-rank="0">Rank 0 (${data.meta.ranks[0].spanUs} µs)</button>
              <button type="button" class="${rank === 1 ? 'is-selected' : ''}" data-sv-rank="1">Rank 1 (${data.meta.ranks[1].spanUs} µs)</button>
            </div>
            <div class="csa-sv-seg" role="group" aria-label="核型筛选">
              ${[['all', '全部'], ['AIC', 'AIC'], ['AIV', 'AIV'], ['MIX', 'MIX 1:2'], ['COMM', '跨 Rank']].map(([key, label]) =>
                `<button type="button" class="${filter === key ? 'is-selected' : ''}" data-sv-filter="${key}">${label}</button>`
              ).join('')}
            </div>
          </div>
        </div>
        <div class="csa-sv-main">
          ${mode === 'assignment' ? renderAssignment(data, stage, visible) : renderPipeline(data, stage, visible)}
        </div>
        <aside class="pto-ide-frame__pane csa-sv-inspector" ${selectedObj ? '' : 'hidden'}>
          ${selectedObj ? inspectorHtml(selectedObj, stage, data) : ''}
        </aside>
      </div>`;
      if (scrollTop != null) host.querySelector('.csa-sv-main').scrollTop = scrollTop;
    }

    host.addEventListener('click', event => {
      const rankBtn = event.target.closest('[data-sv-rank]');
      if (rankBtn) {
        rank = Number(rankBtn.dataset.svRank);
        options.onRankChange?.(rank);
        render();
        return;
      }
      const filterBtn = event.target.closest('[data-sv-filter]');
      if (filterBtn) {
        filter = filterBtn.dataset.svFilter;
        render();
        return;
      }
      if (event.target.closest('[data-sv-close]')) {
        selectedKernel = null;
        render({ preserveScroll: true });
        return;
      }
      const openQk = event.target.closest('[data-sv-open-qkpv]');
      if (openQk) {
        options.onOpenQkPv?.(openQk.dataset.svOpenQkpv);
        return;
      }
      const goGraph = event.target.closest('[data-sv-goto-graph]');
      if (goGraph) {
        options.onNavigateGraph?.(goGraph.dataset.svGotoGraph);
        return;
      }
      const srcBtn = event.target.closest('[data-sv-source]');
      if (srcBtn) {
        const [f, a, z] = srcBtn.dataset.svSource.split(':');
        options.onSource?.(f, Number(a), Number(z));
        return;
      }
      const kRow = event.target.closest('[data-sv-kernel]');
      if (kRow) {
        const name = kRow.dataset.svKernel;
        selectedKernel = selectedKernel === name ? null : name;
        render({ preserveScroll: true });
      }
    });

    return {
      update(nextMode, nextStageId, nextRank) {
        const stageChanged = stageId !== nextStageId || mode !== nextMode;
        mode = nextMode;
        stageId = nextStageId;
        if (nextRank != null) rank = nextRank;
        if (stageChanged) {
          selectedKernel = null;
          filter = 'all';
        }
        render();
      },
    };
  }

  global.PtoCsaStageViews = Object.freeze({ mount });
})(window);
