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

  const STEP_COLORS = ['#80b7f7', '#4fb899', '#f2b857', '#a87ef0', '#f47b8c', '#56c7d6', '#9be05a'];
  const fmtUs = v => (Number.isFinite(v) ? `${v.toFixed(1)} µs` : '—');
  const fmtShortUs = v => (!Number.isFinite(v) ? '—' : v >= 100 ? `${Math.round(v)} µs` : `${v.toFixed(1)} µs`);

  function mount(host, options) {
    let mode = 'assignment'; // 'assignment' | 'pipeline'
    let stageId = 'all';
    let rank = 0;
    let filter = 'all'; // 'all' | 'AIC' | 'AIV' | 'MIX' | 'COMM'
    let selectedKernel = null;
    let pipelineViewMode = 'trace'; // 'trace' (24 Core Groups) | 'kernels' (kernel summary timeline)
    let rtSortBy = 'group'; // 'group' | 'duration'
    let rtShowSetup = null; // null = auto (>=12%)
    let rtIncludePrep = false;
    let rtHideIdle = false;
    let rtHighlightStep = null;
    let rtSelectedGroup = null;

    function getData() {
      return global.PTO_CSA_RUNTIME_DATA;
    }

    function getStageTrace(sid) {
      return global.PTO_CSA_STAGE_TRACES?.[sid] || null;
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

    function renderStageRuntimeTrace(data, stage, kernels, traceDef) {
      const rk = traceDef.ranks?.[String(rank)] || traceDef.ranks?.['0'];
      if (!rk || !rk.kernels) {
        return `<div class="csa-sv-rt-host"><div class="csa-rt-empty">当前 Rank 未采集到核组实测切片。</div></div>`;
      }
      const allowedNames = new Set(kernels.map(k => k.name));
      const metaByName = new Map(kernels.map(k => [k.name, k]));
      const prepSet = new Set(traceDef.prepSteps || []);
      const hasPrep = (traceDef.prepSteps || []).some(n => allowedNames.has(n) && rk.kernels[n]?.length);

      const visible = [];
      for (const [kname, entries] of Object.entries(rk.kernels)) {
        if (!allowedNames.has(kname)) continue;
        if (hasPrep && !rtIncludePrep && prepSet.has(kname)) continue;
        for (const [core, startUs, durUs, computeUs, setupUs] of entries) {
          visible.push({
            core,
            startUs,
            endUs: startUs + durUs,
            durUs,
            computeUs,
            setupUs,
            name: kname,
            unit: core < 24 ? 'aic' : 'aiv',
          });
        }
      }
      if (!visible.length) {
        return `<div class="csa-sv-rt-host"><div class="csa-rt-empty">当前筛选条件下无匹配的核上切片。</div></div>`;
      }

      const originUs = Math.min(...visible.map(s => s.startUs));
      const endUs = Math.max(...visible.map(s => s.endUs));
      const rawSpanUs = Math.max(0.5, endUs - originUs);
      const displaySpanUs = Math.max(rawSpanUs * 1.03, 1);
      const endPct = Math.min(100, (rawSpanUs / displaySpanUs) * 100);

      const stepMap = new Map();
      for (const s of visible) {
        let st = stepMap.get(s.name);
        if (!st) {
          const km = metaByName.get(s.name);
          st = {
            name: s.name,
            coreType: km?.isComm ? 'COMM' : (km?.coreType || (s.unit === 'aic' ? 'AIC' : 'AIV')),
            startUs: Infinity,
            endUs: -Infinity,
            count: 0,
            slotUs: 0,
            computeUs: 0,
            setupUs: 0,
          };
          stepMap.set(s.name, st);
        }
        st.startUs = Math.min(st.startUs, s.startUs);
        st.endUs = Math.max(st.endUs, s.endUs);
        st.count++;
        st.slotUs += s.durUs;
        st.computeUs += s.computeUs;
        st.setupUs += s.setupUs;
      }
      const steps = [...stepMap.values()].sort((a, b) => (a.startUs - b.startUs) || (a.endUs - b.endUs));
      steps.forEach((st, idx) => {
        st.color = STEP_COLORS[idx % STEP_COLORS.length];
        st.spanUs = Math.max(0, st.endUs - st.startUs);
      });
      const stepByName = new Map(steps.map(st => [st.name, st]));
      const multiStep = steps.length > 1;
      if (rtHighlightStep && !stepByName.has(rtHighlightStep)) rtHighlightStep = null;

      const totalSlotUs = visible.reduce((a, s) => a + s.durUs, 0);
      const totalSetupUs = visible.reduce((a, s) => a + s.setupUs, 0);
      const stageSetupShare = totalSetupUs / Math.max(1e-6, totalSlotUs);
      const showSetup = rtShowSetup != null ? rtShowSetup : stageSetupShare >= 0.12;

      const groups = Array.from({ length: 24 }, (_, g) => {
        const aicCore = g, aiv0Core = 24 + 2 * g, aiv1Core = 24 + 2 * g + 1;
        const aicSlices = visible.filter(s => s.core === aicCore).sort((a, b) => a.startUs - b.startUs);
        const aiv0Slices = visible.filter(s => s.core === aiv0Core).sort((a, b) => a.startUs - b.startUs);
        const aiv1Slices = visible.filter(s => s.core === aiv1Core).sort((a, b) => a.startUs - b.startUs);
        const sum = (arr, k) => arr.reduce((acc, x) => acc + x[k], 0);
        const allG = [...aicSlices, ...aiv0Slices, ...aiv1Slices];
        const aicDurUs = sum(aicSlices, 'durUs'), aiv0DurUs = sum(aiv0Slices, 'durUs'), aiv1DurUs = sum(aiv1Slices, 'durUs');
        const aicComputeUs = sum(aicSlices, 'computeUs'), aiv0ComputeUs = sum(aiv0Slices, 'computeUs'), aiv1ComputeUs = sum(aiv1Slices, 'computeUs');
        const aicSetupUs = sum(aicSlices, 'setupUs'), aiv0SetupUs = sum(aiv0Slices, 'setupUs'), aiv1SetupUs = sum(aiv1Slices, 'setupUs');
        const totalDurUs = aicDurUs + aiv0DurUs + aiv1DurUs;
        const totalSetupG = aicSetupUs + aiv0SetupUs + aiv1SetupUs;
        const firstStartUs = allG.length ? Math.min(...allG.map(s => s.startUs)) : originUs;
        const lastEndUs = allG.length ? Math.max(...allG.map(s => s.endUs)) : originUs;
        return {
          group: g,
          aicCore, aiv0Core, aiv1Core,
          aicSlices, aiv0Slices, aiv1Slices,
          aicDurUs, aiv0DurUs, aiv1DurUs,
          aicComputeUs, aiv0ComputeUs, aiv1ComputeUs,
          aicSetupUs, aiv0SetupUs, aiv1SetupUs,
          totalDurUs,
          totalSetupUs: totalSetupG,
          setupPct: (totalSetupG / Math.max(1e-6, totalDurUs)) * 100,
          sliceCount: allG.length,
          firstStartUs, lastEndUs,
          spanUs: allG.length ? Math.max(0, lastEndUs - firstStartUs) : 0,
          tier: 'fast',
        };
      });

      const activeGroups = groups.filter(g => g.sliceCount > 0);
      const activeDurs = activeGroups.map(g => g.totalDurUs);
      const minActiveDur = activeDurs.length ? Math.min(...activeDurs) : 0;
      const maxActiveDur = activeDurs.length ? Math.max(...activeDurs) : 0;
      const ratio = minActiveDur > 0.01 ? maxActiveDur / minActiveDur : 1;
      const slowCut = minActiveDur + (maxActiveDur - minActiveDur) * 0.5;
      for (const g of groups) {
        if (!g.sliceCount) g.tier = 'idle';
        else if (ratio >= 1.25 && g.totalDurUs >= slowCut) g.tier = 'slow';
        else g.tier = 'fast';
      }

      const displayGroups = (rtHideIdle ? activeGroups : groups).slice().sort((a, b) =>
        rtSortBy === 'duration' ? (b.totalDurUs - a.totalDurUs) || (a.group - b.group) : a.group - b.group
      );

      const niceAxisStep = span => {
        const candidates = [1, 2, 5, 10, 20, 25, 50, 100, 150, 200, 250, 500, 1000];
        const target = span / 5.5;
        return candidates.find(c => c >= target) || Math.ceil(target / 500) * 500;
      };
      const tickStep = niceAxisStep(rawSpanUs);
      const axisTicks = [];
      for (let t = 0; t <= rawSpanUs + 1e-6; t += tickStep) {
        const pct = (t / displaySpanUs) * 100;
        if (endPct - pct < 11 && t > 0) continue;
        axisTicks.push(`<span style="left:${pct.toFixed(2)}%">${t >= 10 || t === 0 ? Math.round(t) : t.toFixed(1)} µs</span>`);
      }

      const barStyle = (s, e) => {
        const left = Math.max(0, ((s - originUs) / displaySpanUs) * 100);
        const width = Math.max(0.32, ((e - s) / displaySpanUs) * 100);
        return `left:${left.toFixed(3)}%;width:${width.toFixed(3)}%`;
      };

      const renderBars = (list, roleCls) => list.map(s => {
        const dimCls = rtHighlightStep && rtHighlightStep !== s.name ? ' is-dimmed' : '';
        const stepColor = stepByName.get(s.name)?.color;
        const bgInline = multiStep && stepColor ? `;background:${stepColor}` : '';
        const colorCls = multiStep ? '' : ` ${roleCls}`;
        const coreLabel = s.core < 24 ? `AIC C${s.core}` : `AIV V${s.core}`;
        const tip = `${s.name} · ${coreLabel}\n相对 +${(s.startUs - originUs).toFixed(1)} → +${(s.endUs - originUs).toFixed(1)} µs\n计算 ${s.computeUs.toFixed(1)} µs · Setup ${s.setupUs.toFixed(1)} µs`;
        if (showSetup && s.setupUs > displaySpanUs * 0.002) {
          const setupEnd = Math.min(s.endUs, s.startUs + s.setupUs);
          return (
            `<span class="csa-rt-bar is-setup${dimCls}" style="${barStyle(s.startUs, setupEnd)}" title="${esc(tip)}"></span>` +
            `<span class="csa-rt-bar${colorCls}${dimCls}" style="${barStyle(setupEnd, s.endUs)}${bgInline}" title="${esc(tip)}"></span>`
          );
        }
        return `<span class="csa-rt-bar${colorCls}${dimCls}" style="${barStyle(s.startUs, s.endUs)}${bgInline}" title="${esc(tip)}"></span>`;
      }).join('');

      const idleCount = 24 - activeGroups.length;
      const metricsParts = [
        `<b>${steps.length}</b> 阶段`,
        `<b>${visible.length}</b> 块`,
        `Setup <b>${(stageSetupShare * 100).toFixed(1)}%</b>`,
        esc(traceDef.source || ''),
      ].filter(Boolean);

      const legendHtml = multiStep
        ? `${showSetup ? '<span><i class="is-setup"></i>Setup 等待</span>' : ''}<span>点击下方阶段胶囊可高亮单步</span>`
        : `<span><i class="is-aic"></i>AIC</span><span><i class="is-aiv0"></i>AIV 0</span><span><i class="is-aiv1"></i>AIV 1</span>${showSetup ? '<span><i class="is-setup"></i>Setup 等待</span>' : ''}`;

      const controlsHtml = `<div class="csa-rt-controls">
        <div class="csa-rt-controls-left">
          <div class="segmented-control segmented-control-muted csa-rt-seg" role="group" aria-label="核组排序">
            <button type="button" class="btn btn-sm${rtSortBy === 'group' ? ' is-selected' : ''}" data-sv-rt-sort="group" aria-pressed="${rtSortBy === 'group'}">Group 序</button>
            <button type="button" class="btn btn-sm${rtSortBy === 'duration' ? ' is-selected' : ''}" data-sv-rt-sort="duration" aria-pressed="${rtSortBy === 'duration'}">耗时降序</button>
          </div>
          <div class="segmented-control segmented-control-muted csa-rt-seg" role="group" aria-label="叠加信号">
            <button type="button" class="btn btn-sm${showSetup ? ' is-selected' : ''}" data-sv-rt-toggle="setup" aria-pressed="${showSetup}">Setup 拆分</button>
            ${hasPrep ? `<button type="button" class="btn btn-sm${rtIncludePrep ? ' is-selected' : ''}" data-sv-rt-toggle="prep" aria-pressed="${rtIncludePrep}">含前序准备</button>` : ''}
            ${idleCount > 0 ? `<button type="button" class="btn btn-sm${rtHideIdle ? ' is-selected' : ''}" data-sv-rt-toggle="hideIdle" aria-pressed="${rtHideIdle}">隐藏空组 (${idleCount})</button>` : ''}
          </div>
          <span class="csa-rt-metrics-inline">${metricsParts.join('<i>·</i>')}</span>
        </div>
        <div class="csa-rt-legend" aria-label="泳道图例">${legendHtml}</div>
      </div>`;

      const stepStripHtml = multiStep ? `<div class="csa-rt-steps" aria-label="阶段流水步骤与交接">${steps.map((st, idx) => {
        const isAct = rtHighlightStep === st.name;
        const isDim = rtHighlightStep && !isAct;
        let linkHtml = '';
        if (idx > 0) {
          const prev = steps[idx - 1];
          const delta = st.startUs - prev.endUs;
          if (delta < -1) linkHtml = `<span class="csa-rt-step-link is-overlap" title="${esc(prev.name)} 与 ${esc(st.name)} 存在真实重叠 ${Math.abs(delta).toFixed(1)} µs">⇢ 重叠 ${Math.abs(delta).toFixed(0)}µs</span>`;
          else if (delta > 2) linkHtml = `<span class="csa-rt-step-link is-gap" title="${esc(prev.name)} 结束至 ${esc(st.name)} 启动之间存在 ${delta.toFixed(1)} µs 间隔">→ 空窗 +${delta.toFixed(0)}µs</span>`;
          else linkHtml = `<span class="csa-rt-step-link">→</span>`;
        }
        const relWin = `${Math.round(st.startUs - originUs)}–${Math.round(st.endUs - originUs)}µs`;
        const tip = `点击仅高亮 ${st.name}（${st.coreType} · ${st.count} 块 · 跨度 ${fmtUs(st.spanUs)} · Setup ${(st.setupUs / Math.max(1e-6, st.slotUs) * 100).toFixed(1)}%）`;
        return `${linkHtml}<button type="button" class="csa-rt-step-chip${isAct ? ' is-active' : ''}${isDim ? ' is-dimmed' : ''}" data-sv-rt-step="${esc(st.name)}" aria-pressed="${isAct}" title="${esc(tip)}"><i style="background:${st.color}"></i><span>${esc(st.name)}</span><small>${st.coreType} · ${relWin}</small></button>`;
      }).join('')}</div>` : '';

      const axisHtml = `<div class="csa-rt-axis">
        <div class="csa-rt-col-id">Group</div>
        <div class="csa-rt-axis-track">
          ${axisTicks.join('')}
          <span class="csa-rt-end-flag is-right-aligned" style="left:${endPct.toFixed(2)}%" title="该 Stage 链路最慢完成时刻（相对起点 ${rawSpanUs.toFixed(1)} µs）">${fmtShortUs(rawSpanUs)} 完成线</span>
        </div>
        <div class="csa-rt-col-stat">组跨度</div>
      </div>`;

      const rowsHtml = displayGroups.map(g => {
        const isSel = rtSelectedGroup === g.group;
        const tierCls = g.tier === 'slow' ? 'is-slow' : g.tier === 'idle' ? 'is-idle' : 'is-fast';
        const setupPctInt = Math.round(g.setupPct);
        const secondStat = !g.sliceCount
          ? `<span class="csa-rt-stat-merge">0 块</span>`
          : showSetup
            ? `<span class="csa-rt-stat-merge${setupPctInt >= 30 ? ' is-warn' : ''}">${setupPctInt}%</span>`
            : `<span class="csa-rt-stat-merge">${g.sliceCount} 块</span>`;
        return `<button type="button" class="csa-rt-row ${tierCls}${isSel ? ' is-selected' : ''}" data-sv-rt-group="${g.group}" aria-pressed="${isSel}">
          <div class="csa-rt-col-id">
            <span class="csa-rt-dot-tier ${tierCls}" aria-hidden="true"></span>
            <strong>G${String(g.group).padStart(2, '0')}</strong>
            <small>C${g.aicCore}</small>
          </div>
          <div class="csa-rt-track">
            <div class="csa-rt-end-overlay" aria-hidden="true"><span class="csa-rt-end-line" style="left:${endPct.toFixed(2)}%"></span></div>
            <div class="csa-rt-sublane">
              <span class="csa-rt-subtag${g.aicSlices.length ? '' : ' is-muted'}">C</span>
              <div class="csa-rt-lane-bg">${renderBars(g.aicSlices, 'is-aic')}</div>
            </div>
            <div class="csa-rt-sublane">
              <span class="csa-rt-subtag${g.aiv0Slices.length ? '' : ' is-muted'}">V0</span>
              <div class="csa-rt-lane-bg">${renderBars(g.aiv0Slices, 'is-aiv0')}</div>
            </div>
            <div class="csa-rt-sublane">
              <span class="csa-rt-subtag${g.aiv1Slices.length ? '' : ' is-muted'}">V1</span>
              <div class="csa-rt-lane-bg">${renderBars(g.aiv1Slices, 'is-aiv1')}</div>
            </div>
          </div>
          <div class="csa-rt-col-stat">
            <span class="csa-rt-stat-main">${g.sliceCount ? fmtShortUs(g.spanUs) : '—'}</span>
            ${secondStat}
          </div>
        </button>`;
      }).join('');

      let inspector = '';
      if (rtSelectedGroup != null) {
        const g = groups[rtSelectedGroup];
        if (g) {
          const allG = [...g.aicSlices, ...g.aiv0Slices, ...g.aiv1Slices];
          const byKernel = new Map();
          for (const s of allG) {
            const cur = byKernel.get(s.name) || {
              name: s.name, count: 0, durUs: 0, setupUs: 0, startUs: Infinity,
              color: stepByName.get(s.name)?.color || (s.unit === 'aic' ? '#80b7f7' : '#a87ef0'),
            };
            cur.count++;
            cur.durUs += s.durUs;
            cur.setupUs += s.setupUs;
            cur.startUs = Math.min(cur.startUs, s.startUs);
            byKernel.set(s.name, cur);
          }
          const kList = [...byKernel.values()].sort((a, b) => a.startUs - b.startUs);
          const tierBadgeCls = g.tier === 'slow' ? 'is-slow' : g.tier === 'idle' ? 'is-idle' : 'is-fast';
          const kListHtml = kList.length
            ? `<div class="csa-rt-klist">${kList.map(k => {
                const sPct = Math.round((k.setupUs / Math.max(1e-6, k.durUs)) * 100);
                return `<button type="button" class="csa-rt-krow" data-sv-rt-step="${esc(k.name)}" title="高亮 ${esc(k.name)}">
                  <span class="csa-rt-kname"><i class="csa-rt-dot" style="background:${k.color}"></i>${esc(k.name)}</span>
                  <span class="csa-rt-kstat">${fmtShortUs(k.durUs)} · ${k.count}块${sPct >= 15 ? ` · S${sPct}%` : ''}</span>
                </button>`;
              }).join('')}</div>`
            : `<p class="csa-rt-note">当前筛选条件下该核组无执行切片。</p>`;

          inspector = `<aside class="csa-rt-inspector" aria-label="Core Group 实测详情">
            <header class="csa-rt-insp-head">
              <div>
                <span class="csa-rt-insp-kicker">RANK ${rank} · ${g.sliceCount ? `+${(g.firstStartUs - originUs).toFixed(0)}→+${(g.lastEndUs - originUs).toFixed(0)} µs` : '无活跃切片'}</span>
                <h3>Group ${String(g.group).padStart(2, '0')} <span class="csa-rt-tier ${tierBadgeCls}">${g.sliceCount ? `${g.sliceCount} 块` : '空载组'}</span></h3>
              </div>
              <button type="button" class="btn btn-ghost btn-sm csa-rt-close" data-sv-rt-close aria-label="关闭详情">×</button>
            </header>
            <div class="csa-rt-insp-body">
              <section class="csa-rt-insp-sec">
                <h4>核组实测（1 AIC + 2 AIV）</h4>
                <table class="csa-rt-mini-table">
                  <thead><tr><th>核</th><th>槽位核时</th><th>计算 / Setup</th></tr></thead>
                  <tbody>
                    <tr><td><i class="csa-rt-dot is-aic"></i>AIC · C${g.aicCore}</td><td>${g.aicSlices.length ? fmtUs(g.aicDurUs) : '—'}</td><td>${g.aicSlices.length ? `${g.aicComputeUs.toFixed(0)} / ${g.aicSetupUs.toFixed(0)} µs` : '0 块'}</td></tr>
                    <tr><td><i class="csa-rt-dot is-aiv0"></i>AIV 0 · V${g.aiv0Core}</td><td>${g.aiv0Slices.length ? fmtUs(g.aiv0DurUs) : '—'}</td><td>${g.aiv0Slices.length ? `${g.aiv0ComputeUs.toFixed(0)} / ${g.aiv0SetupUs.toFixed(0)} µs` : '0 块'}</td></tr>
                    <tr><td><i class="csa-rt-dot is-aiv1"></i>AIV 1 · V${g.aiv1Core}</td><td>${g.aiv1Slices.length ? fmtUs(g.aiv1DurUs) : '—'}</td><td>${g.aiv1Slices.length ? `${g.aiv1ComputeUs.toFixed(0)} / ${g.aiv1SetupUs.toFixed(0)} µs` : '0 块'}</td></tr>
                  </tbody>
                </table>
              </section>
              <section class="csa-rt-insp-sec">
                <h4>本核组执行的阶段任务（${kList.length}）</h4>
                ${kListHtml}
              </section>
              <div class="csa-rt-actions">
                ${stage ? `<button type="button" class="btn btn-sm csa-rt-cta-primary" data-sv-goto-graph="${esc(stage.id)}">在 Graph 中定位 ↗</button>` : ''}
                ${(stage?.source || []).slice(0, 2).map(([f, a, z]) =>
                  `<button type="button" class="btn btn-ghost btn-sm" data-sv-source="${esc(f)}:${a}:${z}">${esc(f)} : L${a}–${z} ↗</button>`
                ).join('')}
              </div>
            </div>
          </aside>`;
        }
      }

      return `<div class="csa-sv-rt-host">
        <div class="csa-rt-shell${inspector ? ' has-inspector' : ''}">
          <div class="csa-rt-main">
            ${controlsHtml}
            ${stepStripHtml}
            <div class="csa-rt-board">
              ${axisHtml}
              <div class="csa-rt-rows">${rowsHtml}</div>
            </div>
          </div>
          ${inspector}
        </div>
      </div>`;
    }

    function render({ preserveScroll = false } = {}) {
      const scrollTop = preserveScroll ? host.querySelector('.csa-sv-main')?.scrollTop : null;
      const rtScrollTop = preserveScroll ? host.querySelector('.csa-rt-rows')?.scrollTop : null;
      const data = getData();
      if (!data) return;
      const { stage, kernels } = currentStageEntries();
      const visible = filteredKernels(kernels);
      const traceDef = mode === 'pipeline' && stageId !== 'all' ? getStageTrace(stageId) : null;
      const useTrace = !!traceDef && pipelineViewMode === 'trace';
      const selectedObj = !useTrace ? (visible.find(k => k.name === selectedKernel) || kernels.find(k => k.name === selectedKernel) || null) : null;
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
            ${traceDef ? `<div class="csa-sv-seg" role="group" aria-label="Pipeline 视图模式">
              <button type="button" class="${useTrace ? 'is-selected' : ''}" data-sv-pmode="trace">Runtime Trace</button>
              <button type="button" class="${!useTrace ? 'is-selected' : ''}" data-sv-pmode="kernels">Kernel 总览</button>
            </div>` : ''}
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
        <div class="csa-sv-main${useTrace ? ' is-rt' : ''}">
          ${mode === 'assignment' ? renderAssignment(data, stage, visible) : (useTrace ? renderStageRuntimeTrace(data, stage, visible, traceDef) : renderPipeline(data, stage, visible))}
        </div>
        <aside class="pto-ide-frame__pane csa-sv-inspector" ${selectedObj ? '' : 'hidden'}>
          ${selectedObj ? inspectorHtml(selectedObj, stage, data) : ''}
        </aside>
      </div>`;
      if (scrollTop != null) {
        const mainEl = host.querySelector('.csa-sv-main');
        if (mainEl) mainEl.scrollTop = scrollTop;
      }
      if (rtScrollTop != null) {
        const rowsEl = host.querySelector('.csa-rt-rows');
        if (rowsEl) rowsEl.scrollTop = rtScrollTop;
      }
    }

    host.addEventListener('click', event => {
      const pmodeBtn = event.target.closest('[data-sv-pmode]');
      if (pmodeBtn) {
        const nextPmode = pmodeBtn.dataset.svPmode;
        if (nextPmode !== pipelineViewMode) {
          pipelineViewMode = nextPmode;
          if (nextPmode === 'trace') {
            if (selectedKernel) {
              rtHighlightStep = selectedKernel;
              const td = getStageTrace(stageId);
              if ((td?.prepSteps || []).includes(selectedKernel)) rtIncludePrep = true;
            }
          } else if (nextPmode === 'kernels') {
            if (rtHighlightStep) selectedKernel = rtHighlightStep;
          }
        }
        render();
        return;
      }
      const rtSortBtn = event.target.closest('[data-sv-rt-sort]');
      if (rtSortBtn) {
        rtSortBy = rtSortBtn.dataset.svRtSort;
        render();
        return;
      }
      const rtStepBtn = event.target.closest('[data-sv-rt-step]');
      if (rtStepBtn) {
        const stepName = rtStepBtn.dataset.svRtStep;
        rtHighlightStep = rtHighlightStep === stepName ? null : stepName;
        selectedKernel = rtHighlightStep;
        render({ preserveScroll: true });
        return;
      }
      const rtToggleBtn = event.target.closest('[data-sv-rt-toggle]');
      if (rtToggleBtn) {
        const key = rtToggleBtn.dataset.svRtToggle;
        if (key === 'setup') {
          rtShowSetup = rtToggleBtn.getAttribute('aria-pressed') !== 'true';
        } else if (key === 'prep') {
          rtIncludePrep = !rtIncludePrep;
          rtHighlightStep = null;
          rtShowSetup = null;
          rtSelectedGroup = null;
        } else if (key === 'hideIdle') {
          rtHideIdle = !rtHideIdle;
        }
        render({ preserveScroll: true });
        return;
      }
      const rtGroupBtn = event.target.closest('[data-sv-rt-group]');
      if (rtGroupBtn) {
        const g = Number(rtGroupBtn.dataset.svRtGroup);
        rtSelectedGroup = rtSelectedGroup === g ? null : g;
        render({ preserveScroll: true });
        return;
      }
      if (event.target.closest('[data-sv-rt-close]')) {
        rtSelectedGroup = null;
        render({ preserveScroll: true });
        return;
      }
      const rankBtn = event.target.closest('[data-sv-rank]');
      if (rankBtn) {
        rank = Number(rankBtn.dataset.svRank);
        rtSelectedGroup = null;
        options.onRankChange?.(rank);
        render();
        return;
      }
      const filterBtn = event.target.closest('[data-sv-filter]');
      if (filterBtn) {
        filter = filterBtn.dataset.svFilter;
        rtHighlightStep = null;
        rtShowSetup = null;
        rtSelectedGroup = null;
        render();
        return;
      }
      if (event.target.closest('[data-sv-close]')) {
        selectedKernel = null;
        rtHighlightStep = null;
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
        rtHighlightStep = selectedKernel;
        render({ preserveScroll: true });
      }
    });

    return {
      update(nextMode, nextStageId, nextRank) {
        const stageChanged = stageId !== nextStageId;
        const modeChanged = mode !== nextMode;
        mode = nextMode;
        stageId = nextStageId;
        if (nextRank != null) rank = nextRank;
        if (stageChanged) {
          selectedKernel = null;
          filter = 'all';
          pipelineViewMode = 'trace';
          rtSortBy = 'group';
          rtShowSetup = null;
          rtIncludePrep = false;
          rtHideIdle = false;
          rtHighlightStep = null;
          rtSelectedGroup = null;
        } else if (modeChanged) {
          if (mode === 'pipeline' && selectedKernel) {
            rtHighlightStep = selectedKernel;
            const td = getStageTrace(stageId);
            if ((td?.prepSteps || []).includes(selectedKernel)) rtIncludePrep = true;
          } else if (mode === 'assignment' && !selectedKernel && rtHighlightStep) {
            selectedKernel = rtHighlightStep;
          }
        }
        render();
      },
    };
  }

  global.PtoCsaStageViews = Object.freeze({ mount });
})(window);
