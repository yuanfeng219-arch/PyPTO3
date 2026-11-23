/* qk_pv 72-core (24 Core Groups) Runtime Trace bridge view — denoised visual hierarchy.
 * Backed by unmodified per-core DFX measurements from _jit_l3_decode_csa_20260903_010617.
 */
(function (global) {
  'use strict';

  const esc = v => String(v ?? '').replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
  const fmtUs = v => (Number.isFinite(v) ? v.toFixed(1) + ' µs' : '—');
  const fmtSigned = v => (Number.isFinite(v) ? (v >= 0 ? '+' : '−') + Math.abs(v).toFixed(1) + ' µs' : '—');

  function mount(host, options) {
    const opts = options || {};
    let state = {
      rank: 0,
      scenario: null,
      selectedGroup: null, // null = full-bleed swimlane; click any group to open compact inspector
      sortBy: 'group', // 'group' | 'duration'
      showAiv1Idle: true,
      showMerge: true,
    };

    function rankData() {
      const root = global.PTO_CSA_QK_RUNTIME_TRACE;
      if (!root || !root.ranks) return null;
      return root.ranks[String(state.rank)] || root.ranks['0'] || null;
    }

    function laneStats(groupIndex) {
      const block = state.scenario?.blocks?.[groupIndex];
      if (!block) return null;
      let original = 0, compressed = 0, invalid = 0;
      for (const entry of block.entries) {
        const item = state.scenario.byId.get(entry.id);
        if (!item) continue;
        if (!item.valid) invalid++;
        else if (item.sparseBlock === 0) original++;
        else compressed++;
      }
      const firstValid = block.entries.find(e => state.scenario.byId.get(e.id)?.valid)?.id ?? block.entries[0]?.id ?? null;
      return { total: block.entries.length, valid: original + compressed, original, compressed, invalid, firstValidId: firstValid };
    }

    function orderedGroups(rd) {
      const list = rd.groups.slice();
      if (state.sortBy === 'duration') {
        list.sort((a, b) => (b.aiv0.endUs - b.aic.startUs) - (a.aiv0.endUs - a.aic.startUs));
      } else {
        list.sort((a, b) => a.group - b.group);
      }
      return list;
    }

    function buildTopStripHtml(rd) {
      const s = rd.summary;
      return (
        '<div class="csa-rt-controls">' +
          '<div class="csa-rt-controls-left">' +
            '<div class="segmented-control segmented-control-muted csa-rt-seg" role="group" aria-label="核组排序">' +
              '<button type="button" class="btn btn-sm' + (state.sortBy === 'group' ? ' is-selected' : '') + '" data-rt-sort="group" aria-pressed="' + (state.sortBy === 'group') + '">Group 序</button>' +
              '<button type="button" class="btn btn-sm' + (state.sortBy === 'duration' ? ' is-selected' : '') + '" data-rt-sort="duration" aria-pressed="' + (state.sortBy === 'duration') + '">耗时降序</button>' +
            '</div>' +
            '<div class="segmented-control segmented-control-muted csa-rt-seg" role="group" aria-label="叠加信号">' +
              '<button type="button" class="btn btn-sm' + (state.showAiv1Idle ? ' is-selected' : '') + '" data-rt-toggle="aiv1" aria-pressed="' + state.showAiv1Idle + '">AIV 1 握手空转</button>' +
              '<button type="button" class="btn btn-sm' + (state.showMerge ? ' is-selected' : '') + '" data-rt-toggle="merge" aria-pressed="' + state.showMerge + '">下游 Merge Setup</button>' +
            '</div>' +
            '<span class="csa-rt-metrics-inline">' +
              '<b>' + s.aicMaxMinRatio.toFixed(2) + '×</b> 极差' +
              '<i>·</i>AIV 0 <b>+' + s.aiv0TailMeanUs.toFixed(0) + ' µs</b>' +
              (state.showMerge ? '<i>·</i>Merge Setup <b>' + s.mergeSetupTotalPct.toFixed(1) + '%</b>' : '') +
            '</span>' +
          '</div>' +
          '<div class="csa-rt-legend" aria-label="泳道图例">' +
            '<span><i class="is-aic"></i>AIC</span>' +
            '<span><i class="is-aiv0"></i>AIV 0</span>' +
            '<span><i class="is-aiv1' + (state.showAiv1Idle ? ' is-hatched' : '') + '"></i>AIV 1 (0 rows)</span>' +
            (state.showMerge ? '<span><i class="is-msetup"></i>Merge Setup</span><span><i class="is-mkern"></i>Merge 计算</span>' : '') +
          '</div>' +
        '</div>'
      );
    }

    function buildAxisHtml(rd, spanUs) {
      const step = spanUs > 950 ? 200 : 150;
      const ticks = [];
      for (let t = 0; t <= spanUs + 0.01; t += step) {
        const pct = (t / spanUs) * 100;
        ticks.push('<span style="left:' + pct.toFixed(2) + '%">' + t.toFixed(0) + ' µs</span>');
      }
      const qkEndPct = ((rd.qkEndUs - rd.windowOriginUs) / spanUs) * 100;
      return (
        '<div class="csa-rt-axis">' +
          '<div class="csa-rt-col-id">Group</div>' +
          '<div class="csa-rt-axis-track">' +
            ticks.join('') +
            '<span class="csa-rt-end-flag" style="left:' + qkEndPct.toFixed(2) + '%" title="qk_pv 最慢 AIV 0 完成时刻">' + (rd.qkEndUs - rd.windowOriginUs).toFixed(0) + ' µs 完成线</span>' +
          '</div>' +
          '<div class="csa-rt-col-stat">AIC 耗时</div>' +
        '</div>'
      );
    }

    function barStyle(startUs, endUs, originUs, spanUs) {
      const left = Math.max(0, ((startUs - originUs) / spanUs) * 100);
      const width = Math.max(0.35, ((endUs - startUs) / spanUs) * 100);
      return 'left:' + left.toFixed(3) + '%;width:' + width.toFixed(3) + '%';
    }

    function buildRowsHtml(rd, spanUs) {
      const origin = rd.windowOriginUs;
      const qkEndPct = ((rd.qkEndUs - origin) / spanUs) * 100;
      const groups = orderedGroups(rd);
      return groups.map(g => {
        const isSel = g.group === state.selectedGroup;
        const isSlow = g.tier === 'slow';
        const avgMergeSetupPct = Math.round(((g.merge0.setupUs + g.merge1.setupUs) / Math.max(0.01, g.merge0.durUs + g.merge1.durUs)) * 100);

        const aiv0TailStart = Math.max(g.aiv0.startUs, g.aic.endUs);
        const hasTail = g.aiv0.endUs > aiv0TailStart + 2;

        const mergeBar = m => {
          if (!state.showMerge) return '';
          const setupEnd = Math.min(m.endUs, m.startUs + m.setupUs);
          const setupPart = m.setupUs > 1
            ? '<span class="csa-rt-bar is-msetup" style="' + barStyle(m.startUs, setupEnd, origin, spanUs) + '"></span>'
            : '';
          const kernPart = '<span class="csa-rt-bar is-mkern" style="' + barStyle(setupEnd, m.endUs, origin, spanUs) + '"></span>';
          return setupPart + kernPart;
        };

        const rowLabel = 'Group ' + String(g.group).padStart(2, '0') + ' · AIC ' + g.aic.durUs.toFixed(1) + ' µs · AIV 0 ' + g.aiv0.durUs.toFixed(1) + ' µs · AIV 1 ' + g.aiv1.durUs.toFixed(1) + ' µs';

        return (
          '<button type="button" class="csa-rt-row' + (isSel ? ' is-selected' : '') + (isSlow ? ' is-slow' : ' is-fast') + '" data-rt-group="' + g.group + '" aria-pressed="' + isSel + '" aria-label="' + esc(rowLabel) + '">' +
            '<div class="csa-rt-col-id">' +
              '<span class="csa-rt-dot-tier ' + (isSlow ? 'is-slow' : 'is-fast') + '" aria-hidden="true"></span>' +
              '<strong>G' + String(g.group).padStart(2, '0') + '</strong>' +
              '<small>C' + g.aic.core + '</small>' +
            '</div>' +
            '<div class="csa-rt-track">' +
              '<div class="csa-rt-end-overlay" aria-hidden="true"><span class="csa-rt-end-line" style="left:' + qkEndPct.toFixed(2) + '%"></span></div>' +
              '<div class="csa-rt-sublane">' +
                '<span class="csa-rt-subtag">C</span>' +
                '<div class="csa-rt-lane-bg"><span class="csa-rt-bar is-aic" style="' + barStyle(g.aic.startUs, g.aic.endUs, origin, spanUs) + '"></span></div>' +
              '</div>' +
              '<div class="csa-rt-sublane">' +
                '<span class="csa-rt-subtag">V0</span>' +
                '<div class="csa-rt-lane-bg">' +
                  '<span class="csa-rt-bar is-aiv0" style="' + barStyle(g.aiv0.startUs, g.aiv0.endUs, origin, spanUs) + '"></span>' +
                  (hasTail ? '<span class="csa-rt-bar is-aiv0-tail" style="' + barStyle(aiv0TailStart, g.aiv0.endUs, origin, spanUs) + '"></span>' : '') +
                  mergeBar(g.merge0) +
                '</div>' +
              '</div>' +
              '<div class="csa-rt-sublane">' +
                '<span class="csa-rt-subtag is-muted">V1</span>' +
                '<div class="csa-rt-lane-bg">' +
                  '<span class="csa-rt-bar is-aiv1' + (state.showAiv1Idle ? ' is-hatched' : '') + '" style="' + barStyle(g.aiv1.startUs, g.aiv1.endUs, origin, spanUs) + '"></span>' +
                  mergeBar(g.merge1) +
                '</div>' +
              '</div>' +
            '</div>' +
            '<div class="csa-rt-col-stat">' +
              '<span class="csa-rt-stat-main">' + g.aic.durUs.toFixed(0) + ' µs</span>' +
              (state.showMerge ? '<span class="csa-rt-stat-merge' + (avgMergeSetupPct >= 50 ? ' is-warn' : '') + '">' + avgMergeSetupPct + '%</span>' : '<span class="csa-rt-stat-sub">+' + g.aiv0TailUs.toFixed(0) + '</span>') +
            '</div>' +
          '</button>'
        );
      }).join('');
    }

    function buildInspectorHtml(rd) {
      if (state.selectedGroup == null) return '';
      const g = rd.groups.find(x => x.group === state.selectedGroup);
      if (!g) return '';
      const ls = laneStats(g.group);
      const isSlow = g.tier === 'slow';
      const m0Pct = g.merge0.durUs > 0 ? Math.round((g.merge0.setupUs / g.merge0.durUs) * 100) : 0;
      const m1Pct = g.merge1.durUs > 0 ? Math.round((g.merge1.setupUs / g.merge1.durUs) * 100) : 0;

      return (
        '<aside class="csa-rt-inspector" aria-label="Core Group 实测详情">' +
          '<header class="csa-rt-insp-head">' +
            '<div>' +
              '<span class="csa-rt-insp-kicker">RANK ' + rd.rank + ' · ' + g.aic.startUs.toFixed(0) + '→' + g.aiv0.endUs.toFixed(0) + ' µs</span>' +
              '<h3>Group ' + String(g.group).padStart(2, '0') + ' <span class="csa-rt-tier ' + (isSlow ? 'is-slow' : 'is-fast') + '">' + (isSlow ? '慢组' : '快组') + '</span></h3>' +
            '</div>' +
            '<button type="button" class="btn btn-ghost btn-sm csa-rt-close" data-rt-close aria-label="关闭详情">×</button>' +
          '</header>' +
          '<div class="csa-rt-insp-body">' +
            '<section class="csa-rt-insp-sec">' +
              '<h4>核组实测（1 AIC + 2 AIV）</h4>' +
              '<table class="csa-rt-mini-table">' +
                '<thead><tr><th>核</th><th>时长</th><th>特征</th></tr></thead>' +
                '<tbody>' +
                  '<tr><td><i class="csa-rt-dot is-aic"></i>AIC · C' + g.aic.core + '</td><td>' + fmtUs(g.aic.durUs) + '</td><td>QK / PV</td></tr>' +
                  '<tr><td><i class="csa-rt-dot is-aiv0"></i>AIV 0 · V' + g.aiv0.core + '</td><td>' + fmtUs(g.aiv0.durUs) + '</td><td class="is-accent">' + fmtSigned(g.aiv0TailUs) + '</td></tr>' +
                  '<tr><td><i class="csa-rt-dot is-aiv1"></i>AIV 1 · V' + g.aiv1.core + '</td><td>' + fmtUs(g.aiv1.durUs) + '</td><td>0 rows 握手</td></tr>' +
                '</tbody>' +
              '</table>' +
            '</section>' +
            '<section class="csa-rt-insp-sec">' +
              '<h4>下游 Merge Setup</h4>' +
              '<dl class="csa-rt-dl">' +
                '<dt>V' + g.aiv0.core + ' / V' + g.aiv1.core + '</dt><dd><b>' + m0Pct + '%</b> / <b>' + m1Pct + '%</b>（' + fmtUs(g.merge0.setupUs) + ' 等待）</dd>' +
              '</dl>' +
              '<p class="csa-rt-note">' + (isSlow
                ? '尾组完成即释放依赖，下游 Merge 几乎零等待。'
                : '早于尾线 ' + Math.max(0, rd.qkEndUs - g.aiv0.endUs).toFixed(0) + ' µs，下游 Merge 提前占核等待。') + '</p>' +
            '</section>' +
            (ls ? (
              '<section class="csa-rt-insp-sec">' +
                '<h4>Lane ' + String(g.group).padStart(2, '0') + ' 迭代构成</h4>' +
                '<dl class="csa-rt-dl">' +
                  '<dt>Work Items</dt><dd>' + ls.total + ' 轮（' + ls.valid + 'V / ' + ls.invalid + 'I）</dd>' +
                  '<dt>KV 路径</dt><dd>' + ls.original + ' Orig · ' + ls.compressed + ' Comp</dd>' +
                '</dl>' +
              '</section>'
            ) : '') +
            '<div class="csa-rt-actions">' +
              '<button type="button" class="btn btn-sm csa-rt-cta-primary" data-rt-open-instance="' + g.group + '">进入单 Work Item 因果泳道 →</button>' +
              '<button type="button" class="btn btn-ghost btn-sm" data-rt-open-lane="' + g.group + '">查看 Lane ' + String(g.group).padStart(2, '0') + ' 迭代矩阵 →</button>' +
              '<button type="button" class="btn btn-ghost btn-sm" data-rt-source="decode_sparse_attn_csa.py:204:313">decode_sparse_attn_csa.py : L204–313 ↗</button>' +
            '</div>' +
          '</div>' +
        '</aside>'
      );
    }

    function render() {
      const rd = rankData();
      if (!rd) {
        host.innerHTML = '<div class="csa-rt-empty">未找到 qk_pv 实测泳道数据。</div>';
        return;
      }
      const spanUs = state.showMerge ? rd.totalSpanUs : Math.max(rd.qkSpanUs * 1.04, 1);
      const hasInspector = state.selectedGroup != null;

      host.innerHTML =
        '<div class="csa-rt-shell' + (hasInspector ? ' has-inspector' : '') + '">' +
          '<div class="csa-rt-main">' +
            buildTopStripHtml(rd) +
            '<div class="csa-rt-board">' +
              buildAxisHtml(rd, spanUs) +
              '<div class="csa-rt-rows">' + buildRowsHtml(rd, spanUs) + '</div>' +
            '</div>' +
          '</div>' +
          buildInspectorHtml(rd) +
        '</div>';
    }

    host.addEventListener('click', event => {
      const btn = event.target.closest('button');
      if (!btn) return;
      if (btn.hasAttribute('data-rt-close')) {
        state.selectedGroup = null;
        render();
        return;
      }
      if (btn.dataset.rtSort) {
        state.sortBy = btn.dataset.rtSort;
        render();
        return;
      }
      if (btn.dataset.rtToggle) {
        const key = btn.dataset.rtToggle;
        if (key === 'aiv1') state.showAiv1Idle = !state.showAiv1Idle;
        else if (key === 'merge') state.showMerge = !state.showMerge;
        render();
        return;
      }
      if (btn.dataset.rtGroup != null) {
        const clicked = Number(btn.dataset.rtGroup);
        state.selectedGroup = state.selectedGroup === clicked ? null : clicked;
        if (state.selectedGroup != null) opts.onSelectGroup?.(state.selectedGroup);
        render();
        return;
      }
      if (btn.dataset.rtOpenLane != null) {
        opts.onOpenLane?.(Number(btn.dataset.rtOpenLane));
        return;
      }
      if (btn.dataset.rtOpenInstance != null) {
        const g = Number(btn.dataset.rtOpenInstance);
        const ls = laneStats(g);
        opts.onOpenInstance?.(g, ls?.firstValidId ?? null);
        return;
      }
      if (btn.dataset.rtSource) {
        const [file, a, z] = btn.dataset.rtSource.split(':');
        opts.onSource?.(file, Number(a), Number(z));
      }
    });

    function update(next) {
      if (!next) return;
      const rankChanged = next.rank != null && next.rank !== state.rank;
      if (next.rank != null) state.rank = next.rank;
      if (next.scenario) state.scenario = next.scenario;
      if (next.selectedGroup !== undefined) {
        state.selectedGroup = next.selectedGroup != null && next.selectedGroup >= 0 && next.selectedGroup <= 23 ? next.selectedGroup : null;
      } else if (rankChanged) {
        state.selectedGroup = null;
      }
      render();
    }

    return { update };
  }

  global.PtoCsaQkRuntimeTrace = Object.freeze({ mount });
})(window);
