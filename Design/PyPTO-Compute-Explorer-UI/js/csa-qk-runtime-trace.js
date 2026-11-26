/* qk_plan -> qk_pv -> csa_merge_pack_publish 72-core (24 Core Groups) Runtime Trace view.
 * Backed by unmodified per-core DFX measurements from _jit_l3_decode_csa_20260903_010617
 * and unified 3-act compute-semantics metadata (valid_block_mask / qk_order).
 */
(function (global) {
  'use strict';

  const esc = v => String(v ?? '').replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
  const fmtUs = v => (Number.isFinite(v) ? v.toFixed(1) + ' µs' : '—');
  const fmtShortUs = v => (Number.isFinite(v) ? Math.round(v) + ' µs' : '—');
  const fmtSigned = v => (Number.isFinite(v) ? (v >= 0 ? '+' : '−') + Math.abs(v).toFixed(1) + ' µs' : '—');

  function mount(host, options) {
    const opts = options || {};
    let state = {
      rank: 0,
      scenario: null,
      selectedGroup: null, // null = no group selected; click any group to open Group inspector
      selectedAct: null, // null | 'act1-qk-plan-barrier' | 'act2-qk-pv-parity' | 'act3-invalid-zero-flow'
      sortBy: 'group', // 'group' | 'duration'
      showPlan: true,
      showAiv1Idle: false, // secondary compiler white-box signal (off by default to focus on 3-act chain)
      showMerge: true,
      highlightStep: null, // null | 'csa_slots_build_valid_qk_plan' | 'qk_pv' | 'csa_merge_pack_publish'
      showOptPanel: false,
      optEnabled: {
        'act1-qk-plan-barrier': true,
        'act2-qk-pv-parity': true,
        'act3-invalid-zero-flow': true,
      },
    };

    function rootData() {
      return global.PTO_CSA_QK_RUNTIME_TRACE || null;
    }

    function rankData() {
      const root = rootData();
      if (!root || !root.ranks) return null;
      return root.ranks[String(state.rank)] || root.ranks['0'] || null;
    }

    function storyActs() {
      return rootData()?.storyChain?.acts || [];
    }

    function currentAct() {
      if (!state.selectedAct) return null;
      return storyActs().find(a => a.id === state.selectedAct) || null;
    }

    function windowBounds(rd) {
      const plan = rd.chain?.qkPlan || null;
      const originUs = state.showPlan && plan ? plan.startUs : rd.windowOriginUs;
      const endUs = state.showMerge ? rd.mergeEndUs : rd.qkEndUs;
      const rawSpanUs = Math.max(1, endUs - originUs);
      return { originUs, endUs, spanUs: rawSpanUs * 1.015, rawSpanUs, plan };
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

    function barStyle(startUs, endUs, originUs, spanUs) {
      const left = Math.max(0, ((startUs - originUs) / spanUs) * 100);
      const width = Math.max(0.35, ((endUs - startUs) / spanUs) * 100);
      return 'left:' + left.toFixed(3) + '%;width:' + width.toFixed(3) + '%';
    }

    function actMetricTag(act, rd) {
      const c = rd.chain;
      if (!c) return act.tag;
      if (act.id === 'act1-qk-plan-barrier') {
        return c.qkPlan.durUs.toFixed(0) + ' µs · 1/72 核';
      }
      if (act.id === 'act2-qk-pv-parity') {
        return rd.summary.aicMaxMinRatio.toFixed(2) + '× · 12 慢 / 12 快';
      }
      if (act.id === 'act3-invalid-zero-flow') {
        return '+' + c.breakdown.qkAiv0TailUs.toFixed(0) + ' µs 写零 · ' + c.workProfile.mergeLoopInvalidPct.toFixed(0) + '% 空算';
      }
      return act.tag;
    }

    function buildStoryStripHtml(rd) {
      const acts = storyActs();
      const c = rd.chain;
      if (!acts.length || !c) return '';
      const proj = c.projectedAfterFix;
      const pills = acts.map(act => {
        const isSel = state.selectedAct === act.id;
        return (
          '<button type="button" class="csa-rt-act-pill' + (isSel ? ' is-active' : '') + '" data-rt-act="' + esc(act.id) + '" aria-pressed="' + isSel + '" title="' + esc(act.title) + '">' +
            '<span class="csa-rt-act-num">' + act.order + '</span>' +
            '<span class="csa-rt-act-copy">' +
              '<strong>' + esc(act.stage) + ' · ' + esc(act.tag) + '</strong>' +
              '<small>' + esc(actMetricTag(act, rd)) + '</small>' +
            '</span>' +
          '</button>'
        );
      }).join('');

      return (
        '<div class="csa-rt-story-strip" role="region" aria-label="valid_block_mask / qk_order 三幕运行诊断">' +
          '<div class="csa-rt-story-head">' +
            '<span class="csa-rt-story-kicker">三步关键链路实测</span>' +
            '<span class="csa-rt-story-kpi">' +
              '<b>' + c.chainSpanUs.toFixed(0) + ' µs</b>' +
              '（占整层 CSA ' + c.chainSharePct.toFixed(1) + '%）' +
              '<i>·</i>三处源码直修预计省 <b>~' + Math.round(proj.totalSavingUs) + ' µs</b>（−' + proj.chainSavingPct.toFixed(0) + '%）' +
            '</span>' +
            '<button type="button" class="btn btn-sm csa-rt-opt-trigger' + (state.showOptPanel ? ' is-active' : '') + '" data-rt-opt-open aria-pressed="' + state.showOptPanel + '">优化对比 & 源码 Diff (3 处) →</button>' +
          '</div>' +
          '<div class="csa-rt-act-row">' + pills + '</div>' +
        '</div>'
      );
    }

    function buildTopStripHtml(rd) {
      const s = rd.summary;
      const c = rd.chain;
      return (
        '<div class="csa-rt-controls">' +
          '<div class="csa-rt-controls-left">' +
            '<div class="segmented-control segmented-control-muted csa-rt-seg" role="group" aria-label="核组排序">' +
              '<button type="button" class="btn btn-sm' + (state.sortBy === 'group' ? ' is-selected' : '') + '" data-rt-sort="group" aria-pressed="' + (state.sortBy === 'group') + '">Group 序</button>' +
              '<button type="button" class="btn btn-sm' + (state.sortBy === 'duration' ? ' is-selected' : '') + '" data-rt-sort="duration" aria-pressed="' + (state.sortBy === 'duration') + '">耗时降序</button>' +
            '</div>' +
            '<div class="segmented-control segmented-control-muted csa-rt-seg" role="group" aria-label="链路阶段与叠加">' +
              (c?.qkPlan ? '<button type="button" class="btn btn-sm' + (state.showPlan ? ' is-selected' : '') + '" data-rt-toggle="plan" aria-pressed="' + state.showPlan + '">入口 qk_plan</button>' : '') +
              '<button type="button" class="btn btn-sm' + (state.showMerge ? ' is-selected' : '') + '" data-rt-toggle="merge" aria-pressed="' + state.showMerge + '">下游 Merge</button>' +
              '<button type="button" class="btn btn-sm' + (state.showAiv1Idle ? ' is-selected' : '') + '" data-rt-toggle="aiv1" aria-pressed="' + state.showAiv1Idle + '" title="编译白盒次要信号：SplitVectorKernel 为 AIV 1 生成 valid_shape=[0,0] 握手空壳">AIV 1 握手</button>' +
            '</div>' +
            '<span class="csa-rt-metrics-inline">' +
              '<b>' + s.aicMaxMinRatio.toFixed(2) + '×</b> 极差' +
              '<i>·</i>写零尾 <b>+' + (c ? c.breakdown.qkAiv0TailUs.toFixed(0) : s.aiv0TailMeanUs.toFixed(0)) + ' µs</b>' +
              (state.showMerge ? '<i>·</i>Merge 等待 <b>' + s.mergeSetupTotalPct.toFixed(1) + '%</b>' : '') +
            '</span>' +
          '</div>' +
          '<div class="csa-rt-legend" aria-label="泳道图例">' +
            (state.showPlan && c?.qkPlan ? '<span><i class="is-plan"></i>qk_plan (1核)</span>' : '') +
            '<span><i class="is-aic"></i>qk_pv 计算 (C / V0)</span>' +
            '<span><i class="is-aiv0-tail"></i>无效写零拖尾</span>' +
            (state.showAiv1Idle ? '<span><i class="is-aiv1 is-hatched"></i>AIV 1 空握手</span>' : '') +
            (state.showMerge ? '<span><i class="is-msetup"></i>Merge 等待 (Setup)</span><span><i class="is-mkern"></i>Merge 计算</span>' : '') +
          '</div>' +
        '</div>'
      );
    }

    function buildStepsHtml(rd, bounds) {
      const origin = bounds.originUs;
      const plan = bounds.plan;
      const qkStart = Math.min(...rd.groups.map(g => Math.min(g.aic.startUs, g.aiv0.startUs, g.aiv1.startUs)));
      const qkEnd = rd.qkEndUs;
      const mergeStart = Math.min(...rd.groups.map(g => Math.min(g.merge0.startUs, g.merge1.startUs)));
      const mergeEnd = rd.mergeEndUs;
      const delta = mergeStart - qkEnd;

      const planAct = state.highlightStep === 'csa_slots_build_valid_qk_plan';
      const planDim = state.highlightStep && !planAct;
      const qkAct = state.highlightStep === 'qk_pv';
      const qkDim = state.highlightStep && !qkAct;
      const mAct = state.highlightStep === 'csa_merge_pack_publish';
      const mDim = state.highlightStep && !mAct;

      let html = '<div class="csa-rt-steps" aria-label="三步链路与交接">';

      if (state.showPlan && plan) {
        const gapPlan = Math.max(0, qkStart - plan.endUs);
        html +=
          '<button type="button" class="csa-rt-step-chip' + (planAct ? ' is-active' : '') + (planDim ? ' is-dimmed' : '') + '" data-rt-step="csa_slots_build_valid_qk_plan" aria-pressed="' + planAct + '" title="点击仅高亮 qk_plan（1/72 核串行 · 跨度 ' + plan.durUs.toFixed(1) + ' µs）">' +
            '<i style="background:#fd6e5d"></i><span>qk_plan</span>' +
            '<small>1 AIV · 0–' + Math.round(plan.endUs - origin) + 'µs</small>' +
          '</button>' +
          '<span class="csa-rt-step-link is-gap" title="qk_plan 完成后释放 qk_order 依赖，间隔 +' + gapPlan.toFixed(1) + ' µs">→ +' + Math.round(gapPlan) + 'µs</span>';
      }

      html +=
        '<button type="button" class="csa-rt-step-chip' + (qkAct ? ' is-active' : '') + (qkDim ? ' is-dimmed' : '') + '" data-rt-step="qk_pv" aria-pressed="' + qkAct + '" title="点击仅高亮 qk_pv（MIX 1:2 · 24 核组 · 12 慢 vs 12 快 · 跨度 ' + rd.qkSpanUs.toFixed(1) + ' µs）">' +
          '<i style="background:#80b7f7"></i><span>qk_pv</span>' +
          '<small>24 核组 · ' + Math.round(qkStart - origin) + '–' + Math.round(qkEnd - origin) + 'µs</small>' +
        '</button>';

      if (state.showMerge) {
        const linkHtml = delta < -1
          ? '<span class="csa-rt-step-link is-overlap" title="12 个快组完成后提前占核启动 Merge，与 12 个慢组重叠 ' + Math.abs(delta).toFixed(1) + ' µs，导致快组 Merge 73% Setup 空等">⇢ 快组抢跑 ' + Math.abs(delta).toFixed(0) + 'µs</span>'
          : '<span class="csa-rt-step-link">→</span>';
        html += linkHtml +
          '<button type="button" class="csa-rt-step-chip' + (mAct ? ' is-active' : '') + (mDim ? ' is-dimmed' : '') + '" data-rt-step="csa_merge_pack_publish" aria-pressed="' + mAct + '" title="点击仅高亮 csa_merge_pack_publish（48 AIV · 66–68% 压缩块循环为无效零块）">' +
            '<i style="background:#4fb899"></i><span>csa_merge_pack_publish</span>' +
            '<small>48 AIV · ' + Math.round(mergeStart - origin) + '–' + Math.round(mergeEnd - origin) + 'µs</small>' +
          '</button>';
      }
      html += '</div>';
      return html;
    }

    function buildAxisHtml(rd, bounds) {
      const { originUs, spanUs, plan } = bounds;
      const step = spanUs > 1100 ? 250 : spanUs > 850 ? 200 : 150;
      const planEndPct = state.showPlan && plan ? ((plan.endUs - originUs) / spanUs) * 100 : null;
      const qkEndPct = ((rd.qkEndUs - originUs) / spanUs) * 100;
      const mergeEndPct = state.showMerge ? ((rd.mergeEndUs - originUs) / spanUs) * 100 : null;
      const ticks = [];
      for (let t = 0; t <= bounds.rawSpanUs + 0.01; t += step) {
        const pct = (t / spanUs) * 100;
        const hitsPlan = planEndPct != null && pct > planEndPct - 4 && pct < planEndPct + 14;
        const hitsQk = Math.abs(pct - qkEndPct) < 8;
        const hitsMerge = mergeEndPct != null && pct > mergeEndPct - 8;
        if (t > 0 && (hitsPlan || hitsQk || hitsMerge)) continue;
        ticks.push('<span style="left:' + pct.toFixed(2) + '%">' + t.toFixed(0) + ' µs</span>');
      }
      const planDurLabel = plan ? Math.round(plan.endUs - originUs) : 0;
      return (
        '<div class="csa-rt-axis">' +
          '<div class="csa-rt-col-id">Group</div>' +
          '<div class="csa-rt-axis-track">' +
            ticks.join('') +
            (planEndPct != null ? '<span class="csa-rt-plan-flag" style="left:' + planEndPct.toFixed(2) + '%" title="qk_plan 单核串行屏障：仅 1 颗 AIV 运行，其余 71/72 核空转至 ' + planDurLabel + ' µs">单核串行 ' + planDurLabel + ' µs</span>' : '') +
            '<span class="csa-rt-end-flag" style="left:' + qkEndPct.toFixed(2) + '%" title="qk_pv 最慢 AIV 0（含写零尾）完成时刻，下游 Merge 在此释放等待">qk_pv 结束 ' + Math.round(rd.qkEndUs - originUs) + ' µs</span>' +
            (mergeEndPct != null ? '<span class="csa-rt-merge-flag" style="left:' + mergeEndPct.toFixed(2) + '%" title="整条三步链路完成时刻">' + Math.round(rd.mergeEndUs - originUs) + ' µs</span>' : '') +
          '</div>' +
          '<div class="csa-rt-col-stat" title="实测 AIC 耗时与有效块构成 V = S(SB0滑窗) + C(CMP压缩)">AIC · V(S+C)</div>' +
        '</div>'
      );
    }

    function buildRowsHtml(rd, bounds) {
      const { originUs, spanUs, plan } = bounds;
      const planEndPct = state.showPlan && plan ? ((plan.endUs - originUs) / spanUs) * 100 : null;
      const qkEndPct = ((rd.qkEndUs - originUs) / spanUs) * 100;
      const groups = orderedGroups(rd);

      const act = state.selectedAct;
      const planDim = (state.highlightStep && state.highlightStep !== 'csa_slots_build_valid_qk_plan') || (act && act !== 'act1-qk-plan-barrier') ? ' is-dimmed' : '';
      const qkDim = (state.highlightStep && state.highlightStep !== 'qk_pv') || (act === 'act1-qk-plan-barrier') ? ' is-dimmed' : '';
      const aicDim = qkDim || (act === 'act3-invalid-zero-flow' ? ' is-dimmed' : '');
      const tailEmph = act === 'act3-invalid-zero-flow' ? ' is-emph' : '';
      const mergeDim = (state.highlightStep && state.highlightStep !== 'csa_merge_pack_publish') || (act === 'act1-qk-plan-barrier') ? ' is-dimmed' : '';

      return groups.map(g => {
        const isSel = g.group === state.selectedGroup;
        const isSlow = g.tier === 'slow';
        const isPlanOwner = state.showPlan && plan && g.group === plan.group;
        const isPlanV0 = isPlanOwner && plan.core === g.aiv0.core;
        const isPlanV1 = isPlanOwner && plan.core === g.aiv1.core;
        const wm = g.workMix || null;

        const aiv0TailStart = Math.max(g.aiv0.startUs, g.aic.endUs);
        const hasTail = g.aiv0.endUs > aiv0TailStart + 2;
        const tailTip = 'AIV 0 队尾无效块写零：+' + g.aiv0TailUs.toFixed(1) + ' µs' + (wm ? '（' + wm.invalid + ' 个无效块 · ' + wm.invalidZeroMiB.toFixed(2) + ' MiB 写零）' : '');

        const mergeBar = m => {
          if (!state.showMerge) return '';
          const setupEnd = Math.min(m.endUs, m.startUs + m.setupUs);
          const setupPart = m.setupUs > 1
            ? '<span class="csa-rt-bar is-msetup' + mergeDim + '" style="' + barStyle(m.startUs, setupEnd, originUs, spanUs) + '" title="Merge 等待 (Setup)：' + m.setupUs.toFixed(1) + ' µs（等待 qk_pv 慢组释放依赖）"></span>'
            : '';
          const kernPart = '<span class="csa-rt-bar is-mkern' + mergeDim + (act === 'act3-invalid-zero-flow' ? ' is-emph' : '') + '" style="' + barStyle(setupEnd, m.endUs, originUs, spanUs) + '" title="Merge 计算：' + m.kernelUs.toFixed(1) + ' µs（其中 66–68% 压缩块循环为无效零块）"></span>';
          return setupPart + kernPart;
        };

        const planBarHtml = isOwnerSub => isOwnerSub
          ? '<span class="csa-rt-bar is-plan' + planDim + (act === 'act1-qk-plan-barrier' ? ' is-emph' : '') + '" style="' + barStyle(plan.startUs, plan.endUs, originUs, spanUs) + '" title="csa_slots_build_valid_qk_plan · ' + plan.coreName + ' · ' + plan.durUs.toFixed(1) + ' µs（单核串行屏障，其余 71/72 核空转）"></span>'
          : '';

        const mixText = wm
          ? '<span class="csa-rt-stat-mix ' + (isSlow ? 'is-slow' : 'is-fast') + '" title="有效项总数 ' + wm.valid + ' V = ' + wm.sb0 + ' 个滑窗 SB0 (~8 µs) + ' + wm.cmp + ' 个压缩 CMP (~39 µs)；另含 ' + wm.invalid + ' 个无效块">' + wm.valid + 'V (' + wm.sb0 + 'S+' + wm.cmp + 'C)</span>'
          : '<span class="csa-rt-stat-sub">+' + g.aiv0TailUs.toFixed(0) + '</span>';

        const rowLabel = 'Group ' + String(g.group).padStart(2, '0') + ' · AIC ' + g.aic.durUs.toFixed(1) + ' µs · AIV 0 ' + g.aiv0.durUs.toFixed(1) + ' µs（写零尾 +' + g.aiv0TailUs.toFixed(1) + ' µs）' + (wm ? ' · ' + wm.sb0 + ' SB0 / ' + wm.cmp + ' CMP / ' + wm.invalid + ' Invalid' : '');

        return (
          '<button type="button" class="csa-rt-row' + (isSel ? ' is-selected' : '') + (isSlow ? ' is-slow' : ' is-fast') + (isPlanOwner && act === 'act1-qk-plan-barrier' ? ' is-plan-focus' : '') + '" data-rt-group="' + g.group + '" aria-pressed="' + isSel + '" aria-label="' + esc(rowLabel) + '">' +
            '<div class="csa-rt-col-id">' +
              '<span class="csa-rt-dot-tier ' + (isSlow ? 'is-slow' : 'is-fast') + '" aria-hidden="true"></span>' +
              '<strong>G' + String(g.group).padStart(2, '0') + '</strong>' +
              '<small>' + (isPlanOwner ? 'Plan·C' + g.aic.core : 'C' + g.aic.core) + '</small>' +
            '</div>' +
            '<div class="csa-rt-track">' +
              '<div class="csa-rt-end-overlay" aria-hidden="true">' +
                (planEndPct != null ? '<span class="csa-rt-plan-line" style="left:' + planEndPct.toFixed(2) + '%"></span>' : '') +
                '<span class="csa-rt-end-line" style="left:' + qkEndPct.toFixed(2) + '%"></span>' +
              '</div>' +
              '<div class="csa-rt-sublane">' +
                '<span class="csa-rt-subtag">C</span>' +
                '<div class="csa-rt-lane-bg"><span class="csa-rt-bar is-aic' + aicDim + '" style="' + barStyle(g.aic.startUs, g.aic.endUs, originUs, spanUs) + '" title="AIC · C' + g.aic.core + ' · ' + g.aic.durUs.toFixed(1) + ' µs"></span></div>' +
              '</div>' +
              '<div class="csa-rt-sublane">' +
                '<span class="csa-rt-subtag">V0</span>' +
                '<div class="csa-rt-lane-bg">' +
                  planBarHtml(isPlanV0) +
                  '<span class="csa-rt-bar is-aiv0' + aicDim + '" style="' + barStyle(g.aiv0.startUs, Math.min(g.aiv0.endUs, aiv0TailStart), originUs, spanUs) + '" title="AIV 0 · V' + g.aiv0.core + ' · 有效块 Softmax"></span>' +
                  (hasTail ? '<span class="csa-rt-bar is-aiv0-tail' + qkDim + tailEmph + '" style="' + barStyle(aiv0TailStart, g.aiv0.endUs, originUs, spanUs) + '" title="' + esc(tailTip) + '"></span>' : '') +
                  mergeBar(g.merge0) +
                '</div>' +
              '</div>' +
              '<div class="csa-rt-sublane">' +
                '<span class="csa-rt-subtag is-muted">V1</span>' +
                '<div class="csa-rt-lane-bg">' +
                  planBarHtml(isPlanV1) +
                  '<span class="csa-rt-bar is-aiv1' + (state.showAiv1Idle ? ' is-hatched' : '') + aicDim + '" style="' + barStyle(g.aiv1.startUs, g.aiv1.endUs, originUs, spanUs) + '" title="AIV 1 · V' + g.aiv1.core + ' · valid_shape=[0,0] 管道握手"></span>' +
                  mergeBar(g.merge1) +
                '</div>' +
              '</div>' +
            '</div>' +
            '<div class="csa-rt-col-stat">' +
              '<span class="csa-rt-stat-main">' + g.aic.durUs.toFixed(0) + ' µs</span>' +
              mixText +
            '</div>' +
          '</button>'
        );
      }).join('');
    }

    function buildActInspectorHtml(rd, act) {
      const c = rd.chain;
      const wp = c.workProfile;
      const proj = c.projectedAfterFix;
      let statsHtml = '';
      let assignActionLabel = '在 Assignment 中验证静态/运行落差 →';
      let graphTarget = 'sparse/attention/qk-pv';
      let graphActionLabel = '在 Graph 中追溯根因路径 →';
      let assignLane = null;

      if (act.id === 'act1-qk-plan-barrier') {
        statsHtml =
          '<dl class="csa-rt-dl">' +
            '<dt>实测耗时</dt><dd><b>' + fmtUs(c.qkPlan.durUs) + '</b>（' + c.qkPlan.coreName + ' / G' + String(c.qkPlan.group).padStart(2, '0') + '）</dd>' +
            '<dt>芯片并行度</dt><dd><b>1 / 72 核</b>（71 核全空转）</dd>' +
            '<dt>可并行块数</dt><dd><b>' + c.qkPlan.tokenTiles + ' 个</b> 8-token 向量块</dd>' +
            '<dt>预计优化后</dt><dd>~' + proj.qkPlanUs.toFixed(0) + ' µs（<b>省 ~' + proj.qkPlanSavingUs.toFixed(0) + ' µs</b>，模型推算）</dd>' +
          '</dl>';
        assignActionLabel = '在 Assignment 中看 qk_plan 队列构建 →';
        graphTarget = 'sparse/attention/plan';
        graphActionLabel = '在 Graph 中看 qk_plan 依赖边界 →';
      } else if (act.id === 'act2-qk-pv-parity') {
        const slowSample = rd.groups.find(g => g.tier === 'slow');
        assignLane = slowSample?.group ?? 0;
        statsHtml =
          '<dl class="csa-rt-dl">' +
            '<dt>实测双峰</dt><dd>12 慢 <b>' + rd.summary.slowAicMeanUs.toFixed(0) + ' µs</b> vs 12 快 <b>' + rd.summary.fastAicMeanUs.toFixed(0) + ' µs</b></dd>' +
            '<dt>块构成反差</dt><dd>慢组 <b>' + wp.slowLaneAvg.cmp + ' CMP / ' + wp.slowLaneAvg.sb0 + ' SB0</b> · 快组 <b>' + wp.fastLaneAvg.cmp + ' CMP / ' + wp.fastLaneAvg.sb0 + ' SB0</b></dd>' +
            '<dt>单块拟合代价</dt><dd>SB0 <b>' + wp.fittedCostUs.sb0Us + ' µs</b> vs CMP <b>' + wp.fittedCostUs.cmpUs + ' µs</b>（<b>' + wp.fittedCostUs.costRatio + '×</b>）</dd>' +
            '<dt>预计优化后</dt><dd>~' + proj.qkAicUs.toFixed(0) + ' µs（<b>省 ~' + proj.qkAicSavingUs.toFixed(0) + ' µs</b>，bit-exact）</dd>' +
          '</dl>';
        assignActionLabel = '在 Assignment 中对照 24-Lane 构成 →';
        graphTarget = 'sparse/attention/qk-pv';
        graphActionLabel = '在 Graph 中看 SB0 vs CMP 访存分支 →';
      } else if (act.id === 'act3-invalid-zero-flow') {
        statsHtml =
          '<dl class="csa-rt-dl">' +
            '<dt>无效空块占比</dt><dd><b>' + wp.invalidItems + ' / ' + wp.totalItems + '</b>（<b>' + wp.invalidPct + '%</b>）</dd>' +
            '<dt>AIV 0 写零尾</dt><dd>慢组 <b>+' + c.breakdown.qkAiv0TailUs.toFixed(1) + ' µs</b> · 快组均值 <b>+' + rd.summary.fastAiv0TailMeanUs.toFixed(1) + ' µs</b></dd>' +
            '<dt>无效 HBM 搬运</dt><dd>写 <b>' + wp.invalidZeroMiB + ' MiB</b> + 读 <b>' + wp.invalidZeroMiB + ' MiB</b> = <b>' + wp.invalidTotalHbmMiB + ' MiB</b></dd>' +
            '<dt>Merge 空算占比</dt><dd><b>' + wp.mergeLoopInvalid + ' / ' + wp.mergeLoopTotal + '</b>（<b>' + wp.mergeLoopInvalidPct + '%</b> 压缩块循环）</dd>' +
            '<dt>预计双端节省</dt><dd><b>~' + Math.round(proj.qkAiv0TailSavingUs + proj.mergeSavingUs) + ' µs</b>（模型推算）</dd>' +
          '</dl>';
        assignActionLabel = '在 Assignment 中查看队尾 54% 无效块 →';
        graphTarget = 'redistribute/publish';
        graphActionLabel = '在 Graph 中追踪缺失的掩码边 →';
      }

      const sourceBtns = (act.sourceRefs || []).map(ref =>
        '<button type="button" class="btn btn-ghost btn-sm" data-rt-source="' + esc(ref.file + ':' + ref.start + ':' + ref.end) + '">' + esc(ref.file + ' · L' + ref.start + '–' + ref.end) + ' ↗</button>'
      ).join('');

      const isGraphRoot = act.graphRole !== 'cost-context';
      const loopLabel = act.primaryLoop || (isGraphRoot ? 'Pipeline → Assignment → Graph → Source' : 'Pipeline → Assignment → Source');
      const traceSectionHtml = isGraphRoot
        ? '<section class="csa-rt-insp-sec">' +
            '<h4>语义溯源（' + esc(loopLabel) + '）</h4>' +
            '<div class="csa-rt-trace-chain">' +
              '<div><b>Pipeline</b><p>' + esc(act.pipelineSignal) + '</p></div>' +
              '<div><b>Assignment</b><p>' + esc(act.assignmentGap) + '</p></div>' +
              '<div><b>Graph</b><p>' + esc(act.graphCause) + '</p></div>' +
            '</div>' +
          '</section>'
        : '<section class="csa-rt-insp-sec">' +
            '<h4>主定位闭环（' + esc(loopLabel) + '）</h4>' +
            '<div class="csa-rt-trace-chain">' +
              '<div><b>Pipeline</b><p>' + esc(act.pipelineSignal) + '</p></div>' +
              '<div><b>Assignment</b><p>' + esc(act.assignmentGap) + '</p></div>' +
            '</div>' +
            (act.costContext
              ? '<details class="csa-rt-opt-context">' +
                  '<summary>辅助语义背景：为何 CMP 比 SB0 贵 ' + wp.fittedCostUs.costRatio + '×？（非计算图拓扑问题）</summary>' +
                  '<p>' + esc(act.costContext) + '</p>' +
                  '<button type="button" class="btn btn-ghost btn-sm" data-rt-open-graph="sparse/attention/qk-pv/selected" data-rt-act-ctx="' + esc(act.id) + '">对照查看 qk_pv 内 SB0 / CMP 访存分支（Graph）→</button>' +
                '</details>'
              : '') +
          '</section>';

      const diff = act.sourceDiff;
      const diffHtml = diff
        ? '<section class="csa-rt-insp-sec csa-rt-diff-sec">' +
            '<h4>源码直修方案（' + esc(diff.lineRange) + '）</h4>' +
            '<p class="csa-rt-note">' + esc(diff.summary) + '</p>' +
            '<pre class="csa-rt-code-snippet"><code>' + esc(diff.after) + '</code></pre>' +
          '</section>'
        : '';

      return (
        '<aside class="csa-rt-inspector csa-rt-act-inspector" aria-label="第 ' + act.order + ' 幕诊断详情">' +
          '<header class="csa-rt-insp-head">' +
            '<div>' +
              '<span class="csa-rt-insp-kicker">ACT 0' + act.order + ' · ' + esc(act.semanticObject) + '</span>' +
              '<h3>' + esc(act.stage) + ' <span class="csa-rt-tier is-slow">' + esc(act.tag) + '</span></h3>' +
            '</div>' +
            '<button type="button" class="btn btn-ghost btn-sm csa-rt-close" data-rt-close-act aria-label="关闭诊断">×</button>' +
          '</header>' +
          '<div class="csa-rt-insp-body">' +
            '<section class="csa-rt-insp-sec">' +
              '<h4>实测证据（Rank ' + rd.rank + '）</h4>' +
              statsHtml +
            '</section>' +
            traceSectionHtml +
            diffHtml +
            '<div class="csa-rt-actions">' +
              '<button type="button" class="btn btn-sm csa-rt-cta-primary" data-rt-open-assignment="' + esc(act.id) + '"' + (assignLane != null ? ' data-rt-assign-lane="' + assignLane + '"' : '') + '>' + esc(assignActionLabel) + '</button>' +
              (isGraphRoot ? '<button type="button" class="btn btn-ghost btn-sm" data-rt-open-graph="' + esc(graphTarget) + '" data-rt-act-ctx="' + esc(act.id) + '">' + esc(graphActionLabel) + '</button>' : '') +
              sourceBtns +
            '</div>' +
          '</div>' +
        '</aside>'
      );
    }

    function buildGroupInspectorHtml(rd) {
      if (state.selectedGroup == null) return '';
      const g = rd.groups.find(x => x.group === state.selectedGroup);
      if (!g) return '';
      const wm = g.workMix || null;
      const ls = laneStats(g.group);
      const isSlow = g.tier === 'slow';
      const m0Pct = g.merge0.durUs > 0 ? Math.round((g.merge0.setupUs / g.merge0.durUs) * 100) : 0;
      const m1Pct = g.merge1.durUs > 0 ? Math.round((g.merge1.setupUs / g.merge1.durUs) * 100) : 0;

      return (
        '<aside class="csa-rt-inspector" aria-label="Core Group 实测详情">' +
          '<header class="csa-rt-insp-head">' +
            '<div>' +
              '<span class="csa-rt-insp-kicker">RANK ' + rd.rank + ' · ' + g.aic.startUs.toFixed(0) + '→' + g.aiv0.endUs.toFixed(0) + ' µs</span>' +
              '<h3>Group ' + String(g.group).padStart(2, '0') + ' <span class="csa-rt-tier ' + (isSlow ? 'is-slow' : 'is-fast') + '">' + (isSlow ? '慢组 · 重压缩块' : '快组 · 轻滑窗块') + '</span></h3>' +
            '</div>' +
            '<button type="button" class="btn btn-ghost btn-sm csa-rt-close" data-rt-close aria-label="关闭详情">×</button>' +
          '</header>' +
          '<div class="csa-rt-insp-body">' +
            '<section class="csa-rt-insp-sec">' +
              '<h4>核组实测（1 AIC + 2 AIV）</h4>' +
              '<table class="csa-rt-mini-table">' +
                '<thead><tr><th>核</th><th>时长</th><th>主因</th></tr></thead>' +
                '<tbody>' +
                  '<tr><td><i class="csa-rt-dot is-aic"></i>AIC · C' + g.aic.core + '</td><td>' + fmtUs(g.aic.durUs) + '</td><td>' + (wm ? wm.sb0 + ' SB0 + ' + wm.cmp + ' CMP' : 'QK / PV') + '</td></tr>' +
                  '<tr><td><i class="csa-rt-dot is-aiv0"></i>AIV 0 · V' + g.aiv0.core + '</td><td>' + fmtUs(g.aiv0.durUs) + '</td><td class="is-accent">' + fmtSigned(g.aiv0TailUs) + ' 写零尾</td></tr>' +
                  '<tr><td><i class="csa-rt-dot is-aiv1"></i>AIV 1 · V' + g.aiv1.core + '</td><td>' + fmtUs(g.aiv1.durUs) + '</td><td>0-row 握手</td></tr>' +
                '</tbody>' +
              '</table>' +
            '</section>' +
            (wm ? (
              '<section class="csa-rt-insp-sec">' +
                '<h4>语义分配构成（Lane ' + String(wm.logicalLane).padStart(2, '0') + '）</h4>' +
                '<dl class="csa-rt-dl">' +
                  '<dt>有效项</dt><dd><b>' + wm.valid + '</b>（<b>' + wm.sb0 + '</b> SB0 滑窗 · <b>' + wm.cmp + '</b> CMP 压缩）</dd>' +
                  '<dt>改排后均衡</dt><dd>' + wm.balancedSb0 + ' SB0 · ' + wm.balancedCmp + ' CMP（消除奇偶偏斜）</dd>' +
                  '<dt>队尾无效块</dt><dd><b>' + wm.invalid + ' 块</b>（AIV 0 白写 <b>' + wm.invalidZeroMiB.toFixed(2) + ' MiB</b> 零）</dd>' +
                '</dl>' +
              '</section>'
            ) : (ls ? (
              '<section class="csa-rt-insp-sec">' +
                '<h4>Lane ' + String(g.group).padStart(2, '0') + ' 迭代构成</h4>' +
                '<dl class="csa-rt-dl">' +
                  '<dt>Work Items</dt><dd>' + ls.total + ' 轮（' + ls.valid + 'V / ' + ls.invalid + 'I）</dd>' +
                  '<dt>KV 路径</dt><dd>' + ls.original + ' Orig · ' + ls.compressed + ' Comp</dd>' +
                '</dl>' +
              '</section>'
            ) : '')) +
            '<section class="csa-rt-insp-sec">' +
              '<h4>下游 Merge 连锁</h4>' +
              '<dl class="csa-rt-dl">' +
                '<dt>Setup 等待</dt><dd><b>' + m0Pct + '%</b> / <b>' + m1Pct + '%</b>（' + fmtUs(g.merge0.setupUs) + '）</dd>' +
                '<dt>Merge 计算</dt><dd>' + fmtUs(g.merge0.kernelUs) + '（含 66–68% 无效零块）</dd>' +
              '</dl>' +
              '<p class="csa-rt-note">' + (isSlow
                ? '尾组吃满重压缩块决定全链路尾线；释放依赖后下游 Merge 零空等直接开始计算。'
                : '早于慢组 ' + Math.max(0, rd.qkEndUs - g.aiv0.endUs).toFixed(0) + ' µs 结束，队尾 12 组同时写零拖尾 +' + g.aiv0TailUs.toFixed(0) + ' µs，随后在 Merge 干等慢组。') + '</p>' +
            '</section>' +
            '<div class="csa-rt-actions">' +
              '<button type="button" class="btn btn-sm csa-rt-cta-primary" data-rt-open-assignment="act2-qk-pv-parity" data-rt-assign-lane="' + (wm ? wm.logicalLane : g.group) + '">在 Assignment 中看 Lane ' + String(wm ? wm.logicalLane : g.group).padStart(2, '0') + ' 构成 →</button>' +
              '<button type="button" class="btn btn-ghost btn-sm" data-rt-open-instance="' + g.group + '">进入单 Work Item 因果泳道 →</button>' +
              '<button type="button" class="btn btn-ghost btn-sm" data-rt-open-lane="' + g.group + '">切换静态 Lane Map →</button>' +
              '<button type="button" class="btn btn-ghost btn-sm" data-rt-open-graph="sparse/attention/qk-pv/selected" data-rt-act-ctx="act2-qk-pv-parity">对照 SB0 / CMP 访存分支代价（Graph）→</button>' +
              '<button type="button" class="btn btn-ghost btn-sm" data-rt-source="decode_sparse_attn_csa.py:184:276">decode_sparse_attn_csa.py : L184–276 ↗</button>' +
            '</div>' +
          '</div>' +
        '</aside>'
      );
    }

    function buildDiffHtml(diff) {
      if (!diff) return '';
      const delLines = String(diff.before || '').split('\n').map(l => '<span class="csa-rt-diff-del">− ' + esc(l) + '</span>').join('');
      const addLines = String(diff.after || '').split('\n').map(l => '<span class="csa-rt-diff-add">+ ' + esc(l) + '</span>').join('');
      return '<div class="csa-rt-diff-box">' + delLines + addLines + '</div>';
    }

    function buildOptInspectorHtml(rd) {
      const c = rd.chain;
      const acts = storyActs();
      if (!c || !acts.length) return '';
      const b = c.breakdown;
      const proj = c.projectedAfterFix;
      const wp = c.workProfile;

      const on1 = !!state.optEnabled['act1-qk-plan-barrier'];
      const on2 = !!state.optEnabled['act2-qk-pv-parity'];
      const on3 = !!state.optEnabled['act3-invalid-zero-flow'];

      const basePlan = b.qkPlanUs;
      const baseGap = b.gapPlanToQkUs;
      const baseAic = b.qkAicMaxUs;
      const baseTail = b.qkAiv0TailUs;
      const baseMerge = b.mergeAfterQkUs;
      const baseTotal = c.chainSpanUs;

      const afterPlan = on1 ? proj.qkPlanUs : basePlan;
      const afterAic = on2 ? proj.qkAicUs : baseAic;
      const afterTail = on3 ? proj.qkAiv0TailUs : baseTail;
      const afterMerge = on3 ? proj.mergeAfterQkUs : baseMerge;
      const afterTotal = afterPlan + baseGap + afterAic + afterTail + afterMerge;
      const savingUs = Math.max(0, baseTotal - afterTotal);
      const savingPct = (savingUs / Math.max(1, baseTotal)) * 100;
      const layerSavingPct = (savingUs / Math.max(1, c.csaTotalSpanUs)) * 100;
      const hbmSavedMiB = on3 ? wp.invalidTotalHbmMiB : 0;

      const segW = v => ((Math.max(0, v) / Math.max(1, baseTotal)) * 100).toFixed(2) + '%';

      const cardsHtml = acts.map(act => {
        const checked = !!state.optEnabled[act.id];
        const estSave = act.id === 'act1-qk-plan-barrier'
          ? Math.round(proj.qkPlanSavingUs)
          : act.id === 'act2-qk-pv-parity'
          ? Math.round(proj.qkAicSavingUs)
          : Math.round(proj.qkAiv0TailSavingUs + proj.mergeSavingUs);
        const srcBtns = (act.sourceRefs || []).map(ref =>
          '<button type="button" class="btn btn-ghost btn-sm" data-rt-source="' + esc(ref.file + ':' + ref.start + ':' + ref.end) + '">' + esc(ref.file + ' · L' + ref.start + '–' + ref.end) + ' ↗</button>'
        ).join('');
        return (
          '<section class="csa-rt-opt-card' + (checked ? ' is-enabled' : '') + '">' +
            '<label class="csa-rt-opt-check">' +
              '<input type="checkbox" data-rt-opt-toggle="' + esc(act.id) + '"' + (checked ? ' checked' : '') + '>' +
              '<span class="csa-rt-opt-title"><b>修复 ' + act.order + ' · ' + esc(act.stage) + '</b><small>' + esc(act.sourceDiff?.lineRange || '') + '</small></span>' +
              '<span class="csa-rt-opt-save">−~' + estSave + ' µs</span>' +
            '</label>' +
            '<p class="csa-rt-note">' + esc(act.sourceDiff?.summary || '') + (act.sourceDiff?.bitExact ? ' · 数值 bit-exact' : '') + '</p>' +
            buildDiffHtml(act.sourceDiff) +
            '<div class="csa-rt-opt-refs">' + srcBtns + '</div>' +
          '</section>'
        );
      }).join('');

      return (
        '<aside class="csa-rt-inspector csa-rt-opt-inspector" aria-label="三处源码直修与前后对比">' +
          '<header class="csa-rt-insp-head">' +
            '<div>' +
              '<span class="csa-rt-insp-kicker">SOURCE-FIX LOOP · RANK ' + rd.rank + '</span>' +
              '<h3>前后对比 & 源码 Diff</h3>' +
            '</div>' +
            '<button type="button" class="btn btn-ghost btn-sm csa-rt-close" data-rt-close-opt aria-label="关闭对比面板">×</button>' +
          '</header>' +
          '<div class="csa-rt-insp-body">' +
            '<section class="csa-rt-insp-sec">' +
              '<h4>实测基线 vs 勾选修复推算</h4>' +
              '<div class="csa-rt-cmp-bars">' +
                '<div class="csa-rt-cmp-row">' +
                  '<div class="csa-rt-cmp-label"><span>实测基线（DFX 原始）</span><b>' + baseTotal.toFixed(1) + ' µs</b></div>' +
                  '<div class="csa-rt-cmp-track">' +
                    '<span class="csa-rt-cmp-seg is-plan" style="width:' + segW(basePlan) + '" title="qk_plan: ' + basePlan.toFixed(1) + ' µs"></span>' +
                    '<span class="csa-rt-cmp-seg is-aic" style="width:' + segW(baseAic) + '" title="qk_pv AIC: ' + baseAic.toFixed(1) + ' µs"></span>' +
                    '<span class="csa-rt-cmp-seg is-tail" style="width:' + segW(baseTail) + '" title="AIV 0 写零尾: ' + baseTail.toFixed(1) + ' µs"></span>' +
                    '<span class="csa-rt-cmp-seg is-merge" style="width:' + segW(baseMerge) + '" title="Merge 尾段: ' + baseMerge.toFixed(1) + ' µs"></span>' +
                  '</div>' +
                '</div>' +
                '<div class="csa-rt-cmp-row">' +
                  '<div class="csa-rt-cmp-label"><span>勾选后推算（模型估算）</span><b class="is-good">~' + afterTotal.toFixed(1) + ' µs</b></div>' +
                  '<div class="csa-rt-cmp-track">' +
                    '<span class="csa-rt-cmp-seg is-plan" style="width:' + segW(afterPlan) + '" title="优化后 qk_plan: ~' + afterPlan.toFixed(1) + ' µs"></span>' +
                    '<span class="csa-rt-cmp-seg is-aic" style="width:' + segW(afterAic) + '" title="优化后 qk_pv AIC: ~' + afterAic.toFixed(1) + ' µs"></span>' +
                    '<span class="csa-rt-cmp-seg is-tail" style="width:' + segW(afterTail) + '" title="优化后 AIV 0 尾: ~' + afterTail.toFixed(1) + ' µs"></span>' +
                    '<span class="csa-rt-cmp-seg is-merge" style="width:' + segW(afterMerge) + '" title="优化后 Merge 尾段: ~' + afterMerge.toFixed(1) + ' µs"></span>' +
                  '</div>' +
                '</div>' +
              '</div>' +
              '<dl class="csa-rt-dl csa-rt-opt-summary">' +
                '<dt>预计链路节省</dt><dd><b>−~' + savingUs.toFixed(1) + ' µs</b>（−' + savingPct.toFixed(1) + '% 链路 · −' + layerSavingPct.toFixed(1) + '% 整层 CSA）</dd>' +
                '<dt>HBM 无效流量</dt><dd>' + (hbmSavedMiB > 0 ? '<b>−' + hbmSavedMiB.toFixed(1) + ' MiB</b>（免写免读无效零块）' : '未启用修复 3（仍读写 ' + wp.invalidTotalHbmMiB.toFixed(1) + ' MiB 零）') + '</dd>' +
              '</dl>' +
              '<p class="csa-rt-note">说明：基线为 DFX 原始实测；优化后数据基于 SB0/CMP 单块拟合代价与无效块比例推算，用于指导源码改动与回归验收。</p>' +
            '</section>' +
            cardsHtml +
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
      const bounds = windowBounds(rd);
      const act = currentAct();
      const inspectorHtml = state.showOptPanel
        ? buildOptInspectorHtml(rd)
        : (state.selectedGroup != null
          ? buildGroupInspectorHtml(rd)
          : (act ? buildActInspectorHtml(rd, act) : ''));
      const hasInspector = Boolean(inspectorHtml);

      const prevScroll = host.querySelector('.csa-rt-rows')?.scrollTop ?? null;
      host.innerHTML =
        '<div class="csa-rt-shell' + (hasInspector ? ' has-inspector' : '') + (state.showOptPanel ? ' has-opt-panel' : '') + '">' +
          '<div class="csa-rt-main">' +
            buildStoryStripHtml(rd) +
            buildTopStripHtml(rd) +
            buildStepsHtml(rd, bounds) +
            '<div class="csa-rt-board">' +
              buildAxisHtml(rd, bounds) +
              '<div class="csa-rt-rows">' + buildRowsHtml(rd, bounds) + '</div>' +
            '</div>' +
          '</div>' +
          inspectorHtml +
        '</div>';
      if (prevScroll != null) {
        const rowsEl = host.querySelector('.csa-rt-rows');
        if (rowsEl) rowsEl.scrollTop = prevScroll;
      }
    }

    host.addEventListener('change', event => {
      const toggle = event.target?.closest?.('[data-rt-opt-toggle]');
      if (!toggle) return;
      const id = toggle.dataset.rtOptToggle;
      state.optEnabled[id] = !!toggle.checked;
      render();
    });

    host.addEventListener('click', event => {
      const btn = event.target.closest('button');
      if (!btn) return;
      if (btn.hasAttribute('data-rt-opt-open')) {
        state.showOptPanel = !state.showOptPanel;
        if (state.showOptPanel) {
          state.selectedGroup = null;
          state.selectedAct = null;
          state.highlightStep = null;
        }
        render();
        return;
      }
      if (btn.hasAttribute('data-rt-close-opt')) {
        state.showOptPanel = false;
        render();
        return;
      }
      if (btn.hasAttribute('data-rt-close')) {
        state.selectedGroup = null;
        opts.onSelectGroup?.(null);
        render();
        return;
      }
      if (btn.hasAttribute('data-rt-close-act')) {
        state.selectedAct = null;
        state.highlightStep = null;
        opts.onSelectAct?.(null);
        render();
        return;
      }
      if (btn.dataset.rtAct) {
        state.showOptPanel = false;
        const nextAct = btn.dataset.rtAct;
        if (state.selectedAct === nextAct) {
          state.selectedAct = null;
          state.highlightStep = null;
        } else {
          state.selectedAct = nextAct;
          state.selectedGroup = null;
          if (nextAct === 'act1-qk-plan-barrier') {
            state.showPlan = true;
            state.highlightStep = 'csa_slots_build_valid_qk_plan';
          } else if (nextAct === 'act2-qk-pv-parity') {
            state.showMerge = true;
            state.sortBy = 'duration';
            state.highlightStep = null;
          } else if (nextAct === 'act3-invalid-zero-flow') {
            state.showMerge = true;
            state.highlightStep = null;
          }
        }
        opts.onSelectAct?.(state.selectedAct);
        render();
        return;
      }
      if (btn.dataset.rtStep) {
        const s = btn.dataset.rtStep;
        state.highlightStep = state.highlightStep === s ? null : s;
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
        if (key === 'plan') {
          state.showPlan = !state.showPlan;
          if (!state.showPlan && state.highlightStep === 'csa_slots_build_valid_qk_plan') state.highlightStep = null;
        } else if (key === 'aiv1') {
          state.showAiv1Idle = !state.showAiv1Idle;
        } else if (key === 'merge') {
          state.showMerge = !state.showMerge;
          if (!state.showMerge && state.highlightStep === 'csa_merge_pack_publish') state.highlightStep = null;
        }
        render();
        return;
      }
      if (btn.dataset.rtGroup != null) {
        const clicked = Number(btn.dataset.rtGroup);
        state.selectedGroup = state.selectedGroup === clicked ? null : clicked;
        if (state.selectedGroup != null) {
          state.selectedAct = null;
          state.showOptPanel = false;
        }
        opts.onSelectGroup?.(state.selectedGroup);
        render();
        return;
      }
      if (btn.dataset.rtOpenAssignment != null) {
        const actId = btn.dataset.rtOpenAssignment || null;
        const lane = btn.dataset.rtAssignLane != null ? Number(btn.dataset.rtAssignLane) : null;
        opts.onOpenAssignment?.({ actId, lane, rank: state.rank });
        return;
      }
      if (btn.dataset.rtOpenGraph != null) {
        opts.onOpenGraph?.(btn.dataset.rtOpenGraph, btn.dataset.rtActCtx || state.selectedAct);
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
      if (next.selectedAct !== undefined) {
        state.selectedAct = next.selectedAct || null;
      }
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
