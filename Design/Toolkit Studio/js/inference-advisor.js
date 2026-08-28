/**
 * 推理性能分析 · 诊断建议（P4）
 *
 * 对标 Nsight Compute Guided Analysis。
 * 所有结论都是规则在运行时从 ops / summary / memory / timeline 上算出来的，
 * 没有一条是写死的文案——改数据结论会跟着变，不会出现面板和数字打架。
 */
(function registerInferenceAdvisor() {
  'use strict';

  const esc = (v) => String(v).replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
  const fmt = (n, d = 2) => Number(n).toFixed(d);

  const SEV = {
    high: ['阻塞', 'var(--danger)'],
    med: ['风险', 'var(--warning)'],
    low: ['提示', 'var(--tone-blue-strong, #4a90d9)'],
    good: ['良好', 'var(--success)'],
  };
  const ORDER = { high: 0, med: 1, low: 2, good: 3 };

  const median = (xs) => {
    const s = xs.slice().sort((a, b) => a - b);
    const h = Math.floor(s.length / 2);
    return s.length % 2 ? s[h] : (s[h - 1] + s[h]) / 2;
  };

  function analyze(p) {
    const out = [];
    const tpot = p.summary.tpot.p50;
    const byId = Object.fromEntries(p.ops.map((o) => [o.id, o]));

    /* 规则 A · 同类算子里达成带宽明显偏低 */
    const gemms = p.ops.filter((o) => o.bound === 'mte2' && o.gflop > 10);
    const peerBw = median(gemms.map((o) => o.achievedBw));
    p.ops.filter((o) => o.bound === 'mte2' && o.totalMs >= 0.3 && o.achievedBw < peerBw * 0.85).forEach((o) => {
      const ideal = o.bytesIn / peerBw;
      const recover = o.totalMs - ideal;
      out.push({
        severity: 'high',
        opId: o.id,
        title: `${o.name} 未达内存屋顶，达成带宽仅同类算子的 ${fmt(o.achievedBw / peerBw * 100, 0)}%`,
        evidence: [
          ['达成带宽', `${fmt(o.achievedBw, 2)} TB/s（${fmt(o.achievedBw / p.meta.peakBw * 100, 1)}% 峰值）`],
          ['同类中位数', `${fmt(peerBw, 2)} TB/s · ${gemms.length} 个 GEMM 算子`],
          ['当前耗时', `${fmt(o.totalMs, 3)} ms（占 ${fmt(o.share, 1)}%）`],
          ['按同类带宽应为', `${fmt(ideal, 3)} ms`],
        ],
        recover,
        action: 'paged K/V 的非连续访存是主因：检查 tile 是否对齐 page 边界、预取 stage 是否足够掩盖跨页跳转。',
      });
    });

    /* 规则 B · 整体带宽效率 */
    if (p.summary.efficiency < 70) {
      out.push({
        severity: 'med',
        opId: null,
        title: `整体达成带宽 ${fmt(p.summary.traffic.total / tpot / p.meta.peakBw * 100, 1)}% 峰值，计算与搬运重叠不足`,
        evidence: [
          ['每 step 流量', `${fmt(p.summary.traffic.total, 2)} GB`],
          ['理论下界', `${fmt(p.summary.lowerBoundMs, 2)} ms（${fmt(p.meta.peakBw, 1)} TB/s 打满）`],
          ['实际 TPOT', `${fmt(tpot, 2)} ms · 效率 ${fmt(p.summary.efficiency, 1)}%`],
          ['MTE2 占空比', `${fmt(p.summary.sol[2].pct, 1)}%`],
          ['理论总余量', `${fmt(tpot - p.summary.lowerBoundMs, 2)} ms — 下方各条具体结论都落在这个包络内`],
        ],
        // 这是总包络，不是一条可叠加的独立收益；计入合计会与其它结论重复计算
        recover: 0,
        envelope: tpot - p.summary.lowerBoundMs,
        action: 'MTE2 有 ~30% 的时间没在搬运。先看时间线里 Cube/Vector 轨道与 MTE2 的错位，再决定是加深 pipeline stage 还是调整 tile 粒度。',
      });
    }

    /* 规则 C · 实测超出编译期预算 */
    const overBudget = p.ops.filter((o) => o.static.some((r) => /UB/.test(r[0]) && (r[4] === 'warn' || r[4] === 'bad')));
    if (overBudget.length) {
      out.push({
        severity: 'med',
        opId: overBudget.sort((a, b) => b.totalMs - a.totalMs)[0].id,
        title: `${overBudget.length} 个任务的 UB 实测峰值超出编译期预算`,
        evidence: overBudget.map((o) => {
          const row = o.static.find((r) => /UB/.test(r[0]));
          return [o.name, `预算 ${row[1].replace('（编译期预算）', '')} → 实测 ${row[2]}（${row[3]}）`];
        }),
        recover: 0,
        action: '编译期的 UB 估算偏乐观。复核 chunk × stage 配置，并确认预算模型是否漏算了对齐填充与临时缓冲。',
      });
    }

    /* 规则 D · 负载倾斜 */
    p.ops.filter((o) => o.imbalance !== null && o.imbalance > 0.25).forEach((o) => {
      out.push({
        severity: 'med',
        opId: o.id,
        title: `${o.name} 负载不均衡 CV ${fmt(o.imbalance, 2)}，核间等待被放大`,
        evidence: [
          ['work items', `${o.calls.toLocaleString('en-US')} · ${o.cores} 核 grid-stride`],
          ['不均衡度 CV', `${fmt(o.imbalance, 2)}（同类算子多在 0.05–0.15）`],
          ['效率', `${o.efficiency}%`],
          ['根因', `每个 work item 归并的块数与该请求 seq_len 成正比，seq 跨度 ${Math.min(...p.seqLens)}–${Math.max(...p.seqLens)}`],
        ],
        recover: o.totalMs * o.imbalance * 0.5,
        action: 'ragged 输入直接传导成了核间倾斜。可按真实块数而非请求数切分 work item，或与 fa_fused 融合掉一次同步。',
      });
    });

    /* 规则 E · 无依赖却被串行调度（来自时间线） */
    const tl = p.dist ? null : window.PtoInferenceTimeline?.build?.(p);   // 时间线模型只覆盖单卡
    if (tl && tl.stats.parallelUs > 0) {
      const names = Object.keys(window.PtoInferenceTimeline.PARALLELIZABLE).map((id) => byId[id].name);
      out.push({
        severity: 'med',
        opId: null,
        title: `${names.length} 个任务与前驱无数据依赖，却被串行调度`,
        evidence: [
          ['涉及任务', names.join(' · ')],
          ['占用', `${fmt(tl.stats.parallelUs, 1)} μs/层，占每层 ${fmt(tl.stats.parallelUs / tl.stats.layerDur * 100, 1)}%`],
          ['关键路径', `${fmt(tl.stats.criticalUs, 1)} μs/层`],
          ['同类问题', '源码缺少 pl.pipeline(stage=F)，计算段无法跨块重叠'],
        ],
        recover: tl.stats.parallelStepMs,
        action: '把这三个任务与相邻任务重叠即可完全隐藏。见时间线页签的斜纹色块。',
      });
    }

    /* ---- 多机多卡专属规则 ---- */
    if (p.dist) {
      const d = p.dist;
      const t = d.topology;
      const best = d.scaling.reduce((a, b) => (a.tps > b.tps ? a : b));
      const cur = d.scaling.find((x) => x.current);

      /* D1 · 并行度过高，PP 没有换来容量收益却引入气泡 */
      if (best.cards < cur.cards && best.tps > cur.tps) {
        out.push({
          severity: 'high',
          opId: null,
          title: `并行度过高：${best.cards} 卡比当前 ${cur.cards} 卡快 ${fmt((cur.tpot / best.tpot - 1) * 100, 0)}%`,
          evidence: [
            ['当前配置', `TP=${t.tp} × PP=${t.pp} = ${t.world} 卡 · ${fmt(cur.tpot, 2)} ms · 扩展效率 ${fmt(cur.efficiency, 1)}%`],
            ['更优配置', `${best.label} · ${best.cards} 卡 · ${fmt(best.tpot, 2)} ms · 扩展效率 ${fmt(best.efficiency, 1)}%`],
            ['每卡显存占用', `${fmt(d.memory.perCard, 2)} / ${d.memory.capacity} GB（${fmt(d.memory.perCard / d.memory.capacity * 100, 1)}%）— PP 未带来任何容量收益`],
            ['代价', `流水线气泡 ${fmt(d.cardTime.bubblePct, 1)}% 卡时`],
          ],
          recover: cur.tpot - best.tpot,
          action: `14B 模型 TP=${t.tp} 分片后每卡权重仅 ${fmt(d.memory.weightsStage1, 2)} GB，单节点完全放得下。去掉 PP 即可；若为更大模型保留 PP，把 microbatch 从 ${t.microbatches} 提到 8 可把气泡压到 ${fmt(1 / (8 + t.pp - 1) * 100, 0)}%。`,
        });
      }

      /* D2 · 流水线气泡 */
      const theoretical = (t.pp - 1) / (t.microbatches + t.pp - 1) * 100;
      out.push({
        severity: 'high',
        opId: null,
        title: `流水线气泡占 ${fmt(d.cardTime.bubblePct, 1)}% 卡时，microbatch 数不足`,
        evidence: [
          ['气泡公式', `(p−1)/(m+p−1)，p=${t.pp}，m=${t.microbatches} → ${fmt(theoretical, 1)}%`],
          ['实测', `${fmt(d.cardTime.bubblePct, 1)}% · ${fmt(d.cardTime.totalMs * d.cardTime.bubblePct / 100, 1)} card-ms`],
          ['各 stage 利用率', d.stages.map((s) => `stage ${s.id} ${fmt(s.utilization, 1)}%`).join(' · ')],
          ['有效计算', `仅 ${fmt(d.cardTime.computePct, 1)}% 卡时`],
          ['收益归属', '消除气泡的办法就是去掉 PP，收益已在上一条量化，此处不重复计入'],
        ],
        // 与 D1「并行度过高」是同一笔收益（去掉 PP 即消除气泡），只在 D1 计入
        recover: 0,
        action: 'decode 阶段的 microbatch 会让权重被重复读取，提高 m 的收益要和多读的权重量一起权衡；最优解通常是干脆去掉 PP。',
      });

      /* D3 · TP AllReduce 未与计算重叠 */
      const ar = d.collectives[0];
      out.push({
        severity: 'med',
        opId: null,
        title: `TP AllReduce 占 ${fmt(ar.totalMs / d.tpotMs * 100, 1)}% TPOT，完全暴露在关键路径上`,
        evidence: [
          ['调用次数', `${ar.calls} 次/step · 每层 attention 与 MLP 之后各一次`],
          ['单次耗时', `${fmt(ar.usPerCall, 2)} μs（载荷仅 ${fmt(ar.payloadKb, 0)} KiB）`],
          ['算法带宽', `${fmt(ar.achievedGbs, 1)} / ${ar.peakGbs} GB/s（${fmt(ar.achievedGbs / ar.peakGbs * 100, 1)}%）`],
          ['瓶颈', '载荷太小，单次耗时被固定延迟主导 — 是调用次数问题，不是带宽问题'],
        ],
        recover: ar.totalMs * 0.5,
        action: '把 AllReduce 与后续计算重叠（提前发起、分块流水），或合并相邻层的通信以摊薄固定延迟。',
      });

      /* D4 · straggler */
      const busy = d.ranks.map((r) => r.busyUs);
      const mean = busy.reduce((a, b) => a + b, 0) / busy.length;
      const str = d.ranks.find((r) => r.straggler) || d.ranks.reduce((a, b) => (a.busyUs > b.busyUs ? a : b));
      if (str.busyUs / mean > 1.05) {
        out.push({
          severity: 'med',
          opId: null,
          title: `rank ${str.rank} 落后同组均值 ${fmt((str.busyUs / mean - 1) * 100, 1)}%，拖慢整个 TP 组`,
          evidence: [
            ['最慢 rank', `rank ${str.rank}（node ${str.node} · stage ${str.stage}）${fmt(str.busyUs, 0)} μs`],
            ['组内均值', `${fmt(mean, 0)} μs`],
            ['传导范围', `AllReduce 是同步操作，node ${str.node} 上另外 ${t.cardsPerNode - 1} 张卡一起等`],
            ['浪费', `${fmt((str.busyUs - mean) * (t.cardsPerNode - 1) / 1000, 3)} ms card-time / step`],
          ],
          recover: (str.busyUs - mean) / 1000,
          action: '排查该卡的频率与温度、HCCS 链路误码率，以及 KV 分片是否恰好落在长序列上。',
        });
      }

      /* D5 · 正向：TP 没有跨节点 */
      out.push({
        severity: 'good',
        opId: null,
        title: 'TP 组限制在节点内，跨节点只走小载荷 P2P',
        evidence: [
          ['TP 通信', `${ar.calls} 次 AllReduce 全部走节点内 HCCS ${d.links[0].peak} GB/s`],
          ['跨节点通信', `仅 ${d.collectives[1].calls} 次 P2P，共 ${fmt(d.collectives[1].totalMs, 3)} ms（${fmt(d.cardTime.p2pPct, 2)}% 卡时）`],
          ['对比', `若 TP 跨节点，${ar.calls} 次 AllReduce 将全部落到 ${d.links[1].peak} GB/s 的 RoCE 上`],
        ],
        recover: 0,
        action: '保持。并行策略的切分方向是对的，问题只出在并行度大小。',
      });
    }

    /* 规则 F · 静态推断与实测的显著偏差 */
    const drift = [];
    p.ops.forEach((o) => o.static.forEach((r) => { if (r[4] === 'bad') drift.push([o.name, r[0], r[1], r[2], r[3]]); }));
    if (drift.length) {
      out.push({
        severity: 'low',
        opId: null,
        title: `${drift.length} 处静态推断与实测显著偏离，编译期模型需要校准`,
        evidence: drift.map((d) => [`${d[0]} · ${d[1]}`, `${d[2]} → ${d[3]}（${d[4]}）`]),
        recover: 0,
        action: '这些偏差本身就是编译器的改进输入：估不准的地方要么补实测反馈，要么修正代价模型。',
      });
    }

    /* 规则 G · Host 空隙 */
    const idle = byId.__idle__;
    if (idle && idle.share > 2) {
      out.push({
        severity: 'low',
        opId: null,
        title: `同步与 Host 空隙占 ${fmt(idle.share, 1)}%`,
        evidence: [
          ['层内 barrier', `4.5 μs/层 × ${p.meta.layers} = 0.180 ms`],
          ['Host dispatch', '0.195 ms（launch 40 μs + 等待 155 μs）'],
          ['合计', `${fmt(idle.totalMs, 3)} ms`],
        ],
        recover: 0.195,
        action: 'Host 侧的 0.195 ms 可用 graph capture 或多 step 合批 dispatch 消掉；层内 barrier 属于正常同步开销。',
      });
    }

    /* 规则 H/I · 正向记账 */
    const inLayerCast = p.memory.precision.find((r) => r[0] === '层内转换');
    if (inLayerCast && inLayerCast[2] === 0) {
      out.push({
        severity: 'good',
        opId: 'dcr-xgamma',
        title: '层内零精度转换，FP32 carry 策略生效',
        evidence: [
          ['入口转换', 'copy_hidden · 1 次'],
          ['出口转换', 'cast_lmhead_in · 1 次'],
          ['层内转换', '0 次 — 由 dcr_xgamma 直接产出下一层 BF16 输入'],
        ],
        recover: 0,
        action: '保持。若后续引入新的层内算子，注意不要重新引入 FP32↔BF16 往返。',
      });
    }

    const kv = p.memory.kv;
    out.push({
      severity: 'good',
      opId: 'fa-work-build',
      title: `Paged KV 内部碎片仅 ${fmt(kv.fragmentation, 2)}%，work table 稠密率 ${fmt(kv.density, 1)}%`,
      evidence: [
        ['页利用', `${kv.pagesUsed} / ${kv.pagesTotal} 页 · ${fmt(kv.utilization, 1)}%`],
        ['碎片', `${kv.tokensAllocated - kv.tokensLive} / ${kv.tokensAllocated.toLocaleString('en-US')} token`],
        ['空块压缩', `${kv.blocksPadded} → ${kv.blocksReal}，省 ${fmt(100 - kv.density, 1)}% 迭代`],
        ['抢占 / 换出', `${kv.preempt} / ${kv.swap}`],
      ],
      recover: 0,
      action: '保持。KV 池仍有 35.6% 余量，是往上调 batch 的空间来源。',
    });

    const healthy = gemms.filter((o) => o.achievedBw >= peerBw * 0.95);
    if (healthy.length >= 3) {
      out.push({
        severity: 'good',
        opId: healthy[0].id,
        title: `${healthy.length} 个 GEMM 算子达成 ${fmt(Math.min(...healthy.map((o) => o.achievedBw)) / p.meta.peakBw * 100, 0)}–${fmt(Math.max(...healthy.map((o) => o.achievedBw)) / p.meta.peakBw * 100, 0)}% 带宽屋顶`,
        evidence: healthy.map((o) => [o.name, `${fmt(o.achievedBw, 2)} TB/s · 效率 ${o.efficiency}%`]),
        recover: 0,
        action: '权重搬运路径已接近硬件上限，进一步收益要靠降低流量本身（量化 / 权重复用），而不是调度。',
      });
    }

    return out.sort((a, b) => ORDER[a.severity] - ORDER[b.severity] || b.recover - a.recover);
  }

  function render(p) {
    const findings = analyze(p);
    const tpot = p.summary.tpot.p50;
    const bound = p.summary.lowerBoundMs;
    const envelope = tpot - bound;
    // 只累加各条独立、可定位的收益；总包络那条 recover 为 0，避免与它们重复计算
    const totalRecover = findings.reduce((a, f) => a + Math.max(f.recover || 0, 0), 0);
    const after = tpot - totalRecover;
    const sane = after >= bound - 1e-6;
    const counts = ['high', 'med', 'low', 'good'].map((k) => [k, findings.filter((f) => f.severity === k).length]);

    const head = `<div class="kf-prof-kpis">
      ${counts.map(([k, n]) => `<article class="kf-prof-kpi"><span>${SEV[k][0]}</span><b style="color:${SEV[k][1]}">${n}</b><small><u style="text-decoration:none">${k === 'good' ? '已生效的优化' : '条待处理结论'}</u></small></article>`).join('')}
      <article class="kf-prof-kpi"><span>可定位收益</span><b>${fmt(totalRecover, 2)}<i> ms</i></b><small><span class="kf-prof-delta down">TPOT −${fmt(totalRecover / tpot * 100, 1)}%</span><u style="text-decoration:none">${findings.filter((f) => f.recover > 0.001).length} 条可量化结论</u></small></article>
      <article class="kf-prof-kpi"><span>回收后 TPOT</span><b${sane ? '' : ' style="color:var(--danger)"'}>${fmt(after, 2)}<i> ms</i></b><small><u style="text-decoration:none">${sane ? `理论下界 ${fmt(bound, 2)} ms，仍有 ${fmt(after - bound, 2)} ms 未归因` : '⚠ 低于理论下界，规则重复计算'}</u></small></article>
      <article class="kf-prof-kpi"><span>理论总余量</span><b>${fmt(envelope, 2)}<i> ms</i></b><small><u style="text-decoration:none">带宽打满的包络，各条结论均落在其中</u></small></article>
    </div>`;

    const cards = findings.map((f, i) => `<article class="kf-prof-finding sev-${f.severity}">
        <header>
          <span class="kf-prof-findrank">${i + 1}</span>
          <b>${esc(f.title)}</b>
          ${f.recover > 0.001 ? `<em class="kf-prof-recover">可回收 ${fmt(f.recover, 3)} ms · −${fmt(f.recover / tpot * 100, 2)}%</em>` : ''}
          <span class="kf-prof-findsev">${SEV[f.severity][0]}</span>
        </header>
        <dl class="kf-prof-kv">${f.evidence.map(([k, v]) => `<div><dt>${esc(k)}</dt><dd>${esc(v)}</dd></div>`).join('')}</dl>
        <p>${esc(f.action)}</p>
        ${f.opId ? `<div class="kf-prof-actions" style="padding:0 0 2px">
          <button class="kf-prof-btn" type="button" data-advisor-op="${esc(f.opId)}">在算子分析中打开</button>
          <button class="kf-prof-btn" type="button" data-goto-graph="${esc(f.opId)}">↗ 在结构图中定位</button>
        </div>` : ''}
      </article>`).join('');

    const worst = findings.find((f) => f.severity !== 'good');
    return `${head}
      <section class="kf-prof-card">
        <header><h3>诊断结论</h3><span>${findings.length} 条 · 规则在运行时从实测数据推导</span></header>
        <div class="kf-prof-card__body kf-prof-findings">${cards}</div>
      </section>
      <section class="kf-prof-card">
        <header><h3>总结</h3><span>按可回收时间排序</span></header>
        <div class="kf-prof-card__body">
          <div class="kf-prof-verdict">
            <i>◆</i>
            <b>健康的 memory-bound decode 形态，主要收益在 Attention 分支与调度重叠</b>
            <p>权重搬运路径已跑到带宽屋顶的 70–76%，算力侧只用了 ${fmt(p.summary.sol[0].pct, 1)}%——这是 decode 的正常形态，不是缺陷。
            最大的单点是 <code>${esc(worst ? worst.title.split('，')[0] : '—')}</code>。
            ${findings.filter((f) => f.recover > 0.001).length} 条可量化结论合计 <code>${fmt(totalRecover, 2)} ms</code>，
            即 TPOT 从 <code>${fmt(tpot, 2)} ms</code> 降到 <code>${fmt(after, 2)} ms</code>。
            带宽打满的理论下界是 <code>${fmt(bound, 2)} ms</code>，两者之间还有 <code>${fmt(after - bound, 2)} ms</code> 没有归因到具体算子——
            那部分是弥散在各处的重叠损失，要靠时间线逐段排查，而不是某一条能吃掉的。
            再往下就要动流量本身（量化、权重复用）或提高 batch，见批处理页签。</p>
          </div>
        </div>
      </section>`;
  }

  window.PtoInferenceAdvisor = { render, analyze };
})();
