/**
 * 推理性能分析 · 访存与缓存（P4）
 *
 * 对标 Nsight Compute Memory Workload + vLLM KV 面板，叠加 Ascend 片上存储层级。
 * 带宽曲线直接由 ops 的 achievedBw × 逐层时长推导，不另立数字。
 */
(function registerInferenceMemory() {
  'use strict';

  const esc = (v) => String(v).replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
  const fmt = (n, d = 2) => Number(n).toFixed(d);

  const cmp = (current, baseline, formatted, direction, c) => {
    if (baseline == null) return '<span class="kf-prof-baseline">基线无数据</span>';
    const d = c?.deltaInfo ? c.deltaInfo(current, baseline, direction || 'neutral') : { status: 'changed', label: '变化', delta: '' };
    return `<span class="kf-prof-baseline">基线 · ${formatted} <em class="so-compare-state ${d.status}">${d.label} ${d.delta}</em></span>`;
  };

  const HBM_COLOR = {
    weights: 'var(--primary)',
    kv: 'var(--warning)',
    workspace: 'var(--tone-blue-strong, #4a90d9)',
    act: 'var(--tone-green-strong, #4caf7d)',
  };

  /* HBM 占用 */
  function hbm(p, b, c) {
    const m = p.memory.hbm;
    const bm = b?.memory.hbm;
    const used = m.items.reduce((a, i) => a + i[2], 0);
    const free = m.capacity - used;
    const bar = m.items.map((i) => `<i style="width:${i[2] / m.capacity * 100}%;background:${HBM_COLOR[i[0]]}" title="${esc(i[1])} ${fmt(i[2], 2)} GB"></i>`).join('')
      + `<i style="width:${free / m.capacity * 100}%;background:var(--surface-4)" title="空闲 ${fmt(free, 2)} GB"></i>`;
    const rows = m.items.map((i) => { const bi = bm?.items.find((item) => item[0] === i[0]); return `<tr>
        <td><span class="kf-prof-swatch" style="background:${HBM_COLOR[i[0]]}"></span>${esc(i[1])}</td>
        <td><b>${fmt(i[2], 2)}</b> GB${bi ? cmp(i[2], bi[2], `${fmt(bi[2], 2)} GB`, 'lower', c) : ''}</td>
        <td>${fmt(i[2] / m.capacity * 100, 1)}%${bi ? cmp(i[2] / m.capacity, bi[2] / bm.capacity, `${fmt(bi[2] / bm.capacity * 100, 1)}%`, 'lower', c) : ''}</td>
        <td class="kf-prof-cmpnote">${esc(i[3])}</td>
      </tr>`; }).join('');
    return `<section class="kf-prof-card">
      <header><h3>HBM 占用构成</h3><span>${fmt(used, 2)} / ${m.capacity} GB · ${fmt(used / m.capacity * 100, 1)}%</span></header>
      <div class="kf-prof-card__body">${bm ? `<div class="kf-prof-series-legend"><span><i></i>当前 Run</span><span class="is-baseline"><i></i>基线</span></div>` : ''}
        <div class="kf-prof-stack" style="height:22px">${bar}</div>
        ${bm ? `<div class="kf-prof-stack is-baseline" style="height:10px;margin-top:4px">${bm.items.map((i) => `<i style="width:${i[2] / bm.capacity * 100}%;background:${HBM_COLOR[i[0]]}"></i>`).join('')}<i style="width:${(bm.capacity - bm.items.reduce((a,item)=>a+item[2],0)) / bm.capacity * 100}%;background:var(--surface-4)"></i></div>` : ''}
        <table class="kf-prof-cmp" style="margin-top:12px">
          <thead><tr><th>项</th><th>占用</th><th>占容量</th><th style="text-align:left">说明</th></tr></thead>
          <tbody>${rows}<tr><td><span class="kf-prof-swatch" style="background:var(--surface-4)"></span>空闲</td><td><b>${fmt(free, 2)}</b> GB${bm ? cmp(free, bm.capacity - bm.items.reduce((a,i)=>a+i[2],0), `${fmt(bm.capacity - bm.items.reduce((a,i)=>a+i[2],0),2)} GB`, 'higher', c) : ''}</td><td>${fmt(free / m.capacity * 100, 1)}%</td><td class="kf-prof-cmpnote">可再容纳 ${Math.floor(free / (p.memory.kv.pageBytesMb / 1000))} 个 KV 页</td></tr></tbody>
        </table>
      </div>
    </section>`;
  }

  /* 每 step 的 HBM 流量 */
  function traffic(p, b, c) {
    const t = p.summary.traffic;
    const bt = b?.summary.traffic;
    const tpot = p.summary.tpot.p50;
    const rows = [
      ['权重读入', t.weights, '每层 660.7 MB × 40 + LM Head 1.56 GB · 每 step 全量读一遍'],
      ['KV Cache 读入', t.kv, `${p.memory.kv.tokensLive.toLocaleString('en-US')} token × 160 KiB · paged 非连续`],
      ['激活与中间量', t.act, '层内 tile、累加器回写、fa 分块 partials'],
    ].map(([label, gb, note], index) => { const baseGb = bt ? [bt.weights, bt.kv, bt.act][index] : null; return `<tr>
        <td>${esc(label)}</td><td><b>${fmt(gb, 2)}</b> GB${baseGb != null ? cmp(gb,baseGb,`${fmt(baseGb,2)} GB`,'lower',c):''}</td>
        <td>${fmt(gb / t.total * 100, 1)}%${baseGb != null ? cmp(gb / t.total, baseGb / bt.total, `${fmt(baseGb / bt.total * 100,1)}%`, 'neutral', c) : ''}</td>
        <td>${fmt(gb / tpot, 2)} TB/s${baseGb != null ? cmp(gb / tpot, baseGb / b.summary.tpot.p50, `${fmt(baseGb / b.summary.tpot.p50,2)} TB/s`, 'higher', c) : ''}</td>
        <td class="kf-prof-cmpnote">${esc(note)}</td>
      </tr>`; }).join('');
    const achieved = t.total / tpot;
    const achievedPct = achieved / p.meta.peakBw * 100;
    const baseAchieved = bt ? bt.total / b.summary.tpot.p50 : null;
    const baseAchievedPct = bt ? baseAchieved / b.meta.peakBw * 100 : null;
    return `<section class="kf-prof-card">
      <header><h3>每 step HBM 流量</h3><span>${fmt(t.total, 2)} GB ÷ ${fmt(tpot, 1)} ms = ${fmt(achieved, 2)} TB/s</span></header>
      <div class="kf-prof-card__body">
        <table class="kf-prof-cmp">
          <thead><tr><th>来源</th><th>字节</th><th>占比</th><th>等效带宽</th><th style="text-align:left">说明</th></tr></thead>
          <tbody>${rows}</tbody>
        </table>
        <div class="kf-prof-sol" style="margin-top:14px">
          <div class="kf-prof-solrow is-bottleneck" data-unit="mte2">
            <span>达成带宽</span>
            <div class="kf-prof-soltrack${bt ? ' is-grouped' : ''}"><div class="kf-prof-solfill" style="width:${achievedPct}%"></div>${bt ? `<div class="kf-prof-solfill is-baseline" style="width:${baseAchievedPct}%"></div>` : ''}</div>
            <div class="kf-prof-solval">${fmt(achievedPct, 1)}%${bt ? cmp(achievedPct, baseAchievedPct, `${fmt(baseAchievedPct,1)}%`, 'higher', c) : ''}</div>
            <div class="kf-prof-soldetail">${fmt(achieved, 2)} / ${fmt(p.meta.peakBw, 1)} TB/s 峰值 · MTE2 占空比 ${fmt(p.summary.sol[2].pct, 1)}%</div>
          </div>
        </div>
      </div>
    </section>`;
  }

  /* 片上存储层级 */
  function onchip(p, b, c) {
    const rows = p.memory.onchip.map(([label, pct, budget, note]) => {
      const baseRow = b?.memory.onchip.find((item) => item[0] === label);
      const over = budget !== null && pct > budget;
      return `<div class="kf-prof-solrow${over ? ' is-bottleneck' : ''}" data-unit="${over ? 'mte2' : 'cube'}">
        <span>${esc(label)}</span>
        <div class="kf-prof-soltrack${baseRow ? ' is-grouped' : ''}">
          <div class="kf-prof-solfill" style="width:${pct}%"></div>
          ${baseRow ? `<div class="kf-prof-solfill is-baseline" style="width:${baseRow[1]}%"></div>` : ''}
          ${budget !== null ? `<i class="kf-prof-budget" style="left:${budget}%" title="编译期预算 ${budget}%"></i>` : ''}
        </div>
        <div class="kf-prof-solval">${pct}%${baseRow ? cmp(pct, baseRow[1], `${fmt(baseRow[1],1)}%`, 'higher', c) : ''}</div>
        <div class="kf-prof-soldetail">${esc(note)}</div>
      </div>`;
    }).join('');
    return `<section class="kf-prof-card">
      <header><h3>片上存储层级</h3><span>竖线 = 编译期预算</span></header>
      <div class="kf-prof-card__body"><div class="kf-prof-sol">${rows}</div></div>
    </section>`;
  }

  /* 带宽曲线：由每个任务的 achievedBw × 该层实际时长铺出来 */
  function bandwidth(p, b, c) {
    const layer = 12;
    const chain = window.PtoInferenceTimeline?.LAYER_CHAIN || [];
    const byId = Object.fromEntries(p.ops.map((o) => [o.id, o]));
    const segs = chain.map((id) => {
      const op = byId[id];
      return { name: op.name, dur: op.perLayer ? op.perLayer[layer] : op.perLayerUs, bw: op.achievedBw, group: op.group };
    });
    const total = segs.reduce((a, s) => a + s.dur, 0);
    const peak = p.meta.peakBw;
    const H = 120;
    let x = 0;
    const bars = segs.map((s) => {
      const w = s.dur / total * 100;
      const h = Math.max(s.bw / peak * 100, 0.6);
      const el = `<i style="left:${x}%;width:${w}%;height:${h}%" class="${s.bw / peak > 0.65 ? 'is-high' : s.bw / peak > 0.3 ? 'is-mid' : 'is-low'}" title="${esc(s.name)} · ${fmt(s.bw, 2)} TB/s · ${fmt(s.dur, 1)} μs"></i>`;
      x += w;
      return el;
    }).join('');
    let bx = 0;
    const baseById = b ? Object.fromEntries(b.ops.map((o) => [o.id, o])) : {};
    const baseSegs = b ? chain.map((id) => { const op = baseById[id]; return { name: op.name, dur: op.perLayer ? op.perLayer[layer] : op.perLayerUs, bw: op.achievedBw }; }) : [];
    const baseTotal = baseSegs.reduce((a, s) => a + s.dur, 0);
    const baseBars = baseSegs.map((s) => { const w = s.dur / baseTotal * 100; const h = Math.max(s.bw / b.meta.peakBw * 100, .6); const el = `<i class="is-baseline" style="left:${bx}%;width:${w}%;height:${h}%" title="基线 · ${esc(s.name)} · ${fmt(s.bw,2)} TB/s"></i>`; bx += w; return el; }).join('');
    return `<section class="kf-prof-card">
      <header><h3>层内 MTE2 带宽曲线</h3><span>L${layer} · 每段高度 = 该任务达成带宽 / ${fmt(peak, 1)} TB/s 峰值</span></header>
      <div class="kf-prof-card__body">
        <div class="kf-prof-bwchart" style="height:${H}px">
          <i class="kf-prof-bwroof" style="bottom:100%"><b>峰值 ${fmt(peak, 1)} TB/s</b></i>
          <i class="kf-prof-bwavg" style="bottom:${p.summary.traffic.total / p.summary.tpot.p50 / peak * 100}%"><b>整步均值 ${fmt(p.summary.traffic.total / p.summary.tpot.p50, 2)}</b></i>
          ${bars}${baseBars}
        </div>
        <div class="kf-prof-histaxis">
          <span>${b ? '当前 / 基线共用' : '当前 Run 使用'} ${fmt(peak,1)} TB/s 绝对刻度</span>
          <span>fa_fused 段掉到 <b>51.7%</b></span>
          <span>${b ? cmp(p.summary.traffic.total / p.summary.tpot.p50, b.summary.traffic.total / b.summary.tpot.p50, `${fmt(b.summary.traffic.total / b.summary.tpot.p50,2)} TB/s`, 'higher', c) : ''}</span>
        </div>
      </div>
    </section>`;
  }

  /* Paged KV */
  function paged(p, b, c) {
    const kv = p.memory.kv;
    const bkv = b?.memory.kv;
    const palette = ['var(--primary)', 'var(--warning)', 'var(--tone-blue-strong, #4a90d9)', 'var(--tone-green-strong, #4caf7d)'];
    let cells = '';
    kv.perRequest.forEach((r, i) => {
      for (let k = 0; k < r.pages; k += 1) {
        cells += `<i style="background:${palette[i % palette.length]};opacity:${0.55 + (i % 4) * 0.15}" title="${esc(r.req)} · 第 ${k + 1}/${r.pages} 页 · seq ${r.seq}"></i>`;
      }
    });
    for (let k = kv.pagesUsed; k < kv.pagesTotal; k += 1) cells += '<i class="is-free" title="空闲页"></i>';

    const maxSeq = Math.max(...kv.perRequest.map((r) => r.seq));
    const seqRows = kv.perRequest.map((r) => `<div class="kf-prof-seqrow">
        <span>${esc(r.req)}</span>
        <div class="kf-prof-seqtrack"><i style="width:${r.seq / maxSeq * 100}%"></i><u style="width:${(r.pages * kv.pageTokens - r.seq) / maxSeq * 100}%"></u></div>
        <b>${r.seq.toLocaleString('en-US')}</b><em>${r.pages} 页</em>
      </div>`).join('');

    return `<div class="kf-prof-grid2">
      <section class="kf-prof-card">
        <header><h3>页池分配位图</h3><span>${kv.pagesUsed} / ${kv.pagesTotal} 页 · 每页 ${kv.pageTokens} token / ${fmt(kv.pageBytesMb, 2)} MB</span></header>
        <div class="kf-prof-card__body">
          <div class="kf-prof-pagegrid">${cells}</div>
          <dl class="kf-prof-kv" style="margin-top:12px">
            <div><dt>已分配</dt><dd>${fmt(kv.bytesAllocated, 2)} / ${fmt(kv.bytesPool, 2)} GB · ${fmt(kv.utilization, 1)}%${bkv ? cmp(kv.bytesAllocated,bkv.bytesAllocated,`${fmt(bkv.bytesAllocated,2)} GB · ${fmt(bkv.utilization,1)}%`,'lower',c):''}</dd></div>
            <div><dt>活跃 token</dt><dd>${kv.tokensLive.toLocaleString('en-US')} / ${kv.tokensAllocated.toLocaleString('en-US')}${bkv ? cmp(kv.tokensLive,bkv.tokensLive,`${bkv.tokensLive.toLocaleString('en-US')} / ${bkv.tokensAllocated.toLocaleString('en-US')}`,'neutral',c):''}</dd></div>
            <div><dt>内部碎片</dt><dd class="kf-prof-eff good">${fmt(kv.fragmentation, 2)}%${bkv ? cmp(kv.fragmentation,bkv.fragmentation,`${fmt(bkv.fragmentation,2)}%`,'lower',c):''}</dd></div>
            <div><dt>命中率</dt><dd>${fmt(kv.hitRate, 1)}%${bkv ? cmp(kv.hitRate,bkv.hitRate,`${fmt(bkv.hitRate,1)}%`,'higher',c):''}</dd></div>
            <div><dt>抢占 / 换出</dt><dd>${kv.preempt} / ${kv.swap}${bkv ? cmp(kv.preempt,bkv.preempt,`${bkv.preempt} / ${bkv.swap}`,'lower',c):''}</dd></div>
          </dl>
        </div>
      </section>
      <section class="kf-prof-card">
        <header><h3>每请求 seq_len 与页占用</h3><span>实线 = 活跃 token · 空心 = 页内填充</span></header>
        <div class="kf-prof-card__body"><div class="kf-prof-seqlist">${seqRows}</div></div>
      </section>
    </div>`;
  }

  /* work table 稠密率 —— fa_work_build 的价值证明 */
  function workTable(p, b, c) {
    const kv = p.memory.kv;
    const cells = Array.from({ length: kv.blocksPadded }, (_, i) => `<i class="${i < kv.blocksReal ? 'is-real' : 'is-pad'}"></i>`).join('');
    return `<section class="kf-prof-card">
      <header><h3>fa_work_table 稠密率</h3><span>MCB 静态上界 ${kv.mcb} × ${p.meta.batch} 请求 = ${kv.blocksPadded} 块</span></header>
      <div class="kf-prof-card__body">
        <div class="kf-prof-blockgrid">${cells}</div>
        <div class="kf-prof-verdict" style="border-color:color-mix(in srgb,var(--success) 36%,var(--border-subtle));background:color-mix(in srgb,var(--success) 8%,transparent)">
          <i style="color:var(--success)">✓</i>
          <b>稠密率 ${fmt(kv.density, 1)}% · 压掉 ${kv.blocksPadded - kv.blocksReal} 个空块${b ? cmp(kv.density,b.memory.kv.density,`${fmt(b.memory.kv.density,1)}%`,'higher',c):''}</b>
          <p>编译期只能按 <code>MCB = ${kv.mcb}</code> 的静态上界分配 ${kv.blocksPadded} 块；<code>fa_work_build</code> 花 2.2 μs/层 读 seq_lens 把 ragged 请求压紧成 ${kv.blocksReal} 个真实块，
          为 <code>fa_fused</code> 省掉 ${fmt((1 - kv.density / 100) * 100, 1)}% 的空块迭代。这是静态推断拿不到、只有实测才能记账的收益。</p>
        </div>
      </div>
    </section>`;
  }

  /* 精度边界 */
  function precision(p, b, c) {
    const rows = p.memory.precision.map(([label, where, count, tone]) => { const br = b?.memory.precision.find((item)=>item[0]===label); return `<tr>
        <td>${esc(label)}</td><td>${esc(where)}</td>
        <td>${count === null ? '—' : `<b>${count}</b> 次`}${br ? cmp(count,br[2],br[2] == null ? '—' : `${br[2]} 次`,'lower',c):''}</td>
        <td class="delta ${tone}">${count === 0 ? '已消除' : '符合预期'}</td>
      </tr>`; }).join('');
    return `<section class="kf-prof-card">
      <header><h3>精度边界记账</h3><span>层内零转换是 FP32 carry 策略的直接收益</span></header>
      <div class="kf-prof-card__body"><table class="kf-prof-cmp">
        <thead><tr><th>转换点</th><th>位置</th><th>次数</th><th>结论</th></tr></thead>
        <tbody>${rows}</tbody>
      </table></div>
    </section>`;
  }

  function render(p, b, c) {
    return `<div class="kf-prof-grid2">${hbm(p,b,c)}${onchip(p,b,c)}</div>`
      + traffic(p,b,c)
      + bandwidth(p,b,c)
      + paged(p,b,c)
      + `<div class="kf-prof-grid2">${workTable(p,b,c)}${precision(p,b,c)}</div>`;
  }

  window.PtoInferenceMemory = { render };
})();
