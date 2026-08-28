/**
 * 推理性能分析 · 批处理与调度（P5）
 *
 * 对标 vLLM / TensorRT-LLM 服务面板。batch 扫描表由权重/KV/激活流量推导，
 * B=16 行与 summary 的 TPOT、流量、吞吐完全一致。
 */
(function registerInferenceServing() {
  'use strict';

  const esc = (v) => String(v).replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
  const fmt = (n, d = 2) => Number(n).toFixed(d);
  const int = (n) => Number(n).toLocaleString('en-US');

  /* 队列与拆分 */
  function queue(p) {
    const s = p.serving;
    const q = s.queue;
    const cards = [
      ['运行中', q.running, `槽位上限 ${p.meta.batch}`],
      ['等待中', q.waiting, `等待 p50 ${q.waitP50} ms · p99 ${q.waitP99} ms`],
      ['抢占', q.preempt, q.preempt === 0 ? 'KV 池未打满，无需抢占' : '需扩容 KV 池'],
      ['重计算', q.recompute, '无换出后重算'],
      ['Chunked prefill', q.chunkedPrefill, '长 prefill 被切片让出算力'],
      ['窗口内请求', s.totalRequests, `${s.lanes.length} 个槽位复用`],
    ];
    return `<div class="kf-prof-kpis">${cards.map(([label, value, sub]) => `
      <article class="kf-prof-kpi"><span>${esc(label)}</span><b>${int(value)}</b><small><u style="text-decoration:none">${esc(sub)}</u></small></article>`).join('')}</div>`;
  }

  /* batch 随时间波动 */
  function batchCurve(p) {
    const s = p.serving;
    const arr = s.batchOverTime;
    const min = Math.min(...arr);
    const max = Math.max(...arr);
    const cap = p.meta.batch;
    const bars = arr.map((v, i) => `<i style="height:${v / cap * 100}%" class="${v === cap ? 'is-full' : ''}" title="t≈${fmt(i / arr.length * s.windowMs / 1000, 2)} s · batch ${v}"></i>`).join('');
    return `<section class="kf-prof-card">
      <header><h3>Batch size 随时间</h3><span>${int(s.steps)} steps / ${fmt(s.windowMs / 1000, 2)} s · 采样 ${arr.length} 点</span></header>
      <div class="kf-prof-card__body">
        <div class="kf-prof-batchchart">${bars}</div>
        <div class="kf-prof-histaxis">
          <span>区间 <b>${min} – ${max}</b></span>
          <span>平均 <b>${fmt(s.batchAvg, 2)}</b></span>
          <span>槽位利用率 <b>${fmt(s.batchAvg / cap * 100, 1)}%</b></span>
          <span>Prefill / Decode <b>${fmt(s.split.prefill, 1)}% / ${fmt(s.split.decode, 1)}%</b></span>
        </div>
      </div>
    </section>`;
  }

  /* 请求生命周期泳道 */
  function lanes(p) {
    const s = p.serving;
    const W = s.windowMs;
    const rows = s.lanes.map((lane) => {
      const items = lane.items.map((it) => {
        const seg = (off, dur, cls, label) => {
          if (dur <= 0) return '';
          const left = (it.t0 + off) / W * 100;
          const width = Math.max(dur / W * 100, 0.06);
          if (left > 100) return '';
          return `<i class="${cls}" style="left:${left}%;width:${Math.min(width, 100 - left)}%" title="${esc(`${it.id} · ${label} ${fmt(dur, 0)} ms`)}"></i>`;
        };
        return seg(0, it.wait, 'is-wait', '排队')
          + seg(it.wait, it.prefill, 'is-prefill', 'prefill')
          + seg(it.wait + it.prefill, it.decode, 'is-decode', 'decode');
      }).join('');
      return `<div class="kf-prof-lane"><span>slot ${String(lane.slot).padStart(2, '0')}</span><div class="kf-prof-lanetrack">${items}</div></div>`;
    }).join('');
    return `<section class="kf-prof-card">
      <header><h3>请求生命周期</h3><span>${s.totalRequests} 个请求在 ${s.lanes.length} 个槽位上滚动复用</span></header>
      <div class="kf-prof-card__body">
        <div class="kf-prof-lanes">${rows}</div>
        <div class="kf-prof-stacklegend" style="margin-top:10px">
          <span><i style="background:color-mix(in srgb,var(--foreground) 30%,transparent)"></i>排队</span>
          <span><i style="background:var(--tone-blue-strong,#4a90d9)"></i>Prefill</span>
          <span><i style="background:var(--primary)"></i>Decode</span>
          <span style="margin-left:auto">0 → ${fmt(W / 1000, 2)} s</span>
        </div>
      </div>
    </section>`;
  }

  /* batch 扫描 —— 容量规划最常引用的一屏 */
  function sweep(p) {
    const s = p.serving.sweep;
    const maxTps = Math.max(...s.map((r) => r.tps));
    const maxTpot = Math.max(...s.map((r) => r.tpot));
    const rows = s.map((r) => `<tr class="${r.current ? 'is-selected' : ''}">
        <td><b>${r.batch}</b>${r.current ? ' <em class="kf-prof-now">当前</em>' : ''}</td>
        <td>${fmt(r.traffic, 2)} GB</td>
        <td>${fmt(r.perToken, 2)} GB</td>
        <td>${fmt(r.bw, 2)} TB/s</td>
        <td><span class="kf-prof-share"><span class="kf-prof-sharetrack"><span class="kf-prof-sharefill" style="width:${r.tpot / maxTpot * 100}%;background:var(--warning)"></span></span>${fmt(r.tpot, 2)} ms</span></td>
        <td><span class="kf-prof-share"><span class="kf-prof-sharetrack"><span class="kf-prof-sharefill" style="width:${r.tps / maxTps * 100}%"></span></span>${int(r.tps)}</span></td>
        <td>${fmt(r.mte2, 1)}%</td>
      </tr>`).join('');
    const cur = s.find((r) => r.current) || s.find((r) => r.batch === p.meta.batch) || s[0];
    const big = s[s.length - 1];
    return `<section class="kf-prof-card">
      <header><h3>Batch 扫描</h3><span>权重流量恒定 27.99 GB，KV 与激活随 batch 线性增长</span></header>
      <div class="kf-prof-tablewrap" style="border:0;border-radius:0">
        <table class="kf-prof-table">
          <thead><tr><th>Batch</th><th>每 step 流量</th><th>每 token 流量</th><th>达成带宽</th><th>TPOT</th><th>吞吐 tok/s</th><th>MTE2 占空比</th></tr></thead>
          <tbody>${rows}</tbody>
        </table>
      </div>
      <div class="kf-prof-card__body">
        <div class="kf-prof-verdict">
          <i>◆</i>
          <b>提吞吐靠增大 batch，代价是单 token 时延</b>
          <p>权重那 27.99 GB 每 step 都要读一遍，与 batch 无关——所以 batch 从 ${cur.batch} 提到 ${big.batch} 时，
          每 token 流量从 <code>${fmt(cur.perToken, 2)} GB</code> 降到 <code>${fmt(big.perToken, 2)} GB</code>，吞吐涨 <code>${fmt(big.tps / cur.tps, 2)}×</code>，
          但 TPOT 从 <code>${fmt(cur.tpot, 2)} ms</code> 升到 <code>${fmt(big.tpot, 2)} ms</code>（<code>+${fmt((big.tpot / cur.tpot - 1) * 100, 0)}%</code>）。
          当前 batch ${cur.batch} 的槽位利用率 ${fmt(p.serving.batchAvg / p.meta.batch * 100, 1)}%、KV 池只用了 ${fmt(p.memory.kv.utilization, 1)}%、抢占 0 次——
          <b>还有余量往上走</b>，上限由 SLA 里能接受的 TPOT 决定，而不是显存。</p>
        </div>
      </div>
    </section>`;
  }

  function render(p) {
    return queue(p) + batchCurve(p) + lanes(p) + sweep(p);
  }

  window.PtoInferenceServing = { render };
})();
