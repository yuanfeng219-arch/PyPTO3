/**
 * 推理性能分析 · 并行与通信（多机多卡专属页签）
 *
 * 只在 profile.dist 存在时出现。对标 Nsight Systems 多 rank 视图 + HCCL/NCCL 分析。
 * 所有数字来自 dist 块，由单卡 profile 按 TP/PP 度推导。
 */
(function registerInferenceParallel() {
  'use strict';

  const esc = (v) => String(v).replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
  const fmt = (n, d = 2) => Number(n).toFixed(d);
  const int = (n) => Number(n).toLocaleString('en-US');

  /* 拓扑图 */
  function topology(p) {
    const d = p.dist;
    const t = d.topology;
    const W = 620; const H = 240;
    const nodeW = 250; const gap = 80;
    const x0 = (W - nodeW * 2 - gap) / 2;

    const node = (i) => {
      const nx = x0 + i * (nodeW + gap);
      const stage = d.stages[i];
      const cards = Array.from({ length: t.cardsPerNode }, (_, k) => {
        const rank = i * t.cardsPerNode + k;
        const r = d.ranks[rank];
        const cx = nx + 16 + (k % 4) * 55;
        const cy = 78 + Math.floor(k / 4) * 46;
        return `<g class="pnode${r.straggler ? ' is-straggler' : ''}" data-rank="${rank}">
          <title>rank ${rank} · node ${i} · stage ${stage.id} · 利用率 ${fmt(r.utilization, 1)}%${r.straggler ? ' · straggler' : ''}</title>
          <rect x="${cx}" y="${cy}" width="46" height="36" rx="6"/>
          <rect class="fill" x="${cx}" y="${cy + 36 - Math.max(36 * r.utilization / 100, 3)}" width="46" height="${Math.max(36 * r.utilization / 100, 3)}" rx="4"/>
          <text x="${cx + 23}" y="${cy + 23}">${rank}</text>
        </g>`;
      }).join('');
      return `<g>
        <rect class="pbox" x="${nx}" y="40" width="${nodeW}" height="150" rx="10"/>
        <text class="ptitle" x="${nx + 12}" y="${32}">Node ${i} · Stage ${stage.id}</text>
        <text class="psub" x="${nx + 12}" y="${60}">layers ${esc(stage.layers)} · ${fmt(stage.weights, 2)} GB/卡</text>
        ${cards}
        <text class="phccs" x="${nx + nodeW / 2}" y="${205}" text-anchor="middle">TP=${t.tp} · HCCS ${d.links[0].peak} GB/s</text>
      </g>`;
    };

    const linkX = x0 + nodeW;
    return `<section class="kf-prof-card">
      <header><h3>拓扑</h3><span>${t.nodes} 节点 × ${t.cardsPerNode} 卡 = ${t.world} rank · 方块填充高度 = 该 rank 利用率</span></header>
      <div class="kf-prof-card__body">
        <svg class="kf-prof-topo" viewBox="0 0 ${W} ${H}" role="img" aria-label="并行拓扑">
          ${node(0)}${node(1)}
          <line class="plink" x1="${linkX}" y1="115" x2="${linkX + gap}" y2="115"/>
          <text class="plinklabel" x="${linkX + gap / 2}" y="105" text-anchor="middle">PP=${t.pp}</text>
          <text class="plinklabel dim" x="${linkX + gap / 2}" y="132" text-anchor="middle">RoCE ${d.links[1].peak} GB/s</text>
        </svg>
        <dl class="kf-prof-kv" style="margin-top:12px">
          <div><dt>并行策略</dt><dd>TP ${t.tp} × PP ${t.pp} × DP ${t.dp} = ${t.world} rank</dd></div>
          <div><dt>Batch / microbatch</dt><dd>${t.batch} = ${t.microbatches} × ${t.mbBatch}</dd></div>
          <div><dt>每 stage 层数</dt><dd>${t.stageLayers} 层</dd></div>
          <div><dt>每卡显存占用</dt><dd class="kf-prof-eff bad">${fmt(d.memory.perCard, 2)} / ${d.memory.capacity} GB · ${fmt(d.memory.perCard / d.memory.capacity * 100, 1)}%</dd></div>
        </dl>
      </div>
    </section>`;
  }

  /* 卡时构成 */
  function cardTime(p) {
    const c = p.dist.cardTime;
    const rows = [
      ['计算', c.computePct, 'cube', '真正在做矩阵与向量运算'],
      ['TP AllReduce', c.commPct, 'mte2', `节点内 ${p.dist.collectives[0].calls} 次 Ring AllReduce，未与计算重叠`],
      ['PP Send / Recv', c.p2pPct, 'mte3', '跨节点点对点，载荷小'],
      ['流水线气泡', c.bubblePct, 'vector', 'stage 之间互相空等'],
    ].map(([label, pct, unit, note]) => `<div class="kf-prof-solrow${label === '流水线气泡' ? ' is-bottleneck' : ''}" data-unit="${unit}">
        <span>${esc(label)}</span>
        <div class="kf-prof-soltrack"><div class="kf-prof-solfill" style="width:${pct}%"></div></div>
        <div class="kf-prof-solval">${fmt(pct, 1)}%</div>
        <div class="kf-prof-soldetail">${esc(note)}</div>
      </div>`).join('');
    return `<section class="kf-prof-card">
      <header><h3>卡时构成</h3><span>${fmt(c.totalMs, 1)} card-ms = ${p.dist.topology.world} 卡 × ${fmt(p.dist.tpotMs, 2)} ms</span></header>
      <div class="kf-prof-card__body">
        <div class="kf-prof-sol">${rows}</div>
        <div class="kf-prof-verdict">
          <i>◆</i>
          <b>只有 ${fmt(c.computePct, 1)}% 的卡时在做有效计算</b>
          <p>PP=${p.dist.topology.pp} 却只切了 ${p.dist.topology.microbatches} 个 microbatch，气泡比 = (p−1)/(m+p−1) = <code>${fmt((p.dist.topology.pp - 1) / (p.dist.topology.microbatches + p.dist.topology.pp - 1) * 100, 0)}%</code>，
          与实测 <code>${fmt(c.bubblePct, 1)}%</code> 吻合。TP AllReduce 另占 <code>${fmt(c.commPct, 1)}%</code> 且完全暴露在关键路径上。
          下方流水线甘特图里 stage 1 开头的空白和 stage 0 结尾的空白就是这部分。</p>
        </div>
      </div>
    </section>`;
  }

  /* 流水线甘特图 */
  function gantt(p) {
    const d = p.dist;
    const total = d.tpotMs * 1000;
    const rows = d.stages.map((st) => {
      const items = d.schedule.filter((s) => s.stage === st.id).map((s) => {
        const left = s.t0 / total * 100;
        const width = Math.max(s.dur / total * 100, 0.15);
        const label = s.kind === 'compute' ? `mb${s.mb}` : s.kind === 'p2p' ? 'P2P' : '气泡';
        const title = `${s.kind === 'compute' ? `microbatch ${s.mb} 计算` : s.kind === 'p2p' ? '跨节点 P2P' : '空等'} · ${fmt(s.dur, 1)} μs · 起 ${fmt(s.t0, 1)} μs`;
        return `<i class="is-${s.kind}" style="left:${left}%;width:${width}%" title="${esc(title)}"><b>${esc(label)}</b></i>`;
      }).join('');
      return `<div class="kf-prof-gantrow">
        <span>Stage ${st.id}<small>node ${st.node} · L${esc(st.layers)}</small></span>
        <div class="kf-prof-gantrack">${items}</div>
        <em>${fmt(st.utilization, 1)}%</em>
      </div>`;
    }).join('');
    return `<section class="kf-prof-card">
      <header><h3>流水线调度</h3><span>一个 decode step · 0 → ${fmt(total, 0)} μs · 右列为该 stage 利用率</span></header>
      <div class="kf-prof-card__body">
        <div class="kf-prof-gantt">${rows}</div>
        <div class="kf-prof-stacklegend" style="margin-top:10px">
          <span><i style="background:var(--primary)"></i>microbatch 计算（含 AllReduce）</span>
          <span><i style="background:var(--tone-blue-strong,#4a90d9)"></i>跨节点 P2P</span>
          <span><i style="background:repeating-linear-gradient(45deg,color-mix(in srgb,var(--danger) 40%,transparent) 0 4px,transparent 4px 8px)"></i>气泡</span>
        </div>
      </div>
    </section>`;
  }

  /* 集合通信统计 */
  function collectives(p) {
    const rows = p.dist.collectives.map((c) => `<tr>
        <td><b>${esc(c.op)}</b><small>${esc(c.algo)} · ${esc(c.scope)}</small></td>
        <td>${int(c.calls)}</td>
        <td>${fmt(c.payloadKb, 0)} KiB</td>
        <td>${fmt(c.busKb, 0)} KiB</td>
        <td>${fmt(c.usPerCall, 2)} μs</td>
        <td><b>${fmt(c.totalMs, 3)}</b> ms</td>
        <td>${fmt(c.totalMs / p.dist.tpotMs * 100, 1)}%</td>
        <td>${fmt(c.achievedGbs, 1)} / ${c.peakGbs} GB/s</td>
      </tr>`).join('');
    const linkRows = p.dist.links.map((l) => `<div class="kf-prof-solrow" data-unit="${l.id === 'hccs' ? 'cube' : 'mte3'}">
        <span>${esc(l.label)}</span>
        <div class="kf-prof-soltrack"><div class="kf-prof-solfill" style="width:${l.achieved / l.peak * 100}%"></div></div>
        <div class="kf-prof-solval">${fmt(l.achieved / l.peak * 100, 1)}%</div>
        <div class="kf-prof-soldetail">${fmt(l.achieved, 1)} / ${l.peak} GB/s · ${esc(l.scope)} · 承载 ${esc(l.carries)}</div>
      </div>`).join('');
    return `<section class="kf-prof-card">
      <header><h3>集合通信</h3><span>每 rank 每 step</span></header>
      <div class="kf-prof-tablewrap" style="border:0;border-radius:0">
        <table class="kf-prof-table">
          <thead><tr><th>操作</th><th>调用</th><th>载荷</th><th>总线流量</th><th>单次</th><th>合计</th><th>占 TPOT</th><th>算法带宽</th></tr></thead>
          <tbody>${rows}</tbody>
        </table>
      </div>
      <div class="kf-prof-card__body">
        <div class="kf-prof-sol">${linkRows}</div>
        <div class="kf-prof-histaxis">
          <span>AllReduce 单次载荷仅 <b>${fmt(p.dist.collectives[0].payloadKb, 0)} KiB</b>，被固定延迟主导</span>
          <span>链路带宽利用率都很低 — 瓶颈是<b>调用次数</b>而非带宽</span>
        </div>
      </div>
    </section>`;
  }

  /* rank 负载分布 */
  function rankLoad(p) {
    const d = p.dist;
    const max = Math.max(...d.ranks.map((r) => r.busyUs));
    const total = d.tpotMs * 1000;
    const bars = d.ranks.map((r) => `<div class="kf-prof-rankbar${r.straggler ? ' is-straggler' : ''}" title="rank ${r.rank} · node ${r.node} · stage ${r.stage}&#10;计算 ${fmt(r.computeUs, 1)} μs · 通信 ${fmt(r.commUs, 1)} μs · 空闲 ${fmt(r.idleUs, 1)} μs&#10;利用率 ${fmt(r.utilization, 1)}%">
        <div class="kf-prof-rankstack">
          <i class="c" style="height:${r.computeUs / total * 100}%"></i>
          <i class="m" style="height:${r.commUs / total * 100}%"></i>
          <i class="i" style="height:${r.idleUs / total * 100}%"></i>
        </div>
        <span>${r.rank}</span>
      </div>`).join('');
    const busy = d.ranks.map((r) => r.busyUs);
    const mean = busy.reduce((a, b) => a + b, 0) / busy.length;
    const cv = Math.sqrt(busy.reduce((a, b) => a + (b - mean) ** 2, 0) / busy.length) / mean;
    const str = d.ranks.find((r) => r.straggler);
    return `<section class="kf-prof-card">
      <header><h3>Rank 负载分布</h3><span>${d.ranks.length} rank · 自下而上：计算 / 通信 / 空闲</span></header>
      <div class="kf-prof-card__body">
        <div class="kf-prof-ranks">${bars}</div>
        <div class="kf-prof-histaxis">
          <span>不均衡度 CV <b>${fmt(cv, 3)}</b></span>
          <span>最慢 <b>rank ${str.rank}</b> ${fmt(str.busyUs, 0)} μs</span>
          <span>最快 <b>rank ${d.ranks.reduce((a, b) => (a.busyUs < b.busyUs ? a : b)).rank}</b> ${fmt(Math.min(...busy), 0)} μs</span>
          <span>落后 <b>${fmt((str.busyUs / mean - 1) * 100, 1)}%</b></span>
        </div>
        <div class="kf-prof-verdict" style="border-color:color-mix(in srgb,var(--warning) 36%,var(--border-subtle))">
          <i>◆</i>
          <b>rank ${str.rank} 比同组均值慢 ${fmt((str.busyUs / mean - 1) * 100, 1)}%</b>
          <p>AllReduce 是同步操作，整个 TP 组会被最慢的那张卡拖住——rank ${str.rank} 多出来的 ${fmt(str.busyUs - mean, 0)} μs 会原样传导给 node ${str.node} 上的另外 ${d.topology.cardsPerNode - 1} 张卡。
          先排查该卡的频率/温度与 HCCS 链路误码，再看是不是 KV 分片恰好落到了长序列上。</p>
        </div>
      </div>
    </section>`;
  }

  /* 扩展性 */
  function scaling(p) {
    const s = p.dist.scaling;
    const maxTps = Math.max(...s.map((x) => x.tps));
    const rows = s.map((x) => `<tr class="${x.current ? 'is-selected' : ''}">
        <td><b>${esc(x.label)}</b>${x.current ? ' <em class="kf-prof-now">当前</em>' : ''}<small>${esc(x.note)}</small></td>
        <td>${x.cards}</td>
        <td>${fmt(x.tpot, 2)} ms</td>
        <td><span class="kf-prof-share"><span class="kf-prof-sharetrack"><span class="kf-prof-sharefill" style="width:${x.tps / maxTps * 100}%"></span></span>${int(x.tps)}</span></td>
        <td>${fmt(x.speedup, 2)}×</td>
        <td class="kf-prof-eff ${x.efficiency >= 80 ? 'good' : x.efficiency >= 50 ? 'mid' : 'bad'}">${fmt(x.efficiency, 1)}%</td>
      </tr>`).join('');
    const best = s.reduce((a, b) => (a.tps > b.tps ? a : b));
    const cur = s.find((x) => x.current);
    return `<section class="kf-prof-card">
      <header><h3>扩展性对照</h3><span>同为 batch ${p.dist.topology.batch}</span></header>
      <div class="kf-prof-tablewrap" style="border:0;border-radius:0">
        <table class="kf-prof-table">
          <thead><tr><th>配置</th><th>卡数</th><th>TPOT</th><th>吞吐 tok/s</th><th>加速比</th><th>扩展效率</th></tr></thead>
          <tbody>${rows}</tbody>
        </table>
      </div>
      <div class="kf-prof-card__body">
        <div class="kf-prof-verdict" style="border-color:color-mix(in srgb,var(--danger) 40%,var(--border-subtle));background:color-mix(in srgb,var(--danger) 8%,transparent)">
          <i style="color:var(--danger)">▲</i>
          <b>${best.cards} 卡比 ${cur.cards} 卡快 ${fmt((cur.tpot / best.tpot - 1) * 100, 0)}%，硬件还少一半</b>
          <p>14B 模型 TP=${p.dist.topology.tp} 分片后每卡权重只有 ${fmt(p.dist.memory.weightsStage1, 2)} GB，加 KV 也才用掉 <code>${fmt(p.dist.memory.perCard / p.dist.memory.capacity * 100, 1)}%</code> 显存——
          <b>根本不需要跨节点</b>。PP=${p.dist.topology.pp} 没有带来任何容量收益，只带来了 <code>${fmt(p.dist.cardTime.bubblePct, 1)}%</code> 的气泡。
          去掉 PP 后扩展效率从 <code>${fmt(cur.efficiency, 1)}%</code> 回到 <code>${fmt(best.efficiency, 1)}%</code>。
          若必须保留 PP（比如为了跑更大的模型），把 microbatch 从 ${p.dist.topology.microbatches} 提到 8 可把气泡压到 <code>${fmt(1 / (8 + p.dist.topology.pp - 1) * 100, 0)}%</code>，
          代价是权重要多读几遍。</p>
        </div>
      </div>
    </section>`;
  }

  function render(p) {
    if (!p.dist) return '';
    return `<div class="kf-prof-grid2">${topology(p)}${cardTime(p)}</div>`
      + gantt(p) + collectives(p) + rankLoad(p) + scaling(p);
  }

  window.PtoInferenceParallel = { render };
})();
