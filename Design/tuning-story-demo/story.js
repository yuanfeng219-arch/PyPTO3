/* Tuning Story — 场景渲染与播放。
 * 页面只负责领域数据与内容；shell、播放条、泳道任务条都调用设计系统的共享 pattern。 */
(() => {
  'use strict';

  const D = window.TS_DATA;
  const $ = (id) => document.getElementById(id);
  const esc = (s) => String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
  const num = (v, d = 0) => Number(v).toLocaleString('en-US', { minimumFractionDigits: d, maximumFractionDigits: d });

  const state = { scene: 0, view: 0, playing: false, timer: null };
  const LANE_COUNT = 6;   // 泳道总览只列最忙的几条；播放条是浮层，行数过多会被遮住

  /* =============================================================== 视图渲染 */
  const V = {
    /* ---- T1 基线卡 ---- */
    baseline() {
      const b = D.detail.baseline;
      return `
        ${kpis([
          ['Total Test Time', `${b.total} µs`, '主指标（待用户确认）'],
          ['Total Tasks', b.tasks, `Exec / Latency ${b.ratio}%`],
          ['Tail OH', `${b.tailOh} µs`, `占 latency ${b.tailPct}%`],
          ['Head OH', `${b.headOh} µs`, '占 latency 2.0%'],
        ])}
        <div class="ts-section">
          <h3 class="ts-section-title">Kernel 排行（基线快照）</h3>
          <table class="ts-table">
            <thead><tr><th>Kernel</th><th class="num">次数</th><th class="num">平均 Exec</th><th class="num">平均 Latency</th><th class="num">Exec%</th><th>备注</th></tr></thead>
            <tbody>${b.kernels.map((k) => `
              <tr${/热点|细碎/.test(k.note) ? ' class="is-focus"' : ''}>
                <td class="name">${esc(k.name)}</td>
                <td class="num">${k.n}</td><td class="num">${k.exec}</td><td class="num">${k.lat}</td><td class="num">${k.pct}%</td>
                <td>${esc(k.note)}</td>
              </tr>`).join('')}</tbody>
            <caption>来源 pypto-lib#314，commit 8f88c8f，a2a3，B=64 / S=1。这张表当前是手抄的，与原始数据之间没有链接。</caption>
          </table>
        </div>
        <div class="ts-section">
          <div class="ts-callout warn">
            <span class="ts-callout-k">测量协议（现状：写在 Issue 正文里，工具不执行）</span>
            每次改动后跑 <code>--enable-l2-swimlane</code>，以 Total Test Time 为主指标；精度不过或编译失败时记录阻塞原因并回退。
          </div>
        </div>`;
    },

    /* ---- T1 调用次序 ---- */
    runs() {
      const max = Math.max(...D.runs.map((r) => r.deviceWall));
      return `
        <p class="ts-lead">同一次运行里，首次调用与稳态调用不是一回事。这一屏的数据来自 <code>host.*.log</code> 的 STRACE span，当前界面上完全不可见。</p>
        <div class="ts-section">
          <h3 class="ts-section-title">device_wall × 调用次序</h3>
          ${D.runs.map((r) => `
            <div class="ts-bar-row">
              <span class="ts-bar-k">${r.rank} · inv=${r.inv}</span>
              <span class="ts-bar"><i class="${r.inv === 1 && r.deviceWall > 10 ? 'low' : 'high'}" style="width:${(r.deviceWall / max * 100).toFixed(1)}%"></i></span>
              <span class="ts-bar-v">${r.deviceWall} ms</span>
            </div>`).join('')}
          <p class="ts-section-note">rank0 首次 45.19 ms 是稳态 5.13 ms 的约 9 倍；但 rank1 首次并不慢，两个 rank 的稳态值也相差约 28%。</p>
        </div>
        <div class="ts-section">
          <div class="ts-callout bad">
            <span class="ts-callout-k">可信度风险</span>
            不区分调用次序、不区分 rank 的数字不可比，而且「冷启动是否出现」本身也不稳定。P0：基线默认排除首次调用，并显示 rank 间差异。
          </div>
        </div>`;
    },

    /* ---- T2 scope 排行 ---- */
    scopeRank(scene) {
      return `
        ${kpis([
          ['整图 wall', `${D.meta.wall} µs`, `${D.meta.scopeCount} 个 scope`],
          ['AIC 占用', `${D.meta.utilAic}%`, '24 核平均'],
          ['AIV 占用', `${D.meta.utilAiv}%`, '48 核平均'],
          ['AICore 任务记录', num(D.meta.taskRecords), 'Worker View'],
        ])}
        <div class="ts-section">
          <h3 class="ts-section-title">按 core-time 排行</h3>
          <table class="ts-table">
            <thead><tr><th>#</th><th>Scope</th><th class="num">task</th><th class="num">Σdur</th><th class="num">占比</th><th class="num">平均</th><th class="num">核数</th><th class="num">起点</th><th class="num">跨度</th></tr></thead>
            <tbody>${D.scopes.map((s, i) => `
              <tr${s.name === scene.focus ? ' class="is-focus"' : ''}>
                <td class="num">${s.rank || i + 1}</td>
                <td class="name">${esc(s.name)}</td>
                <td class="num">${s.n}</td>
                <td class="num">${num(s.sum)}</td>
                <td class="num">${s.pct}%</td>
                <td class="num">${s.avg}</td>
                <td class="num">${s.cores}</td>
                <td class="num">${num(s.start)}</td>
                <td class="num">${num(s.span)}</td>
              </tr>`).join('')}</tbody>
            <caption>单位 µs。来源：本地 merged_swimlane，只统计 Worker View 的 AIC_* / AIV_* 轨道，避免与 Scheduler View 重复计数。</caption>
          </table>
        </div>
        <div class="ts-section">
          <div class="ts-callout">
            <span class="ts-callout-k">读法</span>
            <code>kv_score_proj</code> 平均每个 task 只有 11.0 µs 却切成 512 个，跨度 764 µs——先看派发与停顿，而不是先看计算。
            排名第 14 的 <code>scatter_softmax_pool</code> 是 6 月调过的 scope：<b>热点会随版本迁移，排名必须与版本绑定。</b>
          </div>
        </div>
        <p class="ts-section-note">缺口（P0）：这张表还差一列 <b>关键路径 slack</b>。只看 Σdur 会重复 §18 的误判——把一个不在关键路径上的 scope 当成首要目标。</p>`;
    },

    /* ---- T2 占用率窗口 ---- */
    windows() {
      const cls = (v) => (v < 30 ? 'low' : v < 60 ? 'mid' : 'high');
      return `
        <p class="ts-lead">把 wall 切成 500 µs 的窗口，逐段统计核占用率。低于约 30% 就不是计算瓶颈，而是停顿或串行。</p>
        <div class="ts-section">
          <h3 class="ts-section-title">AIC 占用率</h3>
          ${D.windows.map((w) => `
            <div class="ts-bar-row">
              <span class="ts-bar-k">${w.from}–${w.to}</span>
              <span class="ts-bar"><i class="${cls(w.aic)}" style="width:${w.aic}%"></i></span>
              <span class="ts-bar-v">${w.aic}%</span>
            </div>`).join('')}
        </div>
        <div class="ts-section">
          <h3 class="ts-section-title">AIV 占用率</h3>
          ${D.windows.map((w) => `
            <div class="ts-bar-row">
              <span class="ts-bar-k">${w.from}–${w.to}</span>
              <span class="ts-bar"><i class="${cls(w.aiv)}" style="width:${w.aiv}%"></i></span>
              <span class="ts-bar-v">${w.aiv}%</span>
            </div>`).join('')}
          <p class="ts-section-note">1000–1500 µs 这段两侧都很低（AIC 15% / AIV 12%），正是 <code>kv_score_proj</code> 512 个碎 task 铺开的位置。</p>
        </div>`;
    },

    /* ---- T3 调度诊断 ---- */
    schedDiag() {
      const s = D.detail.sched;
      return `
        <div class="ts-section">
          <div class="ts-callout bad">
            <span class="ts-callout-k">诊断结论（P0：应由工具一句话给出，而不是让用户自己拼三份数据）</span>
            这段窗口是<b>依赖串行型</b>，不是计算瓶颈：task 数以百计，却只落在 1–6 个核上；根因是循环体内对共享 GM 句柄反复重新赋值，形成跨迭代写依赖链。
          </div>
        </div>
        <div class="ts-section">
          <h3 class="ts-section-title">判定依据</h3>
          <table class="ts-table">
            <thead><tr><th>指标</th><th class="num">读数</th><th>说明</th></tr></thead>
            <tbody>${s.rows.map((r) => `
              <tr><td>${esc(r.k)}</td>
                  <td class="num"><span class="ts-verdict ${r.s === 'bad' ? 'bad' : 'warn'}">${esc(r.v)}</span></td>
                  <td>${esc(r.d)}</td></tr>`).join('')}</tbody>
            <caption>前两行来自 pypto-lib#314，中间三行来自调优日志 §8，最后一行来自本地 chip_swimlane_records 的 aicpu_scheduler_phases。</caption>
          </table>
        </div>
        <div class="ts-section">
          <h3 class="ts-section-title">依赖图（本地 deps.json）</h3>
          ${kpis([
            ['task', D.meta.deps.tasks, ''],
            ['张量', D.meta.deps.tensors, ''],
            ['依赖边', D.meta.deps.edges, 'source=creator / explicit'],
          ])}
          <p class="ts-section-note">每条边带 <code>flags=[wait, retain]</code> 与 <code>tensor_id</code>，已经足够回答「这个核在等谁」——但当前要在 deps_viewer 和 Perfetto 之间人工对照。</p>
        </div>`;
    },

    /* ---- T3 三次拆分实验 ---- */
    passSplit() {
      const s = D.detail.sched;
      const max = Math.max(...s.splits.map((x) => x.total));
      return `
        <p class="ts-lead">假设要靠实验证伪。三次真机拆分之后才确认：scatter 必须和 softmax 留在同一个 pass，单独成 pass 会塌成 1 核。</p>
        <div class="ts-section">
          ${s.splits.map((x) => `
            <div class="ts-bar-row">
              <span class="ts-bar-k">${esc(x.name)}</span>
              <span class="ts-bar"><i class="${x.ok ? 'high' : 'low'}" style="width:${(x.total / max * 100).toFixed(1)}%"></i></span>
              <span class="ts-bar-v">${num(x.total)} µs</span>
            </div>
            <p class="ts-section-note" style="margin:2px 0 8px 100px">${esc(x.cores)} · ${esc(x.note)}</p>`).join('')}
        </div>
        <div class="ts-section">
          <div class="ts-callout warn">
            <span class="ts-callout-k">被推翻的假设（应当保留在记录里）</span>
            另一处 scope 曾被判断为 HBM 带宽瓶颈。减少重读后单 task 确实更快，但 Total 从 582 µs 退化到 621 µs——带宽假设不成立，真因是依赖与调度停顿。
          </div>
        </div>`;
    },

    /* ---- T4 源码改动 ---- */
    sourceDiff() {
      const s = D.detail.source;
      const mark = (lines, idx) => lines.map((l, i) => (i === idx ? `<span class="hl">${esc(l)}</span>` : esc(l))).join('\n');
      return `
        <p class="ts-lead">根因落到一行缩进：<code>${esc(s.file)}</code></p>
        <div class="ts-section ts-diff">
          <div>
            <div class="ts-diff-head"><span class="k">改前</span><span class="v ts-verdict bad">5 核 / 512 task</span></div>
            <pre class="ts-code">${mark(s.before, 0)}</pre>
            <p class="ts-section-note">${esc(s.beforeNote)}</p>
          </div>
          <div>
            <div class="ts-diff-head"><span class="k">改后</span><span class="v ts-verdict ok">42 核 / 128 task</span></div>
            <pre class="ts-code">${mark(s.after, 0)}</pre>
            <p class="ts-section-note">${esc(s.afterNote)}</p>
          </div>
        </div>
        <div class="ts-section">
          <table class="ts-table">
            <thead><tr><th>指标</th><th class="num">改前</th><th class="num">改后</th><th class="num">变化</th></tr></thead>
            <tbody>${s.metrics.map((m) => `
              <tr><td>${esc(m.k)}</td><td class="num">${esc(m.before)}</td><td class="num">${esc(m.after)}</td>
                  <td class="num"><span class="ts-verdict ok">${esc(m.delta)}</span></td></tr>`).join('')}</tbody>
            <caption>来源：调优日志 §8，两次运行 1039.7 / 1012.0 µs，uniform 与 hetero start_pos 两种输入精度全部通过。</caption>
          </table>
        </div>`;
    },

    /* ---- T4 反直觉：拆开融合 ---- */
    unmix() {
      return `
        <p class="ts-lead">融合并不总是更快。当被融合的 vector 收尾与关键路径上的另一个 vector scope 抢同一批 AIV 核，而本 scope 的产物又有大量下游余量时，<b>拆开反而更快</b>。</p>
        <div class="ts-section ts-diff">
          <div>
            <div class="ts-diff-head"><span class="k">MIX（原融合）</span><span class="v ts-verdict bad">2222.5 µs</span></div>
            <pre class="ts-code">for hg_idx in pl.spmd(..., name_hint="qproj"):
    for h_inner in pl.pipeline(16, stage=2):
        for qb in pl.pipeline(...):
            <span class="cm"># matmul -> col_acc (cube)</span>
        for tc in pl.pipeline(...):
            <span class="hl"># dequant (vec) 紧贴 matmul，钉在本窗口</span></pre>
            <p class="ts-section-note">dequant 被钉在 [300–420 µs] 窗口，抢走关键路径 <code>qr_proj_aiv</code> 需要的 AIV 核。</p>
          </div>
          <div>
            <div class="ts-diff-head"><span class="k">UN-MIX（采用）</span><span class="v ts-verdict ok">2153.4 µs</span></div>
            <pre class="ts-code">for hg_idx in pl.spmd(..., name_hint="qproj_matmul"):
    <span class="cm"># 纯 matmul，cube -> INT32 GM</span>

for hg_idx in pl.spmd(..., name_hint="qproj_dequant"):
    <span class="hl"># 独立 scope：调度器推迟到 AIV 空闲的 [485–566 µs]</span></pre>
            <p class="ts-section-note"><code>qr_proj_aiv</code> 拿满 48 核，442 µs → 369 µs 提前完成。代价是多一个 INT32 GM 中转。</p>
          </div>
        </div>
        <div class="ts-section">
          <div class="ts-callout">
            <span class="ts-callout-k">这条判断需要什么信息</span>
            要知道「产物离消费点有多远（slack）」和「vec 部分是否与关键路径抢核」——两者都依赖 T2 的关键路径能力，当前工具没有。
          </div>
        </div>`;
    },

    /* ---- T5 意图核对 ---- */
    intent() {
      const label = { yes: '已生效', partial: '部分生效', no: '未生效' };
      return `
        <p class="ts-lead">用户写下的意图，不一定被编译器执行。这一屏是本场景最独特的价值：<b>请求值、实际值、证据位置</b>并排。</p>
        <div class="ts-section">
          <table class="ts-table">
            <thead><tr><th>意图</th><th>请求值</th><th>实际值</th><th>状态</th><th>证据</th></tr></thead>
            <tbody>${D.detail.intents.map((i) => `
              <tr${i.ok !== 'yes' ? ' class="is-focus"' : ''}>
                <td>${esc(i.intent)}</td>
                <td class="name">${esc(i.req)}</td>
                <td>${esc(i.act)}</td>
                <td><span class="ts-intent-state ${i.ok}">${label[i.ok]}</span></td>
                <td class="name" style="font-size:11px">${esc(i.ev)}</td>
              </tr>`).join('')}</tbody>
          </table>
        </div>
        <div class="ts-section">
          <div class="ts-callout bad">
            <span class="ts-callout-k">现状：编译器「没做」的事没有任何信号</span>
            NZ 声明要靠用户自己写探针、diff 两份 <code>.pto</code> 才能发现被忽略；<code>stage=2</code> 只在提示日志第 N 行里说明没放下。这两件事都应该在编译结果里直接说明。
          </div>
        </div>`;
    },

    /* ---- T5 五层对照 ---- */
    layers() {
      return `
        <p class="ts-lead">同一个 scope <code>kv_score_proj</code> 在五层里的形态。当前要在编辑器、52 个 pass dump、.pto、.cpp 和提示日志之间人工对照。</p>
        <div class="ts-section">
          ${D.detail.layers.map((l) => `
            <div class="ts-layer">
              <div><div class="ts-layer-k">${esc(l.layer)}</div><div class="ts-layer-file">${esc(l.file)}</div></div>
              <pre class="ts-code">${esc(l.body)}</pre>
            </div>`).join('')}
        </div>`;
    },

    /* ---- T5 提示分诊 ---- */
    hints() {
      const h = D.detail.hints.byFile;
      const max = Math.max(...h.map((x) => x.n));
      return `
        ${kpis([
          ['性能提示总数', D.meta.hints.total, '一次构建'],
          ['PH001', D.meta.hints.ph001, '最内维 < 512B'],
          ['PH-MR-001', D.meta.hints.phmr001, '流水深度放不下'],
          ['Pass dump', D.meta.passes, '每个约 1.1 MB'],
        ])}
        <div class="ts-section">
          <h3 class="ts-section-title">按文件分布（当前按编译顺序输出，没有优先级）</h3>
          ${h.map((x) => `
            <div class="ts-bar-row">
              <span class="ts-bar-k">${esc(x.file)}</span>
              <span class="ts-bar"><i class="${x.hot ? 'mid' : ''}" style="width:${(x.n / max * 100).toFixed(1)}%"></i></span>
              <span class="ts-bar-v">${x.n}</span>
            </div>`).join('')}
          <p class="ts-section-note">高亮的两个文件才是本轮热点所在。提示最多的 <code>qkv_proj_rope.py</code>（62 条）这一轮并不相关——<b>P0：提示要与热点 join 后再排序。</b></p>
        </div>`;
    },

    /* ---- T6 容量预算 ---- */
    budget() {
      return `
        <p class="ts-lead">各级存储的实际上限来自编译器报告，而不是硬件手册。两者不一致时，用户很容易拿错数字。</p>
        <div class="ts-section">
          ${D.detail.budget.map((b) => {
            const pct = (b.used / b.limit * 100);
            return `
            <div class="ts-bar-row">
              <span class="ts-bar-k">${esc(b.space)}</span>
              <span class="ts-bar"><i class="${pct > 95 ? 'low' : pct > 60 ? 'mid' : 'high'}" style="width:${pct.toFixed(1)}%"></i></span>
              <span class="ts-bar-v">${pct.toFixed(1)}%</span>
            </div>
            <p class="ts-section-note" style="margin:2px 0 10px 100px">
              ${b.used} KB / ${b.limit} KB · ${esc(b.fn)} · ${b.refs} MemRefs
              ${b.blocks.length ? '<br>' + b.blocks.map((x) => `<code>${esc(x.n)}</code> ${x.kb} KB ${esc(x.range)} 生命周期 ${esc(x.live)}`).join('<br>') : ''}
            </p>`;
          }).join('')}
        </div>
        <div class="ts-section">
          <div class="ts-callout warn">
            <span class="ts-callout-k">上限要带来源</span>
            编译器报告里 Vec 上限是 <b>184.0 KB</b>，而调优日志中常写 192 KB。每个上限都应标注来源（编译器 / 硬件手册）与平台（a2a3 / a5）。
          </div>
        </div>`;
    },

    /* ---- T6 撞墙记录 ---- */
    walls() {
      return `
        <p class="ts-lead">每一次「再加大一档」都要靠编译试错试出来，而报错只给总量、不给明细。</p>
        <div class="ts-section">
          <table class="ts-table">
            <thead><tr><th>想做的事</th><th>撞到哪一级</th><th>报错 / 计算</th><th>最终处理</th></tr></thead>
            <tbody>${D.detail.walls.map((w) => `
              <tr><td>${esc(w.want)}</td>
                  <td><span class="ts-verdict bad">${esc(w.wall)}</span></td>
                  <td class="name" style="font-size:11px">${esc(w.detail)}</td>
                  <td>${esc(w.fix)}</td></tr>`).join('')}</tbody>
            <caption>来源：调优日志 §2 §19 §20 与 pypto-lib#665。P0：这些结论应当在调参数时就以预算形式给出，而不是编译后才知道。</caption>
          </table>
        </div>`;
    },

    /* ---- T6 搬运形态 ---- */
    movement() {
      const m = D.detail.movement;
      return `
        <div class="ts-section">
          <div class="ts-callout">
            <span class="ts-callout-k">不是带宽墙，是事务过碎</span>
            每条通道只有约 ${m.before.gbps} GB/s，远低于 HBM 上限，所以瓶颈是 transaction 笔数。<b>下「带宽墙」结论之前，先检查搬运粒度。</b>
          </div>
        </div>
        <div class="ts-section ts-diff">
          <div>
            <div class="ts-diff-head"><span class="k">改前</span><span class="v">${esc(m.before.name)}</span></div>
            <p class="ts-section-note" style="margin-top:0">${esc(m.before.desc)}</p>
            <pre class="ts-code">wkv_tile = wkv[k0 : k0 + K_TILE, o0 : o0 + OUT_TILE]
kv_acc = pl.matmul_acc(kv_acc, x_tile, wkv_tile)</pre>
          </div>
          <div>
            <div class="ts-diff-head"><span class="k">改后</span><span class="v ts-verdict ok">${esc(m.after.name)}</span></div>
            <p class="ts-section-note" style="margin-top:0">${esc(m.after.desc)}</p>
            <pre class="ts-code">wkv_tile = wkv[o0 : o0 + OUT_TILE, k0 : k0 + K_TILE]
kv_acc = pl.matmul_acc(kv_acc, x_tile, wkv_tile, <span class="hl">b_trans=True</span>)</pre>
          </div>
        </div>
        <div class="ts-section">
          <table class="ts-table">
            <thead><tr><th>指标</th><th class="num">ND2NZ</th><th class="num">b_trans (DN2ZN)</th><th class="num">变化</th></tr></thead>
            <tbody>${m.results.map((r) => `
              <tr><td>${esc(r.k)}</td><td class="num">${esc(r.before)}</td><td class="num">${esc(r.after)}</td>
                  <td class="num"><span class="ts-verdict ok">${esc(r.delta)}</span></td></tr>`).join('')}</tbody>
            <caption>来源：调优日志 §19，PR pypto-lib#628。本地 .pto 中可以看到生效后的 right tile 为 bf16[256,64] row_major/col_major。</caption>
          </table>
        </div>`;
    },

    /* ---- T7 核内构成 ---- */
    incore() {
      const s = D.detail.incoreSpan;
      return `
        ${kpis([
          ['veccore span', `${s.before} µs`, `块读后 ${s.after} µs（${s.delta}）`],
          ['MTE2 占比', '69%', '256 笔 [1,64] 散读'],
          ['VECTOR 占比', '63%', '大部分是搬运，不是计算'],
          ['真正的 softmax', '< 1 µs', 'VEXP + VDIV + VCADD'],
        ])}
        <div class="ts-section">
          <h3 class="ts-section-title">按「时间花在哪」归类，而不是按硬件单元罗列</h3>
          <table class="ts-table">
            <thead><tr><th>单元</th><th>归类</th><th class="num">占比</th><th class="num">改前 cycles</th><th class="num">改后 cycles</th><th class="num">变化</th><th>构成</th></tr></thead>
            <tbody>${D.detail.incore.map((r) => `
              <tr><td class="name">${esc(r.unit)}</td>
                  <td><span class="ts-verdict warn">${esc(r.cat)}</span></td>
                  <td class="num">${r.pct}%</td>
                  <td class="num">${num(r.cyclesBefore)}</td>
                  <td class="num">${num(r.cyclesAfter)}</td>
                  <td class="num"><span class="ts-verdict ok">${esc(r.delta)}</span></td>
                  <td>${esc(r.detail)}</td></tr>`).join('')}</tbody>
            <caption>来源：调优日志 §20，a2a3 op-sim 单 task。P0：默认分类应回答「搬运还是计算」，而不是让用户自己把指令归类。</caption>
          </table>
        </div>
        <div class="ts-section">
          <div class="ts-callout">
            <span class="ts-callout-k">杠杆</span>
            窗口起点必然是 block_size 的倍数，因此可以把逐行 <code>[1,64]</code> 散读换成逐块 <code>[8,64]</code> 跨步读：事务笔数 256 → 32，逐行 staging 消失。softmax 的数学一字未改，<code>max_error_ratio = 0.0</code>。
          </div>
        </div>`;
    },

    /* ---- T7 采样有效性 ---- */
    traceValid() {
      return `
        <div class="ts-section">
          <div class="ts-callout bad">
            <span class="ts-callout-k">这次 trace 不可用</span>
            kernel 带数据门控 <code>position_ids % 128 >= 126</code>，而自动生成的 golden 把 <code>position_ids</code> 清零 → 门控不进入 → 循环跑 0 轮 → trace 退化，kernel 看起来「很快」。
          </div>
        </div>
        <div class="ts-section">
          <h3 class="ts-section-title">同类风险：replay 脚本的默认输入</h3>
          <pre class="ts-code">def _inline_inputs():
    <span class="cm">"""Materialise inputs from shape / dtype info embedded at compile time."""</span>
    x_hc__ssa_v0 = torch.randn((2, 1, 4, 4096), dtype=torch.float32)
    <span class="hl"># NOTE: dynamic dim(s) filled with 1 — edit as needed</span></pre>
          <p class="ts-section-note">来源：本地 <code>debug/run.py</code>。默认输入是随机数、动态维填 1，同样可能让带门控的 kernel 跑不进真实分支。</p>
        </div>
        <div class="ts-section">
          <div class="ts-callout warn">
            <span class="ts-callout-k">另一个计时陷阱</span>
            开启 PMU 会让 wall 膨胀约 15%（343 µs → 401 µs）。计时与计数必须分两次运行采集，开了 PMU 的运行应自动标注「wall 不可用于性能对比」。
          </div>
        </div>`;
    },

    /* ---- T8 A/B ---- */
    abTable() {
      const v = { base: ['基线', 'neutral'], noisy: ['锚点漂移，不可比', 'bad'], ok: ['可比', 'ok'] };
      return `
        <p class="ts-lead">三版对比里，只有一版的数字可以用。判断依据不是「谁的 wall 低」，而是<b>与改动无关的锚点 scope 有没有漂移</b>。</p>
        <div class="ts-section">
          <table class="ts-table">
            <thead><tr><th>版本</th><th class="num">整图 wall</th><th class="num">softmax_pool core-time</th><th class="num">qk_pv（锚点）</th><th>比较资格</th></tr></thead>
            <tbody>${D.detail.ab.map((r) => `
              <tr${r.verdict === 'ok' ? ' class="is-focus"' : ''}>
                <td>${esc(r.ver)}</td>
                <td class="num">${r.wall} µs</td>
                <td class="num">${num(r.target)} µs<span style="color:var(--foreground-muted)"> · n=${r.targetN}</span></td>
                <td class="num">${num(r.anchor)} µs</td>
                <td><span class="ts-verdict ${v[r.verdict][1]}">${v[r.verdict][0]}</span></td>
              </tr>`).join('')}</tbody>
            <caption>来源：调优日志 §20，真机 a2a3，3 跑。整图全程 bit-identical，max_error_ratio = 0.0。</caption>
          </table>
        </div>
        <div class="ts-section">
          <div class="ts-callout bad">
            <span class="ts-callout-k">差点被骗</span>
            R1 的 wall 只降了 27 µs，而那次运行里与 softmax_pool 完全无关的 <code>qk_pv</code> 比正常快了约 2,500 µs——R1 的收益大半来自那次 session 跑得偏快。
            只有原始版本与 R2 的锚点基本一致（22301 vs 21918），<b>−113.5 µs（−8.7%）才是真实归因</b>。
          </div>
        </div>
        <p class="ts-section-note">P0：锚点应由工具自动推荐（与改动无关、耗时大、历史稳定），并在结果表里显示漂移；不满足比较资格时把提升百分比置灰。</p>`;
    },

    /* ---- T8 参数扫描 ---- */
    sweep() {
      const max = Math.max(...D.detail.sweep.map((x) => x.total));
      return `
        <p class="ts-lead">参数不是越多越好，存在拐点。当前要手工改常量、逐个运行、手工记录。</p>
        <div class="ts-section">
          ${D.detail.sweep.map((x) => `
            <div class="ts-bar-row">
              <span class="ts-bar-k">${x.n} task · ${esc(x.tile)}</span>
              <span class="ts-bar"><i class="${x.best ? 'high' : 'mid'}" style="width:${(x.total / max * 100).toFixed(1)}%"></i></span>
              <span class="ts-bar-v">${x.total} µs</span>
            </div>`).join('')}
          <p class="ts-section-note">4 个 task 是拐点：更少会丢掉重叠，更多则不降反升。来源：调优日志 §6。</p>
        </div>
        <div class="ts-section">
          <div class="ts-callout">
            <span class="ts-callout-k">机会点</span>
            扫描轴、合法区间（来自 T6 的容量预算）、越界区和数值失败区应当画在同一张图上，并显示这批实验预计占用的设备时长。
          </div>
        </div>`;
    },

    /* ---- T9 影响面 ---- */
    impact() {
      return `
        <p class="ts-lead">改一个权重签名，要一路传播到 4 个文件；另有两条同名路径<b>不能改</b>。当前靠全局搜索。</p>
        <div class="ts-section">
          <table class="ts-table">
            <thead><tr><th>文件</th><th>需要的改动</th><th>判定</th></tr></thead>
            <tbody>${D.detail.impact.map((r) => `
              <tr${r.must ? ' class="is-focus"' : ''}>
                <td class="name">${esc(r.file)}</td>
                <td>${esc(r.change)}</td>
                <td><span class="ts-verdict ${r.must ? 'warn' : 'neutral'}">${r.must ? '必须同步改' : '不受影响'}</span></td>
              </tr>`).join('')}</tbody>
          </table>
        </div>
        <div class="ts-section">
          <div class="ts-callout bad">
            <span class="ts-callout-k">漏改的代价</span>
            只漏改 decode_layer / decode_fwd 时，列越界在运行时报的是 <b>507018</b>（device drain），而不是 shape 错误，迷惑性很强。
            另一次 CI 报错实际来自分支基于旧 base 的另一个文件，与本次改动无关——<b>本地与 CI 的版本偏差会制造假问题</b>。
          </div>
        </div>`;
    },

    /* ---- T9 经验卡 ---- */
    recipe() {
      const r = D.detail.recipe;
      return `
        <div class="ts-section">
          <h3 class="ts-section-title">经验卡（P0：没有适用条件和反例就不能发布）</h3>
          <div class="ts-callout">
            <span class="ts-callout-k">法则</span>
            ${esc(r.rule)}
          </div>
        </div>
        <div class="ts-section">
          <h3 class="ts-section-title">适用条件</h3>
          <table class="ts-table">
            <tbody>${r.scope.map((s) => `<tr><td>${esc(s)}</td></tr>`).join('')}</tbody>
          </table>
        </div>
        <div class="ts-section">
          <h3 class="ts-section-title">证据</h3>
          <p style="margin:0">${esc(r.evidence)}</p>
        </div>
        <div class="ts-section">
          <div class="ts-callout warn">
            <span class="ts-callout-k">反例（同一手段，条件不同结论相反）</span>
            ${esc(r.counter)}
          </div>
        </div>
        <p class="ts-section-note">反例出现在另一个仓库、另一个时间点，与原经验之间当前没有任何关联。P0：关联到反例时应自动提示经验卡的作者。</p>`;
    },
  };

  function kpis(rows) {
    return `<div class="ts-kpis">${rows.map(([k, v, s]) => `
      <div class="ts-kpi"><div class="ts-kpi-k">${esc(k)}</div><div class="ts-kpi-v">${esc(v)}</div>
      ${s ? `<div class="ts-kpi-s">${esc(s)}</div>` : ''}</div>`).join('')}</div>`;
  }

  /* ============================================================= 泳道总览 */
  const colormap = window.PtoSwimlaneTaskPattern?.createTaskColormap?.() || null;

  function buildLanes() {
    const host = $('vizLanes');
    host.innerHTML = D.scopes.slice(0, LANE_COUNT).map((s) => `
      <div class="ts-lane" data-lane="${esc(s.name)}">
        <span class="ts-lane-k" title="${esc(s.name)}">${esc(s.name)}</span>
        <canvas class="ts-lane-canvas" data-canvas="${esc(s.name)}"></canvas>
        <span class="ts-lane-v">${s.cores} 核</span>
      </div>`).join('');
  }

  function drawLanes(focus) {
    const P = window.PtoSwimlaneTaskPattern;
    if (!P) return;
    const css = getComputedStyle(document.documentElement);
    const muted = css.getPropertyValue('--foreground-disabled').trim() || 'rgba(255,255,255,.25)';
    D.scopes.slice(0, LANE_COUNT).forEach((s) => {
      const canvas = $('vizLanes').querySelector(`[data-canvas="${CSS.escape(s.name)}"]`);
      if (!canvas) return;
      const dpr = window.devicePixelRatio || 1;
      const w = canvas.clientWidth || 300;
      const h = canvas.clientHeight || 14;
      canvas.width = Math.round(w * dpr);
      canvas.height = Math.round(h * dpr);
      const ctx = canvas.getContext('2d');
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.clearRect(0, 0, w, h);

      // 轨道底：整图 wall 的时间范围
      ctx.fillStyle = muted;
      ctx.globalAlpha = 0.18;
      ctx.fillRect(0, h / 2 - 0.5, w, 1);
      ctx.globalAlpha = 1;

      const x = (s.start / D.meta.wall) * w;
      const bw = Math.max(3, (s.span / D.meta.wall) * w);
      const isFocus = focus === s.name;
      const base = colormap ? colormap.colorForTask({ colorKey: s.kind, label: s.name }, 'engine') : '#5f6775';
      ctx.globalAlpha = focus && !isFocus ? 0.34 : 1;
      P.drawTaskBar(ctx, {
        ctx, x, y: 1, width: bw, height: h - 2,
        baseColor: base,
        isSelected: isFocus,
        isEmphasized: isFocus,
        fontFamily: css.getPropertyValue('--font-mono').trim() || 'monospace',
        task: { label: s.name, displayName: s.name, laneKind: s.kind, totalCycle: s.sum },
      });
      ctx.globalAlpha = 1;
      canvas.__ptoSwimlaneTask = {
        label: s.name, displayName: s.name, laneKind: s.kind,
        totalCycle: s.sum, clcCycle: s.avg, status: `${s.n} task · ${s.cores} 核`,
      };
    });
  }

  /* ================================================================ 渲染 */
  function renderExplorer() {
    const byPhase = D.phases.map((p) => {
      const items = D.scenes.map((s, i) => ({ s, i })).filter(({ s }) => s.phase === p.id);
      return `
        <div class="ts-group-label">${esc(p.name)}</div>
        ${items.map(({ s, i }) => `
          <button class="ts-scene${i === state.scene ? ' is-selected' : ''}" type="button" role="tab"
                  aria-selected="${i === state.scene}" data-scene="${i}">
            <span class="ts-scene-id">${esc(s.id)}</span>
            <span><span class="ts-scene-name">${esc(s.name)}</span>
            <span class="ts-scene-q">${esc(s.question)}</span></span>
          </button>`).join('')}`;
    }).join('');
    $('sceneList').innerHTML = byPhase;
  }

  function renderPhaseBar() {
    const cur = D.scenes[state.scene];
    $('phaseBar').innerHTML = D.phases.map((p) => {
      const dots = p.scenes.map((id) => {
        const idx = D.scenes.findIndex((s) => s.id === id);
        const cls = idx === state.scene ? 'is-current' : idx < state.scene ? 'is-done' : '';
        return `<span class="ts-phase-dot ${cls}"></span>`;
      }).join('');
      return `<span class="ts-phase${p.id === cur.phase ? ' is-active' : ''}" role="listitem">${esc(p.name)}<span class="ts-phase-dots">${dots}</span></span>`;
    }).join('');
  }

  function renderScene() {
    const sc = D.scenes[state.scene];
    state.view = Math.min(state.view, sc.views.length - 1);

    $('sceneTitle').textContent = `${sc.id} · ${sc.name}`;
    $('sceneMeta').textContent = sc.meta;
    $('viewTabs').innerHTML = sc.views.map((v, i) => `
      <button class="tab-control-item${i === state.view ? ' is-selected' : ''}" type="button" role="tab"
              aria-selected="${i === state.view}" data-view="${i}">${esc(v.label)}</button>`).join('');

    const view = sc.views[state.view];
    $('stage').innerHTML = `<div>${V[view.render](sc)}</div>`;
    $('stage').scrollTop = 0;

    // Inspector
    $('inspectorMeta').textContent = `${sc.id} · ${state.scene + 1}/${D.scenes.length}`;
    $('inspector').innerHTML = `
      <section class="inspector-section">
        <div class="inspector-section-head"><span class="inspector-section-title">关键问题</span></div>
        <p style="margin:0;color:var(--foreground);font:var(--type-body)">${esc(sc.question)}</p>
      </section>
      <section class="inspector-section">
        <div class="inspector-section-head"><span class="inspector-section-title">用户在做什么</span>
          <span class="inspector-section-kicker">现状作业过程</span></div>
        <ul class="ts-ins-list">${sc.actions.map((a) => `<li>${a}</li>`).join('')}</ul>
      </section>
      <section class="inspector-section">
        <div class="inspector-section-head"><span class="inspector-section-title">卡点</span></div>
        <ul class="ts-ins-list pain">${sc.pains.map((a) => `<li>${esc(a)}</li>`).join('')}</ul>
      </section>
      <section class="inspector-section">
        <div class="inspector-section-head"><span class="inspector-section-title">机会点</span>
          <span class="inspector-section-kicker">UX 重点设计</span></div>
        ${sc.opps.map((o) => `<div class="ts-opp"><span class="ts-pri ${o.p}">${o.p}</span><span class="ts-opp-t">${esc(o.t)}</span></div>`).join('')}
      </section>
      <section class="inspector-section">
        <div class="inspector-section-head"><span class="inspector-section-title">对象与原型</span></div>
        <div class="ts-obj">${esc(sc.objs)}</div>
        <div class="ts-obj-proto">可承载：${esc(sc.proto)}</div>
      </section>
      <section class="inspector-section">
        <div class="inspector-section-head"><span class="inspector-section-title">数据来源</span>
          <span class="inspector-section-kicker">${sc.sources.length} 项</span></div>
        ${sc.sources.map((s) => `
          <div class="ts-src">
            <div class="ts-src-head"><span class="ts-prov ${s.t}">${s.t}</span><span class="ts-src-s">${esc(s.s)}</span></div>
            <div class="ts-src-d">${esc(s.d)}</div>
          </div>`).join('')}
      </section>`;

    // 终端与泳道
    $('terminalBody').innerHTML = sc.terminal.map((l) => `<p class="ts-term-line${l.startsWith('$') ? ' cmd' : ''}">${esc(l)}</p>`).join('');
    $('vizMeta').textContent = sc.focus ? `聚焦 ${sc.focus}` : '本场景无聚焦对象';
    $('vizNote').textContent = sc.vizNote;
    drawLanes(sc.focus);

    // 状态条
    $('status').innerHTML = [
      `<span>场景 <b>${sc.id} · ${state.scene + 1}/${D.scenes.length}</b></span>`,
      `<span class="sep">|</span><span>case <b>decode_csa_test</b></span>`,
      `<span>rank <b>0</b> · <b>${D.meta.platform}</b></span>`,
      `<span>wall <b>${D.meta.wall} µs</b></span>`,
      `<span>AIC <b>${D.meta.utilAic}%</b> · AIV <b>${D.meta.utilAiv}%</b></span>`,
      `<span>scope <b>${D.meta.scopeCount}</b></span>`,
      `<span>pass <b>${D.meta.passes}</b> · 提示 <b>${D.meta.hints.total}</b></span>`,
      `<span class="sep">|</span><span>情绪 <b>${esc(sc.emotionLabel)}</b></span>`,
    ].join('');

    renderExplorer();
    renderPhaseBar();
    syncPlayback();
  }

  /* ============================================================== 播放条 */
  let pbEls = null;

  function syncPlayback() {
    if (!pbEls) return;
    const sc = D.scenes[state.scene];
    if (pbEls.scrubber) pbEls.scrubber.value = String(state.scene);
    if (pbEls.label) pbEls.label.textContent = `${state.scene} / ${D.scenes.length - 1}`;
    if (pbEls.opname) pbEls.opname.textContent = `${sc.id} · ${sc.name}`;
  }

  function goto(i, stopPlay) {
    const n = D.scenes.length;
    state.scene = ((i % n) + n) % n;
    state.view = 0;
    if (stopPlay) stopPlayback();
    renderScene();
  }

  function startPlayback() {
    if (state.timer) return;
    state.playing = true;
    state.timer = setInterval(() => {
      if (state.scene >= D.scenes.length - 1) { stopPlayback(); return; }
      goto(state.scene + 1, false);
    }, 4200);
  }

  function stopPlayback() {
    state.playing = false;
    if (state.timer) { clearInterval(state.timer); state.timer = null; }
  }

  function wirePlayback() {
    // ide-frame 已挂载播放条并接管折叠、图标与 scrubber 的内部状态；
    // 这里只在同一批控件上追加场景切换的业务处理，不复刻播放条外壳。
    const p = 'ide-floating-playback-1';
    pbEls = {
      play: $(`${p}-play`),
      back: $(`${p}-step-back`),
      fwd: $(`${p}-step-fwd`),
      replay: $(`${p}-replay`),
      scrubber: $(`${p}-scrubber`),
      label: $(`${p}-scrubber-label`),
      opname: $(`${p}-scrubber-opname`),
    };
    pbEls.play?.addEventListener('click', () => { state.playing ? stopPlayback() : startPlayback(); });
    pbEls.back?.addEventListener('click', () => goto(state.scene - 1, true));
    pbEls.fwd?.addEventListener('click', () => goto(state.scene + 1, true));
    pbEls.replay?.addEventListener('click', () => goto(0, true));
    pbEls.scrubber?.addEventListener('input', () => goto(Number(pbEls.scrubber.value) || 0, true));
  }

  /* ================================================================ 启动 */
  function boot() {
    buildLanes();

    $('sceneList').addEventListener('click', (e) => {
      const btn = e.target.closest('[data-scene]');
      if (btn) goto(Number(btn.dataset.scene), true);
    });
    $('viewTabs').addEventListener('click', (e) => {
      const btn = e.target.closest('[data-view]');
      if (!btn) return;
      state.view = Number(btn.dataset.view);
      renderScene();
    });

    document.addEventListener('keydown', (e) => {
      if (e.target.matches('input, textarea')) return;
      if (e.key === 'ArrowRight') { goto(state.scene + 1, true); e.preventDefault(); }
      if (e.key === 'ArrowLeft') { goto(state.scene - 1, true); e.preventDefault(); }
    });

    $('themeBtn').addEventListener('click', () => {
      const root = document.documentElement;
      root.dataset.theme = root.dataset.theme === 'light' ? 'dark' : 'light';
      try { localStorage.setItem('ts-theme', root.dataset.theme); } catch (err) { /* 私密模式下忽略 */ }
      drawLanes(D.scenes[state.scene].focus);
    });
    try {
      const t = localStorage.getItem('ts-theme');
      if (t === 'light' || t === 'dark') document.documentElement.dataset.theme = t;
    } catch (err) { /* 私密模式下忽略 */ }

    let raf = null;
    window.addEventListener('resize', () => {
      cancelAnimationFrame(raf);
      raf = requestAnimationFrame(() => drawLanes(D.scenes[state.scene].focus));
    });

    // 泳道悬停提示走共享 pattern，不自建 tooltip
    window.PtoSwimlaneTaskPattern?.initHoverTooltip?.({
      root: $('vizLanes'),
      targets: '.ts-lane-canvas',
      appendTo: $('viz'),
      getTask: (target) => target.__ptoSwimlaneTask || null,
    });

    wirePlayback();
    renderScene();
  }

  // ide-frame 的 pattern.js 在 DOMContentLoaded / 立即执行时完成 initAll，
  // 播放条控件此时已经在 DOM 里；这里在其之后启动业务逻辑。
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', boot, { once: true });
  } else {
    boot();
  }
})();
