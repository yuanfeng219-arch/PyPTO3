/**
 * 融合算子推荐 · 停靠在模型结构视图右侧
 *
 * 四个页签：替换方案 / 未推荐原因 / 算子核对 / 验证计划
 * 数据全部来自 fusion-rules-data.js（model-infer-fusion skill 口径），
 * 本文件只负责渲染与交互。
 *
 * 方向 C 的两条表达线：
 *   cross 方案 —— 结构图上真实合并多个 L2 模块，面板给「融合前/融合后」开关
 *   intra 方案 —— 结构图只标位置，关系变化用「当前调用 → 替换后调用」代码对照
 */
(function registerFusionAdvisor() {
  'use strict';

  const MODEL_ID = 'deepseek-v4-flash';

  const TABS = [
    { id: 'plans', label: '替换方案' },
    { id: 'blocked', label: '未推荐原因' },
    { id: 'apis', label: '算子核对' },
    { id: 'validation', label: '验证计划' },
  ];

  const state = {
    open: false,
    tab: 'plans',
    selected: 'mod-mla-prolog',
    fusionMode: 'before',
    lastFocus: null,
  };

  let root = null;

  const rules = () => window.PtoFusionRules;
  const viz = () => window.PtoDeepSeekV4ModelViz;
  const esc = (v) => String(v).replace(/[&<>"']/g, (ch) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[ch]));
  const int = (n) => (n === null || n === undefined ? '—' : Math.round(Number(n)).toLocaleString('en-US'));

  const view = () => rules().evaluate();

  /** 手风琴当前展开的方案；全部收起时为 null */
  function selectedItem(model) {
    const data = model || view();
    return state.selected ? (data.items.find((x) => x.id === state.selected) || null) : null;
  }

  /** 需要「一个当前方案」但允许回退的地方（验证计划、对比视图） */
  function currentItem(model) {
    const data = model || view();
    return selectedItem(data) || data.actionable[0] || data.items[0] || null;
  }

  /** 展开哪条就把哪条投到结构图上；全部收起时把图恢复原样 */
  function syncGraphToSelection() {
    const item = selectedItem();
    if (item) pushPlanToGraph(item);
    else clearPlanFromGraph();
  }

  function pushPlanToGraph(item) {
    if (!item || !viz() || !viz().setFusionPlan) return;
    viz().setFusionPlan({
      id: item.id,
      kind: item.kind,
      label: (item.apis && item.apis[0]) || item.title,
      sub: item.kind === 'cross' ? item.nodes.length + ' 个模块合并为 1 个' : item.title,
      nodes: item.nodes,
    });
    if (viz().setFusionMode) viz().setFusionMode(state.fusionMode);
  }

  function clearPlanFromGraph() {
    if (viz() && viz().clearFusionPlan) viz().clearFusionPlan();
  }

  /* ---------------- 通用片段 ---------------- */

  const statusChip = (meta) => `<i class="kf-fus-level is-${meta.id}" title="${esc(meta.desc)}">${esc(meta.label)}</i>`;

  const kindChip = (item) => (item.kind === 'cross'
    ? `<i class="kf-fus-tag is-cross" title="${esc(rules().KIND.cross.desc)}">跨模块 · ${item.nodes.length} 模块</i>`
    : `<i class="kf-fus-tag is-cat" title="${esc(rules().KIND.intra.desc)}">模块内</i>`);

  const nodeChips = (nodes) => nodes
    .map((id) => `<button class="kf-fus-node" type="button" data-goto-node="${esc(id)}" title="在结构图中定位 ${esc(id)}">${esc(id)}</button>`)
    .join('');

  const apiChips = (item) => (item.apiList || [])
    .map((a) => `<a class="kf-fus-api ${a.listed ? '' : 'is-unlisted'}" href="${esc(a.doc)}" target="_blank" rel="noreferrer" title="${esc(a.note)}${a.listed ? '' : ' · 不在算子总表内'}">${esc(a.name)}${a.beta ? '<em>beta</em>' : ''}${a.listed ? '' : '<em class="warn">未登记</em>'}</a>`)
    .join('');

  const codeBlock = (lines, tone) => (lines && lines.length
    ? `<pre class="kf-fus-code ${tone ? 'is-' + tone : ''}">${lines.map((l) => `<span class="kf-fus-codeline"><code>${esc(l[0])}</code>${l[1] ? `<em>${esc(l[1])}</em>` : ''}</span>`).join('')}</pre>`
    : '');

  /* ---------------- 替换方案 ---------------- */

  function renderKpis(data) {
    const s = data.summary;
    const c = rules().CONFIG;
    return `<div class="kf-fus-kpis">
      <div class="kf-fus-kpi"><span>可落地方案</span><b>${s.actionable}</b><small>可直接替换 ${s.byStatus.ready} · 需前置改造 ${s.byStatus.prep}</small></div>
      <div class="kf-fus-kpi"><span>跨模块超级融合</span><b class="is-strong">${s.cross}</b><small>结构图上可见合并</small></div>
      <div class="kf-fus-kpi"><span>模块内融合</span><b>${s.intra}</b><small>看代码级调用对照</small></div>
      <div class="kf-fus-kpi"><span>候选算子</span><b>${s.apiCount}</b><small>${s.riskyApis} 个命名存疑，须先核对</small></div>
      <div class="kf-fus-kpi"><span>无现成算子</span><b>${s.handoff}</b><small>转新增融合算子需求</small></div>
      <div class="kf-fus-kpi"><span>未推荐</span><b>${s.byStatus.constraint + s.byStatus.blocked + s.byStatus.none}</b><small>约束待确认 / 不适配 / 无算子</small></div>
    </div>`;
  }

  function renderList(data) {
    if (!data.actionable.length) return '<div class="kf-fus-empty">当前没有可落地的替换方案。</div>';
    const openId = state.selected;
    const rows = data.actionable.map((item, index) => {
      const open = item.id === openId;
      return `<div class="kf-fus-item ${open ? 'is-open' : ''}">
        <button type="button" class="kf-fus-row" data-plan-pick="${esc(item.id)}" aria-expanded="${open}">
          <span class="kf-fus-row__rank">${index + 1}</span>
          <span class="kf-fus-row__main">
            <b>${esc(item.title)}</b>
            <small>${esc(item.module)}</small>
            <span class="kf-fus-row__tags">
              ${kindChip(item)}
              ${statusChip(item.statusMeta)}
              <i class="kf-fus-tag">${item.layers} 层生效</i>
            </span>
          </span>
          <span class="kf-fus-row__api">${esc((item.apis && item.apis[0]) || '—')}${item.apis && item.apis.length > 1 ? ` +${item.apis.length - 1}` : ''}</span>
          <span class="kf-fus-row__chev" aria-hidden="true">⌄</span>
        </button>
        ${open ? renderDetail(item) : ''}
      </div>`;
    }).join('');
    return `<div class="kf-fus-list">${rows}</div>`;
  }

  /** cross：图上真有变化，给开关；intra：图上没变化，说清楚为什么 */
  function renderGraphLink(item) {
    if (item.kind === 'cross' && item.nodes.length > 1) {
      return `<div class="kf-fus-block">
        <h4>结构图上的变化<small>左侧画布联动</small></h4>
        <div class="kf-fus-modeswitch" role="group" aria-label="结构图视图">
          <button type="button" data-fusion-mode="before" class="${state.fusionMode === 'before' ? 'is-active' : ''}">融合前 · ${item.nodes.length} 个模块</button>
          <button type="button" data-fusion-mode="after" class="${state.fusionMode === 'after' ? 'is-active' : ''}">融合后 · 1 个算子</button>
        </div>
        <div class="kf-fus-nodes">${nodeChips(item.nodes)}</div>
        <p class="kf-fus-note">切到「融合后」，左侧整网图会把这 ${item.nodes.length} 个模块合并成一个节点，模块之间的依赖被这个 API 吃掉，外部边重新接到融合节点上。要左右并排看，用右上角的「替换前后对比」。</p>
      </div>`;
    }
    return `<div class="kf-fus-block">
      <h4>结构图上的位置<small>模块内融合</small></h4>
      <div class="kf-fus-nodes">${nodeChips(item.nodes)}</div>
      <p class="kf-fus-note">这类融合发生在模块内部，整网结构图是模块级的，<b>拓扑不会变化</b>——图上只标位置。关系变化看下面的调用对照。</p>
    </div>`;
  }

  function renderCodeDiff(item) {
    if (!item.before || !item.before.length) return '';
    const after = item.after && item.after.length
      ? codeBlock(item.after, 'after')
      : '<p class="kf-fus-note">无对应 torch_npu 调用，见「未推荐原因」。</p>';
    return `<div class="kf-fus-block">
      <h4>调用对照<small>${esc(rules().SOURCE)}</small></h4>
      <div class="kf-fus-diff">
        <div><span class="kf-fus-difflabel is-before">当前实现</span>${codeBlock(item.before, 'before')}</div>
        <div><span class="kf-fus-difflabel is-after">替换后</span>${after}</div>
      </div>
    </div>`;
  }

  function listBlock(title, items, tone, sub) {
    if (!items || !items.length) return '';
    return `<div class="kf-fus-block${tone ? ` is-${tone}` : ''}">
      <h4>${esc(title)}${sub ? `<small>${esc(sub)}</small>` : ''}</h4>
      <ul class="kf-fus-bullets">${items.map((x) => `<li>${esc(x)}</li>`).join('')}</ul>
    </div>`;
  }

  function renderStages(item) {
    if (!item.stages) return '';
    const w = rules().WORKLOAD;
    return `<div class="kf-fus-block">
      <h4>Prefill / Decode 差异<small>同一模块两阶段可能走不同算子路径</small></h4>
      <div class="kf-fus-stages">
        <div><span>${esc(w.prefill.label)}<em>${esc(w.prefill.note)}</em></span><p>${esc(item.stages.prefill)}</p></div>
        <div><span>${esc(w.decode.label)}<em>${esc(w.decode.note)}</em></span><p>${esc(item.stages.decode)}</p></div>
      </div>
    </div>`;
  }

  function renderConstraints(item) {
    if (!item.constraints || !item.constraints.length) return '';
    return `<div class="kf-fus-block is-warn">
      <h4>待确认约束<small>${item.unchecked} 项未核对</small></h4>
      ${item.constraints.map((x) => `<div class="kf-fus-constraint">
        <i>${x.checked ? '✓' : '?'}</i>
        <b>${esc(x.item)}</b>
        <p>${esc(x.why)}</p>
      </div>`).join('')}
      <p class="kf-fus-note">skill 第四步要求每个候选 API 都查过官方详情文档。当前分析机没有 torch_npu 环境，脚本会降级到 12 条兜底集，所以这些约束一律记为<b>未核对</b>，需在有环境的机器上用 <code>torch_npu_query.py show</code> 补齐。</p>
    </div>`;
  }

  function renderDetail(item) {
    if (!item) return '';
    // 手风琴展开体：标题、状态、类型已经在行头上了，这里不再重复
    return `<div class="kf-fus-panel">
      <div class="kf-fus-panelhead">
        <div class="kf-fus-apis">${apiChips(item)}</div>
        <div class="kf-fus-detailactions">
          <button class="kf-fus-btn" type="button" data-plan-verify="${esc(item.id)}">验证计划</button>
          <button class="kf-fus-btn" type="button" data-export="${esc(item.id)}">导出 JSON</button>
        </div>
      </div>
      <div class="kf-fus-detailbody">
        ${renderGraphLink(item)}
        ${renderCodeDiff(item)}
        ${listBlock('覆盖的子链路', item.covers, 'good')}
        ${listBlock('未覆盖', item.uncovered, null, '需另找算子或保持原实现')}
        ${listBlock('前置改造', item.prep, 'warn')}
        ${renderStages(item)}
        ${renderConstraints(item)}
        <div class="kf-fus-block">
          <h4>最小验证切口</h4>
          <p class="kf-fus-fallback">${esc(item.verify)}</p>
        </div>
        <div class="kf-fus-block">
          <h4>回退方案</h4>
          <p class="kf-fus-note">${esc(item.fallback)}</p>
        </div>
        ${item.note ? `<p class="kf-fus-note">${esc(item.note)}</p>` : ''}
        <p class="kf-fus-note">参考链路：${esc(item.reference)}</p>
      </div>
    </div>`;
  }

  function renderPlans(data) {
    return `${renderKpis(data)}${renderList(data)}`;
  }

  /* ---------------- 未推荐原因 ---------------- */

  function renderBlocked(data) {
    const group = (status) => data.deferred.filter((x) => x.status === status);
    const card = (x, body) => `<article class="kf-fus-blockcard is-${x.status}">
      <header>
        <div>${statusChip(x.statusMeta)}<h4>${esc(x.title)}</h4></div>
        <span>${esc((x.apis && x.apis[0]) || '无候选算子')}</span>
      </header>
      <p class="kf-fus-pattern">${esc(x.module)}</p>
      <div class="kf-fus-nodes">${nodeChips(x.nodes)}</div>
      ${body}
    </article>`;

    const blockedCards = group('blocked').map((x) => card(x, `
      ${(x.blockers || []).map((b) => `<div class="kf-fus-reason"><i>${esc(b.code)}</i><p>${esc(b.note)}</p></div>`).join('')}
      ${codeBlock(x.before, 'before')}
      <p class="kf-fus-unblock"><b>回退</b>${esc(x.fallback)}</p>
      ${x.note ? `<p class="kf-fus-note">${esc(x.note)}</p>` : ''}
    `)).join('');

    const noneCards = group('none').map((x) => card(x, `
      <p class="kf-fus-note"><b>为什么没有算子</b>${esc(x.handoff.why)}</p>
      <p class="kf-fus-note"><b>移交内容</b>${esc(x.handoff.payload)}</p>
      <p class="kf-fus-unblock"><b>下一步</b>${esc(x.handoff.next)}</p>
    `)).join('');

    const constraintCards = group('constraint').map((x) => card(x, `
      ${(x.constraints || []).map((cst) => `<div class="kf-fus-constraint"><i>?</i><b>${esc(cst.item)}</b><p>${esc(cst.why)}</p></div>`).join('')}
      <p class="kf-fus-unblock"><b>解除条件</b>查 ${esc((x.apis || []).join(' / ') || '相关算子')} 的官方详情文档，确认上述约束后可升为可落地方案</p>
    `)).join('');

    return `<div class="kf-fus-pane">
      ${constraintCards ? `<section class="kf-fus-section"><header><h3>约束待确认</h3><span>${group('constraint').length} 条 · 查文档后才能定</span></header>${constraintCards}</section>` : ''}
      ${blockedCards ? `<section class="kf-fus-section"><header><h3>不适配</h3><span>${group('blocked').length} 条 · 附硬约束证据</span></header>${blockedCards}</section>` : ''}
      ${noneCards ? `<section class="kf-fus-section"><header><h3>无现成算子 · 转新增融合算子需求</h3><span>${group('none').length} 条</span></header>${noneCards}</section>` : ''}
      <section class="kf-fus-section">
        <header><h3>为什么 V4 的 Residual + Norm 类融合普遍不成立</h3><span>结构性原因</span></header>
        <p class="kf-fus-note">参考链路里高频使用的 <code>npu_add_rms_norm</code>、<code>npu_moe_distribute_combine_add_rms_norm</code> 都假设 <code>x = norm(residual + x)</code>。DeepSeek V4 Flash 用 mHC 取代了普通残差：残差在 <code>hc_post</code> 里按 Sinkhorn 出来的 post/comb 权重合并（${esc(rules().SOURCE)}:683-686、700），且每层出现两次（attn 前后、ffn 前后）。这不是参数不匹配，是结构不同。</p>
      </section>
    </div>`;
  }

  /* ---------------- 算子核对 ---------------- */

  function renderApis() {
    const list = rules().apiChecklist();
    const unlisted = list.filter((x) => !x.listed);
    const rows = list.map((a) => `<tr class="${a.listed ? '' : 'is-hit'}">
      <td><code>${esc(a.name)}</code>${a.beta ? ' <i class="kf-fus-tag">beta</i>' : ''}</td>
      <td>${a.listed ? '<span class="kf-fus-muted">在总表</span>' : '<i class="kf-fus-hit">不在总表</i>'}</td>
      <td>${esc(a.note)}${a.alias ? `<small>别名：${esc(a.alias)}</small>` : ''}</td>
      <td>${a.modules.map((m) => esc(m)).join('、')}</td>
      <td><a class="kf-fus-link" href="${esc(a.doc)}" target="_blank" rel="noreferrer">文档 ↗</a></td>
    </tr>`).join('');

    return `<div class="kf-fus-pane">
      <section class="kf-fus-section">
        <header><h3>候选算子核对清单</h3><span>${list.length} 个 · ${unlisted.length} 个不在总表内</span></header>
        <table class="kf-fus-table">
          <thead><tr><th>算子</th><th>总表</th><th>说明</th><th>用在哪</th><th>文档</th></tr></thead>
          <tbody>${rows}</tbody>
        </table>
        <p class="kf-fus-note">算子总表是 skill 附带的版本快照（144 个），它自己声明「只作为候选目录，进入候选清单后必须用本地 docstring 或在线文档核对当前 torch_npu 版本是否存在、签名是否一致」。下面这几处参考链路与总表命名不一致，是必须先核实的：</p>
        <ul class="kf-fus-bullets">
          <li><code>npu_add_rms_norm</code> — 三份 attention 参考链路都在用，但不在总表内（总表只有 <code>npu_rms_norm</code>）</li>
          <li><code>npu_lightning_indexer_quant</code>（参考文档）↔ <code>npu_quant_lightning_indexer</code>（总表）</li>
          <li><code>npu_sparse_flash_attention_antiquant</code>（参考文档）↔ <code>npu_kv_quant_sparse_flash_attention</code>（总表）</li>
          <li><code>npu_kv_rmsnorm_rope_cache_v2</code>（参考文档）↔ <code>npu_kv_rmsnorm_rope_cache</code>（总表无 v2）</li>
        </ul>
      </section>
      <section class="kf-fus-section">
        <header><h3>怎么核对</h3><span>skill 第三步</span></header>
        <pre class="kf-fus-code"><span class="kf-fus-codeline"><code>python3 scripts/torch_npu_query.py show &lt;api_name&gt;</code><em>单 API 详情</em></span><span class="kf-fus-codeline"><code>python3 scripts/torch_npu_query.py search "&lt;keyword&gt;"</code><em>反向搜索</em></span><span class="kf-fus-codeline"><code>python3 scripts/torch_npu_query.py list --prefix npu_</code><em>枚举算子名</em></span></pre>
        <p class="kf-fus-note">数据源是 torch_npu 安装包内的 <code>_op_plugin_docs.py</code>，与安装版本严格绑定。<b>没有 torch_npu 环境时脚本会降级到内嵌的 12 条兜底集</b>，覆盖面很小——本面板里所有约束因此都标为未核对。</p>
      </section>
    </div>`;
  }

  /* ---------------- 验证计划 ---------------- */

  function renderValidation(data) {
    const item = currentItem(data);
    const steps = [
      ['完成前置改造', item && item.prep && item.prep.length ? item.prep.join('；') : '本方案无前置改造'],
      ['替换该模块的算子代码', item ? ((item.apis || []).join(' / ') || '—') : '—'],
      ['精度对比', '融合前后输出逐元素对齐'],
      ['性能对比', '确认有收益，而不是只看理论 launch 数'],
      ['通过则保留，继续下一个模块', '每次只落一个可独立验证的链路'],
      ['失败则回退并记录', '记录报错、尝试过的参数、不适配证据；确认无现成 API 且有明确收益时转新增算子需求'],
    ];
    return `<div class="kf-fus-pane">
      <section class="kf-fus-section">
        <header><h3>替换流程 · skill 第五步</h3><span>当前方案：${item ? esc(item.title) : '—'}</span></header>
        <ol class="kf-fus-gates">${steps.map((s, i) => `<li><i>${i + 1}</i><div><b>${esc(s[0])}</b><p>${esc(s[1])}</p></div></li>`).join('')}</ol>
        ${item ? `<div class="kf-fus-block"><h4>最小验证切口</h4><p class="kf-fus-fallback">${esc(item.verify)}</p></div>` : ''}
      </section>
      <section class="kf-fus-section">
        <header><h3>待处理队列</h3><span>${data.actionable.length} 个可落地方案</span></header>
        <table class="kf-fus-table">
          <thead><tr><th>方案</th><th>状态</th><th>类型</th><th>前置改造</th><th>未核约束</th></tr></thead>
          <tbody>${data.actionable.map((x) => `<tr>
            <td><button class="kf-fus-link" type="button" data-plan-jump="${esc(x.id)}">${esc(x.title)}</button></td>
            <td>${statusChip(x.statusMeta)}</td>
            <td>${x.kind === 'cross' ? '跨模块' : '模块内'}</td>
            <td>${x.prep && x.prep.length ? x.prep.length + ' 项' : '<span class="kf-fus-muted">无</span>'}</td>
            <td>${x.unchecked ? `<i class="kf-fus-hit">${x.unchecked}</i>` : '<span class="kf-fus-muted">—</span>'}</td>
          </tr>`).join('')}</tbody>
        </table>
        <p class="kf-fus-note">skill 第五步要求：每次优先落一个可独立验证的候选链路，完成精度对齐与性能观察后再继续下一个；不得跳过任何已进入候选清单的模块，无法实施的也必须记录失败证据与阻塞原因。</p>
      </section>
    </div>`;
  }

  /* ---------------- 外壳 ---------------- */

  function shell(data) {
    const c = rules().CONFIG;
    const s = data.summary;
    const tabs = TABS.map((t) => {
      const badge = t.id === 'plans' ? s.actionable
        : t.id === 'blocked' ? data.deferred.length
          : t.id === 'apis' ? s.apiCount : '';
      return `<button type="button" class="${state.tab === t.id ? 'is-active' : ''}" data-fus-tab="${t.id}" role="tab" aria-selected="${state.tab === t.id}">${esc(t.label)}${badge ? `<em>${badge}</em>` : ''}</button>`;
    }).join('');

    return `<aside class="kf-fus-drawer" role="region" aria-label="融合算子推荐">
      <header class="kf-fus-head">
        <div>
          <div class="kf-fus-crumb"><button type="button" data-fus-close>模型结构</button><span>/</span><button type="button" data-fus-close>DeepSeek V4 Flash</button><span>/</span><b>融合算子推荐</b></div>
          <h2>torch_npu 替换方案</h2>
          <p>skill: model-infer-fusion · 源码 ${esc(rules().SOURCE)}</p>
        </div>
        <div class="kf-fus-headactions">
          <button class="kf-fus-btn is-primary" type="button" data-compare aria-haspopup="dialog" aria-controls="modelCompare">替换前后对比 ↗</button>
          <button class="kf-fus-close" type="button" data-fus-close aria-label="关闭">✕</button>
        </div>
      </header>
      <div class="kf-fus-context">
        <span class="kf-fus-ctxitem">Hidden <b>${int(c.dim)}</b> · <b>${c.n_layers}</b> 层</span>
        <span class="kf-fus-ctxsep"></span>
        <span class="kf-fus-ctxitem">MLA + Indexer · 压缩层 <b>${c.compressLayers}</b> / indexer <b>${c.indexerLayers}</b></span>
        <span class="kf-fus-ctxsep"></span>
        <span class="kf-fus-ctxitem">MoE <b>${c.n_routed_experts}+${c.n_shared_experts}</b> · top-k <b>${c.n_activated_experts}</b></span>
        <span class="kf-fus-ctxsep"></span>
        <span class="kf-fus-ctxitem">精度 <b>${esc(c.expert_dtype.toUpperCase())}</b> / <b>${esc(c.dtype.toUpperCase())}</b></span>
      </div>
      <nav class="kf-fus-tabs" role="tablist">${tabs}</nav>
      <div class="kf-fus-body" id="fusBody"></div>
    </aside>`;
  }

  function renderBody() {
    const body = root && root.querySelector('#fusBody');
    if (!body) return;
    const data = view();
    if (state.tab === 'plans') body.innerHTML = renderPlans(data);
    else if (state.tab === 'blocked') body.innerHTML = renderBlocked(data);
    else if (state.tab === 'apis') body.innerHTML = renderApis();
    else body.innerHTML = renderValidation(data);
    body.scrollTop = 0;
  }

  function renderAll() {
    if (!root) return;
    root.innerHTML = shell(view());
    renderBody();
  }

  /* ---------------- 交互 ---------------- */

  function download(name, text) {
    const blob = new Blob([text], { type: 'application/json;charset=utf-8' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = name;
    document.body.appendChild(a);
    a.click();
    a.remove();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  }

  function onClick(event) {
    const t = event.target;
    if (t.closest('[data-fus-close]')) { close(); return; }

    const tab = t.closest('[data-fus-tab]');
    if (tab) {
      state.tab = tab.dataset.fusTab;
      renderAll();
      if (state.tab === 'plans') syncGraphToSelection();
      return;
    }

    const mode = t.closest('[data-fusion-mode]');
    if (mode) {
      state.fusionMode = mode.dataset.fusionMode;
      if (viz() && viz().setFusionMode) viz().setFusionMode(state.fusionMode);
      renderBody();
      return;
    }

    const pick = t.closest('[data-plan-pick]');
    if (pick) {
      const id = pick.dataset.planPick;
      // 再点已展开的那条就收起；点别的就切过去（单开手风琴）
      state.selected = state.selected === id ? null : id;
      state.fusionMode = 'before';
      renderBody();
      syncGraphToSelection();
      // 展开后把这条滚进可视区，避免展开体把行头顶出屏幕
      if (state.selected) {
        const el = root.querySelector(`[data-plan-pick="${CSS.escape(id)}"]`);
        if (el) el.scrollIntoView({ block: 'nearest' });
      }
      return;
    }

    const jump = t.closest('[data-plan-jump]');
    if (jump) {
      state.selected = jump.dataset.planJump;
      state.tab = 'plans';
      renderAll();
      syncGraphToSelection();
      return;
    }

    const cmp = t.closest('[data-compare]');
    if (cmp) {
      const data = view();
      // 只有跨模块方案在整网图上有拓扑变化；默认叠加全部这类方案，
      // 也就是「全部替换完之后整网长什么样」
      const crossPlans = data.items
        .filter((x) => rules().changesTopology(x) && rules().isActionable(x))
        .map((x) => ({ id: x.id, kind: x.kind, label: (x.apis && x.apis[0]) || x.title, sub: x.nodes.length + ' 个模块合并为 1 个', nodes: x.nodes }));
      const current = currentItem(data);
      if (viz() && viz().openCompare) {
        viz().openCompare({
          plans: crossPlans,
          // 切到「单个方案」时用当前选中的（若它本身就是跨模块方案）
          activeId: current && rules().changesTopology(current) ? current.id : (crossPlans[0] && crossPlans[0].id),
          scope: 'all',
        });
      }
      return;
    }

    const verify = t.closest('[data-plan-verify]');
    if (verify) { state.selected = verify.dataset.planVerify; state.tab = 'validation'; renderAll(); return; }

    const one = t.closest('[data-export]');
    if (one) {
      const item = view().items.find((x) => x.id === one.dataset.export);
      if (item) {
        download(`${item.id}.json`, JSON.stringify({
          module_id: item.id,
          module: item.title,
          kind: item.kind,
          graph_nodes: item.nodes,
          candidate_apis: item.apis,
          status: item.status,
          covers: item.covers,
          uncovered: item.uncovered,
          prep: item.prep,
          constraints: item.constraints,
          stages: item.stages,
          minimal_verification: item.verify,
          fallback: item.fallback,
          handoff: item.handoff || null,
          blockers: item.blockers || [],
          source: rules().SOURCE,
        }, null, 2));
      }
      return;
    }

    const node = t.closest('[data-goto-node]');
    if (node && viz() && viz().focusNode) viz().focusNode(node.dataset.gotoNode);
  }

  /* ---------------- 生命周期 ---------------- */

  function open() {
    const host = document.getElementById('modelArchitectureView');
    if (!host || !rules()) return;
    if (!root) {
      root = document.createElement('div');
      root.className = 'kf-fus-root';
      root.id = 'fusionAdvisor';
      host.appendChild(root);
      root.addEventListener('click', onClick);
    }
    state.lastFocus = document.activeElement;
    state.open = true;
    root.hidden = false;
    host.classList.add('is-fusion-docked');
    renderAll();
    if (state.tab === 'plans') syncGraphToSelection();
  }

  function close() {
    if (!root) return;
    state.open = false;
    root.hidden = true;
    const host = document.getElementById('modelArchitectureView');
    if (host) host.classList.remove('is-fusion-docked');
    clearPlanFromGraph();
    if (state.lastFocus && state.lastFocus.isConnected) state.lastFocus.focus();
  }

  function syncEntry() {
    const active = (window.PtoModelArchitectureState || {}).active || 'qwen3';
    const supported = active === MODEL_ID && !!rules();
    document.querySelectorAll('[data-open-fusion]').forEach((el) => { el.hidden = !supported; });
    if (!supported) { if (state.open) close(); return; }
    const data = view();
    const stat = document.querySelector('[data-fusion-stat]');
    if (stat) stat.innerHTML = `<i></i>可落地 <b>${data.summary.actionable}</b> · 跨模块 <b>${data.summary.cross}</b> · 未推荐 <b>${data.deferred.length}</b>`;
    const meta = document.querySelector('[data-fusion-meta]');
    if (meta) meta.textContent = 'model-infer-fusion · torch_npu 替换';
  }

  const scheduleSync = () => setTimeout(syncEntry, 0);

  document.addEventListener('keydown', (event) => {
    if (event.key === 'Escape' && state.open) { event.stopPropagation(); close(); }
  });

  document.addEventListener('click', (event) => {
    if (event.target.closest('[data-open-fusion]')) {
      event.preventDefault();
      event.stopPropagation();
      open();
      return;
    }
    if (event.target.closest('[data-model-id]')) { scheduleSync(); return; }
    const rail = event.target.closest('[data-activity-view]');
    if (rail) {
      if (rail.dataset.activityView !== 'model' && state.open) close();
      scheduleSync();
    }
  }, true);

  document.addEventListener('DOMContentLoaded', scheduleSync);

  // 对比视图关掉后，把当前方案重新投回单图视图
  function onCompareClosed() {
    if (state.open && state.tab === 'plans') syncGraphToSelection();
  }

  window.PtoFusionAdvisor = { open, close, syncEntry, onCompareClosed, isOpen: () => state.open };
})();
