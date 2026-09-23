/**
 * 整网融合推荐器 · 面板
 *
 * 挂在 torch_npu 部署态模型上，和官方结构上的「torch_npu 替换方案」是两套东西：
 *
 *   替换方案面板 —— skill: model-infer-fusion —— 现有算子怎么替换现有代码
 *   本面板       —— 昇腾大模型推理融合规则目录 —— 这张图里还有什么值得做成融合算子
 *
 * 四个页签对应文档的四段：候选（§18.1 最小推荐包）、未推荐（§19 原因码）、
 * 收益与排序（§5）、验证闭环（§20）。场景开关对应 §16：prefill 与 decode
 * 必须分别打分，不能互相外推。
 */
(function registerFusionCatalogAdvisor() {
  'use strict';

  const NPU_MODEL_ID = 'deepseek-v4-flash-npu';

  const TABS = [
    { id: 'candidates', label: '融合候选' },
    { id: 'deferred', label: '未推荐' },
    { id: 'ranking', label: '收益与排序' },
    { id: 'validation', label: '验证闭环' },
  ];

  const state = {
    open: false,
    tab: 'candidates',
    scenario: 'prefill',
    selected: 'cand-swiglu',
    lastFocus: null,
  };

  let root = null;

  const cat = () => window.PtoFusionCatalog;
  const viz = () => window.PtoDeepSeekV4ModelViz;
  const esc = (v) => String(v).replace(/[&<>"']/g, (ch) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[ch]));
  const num = (n) => (n === null || n === undefined ? '—' : Math.round(Number(n)).toLocaleString('en-US'));
  const bytes = (n) => (n >= 1024 * 1024 ? (n / 1024 / 1024).toFixed(1) + ' MB' : n >= 1024 ? (n / 1024).toFixed(1) + ' KB' : num(n) + ' B');

  const view = () => cat().evaluate(state.scenario);

  function selectedItem(model) {
    const data = model || view();
    return state.selected ? (data.items.find((x) => x.id === state.selected) || null) : null;
  }

  function currentItem(model) {
    const data = model || view();
    return selectedItem(data) || data.items[0] || null;
  }

  /** 展开哪条就把哪条在部署态整网上定位出来；全部收起就收起下钻 */
  function syncGraph() {
    const item = selectedItem();
    if (item && item.site) viz()?.focusCandidate?.(item.site);
    else viz()?.clearCandidate?.();
  }

  /* ---------------- 通用片段 ---------------- */

  const levelChip = (meta) => `<i class="kf-fcat-level is-${meta.id}" title="${esc(meta.desc)}">${esc(meta.label)}</i>`;

  const tag = (text, title, cls) => `<i class="kf-fcat-tag ${cls || ''}"${title ? ` title="${esc(title)}"` : ''}>${esc(text)}</i>`;

  const bullets = (title, items, tone, sub) => (items && items.length
    ? `<div class="kf-fcat-block${tone ? ` is-${tone}` : ''}">
        <h4>${esc(title)}${sub ? `<small>${esc(sub)}</small>` : ''}</h4>
        <ul class="kf-fcat-bullets">${items.map((x) => `<li>${esc(x)}</li>`).join('')}</ul>
      </div>`
    : '');

  const stepChips = (site) => {
    if (!site || !site.module) return '<span class="kf-fcat-muted">整网上没有对应位置</span>';
    const specs = viz() && viz().drillSpecs;
    const graph = viz() && viz().npuGraph && viz().npuGraph();
    const owner = graph && graph.nodes.find((n) => n.id === site.module);
    const spec = (specs && specs[site.module]) || null;
    const labels = (site.steps || []).map((i) => (spec && spec.steps[i] ? spec.steps[i][0] : site.module + '[' + i + ']'));
    const head = `<button class="kf-fcat-node" type="button" data-goto-site title="在整网上定位并展开">${esc(owner ? owner.label : site.module)}</button>`;
    return head + labels.map((l) => `<i class="kf-fcat-step">${esc(l)}</i>`).join('');
  };

  /* ---------------- 融合候选 ---------------- */

  function renderKpis(data) {
    const s = data.summary;
    const scen = data.scenarioMeta;
    return `<div class="kf-fcat-kpis">
      <div class="kf-fcat-kpi"><span>融合候选</span><b>${s.total}</b><small>强推荐 ${s.strong} · 条件推荐 ${s.conditional}</small></div>
      <div class="kf-fcat-kpi"><span>未推荐</span><b>${s.deferred}</b><small>谨慎 / 不推荐 · 带原因码</small></div>
      <div class="kf-fcat-kpi"><span>可省 launch</span><b>${num(s.launchSaved)}</b><small>整网单 token 累计</small></div>
      <div class="kf-fcat-kpi"><span>当前场景</span><b class="is-strong">${esc(scen.label)}</b><small>${esc(scen.cares)}</small></div>
    </div>`;
  }

  function renderList(data) {
    if (!data.items.length) return '<div class="kf-fcat-empty">当前场景下没有生成融合候选。</div>';
    return `<div class="kf-fcat-list">${data.items.map((item, index) => {
      const open = item.id === state.selected;
      const neg = item.netBenefit.net < 0;
      return `<div class="kf-fcat-item ${open ? 'is-open' : ''} ${neg ? 'is-negative' : ''}">
        <button type="button" class="kf-fcat-row" data-cand-pick="${esc(item.id)}" aria-expanded="${open}">
          <span class="kf-fcat-row__rank">${index + 1}</span>
          <span class="kf-fcat-row__main">
            <b>${esc(item.title)}</b>
            <code>${esc(item.pattern)}</code>
            <span class="kf-fcat-row__tags">
              ${tag(item.ruleId, '规则 ID')}
              ${levelChip(item.levelMeta)}
              ${tag(item.shapeMeta.label, item.shapeMeta.desc, 'is-shape')}
              ${tag('×' + item.invocations + ' 次/token', item.invocationNote)}
            </span>
          </span>
          <span class="kf-fcat-row__score ${neg ? 'is-negative' : ''}">
            <b>${item.netBenefit.net > 0 ? '+' : ''}${num(item.netBenefit.net)}</b>
            <small>score ${item.score.total > 0 ? '+' : ''}${item.score.total}</small>
          </span>
          <span class="kf-fcat-row__chev" aria-hidden="true">⌄</span>
        </button>
        ${open ? renderDetail(item, data) : ''}
      </div>`;
    }).join('')}</div>`;
  }

  function renderScoreBar(item) {
    const max = Math.max(...item.score.terms.map((t) => Math.abs(t.value)), 1);
    return `<div class="kf-fcat-score">
      ${item.score.terms.map((t) => `<div class="kf-fcat-scorerow ${t.value >= 0 ? 'is-plus' : 'is-minus'}">
        <span>${esc(t.label)}</span>
        <i><em style="width:${Math.round(Math.abs(t.value) / max * 100)}%"></em></i>
        <b>${t.value >= 0 ? '+' : ''}${t.value}</b>
        <small>${t.weight > 0 ? '+' : ''}${t.weight} × ${t.factor}</small>
      </div>`).join('')}
      <div class="kf-fcat-scoretotal"><span>FusionScore</span><b>${item.score.total > 0 ? '+' : ''}${item.score.total}</b><i class="kf-fcat-prov">provisional</i></div>
    </div>`;
  }

  function renderTraffic(item) {
    const t = item.traffic;
    return `<div class="kf-fcat-block">
      <h4>收益估算<small>文档 §5.4 · 单 token 单次调用</small></h4>
      <dl class="kf-fcat-metrics">
        <div><dt>gm_saved_bytes</dt><dd>${bytes(t.gm)}<em>${esc(t.gmNote)}</em></dd></div>
        <div><dt>launch_saved</dt><dd>${t.launchBefore} → ${t.launchAfter}<em>${t.launchBefore - t.launchAfter} 次 kernel launch</em></dd></div>
        <div><dt>live_working_set</dt><dd>${bytes(t.liveSet)}<em>${esc(t.liveNote)}</em></dd></div>
        <div><dt>sync_added</dt><dd>${t.syncAdded}<em>${esc(t.syncNote)}</em></dd></div>
      </dl>
      <p class="kf-fcat-note">分析机没有 profile 与 capability 数据，按文档 §5.4 要求，分数一律标记为 <b>provisional</b>，不给百分制结论。以上字节数是按 shape 与 dtype 推算的估计值。</p>
    </div>`;
  }

  function renderNet(item, data) {
    const n = item.netBenefit;
    return `<div class="kf-fcat-block">
      <h4>整网贡献<small>文档 §5.5 net_benefit</small></h4>
      <div class="kf-fcat-net">
        <span><b>${n.perInvocation > 0 ? '+' : ''}${n.perInvocation}</b><small>单次收益</small></span>
        <i>×</i>
        <span><b>${n.invocations}</b><small>invocation_count</small></span>
        <i>−</i>
        <span><b>${n.integrationCost + n.validationCost}</b><small>集成 ${n.integrationCost} + 验证 ${n.validationCost}</small></span>
        <i>=</i>
        <span class="is-total"><b>${n.net > 0 ? '+' : ''}${num(n.net)}</b><small>整网贡献指数</small></span>
      </div>
      <p class="kf-fcat-note">${esc(item.invocationNote)}。execution_weight：${esc(n.execNote)}。因子是序数不是物理量，所以 token 数只作场景说明，不乘进指数里。</p>
      ${n.net < 0 ? `<p class="kf-fcat-warn">当前是 <b>${esc(data.scenarioMeta.label)}</b> 场景，本候选净收益为负。文档 §16 要求 prefill 与 decode 分别打分 —— 换到另一个场景再看，不要把结论互相迁移。</p>` : ''}
    </div>`;
  }

  function renderDetail(item, data) {
    const b = item.boundary;
    return `<div class="kf-fcat-panel">
      <div class="kf-fcat-panelhead">
        <div class="kf-fcat-chips">
          ${tag(item.modeMeta.label, item.modeMeta.desc, 'is-mode')}
          ${tag('pypto_status: ' + item.pyptoLabel, '文档 §2.2 二级可行性校验')}
          ${tag('materialization: ' + item.materialization, '文档 §4.4')}
          ${tag(item.priority, '文档 §6 优先级')}
        </div>
        <div class="kf-fcat-actions">
          <button class="kf-fcat-btn" type="button" data-cand-validate="${esc(item.id)}">验证计划</button>
          <button class="kf-fcat-btn" type="button" data-cand-export="${esc(item.id)}">导出 JSON</button>
        </div>
      </div>
      <div class="kf-fcat-panelbody">

        <div class="kf-fcat-block">
          <h4>子图位置<small>点击在整网上展开并高亮</small></h4>
          <div class="kf-fcat-nodes">${stepChips(item.site)}</div>
          <p class="kf-fcat-note">model_context：<code>${esc(item.modelContext)}</code> · 归一化角色：${item.roles.map((r) => `<code>${esc(r)}</code>`).join(' → ')}</p>
        </div>

        <div class="kf-fcat-block">
          <h4>融合边界<small>文档 §18.1 最小推荐包</small></h4>
          <dl class="kf-fcat-boundary">
            <div><dt>start</dt><dd>${esc(b.start)}</dd></div>
            <div><dt>end</dt><dd>${esc(b.end)}</dd></div>
            <div><dt>保留物化</dt><dd>${b.preserve.length ? b.preserve.map(esc).join('；') : '无'}</dd></div>
            <div><dt>共享消费者</dt><dd>${b.shared.length ? b.shared.map(esc).join('；') : '无'}</dd></div>
            <div><dt>consumer_summary</dt><dd>多消费者 ${item.consumers.multi ? '是' : '否'} · 跨阶段复用 ${item.consumers.crossStage ? '是' : '否'}</dd></div>
          </dl>
          <p class="kf-fcat-note">${esc(item.semantics)}</p>
        </div>

        ${bullets('收益来源', item.benefits, 'good')}
        ${bullets('必要条件', item.requirements, 'warn', '硬约束 · 文档 §17')}
        ${bullets('风险', item.risks, 'risk')}

        ${renderTraffic(item)}
        ${renderNet(item, data)}

        <div class="kf-fcat-block">
          <h4>FusionScore 拆解<small>文档 §5.3</small></h4>
          ${renderScoreBar(item)}
        </div>

        ${bullets('回退链', item.fallback, null, '文档 §18.1 · 更小子图 → 复合/编排 → 库实现')}

        <div class="kf-fcat-block">
          <h4>证据<small>文档 §22 规则维护</small></h4>
          <p class="kf-fcat-note">置信度：<b>${esc({ validated: '已验证', experience: '经验规则', hypothesis: '待验证假设' }[item.evidence.status] || item.evidence.status)}</b></p>
          <ul class="kf-fcat-bullets">${item.evidence.refs.map((r) => `<li><code>${esc(r)}</code></li>`).join('')}</ul>
        </div>
      </div>
    </div>`;
  }

  function renderCandidates(data) {
    return `${renderKpis(data)}${renderList(data)}`;
  }

  /* ---------------- 未推荐 ---------------- */

  function renderDeferred(data) {
    const reasons = cat().REASONS;
    return `<div class="kf-fcat-deferred">
      <p class="kf-fcat-lead">文档 §19 要求不能只输出「不可融合」，每条都要带稳定的原因码。以下候选保留在目录里，条件变化时会重新进入推荐列表。</p>
      ${data.deferred.map((x) => `<article class="kf-fcat-defcard is-${x.level}">
        <header>
          <div>
            <b>${esc(x.title)}</b>
            <code>${esc(x.pattern)}</code>
          </div>
          <div class="kf-fcat-defmeta">${tag(x.ruleId, '规则 ID')}${levelChip(x.levelMeta)}</div>
        </header>
        <div class="kf-fcat-reasons">${x.reasons.map((r) => `<i class="kf-fcat-reason" title="${esc(reasons[r] || '')}">${esc(r)}</i>`).join('')}</div>
        <p>${esc(x.why)}</p>
        <p class="kf-fcat-next"><span>下一步</span>${esc(x.next)}${x.mode ? ` · 建议实现路径 <code>${esc(x.mode)}</code>` : ''}</p>
        ${x.site && x.site.module ? `<button class="kf-fcat-node" type="button" data-goto-defer="${esc(x.id)}">在整网上定位</button>` : ''}
      </article>`).join('')}
    </div>`;
  }

  /* ---------------- 收益与排序 ---------------- */

  function renderRanking(data) {
    const w = cat().WEIGHTS;
    const scen = cat().SCENARIOS;
    return `<div class="kf-fcat-ranking">
      <div class="kf-fcat-block">
        <h4>排序公式<small>文档 §5.3 · MVP 启发式</small></h4>
        <pre class="kf-fcat-formula">FusionScore =
${w.map((x) => `  ${x.sign > 0 ? '+' : '−'} ${x.w} × ${x.label}`).join('\n')}</pre>
        <p class="kf-fcat-note">正式版本应使用 profile 数据校准权重，并按 prefill、decode、batch size、sequence length 分别建模。当前所有分数都是 <b>provisional</b>。</p>
      </div>

      <div class="kf-fcat-block">
        <h4>场景对照<small>文档 §16 FUS-DECODE-002</small></h4>
        <div class="kf-fcat-scencompare">
          <div class="kf-fcat-scenhead"><span>候选</span><b>${esc(scen.prefill.label)}</b><b>${esc(scen.decode.label)}</b><span>差异</span></div>
          ${(() => {
    const p = cat().evaluate('prefill');
    const d = cat().evaluate('decode');
    return p.items.map((item) => {
      const other = d.items.find((x) => x.id === item.id);
      const delta = other.netBenefit.net - item.netBenefit.net;
      return `<div class="kf-fcat-scenrow">
              <span>${esc(item.title)}</span>
              <b>${item.netBenefit.net > 0 ? '+' : ''}${num(item.netBenefit.net)}</b>
              <b class="${other.netBenefit.net < 0 ? 'is-negative' : ''}">${other.netBenefit.net > 0 ? '+' : ''}${num(other.netBenefit.net)}</b>
              <span class="${delta >= 0 ? 'is-up' : 'is-down'}">${delta >= 0 ? '▲' : '▼'} ${num(Math.abs(delta))}</span>
            </div>`;
    }).join('');
  })()}
        </div>
        <p class="kf-fcat-note">${esc(scen.prefill.label)} 更看重 ${esc(scen.prefill.cares)}；${esc(scen.decode.label)} 更看重 ${esc(scen.decode.cares)}。同一条链在两个场景下的排名可以完全不同 —— 这正是文档禁止把 prefill 结论迁移到 decode 的原因。</p>
      </div>

      <div class="kf-fcat-block">
        <h4>整网聚合<small>文档 §5.5</small></h4>
        <div class="kf-fcat-aggr">
          <div class="kf-fcat-aggrhead"><span>候选</span><span>单次</span><span>×次数</span><span>集成+验证</span><span>整网贡献</span></div>
          ${data.items.map((x) => `<div class="kf-fcat-aggrrow">
            <span>${esc(x.title)}</span>
            <b>${x.score.total > 0 ? '+' : ''}${x.score.total}</b>
            <b>${x.invocations}</b>
            <b>−${x.netBenefit.integrationCost + x.netBenefit.validationCost}</b>
            <b class="is-total ${x.netBenefit.net < 0 ? 'is-negative' : ''}">${x.netBenefit.net > 0 ? '+' : ''}${num(x.netBenefit.net)}</b>
          </div>`).join('')}
        </div>
        <p class="kf-fcat-note">Embedding 掩码链的单次分数最高之一，但整网只调用一次，聚合后排到后面；SwiGLU 单次分数不是最高，却因为 43 层每 token 必过而稳居第一。文档 §5.5 要求的就是同时展示这两面。</p>
      </div>
    </div>`;
  }

  /* ---------------- 验证闭环 ---------------- */

  function renderValidation(data) {
    const item = currentItem(data);
    if (!item) return '<div class="kf-fcat-empty">先在「融合候选」里选一条。</div>';
    const gates = cat().GATES;
    return `<div class="kf-fcat-validation">
      <div class="kf-fcat-current"><span>当前候选</span><b>${esc(item.title)}</b><code>${esc(item.ruleId)}</code></div>
      <p class="kf-fcat-lead">文档 §20：每个进入实现的候选至少经过四道门。前一道不过就不进下一道。</p>
      <ol class="kf-fcat-gates">
        ${gates.map((g, i) => `<li class="kf-fcat-gate">
          <i>${i + 1}</i>
          <div>
            <b>${esc(g.label)}</b>
            <small>${esc(g.desc)}</small>
            <p>${esc(item.validation[g.id])}</p>
          </div>
        </li>`).join('')}
      </ol>
      <div class="kf-fcat-block">
        <h4>Baseline 集<small>文档 §20</small></h4>
        <ul class="kf-fcat-bullets">${cat().BASELINES.map((b) => `<li>${esc(b)}</li>`).join('')}</ul>
      </div>
      <div class="kf-fcat-block is-warn">
        <h4>进入算子开发前必须明确</h4>
        <p class="kf-fcat-note">文档 §18.1：这是「模型层推荐」「可编译草图」还是「已验证实现」。当前候选的 pypto_status 是 <b>${esc(item.pyptoLabel)}</b> —— 模型层推荐不因为 PyPTO 尚未支持某条实现路径而消失，只降低实现置信度并提供回退链。</p>
      </div>
    </div>`;
  }

  /* ---------------- 外壳 ---------------- */

  function shell(data) {
    const s = data.summary;
    const tabs = TABS.map((t) => {
      const badge = t.id === 'candidates' ? s.total : t.id === 'deferred' ? s.deferred : '';
      return `<button type="button" class="${state.tab === t.id ? 'is-active' : ''}" data-fcat-tab="${t.id}" role="tab" aria-selected="${state.tab === t.id}">${esc(t.label)}${badge ? `<em>${badge}</em>` : ''}</button>`;
    }).join('');

    const scen = Object.values(cat().SCENARIOS).map((x) => `<button type="button" data-fcat-scenario="${x.id}" class="${state.scenario === x.id ? 'is-active' : ''}" title="${esc(x.cares)}">${esc(x.label)}</button>`).join('');

    return `<aside class="kf-fcat-drawer" role="region" aria-label="整网融合推荐器">
      <header class="kf-fcat-head">
        <div>
          <h2>整网融合推荐器</h2>
          <p>规则目录：${esc(cat().SOURCE)} · ${esc(cat().DOC_DATE)}</p>
        </div>
        <div class="kf-fcat-headactions">
          <div class="kf-fcat-scenario" role="group" aria-label="推理场景">${scen}</div>
          <button class="kf-fcat-close" type="button" data-fcat-close aria-label="关闭">✕</button>
        </div>
      </header>
      <div class="kf-fcat-context">
        <span>作用对象 <b>torch_npu 部署态整网</b></span>
        <span class="kf-fcat-ctxsep"></span>
        <span>回答 <b>还有什么值得做成融合算子</b></span>
        <span class="kf-fcat-ctxsep"></span>
        <span class="kf-fcat-prov">分数 provisional · 无 profile 数据</span>
      </div>
      <nav class="kf-fcat-tabs" role="tablist">${tabs}</nav>
      <div class="kf-fcat-body" id="fcatBody"></div>
    </aside>`;
  }

  function renderBody() {
    const body = root && root.querySelector('#fcatBody');
    if (!body) return;
    const data = view();
    if (state.tab === 'candidates') body.innerHTML = renderCandidates(data);
    else if (state.tab === 'deferred') body.innerHTML = renderDeferred(data);
    else if (state.tab === 'ranking') body.innerHTML = renderRanking(data);
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

  function exportCandidate(item) {
    // 文档 §18 的推荐结果数据结构
    download(item.id + '.json', JSON.stringify({
      candidate_id: item.id,
      rule_id: item.ruleId,
      category: item.category,
      priority: item.priority,
      nodes: item.site ? [item.site.module].concat((item.site.steps || []).map((i) => item.site.module + '__' + i)) : [],
      pattern: item.pattern,
      recommendation: item.level,
      fusion_shape: item.shape,
      model_context: item.modelContext,
      normalized_roles: item.roles,
      fusion_boundary: {
        start: item.boundary.start,
        end: item.boundary.end,
        preserve_outputs: item.boundary.preserve,
        shared_consumers: item.boundary.shared,
      },
      implementation_mode: item.mode,
      pypto_status: item.pyptoStatus,
      materialization_decision: item.materialization,
      consumer_summary: {
        intermediate_multi_consumer: item.consumers.multi,
        cross_stage_reuse: item.consumers.crossStage,
      },
      gm_roundtrip_saved: { bytes: item.traffic.gm, confidence: 'estimate' },
      resource_estimate: {
        intermediate_bytes: item.traffic.liveSet,
        estimated_launch_saved: item.traffic.launchBefore - item.traffic.launchAfter,
        sync_added: item.traffic.syncAdded,
      },
      scenario: state.scenario,
      score: item.score.total,
      score_terms: item.score.terms,
      score_confidence: 'provisional',
      net_benefit: item.netBenefit,
      benefits: item.benefits,
      requirements: item.requirements,
      risks: item.risks,
      validation: item.validation,
      fallback_chain: item.fallback,
      evidence: item.evidence,
      source: cat().SOURCE,
    }, null, 2));
  }

  function onClick(event) {
    const t = event.target;
    if (t.closest('[data-fcat-close]')) { close(); return; }

    const tab = t.closest('[data-fcat-tab]');
    if (tab) {
      state.tab = tab.dataset.fcatTab;
      renderAll();
      if (state.tab === 'candidates') syncGraph();
      return;
    }

    const scen = t.closest('[data-fcat-scenario]');
    if (scen) {
      state.scenario = scen.dataset.fcatScenario;
      renderAll();
      return;
    }

    const pick = t.closest('[data-cand-pick]');
    if (pick) {
      const id = pick.dataset.candPick;
      state.selected = state.selected === id ? null : id;
      renderBody();
      syncGraph();
      if (state.selected) {
        const el = root.querySelector(`[data-cand-pick="${CSS.escape(id)}"]`);
        if (el) el.scrollIntoView({ block: 'nearest' });
      }
      return;
    }

    if (t.closest('[data-goto-site]')) { syncGraph(); return; }

    const defer = t.closest('[data-goto-defer]');
    if (defer) {
      const x = view().deferred.find((d) => d.id === defer.dataset.gotoDefer);
      if (x && x.site) viz()?.focusCandidate?.(x.site);
      return;
    }

    const val = t.closest('[data-cand-validate]');
    if (val) { state.selected = val.dataset.candValidate; state.tab = 'validation'; renderAll(); return; }

    const exp = t.closest('[data-cand-export]');
    if (exp) {
      const item = view().items.find((x) => x.id === exp.dataset.candExport);
      if (item) exportCandidate(item);
    }
  }

  /* ---------------- 生命周期 ---------------- */

  function open() {
    const host = document.getElementById('modelArchitectureView');
    if (!host || !cat()) return;
    if (!root) {
      root = document.createElement('div');
      root.className = 'kf-fcat-root';
      root.id = 'fusionCatalog';
      host.appendChild(root);
      root.addEventListener('click', onClick);
    }
    state.lastFocus = document.activeElement;
    state.open = true;
    root.hidden = false;
    host.classList.add('is-catalog-docked');
    renderAll();
    if (state.tab === 'candidates') syncGraph();
  }

  function close() {
    if (!root) return;
    state.open = false;
    root.hidden = true;
    const host = document.getElementById('modelArchitectureView');
    if (host) host.classList.remove('is-catalog-docked');
    viz()?.clearCandidate?.();
    if (state.lastFocus && state.lastFocus.isConnected) state.lastFocus.focus();
  }

  /** 只有 torch_npu 部署态模型挂这个面板 */
  function syncEntry() {
    const active = (window.PtoModelArchitectureState || {}).active;
    const supported = active === NPU_MODEL_ID && !!cat();
    document.querySelectorAll('[data-open-catalog]').forEach((el) => { el.hidden = !supported; });
    if (!supported && state.open) close();
  }

  const scheduleSync = () => setTimeout(syncEntry, 0);

  document.addEventListener('keydown', (event) => {
    if (event.key === 'Escape' && state.open) { event.stopPropagation(); close(); }
  });

  document.addEventListener('click', (event) => {
    if (event.target.closest('[data-open-catalog]')) {
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

  window.PtoFusionCatalogAdvisor = { open, close, syncEntry, isOpen: () => state.open };
})();
